from flask import Flask, render_template, request, redirect, url_for
import os
import logging

from modules.pdf_loader import extract_pdf_text
from modules.text_preprocessing import clean_text_for_indexing, clean_text_for_display
from modules.chunking import split_into_chunks
from modules.vectorizer import create_vectorizer
from modules.search_engine import search
from modules.answer_builder import build_answer
from modules.config import (
    DATA_FOLDER,
    UPLOAD_FOLDER,
    DEFAULTS,
    UPLOAD_CONFIG,
    LOGGING_CONFIG,
)

from modules.db_layer import (
    DocumentManager,
    ChunkManager,
    VectorManager,
    VectorizerManager,
    AuditManager,
    DataExportManager,
)
from modules.db import get_db


logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

try:
    db = get_db()
    logger.info("Database initialized on startup")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise


def load_metadata_safe():
    """Load metadata from database."""
    try:
        metadata = DataExportManager.get_metadata_as_list()
        logger.debug(f"Loaded {len(metadata)} metadata items from database")
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return []


def is_pdf_file(filename):
    if not filename:
        return False
    return filename.lower().endswith(".pdf")


def detect_query_type(query):
    query_lower = query.strip().lower()

    if any(word in query_lower for word in ["difference between", "compare", "comparison", "distinguish"]):
        return "comparison"
    if any(word in query_lower for word in ["layers", "types", "steps", "components", "features"]):
        return "list"
    if any(word in query_lower for word in ["what is", "define", "meaning of"]):
        return "definition"
    if any(word in query_lower for word in ["explain", "describe", "discuss"]):
        return "explanation"

    return "general"


def get_system_stats():
    metadata = load_metadata_safe()
    documents = DocumentManager.get_all_documents()

    total_chunks = len(metadata)
    doc_types = sorted({doc.get("type", "unknown") for doc in documents})
    subjects = sorted({doc.get("subject", "General") for doc in documents})

    docs = [
        {
            "id": doc.get("id"),
            "filename": doc.get("filename"),
            "type": doc.get("type", "unknown"),
            "subject": doc.get("subject", "General"),
            "total_chunks": doc.get("total_chunks", 0),
        }
        for doc in documents
    ]

    return {
        "total_chunks": total_chunks,
        "total_documents": len(documents),
        "documents": docs,
        "doc_types": doc_types,
        "subjects": subjects,
    }


def rebuild_search_index():
    """Rebuild the search index for the full active corpus."""
    metadata = load_metadata_safe()

    if not metadata:
        logger.info("No metadata found. Clearing vector index state.")
        if not VectorManager.replace_all_tfidf_vectors([], []):
            raise RuntimeError("Failed to clear TF-IDF vectors")
        if not VectorManager.replace_all_semantic_vectors([], []):
            raise RuntimeError("Failed to clear semantic vectors")
        if not VectorizerManager.clear_vectorizers():
            raise RuntimeError("Failed to clear vectorizer state")
        return 0

    corpus = [item["indexed_chunk"] for item in metadata]
    chunk_ids = DataExportManager.get_chunk_ids_in_order()

    if len(chunk_ids) != len(corpus):
        raise ValueError(
            f"Chunk/index mismatch: {len(chunk_ids)} chunk ids for {len(corpus)} corpus items"
        )

    vectorizer, vectors, semantic_vectors = create_vectorizer(corpus)

    if not VectorizerManager.save_vectorizer(vectorizer):
        raise RuntimeError("Failed to save vectorizer")
    if not VectorManager.replace_all_tfidf_vectors(chunk_ids, vectors):
        raise RuntimeError("Failed to store TF-IDF vectors")
    if not VectorManager.replace_all_semantic_vectors(chunk_ids, semantic_vectors):
        raise RuntimeError("Failed to store semantic vectors")

    logger.info(f"Rebuilt search index for {len(corpus)} chunks")
    return len(corpus)


@app.route("/")
def home():
    stats = get_system_stats()
    return render_template("index_modern.html", stats=stats)


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "pdf" not in request.files:
            logger.warning("No PDF file in upload request")
            return "<h3>Error: No file uploaded</h3>"

        file = request.files["pdf"]
        doc_type = request.form.get("doc_type", DEFAULTS["doc_type"]).strip().lower()
        subject = request.form.get("subject", DEFAULTS["subject"]).strip()

        if file.filename == "":
            logger.warning("Empty filename in upload request")
            return "<h3>Error: No selected file</h3>"

        if not is_pdf_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return "<h3>Error: Invalid file type. Upload PDF only.</h3>"

        if not subject:
            subject = DEFAULTS["subject"]

        if DocumentManager.document_exists(file.filename):
            logger.info(f"Duplicate file attempted: {file.filename}")
            return f"<h3>Warning: File '{file.filename}' already uploaded.</h3>"

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        logger.info(f"File saved: {file_path}")

        pages = extract_pdf_text(file_path)
        if not pages:
            logger.warning(f"No readable text found in: {file.filename}")
            return "<h3>Error: No readable text found (possibly scanned PDF)</h3>"

        new_chunks = []

        for p in pages:
            readable_page_text = clean_text_for_display(p["text"])
            chunks = split_into_chunks(readable_page_text)

            for chunk in chunks:
                indexed_chunk = clean_text_for_indexing(chunk, remove_stopwords=True)

                if len(indexed_chunk.strip()) < 10:
                    continue

                new_chunks.append(
                    {
                        "chunk": chunk,
                        "indexed_chunk": indexed_chunk,
                        "page": p["page"],
                    }
                )

        if len(new_chunks) == 0:
            logger.warning(f"No valid chunks extracted from: {file.filename}")
            return "<h3>Error: No readable text found</h3>"

        try:
            doc_id = DocumentManager.add_document(
                filename=file.filename,
                doc_type=doc_type,
                subject=subject,
            )
            logger.info(f"Document added to database: {file.filename} (ID: {doc_id})")

            chunk_ids = ChunkManager.add_chunks(doc_id, new_chunks)
            logger.info(f"Added {len(chunk_ids)} chunks to database")

            DocumentManager.update_chunk_count(doc_id, len(chunk_ids))

            total_chunks = rebuild_search_index()

            AuditManager.log_action(
                "document_uploaded",
                doc_id,
                f"File: {file.filename}, Chunks: {len(chunk_ids)}",
            )

            logger.info(f"Successfully processed PDF: {file.filename}")

            return f"""
            <h2>Upload successful</h2>
            <p>Chunks added: {len(new_chunks)}</p>
            <p>Total chunks in system: {total_chunks}</p>
            <a href="/">Go Back</a>
            """

        except Exception as e:
            logger.error(f"Error processing chunks: {e}", exc_info=True)
            return f"<h3>Error processing PDF: {str(e)}</h3>"

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return f"<h3>Unexpected error: {str(e)}</h3>"


@app.route("/search", methods=["POST"])
def search_route():
    try:
        query = request.form.get("query", "").strip()
        if not query:
            logger.warning("Empty search query")
            return "<h3>Error: Please enter a search query</h3>"

        search_type = request.form.get("search_type", DEFAULTS["search_type"])
        subject_filter = request.form.get("subject_filter", DEFAULTS["subject_filter"])

        logger.info(f"Search query: {query[:50]} | Type: {search_type} | Subject: {subject_filter}")

        try:
            metadata = load_metadata_safe()
            logger.info(f"Loaded {len(metadata)} metadata items")

            if not metadata:
                logger.warning("Search attempted but no data available")
                return "<h3>Error: Please upload PDFs first.</h3>"

            vectorizer = VectorizerManager.get_vectorizer()
            if vectorizer is None:
                logger.warning("Vectorizer not found in database")
                return "<h3>Error: System not initialized. Please upload PDFs first.</h3>"

            chunk_ids, vectors = VectorManager.get_all_tfidf_vectors()
            if not vectors:
                logger.warning("No TF-IDF vectors found in database")
                return "<h3>Error: No search data available. Please upload PDFs first.</h3>"

            if len(chunk_ids) != len(metadata):
                logger.error(
                    "Search index mismatch: %s TF-IDF vectors for %s metadata items",
                    len(chunk_ids),
                    len(metadata),
                )
                return "<h3>Error: Search index is out of sync. Please re-upload or rebuild the data.</h3>"

            logger.info(f"Retrieved {len(vectors)} vectors from database for search")

        except Exception as e:
            logger.error(f"Error loading search data: {e}")
            return f"<h3>Error loading data: {str(e)}</h3>"

        cleaned_query = clean_text_for_indexing(query, remove_stopwords=True)
        logger.info(f"Cleaned query: {cleaned_query}")

        results = search(
            cleaned_query,
            vectorizer,
            vectors,
            metadata,
            top_k=None,
            search_type=search_type,
            subject_filter=subject_filter,
        )

        logger.info(f"Search returned {len(results)} results")

        if not results:
            logger.info(f"No results found for query: {query[:50]}")
            answer = "No relevant results found for your query."
        else:
            try:
                answer = build_answer(query, results)
                logger.info(f"Generated answer: {len(answer)} chars")
            except Exception as e:
                logger.error(f"Error building answer: {e}", exc_info=True)
                if results:
                    answer = f"Here's relevant information from your documents:\n\n{results[0].get('chunk', '')[:500]}"
                else:
                    answer = "Could not generate answer."

        try:
            AuditManager.log_action("search_performed", None, f"Query: {query[:100]}")
        except Exception as e:
            logger.error(f"Error logging to audit trail: {e}")

        logger.info(f"Search completed. Results: {len(results)}, Answer length: {len(answer)}")

        return render_template(
            "results_modern.html",
            results=results,
            query=query,
            answer=answer,
        )

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return f"<h3>Error: {str(e)}</h3>"


@app.route("/delete_document", methods=["POST"])
def delete_document():
    try:
        doc_id = request.form.get("doc_id")
        if not doc_id or not doc_id.isdigit():
            logger.warning("Invalid document delete request")
            return redirect(url_for("home"))

        doc_id = int(doc_id)
        document = DocumentManager.get_document(doc_id)
        if not document:
            logger.warning(f"Document not found for deletion: {doc_id}")
            return redirect(url_for("home"))

        filename = document.get("filename")
        deleted = DocumentManager.delete_document(doc_id)
        if deleted:
            if filename:
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted uploaded file: {file_path}")

            rebuild_search_index()
            logger.info(f"Deleted document {doc_id} ({filename})")
        else:
            logger.error(f"Failed to delete document {doc_id} ({filename})")

        return redirect(url_for("home"))
    except Exception as e:
        logger.error(f"Delete document error: {e}", exc_info=True)
        return redirect(url_for("home"))


@app.route("/clear", methods=["POST"])
def clear_data():
    try:
        logger.warning("CLEARING ALL DATA FROM DATABASE")

        try:
            db = get_db()

            with db.get_cursor() as cursor:
                cursor.execute("DELETE FROM vectors")
                logger.info("Deleted all vectors")

                cursor.execute("DELETE FROM semantic_vectors")
                logger.info("Deleted all semantic vectors")

                cursor.execute("DELETE FROM chunks")
                logger.info("Deleted all chunks")

                cursor.execute("DELETE FROM documents")
                logger.info("Deleted all documents")

                cursor.execute("DELETE FROM vectorizer_state")
                logger.info("Deleted vectorizer state")

                cursor.execute("DELETE FROM audit_log")
                logger.info("Deleted audit log")

            logger.info("All database tables cleared successfully")
        except Exception as e:
            logger.error(f"Error during database deletion: {e}", exc_info=True)
            return redirect(url_for("home"))

        try:
            pdf_files_deleted = 0
            if os.path.exists(UPLOAD_FOLDER):
                for filename in os.listdir(UPLOAD_FOLDER):
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Deleted PDF: {file_path}")
                        pdf_files_deleted += 1

            logger.info(f"Deleted {pdf_files_deleted} PDF files from {UPLOAD_FOLDER}")
        except Exception as e:
            logger.error(f"Error deleting PDF files: {e}")

        try:
            AuditManager.log_action(
                "data_cleared",
                None,
                "All data cleared from system",
            )
        except Exception as e:
            logger.error(f"Error logging clear action: {e}")

        logger.warning("ALL DATA CLEARED SUCCESSFULLY")
        return redirect(url_for("home"))

    except Exception as e:
        logger.error(f"Clear error: {e}", exc_info=True)
        return redirect(url_for("home"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
