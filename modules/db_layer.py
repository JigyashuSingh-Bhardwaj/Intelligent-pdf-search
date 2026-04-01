"""
Data access layer abstraction
Provides high-level API for interacting with database
Decouples application from database implementation
"""

import pickle
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

from modules.db import get_db

logger = logging.getLogger(__name__)


class DocumentManager:
    """Manage document operations"""
    
    @staticmethod
    def add_document(filename: str, doc_type: str, subject: str, file_hash: str = None) -> int:
        """Add new document to database"""
        try:
            db = get_db()
            
            if not file_hash:
                file_hash = hashlib.md5(filename.encode()).hexdigest()
            
            with db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO documents (filename, type, subject, file_hash, status)
                    VALUES (?, ?, ?, ?, 'active')
                """, (filename, doc_type, subject, file_hash))
                
                doc_id = cursor.lastrowid
                logger.info(f"Document added: {filename} (ID: {doc_id})")
                return doc_id
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    @staticmethod
    def get_document(doc_id: int) -> Optional[Dict]:
        """Get document by ID"""
        try:
            db = get_db()
            result = db.execute_query(
                "SELECT * FROM documents WHERE id = ?", 
                (doc_id,)
            )
            return dict(result[0]) if result else None
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None
    
    @staticmethod
    def get_all_documents() -> List[Dict]:
        """Get all active documents"""
        try:
            db = get_db()
            results = db.execute_query(
                "SELECT * FROM documents WHERE status = 'active' ORDER BY upload_date DESC"
            )
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []
    
    @staticmethod
    def update_chunk_count(doc_id: int, count: int) -> bool:
        """Update chunk count for document"""
        try:
            db = get_db()
            db.execute_update(
                "UPDATE documents SET total_chunks = ? WHERE id = ?",
                (count, doc_id)
            )
            logger.info(f"Updated chunk count for document {doc_id}: {count}")
            return True
        except Exception as e:
            logger.error(f"Error updating chunk count: {e}")
            return False
    
    @staticmethod
    def delete_document(doc_id: int) -> bool:
        """Delete document and all associated data"""
        try:
            db = get_db()
            
            with db.get_cursor() as cursor:
                # Delete in correct order to avoid FK constraint issues
                # 1. Delete vectors first
                cursor.execute("""
                    DELETE FROM vectors 
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)
                """, (doc_id,))
                logger.debug(f"Deleted vectors for document {doc_id}")
                
                # 2. Delete semantic vectors
                cursor.execute("""
                    DELETE FROM semantic_vectors 
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)
                """, (doc_id,))
                logger.debug(f"Deleted semantic vectors for document {doc_id}")
                
                # 3. Delete chunks
                cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
                logger.debug(f"Deleted chunks for document {doc_id}")
                
                # 4. Delete document
                cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                logger.debug(f"Deleted document {doc_id}")
                
                # Log to audit trail
                cursor.execute("""
                    INSERT INTO audit_log (action, document_id, details)
                    VALUES (?, ?, ?)
                """, ("document_deleted", doc_id, "Document permanently deleted"))
                
            logger.info(f"Document deleted: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    @staticmethod
    def document_exists(filename: str) -> bool:
        """Check if document already exists"""
        try:
            db = get_db()
            result = db.execute_query(
                "SELECT id FROM documents WHERE filename = ? AND status = 'active'",
                (filename,)
            )
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False


class ChunkManager:
    """Manage chunk operations"""
    
    @staticmethod
    def add_chunks(doc_id: int, chunks_data: List[Dict]) -> List[int]:
        """Add multiple chunks for a document"""
        try:
            db = get_db()
            chunk_ids = []
            
            with db.get_cursor() as cursor:
                for chunk in chunks_data:
                    cursor.execute("""
                        INSERT INTO chunks 
                        (document_id, chunk_text, indexed_chunk, page)
                        VALUES (?, ?, ?, ?)
                    """, (
                        doc_id,
                        chunk.get('chunk', ''),
                        chunk.get('indexed_chunk', ''),
                        chunk.get('page', 0)
                    ))
                    chunk_ids.append(cursor.lastrowid)
            
            logger.info(f"Added {len(chunk_ids)} chunks for document {doc_id}")
            return chunk_ids
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            raise
    
    @staticmethod
    def get_all_chunks() -> List[Dict]:
        """Get all chunks with metadata"""
        try:
            db = get_db()
            results = db.execute_query("""
                SELECT c.*, d.filename, d.type, d.subject
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.status = 'active'
                ORDER BY c.document_id, c.page
            """)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []
    
    @staticmethod
    def get_chunks_by_document(doc_id: int) -> List[Dict]:
        """Get all chunks for a document"""
        try:
            db = get_db()
            results = db.execute_query(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY page",
                (doc_id,)
            )
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting chunks for document {doc_id}: {e}")
            return []
    
    @staticmethod
    def delete_chunks_by_document(doc_id: int) -> bool:
        """Delete all chunks for a document"""
        try:
            db = get_db()
            db.execute_update("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            logger.info(f"Deleted chunks for document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False


class VectorManager:
    """Manage vector storage"""
    
    @staticmethod
    def add_tfidf_vectors(chunk_ids: List[int], vectors_data) -> bool:
        """Store TF-IDF vectors"""
        try:
            db = get_db()
            
            # Convert sparse matrix to dense for storage
            dense_vectors = vectors_data.toarray() if hasattr(vectors_data, 'toarray') else vectors_data
            
            with db.get_cursor() as cursor:
                for i, chunk_id in enumerate(chunk_ids):
                    vector_bytes = pickle.dumps(dense_vectors[i])
                    cursor.execute("""
                        INSERT INTO vectors (chunk_id, vector_data)
                        VALUES (?, ?)
                    """, (chunk_id, vector_bytes))
            
            logger.info(f"Stored {len(chunk_ids)} TF-IDF vectors")
            return True
        except Exception as e:
            logger.error(f"Error storing TF-IDF vectors: {e}")
            return False
    
    @staticmethod
    def add_semantic_vectors(chunk_ids: List[int], vectors_data) -> bool:
        """Store semantic embeddings"""
        try:
            db = get_db()
            
            with db.get_cursor() as cursor:
                for i, chunk_id in enumerate(chunk_ids):
                    embedding_bytes = pickle.dumps(vectors_data[i])
                    cursor.execute("""
                        INSERT INTO semantic_vectors (chunk_id, embedding)
                        VALUES (?, ?)
                    """, (chunk_id, embedding_bytes))
            
            logger.info(f"Stored {len(chunk_ids)} semantic vectors")
            return True
        except Exception as e:
            logger.error(f"Error storing semantic vectors: {e}")
            return False
    
    @staticmethod
    def get_all_tfidf_vectors() -> Tuple[List[int], List]:
        """Get all TF-IDF vectors as dense array"""
        try:
            db = get_db()
            results = db.execute_query("""
                SELECT chunk_id, vector_data FROM vectors
                ORDER BY chunk_id
            """)
            
            chunk_ids = []
            vectors = []
            
            for row in results:
                chunk_ids.append(row['chunk_id'])
                vectors.append(pickle.loads(row['vector_data']))
            
            logger.info(f"Retrieved {len(vectors)} TF-IDF vectors")
            return chunk_ids, vectors
        except Exception as e:
            logger.error(f"Error retrieving TF-IDF vectors: {e}")
            return [], []
    
    @staticmethod
    def get_all_semantic_vectors() -> Tuple[List[int], List]:
        """Get all semantic vectors as array"""
        try:
            db = get_db()
            results = db.execute_query("""
                SELECT chunk_id, embedding FROM semantic_vectors
                ORDER BY chunk_id
            """)
            
            chunk_ids = []
            embeddings = []
            
            for row in results:
                chunk_ids.append(row['chunk_id'])
                embeddings.append(pickle.loads(row['embedding']))
            
            logger.info(f"Retrieved {len(embeddings)} semantic vectors")
            return chunk_ids, embeddings
        except Exception as e:
            logger.error(f"Error retrieving semantic vectors: {e}")
            return [], []


class VectorizerManager:
    """Manage vectorizer state"""
    
    @staticmethod
    def save_vectorizer(vectorizer_obj) -> bool:
        """Save vectorizer pickle to database"""
        try:
            db = get_db()
            vectorizer_bytes = pickle.dumps(vectorizer_obj)
            
            # Mark previous as not current
            db.execute_update("UPDATE vectorizer_state SET is_current = 0")
            
            # Insert new version
            db.execute_update("""
                INSERT INTO vectorizer_state (vectorizer_pickle, is_current)
                VALUES (?, 1)
            """, (vectorizer_bytes,))
            
            logger.info("Vectorizer saved to database")
            return True
        except Exception as e:
            logger.error(f"Error saving vectorizer: {e}")
            return False
    
    @staticmethod
    def get_vectorizer():
        """Get current vectorizer from database"""
        try:
            db = get_db()
            result = db.execute_query(
                "SELECT vectorizer_pickle FROM vectorizer_state WHERE is_current = 1 LIMIT 1"
            )
            
            if result:
                vectorizer = pickle.loads(result[0]['vectorizer_pickle'])
                logger.info("Vectorizer retrieved from database")
                return vectorizer
            
            logger.warning("No vectorizer found in database")
            return None
        except Exception as e:
            logger.error(f"Error retrieving vectorizer: {e}")
            return None


class AuditManager:
    """Manage audit logging"""
    
    @staticmethod
    def log_action(action: str, document_id: int = None, details: str = None) -> bool:
        """Log an action to audit trail"""
        try:
            db = get_db()
            db.execute_update("""
                INSERT INTO audit_log (action, document_id, details)
                VALUES (?, ?, ?)
            """, (action, document_id, details))
            
            logger.debug(f"Audit logged: {action}")
            return True
        except Exception as e:
            logger.error(f"Error logging audit: {e}")
            return False
    
    @staticmethod
    def get_audit_log(limit: int = 100) -> List[Dict]:
        """Get recent audit entries"""
        try:
            db = get_db()
            results = db.execute_query("""
                SELECT * FROM audit_log
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error retrieving audit log: {e}")
            return []


class DataExportManager:
    """Export and import data"""
    
    @staticmethod
    def get_metadata_as_list() -> List[Dict]:
        """Get all metadata as list format (compatible with pickle format)"""
        try:
            db = get_db()
            results = db.execute_query("""
                SELECT 
                    c.chunk_text as chunk,
                    c.indexed_chunk,
                    c.page,
                    d.filename as document,
                    d.type,
                    d.subject
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.status = 'active'
                ORDER BY c.id
            """)
            
            metadata = [dict(row) for row in results]
            logger.info(f"Exported {len(metadata)} metadata items")
            return metadata
        except Exception as e:
            logger.error(f"Error exporting metadata: {e}")
            return []
