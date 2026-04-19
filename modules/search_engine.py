import re
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from modules.db_layer import VectorManager
from modules.config import (
    SEMANTIC_MODEL_NAME,
    SEARCH_CONFIG,
    SEARCH_SCORE_BOOSTS,
    SEARCH_TERM_CONFIG,
    ANSWER_KEYWORDS,
    LOGGING_CONFIG
)

# Setup logging
logging.basicConfig(level=getattr(logging, LOGGING_CONFIG["level"]))
logger = logging.getLogger(__name__)

# 🔹 Load semantic model (singleton pattern to avoid duplicates)
_semantic_model = None
_semantic_model_available = True


def get_semantic_model():
    global _semantic_model, _semantic_model_available
    if not _semantic_model_available:
        return None
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading semantic model: {SEMANTIC_MODEL_NAME}")
            _semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        except Exception as e:
            _semantic_model_available = False
            logger.warning(
                "Semantic model unavailable. Falling back to TF-IDF only. Error: %s",
                e
            )
            return None
    return _semantic_model


def semantic_search_available():
    return get_semantic_model() is not None


def get_semantic_vectors():
    if not semantic_search_available():
        return None
    try:
        _, semantic_vectors = VectorManager.get_all_semantic_vectors()
        if semantic_vectors:
            return np.array(
                [np.asarray(vector, dtype=np.float32).flatten() for vector in semantic_vectors],
                dtype=np.float32
            )
        logger.warning("Semantic vectors not found in database. Will fall back to TF-IDF only.")
    except Exception as e:
        logger.error(f"Error loading semantic vectors from database: {e}")
    return None

def normalize_text(text):
    return re.sub(r'\s+', ' ', text.lower()).strip()


def clean_terms(text):
    return re.findall(r'\b\w+\b', normalize_text(text))


def detect_query_intent(query):
    q = normalize_text(query)

    if any(word in q for word in ANSWER_KEYWORDS["comparison_words"]):
        return "comparison"
    if any(word in q for word in ANSWER_KEYWORDS["definition_words"]):
        return "definition"
    if any(word in q for word in ANSWER_KEYWORDS["list_words"]):
        return "list"
    if any(word in q for word in ANSWER_KEYWORDS["importance_words"]):
        return "importance"
    if any(word in q for word in ANSWER_KEYWORDS["explanation_words"]):
        return "explanation"

    return "general"


def extract_core_query(query):
    stop_phrases = ANSWER_KEYWORDS["stop_words"]
    q = normalize_text(query)

    for phrase in stop_phrases:
        q = q.replace(phrase, "").strip()

    return q


def is_definition_chunk(chunk, concept):
    chunk_lower = normalize_text(chunk)
    concept_lower = normalize_text(concept)

    patterns = [
        f"{concept_lower} is",
        f"the {concept_lower} is",
        f"{concept_lower} refers to",
        f"{concept_lower} is defined as",
    ]

    return any(pattern in chunk_lower for pattern in patterns)


def is_overview_query(query):
    q = normalize_text(query)
    has_model_term = any(word in q for word in ANSWER_KEYWORDS["overview_words"])
    has_explain_term = any(word in q for word in ANSWER_KEYWORDS["explanation_words"] + ANSWER_KEYWORDS["definition_words"])
    return has_model_term and has_explain_term


def count_layer_mentions(text):
    lowered = normalize_text(text)
    return sum(1 for keyword in ANSWER_KEYWORDS["layer_keywords"] if keyword in lowered)


def is_layer_overview_chunk(chunk):
    lowered = normalize_text(chunk)
    return (
        count_layer_mentions(lowered) >= 3
        or any(marker in lowered for marker in ANSWER_KEYWORDS["overview_markers"])
    )


def is_comparison_chunk(chunk):
    lowered = normalize_text(chunk)
    return any(marker in lowered for marker in ANSWER_KEYWORDS["comparison_markers"] + ["tcp/ip model", "comparison -", "s. no"])


def is_single_layer_chunk(chunk, core_query):
    lowered = normalize_text(chunk)
    layer_mentions = [marker for marker in ANSWER_KEYWORDS["detail_layer_markers"] if marker in lowered]
    if len(layer_mentions) != 1:
        return False
    if core_query and normalize_text(core_query) in lowered:
        return False
    return count_layer_mentions(lowered) < 3


def is_heading_noise_chunk(chunk):
    lowered = normalize_text(chunk)
    return any(marker in lowered for marker in ANSWER_KEYWORDS["heading_noise_markers"])


def search(query, vectorizer, vectors, metadata, top_k=None, search_type="all", subject_filter="all"):
    """
    Search for relevant chunks based on query using TF-IDF + semantic similarity
    
    Args:
        query: Search query string
        vectorizer: TF-IDF vectorizer object
        vectors: TF-IDF vectors
        metadata: List of chunk metadata
        top_k: Number of results (default from config)
        search_type: Filter by document type
        subject_filter: Filter by subject
        
    Returns:
        List of scored results
    """
    if top_k is None:
        top_k = SEARCH_CONFIG["top_k"]
    
    try:
        query_clean = normalize_text(query)
        query_terms = [term for term in clean_terms(query_clean) if len(term) > SEARCH_TERM_CONFIG["min_term_length"]]

        logger.debug(f"Search query: {query} | Intent: {detect_query_intent(query)}")

        # Convert vectors to proper format for cosine_similarity
        if isinstance(vectors, list):
            # Vectors is a list - need to convert to dense array or sparse matrix
            from scipy.sparse import issparse, vstack
            
            valid_vectors = []
            has_sparse = False
            for v in vectors:
                if issparse(v):
                    valid_vectors.append(v)
                    has_sparse = True
                elif isinstance(v, (list, np.ndarray)):
                    valid_vectors.append(v)
            
            if valid_vectors:
                if has_sparse:
                    # Convert all to dense for consistent handling
                    vectors_list = [v.toarray().flatten() if issparse(v) else np.asarray(v).flatten() for v in valid_vectors]
                else:
                    # Flatten all vectors to ensure consistent shape
                    vectors_list = [np.asarray(v).flatten() for v in valid_vectors]
                
                # Find max length and pad all vectors to that length
                max_len = max(len(v) for v in vectors_list)
                vectors_padded = []
                for v in vectors_list:
                    if len(v) < max_len:
                        # Pad with zeros to match max length
                        v_padded = np.zeros(max_len, dtype=np.float32)
                        v_padded[:len(v)] = v
                        vectors_padded.append(v_padded)
                    else:
                        vectors_padded.append(np.asarray(v, dtype=np.float32))
                
                vectors_array = np.array(vectors_padded, dtype=np.float32)
                logger.debug(f"Vectors padded to shape: {vectors_array.shape}")
            else:
                logger.error("No valid vectors found")
                return []
        else:
            vectors_array = vectors
            logger.debug(f"Vectors already in proper format: {vectors_array.shape if hasattr(vectors_array, 'shape') else 'unknown'}")
        
        logger.info(f"Processing {len(vectors)} vectors for search")

        # 🔹 TF-IDF similarity
        query_vector = vectorizer.transform([query_clean])
        
        # Convert query vector if sparse
        from scipy.sparse import issparse
        if issparse(query_vector):
            query_vector = query_vector.toarray()
        
        tfidf_similarities = cosine_similarity(query_vector, vectors_array).flatten()

        # 🔹 Semantic similarity
        semantic_vectors = get_semantic_vectors()
        semantic_model = get_semantic_model()

        if semantic_vectors is not None and semantic_model is not None:
            query_embedding = semantic_model.encode([query_clean])
            # Ensure semantic_vectors has same length as tfidf_similarities
            if len(semantic_vectors) < len(tfidf_similarities):
                # Pad with zeros if we have fewer semantic vectors than TF-IDF vectors
                padding_size = len(tfidf_similarities) - len(semantic_vectors)
                zero_padding = np.zeros((padding_size, semantic_vectors.shape[1] if hasattr(semantic_vectors, 'shape') else query_embedding.shape[1]))
                semantic_vectors = np.vstack([semantic_vectors, zero_padding])
                logger.warning(f"Padded semantic vectors from {len(semantic_vectors)-padding_size} to {len(semantic_vectors)}")
            
            semantic_similarities = cosine_similarity(query_embedding, semantic_vectors).flatten()
            similarities = (SEARCH_CONFIG["tfidf_weight"] * tfidf_similarities + 
                          SEARCH_CONFIG["semantic_weight"] * semantic_similarities)
            logger.debug("Using hybrid TF-IDF + semantic search")
        else:
            similarities = tfidf_similarities
            logger.warning("Semantic vectors not available, using TF-IDF only")

        intent = detect_query_intent(query)
        core_query = extract_core_query(query)
        overview_query = is_overview_query(query)

        scored_results = []
        
        # Use dynamic min similarity - if no good results, lower threshold
        base_min_similarity = SEARCH_CONFIG["min_similarity"]
        min_sim_threshold = base_min_similarity
        
        for idx, score in enumerate(similarities):
            # If this is a good score, include it
            if score >= min_sim_threshold:
                pass  # Include
            # If very low score, skip (avoid noise)
            elif score < 0.01:
                continue
            # For borderline cases, include them (better to have more results)
            else:
                pass  # Include borderline results
                
            item = metadata[idx]

            item_type = item.get("type", "").lower()
            item_subject = item.get("subject", "General").strip().lower()

            chunk = item.get("chunk_text", item.get("chunk", ""))
            indexed_chunk = item.get("indexed_chunk", chunk)

            chunk_clean = normalize_text(indexed_chunk)
            chunk_words = set(clean_terms(chunk_clean))

            if search_type != "all" and item_type != search_type.lower():
                continue

            if subject_filter != "all" and item_subject != subject_filter.strip().lower():
                continue

            final_score = float(score)
            reasons = []

            # Exact match
            if query_clean in chunk_clean:
                final_score += SEARCH_SCORE_BOOSTS["exact_match"]
                reasons.append("Exact match")

            # Core concept
            if core_query and core_query in chunk_clean:
                final_score += SEARCH_SCORE_BOOSTS["core_concept_match"]
                reasons.append("Core concept match")

            # Coverage
            matched_terms = sum(1 for term in query_terms if term in chunk_words)
            if query_terms:
                coverage_ratio = matched_terms / len(query_terms)
                final_score += SEARCH_SCORE_BOOSTS["high_term_coverage"] * coverage_ratio
                if coverage_ratio > 0.5:
                    reasons.append("High term coverage")

            # Chunk size
            chunk_word_count = len(chunk.split())
            if chunk_word_count < 20:
                final_score += SEARCH_SCORE_BOOSTS["small_chunk_penalty"]
            elif 35 <= chunk_word_count <= 110:
                final_score += SEARCH_SCORE_BOOSTS["optimal_chunk_size"]

            # Intent boosts
            if intent == "definition":
                if is_definition_chunk(chunk, core_query):
                    final_score += SEARCH_SCORE_BOOSTS["definition_detected"]
                    reasons.append("Definition detected")

            if overview_query:
                if is_layer_overview_chunk(chunk):
                    final_score += SEARCH_SCORE_BOOSTS["overview_chunk_boost"]
                    reasons.append("Overview chunk")
                if count_layer_mentions(chunk) >= 5:
                    final_score += SEARCH_SCORE_BOOSTS["layer_list_boost"]
                    reasons.append("Layer list")
                if is_comparison_chunk(chunk):
                    final_score += SEARCH_SCORE_BOOSTS["comparison_chunk_penalty"]
                    reasons.append("Comparison penalty")
                if is_single_layer_chunk(chunk, core_query):
                    final_score += SEARCH_SCORE_BOOSTS["single_layer_chunk_penalty"]
                    reasons.append("Single-layer penalty")
                if is_heading_noise_chunk(chunk):
                    final_score += SEARCH_SCORE_BOOSTS["heading_noise_penalty"]
                    reasons.append("Heading penalty")

            scored_results.append({
                "score": round(final_score, 4),
                "base_score": round(float(score), 4),
                "chunk": chunk,
                "page": item.get("page", ""),
                "document": item.get("document", ""),
                "type": item.get("type", "unknown"),
                "subject": item.get("subject", "General"),
                "reasons": reasons
            })

        # 🔹 Sort results
        scored_results.sort(key=lambda x: x["score"], reverse=True)

        # 🔹 Remove duplicates
        unique_results = []
        seen_chunks = []

        for result in scored_results:
            chunk_text = normalize_text(result["chunk"])

            is_duplicate = False

            for seen in seen_chunks:
                overlap = len(set(chunk_text.split()) & set(seen.split()))
                union = len(set(chunk_text.split()) | set(seen.split()))
                jaccard_sim = overlap / union if union else 0

                prefix_sim = chunk_text[:120] == seen[:120]

                if jaccard_sim > SEARCH_CONFIG["jaccard_threshold"] or prefix_sim:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)
                seen_chunks.append(chunk_text)

            if len(unique_results) >= top_k:
                break

        logger.info(f"Search returned {len(unique_results)} results for query: {query[:50]}")
        return unique_results
        
    except Exception as e:
        logger.error(f"Error in search function: {e}", exc_info=True)
        return []
