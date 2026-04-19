"""
Centralized configuration for Intelligent PDF Search
All constants, thresholds, and paths defined here for easy tuning
"""

import os

# ==================== FILE PATHS ====================
DATA_FOLDER = "data"
UPLOAD_FOLDER = "uploads/pdfs"

DATA_PATHS = {
    "metadata": os.path.join(DATA_FOLDER, "metadata.pkl"),
    "vectorizer": os.path.join(DATA_FOLDER, "tfidf_vectorizer.pkl"),
    "vectors": os.path.join(DATA_FOLDER, "vectors.pkl"),
    "semantic_vectors": os.path.join(DATA_FOLDER, "semantic_vectors.pkl")
}

# ==================== SEMANTIC MODEL ====================
SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2'

# ==================== VECTORIZER SETTINGS ====================
VECTORIZER_CONFIG = {
    "ngram_range": (1, 2),
    "stop_words": 'english',
    "min_df": 1,
    "sublinear_tf": True
}

# ==================== SEARCH SETTINGS ====================
SEARCH_CONFIG = {
    "top_k": 5,
    "tfidf_weight": 0.6,
    "semantic_weight": 0.4,
    "min_similarity": 0.01,  # Lowered to capture more results
    "duplicate_threshold": 0.7,
    "jaccard_threshold": 0.8,  # Increased to allow more diverse results
}

# ==================== SEARCH SCORING BOOSTS ====================
SEARCH_SCORE_BOOSTS = {
    "exact_match": 0.30,  # Increased from 0.20
    "core_concept_match": 0.40,  # Increased from 0.25
    "high_term_coverage": 0.25,  # Increased from 0.15
    "optimal_chunk_size": 0.15,  # Increased from 0.05
    "definition_detected": 0.50,  # Increased from 0.40
    "small_chunk_penalty": -0.05,  # Reduced penalty from -0.15
    "overview_chunk_boost": 0.55,
    "layer_list_boost": 0.75,
    "comparison_chunk_penalty": -0.55,
    "single_layer_chunk_penalty": -0.35,
    "heading_noise_penalty": -0.45,
}

# ==================== CHUNK SETTINGS ====================
CHUNK_CONFIG = {
    "chunk_size": 90,  # words per chunk
    "overlap_sentences": 1,
    "min_sentence_length": 20,
    "min_chunk_length": 10,  # minimum characters to save chunk
}

# ==================== ANSWER BUILDING ====================
ANSWER_CONFIG = {
    "max_sentences": 5,
    "sentence_similarity_threshold": 0.7,
    "definition_similarity_boost": 1.0,
    "concept_presence_boost": 0.3,
    "layer_detection_boost": 1.2,
    "layer_keyword_boost": 0.5,
    "tcp_ip_penalty": -1.0,
    "bad_sentence_penalty": -0.8,
    "min_sentence_length": 20,
    "max_sentences_to_scan": 24,
    "min_query_coverage": 0.2,
    "result_score_weight": 0.35,
    "coverage_weight": 1.2,
    "query_phrase_boost": 0.75,
    "comparison_term_boost": 0.6,
    "process_term_boost": 0.5,
    "importance_term_boost": 0.45,
    "noise_penalty": -1.2,
    "overview_sentence_boost": 0.8,
    "layer_list_sentence_boost": 1.0,
    "comparison_sentence_penalty": -0.7,
    "single_layer_sentence_penalty": -0.45,
    "heading_sentence_penalty": -0.5,
}

# ==================== ANSWER BUILDING KEYWORDS ====================
ANSWER_KEYWORDS = {
    "stop_words": ["what is", "define", "explain", "describe", "meaning of"],
    "comparison_words": ["difference", "compare", "comparison", "vs", "versus", "distinguish"],
    "definition_words": ["what is", "define", "definition", "meaning"],
    "list_words": ["layers", "types", "functions", "steps", "components", "features"],
    "importance_words": ["importance", "advantages", "benefits", "why important"],
    "explanation_words": ["explain", "describe", "discuss"],
    "process_words": ["how", "working", "workflow", "process", "procedure", "algorithm"],
    "overview_words": ["model", "overview", "architecture", "layers", "seven layers", "7 layers"],
    "layer_keywords": ["physical", "data link", "network", "transport", "session", "presentation", "application"],
    "bad_sentence_words": ["unlike", "whereas", "comparison", "important", "advantage", "benefit"],
    "tcp_keywords": ["tcp/ip", "tcp ip"],
    "query_filler_words": [
        "what", "which", "who", "where", "when", "why", "how",
        "explain", "describe", "discuss", "define", "list", "write",
        "short", "note", "notes", "about", "with", "and", "the", "for"
    ],
    "comparison_markers": ["difference", "different", "whereas", "while", "compared", "comparison", "unlike"],
    "process_markers": ["step", "process", "procedure", "first", "next", "then", "finally"],
    "importance_markers": ["important", "importance", "benefit", "advantage", "useful", "significant"],
    "overview_markers": ["consists of", "seven layers", "7 layers", "divided into", "model has", "reference model"],
    "heading_noise_markers": ["unit", "introduction", "comparison", "organization of", "s. no"],
    "detail_layer_markers": ["transport layer", "network layer", "data link layer", "physical layer", "session layer", "presentation layer", "application layer"],
}

# ==================== PDF PROCESSING ====================
PDF_CONFIG = {
    "min_page_length": 30,  # minimum characters per page to save
    "max_repeated_line_length": 90,
    "repeated_line_min_occurrences": 2,
    "min_line_length": 3,
}

# ==================== UPLOAD SETTINGS ====================
UPLOAD_CONFIG = {
    "allowed_extensions": {"pdf"},
    "max_file_size_mb": 50,
}

# ==================== LOGGING ====================
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "logs/app.log"
}

# ==================== DATABASE ====================
DATABASE_CONFIG = {
    "type": "sqlite",  # sqlite, mysql, postgresql (future)
    "path": "data/intelligent_pdf_search.db",
    "journal_mode": "WAL",  # Write-Ahead Logging for better concurrency
    "enable_foreign_keys": True,
    "backup_dir": "data/backups",
}

# ==================== SEARCH TERM COVERAGE ====================
SEARCH_TERM_CONFIG = {
    "min_term_length": 1,  # minimum characters in a query term
}

# ==================== PDF UPLOAD VALIDATION ====================
PDF_VALIDATION = {
    "valid_extensions": [".pdf"],
    "min_page_content_length": 30,
}

# ==================== DEFAULT VALUES ====================
DEFAULTS = {
    "doc_type": "notes",
    "subject": "General",
    "search_type": "all",
    "subject_filter": "all",
}
