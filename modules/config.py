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
    "max_sentences": 4,
    "sentence_similarity_threshold": 0.7,
    "definition_similarity_boost": 1.0,
    "concept_presence_boost": 0.3,
    "layer_detection_boost": 1.2,
    "layer_keyword_boost": 0.5,
    "tcp_ip_penalty": -1.0,
    "bad_sentence_penalty": -0.8,
    "min_sentence_length": 20,
}

# ==================== ANSWER BUILDING KEYWORDS ====================
ANSWER_KEYWORDS = {
    "stop_words": ["what is", "define", "explain", "describe", "meaning of"],
    "comparison_words": ["difference", "compare", "comparison", "vs", "versus", "distinguish"],
    "definition_words": ["what is", "define", "definition", "meaning"],
    "list_words": ["layers", "types", "functions", "steps", "components", "features"],
    "importance_words": ["importance", "advantages", "benefits", "why important"],
    "explanation_words": ["explain", "describe", "discuss"],
    "layer_keywords": ["physical", "data link", "network", "transport", "session", "presentation", "application"],
    "bad_sentence_words": ["unlike", "whereas", "comparison", "important", "advantage", "benefit"],
    "tcp_keywords": ["tcp/ip", "tcp ip"],
}

# ==================== PDF PROCESSING ====================
PDF_CONFIG = {
    "min_page_length": 30,  # minimum characters per page to save
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
