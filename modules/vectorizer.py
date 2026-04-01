import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from modules.search_engine import get_semantic_model
from modules.config import VECTORIZER_CONFIG, SEMANTIC_MODEL_NAME, LOGGING_CONFIG

# Setup logging
logging.basicConfig(level=getattr(logging, LOGGING_CONFIG["level"]))
logger = logging.getLogger(__name__)


def create_vectorizer(corpus):
    """
    Create TF-IDF vectorizer and semantic embeddings from corpus
    
    Args:
        corpus: List of text documents/chunks
        
    Returns:
        Tuple of (vectorizer, tfidf_vectors, semantic_vectors)
    """
    try:
        logger.info(f"Creating vectorizer with {len(corpus)} documents")
        
        # Create TF-IDF vectorizer with config
        vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG)
        tfidf_vectors = vectorizer.fit_transform(corpus)
        logger.info(f"Created TF-IDF vectors: {tfidf_vectors.shape}")

        # Create semantic embeddings using singleton model
        semantic_model = get_semantic_model()
        logger.info("Encoding corpus with semantic model...")
        semantic_vectors = semantic_model.encode(corpus, show_progress_bar=True)
        logger.info(f"Created semantic vectors: {semantic_vectors.shape}")

        return vectorizer, tfidf_vectors, semantic_vectors
        
    except Exception as e:
        logger.error(f"Error in create_vectorizer: {e}", exc_info=True)
        raise


def create_semantic_vectors(corpus):
    """
    Create semantic embeddings for a corpus (standalone function)
    
    Args:
        corpus: List of text documents/chunks
        
    Returns:
        Semantic vectors (numpy array)
    """
    try:
        semantic_model = get_semantic_model()
        logger.info(f"Creating semantic vectors for {len(corpus)} documents")
        vectors = semantic_model.encode(corpus, show_progress_bar=True)
        return vectors
    except Exception as e:
        logger.error(f"Error in create_semantic_vectors: {e}", exc_info=True)
        raise
    