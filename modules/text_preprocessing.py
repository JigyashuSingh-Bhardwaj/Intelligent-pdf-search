import re
import logging
import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)


def _load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
        except Exception as e:
            logger.warning(f"NLTK stopwords resource unavailable: {e}")
            return set()


stop_words = _load_stopwords()


def clean_text_for_indexing(text, remove_stopwords=False):
    """
    Clean text for TF-IDF indexing and query matching.
    Produces compact normalized text for retrieval.
    """
    if not text:
        return ""

    text = text.lower()

    # Remove non-alphanumeric noise
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    # Remove very short noisy tokens
    words = [word for word in words if len(word) > 1]

    if remove_stopwords:
        words = [word for word in words if word not in stop_words]

    return " ".join(words)


def clean_text_for_display(text):
    """
    Clean text for storage, chunking, answer building, and display.
    Preserves punctuation and sentence structure.
    """
    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text).strip()

    # Remove extra spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    return text