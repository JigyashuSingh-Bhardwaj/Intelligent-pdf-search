import re

def is_heading(sentence):
    sentence = sentence.strip().lower()
    patterns = [
        r'^topic\s+\d+\s*:',
        r'^unit\s+\d+\s*:',
        r'^chapter\s+\d+\s*:',
        r'^\d+\.\s+[a-zA-Z][a-zA-Z\s\-]+$'
    ]
    return any(re.match(pattern, sentence) for pattern in patterns)


def split_into_chunks(text, chunk_size=90, overlap_sentences=1):
    if not text or not text.strip():
        return []

    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # If a new heading starts, close previous chunk first
        if is_heading(sentence) and current_chunk:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_length = sentence_word_count
            continue

        if current_chunk and current_length + sentence_word_count > chunk_size:
            chunks.append(" ".join(current_chunk).strip())

            if overlap_sentences > 0:
                current_chunk = current_chunk[-overlap_sentences:]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks