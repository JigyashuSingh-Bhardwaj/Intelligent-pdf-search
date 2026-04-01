import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def clean_sentence(sentence):
    # Remove "Topic 1:" etc
    sentence = re.sub(r'^topic\s*\d+\s*:\s*', '', sentence.lower())
    return sentence.strip().capitalize()


def split_sentences(text):
    text = normalize_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [clean_sentence(s) for s in sentences if len(s.strip()) > 20]


def remove_similar_sentences(sentences, threshold=0.7):
    unique = []

    for s in sentences:
        words = set(re.findall(r'\b\w+\b', s.lower()))
        duplicate = False

        for u in unique:
            u_words = set(re.findall(r'\b\w+\b', u.lower()))
            overlap = len(words & u_words) / max(len(words | u_words), 1)

            if overlap > threshold:
                duplicate = True
                break

        if not duplicate:
            unique.append(s)

    return unique


# 🔥 Extract main concept
def extract_core_query(query):
    q = query.lower()
    stop_words = ["what is", "define", "explain", "describe", "meaning of"]

    for w in stop_words:
        q = q.replace(w, "").strip()

    return q


def detect_query_intent(query):
    q = query.lower()

    if any(w in q for w in ["difference", "compare", "vs"]):
        return "comparison"
    if any(w in q for w in ["layers", "types", "steps"]):
        return "list"
    if any(w in q for w in ["importance", "advantages"]):
        return "importance"
    if any(w in q for w in ["what is", "define", "meaning"]):
        return "definition"

    return "general"


# 🔥 Definition detection
def is_definition_like(sentence, concept):
    s = sentence.lower()
    c = concept.lower()

    patterns = [
        f"{c} is",
        f"the {c} is",
        f"{c} refers to",
        f"{c} is defined as",
        f"{c} can be defined as"
    ]

    return any(p in s for p in patterns)


# 🔥 Detect OSI layer list
def is_layer_sentence(sentence):
    s = sentence.lower()

    keywords = [
        "physical", "data link", "network",
        "transport", "session", "presentation", "application"
    ]

    count = sum(1 for k in keywords if k in s)

    return count >= 3


# 🔥 Bad sentences (noise)
def is_bad_sentence(sentence):
    s = sentence.lower()

    bad_words = [
        "unlike", "whereas", "comparison",
        "important", "advantage", "benefit"
    ]

    return any(w in s for w in bad_words)


def score_sentence(sentence, base_score, concept, intent):
    score = base_score
    s = sentence.lower()

    # 🔥 Strong definition boost
    if is_definition_like(sentence, concept):
        score += 1.0

    if concept in s:
        score += 0.3

    # 🔥 LIST INTENT
    if intent == "list":
        if is_layer_sentence(sentence):
            score += 1.2
        if "layers" in s:
            score += 0.5
        if "tcp/ip" in s or "tcp ip" in s:
            score -= 1.0

    # 🔥 DEFINITION CLEANING
    if intent == "definition" and is_bad_sentence(sentence):
        score -= 0.8

    return score


def format_answer(query, sentences):
    if not sentences:
        return "No proper answer could be generated."

    answer = " ".join(sentences)
    answer = re.sub(r'\s+', ' ', answer)

    return f"{query}:\n\n{answer}\n\n(This answer is generated from your uploaded documents.)"


def build_answer(query, results, max_sentences=4):
    if not results:
        return "No relevant content found."

    intent = detect_query_intent(query)
    concept = extract_core_query(query)

    all_sentences = []

    # 🔹 Extract sentences
    for item in results[:3]:
        chunk = item.get("chunk_text", item.get("chunk", ""))
        sentences = split_sentences(chunk)
        all_sentences.extend(sentences)

    if not all_sentences:
        return "Could not extract meaningful answer."

    all_sentences = remove_similar_sentences(all_sentences)

    # 🔹 Rank sentences
    vectorizer = TfidfVectorizer()
    vecs = vectorizer.fit_transform(all_sentences + [concept])

    query_vec = vecs[-1]
    sent_vecs = vecs[:-1]

    sims = cosine_similarity(query_vec, sent_vecs).flatten()

    ranked = []
    for s, sim in zip(all_sentences, sims):
        ranked.append((s, score_sentence(s, sim, concept, intent)))

    ranked.sort(key=lambda x: x[1], reverse=True)

    selected = []

    # 🔥 FORCE DEFINITION FIRST
    if intent == "definition":
        for s, sc in ranked:
            if is_definition_like(s, concept) and not is_bad_sentence(s):
                selected.append(s)
                break

    # 🔥 FORCE LAYER LIST FIRST
    if intent == "list":
        for s, sc in ranked:
            if is_layer_sentence(s):
                selected.append(s)
                break

    # 🔥 ADD SUPPORTING CLEAN SENTENCES
    for s, sc in ranked:
        if s in selected:
            continue

        if intent == "definition" and is_bad_sentence(s):
            continue

        if intent == "list":
            if "tcp/ip" in s.lower() or "tcp ip" in s.lower():
                continue

        selected.append(s)

        if len(selected) >= max_sentences:
            break

    selected = remove_similar_sentences(selected)
    selected = selected[:max_sentences]

    return format_answer(query, selected)