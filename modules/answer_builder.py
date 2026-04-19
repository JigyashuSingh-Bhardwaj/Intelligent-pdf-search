import logging
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.config import ANSWER_CONFIG, ANSWER_KEYWORDS, LOGGING_CONFIG


logging.basicConfig(level=getattr(logging, LOGGING_CONFIG["level"]))
logger = logging.getLogger(__name__)


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_match(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def tokenize(text):
    return re.findall(r"\b[a-z0-9][a-z0-9/\-]*\b", normalize_for_match(text))


def clean_sentence(sentence):
    sentence = normalize_text(sentence)
    sentence = re.sub(r"^(topic|unit|chapter)\s*\d+\s*:?\s*", "", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\s+", " ", sentence).strip(" -:")
    return sentence


def split_sentences(text):
    text = normalize_text(text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []

    for part in parts:
        cleaned = clean_sentence(part)
        if len(cleaned) < ANSWER_CONFIG["min_sentence_length"]:
            continue
        if is_noise_sentence(cleaned):
            continue
        sentences.append(cleaned)

    return sentences


def remove_similar_sentences(sentences, threshold=None):
    if threshold is None:
        threshold = ANSWER_CONFIG["sentence_similarity_threshold"]

    unique = []
    for sentence in sentences:
        words = set(tokenize(sentence))
        duplicate = False

        for existing in unique:
            existing_words = set(tokenize(existing))
            union = words | existing_words
            overlap = len(words & existing_words) / max(len(union), 1)
            if overlap > threshold:
                duplicate = True
                break

        if not duplicate:
            unique.append(sentence)

    return unique


def get_query_terms(query):
    filler_words = set(ANSWER_KEYWORDS["query_filler_words"])
    return [
        term for term in tokenize(query)
        if len(term) > 1 and term not in filler_words
    ]


def extract_core_query(query):
    q = normalize_for_match(query)
    for phrase in ANSWER_KEYWORDS["stop_words"]:
        q = q.replace(phrase, " ").strip()
    q = re.sub(r"\s+", " ", q).strip()
    return q


def detect_query_intent(query):
    q = normalize_for_match(query)

    if any(word in q for word in ANSWER_KEYWORDS["comparison_words"]):
        return "comparison"
    if any(word in q for word in ANSWER_KEYWORDS["definition_words"]):
        return "definition"
    if any(word in q for word in ANSWER_KEYWORDS["list_words"]):
        return "list"
    if any(word in q for word in ANSWER_KEYWORDS["process_words"]):
        return "process"
    if any(word in q for word in ANSWER_KEYWORDS["importance_words"]):
        return "importance"
    if any(word in q for word in ANSWER_KEYWORDS["explanation_words"]):
        return "explanation"
    return "general"


def is_overview_query(query):
    q = normalize_for_match(query)
    has_model_term = any(word in q for word in ANSWER_KEYWORDS["overview_words"])
    has_explain_term = any(word in q for word in ANSWER_KEYWORDS["explanation_words"] + ANSWER_KEYWORDS["definition_words"])
    return has_model_term and has_explain_term


def extract_comparison_topics(query):
    lowered = normalize_text(query)
    patterns = [
        r"difference between\s+(.+?)\s+and\s+(.+)",
        r"compare\s+(.+?)\s+and\s+(.+)",
        r"(.+?)\s+vs\.?\s+(.+)",
        r"(.+?)\s+versus\s+(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            left = match.group(1).strip(" .,:;")
            right = match.group(2).strip(" .,:;")
            return left, right

    return None, None


def is_definition_like(sentence, concept):
    if not concept:
        return False

    s = normalize_for_match(sentence)
    c = normalize_for_match(concept)
    patterns = [
        f"{c} is",
        f"the {c} is",
        f"{c} refers to",
        f"{c} is defined as",
        f"{c} can be defined as",
    ]
    return any(pattern in s for pattern in patterns)


def is_noise_sentence(sentence):
    lowered = sentence.lower()

    if "http://" in lowered or "https://" in lowered or "www." in lowered:
        return True
    if any(token in lowered for token in ["downloaded from", "all rights reserved", "copyright"]):
        return True
    if re.match(r"^(fig|figure|table|image|diagram)\.?\s*\d+", lowered):
        return True
    if any(token in lowered for token in ["branch code", "course code", "subject code", "faculty name", "submitted by"]):
        return True
    if re.match(r"^(prof|professor|dr)\.?\s+[a-z]", lowered):
        return True
    if re.fullmatch(r"[A-Z0-9 .:/()_-]+", sentence) and len(sentence.split()) <= 6:
        return True
    if any(marker in lowered for marker in ["s. no", "comparison -", "unit -", "unit –"]):
        return True

    return False


def is_layer_sentence(sentence):
    lowered = sentence.lower()
    count = sum(1 for keyword in ANSWER_KEYWORDS["layer_keywords"] if keyword in lowered)
    return count >= 3


def count_layer_mentions(sentence):
    lowered = sentence.lower()
    return sum(1 for keyword in ANSWER_KEYWORDS["layer_keywords"] if keyword in lowered)


def is_overview_sentence(sentence):
    lowered = sentence.lower()
    return (
        count_layer_mentions(sentence) >= 3
        or any(marker in lowered for marker in ANSWER_KEYWORDS["overview_markers"])
    )


def is_comparison_sentence(sentence):
    lowered = sentence.lower()
    return any(marker in lowered for marker in ANSWER_KEYWORDS["comparison_markers"] + ["tcp/ip model", "comparison -", "s. no"])


def is_single_layer_detail_sentence(sentence, concept):
    lowered = sentence.lower()
    matches = [marker for marker in ANSWER_KEYWORDS["detail_layer_markers"] if marker in lowered]
    if len(matches) != 1:
        return False
    if concept and normalize_for_match(concept) in normalize_for_match(sentence):
        return False
    return count_layer_mentions(sentence) < 3


def is_heading_noise_sentence(sentence):
    lowered = sentence.lower()
    return any(marker in lowered for marker in ANSWER_KEYWORDS["heading_noise_markers"])


def is_bad_sentence(sentence):
    lowered = sentence.lower()
    return any(word in lowered for word in ANSWER_KEYWORDS["bad_sentence_words"])


def sentence_query_coverage(sentence, query_terms):
    if not query_terms:
        return 0.0

    sentence_terms = set(tokenize(sentence))
    matched = sum(1 for term in query_terms if term in sentence_terms)
    return matched / len(query_terms)


def compute_similarity_scores(sentences, concept):
    if not sentences:
        return []

    try:
        vectorizer = TfidfVectorizer()
        vecs = vectorizer.fit_transform(sentences + [concept or ""])
        query_vec = vecs[-1]
        sent_vecs = vecs[:-1]
        return cosine_similarity(query_vec, sent_vecs).flatten().tolist()
    except Exception as e:
        logger.error(f"Error in sentence ranking: {e}")
        return [0.4] * len(sentences)


def score_sentence(sentence, concept, intent, query_terms, result_score, similarity, overview_query=False):
    score = float(similarity)
    lowered = sentence.lower()
    coverage = sentence_query_coverage(sentence, query_terms)

    score += coverage * ANSWER_CONFIG["coverage_weight"]
    score += float(result_score or 0) * ANSWER_CONFIG["result_score_weight"]

    if concept and normalize_for_match(concept) in normalize_for_match(sentence):
        score += ANSWER_CONFIG["concept_presence_boost"]

    if concept and is_definition_like(sentence, concept):
        score += ANSWER_CONFIG["definition_similarity_boost"]

    if concept and normalize_for_match(concept) in normalize_for_match(sentence):
        score += ANSWER_CONFIG["query_phrase_boost"]

    if intent == "comparison" and any(marker in lowered for marker in ANSWER_KEYWORDS["comparison_markers"]):
        score += ANSWER_CONFIG["comparison_term_boost"]
    if intent == "process" and any(marker in lowered for marker in ANSWER_KEYWORDS["process_markers"]):
        score += ANSWER_CONFIG["process_term_boost"]
    if intent == "importance" and any(marker in lowered for marker in ANSWER_KEYWORDS["importance_markers"]):
        score += ANSWER_CONFIG["importance_term_boost"]
    if intent == "list" and is_layer_sentence(sentence):
        score += ANSWER_CONFIG["layer_detection_boost"]
    if intent == "definition" and is_bad_sentence(sentence):
        score += ANSWER_CONFIG["bad_sentence_penalty"]
    if is_noise_sentence(sentence):
        score += ANSWER_CONFIG["noise_penalty"]

    if overview_query:
        if is_overview_sentence(sentence):
            score += ANSWER_CONFIG["overview_sentence_boost"]
        if count_layer_mentions(sentence) >= 5:
            score += ANSWER_CONFIG["layer_list_sentence_boost"]
        if is_comparison_sentence(sentence):
            score += ANSWER_CONFIG["comparison_sentence_penalty"]
        if is_single_layer_detail_sentence(sentence, concept):
            score += ANSWER_CONFIG["single_layer_sentence_penalty"]
        if is_heading_noise_sentence(sentence):
            score += ANSWER_CONFIG["heading_sentence_penalty"]

    return score, coverage


def collect_sentence_candidates(query, results):
    intent = detect_query_intent(query)
    concept = extract_core_query(query)
    query_terms = get_query_terms(query)
    overview_query = is_overview_query(query)
    candidates = []

    for idx, item in enumerate(results[:6]):
        chunk = item.get("chunk", "")
        if not chunk or len(chunk.strip()) < 10:
            continue

        sentences = split_sentences(chunk)
        if not sentences:
            continue

        similarities = compute_similarity_scores(sentences, concept)
        for sentence, similarity in zip(sentences, similarities):
            score, coverage = score_sentence(
                sentence=sentence,
                concept=concept,
                intent=intent,
                query_terms=query_terms,
                result_score=item.get("score", 0),
                similarity=similarity,
                overview_query=overview_query,
            )
            candidates.append({
                "sentence": sentence,
                "score": score,
                "coverage": coverage,
                "document": item.get("document", ""),
                "subject": item.get("subject", ""),
                "page": item.get("page", ""),
                "result_rank": idx,
            })

    return candidates, intent, concept, query_terms, overview_query


def dedupe_candidates(candidates):
    seen = []
    unique = []

    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        sentence = candidate["sentence"]
        normalized = normalize_for_match(sentence)
        is_duplicate = False

        for existing in seen:
            left = set(tokenize(normalized))
            right = set(tokenize(existing))
            union = left | right
            overlap = len(left & right) / max(len(union), 1)
            if overlap > ANSWER_CONFIG["sentence_similarity_threshold"]:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(candidate)
            seen.append(normalized)

    return unique


def prioritize_candidates(candidates, intent, concept, query_terms, overview_query=False):
    prioritized = []
    general_pool = []

    for candidate in candidates:
        sentence = candidate["sentence"]
        coverage = candidate["coverage"]

        min_query_coverage = ANSWER_CONFIG["min_query_coverage"]
        if overview_query:
            min_query_coverage = max(min_query_coverage, 0.3)

        if coverage < min_query_coverage and intent != "general":
            continue

        if overview_query and is_overview_sentence(sentence):
            prioritized.append(candidate)
        elif intent == "definition" and is_definition_like(sentence, concept):
            prioritized.append(candidate)
        elif intent == "comparison" and any(marker in sentence.lower() for marker in ANSWER_KEYWORDS["comparison_markers"]):
            prioritized.append(candidate)
        elif intent == "list" and (is_layer_sentence(sentence) or coverage >= 0.5):
            prioritized.append(candidate)
        elif intent == "process" and any(marker in sentence.lower() for marker in ANSWER_KEYWORDS["process_markers"]):
            prioritized.append(candidate)
        elif intent == "importance" and any(marker in sentence.lower() for marker in ANSWER_KEYWORDS["importance_markers"]):
            prioritized.append(candidate)
        else:
            general_pool.append(candidate)

    merged = prioritized + general_pool
    return merged[:ANSWER_CONFIG["max_sentences_to_scan"]]


def build_context_line(selected_candidates):
    documents = []
    for candidate in selected_candidates:
        document = candidate["document"]
        if document and document not in documents:
            documents.append(document)

    if not documents:
        return ""

    shown = ", ".join(documents[:2])
    if len(documents) > 2:
        shown += f" and {len(documents) - 2} more"

    return f"Context: Based on material from {shown}."


def format_definition_answer(query, selected_candidates):
    summary = selected_candidates[0]["sentence"]
    lines = [f"{query}", "", f"Answer: {summary}"]

    supporting = [item["sentence"] for item in selected_candidates[1:4]]
    if supporting:
        lines.append("")
        lines.append("Key Points:")
        for sentence in supporting:
            lines.append(f"- {sentence}")

    context_line = build_context_line(selected_candidates)
    if context_line:
        lines.extend(["", context_line])

    return "\n".join(lines)


def format_list_answer(query, selected_candidates, title="Key Points"):
    lines = [query, ""]
    if selected_candidates:
        lines.append("Answer:")
        lines.append(selected_candidates[0]["sentence"])

    if len(selected_candidates) > 1:
        lines.append("")
        lines.append(f"{title}:")
        for candidate in selected_candidates[1:]:
            lines.append(f"- {candidate['sentence']}")

    context_line = build_context_line(selected_candidates)
    if context_line:
        lines.extend(["", context_line])

    return "\n".join(lines)


def format_comparison_answer(query, selected_candidates):
    left_topic, right_topic = extract_comparison_topics(query)
    lines = [query, "", "Comparison Summary:"]

    if selected_candidates:
        lines.append(selected_candidates[0]["sentence"])

    left_group = []
    right_group = []
    differences = []

    for candidate in selected_candidates[1:]:
        sentence = candidate["sentence"]
        lowered = sentence.lower()

        if left_topic and left_topic.lower() in lowered:
            left_group.append(sentence)
        if right_topic and right_topic.lower() in lowered:
            right_group.append(sentence)
        if any(marker in lowered for marker in ANSWER_KEYWORDS["comparison_markers"]):
            differences.append(sentence)

    if differences:
        lines.extend(["", "Differences:"])
        for sentence in differences[:3]:
            lines.append(f"- {sentence}")

    if left_topic and left_group:
        lines.extend(["", f"{left_topic.strip()}:"])
        for sentence in left_group[:2]:
            lines.append(f"- {sentence}")

    if right_topic and right_group:
        lines.extend(["", f"{right_topic.strip()}:"])
        for sentence in right_group[:2]:
            lines.append(f"- {sentence}")

    context_line = build_context_line(selected_candidates)
    if context_line:
        lines.extend(["", context_line])

    return "\n".join(lines)


def format_overview_answer(query, selected_candidates):
    lines = [query, ""]

    if selected_candidates:
        lines.append("Overview:")
        lines.append(selected_candidates[0]["sentence"])

    supporting = [candidate["sentence"] for candidate in selected_candidates[1:]]
    layer_list = [sentence for sentence in supporting if count_layer_mentions(sentence) >= 5]
    remaining = [sentence for sentence in supporting if sentence not in layer_list]

    if layer_list:
        lines.extend(["", "Seven Layers:"])
        for sentence in layer_list[:2]:
            lines.append(f"- {sentence}")

    if remaining:
        lines.extend(["", "Key Points:"])
        for sentence in remaining[:3]:
            lines.append(f"- {sentence}")

    context_line = build_context_line(selected_candidates)
    if context_line:
        lines.extend(["", context_line])

    return "\n".join(lines)


def format_answer(query, selected_candidates, intent):
    if not selected_candidates:
        return "No proper answer could be generated."

    if is_overview_query(query):
        return format_overview_answer(query, selected_candidates)
    if intent == "definition":
        return format_definition_answer(query, selected_candidates)
    if intent == "comparison":
        return format_comparison_answer(query, selected_candidates)
    if intent == "list":
        return format_list_answer(query, selected_candidates, title="Structured Points")
    if intent == "process":
        return format_list_answer(query, selected_candidates, title="Process Flow")
    if intent == "importance":
        return format_list_answer(query, selected_candidates, title="Why It Matters")
    return format_list_answer(query, selected_candidates, title="Supporting Points")


def fallback_answer(query, results):
    if results and results[0].get("chunk"):
        return f"{query}\n\nRelevant context:\n{results[0]['chunk'][:500]}"
    return "Error generating answer from results."


def build_answer(query, results, max_sentences=None):
    if max_sentences is None:
        max_sentences = ANSWER_CONFIG["max_sentences"]

    try:
        if not results:
            logger.warning(f"No results provided to build_answer for query: {query}")
            return "No relevant content found."

        candidates, intent, concept, query_terms, overview_query = collect_sentence_candidates(query, results)
        if not candidates:
            logger.warning(f"No answer candidates built for query: {query}")
            return fallback_answer(query, results)

        candidates = dedupe_candidates(candidates)
        candidates = prioritize_candidates(candidates, intent, concept, query_terms, overview_query=overview_query)
        selected = candidates[:max_sentences]

        if not selected:
            logger.warning(f"No sentences selected after filtering for query: {query}")
            return fallback_answer(query, results)

        logger.info(f"Generated answer with {len(selected)} sentences for query: {query[:50]}")
        return format_answer(query, selected, intent)

    except Exception as e:
        logger.error(f"Error in build_answer: {e}", exc_info=True)
        return fallback_answer(query, results)
