import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.config import ANSWER_CONFIG, ANSWER_KEYWORDS, LOGGING_CONFIG

# Setup logging
logging.basicConfig(level=getattr(logging, LOGGING_CONFIG["level"]))
logger = logging.getLogger(__name__)


def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def clean_sentence(sentence):
    # Remove "Topic 1:" etc
    sentence = re.sub(r'^topic\s*\d+\s*:\s*', '', sentence.lower())
    return sentence.strip().capitalize()


def split_sentences(text):
    text = normalize_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [clean_sentence(s) for s in sentences if len(s.strip()) > ANSWER_CONFIG["min_sentence_length"]]


def remove_similar_sentences(sentences, threshold=None):
    if threshold is None:
        threshold = ANSWER_CONFIG["sentence_similarity_threshold"]
    
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


def extract_core_query(query):
    q = query.lower()
    for w in ANSWER_KEYWORDS["stop_words"]:
        q = q.replace(w, "").strip()
    return q


def detect_query_intent(query):
    q = query.lower()

    if any(w in q for w in ANSWER_KEYWORDS["comparison_words"]):
        return "comparison"
    if any(w in q for w in ANSWER_KEYWORDS["list_words"]):
        return "list"
    if any(w in q for w in ANSWER_KEYWORDS["definition_words"]):
        return "definition"
    
    # Detect topology-related queries
    if any(w in q for w in ["topology", "structure", "architecture"]):
        return "topology"

    return "general"


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


def is_advantages_sentence(sentence):
    s = sentence.lower()
    advantages_keywords = ["advantage", "benefit", "beneficial", "easy", "suitable", "good", "efficient", "effective"]
    return any(kw in s for kw in advantages_keywords)


def is_disadvantages_sentence(sentence):
    s = sentence.lower()
    disadvantages_keywords = ["disadvantage", "drawback", "limitation", "problem", "difficult", "hard", "inefficient", "susceptible"]
    return any(kw in s for kw in disadvantages_keywords)


def is_expansion_sentence(sentence):
    s = sentence.lower()
    expansion_keywords = ["expand", "cable", "connect", "join", "growth", "scalable", "extend"]
    return any(kw in s for kw in expansion_keywords)


def is_layer_sentence(sentence):
    s = sentence.lower()
    keywords = ANSWER_KEYWORDS["layer_keywords"]
    count = sum(1 for k in keywords if k in s)
    return count >= 3


def is_bad_sentence(sentence):
    s = sentence.lower()
    bad_words = ANSWER_KEYWORDS["bad_sentence_words"]
    return any(w in s for w in bad_words)


def score_sentence(sentence, base_score, concept, intent):
    score = base_score
    s = sentence.lower()

    # Strong definition boost
    if is_definition_like(sentence, concept):
        score += ANSWER_CONFIG["definition_similarity_boost"]

    if concept in s:
        score += ANSWER_CONFIG["concept_presence_boost"]

    # Topology-specific scoring
    if intent == "topology":
        if is_definition_like(sentence, concept):
            score += 1.5  # Boost definitions
        if is_advantages_sentence(sentence):
            score += 1.0
        if is_disadvantages_sentence(sentence):
            score += 1.0
        if is_expansion_sentence(sentence):
            score += 0.8

    # LIST INTENT
    if intent == "list":
        if is_layer_sentence(sentence):
            score += ANSWER_CONFIG["layer_detection_boost"]
        if "layers" in s:
            score += ANSWER_CONFIG["layer_keyword_boost"]
        if any(tcp in s for tcp in ANSWER_KEYWORDS["tcp_keywords"]):
            score += ANSWER_CONFIG["tcp_ip_penalty"]

    # DEFINITION CLEANING
    if intent == "definition" and is_bad_sentence(sentence):
        score += ANSWER_CONFIG["bad_sentence_penalty"]

    return score


def categorize_sentence(sentence, concept, intent):
    """Categorize sentences into logical groups"""
    if intent == "topology":
        if is_definition_like(sentence, concept):
            return "definition"
        elif is_advantages_sentence(sentence):
            return "advantages"
        elif is_disadvantages_sentence(sentence):
            return "disadvantages"
        elif is_expansion_sentence(sentence):
            return "expansion"
        else:
            return "general"
    elif intent == "definition":
        return "definition"
    elif intent == "list":
        return "list"
    else:
        return "general"


def format_topology_answer(query, categorized_sentences):
    """Format topology-specific answers with organized sections"""
    sections = {
        "definition": [],
        "expansion": [],
        "advantages": [],
        "disadvantages": [],
        "general": []
    }
    
    # Populate sections
    for category, sentences in categorized_sentences.items():
        if sentences and category in sections:
            sections[category] = sentences[:2]  # Limit to 2 per section
    
    # Build answer
    answer_parts = [f"**{query}**\n"]
    
    # Definition
    if sections["definition"]:
        answer_parts.append("**Definition:**")
        for s in sections["definition"]:
            answer_parts.append(f"- {s}")
        answer_parts.append("")
    
    # Expansion characteristics
    if sections["expansion"]:
        answer_parts.append("**Expansion & Connectivity:**")
        for s in sections["expansion"]:
            answer_parts.append(f"- {s}")
        answer_parts.append("")
    
    # Advantages
    if sections["advantages"]:
        answer_parts.append("**Advantages:**")
        for s in sections["advantages"]:
            answer_parts.append(f"- {s}")
        answer_parts.append("")
    
    # Disadvantages
    if sections["disadvantages"]:
        answer_parts.append("**Disadvantages:**")
        for s in sections["disadvantages"]:
            answer_parts.append(f"- {s}")
        answer_parts.append("")
    
    # General info
    if sections["general"]:
        answer_parts.append("**Additional Information:**")
        for s in sections["general"]:
            answer_parts.append(f"- {s}")
        answer_parts.append("")
    
    answer_text = "\n".join(answer_parts)
    answer_text += "\n(This answer is generated from your uploaded documents.)"
    
    return answer_text


def format_answer(query, sentences, intent="general"):
    if not sentences:
        return "No proper answer could be generated."

    # For topology queries, use structured format
    if intent == "topology":
        # Categorize sentences
        categorized = {
            "definition": [],
            "expansion": [],
            "advantages": [],
            "disadvantages": [],
            "general": []
        }
        
        concept = extract_core_query(query)
        for s in sentences:
            category = categorize_sentence(s, concept, intent)
            categorized[category].append(s)
        
        return format_topology_answer(query, categorized)
    
    # Default format
    answer = " ".join(sentences)
    answer = re.sub(r'\s+', ' ', answer)

    return f"{query}:\n\n{answer}\n\n(This answer is generated from your uploaded documents.)"


def build_answer(query, results, max_sentences=None):
    """
    Build a coherent answer from search results using intelligent sentence selection
    with enhanced categorization and formatting based on query intent.
    
    Args:
        query: Original search query
        results: List of search result chunks
        max_sentences: Maximum sentences in answer (default from config)
        
    Returns:
        Formatted answer string
    """
    if max_sentences is None:
        max_sentences = ANSWER_CONFIG["max_sentences"]
    
    try:
        if not results:
            logger.warning(f"No results provided to build_answer for query: {query}")
            return "No relevant content found."
        
        logger.info(f"Building answer from {len(results)} results for query: {query[:50]}")

        intent = detect_query_intent(query)
        concept = extract_core_query(query)
        logger.debug(f"Building answer | Intent: {intent} | Concept: {concept}")

        all_sentences = []

        # Extract sentences from top results
        for idx, item in enumerate(results[:5]):  # Check up to 5 results
            chunk = item.get("chunk", "")
            if not chunk or len(chunk.strip()) < 10:
                logger.debug(f"Skipping result {idx}: empty or too short")
                continue
            
            logger.debug(f"Processing result {idx}: {len(chunk)} chars")
            sentences = split_sentences(chunk)
            logger.debug(f"Extracted {len(sentences)} sentences from result {idx}")
            all_sentences.extend(sentences)

        if not all_sentences:
            logger.warning(f"Could not extract sentences from {len(results)} results for query: {query}")
            # Return the first chunk as-is if no sentences extracted
            if results and results[0].get("chunk"):
                return f"Answer based on search results:\n\n{results[0]['chunk'][:500]}"
            return "Could not extract meaningful answer."

        logger.info(f"Total sentences extracted: {len(all_sentences)}")
        all_sentences = remove_similar_sentences(all_sentences)
        logger.info(f"After removing duplicates: {len(all_sentences)} sentences")

        # Rank sentences
        try:
            vectorizer = TfidfVectorizer()
            vecs = vectorizer.fit_transform(all_sentences + [concept])

            query_vec = vecs[-1]
            sent_vecs = vecs[:-1]

            sims = cosine_similarity(query_vec, sent_vecs).flatten()
        except Exception as e:
            logger.error(f"Error in sentence ranking: {e}")
            sims = [0.5] * len(all_sentences)

        ranked = []
        for s, sim in zip(all_sentences, sims):
            ranked.append((s, score_sentence(s, sim, concept, intent)))

        ranked.sort(key=lambda x: x[1], reverse=True)

        selected = []

        # Intent-specific selection logic
        if intent == "topology":
            # For topology: prioritize different categories
            categories = {
                "definition": [],
                "expansion": [],
                "advantages": [],
                "disadvantages": [],
                "general": []
            }
            
            # Categorize ranked sentences
            for s, sc in ranked:
                category = categorize_sentence(s, concept, intent)
                categories[category].append((s, sc))
            
            # Select from each category
            for cat in ["definition", "expansion", "advantages", "disadvantages", "general"]:
                for s, sc in categories[cat]:
                    if len(selected) >= max_sentences * 2:  # Allow more variety
                        break
                    selected.append(s)
            
            selected = selected[:max_sentences * 2]
        
        elif intent == "definition":
            # FORCE DEFINITION FIRST
            for s, sc in ranked:
                if is_definition_like(s, concept) and not is_bad_sentence(s):
                    selected.append(s)
                    break

            # ADD SUPPORTING CLEAN SENTENCES
            for s, sc in ranked:
                if s in selected:
                    continue
                if is_bad_sentence(s):
                    continue
                selected.append(s)
                if len(selected) >= max_sentences:
                    break
        
        elif intent == "list":
            # FORCE LAYER LIST FIRST
            for s, sc in ranked:
                if is_layer_sentence(s):
                    selected.append(s)
                    break

            # ADD SUPPORTING CLEAN SENTENCES
            for s, sc in ranked:
                if s in selected:
                    continue
                if any(tcp in s.lower() for tcp in ANSWER_KEYWORDS["tcp_keywords"]):
                    continue
                selected.append(s)
                if len(selected) >= max_sentences:
                    break
        
        else:
            # General: just use ranked order
            for s, sc in ranked:
                selected.append(s)
                if len(selected) >= max_sentences:
                    break

        selected = remove_similar_sentences(selected)
        selected = selected[:max_sentences]

        if not selected:
            logger.warning(f"No sentences selected after filtering for query: {query}")
            # Return first chunk as fallback
            if results and results[0].get("chunk"):
                return f"Here's relevant information:\n\n{results[0]['chunk'][:500]}"
            return "Could not generate answer from results."

        logger.info(f"Generated answer with {len(selected)} sentences for query: {query[:50]}")
        return format_answer(query, selected, intent)
        
    except Exception as e:
        logger.error(f"Error in build_answer: {e}", exc_info=True)
        # Fallback: return first chunk
        if results and results[0].get("chunk"):
            return f"Answer (from search results):\n\n{results[0]['chunk'][:500]}"
        return "Error generating answer from results."
