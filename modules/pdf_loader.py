import re
from collections import Counter

import PyPDF2

from modules.config import PDF_CONFIG


NOISE_LINE_PATTERNS = [
    r"https?://",
    r"www\.",
    r"download(ed)? from",
    r"available at",
    r"all rights reserved",
    r"copyright",
    r"page\s+\d+",
    r"^\d+\s*$",
    r"^fig(?:ure)?\.?\s*\d+",
    r"^table\.?\s*\d+",
    r"^image\.?\s*\d+",
    r"^diagram\.?\s*\d+",
    r"^\(?[a-z]\)\s*$",
    r"^department of",
    r"^branch\s*:?",
    r"^branch code\s*:?",
    r"^course code\s*:?",
    r"^subject code\s*:?",
    r"^unit code\s*:?",
    r"^prepared by",
    r"^submitted by",
    r"^faculty name",
    r"^prof(?:essor)?\.?\s+",
    r"^dr\.?\s+[a-z]",
]


def normalize_line(line):
    line = line.replace("\t", " ")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def clean_extracted_text(text):
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\bnode tonode\b", "node-to-node", text, flags=re.IGNORECASE)
    text = re.sub(r"\bend toend\b", "end-to-end", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def looks_like_code_or_branch_line(line):
    lowered = line.lower()
    if any(token in lowered for token in ["branch code", "course code", "subject code", "paper code"]):
        return True
    return bool(re.search(r"\b[A-Z]{2,6}[-/]?\d{2,4}[A-Z]?\b", line))


def looks_like_person_line(line):
    lowered = line.lower()
    if any(token in lowered for token in ["prof.", "professor", "faculty", "lecturer", "submitted by", "prepared by"]):
        return True
    return False


def looks_like_noise_line(line):
    if not line:
        return True

    lowered = line.lower()

    if len(line) < PDF_CONFIG["min_line_length"]:
        return True

    if any(re.search(pattern, lowered) for pattern in NOISE_LINE_PATTERNS):
        return True

    if looks_like_code_or_branch_line(line):
        return True

    if looks_like_person_line(line):
        return True

    word_count = len(line.split())
    if word_count <= 4 and re.fullmatch(r"[A-Z0-9 .:/()_-]+", line):
        return True

    if sum(ch.isdigit() for ch in line) >= max(4, len(line) // 3) and word_count <= 6:
        return True

    return False


def get_repeated_noise_lines(raw_pages):
    counter = Counter()

    for page_text in raw_pages:
        seen_on_page = set()
        for raw_line in page_text.splitlines():
            line = normalize_line(raw_line)
            if not line:
                continue

            canonical = re.sub(r"\b\d+\b", "#", line.lower())
            if canonical in seen_on_page:
                continue

            seen_on_page.add(canonical)
            counter[canonical] += 1

    repeated = set()
    for canonical, count in counter.items():
        if (
            count >= PDF_CONFIG["repeated_line_min_occurrences"]
            and len(canonical) <= PDF_CONFIG["max_repeated_line_length"]
            and looks_like_noise_line(canonical)
        ):
            repeated.add(canonical)

    return repeated


def merge_content_lines(lines):
    merged = []

    for line in lines:
        if not merged:
            merged.append(line)
            continue

        previous = merged[-1]

        if previous.endswith("-"):
            merged[-1] = previous[:-1] + line.lstrip()
            continue

        if previous.endswith((".", "!", "?", ":")):
            merged.append(line)
            continue

        if line[:1].islower():
            merged[-1] = previous + " " + line
            continue

        if len(previous.split()) < 8:
            merged[-1] = previous + " " + line
            continue

        merged.append(line)

    return "\n".join(merged)


def strip_page_noise(page_text, repeated_noise_lines):
    cleaned_lines = []

    for raw_line in page_text.splitlines():
        line = normalize_line(raw_line)
        if not line:
            continue

        canonical = re.sub(r"\b\d+\b", "#", line.lower())
        if canonical in repeated_noise_lines:
            continue

        if looks_like_noise_line(line):
            continue

        cleaned_lines.append(line)

    return merge_content_lines(cleaned_lines)


def extract_pdf_text(file_path):
    pages = []
    raw_pages = []

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            text = page.extract_text() or ""
            raw_pages.append(text)

    repeated_noise_lines = get_repeated_noise_lines(raw_pages)

    for page_num, raw_text in enumerate(raw_pages, start=1):
        page_without_noise = strip_page_noise(raw_text, repeated_noise_lines)
        cleaned_text = clean_extracted_text(page_without_noise)

        if len(cleaned_text) > PDF_CONFIG["min_page_length"]:
            pages.append({
                "page": page_num,
                "text": cleaned_text,
            })

    return pages
