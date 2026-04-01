import re
import PyPDF2


def clean_extracted_text(text):
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r", "\n")

    # Join broken lines inside paragraphs
    text = re.sub(r'(?<![.!?:])\n+', ' ', text)

    # Keep stronger paragraph breaks where they may matter
    text = re.sub(r'\n{2,}', '\n', text)

    # Fix multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove unwanted spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Add space after punctuation if missing
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)

    # Fix common joined academic words
    text = re.sub(r'\bnode tonode\b', 'node-to-node', text, flags=re.IGNORECASE)
    text = re.sub(r'\bend toend\b', 'end-to-end', text, flags=re.IGNORECASE)

    # Remove repeated whitespace again
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_pdf_text(file_path):
    pages = []

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if text:
                cleaned_text = clean_extracted_text(text)

                if len(cleaned_text) > 30:
                    pages.append({
                        "page": page_num + 1,
                        "text": cleaned_text
                    })

    return pages