import io
import re
from collections import Counter

import trafilatura
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(uploaded_file) -> str:
    """Extract text from a PDF file."""
    try:
        pdf_bytes = uploaded_file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

        return clean_text("\n".join(text))
    except Exception as exc:
        return f"PDF read error: {exc}"


def read_txt(uploaded_file) -> str:
    """Read plain text file."""
    try:
        raw = uploaded_file.read()
        return clean_text(raw.decode("utf-8", errors="ignore"))
    except Exception as exc:
        return f"TXT read error: {exc}"


def read_docx(uploaded_file) -> str:
    """Extract text from DOCX."""
    try:
        doc = Document(uploaded_file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return clean_text("\n".join(paragraphs))
    except Exception as exc:
        return f"DOCX read error: {exc}"


def extract_text_from_file(uploaded_file) -> str:
    """Detect file type and extract text."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif filename.endswith(".txt"):
        return read_txt(uploaded_file)
    elif filename.endswith(".docx"):
        return read_docx(uploaded_file)
    else:
        return "Unsupported file format. Please upload PDF, TXT, or DOCX."


def extract_text_from_url(url: str) -> str:
    """Extract main text content from a web article."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "Could not fetch content from the URL."

        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True
        )

        if not extracted:
            return "Could not extract article text from the URL."

        return clean_text(extracted)
    except Exception as exc:
        return f"URL extraction error: {exc}"


def chunk_text_by_words(text: str, chunk_size: int = 700):
    """Split long text into smaller chunks."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def get_mode_lengths(mode: str):
    """Return min and max summary lengths based on mode."""
    mapping = {
        "Short": (30, 80),
        "Medium": (60, 140),
        "Detailed": (100, 220),
    }
    return mapping.get(mode, (60, 140))


def simple_bulletize(summary: str):
    """Convert summary into bullet points using sentence splitting."""
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    bullets = [s.strip() for s in sentences if len(s.strip()) > 10]
    return bullets


def extract_keywords(text: str, top_n: int = 10):
    """Extract top keywords using frequency after stopword removal."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s-]", " ", text)

    words = text.split()
    filtered = [
        w for w in words
        if w not in ENGLISH_STOP_WORDS
        and len(w) > 2
        and not w.isdigit()
    ]

    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]