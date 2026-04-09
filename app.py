import streamlit as st
from transformers import pipeline

from utils import (
    extract_keywords,
    extract_text_from_file,
    extract_text_from_url,
    chunk_text_by_words,
    clean_text,
    get_mode_lengths,
    simple_bulletize,
)

st.set_page_config(
    page_title="AI Text Summarizer Pro",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #1f2937, #111827);
        padding: 28px;
        border-radius: 20px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.9;
    }

    .feature-card {
        background: #f8fafc;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }

    .summary-box {
        background: #ffffff;
        padding: 20px;
        border-radius: 18px;
        border-left: 6px solid #2563eb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .keyword-chip {
        display: inline-block;
        background: #e0ecff;
        color: #1d4ed8;
        padding: 6px 12px;
        border-radius: 999px;
        margin: 4px 6px 4px 0;
        font-size: 0.9rem;
        font-weight: 500;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3rem;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        box-shadow: 0 6px 16px rgba(37,99,235,0.25);
    }

    div.stDownloadButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3rem;
        font-weight: 600;
        border: 1px solid #d1d5db;
        background: white;
        color: #111827;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Session state defaults
# -----------------------------
defaults = {
    "current_input_text": "",
    "summary_output": "",
    "translated_output": "",
    "last_url": "",
    "url_text": "",
    "last_uploaded_name": "",
    "file_text": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")


@st.cache_resource
def load_translator(model_name: str):
    return pipeline("translation", model=model_name)


TRANSLATION_MODELS = {
    "None": None,
    "English → Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "English → French": "Helsinki-NLP/opus-mt-en-fr",
    "English → German": "Helsinki-NLP/opus-mt-en-de",
    "English → Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English → Italian": "Helsinki-NLP/opus-mt-en-it",
}


# -----------------------------
# Helper functions
# -----------------------------
def summarize_long_text(text: str, mode: str = "Medium") -> str:
    text = clean_text(text)
    if not text:
        return ""

    summarizer = load_summarizer()
    min_len, max_len = get_mode_lengths(mode)

    chunks = list(chunk_text_by_words(text, chunk_size=700))
    partial_summaries = []

    for chunk in chunks:
        if len(chunk.split()) < 40:
            partial_summaries.append(chunk)
            continue

        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )
        partial_summaries.append(result[0]["summary_text"])

    combined = " ".join(partial_summaries)

    if len(combined.split()) > 220:
        result = summarizer(
            combined,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )
        return result[0]["summary_text"]

    return combined


def translate_summary(summary_text: str, translation_choice: str) -> str:
    if translation_choice == "None":
        return summary_text

    model_name = TRANSLATION_MODELS.get(translation_choice)
    if not model_name:
        return summary_text

    translator = load_translator(model_name)

    chunks = list(chunk_text_by_words(summary_text, chunk_size=250))
    translated_parts = []

    for chunk in chunks:
        result = translator(chunk, max_length=300)
        translated_parts.append(result[0]["translation_text"])

    return " ".join(translated_parts)


def reset_outputs():
    st.session_state["summary_output"] = ""
    st.session_state["translated_output"] = ""


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🧠 AI Text Summarizer Pro</div>
        <div class="hero-subtitle">
            Summarize articles, blogs, reports, research papers, PDFs, TXT, and DOCX files with a cleaner premium interface.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.selectbox("Summary Mode", ["Short", "Medium", "Detailed"], index=1)
    output_style = st.selectbox("Output Style", ["Paragraph", "Bullet Points"])
    translation_choice = st.selectbox(
        "Translate Summary",
        list(TRANSLATION_MODELS.keys()),
        index=0,
    )
    show_keywords = st.checkbox("Show Keywords", value=True)
    show_text_preview = st.checkbox("Show Extracted/Input Text Preview", value=True)

# -----------------------------
# Input tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["✍️ Paste Text", "🔗 Article URL", "📄 Upload File"])

with tab1:
    pasted_text = st.text_area(
        "Paste your text here",
        value="",
        height=300,
        placeholder="Paste article, report, blog, or research paper text...",
        key="paste_text_input",
    )

    if st.button("Use Pasted Text"):
        st.session_state["current_input_text"] = clean_text(pasted_text)
        reset_outputs()
        st.success("Pasted text loaded successfully.")

with tab2:
    article_url = st.text_input(
        "Enter article URL",
        value="",
        placeholder="https://example.com/article",
        key="article_url_input",
    )

    if st.button("Extract Text From URL"):
        if not article_url.strip():
            st.warning("Please enter a valid URL.")
        else:
            # If URL changes, wipe old content and summary
            if article_url.strip() != st.session_state["last_url"]:
                st.session_state["url_text"] = ""
                reset_outputs()

            with st.spinner("Extracting article text..."):
                extracted = extract_text_from_url(article_url.strip())

            st.session_state["last_url"] = article_url.strip()
            st.session_state["url_text"] = extracted
            st.session_state["current_input_text"] = extracted
            reset_outputs()

            if extracted.startswith("Could not") or extracted.startswith("URL extraction error"):
                st.error(extracted)
            else:
                st.success("Fresh text extracted from the URL.")

    if show_text_preview and st.session_state["url_text"]:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Extracted URL Text")
        st.text_area(
            "URL Content Preview",
            st.session_state["url_text"],
            height=220,
            key="url_preview",
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    uploaded_file = st.file_uploader(
        "Upload a PDF, TXT, or DOCX file",
        type=["pdf", "txt", "docx"],
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state["last_uploaded_name"]:
            reset_outputs()
            st.session_state["file_text"] = ""
            st.session_state["last_uploaded_name"] = uploaded_file.name

        with st.spinner("Reading uploaded file..."):
            file_text = extract_text_from_file(uploaded_file)

        st.session_state["file_text"] = file_text
        st.session_state["current_input_text"] = file_text
        reset_outputs()

        if file_text.startswith("Unsupported") or "error" in file_text.lower():
            st.error(file_text)
        else:
            st.success(f"Loaded file: {uploaded_file.name}")

    if show_text_preview and st.session_state["file_text"]:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Extracted File Text")
        st.text_area(
            "File Content Preview",
            st.session_state["file_text"],
            height=220,
            key="file_preview",
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### 🚀 Generate Summary")

col_a, col_b = st.columns([3, 1])

with col_a:
    generate_clicked = st.button("Generate Summary")

with col_b:
    clear_clicked = st.button("Clear Output")

if clear_clicked:
    reset_outputs()
    st.session_state["current_input_text"] = ""
    st.session_state["url_text"] = ""
    st.session_state["file_text"] = ""
    st.session_state["last_url"] = ""
    st.session_state["last_uploaded_name"] = ""
    st.rerun()

if generate_clicked:
    text = clean_text(st.session_state["current_input_text"])

    if not text:
        st.warning("Please load text from paste, URL, or file first.")
    elif len(text.split()) < 30:
        st.warning("Text is too short to summarize properly.")
    else:
        with st.spinner("Generating summary..."):
            try:
                summary = summarize_long_text(text, mode=mode)
                translated_summary = translate_summary(summary, translation_choice)

                st.session_state["summary_output"] = summary
                st.session_state["translated_output"] = translated_summary

            except Exception as exc:
                st.error(f"Error: {exc}")

# -----------------------------
# Output section
# -----------------------------
if st.session_state["translated_output"]:
    final_output = st.session_state["translated_output"]
    source_text = st.session_state["current_input_text"]

    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
    st.subheader("📌 Summary Result")

    if output_style == "Bullet Points":
        bullets = simple_bulletize(final_output)
        for bullet in bullets:
            st.markdown(f"- {bullet}")
    else:
        st.write(final_output)

    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="⬇️ Download Summary",
            data=final_output,
            file_name="summary.txt",
            mime="text/plain",
        )

    with c2:
        st.code(final_output, language=None)

    st.markdown('<p class="small-note">You can copy the summary directly from the box above.</p>', unsafe_allow_html=True)

    st.markdown("### 📊 Summary Statistics")
    stat1, stat2, stat3 = st.columns(3)

    original_words = len(source_text.split())
    summary_words = len(final_output.split())
    compression = round((1 - (summary_words / max(original_words, 1))) * 100, 2)

    stat1.metric("Original Words", original_words)
    stat2.metric("Summary Words", summary_words)
    stat3.metric("Compression", f"{compression}%")

    if show_keywords:
        st.markdown("### 🔑 Top Keywords")
        keywords = extract_keywords(source_text, top_n=10)
        chips_html = "".join(
            [f'<span class="keyword-chip">{kw}</span>' for kw in keywords]
        )
        st.markdown(chips_html, unsafe_allow_html=True)