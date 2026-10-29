"""Streamlit interface for the RAG technical documentation assistant."""

import os

import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import APIError

from rag_assistant.documents import extract_chunks
from rag_assistant.generation import generate_answer
from rag_assistant.retrieval import GeminiEmbedder, VectorIndex


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "")
if api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
    api_key = ""

st.set_page_config(page_title="Documentation Assistant", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.4rem;
        padding-bottom: 3rem;
    }
    h1 {
        font-size: 2.35rem !important;
        line-height: 1.2 !important;
        letter-spacing: -0.035em !important;
    }
    h2 {
        font-size: 1.45rem !important;
        line-height: 1.3 !important;
    }
    h3 {
        font-size: 1.12rem !important;
    }
    p, label, [data-testid="stCaptionContainer"] {
        font-size: 1rem;
        line-height: 1.6;
    }
    [data-testid="stFileUploader"] {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        background: rgba(128, 128, 128, 0.035);
    }
    [data-testid="stChatMessage"] {
        padding: 1rem 1.2rem;
        border-radius: 14px;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 1rem;
    }
    @media (max-width: 700px) {
        .block-container {
            padding: 1.4rem 1rem 2rem;
        }
        h1 {
            font-size: 1.85rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 Technical Documentation Assistant")
st.markdown(
    "Ask questions about your technical docs and get clear answers grounded in "
    "retrieved passages, with sources you can verify."
)

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []

st.header("Add your documentation")
st.caption("Upload PDF, Markdown, or plain-text files to build a searchable knowledge base.")
uploads = st.file_uploader(
    "Choose documentation files",
    type=["pdf", "md", "txt"],
    accept_multiple_files=True,
    help="Supported formats: PDF, Markdown (.md), and plain text (.txt).",
)

index_col, files_col = st.columns([1, 2], vertical_alignment="center")
with index_col:
    if st.button(
        "Index documents",
        type="primary",
        disabled=not uploads or not api_key,
        use_container_width=True,
    ):
        chunks = []
        try:
            for upload in uploads or []:
                chunks.extend(extract_chunks(upload.name, upload.getvalue()))
            if not chunks:
                st.error("No readable text was found in the selected documents.")
            else:
                with st.spinner("Creating embeddings and indexing passages..."):
                    index = VectorIndex(GeminiEmbedder(api_key))
                    index.add_documents(chunks)
                st.session_state.vector_index = index
                st.session_state.indexed_files = sorted({chunk.source for chunk in chunks})
                st.session_state.messages = []
                st.success(
                    f"Indexed {len(chunks)} passages from "
                    f"{len(st.session_state.indexed_files)} files."
                )
        except (ValueError, UnicodeDecodeError) as error:
            st.error(f"Could not index documents: {error}")
        except APIError as error:
            st.error(f"Gemini document embedding failed: {error}")
        except RuntimeError as error:
            st.error(f"Gemini returned an invalid embedding response: {error}")

with files_col:
    if st.session_state.vector_index is not None:
        file_list = ", ".join(st.session_state.indexed_files)
        st.markdown(
            f"**Indexed:** {st.session_state.vector_index.size} passages "
            f"from {len(st.session_state.indexed_files)} file(s)"
        )
        st.caption(file_list)
    elif not api_key:
        st.info("Add `GEMINI_API_KEY` to the project `.env` file to enable indexing.")
    else:
        st.caption("Your selected files will appear here after indexing.")

if st.session_state.vector_index is None:
    st.info("Upload your files above and select **Index documents** to start asking questions.")

with st.sidebar:
    st.header("Gemini API")
    st.caption(
        "Your Gemini API key is read from the project .env file. Document passages "
        "are sent to Google for embedding and answer generation."
    )
    if not api_key:
        st.info("Set GEMINI_API_KEY in the project .env file to enable indexing and answers.")

    if st.session_state.vector_index is not None:
        st.subheader("Current index")
        st.write(f"{st.session_state.vector_index.size} searchable passages")
        st.write(f"{len(st.session_state.indexed_files)} indexed file(s)")
        if st.button("Clear indexed documents"):
            st.session_state.vector_index = None
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.rerun()
    st.divider()

if st.session_state.vector_index is not None:
    st.header("Ask your documentation")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", []):
                with st.expander(source["label"]):
                    st.write(source["content"])
    question = st.chat_input("Ask a question about your documentation")
else:
    question = None
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documentation and preparing an answer..."):
                results = st.session_state.vector_index.search(question, top_k=5)
                answer = generate_answer(question, results, api_key)
            st.markdown(answer)
            sources = [
                {
                    "label": (
                        f"[{number}] {result.chunk.source}"
                        f"{f' — page {result.chunk.page}' if result.chunk.page is not None else ''}"
                    ),
                    "content": result.chunk.content,
                }
                for number, result in enumerate(results, start=1)
            ]
            for source in sources:
                with st.expander(source["label"]):
                    st.write(source["content"])
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except APIError as error:
            st.error(f"Gemini answer generation failed: {error}")
        except RuntimeError as error:
            st.error(f"Gemini returned an invalid answer response: {error}")
        except ValueError as error:
            st.error(str(error))
