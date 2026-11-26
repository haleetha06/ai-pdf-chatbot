import os

import streamlit as st
from dotenv import load_dotenv

from rag import Retriever, ask_gemini, build_prompt, chunk_pages, extract_text

load_dotenv()  # reads GEMINI_API_KEY from a local .env file (never committed to GitHub)

st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄")
st.title("📄 AI PDF Chatbot")
st.caption("Upload a PDF and ask questions about it — powered by RAG + Google Gemini")

# ---- Sidebar: settings ----
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Gemini API key", value=os.getenv("GEMINI_API_KEY", ""), type="password",
        help="Free key from https://aistudio.google.com/apikey",
    )
    model = st.text_input("Model", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    top_k = st.slider("Chunks to retrieve", 1, 8, 4)
    st.markdown("---")
    st.markdown("**How it works**\n1. PDF → text\n2. Text → chunks\n3. Find relevant chunks (TF-IDF)\n4. LLM answers from those chunks")

# ---- Upload and index the PDF ----
uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is None:
    st.info("👆 Upload a PDF to get started.")
    st.stop()

if st.session_state.get("file_name") != uploaded.name:
    with st.spinner("Reading and indexing the PDF..."):
        pages = extract_text(uploaded)
        if not pages:
            st.error("No text found. This PDF may be a scanned image.")
            st.stop()
        chunks = chunk_pages(pages)
        st.session_state.retriever = Retriever(chunks)
        st.session_state.file_name = uploaded.name
        st.session_state.messages = []
        st.session_state.stats = (len(pages), len(chunks))

n_pages, n_chunks = st.session_state.stats
st.success(f"✅ Indexed **{uploaded.name}** — {n_pages} pages, {n_chunks} chunks")

# ---- Chat ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        results = st.session_state.retriever.search(question, top_k=top_k)

        if not results:
            answer = "I couldn't find anything related to that in the document."
        elif not api_key:
            answer = ("⚠️ No API key set, so here are the most relevant passages instead:\n\n"
                      + "\n\n".join(f"**Page {r['page']}:** {r['text']}" for r in results))
        else:
            try:
                with st.spinner("Thinking..."):
                    answer = ask_gemini(build_prompt(question, results), api_key, model)
            except Exception as e:
                answer = f"❌ Error from Gemini: {e}"

        st.markdown(answer)
        if results:
            with st.expander("📚 Sources used"):
                for r in results:
                    st.markdown(f"**Page {r['page']}** (relevance {r['score']:.2f})\n\n> {r['text'][:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
