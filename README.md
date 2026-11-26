# 📄 AI PDF Chatbot (RAG + Google Gemini)

An AI chatbot that lets you **upload any PDF and ask questions about it** in plain English. It uses **Retrieval-Augmented Generation (RAG)**: instead of sending the whole document to the AI, it first finds the most relevant parts and then asks the LLM to answer **only from those parts**, with page references.

## ✨ Features

- 📤 Upload any text-based PDF
- 🔍 Finds the most relevant passages using TF-IDF + cosine similarity
- 🤖 Generates answers with Google Gemini, grounded in the document
- 📚 Shows the source pages used for every answer (reduces hallucination)
- 💬 Chat interface with conversation history
- 🔐 API key loaded from a `.env` file — never stored in code

## 🧠 How It Works (RAG Pipeline)

```
PDF ──► Extract text ──► Split into chunks ──► TF-IDF vectors
                                                     │
Question ──► TF-IDF vector ──► Cosine similarity ────┘
                                     │
                          Top-k relevant chunks
                                     │
                     Prompt (question + chunks) ──► Gemini LLM ──► Answer + page sources
```

1. **Text extraction** — `pypdf` reads each page
2. **Chunking** — text is split into ~150-word chunks with 30-word overlap, so sentences at a boundary aren't lost
3. **Retrieval** — chunks and the question are converted into TF-IDF vectors; the chunks with the highest cosine similarity are selected
4. **Generation** — the selected chunks are put into a prompt that tells the LLM to answer only from this context and cite pages

## 🛠️ Tech Stack

Python · Streamlit · Google Gemini API · Scikit-learn (TF-IDF) · pypdf

## ▶️ How to Run

```bash
git clone https://github.com/YOUR-USERNAME/ai-pdf-chatbot.git
cd ai-pdf-chatbot
pip install -r requirements.txt
```

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey), then copy `.env.example` to `.env` and paste your key:

```
GEMINI_API_KEY=your-api-key-here
```

Run the app:

```bash
streamlit run app.py
```

> Without an API key, the app still works in "retrieval only" mode and shows the most relevant passages.

## 📁 Project Structure

```
ai-pdf-chatbot/
├── app.py            # Streamlit user interface
├── rag.py            # RAG logic: extraction, chunking, retrieval, prompt, LLM call
├── requirements.txt
├── .env.example      # Template for API key (real .env is git-ignored)
└── .gitignore
```

## 🚀 Future Improvements

- Use semantic embeddings (e.g. sentence-transformers) + a vector database (FAISS / ChromaDB) for better retrieval
- Support multiple PDFs at once
- OCR support for scanned PDFs
- Deploy on Streamlit Community Cloud
