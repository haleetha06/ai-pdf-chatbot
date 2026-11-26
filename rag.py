"""
Core RAG (Retrieval-Augmented Generation) logic.

Pipeline:
  1. Extract text from the PDF
  2. Split the text into overlapping chunks
  3. Index the chunks with TF-IDF vectors
  4. For a question, retrieve the most similar chunks (cosine similarity)
  5. Send question + retrieved chunks to an LLM (Google Gemini) to write the answer
"""
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text(pdf_file) -> list[dict]:
    """Return a list of {'page': n, 'text': ...} for every page that has text."""
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": " ".join(text.split())})
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 150, overlap: int = 30) -> list[dict]:
    """Split each page into chunks of ~chunk_size words, overlapping by `overlap` words
    so that sentences on a chunk boundary are not lost."""
    chunks = []
    step = chunk_size - overlap
    for p in pages:
        words = p["text"].split()
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + chunk_size])
            if piece:
                chunks.append({"page": p["page"], "text": piece})
            if start + chunk_size >= len(words):
                break
    return chunks


class Retriever:
    """Finds the chunks most relevant to a question using TF-IDF + cosine similarity."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search(self, question: str, top_k: int = 4) -> list[dict]:
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        best = scores.argsort()[::-1][:top_k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in best if scores[i] > 0]


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in context_chunks)
    return (
        "You are a helpful assistant that answers questions about a document.\n"
        "Answer ONLY using the context below. If the answer is not in the context, "
        "say \"I couldn't find that in the document.\" Mention the page numbers you used.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def ask_gemini(prompt: str, api_key: str, model: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text
