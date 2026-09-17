"""
Core RAG (Retrieval-Augmented Generation) pipeline, shared by the CLI
(rag.py) and the API server (server.py).

    PDFs -> section-aware chunks -> embeddings -> vector store
                                        |
    question -> embed -> cosine similarity -> top-k chunks
                                        |
                          prompt = question + retrieved chunks
                                        |
                                 Gemini -> answer
"""

import os
import re
import glob

import numpy as np
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

import ocr_core

# Loads GEMINI_API_KEY (and anything else) from a .env file next to this
# file, if present, so you don't have to re-export it in every new terminal.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(os.path.dirname(__file__), ".rag_index_cache.npz")
CHUNK_SIZE = 800   # characters per chunk, used when a section is too big to keep whole
CHUNK_OVERLAP = 100
TOP_K = 4
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

# A line like "EDUCATION" or "WORK EXPERIENCE" is treated as a section header:
# short, mostly uppercase letters, no sentence punctuation.
SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 &/\-]{2,40}$")


def normalize_whitespace(text):
    """Collapse runs of whitespace (including the character-by-character
    spacing some PDF templates produce, e.g. "U n i v e r s i t y") down to
    single spaces, without merging separate lines together."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines)


def load_pdfs(data_dir=DATA_DIR):
    """Read every PDF in data_dir and return {filename: full_text}.

    Uses pdfplumber rather than pypdf because it's generally better at
    preserving reading order for multi-column layouts (common in resumes),
    and its extract_text() already groups characters into words based on
    their positions on the page.

    Scanned PDFs have no embedded text layer, so extract_text() comes back
    empty for them; ocr_core.ocr_pdf() is used as a fallback in that case.
    """
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))
    documents = {}
    for path in pdf_paths:
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if ocr_core.is_text_empty(text):
            text = ocr_core.ocr_pdf(path)
        documents[os.path.basename(path)] = normalize_whitespace(text)
    return documents


def fingerprint_pdfs(data_dir):
    """Cheap signature of the PDFs in data_dir (name, size, mtime), used to
    tell whether a cached index is stale without re-reading file contents."""
    pdf_paths = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    parts = [
        f"{os.path.basename(path)}:{os.path.getsize(path)}:{int(os.path.getmtime(path))}"
        for path in pdf_paths
    ]
    return "|".join(parts)


def load_index_cache(cache_path, fingerprint):
    """Load a previously saved (document_names, chunks, sources, embeddings)
    tuple from cache_path, or return None if there's no cache or it was built
    from different PDFs (OCR + embedding are slow, so this avoids redoing
    them on every process restart when data_dir hasn't changed)."""
    if not os.path.exists(cache_path):
        return None
    with np.load(cache_path, allow_pickle=True) as data:
        if str(data["fingerprint"]) != fingerprint:
            return None
        return (
            list(data["document_names"]),
            list(data["chunks"]),
            list(data["sources"]),
            data["embeddings"],
        )


def save_index_cache(cache_path, fingerprint, document_names, chunks, sources, embeddings):
    np.savez(
        cache_path,
        fingerprint=fingerprint,
        document_names=np.array(document_names, dtype=object),
        chunks=np.array(chunks, dtype=object),
        sources=np.array(sources, dtype=object),
        embeddings=embeddings,
    )


def is_section_header(line):
    stripped = line.strip()
    if not stripped or len(stripped) > 40:
        return False
    letters = [c for c in stripped if c.isalpha()]
    # require a handful of letters so short acronyms/dates aren't mistaken
    # for headers, and require the line to be all-uppercase.
    return len(letters) >= 3 and stripped == stripped.upper() and bool(SECTION_HEADER_RE.match(stripped))


def split_into_sections(text):
    """Split a document into (header, body) sections using ALL-CAPS lines
    (e.g. "EDUCATION", "SKILLS", "EXPERIENCE") as section boundaries. Text
    before the first detected header is kept under a generic header."""
    sections = []
    current_header = "General"
    current_lines = []
    for line in text.splitlines():
        if is_section_header(line):
            if current_lines:
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))
    return [(h, b) for h, b in sections if b]


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunking by character count, with overlap so
    an idea that spans a chunk boundary still appears whole in one chunk.

    A trailing remainder no bigger than the overlap is folded into the
    previous chunk instead of becoming its own chunk: that remainder is
    already covered by the previous chunk's overlap window, so splitting it
    out would only produce a tiny, often mid-word, near-content-free chunk
    (e.g. a 1406-char section with chunk_size=800/overlap=100 would
    otherwise end with a stray final chunk of just "ience.").
    """
    chunks = []
    start = 0
    while start < len(text):
        remaining = len(text) - start
        if chunks and remaining <= overlap:
            leftover = text[start:].strip()
            if leftover:
                chunks[-1] = (chunks[-1] + " " + leftover).strip()
            break
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_section(header, body, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk a single section, keeping it whole if it fits, and always
    prefixing chunks with their section header so retrieval still knows
    which part of the resume/document a chunk came from even after the
    surrounding structure is gone."""
    if len(body) <= chunk_size:
        return [f"{header}\n{body}"]
    return [f"{header}\n{piece}" for piece in chunk_text(body, chunk_size, overlap)]


def build_index(documents):
    """Turn {filename: text} into a flat list of chunks with source labels,
    splitting each document into sections first so a chunk doesn't straddle
    unrelated parts of the document (e.g. SKILLS bleeding into EDUCATION)."""
    chunks = []
    sources = []
    for filename, text in documents.items():
        for header, body in split_into_sections(text):
            for c in chunk_section(header, body):
                chunks.append(c)
                sources.append(filename)
    return chunks, sources


def embed_texts(model, texts):
    """Batch-encode texts into normalized embedding vectors."""
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings)


def retrieve(model, query, chunks, sources, embeddings, k=TOP_K):
    """Embed the query and return the top-k most similar chunks (cosine
    similarity — since embeddings are normalized, this is just a dot product)."""
    query_embedding = embed_texts(model, [query])[0]
    scores = embeddings @ query_embedding  # cosine similarity, shape (num_chunks,)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(chunks[i], sources[i], float(scores[i])) for i in top_indices]


def build_prompt(query, retrieved):
    """Stuff the retrieved chunks into a context block ahead of the question."""
    context = "\n\n".join(
        f"[Source: {source}]\n{chunk}" for chunk, source, _ in retrieved
    )
    return f"""Answer the question using only the context below. \
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {query}"""


def ask_gemini(client, prompt):
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def ask_gemini_stream(client, prompt):
    """Streaming variant of ask_gemini: yields text fragments as they arrive
    instead of waiting for the full response."""
    for chunk in client.models.generate_content_stream(model=GEMINI_MODEL, contents=prompt):
        if chunk.text:
            yield chunk.text


class RagIndex:
    """Loads PDFs and builds the embedding index once, so repeated queries
    (e.g. from a chat UI) don't reload the embedding model or re-embed the
    documents on every request."""

    def __init__(self, data_dir=DATA_DIR, cache_path=CACHE_PATH):
        self.data_dir = data_dir
        self.cache_path = cache_path
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.genai_client = genai.Client()
        self.reload()

    def reload(self):
        """Rebuild the index, unless a cache on disk already matches the
        current PDFs in data_dir -- OCR and embedding are slow, so a
        matching cache is loaded instead of redoing that work on every
        process restart. Adding, removing, or replacing a PDF changes its
        fingerprint, which invalidates the cache automatically."""
        fingerprint = fingerprint_pdfs(self.data_dir)
        cached = load_index_cache(self.cache_path, fingerprint)
        if cached:
            self.document_names, self.chunks, self.sources, self.embeddings = cached
            return

        documents = load_pdfs(self.data_dir)
        self.document_names = list(documents.keys())
        self.chunks, self.sources = build_index(documents)
        if self.chunks:
            self.embeddings = embed_texts(self.embed_model, self.chunks)
        else:
            self.embeddings = np.empty((0, 0))
        save_index_cache(
            self.cache_path, fingerprint, self.document_names, self.chunks, self.sources, self.embeddings
        )

    def ask(self, query, k=TOP_K):
        if not self.chunks:
            raise ValueError(f"No PDFs found in {self.data_dir}")
        retrieved = retrieve(
            self.embed_model, query, self.chunks, self.sources, self.embeddings, k=k
        )
        prompt = build_prompt(query, retrieved)
        answer = ask_gemini(self.genai_client, prompt)
        return {
            "answer": answer,
            "retrieved": [
                {"source": source, "score": score, "text": chunk}
                for chunk, source, score in retrieved
            ],
        }

    def ask_stream(self, query, k=TOP_K):
        """Generator variant of ask(): yields dicts shaped for the NDJSON
        wire format the streaming API endpoint sends to the frontend —
        one "retrieved" event, then zero or more "token" events, then a
        final "done" event."""
        if not self.chunks:
            raise ValueError(f"No PDFs found in {self.data_dir}")
        retrieved = retrieve(
            self.embed_model, query, self.chunks, self.sources, self.embeddings, k=k
        )
        yield {
            "type": "retrieved",
            "chunks": [
                {"source": source, "score": score, "text": chunk}
                for chunk, source, score in retrieved
            ],
        }
        prompt = build_prompt(query, retrieved)
        for fragment in ask_gemini_stream(self.genai_client, prompt):
            yield {"type": "token", "text": fragment}
        yield {"type": "done"}
