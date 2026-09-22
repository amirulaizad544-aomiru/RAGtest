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
TOP_K = 4         # chunks pulled from the main (non-resume) memory
RESUME_TOP_K = 6  # chunks pulled from the resume memory, only for comparison questions
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

# A line like "EDUCATION" or "WORK EXPERIENCE" is treated as a section header:
# short, mostly uppercase letters, no sentence punctuation.
SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 &/\-]{2,40}$")

# Filename heuristic used to split documents into two separate memories (see
# RagIndex): anything with "resume" or "cv" in its filename goes into the
# resume memory, everything else (employment letter, NDA, ...) goes into the
# main memory that's searched by default.
RESUME_FILENAME_RE = re.compile(r"resume|\bcv\b", re.IGNORECASE)


def is_resume_filename(filename):
    return bool(RESUME_FILENAME_RE.search(filename))


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


CACHE_FIELDS = (
    "document_names",
    "chunks", "sources", "embeddings",                    # main memory
    "resume_chunks", "resume_sources", "resume_embeddings",  # resume memory
)


def load_index_cache(cache_path, fingerprint):
    """Load a previously saved index (main memory + resume memory) from
    cache_path, or return None if there's no cache, it was built from
    different PDFs, or it predates the two-memory split (OCR + embedding are
    slow, so this avoids redoing them on every process restart when data_dir
    hasn't changed)."""
    if not os.path.exists(cache_path):
        return None
    with np.load(cache_path, allow_pickle=True) as data:
        if str(data["fingerprint"]) != fingerprint or not all(f in data for f in CACHE_FIELDS):
            return None
        return {
            "document_names": list(data["document_names"]),
            "chunks": list(data["chunks"]),
            "sources": list(data["sources"]),
            "embeddings": data["embeddings"],
            "resume_chunks": list(data["resume_chunks"]),
            "resume_sources": list(data["resume_sources"]),
            "resume_embeddings": data["resume_embeddings"],
        }


def save_index_cache(cache_path, fingerprint, index):
    np.savez(
        cache_path,
        fingerprint=fingerprint,
        document_names=np.array(index["document_names"], dtype=object),
        chunks=np.array(index["chunks"], dtype=object),
        sources=np.array(index["sources"], dtype=object),
        embeddings=index["embeddings"],
        resume_chunks=np.array(index["resume_chunks"], dtype=object),
        resume_sources=np.array(index["resume_sources"], dtype=object),
        resume_embeddings=index["resume_embeddings"],
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


# Words/phrases used only as a fallback (see classify_wants_resume) if the
# Gemini classification call itself fails, e.g. a network hiccup -- so a
# transient API error doesn't silently make the resume unreachable.
_RESUME_KEYWORDS_FALLBACK = (
    "compare", "comparison", "difference", "differences", "differ",
    "versus", " vs ", " vs. ", "consistent", "inconsist", "match",
    "cross-reference", "cross reference", "discrepanc", "align with", "same as",
    "relevant", "relate", "related", "relation",
)


def _keyword_wants_resume(query):
    lowered = f" {query.lower()} "
    return bool(RESUME_FILENAME_RE.search(query)) or any(k in lowered for k in _RESUME_KEYWORDS_FALLBACK)


def classify_wants_resume(client, query):
    """Ask Gemini whether answering this question needs the resume pulled in
    alongside the main-memory documents -- either to compare/cross-check it
    against them, or because the question is about the resume itself (e.g.
    "is this relevant to my resume?"). Uses a tiny yes/no prompt with a
    5-token cap so the extra round-trip stays fast and cheap.

    Falls back to a keyword heuristic if the classification call itself
    errors (e.g. a network hiccup), rather than silently never using the
    resume for the rest of that request.
    """
    prompt = (
        f'Question: "{query}"\n\n'
        "Does answering this question require the user's resume/CV -- either "
        "to look at it directly, or to compare/cross-check it against other "
        "documents (e.g. an employment letter or NDA)? Reply with exactly one "
        "word: yes or no."
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"max_output_tokens": 5, "temperature": 0},
        )
        return response.text.strip().lower().startswith("y")
    except Exception:
        return _keyword_wants_resume(query)


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
    """Loads PDFs and builds two separate embedding indexes once, so repeated
    queries (e.g. from a chat UI) don't reload the embedding model or
    re-embed the documents on every request:

      - self.chunks/sources/embeddings: the "main memory" -- every document
        except the resume (employment letter, NDA, ...). This is what a
        normal question is answered from.
      - self.resume_chunks/resume_sources/resume_embeddings: the "resume
        memory" -- only pulled in on top of the main memory when the
        question looks like it wants the resume compared/cross-checked
        against the other documents (see is_comparison_query).
    """

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
            self.document_names = cached["document_names"]
            self.chunks, self.sources, self.embeddings = (
                cached["chunks"], cached["sources"], cached["embeddings"],
            )
            self.resume_chunks, self.resume_sources, self.resume_embeddings = (
                cached["resume_chunks"], cached["resume_sources"], cached["resume_embeddings"],
            )
            return

        documents = load_pdfs(self.data_dir)
        self.document_names = list(documents.keys())

        main_documents = {name: text for name, text in documents.items() if not is_resume_filename(name)}
        resume_documents = {name: text for name, text in documents.items() if is_resume_filename(name)}

        self.chunks, self.sources = build_index(main_documents)
        self.embeddings = embed_texts(self.embed_model, self.chunks) if self.chunks else np.empty((0, 0))

        self.resume_chunks, self.resume_sources = build_index(resume_documents)
        self.resume_embeddings = (
            embed_texts(self.embed_model, self.resume_chunks) if self.resume_chunks else np.empty((0, 0))
        )

        save_index_cache(self.cache_path, fingerprint, {
            "document_names": self.document_names,
            "chunks": self.chunks, "sources": self.sources, "embeddings": self.embeddings,
            "resume_chunks": self.resume_chunks, "resume_sources": self.resume_sources,
            "resume_embeddings": self.resume_embeddings,
        })

    def _retrieve_for(self, query, k, resume_k):
        """Top-k chunks from the main memory, plus (only if Gemini judges the
        question needs the resume, and only if a resume was actually found)
        top resume_k chunks from the resume memory, so Gemini has both sides
        to compare."""
        retrieved = (
            retrieve(self.embed_model, query, self.chunks, self.sources, self.embeddings, k=k)
            if self.chunks else []
        )
        if self.resume_chunks and classify_wants_resume(self.genai_client, query):
            retrieved += retrieve(
                self.embed_model, query, self.resume_chunks, self.resume_sources,
                self.resume_embeddings, k=resume_k,
            )
        return retrieved

    def ask(self, query, k=TOP_K, resume_k=RESUME_TOP_K):
        if not self.chunks and not self.resume_chunks:
            raise ValueError(f"No PDFs found in {self.data_dir}")
        retrieved = self._retrieve_for(query, k, resume_k)
        prompt = build_prompt(query, retrieved)
        answer = ask_gemini(self.genai_client, prompt)
        return {
            "answer": answer,
            "retrieved": [
                {"source": source, "score": score, "text": chunk}
                for chunk, source, score in retrieved
            ],
        }

    def ask_stream(self, query, k=TOP_K, resume_k=RESUME_TOP_K):
        """Generator variant of ask(): yields dicts shaped for the NDJSON
        wire format the streaming API endpoint sends to the frontend —
        one "retrieved" event, then zero or more "token" events, then a
        final "done" event."""
        if not self.chunks and not self.resume_chunks:
            raise ValueError(f"No PDFs found in {self.data_dir}")
        retrieved = self._retrieve_for(query, k, resume_k)
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
