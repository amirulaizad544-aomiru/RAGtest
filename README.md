# Mini RAG: PDF Q&A over Gemini

A small, un-abstracted RAG (Retrieval-Augmented Generation) pipeline so you
can see every step instead of hiding it behind a framework.

## What RAG actually is, in this project's terms

1. **Chunking** — PDFs are long, and embedding models work best on small
   pieces of text. `chunk_text()` splits each document into ~800-character
   windows with a 100-character overlap, so an idea that straddles a chunk
   boundary still shows up whole somewhere.

2. **Embedding** — each chunk is converted into a vector (a list of numbers)
   that captures its meaning, using a local model
   (`sentence-transformers/all-MiniLM-L6-v2`). Similar meanings end up as
   similar vectors — this is what lets us search by *meaning* instead of
   exact keywords.

3. **Vector store** — here it's just a NumPy array of all chunk vectors. A
   real system would use a proper vector database, but the underlying idea
   (store vectors, search by similarity) is identical.

4. **Retrieval** — your question is embedded the same way, then compared
   against every chunk vector with cosine similarity (a dot product, since
   the vectors are normalized). The top-k most similar chunks are pulled out.

5. **Generation** — the retrieved chunks are "stuffed" into a prompt ahead of
   your question, and sent to Gemini. Gemini answers using that context
   instead of (or in addition to) what it already knows — this is what makes
   answers grounded in *your* documents.

Running the script prints all of this: the chunk count, the retrieved chunks
with their similarity scores, the full prompt sent to Gemini, and the final
answer — so you can see the whole pipeline, not just the output.

## Setup

```
pip install -r requirements.txt
```

Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
(no card required for the free tier — it has generous per-minute/per-day
limits, plenty for this project), then set it:

- PowerShell: `$env:GEMINI_API_KEY = "..."`
- bash: `export GEMINI_API_KEY=...`

Drop one or more PDF files into `data/`.

The first run downloads the embedding model (~80MB) from Hugging Face, so it
needs internet access once.

## Run (CLI)

```
python rag.py "What does the document say about X?"
```

## Run (chat UI)

The pipeline logic lives in `rag_core.py`, shared by the CLI (`rag.py`) and
an API server (`server.py`) that a Next.js chat frontend talks to.

Terminal 1 — API server:

```
uvicorn server:app --reload --port 8000
```

Terminal 2 — frontend:

```
cd frontend
npm install
npm run dev
```

Then open http://localhost:3000 and chat with your PDFs. The frontend calls
`NEXT_PUBLIC_RAG_API_URL` (defaults to `http://localhost:8000`, see
`frontend/.env.local.example`) — no PDFs or Gemini key ever touch the
frontend, only the backend holds those.

## Things to try

- Ask a question the PDF clearly answers — check the retrieved chunks are
  actually relevant and the answer is grounded in them.
- Ask a question the PDF *can't* answer — see how retrieval + Gemini handle
  missing context (this is one of RAG's real limits: it can only answer from
  what got retrieved).
- Change `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `TOP_K` in `rag.py` and see how
  retrieval quality changes.
- Swap `EMBEDDING_MODEL` for a different `sentence-transformers` model and
  compare.
