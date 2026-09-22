"""
Small API server exposing the RAG pipeline for the Next.js chat frontend.

Usage:
    pip install -r requirements.txt
    set GEMINI_API_KEY=...          (PowerShell: $env:GEMINI_API_KEY="...")
    put one or more PDFs in ./data
    uvicorn server:app --reload --port 8000
"""

import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_core import RagIndex

app = FastAPI(title="Mini RAG API")

# Next.js dev server runs on localhost:3000 by default, but falls back to
# 3001, 3002, ... if that port is already taken by another project, so allow
# a small range of likely dev ports rather than just 3000.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:300\d",
    allow_methods=["*"],
    allow_headers=["*"],
)

_index = None


def get_index():
    global _index
    if _index is None:
        if not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set on the server.")
        _index = RagIndex()
    return _index


class AskRequest(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")
    index = get_index()
    if not index.chunks:
        raise HTTPException(status_code=400, detail=f"No PDFs found in {index.data_dir}.")
    try:
        return index.ask(req.question)
    except Exception as exc:  # surface Gemini/embedding errors to the frontend
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """Same as /api/ask, but streams the answer token-by-token as newline-
    delimited JSON (NDJSON) instead of waiting for the full response:
        {"type": "retrieved", "chunks": [...]}
        {"type": "token", "text": "..."}      (repeated)
        {"type": "done"}
    or, in place of "done", on failure:
        {"type": "error", "message": "..."}
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")
    index = get_index()
    if not index.chunks:
        raise HTTPException(status_code=400, detail=f"No PDFs found in {index.data_dir}.")

    def event_stream():
        try:
            for event in index.ask_stream(req.question):
                yield json.dumps(event) + "\n"
        except Exception as exc:
            # Headers are already sent by this point, so a failure here has
            # to be signaled inside the stream rather than as an HTTP error.
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
