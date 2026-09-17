"""
CLI entry point for the mini RAG pipeline. See rag_core.py for the pipeline
itself, and server.py for the API used by the Next.js chat frontend.

Usage:
    pip install -r requirements.txt
    set GEMINI_API_KEY=...          (PowerShell: $env:GEMINI_API_KEY="...")
    put one or more PDFs in ./data
    python rag.py "What does the document say about X?"
"""

import os
import sys

from rag_core import RagIndex, DATA_DIR


def main():
    if len(sys.argv) < 2:
        print('Usage: python rag.py "your question"')
        sys.exit(1)
    query = sys.argv[1]

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set. Set it before running.")
        sys.exit(1)

    print(f"Loading PDFs from {DATA_DIR} ...")
    index = RagIndex()
    print(f"Loaded {len(index.document_names)} document(s): {', '.join(index.document_names)}")
    print(f"Split into {len(index.chunks)} chunks.")

    result = index.ask(query)

    print("\n--- Retrieved chunks ---")
    for chunk in result["retrieved"]:
        preview = chunk["text"][:150].replace("\n", " ")
        print(f"[{chunk['score']:.3f}] ({chunk['source']}) {preview}...")

    print("\n--- Answer ---")
    print(result["answer"])


if __name__ == "__main__":
    main()
