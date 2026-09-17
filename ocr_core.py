"""
OCR fallback for scanned PDFs.

pdfplumber's extract_text() (used in rag_core.load_pdfs) only reads a PDF's
embedded text layer. A scanned PDF is just page images with no text layer,
so extract_text() silently returns nothing for it. This module renders such
pages to images and runs OCR on them instead.

Requires the Tesseract OCR engine, which is a separate system install (not
a pip package):
    Windows: https://github.com/UB-Mannheim/tesseract/wiki
    macOS:   brew install tesseract
    Linux:   apt install tesseract-ocr

If tesseract.exe isn't on PATH, point to it with a TESSERACT_CMD env var
(e.g. in .env: TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe).
"""

import io
import os

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

OCR_DPI = 300
MIN_TEXT_CHARS = 20  # below this, treat a pdfplumber extraction as empty/noise


def is_text_empty(text, min_chars=MIN_TEXT_CHARS):
    """Heuristic for "this PDF has no real text layer": pdfplumber returned
    next to nothing, so it's most likely a scan rather than a short document."""
    return len((text or "").strip()) < min_chars


def ocr_pdf(path, dpi=OCR_DPI):
    """Render every page of a PDF to an image and OCR it, returning the
    concatenated text. Used as a fallback for scanned PDFs."""
    tesseract_cmd = os.environ.get("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    zoom = dpi / 72  # PDF points are 72 per inch
    matrix = fitz.Matrix(zoom, zoom)
    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            text_parts.append(pytesseract.image_to_string(image))
    return "\n".join(text_parts)
