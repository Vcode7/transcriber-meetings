"""
doc_extractor.py — Offline text extraction from uploaded agenda/context files.

Supported formats:
  - PDF   : pypdf
  - DOCX  : python-docx
  - PPTX  : python-pptx
  - TXT   : plain read
  - MD    : plain read
  - PNG / JPG / JPEG / WEBP : pytesseract OCR (gracefully skipped if Tesseract not installed)

.doc (old binary Word) is not supported — users must convert to .docx first.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_TESSERACT_AVAILABLE: Optional[bool] = None


def _check_tesseract() -> bool:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is not None:
        return _TESSERACT_AVAILABLE
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _TESSERACT_AVAILABLE = True
    except Exception:
        _TESSERACT_AVAILABLE = False
    return _TESSERACT_AVAILABLE


def _extract_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text.strip())
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"[DocExtractor] PDF extraction failed for {path}: {e}")
        return ""


def _extract_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    parts.append(row_text)
        return "\n".join(parts)
    except Exception as e:
        logger.warning(f"[DocExtractor] DOCX extraction failed for {path}: {e}")
        return ""


def _extract_pptx(path: str) -> str:
    try:
        from pptx import Presentation
        prs = Presentation(path)
        parts = []
        for slide_idx, slide in enumerate(prs.slides, 1):
            slide_texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())
            if slide_texts:
                parts.append(f"[Slide {slide_idx}]\n" + "\n".join(slide_texts))
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"[DocExtractor] PPTX extraction failed for {path}: {e}")
        return ""


def _extract_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"[DocExtractor] Text file read failed for {path}: {e}")
        return ""


def _extract_image_ocr(path: str) -> str:
    if not _check_tesseract():
        logger.warning(
            f"[DocExtractor] Tesseract OCR not available — skipping image {os.path.basename(path)}. "
            "Install Tesseract-OCR to enable image text extraction."
        )
        return ""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        logger.warning(f"[DocExtractor] OCR failed for {path}: {e}")
        return ""


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".webp",
}

UNSUPPORTED_EXTENSIONS = {
    ".doc", ".ppt", ".xls", ".xlsx",
}


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract plain text from a document or image file."""
    ext = os.path.splitext(filename.lower())[1]

    if ext in UNSUPPORTED_EXTENSIONS:
        logger.warning(
            f"[DocExtractor] Unsupported format '{ext}' for '{filename}'. "
            "For .doc use .docx; for .ppt use .pptx."
        )
        return ""

    if ext == ".pdf":
        text = _extract_pdf(file_path)
    elif ext == ".docx":
        text = _extract_docx(file_path)
    elif ext == ".pptx":
        text = _extract_pptx(file_path)
    elif ext in (".txt", ".md"):
        text = _extract_text(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        text = _extract_image_ocr(file_path)
    else:
        logger.warning(f"[DocExtractor] Unknown extension '{ext}' for '{filename}' — skipping.")
        return ""

    logger.info(f"[DocExtractor] Extracted {len(text)} chars from '{filename}' (format={ext})")
    return text


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for change detection."""
    import hashlib
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
