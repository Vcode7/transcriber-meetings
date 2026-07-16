"""
doc_extractor.py — Offline text extraction from uploaded agenda/context files.

Supported formats:
  - PDF   : PyMuPDF (fitz) text extraction + PyMuPDF embedded image extraction + Tesseract OCR
  - DOCX  : python-docx text extraction + inline image extraction + Tesseract OCR
  - PPTX  : python-pptx text extraction + picture shape extraction + Tesseract OCR
  - TXT   : plain read
  - MD    : plain read
  - PNG / JPG / JPEG / WEBP : pytesseract OCR
  - XLSX / XLS / CSV        : pandas spreadsheet sheet-by-sheet extraction + row indices tracking
"""
from __future__ import annotations

import logging
import os
import re
import io
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


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run pytesseract OCR on in-memory image bytes."""
    if not _check_tesseract():
        return ""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        logger.warning(f"[DocExtractor] OCR failed for image bytes: {e}")
        return ""


def _is_duplicate(selectable_text: str, ocr_text: str) -> bool:
    """Check if the OCR extracted text is already contained/duplicated in the selectable text."""
    clean_sel = re.sub(r'[^a-z0-9]', '', selectable_text.lower())
    clean_ocr = re.sub(r'[^a-z0-9]', '', ocr_text.lower())
    if not clean_ocr:
        return True
    
    # If the clean OCR text is fully contained in selectable text
    if clean_ocr in clean_sel:
        return True
    
    # Calculate word-level overlap
    ocr_words = ocr_text.lower().split()
    if not ocr_words:
        return True
    sel_words = set(selectable_text.lower().split())
    overlap = sum(1 for w in ocr_words if w in sel_words)
    ratio = overlap / len(ocr_words)
    if ratio > 0.85:
        return True
    return False


def _extract_pdf(path: str) -> str:
    """Extract plain text and run OCR on embedded images in page order."""
    try:
        import fitz
        doc = fitz.open(path)
        parts = []
        
        for page_idx, page in enumerate(doc, 1):
            page_parts = []
            # Extract selectable text
            text = page.get_text()
            if text.strip():
                page_parts.append(text.strip())
            
            # Extract page images
            try:
                images = page.get_images(full=True)
                for img_info in images:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    ocr_text = _ocr_image_bytes(image_bytes)
                    if ocr_text and ocr_text.strip():
                        if not _is_duplicate(text, ocr_text):
                            page_parts.append(f"[Embedded Image OCR: {ocr_text.strip()}]")
            except Exception as img_err:
                logger.warning(f"[DocExtractor] PDF image extraction failed on page {page_idx} of {path}: {img_err}")
                
            if page_parts:
                parts.append("\n\n".join(page_parts))
                
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"[DocExtractor] PDF PyMuPDF extraction failed for {path}: {e}. Falling back to pypdf.")
        # Fallback to pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    parts.append(text.strip())
            return "\n\n".join(parts)
        except Exception as pypdf_err:
            logger.warning(f"[DocExtractor] PDF fallback pypdf extraction also failed: {pypdf_err}")
            return ""


def _extract_docx(path: str) -> str:
    """Extract plain text and run OCR on embedded drawings/images inside paragraphs and tables."""
    try:
        from docx import Document
        doc = Document(path)
        parts = []
        
        # 1. Paragraphs with embedded drawings
        for para in doc.paragraphs:
            para_parts = []
            if para.text.strip():
                para_parts.append(para.text.strip())
            
            # Look for drawings inside runs
            for run in para.runs:
                try:
                    drawings = run._r.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing')
                    for drawing in drawings:
                        embeds = drawing.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                        for embed in embeds:
                            rId = embed.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                            if rId and rId in doc.part.related_parts:
                                image_part = doc.part.related_parts[rId]
                                image_bytes = image_part.blob
                                ocr_text = _ocr_image_bytes(image_bytes)
                                if ocr_text and ocr_text.strip():
                                    if not _is_duplicate(para.text, ocr_text):
                                        para_parts.append(f"[Embedded Image OCR: {ocr_text.strip()}]")
                except Exception:
                    pass
            if para_parts:
                parts.append("\n".join(para_parts))
                
        # 2. Tables with embedded drawings
        for table in doc.tables:
            for row in table.rows:
                cells_text = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    cell_parts = [cell_text] if cell_text else []
                    
                    # Inspect cell paragraphs for runs with drawings
                    for para in cell.paragraphs:
                        for run in para.runs:
                            try:
                                drawings = run._r.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing')
                                for drawing in drawings:
                                    embeds = drawing.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                                    for embed in embeds:
                                        rId = embed.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                                        if rId and rId in doc.part.related_parts:
                                            image_part = doc.part.related_parts[rId]
                                            image_bytes = image_part.blob
                                            ocr_text = _ocr_image_bytes(image_bytes)
                                            if ocr_text and ocr_text.strip():
                                                if not _is_duplicate(cell_text, ocr_text):
                                                    cell_parts.append(f"[Embedded Image OCR: {ocr_text.strip()}]")
                            except Exception:
                                pass
                    if cell_parts:
                        cells_text.append("\n".join(cell_parts))
                row_text = " | ".join(cells_text)
                if row_text.strip():
                    parts.append(row_text)
                    
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"[DocExtractor] DOCX extraction failed for {path}: {e}")
        return ""


def _extract_pptx(path: str) -> str:
    """Extract plain text and run OCR on slide embedded Picture shapes."""
    try:
        from pptx import Presentation
        prs = Presentation(path)
        parts = []
        
        for slide_idx, slide in enumerate(prs.slides, 1):
            slide_parts = []
            
            # Extract normal slide text
            slide_texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())
            selectable_text = "\n".join(slide_texts)
            if slide_texts:
                slide_parts.append(selectable_text)
            
            # Extract embedded pictures & run OCR
            for shape in slide.shapes:
                try:
                    # shape_type 13 is Picture
                    if shape.shape_type == 13:
                        image_bytes = shape.image.blob
                        ocr_text = _ocr_image_bytes(image_bytes)
                        if ocr_text and ocr_text.strip():
                            if not _is_duplicate(selectable_text, ocr_text):
                                slide_parts.append(f"[Embedded Image OCR: {ocr_text.strip()}]")
                except Exception:
                    pass
            
            if slide_parts:
                parts.append(f"[Slide {slide_idx}]\n" + "\n\n".join(slide_parts))
                
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


def _format_dataframe(df: str, filename: str, sheet_name: str) -> str:
    import pandas as pd
    # df is pd.DataFrame
    # Drop completely empty rows and columns
    df = df.dropna(how="all").dropna(how="all", axis=1)
    if df.empty:
        return ""
    
    df = df.fillna("")
    cols = [str(c).strip() for c in df.columns]
    
    block_size = 20
    formatted_blocks = []
    
    for start_idx in range(0, len(df), block_size):
        end_idx = min(start_idx + block_size, len(df))
        df_block = df.iloc[start_idx:end_idx]
        
        row_numbers = [idx + 2 for idx in df_block.index]
        first_row = row_numbers[0]
        last_row = row_numbers[-1]
        
        block_lines = [
            f"Source: {filename}",
            f"Sheet: {sheet_name}",
            f"Rows: {first_row}-{last_row}",
            "---------------------"
        ]
        
        header_line = " | ".join(cols)
        block_lines.append(f"Headers: {header_line}")
        
        for idx, row in df_block.iterrows():
            row_num = idx + 2
            row_vals = [str(val).strip() for val in row.values]
            row_str = " | ".join(row_vals)
            block_lines.append(f"[Row {row_num}] {row_str}")
            
        formatted_blocks.append("\n".join(block_lines))
        
    return "\n\n".join(formatted_blocks)


def _extract_spreadsheet(path: str, filename: str) -> str:
    """Extract worksheets from excel sheets/csv tables, partitioning rows with tracking metadata."""
    import pandas as pd
    ext = os.path.splitext(filename.lower())[1]
    parts = []
    
    try:
        if ext == ".csv":
            sheet_name = os.path.splitext(filename)[0]
            df = pd.read_csv(path)
            formatted = _format_dataframe(df, filename, sheet_name)
            if formatted:
                parts.append(formatted)
        else:
            excel_file = pd.ExcelFile(path)
            for sheet_name in excel_file.sheet_names:
                df = excel_file.parse(sheet_name)
                formatted = _format_dataframe(df, filename, sheet_name)
                if formatted:
                    parts.append(formatted)
                    
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"[DocExtractor] Spreadsheet extraction failed for {filename}: {e}")
        return ""


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".webp",
    ".xlsx", ".xls", ".csv",
}

UNSUPPORTED_EXTENSIONS = {
    ".doc", ".ppt",
}


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract plain text from a document, image, or spreadsheet file."""
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
    elif ext in (".xlsx", ".xls", ".csv"):
        text = _extract_spreadsheet(file_path, filename)
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
