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

Tesseract is always invoked with CREATE_NO_WINDOW so no console/cmd window
flickers in the packaged (.exe) application.  The pytesseract subprocess flags
are applied once at import time via _configure_tesseract_subprocess().
"""
from __future__ import annotations

import logging
import os
import re
import io
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# ── Windows silent-subprocess constants ────────────────────────────────────────
# CREATE_NO_WINDOW (0x08000000) prevents a cmd/console window from appearing
# when tesseract.exe is spawned from a windowed / packaged Python process.
_CREATE_NO_WINDOW = 0x08000000

# ── Tesseract availability (cached after first check) ─────────────────────────
_TESSERACT_AVAILABLE: Optional[bool] = None
_TESSERACT_CONFIGURED: bool = False


def _get_bundled_tesseract_path() -> Optional[str]:
    """
    Return the path to a bundled tesseract.exe when running inside a
    PyInstaller-frozen executable, or None if running in development.

    Convention: the installer places Tesseract-OCR inside
      <app_root>/runtime/tesseract/tesseract.exe
    where <app_root> is the directory that contains backend.exe / launcher.exe.
    """
    if not getattr(sys, "frozen", False):
        return None  # development mode — rely on system PATH

    # sys.executable → …/Application/backend/backend.exe
    # app_root       → …/Application/
    try:
        exe_dir = os.path.dirname(sys.executable)
        app_root = os.path.dirname(exe_dir)
        candidate = os.path.join(app_root, "runtime", "tesseract", "tesseract.exe")
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    return None


def _configure_tesseract_subprocess() -> None:
    """
    Patch pytesseract's internal subprocess call once so that:
      1. The correct tesseract.exe is used in the packaged app.
      2. No console/cmd window ever appears (CREATE_NO_WINDOW + SW_HIDE).

    This must be called before the first pytesseract.image_to_string() call.
    It is idempotent — repeated calls are a no-op.
    """
    global _TESSERACT_CONFIGURED
    if _TESSERACT_CONFIGURED:
        return

    try:
        import pytesseract

        # ── 1. Point to bundled binary if available ────────────────────────────
        bundled = _get_bundled_tesseract_path()
        if bundled:
            pytesseract.pytesseract.tesseract_cmd = bundled
            logger.info(f"[DocExtractor] Using bundled Tesseract: {bundled}")

        # ── 2. Patch subprocess so no window ever appears ─────────────────────
        # pytesseract.run_tesseract() uses subprocess.Popen internally.
        # We monkey-patch the module-level Popen reference so that every call
        # automatically carries CREATE_NO_WINDOW + STARTF_USESHOWWINDOW.
        if sys.platform == "win32":
            _orig_popen = subprocess.Popen

            class _SilentPopen(_orig_popen):  # type: ignore[misc]
                """Subprocess.Popen subclass that always hides the console window."""

                def __init__(self, *args, **kwargs):
                    # Build STARTUPINFO
                    si = kwargs.pop("startupinfo", None) or subprocess.STARTUPINFO()
                    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    si.wShowWindow = subprocess.SW_HIDE
                    kwargs["startupinfo"] = si

                    # Merge creation flags
                    flags = kwargs.pop("creationflags", 0)
                    kwargs["creationflags"] = flags | _CREATE_NO_WINDOW

                    super().__init__(*args, **kwargs)

            # Patch only the subprocess reference inside the pytesseract module
            # so we do not affect the rest of the application.
            import pytesseract.pytesseract as _pt_core
            _pt_core.subprocess.Popen = _SilentPopen  # type: ignore[attr-defined]
            logger.debug("[DocExtractor] Patched pytesseract to use CREATE_NO_WINDOW")

        _TESSERACT_CONFIGURED = True

    except Exception as exc:
        logger.warning(f"[DocExtractor] Could not configure Tesseract subprocess: {exc}")


def _check_tesseract() -> bool:
    """Return True if Tesseract is installed and callable (result is cached)."""
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is not None:
        return _TESSERACT_AVAILABLE
    try:
        _configure_tesseract_subprocess()
        import pytesseract
        pytesseract.get_tesseract_version()
        _TESSERACT_AVAILABLE = True
        logger.info("[DocExtractor] Tesseract OCR is available")
    except Exception as exc:
        _TESSERACT_AVAILABLE = False
        logger.warning(f"[DocExtractor] Tesseract OCR not available: {exc}")
    return _TESSERACT_AVAILABLE


def _ocr_image_bytes(image_bytes: bytes, *, label: str = "<bytes>") -> str:
    """
    Run pytesseract OCR on in-memory image bytes.

    Retries at most once on transient failure.  If the retry also fails,
    the image is silently skipped and an empty string is returned.
    """
    if not _check_tesseract():
        return ""

    max_attempts = 2
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            from PIL import Image
            import pytesseract
            _configure_tesseract_subprocess()
            img = Image.open(io.BytesIO(image_bytes))
            return pytesseract.image_to_string(img).strip()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                logger.warning(
                    f"[DocExtractor] OCR attempt {attempt}/{max_attempts} failed for {label}: {exc} — retrying"
                )
            else:
                logger.warning(
                    f"[DocExtractor] OCR skipped after {max_attempts} attempt(s) for {label}: {exc}"
                )

    return ""


def _ocr_image_file(path: str) -> str:
    """
    Run pytesseract OCR directly on an image file path.

    Retries at most once on transient failure.  If the retry also fails,
    the image is silently skipped.
    """
    if not _check_tesseract():
        logger.warning(
            f"[DocExtractor] Tesseract OCR not available — skipping image {os.path.basename(path)}. "
            "Install Tesseract-OCR to enable image text extraction."
        )
        return ""

    max_attempts = 2
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            from PIL import Image
            import pytesseract
            _configure_tesseract_subprocess()
            img = Image.open(path)
            return pytesseract.image_to_string(img).strip()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                logger.warning(
                    f"[DocExtractor] OCR attempt {attempt}/{max_attempts} failed for "
                    f"{os.path.basename(path)}: {exc} — retrying"
                )
            else:
                logger.warning(
                    f"[DocExtractor] OCR skipped after {max_attempts} attempt(s) for "
                    f"{os.path.basename(path)}: {exc}"
                )

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
                    ocr_text = _ocr_image_bytes(
                        image_bytes,
                        label=f"PDF page {page_idx} img xref={xref}",
                    )
                    if ocr_text and not _is_duplicate(text, ocr_text):
                        page_parts.append(f"[Embedded Image OCR: {ocr_text}]")
            except Exception as img_err:
                logger.warning(
                    f"[DocExtractor] PDF image extraction failed on page {page_idx} of {path}: {img_err}"
                )

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
                    drawings = run._r.findall(
                        './/{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing'
                    )
                    for drawing in drawings:
                        embeds = drawing.findall(
                            './/{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
                        )
                        for embed in embeds:
                            rId = embed.get(
                                '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
                            )
                            if rId and rId in doc.part.related_parts:
                                image_part = doc.part.related_parts[rId]
                                image_bytes = image_part.blob
                                ocr_text = _ocr_image_bytes(
                                    image_bytes, label=f"DOCX para embed rId={rId}"
                                )
                                if ocr_text and not _is_duplicate(para.text, ocr_text):
                                    para_parts.append(f"[Embedded Image OCR: {ocr_text}]")
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
                                drawings = run._r.findall(
                                    './/{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing'
                                )
                                for drawing in drawings:
                                    embeds = drawing.findall(
                                        './/{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
                                    )
                                    for embed in embeds:
                                        rId = embed.get(
                                            '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
                                        )
                                        if rId and rId in doc.part.related_parts:
                                            image_part = doc.part.related_parts[rId]
                                            image_bytes = image_part.blob
                                            ocr_text = _ocr_image_bytes(
                                                image_bytes,
                                                label=f"DOCX table cell embed rId={rId}",
                                            )
                                            if ocr_text and not _is_duplicate(cell_text, ocr_text):
                                                cell_parts.append(f"[Embedded Image OCR: {ocr_text}]")
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


def _process_pptx_shape(shape, slide_idx: int, slide_parts: List[str], extracted_set: Set[str]) -> None:
    """Recursively extract content from a pptx shape."""
    # 1. Group shapes (recurse into sub-shapes)
    if hasattr(shape, "shapes"):
        try:
            for sub_shape in shape.shapes:
                _process_pptx_shape(sub_shape, slide_idx, slide_parts, extracted_set)
        except Exception as grp_err:
            logger.warning(f"[DocExtractor] Failed iterating sub-shapes of group on slide {slide_idx}: {grp_err}")
        return

    # 2. Table shape
    if hasattr(shape, "has_table") and shape.has_table:
        try:
            table = shape.table
            table_text_lines = []
            for row in table.rows:
                row_cells = []
                for cell in row.cells:
                    cell_text = cell.text.strip() if hasattr(cell, "text") else ""
                    if cell_text:
                        row_cells.append(cell_text)
                if row_cells:
                    table_text_lines.append(" | ".join(row_cells))
            if table_text_lines:
                table_text = "[Table]\n" + "\n".join(table_text_lines)
                norm = re.sub(r'\s+', ' ', table_text.lower())
                if not any(norm in x or x in norm for x in extracted_set):
                    extracted_set.add(norm)
                    slide_parts.append(table_text)
        except Exception as tbl_err:
            logger.warning(f"[DocExtractor] Failed extracting table on slide {slide_idx}: {tbl_err}")
        return

    # 3. Chart shape
    if hasattr(shape, "has_chart") and shape.has_chart:
        try:
            chart = shape.chart
            chart_texts = []
            
            # Title
            if chart.has_title:
                try:
                    if chart.chart_title.has_text_frame and chart.chart_title.text_frame.text:
                        chart_texts.append(f"Title: {chart.chart_title.text_frame.text.strip()}")
                except Exception:
                    pass

            # Series & Categories
            try:
                for plot in chart.plots:
                    series_names = [s.name.strip() for s in plot.series if s.name]
                    if series_names:
                        chart_texts.append(f"Series: {', '.join(series_names)}")
                    if hasattr(plot, "categories"):
                        cat_labels = []
                        for cat in plot.categories:
                            if hasattr(cat, "label") and cat.label:
                                cat_labels.append(str(cat.label).strip())
                            elif cat:
                                cat_labels.append(str(cat).strip())
                        if cat_labels:
                            chart_texts.append(f"Categories: {', '.join(cat_labels)}")
            except Exception:
                pass

            # Category Axis Title
            try:
                if hasattr(chart, "category_axis") and chart.category_axis.has_title:
                    axis_title = chart.category_axis.axis_title.text_frame.text.strip()
                    if axis_title:
                        chart_texts.append(f"Category Axis Title: {axis_title}")
            except Exception:
                pass

            if chart_texts:
                chart_text = "[Chart]\n" + "\n".join(chart_texts)
                norm = re.sub(r'\s+', ' ', chart_text.lower())
                if not any(norm in x or x in norm for x in extracted_set):
                    extracted_set.add(norm)
                    slide_parts.append(chart_text)
        except Exception as chart_err:
            logger.warning(f"[DocExtractor] Failed extracting chart on slide {slide_idx}: {chart_err}")
        return

    # 4. Standard shape/Text frame
    if hasattr(shape, "has_text_frame") and shape.has_text_frame:
        try:
            text = shape.text_frame.text.strip()
            if text:
                norm = re.sub(r'\s+', ' ', text.lower())
                if not any(norm in x or x in norm for x in extracted_set):
                    extracted_set.add(norm)
                    slide_parts.append(text)
        except Exception as txt_err:
            logger.warning(f"[DocExtractor] Failed extracting text frame on slide {slide_idx}: {txt_err}")
    elif hasattr(shape, "text") and shape.text.strip():
        try:
            text = shape.text.strip()
            norm = re.sub(r'\s+', ' ', text.lower())
            if not any(norm in x or x in norm for x in extracted_set):
                extracted_set.add(norm)
                slide_parts.append(text)
        except Exception as txt_err:
            logger.warning(f"[DocExtractor] Failed extracting text property on slide {slide_idx}: {txt_err}")

    # 5. Picture shape
    is_picture = False
    try:
        if shape.shape_type == 13 or hasattr(shape, "image"):
            is_picture = True
    except Exception:
        pass

    if is_picture:
        try:
            image_bytes = shape.image.blob
            ocr_text = _ocr_image_bytes(
                image_bytes, label=f"PPTX slide {slide_idx} shape {shape.shape_id}"
            )
            if ocr_text:
                norm = re.sub(r'\s+', ' ', ocr_text.lower())
                if not any(norm in x or x in norm for x in extracted_set):
                    extracted_set.add(norm)
                    slide_parts.append(f"[Embedded Image OCR: {ocr_text}]")
        except Exception as ocr_err:
            logger.warning(f"[DocExtractor] OCR failed on slide {slide_idx} image (shape_id={getattr(shape, 'shape_id', 'unknown')}): {ocr_err}")

    # 6. XML Text fallback (for SmartArt, drawing ML shapes, custom graphic objects, etc.)
    try:
        xml_texts = []
        if hasattr(shape, "element") and shape.element is not None:
            for node in shape.element.iter():
                if node.tag.endswith("}t") and node.text:
                    txt = node.text.strip()
                    if txt:
                        xml_texts.append(txt)
        if xml_texts:
            full_xml_text = " ".join(xml_texts)
            norm = re.sub(r'\s+', ' ', full_xml_text.lower())
            if not any(norm in x or x in norm for x in extracted_set):
                unique_xml_texts = []
                for t in xml_texts:
                    t_norm = re.sub(r'\s+', ' ', t.lower())
                    if not any(t_norm in x or x in t_norm for x in extracted_set):
                        unique_xml_texts.append(t)
                        extracted_set.add(t_norm)
                if unique_xml_texts:
                    slide_parts.append(" ".join(unique_xml_texts))
    except Exception as xml_err:
        logger.debug(f"[DocExtractor] XML text extraction failed on shape: {xml_err}")


def _extract_pptx(path: str) -> str:
    """Extract recursively all slide content (text, tables, charts, SmartArt, notes, pictures OCR) in order."""
    try:
        from pptx import Presentation
        prs = Presentation(path)
        parts = []

        for slide_idx, slide in enumerate(prs.slides, 1):
            slide_parts = []
            extracted_set = set()

            # Process all slide shapes recursively
            for shape in slide.shapes:
                _process_pptx_shape(shape, slide_idx, slide_parts, extracted_set)

            # Process speaker notes
            try:
                if slide.notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        norm = re.sub(r'\s+', ' ', notes.lower())
                        if not any(norm in x or x in norm for x in extracted_set):
                            extracted_set.add(norm)
                            slide_parts.append(f"[Speaker Notes]\n{notes}")
            except Exception as notes_err:
                logger.debug(f"[DocExtractor] Speaker notes extraction failed on slide {slide_idx}: {notes_err}")

            # Combine slide parts (if empty, we still output the slide marker to preserve indices)
            slide_content = "\n\n".join(slide_parts)
            if slide_content:
                parts.append(f"[Slide {slide_idx}]\n{slide_content}")
            else:
                parts.append(f"[Slide {slide_idx}]")

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
    """OCR a standalone image file (PNG, JPG, JPEG, WEBP)."""
    return _ocr_image_file(path)


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
