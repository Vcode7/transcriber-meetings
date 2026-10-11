"""Scanned-PDF extraction: page classification, rendering, OCR and reading order."""
import io
import os

import pytest
from PIL import Image, ImageDraw, ImageFont

from services.doc_extractor import _extract_pdf, extract_text_from_file, ocr_pdf_pages_detailed
from services.ocr_settings import OcrSettings, use_ocr_settings

fitz = pytest.importorskip("fitz")


@pytest.fixture(autouse=True)
def _pin_default_ocr_settings():
    with use_ocr_settings(OcrSettings()):
        yield


def _font(size: int = 40):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _scan_jpeg(lines, size=(1240, 1754)) -> bytes:
    """Simulate a 150-dpi A4 scan as a JPEG (what scanners embed in PDFs)."""
    img = Image.new("RGB", size, (250, 250, 248))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((120, 160 + i * 120), line, fill=(20, 20, 20), font=_font())
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _build_pdf(path, *, rotate_scan=False, stamped_header=False):
    doc = fitz.open()

    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Page one has a real text layer about budgets.", fontsize=12)
    page.insert_text((72, 120), "It lists spending by department for the quarter.", fontsize=12)

    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=_scan_jpeg(["Scanned Minutes Alpha", "Decision approved by board"]))
    if stamped_header:
        page.insert_text((40, 30), "CONFIDENTIAL - internal use only - page 2", fontsize=9)

    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=_scan_jpeg(["Second Scan Bravo", "Follow up next quarter"]))
    if rotate_scan:
        page.set_rotation(90)

    doc.save(path)
    doc.close()


def test_scanned_pages_are_ocrd_in_page_order(tmp_path):
    pdf = str(tmp_path / "mixed.pdf")
    _build_pdf(pdf, stamped_header=True)

    text = _extract_pdf(pdf)

    assert "real text layer" in text
    assert "Alpha" in text and "Bravo" in text
    # Page order preserved: text page, scan 1, scan 2
    assert text.index("real text layer") < text.index("Alpha") < text.index("Bravo")
    # Scanned pages become page text, not embedded-image annotations
    assert "[Embedded Image OCR" not in text
    # The document was closed: the file can be deleted (Windows file locking)
    os.remove(pdf)


def test_rotated_scan_page_is_rendered_upright(tmp_path):
    pdf = str(tmp_path / "rotated.pdf")
    _build_pdf(pdf, rotate_scan=True)

    with use_ocr_settings(OcrSettings(auto_orient=True)):
        text = extract_text_from_file(pdf, "rotated.pdf")

    assert "Bravo" in text or "quarter" in text


def test_never_mode_skips_page_rendering(tmp_path):
    pdf = str(tmp_path / "never.pdf")
    _build_pdf(pdf)

    _, details, total = ocr_pdf_pages_detailed(pdf, OcrSettings(pdf_ocr_mode="never"), max_pages=10)

    assert total == 3
    assert all(d["source"] == "text_layer" for d in details.values())
    assert details[2]["render_dpi"] is None


def test_detailed_results_report_ocr_pages(tmp_path):
    pdf = str(tmp_path / "detail.pdf")
    _build_pdf(pdf)

    text, details, total = ocr_pdf_pages_detailed(pdf, OcrSettings(), max_pages=2, keep_images=True)

    assert total == 3 and sorted(details) == [1, 2]
    assert details[1]["source"] == "text_layer"
    assert details[2]["source"] == "ocr"
    assert details[2]["render_dpi"] == 200
    result = details[2]["ocr_result"]
    assert result is not None and result.image is not None and result.lines
    assert "Alpha" in text and "Bravo" not in text
