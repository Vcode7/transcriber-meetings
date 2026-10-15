"""
Structured OCR text (services/ocr_layout.py) from synthetic region boxes, the spacing fixes in
services/ocr_engine.py, and rendered agenda pages through the real PP-OCRv6 pipeline.
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from services.ocr_engine import OcrLine, _clean_ocr_text, get_ocr_engine, ocr_image_detailed, pad_box
from services.ocr_layout import build_layout
from services.ocr_settings import OcrSettings, use_ocr_settings
from services.text_tables import strip_layout_markup

H = 20  # height of the synthetic regions


@pytest.fixture(autouse=True)
def _pin_default_ocr_settings():
    """Run against the default OCR settings, not whatever is saved in the dev database."""
    with use_ocr_settings(OcrSettings()):
        yield


def _r(x0, y, text, w=None):
    w = w or 11 * len(text)
    return OcrLine(box=[[x0, y], [x0 + w, y], [x0 + w, y + H], [x0, y + H]], text=text, score=0.95)


# ── Layout from region boxes ──────────────────────────────────────────────────

def test_aligned_rows_become_one_markdown_table():
    items = [_r(10, 10, "Board Meeting Agenda"),
             _r(10, 60, "Time"), _r(150, 60, "Topic"), _r(600, 60, "Presenter"),
             _r(10, 95, "10:00"), _r(150, 95, "Opening remarks"), _r(600, 95, "Chair"),
             _r(10, 130, "10:30"), _r(150, 130, "Budget review"), _r(600, 130, "CFO")]
    layout = build_layout(items, OcrSettings())
    assert layout.table_count == 1
    assert layout.text == (
        "Board Meeting Agenda\n\n"
        "[Table: 3 rows x 3 cols]\n"
        "| Time | Topic | Presenter |\n"
        "| --- | --- | --- |\n"
        "| 10:00 | Opening remarks | Chair |\n"
        "| 10:30 | Budget review | CFO |"
    )
    assert "|" not in layout.plain_text and "10:00 Opening remarks Chair" in layout.plain_text
    assert len(layout.lines) == 10  # the title and nine cells, for previews


def test_short_cells_survive_cleanup():
    """Plain line cleanup drops lines under two letters/digits; table cells are kept."""
    items = [_r(10, 10, "1"), _r(80, 10, "Opening"), _r(10, 45, "2"), _r(80, 45, "Budget")]
    assert build_layout(items, OcrSettings()).text == (
        "[Table: 2 rows x 2 cols]\n| 1 | Opening |\n| --- | --- |\n| 2 | Budget |")


def test_wrapped_cell_lines_join_their_row():
    items = [_r(10, 10, "No."), _r(80, 10, "Agenda Item"), _r(500, 10, "Owner"),
             _r(10, 45, "1"), _r(80, 45, "Approval of the minutes of the"), _r(500, 45, "Finance"),
             _r(80, 67, "previous meeting"), _r(500, 67, "Controller"),   # tight below, no first-column cell
             _r(10, 100, "2"), _r(80, 100, "Any other business"), _r(500, 100, "Chair")]
    text = build_layout(items, OcrSettings()).text
    assert "| 1 | Approval of the minutes of the previous meeting | Finance Controller |" in text
    assert text.startswith("[Table: 3 rows x 3 cols]")


def test_line_with_its_own_first_column_cell_starts_a_new_row():
    items = [_r(10, 10, "10:00"), _r(100, 10, "Opening"), _r(10, 32, "10:15"), _r(100, 32, "Budget"),
             _r(10, 54, "10:30"), _r(100, 54, "Hiring")]
    assert build_layout(items, OcrSettings()).text.count("\n| 10:") == 3


def test_side_by_side_prose_is_not_a_table():
    left = ["The quarterly review covered the", "progress of all regional teams and", "the status of the upgrade."]
    right = ["The finance team presented the", "revised budget, which includes a", "contingency of five percent."]
    items = [_r(10, 10 + 30 * i, t) for i, t in enumerate(left)] + [_r(500, 10 + 30 * i, t) for i, t in enumerate(right)]
    assert build_layout(items, OcrSettings()).table_count == 0
    columns = build_layout(items, OcrSettings(column_detection=True)).text
    assert columns.index("the status of the upgrade.") < columns.index("The finance team presented the")


def test_column_detection_keeps_tables_with_a_label_column():
    items = [_r(10, 10, "10:00"), _r(300, 10, "Opening remarks by the chair"),
             _r(10, 45, "10:30"), _r(300, 45, "Budget review for next year")]
    assert build_layout(items, OcrSettings(column_detection=True)).table_count == 1


def test_line_above_a_table_is_not_absorbed():
    """A two-field line whose halves line up with the table columns but sits far above it."""
    items = [_r(10, 10, "Venue: Room 2"), _r(178, 10, "Date: 14 Oct")]
    for i, (a, b) in enumerate([("Time", "Topic"), ("10:00", "Opening"), ("10:30", "Budget"), ("11:00", "Hiring")]):
        items += [_r(10, 75 + 30 * i, a), _r(178, 75 + 30 * i, b)]
    text = build_layout(items, OcrSettings()).text
    assert text.startswith("Venue: Room 2\tDate: 14 Oct\n\n[Table: 4 rows x 2 cols]\n| Time | Topic |")


def test_wide_gap_inside_a_line_is_a_tab_and_far_regions_stay_apart():
    near = build_layout([_r(10, 10, "Venue: Room 2"), _r(178, 10, "Date: 14 Oct")], OcrSettings())
    assert near.text == "Venue: Room 2\tDate: 14 Oct"
    far = build_layout([_r(10, 10, "Left column"), _r(400, 10, "Right column")], OcrSettings())
    assert far.text == "Left column\nRight column"


def test_paragraph_gaps_become_blank_lines():
    items = [_r(10, 10, "First paragraph line one"), _r(10, 34, "First paragraph line two"),
             _r(10, 90, "Second paragraph")]
    assert build_layout(items, OcrSettings()).text == (
        "First paragraph line one\nFirst paragraph line two\n\nSecond paragraph")


# ── Spacing ───────────────────────────────────────────────────────────────────

def test_separator_runs_become_a_gap_instead_of_joining_words():
    assert _clean_ocr_text("Introduction.........10:00") == "Introduction\t10:00"
    assert _clean_ocr_text("Closing Remarks - - - - - 12:00") == "Closing Remarks\t12:00"
    assert _clean_ocr_text("Strategic Plan________11:15") == "Strategic Plan\t11:15"
    assert _clean_ocr_text("Überprüfung läuft\n会议纪要\n|||---") == "Überprüfung läuft\n会议纪要"


def test_pad_box_grows_along_the_box_and_stays_in_the_image():
    box = np.array([[10, 10], [110, 10], [110, 30], [10, 30]], dtype=np.float32)
    assert pad_box(box, 0.1, 200, 200).tolist() == [[8, 8], [112, 8], [112, 32], [8, 32]]
    assert pad_box(box, 0.5, 115, 200).tolist() == [[0, 0], [115, 0], [115, 40], [0, 40]]
    assert pad_box(box, 0.0, 200, 200) is box


# ── Real pipeline ─────────────────────────────────────────────────────────────

def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        pytest.skip("Arial is not available")


def _ruled_agenda(size=18):
    rows = [["S.No", "Time", "Agenda Item", "Presenter"],
            ["1", "10:00 AM", "Opening remarks and welcome address", "Chairperson"],
            ["2", "10:15 AM", "Review of previous meeting minutes", "Secretary"],
            ["3", "10:45 AM", "Quarterly budget and expenditure report", "Finance Head"]]
    col_w, row_h, font = [90, 150, 560, 230], int(size * 2.2), _font(size)
    img = Image.new("RGB", (sum(col_w) + 80, row_h * len(rows) + 160), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), "Board Meeting Agenda", font=_font(int(size * 1.4)), fill="black")
    for r, row in enumerate(rows):
        x = 40
        for c, cell in enumerate(row):
            draw.rectangle([x, 110 + r * row_h, x + col_w[c], 110 + (r + 1) * row_h], outline="black", width=2)
            draw.text((x + 10, 110 + r * row_h + (row_h - size) // 2), cell, font=font, fill="black")
            x += col_w[c]
    return np.array(img)


def test_scanned_agenda_table_keeps_rows_and_columns():
    result = ocr_image_detailed(_ruled_agenda(), OcrSettings())
    assert result.table_count == 1
    assert "| S.No | Time | Agenda Item | Presenter |" in result.text
    assert "| 1 | 10:00 AM | Opening remarks and welcome address | Chairperson |" in result.text
    assert result.text.startswith("Board Meeting Agenda")
    assert result.word_count == len(strip_layout_markup(result.text).split())  # words, not markup


def test_structured_output_off_returns_plain_lines():
    result = ocr_image_detailed(_ruled_agenda(), OcrSettings(structured_output=False))
    assert result.table_count == 0 and "[Table:" not in result.text
    assert "Opening remarks and welcome address" in result.text


def test_wrong_upside_down_guess_is_overruled_by_the_recognizer(monkeypatch):
    """The 0/180 classifier sees each line squeezed to 192 px and has flagged upright lines as
    upside down with 0.93 confidence (the line then read as 'ct6'). A flip is kept only when the
    flipped crop reads more confidently."""
    font = _font(22)
    img = Image.new("RGB", (900, 80), "white")
    ImageDraw.Draw(img).text((10, 25), "Action item: IT team to finalise the vendor list", font=font, fill="black")
    engine = get_ocr_engine()
    monkeypatch.setattr(engine, "classify_orientation", lambda crops: [(1, 0.99)] * len(crops))
    text = engine.run(np.array(img), OcrSettings(auto_orient=False)).text
    assert text == "Action item: IT team to finalise the vendor list"
