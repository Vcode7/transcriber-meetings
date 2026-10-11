import os
import socket
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pathlib import Path
from services.ocr_engine import (
    get_ocr_engine,
    unload_ocr_engine,
    is_ocr_available,
    preprocess_for_ocr,
    ocr_image_ndarray,
    ocr_image_bytes,
    ocr_image_file,
    ocr_image_detailed,
    ocr_engine_lease,
    get_model_dir,
    merge_lines,
    sort_boxes,
    xy_cut_order,
    OcrLine,
    PPOCRv6Engine,
)
from services.ocr_settings import OcrSettings, preset_settings, use_ocr_settings


@pytest.fixture(autouse=True)
def _pin_default_ocr_settings():
    """Run against the default OCR settings, not whatever is saved in the dev database."""
    with use_ocr_settings(OcrSettings()):
        yield


def test_ppocrv6_model_files_exist_in_download_dir():
    """Verify all PP-OCRv6 Medium model files exist locally in download/ppocrv6/."""
    model_dir = get_model_dir()
    assert model_dir.is_dir(), f"Model directory {model_dir} does not exist"
    
    det_file = model_dir / "PP-OCRv6_medium_det.onnx"
    rec_file = model_dir / "PP-OCRv6_medium_rec.onnx"
    cls_file = model_dir / "ch_ppocr_mobile_v2.0_cls.onnx"
    dict_file = model_dir / "ppocrv6_dict.txt"

    assert det_file.exists() and det_file.stat().st_size > 10_000_000, "Detection model missing or too small"
    assert rec_file.exists() and rec_file.stat().st_size > 10_000_000, "Recognition model missing or too small"
    assert cls_file.exists() and cls_file.stat().st_size > 100_000, "Classifier model missing or too small"
    assert dict_file.exists() and dict_file.stat().st_size > 10_000, "Dictionary file missing or too small"


def test_ppocrv6_engine_availability_and_offline():
    """Verify PP-OCRv6 Medium engine initialises successfully without network access."""
    # Temporarily monkeypatch socket to ensure zero outbound connection attempts
    orig_connect = socket.socket.connect
    def blocked_connect(*args, **kwargs):
        raise RuntimeError("Network call attempted during offline OCR operation!")

    socket.socket.connect = blocked_connect  # type: ignore[assignment]
    try:
        assert is_ocr_available() is True
        engine = get_ocr_engine()
        assert engine is not None
        assert isinstance(engine, PPOCRv6Engine)
    finally:
        socket.socket.connect = orig_connect  # type: ignore[assignment]


def test_preprocess_for_ocr_preserves_continuous_channels():
    """Verify preprocess_for_ocr upscales image to min_width and maintains continuous RGB channels."""
    img_np = np.ones((100, 200, 3), dtype=np.uint8) * 200
    prep = preprocess_for_ocr(img_np, min_width=1280)
    assert isinstance(prep, np.ndarray)
    assert prep.shape[1] >= 1280
    assert prep.ndim == 3
    assert prep.shape[2] == 3


def test_ocr_printed_text():
    """Verify PP-OCRv6 extracts printed multi-line text accurately."""
    img_pil = Image.new("RGB", (650, 180), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    draw.text((25, 30), "Transcriber AI Meeting Pipeline", fill=(0, 0, 0))
    draw.text((25, 90), "Quarterly Financial Overview 2026", fill=(0, 0, 0))
    img_np = np.array(img_pil)

    clean_text, avg_conf, word_count, char_count = ocr_image_ndarray(img_np, label="printed_test")
    assert "Transcriber" in clean_text or "Meeting" in clean_text or "Pipeline" in clean_text
    assert "Financial" in clean_text or "2026" in clean_text or "Overview" in clean_text
    assert avg_conf > 0.50
    assert word_count >= 3
    assert char_count >= 10


def test_ocr_rotated_text_180():
    """Verify orientation classifier detects and decodes 180-degree upside down text."""
    img_pil = Image.new("RGB", (600, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    draw.text((30, 50), "Executive Action Item Approved", fill=(0, 0, 0))
    rotated = img_pil.rotate(180, expand=True)
    img_np = np.array(rotated)

    clean_text, avg_conf, word_count, char_count = ocr_image_ndarray(img_np, label="rotated_180")
    assert "Executive" in clean_text or "Action" in clean_text or "Approved" in clean_text
    assert avg_conf > 0.40


def test_ocr_noisy_and_low_quality_image():
    """Verify PP-OCRv6 handles blurry or noisy inputs typical of compressed video frames."""
    img_pil = Image.new("RGB", (500, 140), color=(240, 240, 240))
    draw = ImageDraw.Draw(img_pil)
    draw.text((20, 45), "System Diagnostic Telemetry", fill=(20, 20, 20))
    blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=0.7))
    img_np = np.array(blurred)

    clean_text, avg_conf, word_count, char_count = ocr_image_ndarray(img_np, label="noisy_test")
    assert "Diagnostic" in clean_text or "Telemetry" in clean_text or "System" in clean_text
    assert word_count > 0


def test_ocr_multiple_text_regions_and_tables():
    """Verify multi-column / tabular layout extraction with reading order sorting."""
    img_pil = Image.new("RGB", (800, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    # Column 1 Header & Row
    draw.text((30, 30), "Item Description", fill=(0, 0, 0))
    draw.text((30, 80), "Server Hardware", fill=(0, 0, 0))
    draw.text((30, 130), "Network Switch", fill=(0, 0, 0))
    # Column 2 Header & Row
    draw.text((450, 30), "Budget Amount", fill=(0, 0, 0))
    draw.text((450, 80), "$15000", fill=(0, 0, 0))
    draw.text((450, 130), "$4500", fill=(0, 0, 0))
    img_np = np.array(img_pil)

    clean_text, avg_conf, word_count, char_count = ocr_image_ndarray(img_np, label="table_test")
    assert "Item Description" in clean_text or "Server" in clean_text
    assert "Budget" in clean_text or "15000" in clean_text
    assert word_count >= 4


def test_ocr_image_bytes_and_file_paths(tmp_path):
    """Verify file and bytes public APIs function correctly."""
    img_pil = Image.new("RGB", (600, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    draw.text((20, 40), "Conference Room Booking System", fill=(0, 0, 0))

    file_path = str(tmp_path / "test_booking.png")
    img_pil.save(file_path)

    # Test file path OCR
    text_file = ocr_image_file(file_path)
    assert "Conference" in text_file or "Booking" in text_file or "Room" in text_file

    # Test bytes OCR
    with open(file_path, "rb") as f:
        img_bytes = f.read()
    text_bytes = ocr_image_bytes(img_bytes, label="test_bytes")
    assert "Conference" in text_bytes or "Booking" in text_bytes or "Room" in text_bytes


def test_unload_ocr_engine_releases_memory():
    """Verify unload_ocr_engine cleanly releases sessions and state."""
    assert is_ocr_available() is True
    engine = get_ocr_engine()
    assert engine is not None
    assert engine.det_session is not None

    unload_ocr_engine()
    from services.ocr_engine import _OCR_ENGINE
    assert _OCR_ENGINE is None


# ── Official PP-OCRv6 pipeline behaviour ──────────────────────────────────────

def _font(size=26):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _render(lines, size=(900, 360), rotate=0):
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x, y, text in lines:
        draw.text((x, y), text, fill=(0, 0, 0), font=_font())
    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor=(255, 255, 255))
    return np.array(img)


_DOC = [
    (30, 30, "Quarterly Review Meeting"),
    (30, 90, "Revenue grew twelve percent"),
    (30, 150, "Action items assigned to Priya"),
    (30, 210, "Next meeting in November"),
]


def _line(x0, y0, x1, y1, text):
    return OcrLine(box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text=text, score=0.9)


@pytest.mark.parametrize("hw,official,legacy", [
    ((2339, 1654), (2336, 1664), (960, 672)),    # A4 @ 200 dpi: native resolution vs 960 px downscale
    ((40, 30), (96, 64), (32, 32)),              # tiny image: min side raised to 64
    ((6000, 3000), (4000, 1984), (960, 480)),    # capped by max_side_limit
    ((1080, 1920), (1088, 1920), (544, 960)),    # 1080p video frame
])
def test_det_resize_follows_official_rules(hw, official, legacy):
    assert PPOCRv6Engine.det_resize_shape(*hw, OcrSettings()) == official
    assert PPOCRv6Engine.det_resize_shape(*hw, preset_settings("legacy")) == legacy


def test_sort_boxes_uses_same_line_tolerance():
    items = [_line(300, 98, 400, 118, "B"), _line(10, 100, 100, 120, "A"), _line(10, 200, 100, 220, "C")]
    assert [it.text for it in sort_boxes(items)] == ["A", "B", "C"]


def test_merge_lines_joins_words_but_not_distant_columns():
    items = [_line(10, 100, 80, 120, "Hello"), _line(90, 101, 160, 121, "world"), _line(600, 100, 700, 120, "Right")]
    assert [ln.text for ln in merge_lines(items, max_gap=2.0)] == ["Hello world", "Right"]


def test_xy_cut_reads_each_column_top_to_bottom():
    left = [_line(10, 100 + i * 30, 200, 120 + i * 30, f"L{i}") for i in range(3)]
    right = [_line(400, 100 + i * 30, 600, 120 + i * 30, f"R{i}") for i in range(3)]
    ordered = xy_cut_order(sort_boxes(left + right))
    assert [it.text for it in ordered] == ["L0", "L1", "L2", "R0", "R1", "R2"]


def test_column_detection_orders_real_two_column_page():
    img = _render(
        [(30, 40, "Alpha first line"), (30, 100, "Alpha second line"),
         (520, 40, "Bravo first line"), (520, 100, "Bravo second line")],
        size=(900, 200),
    )
    cfg = OcrSettings(column_detection=True)
    text = ocr_image_detailed(img, cfg).text
    assert text.index("Alpha second") < text.index("Bravo first")


def test_batched_and_single_recognition_agree():
    engine = get_ocr_engine()
    bgr = _render(_DOC)[:, :, ::-1].copy()
    boxes = engine.detect_text(bgr, OcrSettings())
    from services.ocr_engine import get_rotate_crop_image
    crops = [get_rotate_crop_image(bgr, box) for box, _ in boxes]
    batched = engine.recognize_batch(crops, OcrSettings(rec_batch_size=6))
    single = engine.recognize_batch(crops, OcrSettings(rec_batch_size=1))
    assert [t for t, _ in batched] == [t for t, _ in single]
    assert all(t for t, _ in batched)


@pytest.mark.parametrize("rotate", [90, 180, 270])
def test_auto_orient_restores_rotated_pages(rotate):
    result = ocr_image_detailed(_render(_DOC, rotate=rotate), OcrSettings(auto_orient=True))
    assert result.rotation in (90, 180, 270)
    assert result.text.index("Quarterly") < result.text.index("November")


def test_deskew_straightens_slightly_rotated_scan():
    result = ocr_image_detailed(_render(_DOC, rotate=3), OcrSettings(deskew=True))
    assert abs(result.skew_angle) > 1.5
    lines = result.text.splitlines()
    assert lines[0].startswith("Quarterly") and lines[-1].endswith("November")


def test_unload_is_deferred_while_a_lease_is_held():
    from services import ocr_engine as engine_module

    with ocr_engine_lease() as engine:
        assert engine is not None
        unload_ocr_engine()
        assert engine_module._OCR_ENGINE is engine   # still alive for the running task
    assert engine_module._OCR_ENGINE is None          # unloaded once the task finished


def test_legacy_preset_still_reads_synthetic_text():
    text, conf, words, _ = ocr_image_ndarray(_render(_DOC), cfg=preset_settings("legacy"))
    assert "Quarterly" in text and words >= 8 and conf > 0.5


def test_non_latin_lines_survive_cleanup():
    from services.ocr_engine import _clean_ocr_text
    assert _clean_ocr_text("Überprüfung läuft\n会议纪要\n|||---") == "Überprüfung läuft\n会议纪要"
