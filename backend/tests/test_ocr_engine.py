import pytest
import os
import numpy as np
from PIL import Image, ImageDraw
from services.ocr_engine import (
    get_ocr_engine,
    is_ocr_available,
    preprocess_for_ocr,
    ocr_image_ndarray,
    ocr_image_bytes,
    ocr_image_file,
)

def test_ocr_engine_availability():
    """Verify RapidOCR engine initialises successfully."""
    assert is_ocr_available() is True
    engine = get_ocr_engine()
    assert engine is not None

def test_preprocess_for_ocr_upscaling_and_channels():
    """Verify preprocess_for_ocr upscales image to min_width and outputs 3-channel BGR image."""
    img_np = np.ones((100, 200, 3), dtype=np.uint8) * 255
    prep = preprocess_for_ocr(img_np, min_width=1280)
    assert isinstance(prep, np.ndarray)
    assert prep.shape[1] >= 1280
    assert prep.ndim == 3

def test_ocr_image_ndarray_with_text():
    """Verify ocr_image_ndarray extracts text from synthetic image numpy array."""
    img_pil = Image.new("RGB", (500, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    draw.text((20, 30), "Project Status Report Q3 2026", fill=(0, 0, 0))
    draw.text((20, 80), "Key Milestone Delivered On Time", fill=(0, 0, 0))
    img_np = np.array(img_pil)

    clean_text, avg_conf, word_count, char_count = ocr_image_ndarray(img_np, label="test_synth")
    assert "Project Status Report" in clean_text or "Q3 2026" in clean_text or "Milestone" in clean_text
    assert avg_conf > 0.0
    assert word_count > 0
    assert char_count > 0

def test_ocr_image_bytes_and_file(tmp_path):
    """Verify ocr_image_bytes and ocr_image_file functions extract text correctly."""
    img_pil = Image.new("RGB", (600, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    draw.text((20, 40), "Agenda Meeting Minutes Overview", fill=(0, 0, 0))
    
    file_path = str(tmp_path / "test_ocr.png")
    img_pil.save(file_path)

    # Test file path OCR
    text_file = ocr_image_file(file_path)
    assert "Agenda Meeting" in text_file or "Overview" in text_file

    # Test bytes OCR
    with open(file_path, "rb") as f:
        img_bytes = f.read()
    text_bytes = ocr_image_bytes(img_bytes, label="test_bytes")
    assert "Agenda Meeting" in text_bytes or "Overview" in text_bytes
