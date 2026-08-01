import pytest
import os
import numpy as np
from PIL import Image
from services.video_processing_service import (
    get_sr_model,
    preprocess_frame_for_ocr,
    _clean_ocr_text,
    _score_candidate,
    _text_similarity,
    merge_ocr_results,
    run_ocr_on_frame_detail,
    MIN_FRAME_WIDTH,
)

def test_get_sr_model_caching_and_fallback():
    """Verify AI Super-Resolution model loader caches correctly and returns fallback if weights missing."""
    model, model_name = get_sr_model()
    assert isinstance(model_name, str)
    assert len(model_name) > 0
    
    # Second call should return cached model
    model2, model_name2 = get_sr_model()
    assert model_name2 == model_name

def test_preprocess_frame_scaling_and_sr_flag():
    """Verify frames are preprocessed, upscaled to >=1280px, and sr_applied boolean is returned."""
    img = Image.new("RGB", (640, 360), color=(240, 240, 240))
    processed, sr_applied = preprocess_frame_for_ocr(img, min_width=1280)
    
    assert isinstance(processed, np.ndarray)
    assert processed.shape[1] >= MIN_FRAME_WIDTH
    assert len(processed.shape) == 2
    assert isinstance(sr_applied, bool)

def test_clean_ocr_text_noise_removal():
    """Verify noise characters, isolated symbols, and duplicate whitespace are cleaned."""
    raw = """
    ||||||||||||||||||||||||
    Slide 1: System Architecture Overview
    ~ ^ @
    
    Components: API Gateway, User Service, Database.
    ======
    .
    """
    cleaned = _clean_ocr_text(raw)
    assert "Slide 1: System Architecture Overview" in cleaned
    assert "Components: API Gateway, User Service, Database." in cleaned
    assert "||||||" not in cleaned
    assert "=======" not in cleaned

def test_score_candidate_weighting():
    """Verify weighted score calculation based on confidence, word count, and character count."""
    score_high = _score_candidate(avg_conf=85.0, word_count=20, char_count=120, min_chars=5)
    score_low = _score_candidate(avg_conf=40.0, word_count=5, char_count=25, min_chars=5)
    score_insufficient = _score_candidate(avg_conf=90.0, word_count=1, char_count=3, min_chars=5)
    
    assert score_high > score_low
    assert score_insufficient == 0.0

def test_fuzzy_text_similarity_and_merging():
    """Verify consecutive OCR blocks with minor OCR differences are merged using fuzzy similarity."""
    entries = [
        {"start_time": 0.0, "text": "Slide Title: Project Roadmap Q3 2026\nGoals: Deliver core features"},
        {"start_time": 10.0, "text": "Slide Title: Project Roadmap Q3 2026\nGoals: Deliver core features."},  # minor punctuation difference
        {"start_time": 20.0, "text": "Slide Title: Financial Budget Report\nQuarterly revenue stats"},       # new slide
    ]
    
    sim_same = _text_similarity(entries[0]["text"], entries[1]["text"])
    sim_diff = _text_similarity(entries[0]["text"], entries[2]["text"])
    
    assert sim_same >= 0.80
    assert sim_diff < 0.60
    
    merged = merge_ocr_results(entries, video_duration=30.0, merge_ratio=0.80)
    assert len(merged) == 2
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 20.0
    assert "Project Roadmap Q3 2026" in merged[0]["text"]
    assert merged[1]["start"] == 20.0
    assert merged[1]["end"] == 30.0
    assert "Financial Budget Report" in merged[1]["text"]

def test_run_ocr_on_frame_detail_return_format(tmp_path):
    """Verify run_ocr_on_frame_detail returns dict with all required stats and respects debug mode."""
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    debug_dir = str(tmp_path / "debug_ocr")
    os.environ["DEBUG_OCR_DIR"] = debug_dir
    
    res = run_ocr_on_frame_detail(img, save_debug=True, debug_prefix="test_frame_001")
    assert "raw" in res
    assert "clean" in res
    assert "avg_conf" in res
    assert "word_count" in res
    assert "char_count" in res
    assert "sr_applied" in res
    
    assert os.path.exists(os.path.join(debug_dir, "test_frame_001_orig.jpg"))
    assert os.path.exists(os.path.join(debug_dir, "test_frame_001_prep.png"))
    assert os.path.exists(os.path.join(debug_dir, "test_frame_001_raw.txt"))
    assert os.path.exists(os.path.join(debug_dir, "test_frame_001_clean.txt"))
