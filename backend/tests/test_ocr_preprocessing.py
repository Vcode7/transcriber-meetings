"""Optional OCR preprocessing steps: identity by default, shape/dtype invariants, EXIF handling."""
import io

import numpy as np
import pytest
from PIL import Image

from services.ocr_preprocessing import apply_preprocessing, cap_long_side, ensure_rgb, load_image_rgb
from services.ocr_settings import OcrSettings


def _sample() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.full((120, 200, 3), 225, dtype=np.uint8)
    img[40:80, 20:180] = 35
    noise = rng.integers(-20, 20, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def test_defaults_leave_the_image_untouched():
    img = _sample()
    out, steps = apply_preprocessing(img, OcrSettings())
    assert steps == []
    assert np.array_equal(out, img)


@pytest.mark.parametrize("overrides,step", [
    ({"grayscale": True}, "grayscale"),
    ({"denoise": "median"}, "denoise:median"),
    ({"denoise": "bilateral"}, "denoise:bilateral"),
    ({"denoise": "nlmeans"}, "denoise:nlmeans"),
    ({"clahe": True}, "clahe"),
    ({"sharpen": True}, "sharpen"),
    ({"binarize": "otsu"}, "binarize:otsu"),
    ({"binarize": "adaptive"}, "binarize:adaptive"),
    ({"upscale_min_side": 300}, "upscale"),
])
def test_each_step_returns_rgb_uint8(overrides, step):
    out, steps = apply_preprocessing(_sample(), OcrSettings(**overrides))
    assert step in steps
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3


def test_steps_run_in_documented_order():
    cfg = OcrSettings(upscale_min_side=300, grayscale=True, denoise="median", clahe=True, sharpen=True, binarize="otsu")
    _, steps = apply_preprocessing(_sample(), cfg)
    assert steps == ["upscale", "grayscale", "denoise:median", "clahe", "sharpen", "binarize:otsu"]


def test_binarize_produces_two_levels():
    out, _ = apply_preprocessing(_sample(), OcrSettings(binarize="otsu"))
    assert set(np.unique(out)) <= {0, 255}


def test_upscale_is_capped_at_four_times():
    out, _ = apply_preprocessing(_sample(), OcrSettings(upscale_min_side=4000))
    assert out.shape[:2] == (480, 800)


def test_upscale_skips_large_images():
    _, steps = apply_preprocessing(_sample(), OcrSettings(upscale_min_side=100))
    assert steps == []


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (200, 100), "white")
    exif = img.getexif()
    exif[0x0112] = 6  # stored sideways: rotate 90° clockwise for display
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    arr = load_image_rgb(buf.getvalue())
    assert arr.shape[:2] == (200, 100)


def test_ensure_rgb_and_cap_long_side():
    assert ensure_rgb(np.zeros((10, 20), np.uint8)).shape == (10, 20, 3)
    assert ensure_rgb(np.zeros((10, 20, 4), np.uint8)).shape == (10, 20, 3)
    assert cap_long_side(np.zeros((1000, 500, 3), np.uint8), 400).shape[:2] == (400, 200)
    small = np.zeros((100, 50, 3), np.uint8)
    assert cap_long_side(small, 400) is small
