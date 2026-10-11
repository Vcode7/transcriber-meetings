"""
ocr_preprocessing.py — Optional image preprocessing for the PP-OCRv6 pipeline.

The official PP-OCRv6 pipeline feeds continuous-tone images straight to the detector,
so every step here is opt-in (see services.ocr_settings). Steps run in a fixed order:

    upscale → grayscale → denoise → CLAHE → sharpen → binarize

Every step takes and returns an H×W×3 uint8 RGB array. A step that fails is logged
and skipped so preprocessing can never break OCR.
"""
from __future__ import annotations

import io
import logging
from typing import List, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from PIL import Image, ImageOps
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ── Loading ───────────────────────────────────────────────────────────────────

def load_image_rgb(src: Union[bytes, bytearray, str]) -> np.ndarray:
    """
    Decode image bytes or a file path to an RGB uint8 array, honouring EXIF
    orientation (phone photos are often stored sideways with a rotation tag).
    """
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow is not installed")
    handle = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    with Image.open(handle) as img:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        return np.array(img.convert("RGB"))


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Return an H×W×3 uint8 array (gray / RGBA inputs converted)."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1) if not _CV2_AVAILABLE else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3].copy() if not _CV2_AVAILABLE else cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


def cap_long_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longer side is at most `max_side` (no-op when already smaller)."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if max_side <= 0 or longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    if _CV2_AVAILABLE:
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return np.array(Image.fromarray(img).resize(new_size, Image.Resampling.LANCZOS))


# ── Individual steps ──────────────────────────────────────────────────────────

def upscale_min_side(img: np.ndarray, min_side: int) -> np.ndarray:
    """Upscale (cubic, ≤4×) when the shorter side is below `min_side`."""
    h, w = img.shape[:2]
    shortest = min(h, w)
    if min_side <= 0 or shortest <= 0 or shortest >= min_side:
        return img
    scale = min(4.0, min_side / float(shortest))
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def denoise(img: np.ndarray, mode: str, strength: int) -> np.ndarray:
    if mode == "median":
        ksize = 3 + 2 * min(3, strength // 8)          # 3, 5, 7 or 9
        return cv2.medianBlur(img, ksize)
    if mode == "bilateral":
        sigma = 25 + 5 * strength
        return cv2.bilateralFilter(img, 9, sigma, sigma)
    if mode == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
    return img


def clahe(img: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """CLAHE on the LAB lightness channel so colours are preserved."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    eq = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid))
    lab = cv2.merge((eq.apply(l_chan), a_chan, b_chan))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask: img + amount × (img − blur)."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def binarize(img: np.ndarray, mode: str, block_size: int, c: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if mode == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "adaptive":
        block = block_size if block_size % 2 == 1 else block_size + 1
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, max(3, block), c
        )
    else:
        return img
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def apply_preprocessing(img: np.ndarray, cfg) -> Tuple[np.ndarray, List[str]]:
    """
    Apply the enabled preprocessing steps from `cfg` (OcrSettings).
    Returns (processed RGB image, names of the steps that were applied).
    """
    img = ensure_rgb(img)
    if not _CV2_AVAILABLE or img.size == 0:
        return img, []

    steps = []
    if cfg.upscale_min_side > 0:
        steps.append(("upscale", lambda x: upscale_min_side(x, cfg.upscale_min_side)))
    if cfg.grayscale:
        steps.append(("grayscale", to_grayscale))
    if cfg.denoise != "none":
        steps.append((f"denoise:{cfg.denoise}", lambda x: denoise(x, cfg.denoise, cfg.denoise_strength)))
    if cfg.clahe:
        steps.append(("clahe", lambda x: clahe(x, cfg.clahe_clip_limit, cfg.clahe_tile_grid)))
    if cfg.sharpen:
        steps.append(("sharpen", lambda x: sharpen(x, cfg.sharpen_amount)))
    if cfg.binarize != "none":
        steps.append((f"binarize:{cfg.binarize}",
                      lambda x: binarize(x, cfg.binarize, cfg.binarize_block_size, cfg.binarize_c)))

    applied: List[str] = []
    for name, fn in steps:
        try:
            before = img.shape
            out = ensure_rgb(fn(img))
            if out is not None and out.size:
                img = out
                if name != "upscale" or out.shape != before:
                    applied.append(name)
        except Exception as exc:
            logger.warning(f"[OCRPreprocess] Step '{name}' failed and was skipped: {exc}")
    return img, applied
