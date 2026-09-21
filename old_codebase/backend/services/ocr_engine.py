"""
ocr_engine.py — Shared OCR engine using RapidOCR + ONNX Runtime.

Replaces Tesseract throughout the application. RapidOCR is a PaddleOCR-derived
inference engine that runs PP-OCRv4 ONNX models directly through ONNX Runtime,
giving excellent Windows compatibility without requiring the Paddle runtime.

Architecture:
  • get_ocr_engine()       — Singleton: initialises RapidOCR once at startup.
  • ocr_image_bytes()      — OCR from in-memory bytes → clean text  (doc_extractor).
  • ocr_image_file()       — OCR from a file path → clean text       (doc_extractor).
  • ocr_image_ndarray()    — OCR from numpy array → (text, conf, words, chars) (video).
  • preprocess_for_ocr()   — Upscale → CLAHE → Sharpen → Binarise.

Offline model behaviour:
  RapidOCR bundles PP-OCRv4 ONNX models directly inside the package
  (rapidocr_onnxruntime/models/).  No runtime downloads are required — the
  engine works 100% offline as soon as the package is installed.

  For the packaged Application (.exe build), include the entire
  rapidocr_onnxruntime package in the PyInstaller bundle and the ONNX
  models will be embedded automatically.
"""
from __future__ import annotations

import io
import logging
import os
import re
from threading import Lock
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional dependency flags ─────────────────────────────────────────────────
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ── Singleton state ───────────────────────────────────────────────────────────
_OCR_ENGINE: Optional[object] = None
_OCR_ENGINE_LOCK = Lock()
_OCR_AVAILABLE: Optional[bool] = None

# ── Tuning constants ──────────────────────────────────────────────────────────
MIN_OCR_SCORE: float = 0.40      # Minimum per-line confidence to keep
MIN_FRAME_WIDTH: int = 1280      # Minimum image width before OCR


# ── Engine initialisation ─────────────────────────────────────────────────────

def get_ocr_engine():
    """
    Return the singleton RapidOCR engine, initialising it on first call.

    Thread-safe.  Returns None if RapidOCR is not available.
    Enforces mutual exclusivity by unloading any active text embedder first.
    """
    global _OCR_ENGINE, _OCR_AVAILABLE

    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            return _OCR_ENGINE

        # Ensure embedding model is unloaded first so both models never co-exist in memory
        try:
            from services.text_embedding_service import unload_text_embedder
            unload_text_embedder()
        except Exception:
            pass

        try:
            from rapidocr_onnxruntime import RapidOCR

            # Register PyTorch CUDA DLL directory if available on Windows
            try:
                import torch
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
                    if os.path.exists(torch_lib):
                        if hasattr(os, "add_dll_directory"):
                            try:
                                os.add_dll_directory(torch_lib)
                            except Exception:
                                pass
                        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass

            logger.info("[OCREngine] Initialising RapidOCR (ONNX Runtime / PP-OCRv4) with CUDA support…")
            try:
                _OCR_ENGINE = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
            except Exception as cuda_exc:
                logger.warning(f"[OCREngine] Failed to init RapidOCR on CUDA: {cuda_exc} — falling back to CPU")
                _OCR_ENGINE = RapidOCR()
            _OCR_AVAILABLE = True
            logger.info("[OCREngine] RapidOCR engine initialised successfully.")

        except Exception as exc:
            _OCR_AVAILABLE = False
            logger.error(
                f"[OCREngine] Failed to initialise RapidOCR: {exc}",
                exc_info=True,
            )

    return _OCR_ENGINE


def unload_ocr_engine() -> None:
    """
    Unload RapidOCR engine from memory and run garbage collection.
    Releases ONNX Runtime sessions, PyTorch/CUDA memory, and Python references.
    """
    global _OCR_ENGINE, _OCR_AVAILABLE
    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            logger.info("[OCREngine] Unloading RapidOCR engine from memory...")
            _OCR_ENGINE = None
            _OCR_AVAILABLE = None
            import gc
            gc.collect()
            try:
                import torch
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("[OCREngine] RapidOCR engine successfully unloaded.")
        else:
            _OCR_AVAILABLE = None


def is_ocr_available() -> bool:
    """Return True if the OCR engine initialised successfully."""
    if _OCR_AVAILABLE is not None:
        return bool(_OCR_AVAILABLE)
    get_ocr_engine()
    return bool(_OCR_AVAILABLE)


# ── Image pre-processing ──────────────────────────────────────────────────────

def preprocess_for_ocr(img_np: np.ndarray, min_width: int = MIN_FRAME_WIDTH) -> np.ndarray:
    """
    Preprocess a uint8 RGB numpy array for optimal OCR accuracy.

    Steps:
      1. Scale to at least *min_width* pixels wide, preserving aspect ratio.
      2. Convert to grayscale.
      3. CLAHE contrast enhancement.
      4. Unsharp-mask sharpening.
      5. Auto-invert if the background is dark.
      6. Otsu binarisation.
      7. Return as 3-channel BGR for RapidOCR.
    """
    if img_np is None or img_np.size == 0:
        return img_np

    h, w = img_np.shape[:2]

    # ── 1. Upscale if needed ──────────────────────────────────────────────────
    if w < min_width:
        scale = min_width / float(w)
        new_w = min_width
        new_h = max(1, int(h * scale))
        if _CV2_AVAILABLE:
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif _PIL_AVAILABLE:
            pil = Image.fromarray(img_np)
            pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_np = np.array(pil)

    if _CV2_AVAILABLE:
        # ── 2. Grayscale ───────────────────────────────────────────────────────
        if img_np.ndim == 3 and img_np.shape[2] >= 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.copy()

        # ── 3. CLAHE contrast ─────────────────────────────────────────────────
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # ── 4. Sharpening (unsharp mask) ─────────────────────────────────────
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        # ── 5. Invert dark backgrounds ────────────────────────────────────────
        if np.mean(sharpened) < 127:
            sharpened = cv2.bitwise_not(sharpened)

        # ── 6. Otsu binarisation ──────────────────────────────────────────────
        _, binary = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # RapidOCR expects BGR 3-channel
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ── PIL fallback ──────────────────────────────────────────────────────────
    if _PIL_AVAILABLE:
        pil = Image.fromarray(img_np).convert("L")  # grayscale
        pil = ImageEnhance.Contrast(pil).enhance(2.0)
        pil = pil.filter(ImageFilter.SHARPEN)
        arr = np.array(pil, dtype=np.uint8)
        if np.mean(arr) < 127:
            arr = 255 - arr
        binary = np.where(arr > 128, 255, 0).astype(np.uint8)
        return np.stack([binary, binary, binary], axis=-1)

    return img_np


# ── Core OCR helpers ──────────────────────────────────────────────────────────

def _parse_rapid_result(
    result: Optional[list],
) -> Tuple[str, float, int, int]:
    """
    Parse raw RapidOCR output into (text, avg_confidence, word_count, char_count).

    RapidOCR returns a list of:
        [bbox_coords, detected_text, confidence_score]
    sorted roughly in reading order (top-to-bottom, left-to-right).
    """
    if not result:
        return "", 0.0, 0, 0

    lines: List[str] = []
    scores: List[float] = []

    for item in result:
        # Each item: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, score]
        if not (isinstance(item, (list, tuple)) and len(item) >= 3):
            continue
        text = str(item[1]).strip()
        try:
            score = float(item[2])
        except (TypeError, ValueError):
            score = 0.0

        if text and score >= MIN_OCR_SCORE:
            lines.append(text)
            scores.append(score)

    full_text = "\n".join(lines)
    avg_conf = float(np.mean(scores)) if scores else 0.0
    word_count = len(re.findall(r"\S+", full_text))
    char_count = len(re.findall(r"[a-zA-Z0-9]", full_text))
    return full_text, avg_conf, word_count, char_count


def _clean_ocr_text(raw: str) -> str:
    """
    Post-process raw OCR output to remove noise.

    Removes:
      • Lines with fewer than 2 alphanumeric characters.
      • Lines with alphanumeric ratio < 35 %.
      • Runs of 3+ repeated punctuation/symbol characters.
      • Duplicate consecutive lines.
      • Excessive whitespace.
    """
    if not raw:
        return ""

    lines = raw.splitlines()
    cleaned: List[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Remove long runs of repeated punctuation
        s = re.sub(r"([\|~_=\-\^\\\/@#\$%\*\+\.])\1{2,}", "", s).strip()
        if not s:
            continue
        alpha = len(re.findall(r"[a-zA-Z0-9]", s))
        non_sp = len(re.sub(r"\s+", "", s))
        # Discard garbage lines
        if alpha < 2:
            continue
        if non_sp > 0 and alpha / non_sp < 0.35:
            continue
        # Normalise internal whitespace
        s = re.sub(r"[ \t]+", " ", s)
        cleaned.append(s)

    # Deduplicate consecutive identical lines
    deduped: List[str] = []
    for line in cleaned:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    return "\n".join(deduped).strip()


def _ocr_np(
    img_np: np.ndarray,
    preprocess: bool = True,
    label: str = "",
    dual_pass: bool = False,
) -> Tuple[str, float, int, int]:
    """
    Run OCR on a numpy array, optionally preprocessing first.

    By default (dual_pass=False), performs a single preprocessed pass per image
    for fast, efficient document extraction.
    """
    engine = get_ocr_engine()
    if engine is None:
        return "", 0.0, 0, 0

    def _infer(arr: np.ndarray) -> Tuple[str, float, int, int]:
        try:
            result, _ = engine(arr)
        except Exception as exc:
            logger.debug(f"[OCREngine] {label}: inference error: {exc}")
            return "", 0.0, 0, 0
        raw_text, conf, wc, cc = _parse_rapid_result(result)
        return _clean_ocr_text(raw_text), conf, wc, cc

    if preprocess:
        prep_np = preprocess_for_ocr(img_np)
        if not dual_pass:
            # Single pass: preprocessed image (fast & accurate)
            return _infer(prep_np)

        text_prep, conf_prep, wc_prep, cc_prep = _infer(prep_np)
        text_orig, conf_orig, wc_orig, cc_orig = _infer(img_np)

        # Pick the winner: original wins only when confidence is clearly better
        if conf_orig > conf_prep + 0.05 or (
            conf_orig >= conf_prep and cc_orig > cc_prep
        ):
            logger.debug(
                f"[OCREngine] {label}: original wins "
                f"(conf={conf_orig:.2f} > {conf_prep:.2f})"
            )
            return text_orig, conf_orig, wc_orig, cc_orig

        logger.debug(
            f"[OCREngine] {label}: preprocessed wins (conf={conf_prep:.2f})"
        )
        return text_prep, conf_prep, wc_prep, cc_prep

    return _infer(img_np)


# ── Public API ────────────────────────────────────────────────────────────────

def ocr_image_bytes(image_bytes: bytes, *, label: str = "<bytes>") -> str:
    """
    Run OCR on in-memory image bytes. Returns cleaned text or "" on failure.

    Filters out small images (<60px width/height or <3600 total pixels) before OCR.
    Runs a single RapidOCR inference pass per image for maximum performance.
    """
    if not image_bytes:
        return ""
    if not is_ocr_available():
        logger.warning(f"[OCREngine] OCR engine not available — skipping {label}")
        return ""

    MIN_DIM = 100
    MIN_AREA = 10000

    try:
        with Image.open(io.BytesIO(image_bytes)) as img_pil:
            w, h = img_pil.size
            if w < MIN_DIM or h < MIN_DIM or (w * h) < MIN_AREA:
                logger.info(f"[OCREngine] Skipping small image {label} ({w}x{h}px < {MIN_DIM}px threshold)")
                return ""
            img_np = np.array(img_pil.convert("RGB"))
    except Exception as img_err:
        logger.warning(f"[OCREngine] Could not open image bytes for {label}: {img_err}")
        return ""

    for attempt in range(1, 3):
        try:
            text, conf, wc, cc = _ocr_np(img_np, preprocess=True, label=label, dual_pass=False)
            if text:
                logger.info(
                    f"[OCREngine] {label} ({w}x{h}px): OCR complete -> conf={conf:.2f}, words={wc}, chars={cc}"
                )
            else:
                logger.info(f"[OCREngine] {label} ({w}x{h}px): OCR finished -> no text detected")
            return text
        except Exception as exc:
            if attempt < 2:
                logger.warning(
                    f"[OCREngine] OCR attempt {attempt}/2 failed for {label}: {exc} — retrying"
                )
            else:
                logger.warning(
                    f"[OCREngine] OCR skipped after 2 attempts for {label}: {exc}"
                )
    return ""


def ocr_image_file(path: str) -> str:
    """
    Run OCR on a standalone image file. Returns cleaned text or "".

    Used by doc_extractor.py for PNG/JPG/JPEG/WEBP files.
    """
    if not os.path.isfile(path):
        logger.warning(f"[OCREngine] Image file not found: {path}")
        return ""
    if not is_ocr_available():
        logger.warning(
            f"[OCREngine] OCR engine not available — skipping {os.path.basename(path)}"
        )
        return ""

    try:
        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)
        text, conf, wc, cc = _ocr_np(
            img_np, preprocess=True, label=os.path.basename(path)
        )
        logger.debug(
            f"[OCREngine] {os.path.basename(path)}: conf={conf:.2f}, words={wc}, chars={cc}"
        )
        return text
    except Exception as exc:
        logger.warning(f"[OCREngine] OCR failed for {os.path.basename(path)}: {exc}")
        return ""


def ocr_image_ndarray(
    img_np: np.ndarray,
    label: str = "frame",
    preprocess: bool = True,
) -> Tuple[str, float, int, int]:
    """
    Run OCR on a uint8 numpy array (H × W × 3, RGB channel order).

    Returns (clean_text, avg_confidence, word_count, char_count).
    Used by video_processing_service.py for per-frame OCR.
    """
    if not is_ocr_available():
        return "", 0.0, 0, 0
    return _ocr_np(img_np, preprocess=preprocess, label=label)


def unload_ocr_engine():
    """
    Unload the singleton RapidOCR engine and release ONNX Runtime / GPU resources.
    Thread-safe. Can be called repeatedly safely.
    """
    global _OCR_ENGINE, _OCR_AVAILABLE
    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            logger.info("[OCREngine] Unloading RapidOCR / ONNX Runtime engine...")
            try:
                if hasattr(_OCR_ENGINE, "text_det"):
                    del _OCR_ENGINE.text_det
                if hasattr(_OCR_ENGINE, "text_cls"):
                    del _OCR_ENGINE.text_cls
                if hasattr(_OCR_ENGINE, "text_rec"):
                    del _OCR_ENGINE.text_rec
            except Exception as exc:
                logger.warning(f"[OCREngine] Exception clearing session references: {exc}")
            del _OCR_ENGINE
            _OCR_ENGINE = None
            _OCR_AVAILABLE = None

            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("[OCREngine] RapidOCR engine unloaded successfully.")

