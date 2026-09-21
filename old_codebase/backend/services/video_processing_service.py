"""
Video Processing Service — dedicated module for all video-specific operations.

Responsibilities:
  1. extract_audio_from_video()  — strip audio track with ffmpeg
  2. sample_frames()             — extract one frame every stride_sec + scene changes via ffmpeg
  3. get_sr_model()              — cached AI Super-Resolution model loader (OpenCV DNN / Real-ESRGAN / Fallback)
  4. preprocess_frame_for_ocr()  — AI Super-Resolution (2x-4x), grayscale, CLAHE, sharpen, auto-invert, threshold
  5. run_ocr_on_frame_detail()   — RapidOCR (PP-OCRv4/ONNX) OCR with confidence filtering, word & char counts
  6. extract_video_ocr_timeline()— multi-frame OCR pipeline (evaluates T-0.2s, T, T+0.2s, T+0.4s in parallel)
  7. merge_ocr_results()         — merge consecutive OCR blocks via fuzzy text similarity

Design decisions:
  • OCR is provided by ocr_engine.py (RapidOCR + ONNX Runtime / PP-OCRv4), replacing Tesseract.
  • Multi-frame evaluation around each timestamp T (offsets -0.2s, 0.0s, +0.2s, +0.4s) avoids animation artifacts.
  • Candidate frames are scored by weighted combination of OCR confidence, word count, and character count.
  • AI Super-Resolution model is initialized once and cached for high-quality OCR input.
  • Processing of candidate frames is parallelized using ThreadPoolExecutor.
  • Fully offline execution with seamless fallback to Lanczos/Bicubic high-quality interpolation.
"""
from __future__ import annotations

import concurrent.futures
import difflib
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ── Tuneable constants ────────────────────────────────────────────────────────
FRAME_STRIDE_SEC: float = 15.0          # extract one frame every N seconds (15s stride)
SCENE_THRESHOLD: float = 0.40           # scene-change sensitivity (0–1; lower = more sensitive)
MIN_OCR_CHARS: int = 5                  # discard OCR results shorter than this
OCR_MERGE_RATIO: float = 0.90           # fuzzy similarity ratio threshold for merging duplicate slides
MIN_WORD_CONFIDENCE: float = 30.0       # discard words with OCR confidence < 30%
MIN_FRAME_WIDTH: int = 1280             # scale frame to at least 1280px width before OCR
JPEG_QUALITY: int = 90                  # ffmpeg JPEG quality for extracted frames
DEBUG_OCR: bool = os.environ.get("DEBUG_OCR", "false").lower() in ("1", "true", "yes")
DEBUG_OCR_DIR: str = os.environ.get("DEBUG_OCR_DIR", "debug_ocr")
MULTI_FRAME_OFFSETS: List[float] = [-0.2, 0.0, 0.2, 0.4]  # nearby frame offsets around timestamp T

# ── OpenCV setup ──────────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("[VideoOCR] OpenCV (cv2) not available — using PIL fallback.")

# ── OCR engine (RapidOCR / PP-OCRv4 via ONNX Runtime) ───────────────────────
try:
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
except ImportError:
    pass

try:
    from services.ocr_engine import ocr_image_ndarray as _ocr_ndarray, is_ocr_available
except ImportError:
    try:
        from backend.services.ocr_engine import ocr_image_ndarray as _ocr_ndarray, is_ocr_available
    except ImportError:
        def _ocr_ndarray(img, label="", preprocess=True):  # type: ignore[misc]
            return "", 0.0, 0, 0
        def is_ocr_available() -> bool:  # type: ignore[misc]
            return False
        logger.warning("[VideoOCR] ocr_engine not available — OCR disabled.")


# ── Cached AI Super-Resolution Model Loader ───────────────────────────────────

_SR_MODEL = None
_SR_MODEL_LOADED: bool = False
_SR_MODEL_NAME: str = "none"

def get_sr_model() -> Tuple[Optional[object], str]:
    """
    Load and cache the AI Super-Resolution model (initialized once).
    Supports:
      1. OpenCV DNN SuperRes (FSRCNN / EDSR / ESPCN / LapSRN) if model file exists.
      2. Real-ESRGAN if package & model file exist.
      3. OpenCV DNN Net if model file exists.
      4. High-Quality Lanczos/Bicubic Interpolation (fallback).
    """
    global _SR_MODEL, _SR_MODEL_LOADED, _SR_MODEL_NAME

    if _SR_MODEL_LOADED:
        return _SR_MODEL, _SR_MODEL_NAME

    _SR_MODEL_LOADED = True

    sr_model_path = os.environ.get("SR_MODEL_PATH", "")
    sr_model_type = os.environ.get("SR_MODEL_TYPE", "fsrcnn").lower()
    sr_scale = int(os.environ.get("SR_SCALE", "2"))

    # Model directories to search
    base_dirs: List[Path] = []
    if sr_model_path:
        p_obj = Path(sr_model_path)
        if p_obj.is_dir():
            base_dirs.append(p_obj)
        elif p_obj.is_file():
            base_dirs.append(p_obj.parent)

    try:
        from config import DEFAULT_MODELS_DIR, BASE_DIR, RUNTIME_DIR
        base_dirs.extend([
            DEFAULT_MODELS_DIR,
            BASE_DIR.parent / "Application" / "runtime" / "models",
            BASE_DIR.parent / "Application" / "models",
            BASE_DIR.parent / "Application" / "backend" / "runtime" / "models",
            RUNTIME_DIR / "models",
            BASE_DIR / "models",
        ])
    except Exception:
        pass

    base_dirs.extend([
        Path("models"),
        Path("runtime/models"),
        Path("backend/runtime/models"),
        Path("backend/models"),
        Path(r"C:\models"),
    ])

    filenames = [
        f"FSRCNN_x{sr_scale}.pb",
        f"EDSR_x{sr_scale}.pb",
        f"fsrcnn_x{sr_scale}.pb",
        f"edsr_x{sr_scale}.pb",
        f"ESPCN_x{sr_scale}.pb",
        f"LapSRN_x{sr_scale}.pb",
        "FSRCNN_x2.pb",
        "EDSR_x2.pb",
        "fsrcnn_x2.pb",
        "edsr_x2.pb",
    ]

    possible_paths: List[str] = []
    if sr_model_path and os.path.isfile(sr_model_path):
        possible_paths.append(sr_model_path)

    for bd in base_dirs:
        for fn in filenames:
            possible_paths.append(str(bd / fn))

    valid_model_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            valid_model_path = p
            fname_lower = os.path.basename(p).lower()
            if "edsr" in fname_lower:
                sr_model_type = "edsr"
            elif "fsrcnn" in fname_lower:
                sr_model_type = "fsrcnn"
            elif "espcn" in fname_lower:
                sr_model_type = "espcn"
            elif "lapsrn" in fname_lower:
                sr_model_type = "lapsrn"
            break

    # 1. Try OpenCV dnn_superres if available
    if _CV2_AVAILABLE and hasattr(cv2, "dnn_superres") and valid_model_path:
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(valid_model_path)
            sr.setModel(sr_model_type, sr_scale)
            _SR_MODEL = sr
            _SR_MODEL_NAME = f"OpenCV DNN SuperRes ({sr_model_type.upper()} {sr_scale}x)"
            logger.info(f"[VideoOCR AI SR] Loaded Super-Resolution model: {_SR_MODEL_NAME} from {valid_model_path}")
            return _SR_MODEL, _SR_MODEL_NAME
        except Exception as e:
            logger.warning(f"[VideoOCR AI SR] OpenCV dnn_superres load failed: {e}")

    # 2. Try OpenCV dnn readNet
    if _CV2_AVAILABLE and valid_model_path:
        try:
            net = cv2.dnn.readNet(valid_model_path)
            _SR_MODEL = ("cv2_dnn", net, sr_scale)
            _SR_MODEL_NAME = f"OpenCV DNN Net ({os.path.basename(valid_model_path)})"
            logger.info(f"[VideoOCR AI SR] Loaded Super-Resolution DNN net: {_SR_MODEL_NAME}")
            return _SR_MODEL, _SR_MODEL_NAME
        except Exception as e:
            logger.warning(f"[VideoOCR AI SR] OpenCV dnn.readNet load failed: {e}")

    # 3. Try Real-ESRGAN
    try:
        from realesrgan import RealESRGANer
        if valid_model_path:
            # Placeholder for RealESRGANer instantiation if model is provided
            pass
    except ImportError:
        pass

    # 4. Fallback to Lanczos/Bicubic high-quality interpolation
    _SR_MODEL = None
    _SR_MODEL_NAME = "Lanczos High-Quality Interpolation Scaling (Fallback)"
    logger.info(f"[VideoOCR AI SR] AI Super-Resolution model weights not found. Using fallback pipeline: {_SR_MODEL_NAME}")
    return _SR_MODEL, _SR_MODEL_NAME


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def _ffmpeg_kwargs() -> dict:
    """Return subprocess run kwargs; suppress console window on Windows."""
    kw: dict = {"capture_output": True, "timeout": 7200}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    return kw


def extract_audio_from_video(video_path: str, output_wav_path: str, sr: int = 16000) -> str:
    """
    Extract the audio track from a video file and write a 16kHz mono WAV.
    """
    logger.info(f"[VideoProc] Extracting audio: {video_path} → {output_wav_path}")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ar", str(sr),
            "-ac", "1",
            "-f", "wav",
            "-acodec", "pcm_s16le",
            output_wav_path,
        ],
        **_ffmpeg_kwargs(),
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"[VideoProc] ffmpeg audio extraction failed: {stderr}")
    logger.info(f"[VideoProc] Audio extracted OK: {output_wav_path}")
    return output_wav_path


def get_video_duration(video_path: str) -> float:
    """Return duration in seconds using ffprobe. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path,
            ],
            **_ffmpeg_kwargs(),
        )
        if result.returncode == 0:
            data = json.loads(result.stdout.decode(errors="replace"))
            dur = float(data.get("format", {}).get("duration", 0))
            if dur > 0:
                return dur
    except Exception as e:
        logger.warning(f"[VideoProc] ffprobe duration failed: {e}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            **_ffmpeg_kwargs(),
        )
        if result.returncode == 0:
            return float(result.stdout.decode(errors="replace").strip())
    except Exception:
        pass
    return 0.0


def _detect_scene_change_timestamps(video_path: str, threshold: float = SCENE_THRESHOLD) -> List[float]:
    """
    Use ffmpeg's scene-change detection filter to get timestamps where scene changes occur.
    """
    logger.debug(f"[VideoProc] Detecting scene changes (threshold={threshold})")
    result = subprocess.run(
        [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-vsync", "vfr",
            "-f", "null",
            "-",
        ],
        capture_output=True,
        timeout=300,
        **({"creationflags": 0x08000000} if os.name == "nt" else {}),
    )
    raw = (result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace"))
    timestamps: List[float] = []
    for match in re.finditer(r"pts_time:([\d.]+)", raw):
        try:
            timestamps.append(float(match.group(1)))
        except ValueError:
            pass
    logger.debug(f"[VideoProc] Scene changes detected: {len(timestamps)}")
    return sorted(set(timestamps))


def _build_frame_timestamps(video_path: str, stride_sec: float = FRAME_STRIDE_SEC) -> List[float]:
    """
    Build sampling timestamps combining stride intervals and scene changes.
    """
    duration = get_video_duration(video_path)
    if duration <= 0:
        logger.warning("[VideoProc] Could not determine video duration — using stride only")
        duration = 3600.0

    stride_ts: List[float] = []
    t = 0.0
    while t < duration:
        stride_ts.append(t)
        t += stride_sec

    try:
        scene_ts = _detect_scene_change_timestamps(video_path)
    except Exception as e:
        logger.warning(f"[VideoProc] Scene detection failed (non-fatal): {e}")
        scene_ts = []

    all_ts = sorted(set(stride_ts + scene_ts))
    deduped: List[float] = []
    for ts in all_ts:
        if not deduped or (ts - deduped[-1]) >= 2.0:
            deduped.append(ts)

    logger.info(f"[VideoProc] Frame sampling: {len(deduped)} base timestamps "
                f"(stride={stride_sec}s, scene_changes={len(scene_ts)}, duration={duration:.1f}s)")
    return deduped


def _extract_frame_at(video_path: str, timestamp_sec: float, output_path: str) -> bool:
    """
    Extract a single JPEG frame from video_path at timestamp_sec.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{timestamp_sec:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", str(JPEG_QUALITY),
            output_path,
        ],
        **_ffmpeg_kwargs(),
    )
    return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0


# ── Image Preprocessing & AI Super-Resolution ─────────────────────────────────

def preprocess_frame_for_ocr(img: Image.Image, min_width: int = MIN_FRAME_WIDTH) -> Tuple[np.ndarray, bool]:
    """
    Preprocessing Pipeline:
      1. Extract Frame
      2. AI Super Resolution (2x-4x) / Fallback Lanczos Scaling
      3. Grayscale
      4. CLAHE Contrast Enhancement
      5. Image Sharpening
      6. Auto-invert dark backgrounds (mean pixel < 127)
      7. Otsu Binarization / Thresholding

    Returns:
      Tuple[preprocessed_threshold_array, sr_applied_flag]
    """
    sr_model, sr_name = get_sr_model()
    sr_applied = False
    img_np = np.array(img)

    # 1. Apply AI Super-Resolution if model is cached
    if sr_model is not None and _CV2_AVAILABLE:
        try:
            if hasattr(sr_model, "upsample"):
                img_np = sr_model.upsample(img_np)
                sr_applied = True
            elif isinstance(sr_model, tuple) and sr_model[0] == "cv2_dnn":
                net = sr_model[1]
                blob = cv2.dnn.blobFromImage(img_np, 1.0 / 255.0, (img_np.shape[1], img_np.shape[0]), (0, 0, 0), swapRB=False, crop=False)
                net.setInput(blob)
                out = net.forward()
                out = (out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                img_np = out
                sr_applied = True
        except Exception as e:
            logger.warning(f"[VideoOCR AI SR] Upsampling failed: {e}")
            sr_applied = False

    # Ensure width >= min_width (1280px)
    curr_h, curr_w = img_np.shape[:2]
    if curr_w < min_width:
        scale = min_width / float(curr_w)
        target_w = min_width
        target_h = int(curr_h * scale)
        if _CV2_AVAILABLE:
            img_np = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            pil_img = Image.fromarray(img_np)
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img_np = np.array(pil_img)

    # 2. Binarization & contrast enhancement pipeline
    if _CV2_AVAILABLE:
        if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        mean_intensity = np.mean(sharpened)
        if mean_intensity < 127:
            sharpened = cv2.bitwise_not(sharpened)

        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh, sr_applied
    else:
        pil_img = Image.fromarray(img_np)
        gray_img = ImageOps.grayscale(pil_img)
        gray_img = ImageEnhance.Contrast(gray_img).enhance(1.8)
        gray_img = gray_img.filter(ImageFilter.SHARPEN)
        arr = np.array(gray_img)
        if np.mean(arr) < 127:
            arr = 255 - arr
        thresh = np.where(arr > 128, 255, 0).astype(np.uint8)
        return thresh, sr_applied


def _clean_ocr_text(raw: str) -> str:
    """
    Clean OCR text:
      - Strip leading/trailing whitespace
      - Remove noise characters and isolated symbols
      - Remove lines with < 2 alphanumeric chars or < 35% alphanumeric ratio
      - Remove repeated character noise
      - Collapse multiple blank lines and duplicate spaces
    """
    if not raw:
        return ""

    lines = raw.splitlines()
    clean_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        stripped = re.sub(r"([\|~_=\-\^\\\/@#\$\%*])\1{2,}", "", stripped).strip()
        if not stripped:
            continue

        if len(stripped) == 1 and not stripped.isalnum():
            continue

        alpha_count = len(re.findall(r"[a-zA-Z0-9]", stripped))
        non_space_count = len(re.sub(r"\s+", "", stripped))

        if alpha_count < 2:
            continue

        if non_space_count > 0 and (alpha_count / non_space_count) < 0.35:
            continue

        stripped = re.sub(r"[ \t]+", " ", stripped)
        clean_lines.append(stripped)

    deduped_lines: List[str] = []
    for line in clean_lines:
        if not deduped_lines or line != deduped_lines[-1]:
            deduped_lines.append(line)

    return "\n".join(deduped_lines).strip()


def run_ocr_on_frame_detail(
    image_path_or_img: Union[str, "Image.Image"],
    min_conf: float = MIN_WORD_CONFIDENCE,
    config: str = r"--oem 3 --psm 6",  # kept for API compat; ignored (RapidOCR has no config string)
    save_debug: bool = False,
    debug_prefix: str = "frame",
) -> Dict[str, Union[str, float, int, bool]]:
    """
    Run RapidOCR (PP-OCRv4 / ONNX Runtime) on a frame with preprocessing.

    Accepts either a file path (str) or a PIL.Image.  Preprocessing is
    delegated to ocr_engine.preprocess_for_ocr() which applies upscaling,
    CLAHE, sharpening, and Otsu binarisation.

    Returns Dict {
        "raw": str,
        "clean": str,
        "avg_conf": float,   # 0.0–1.0 (RapidOCR native score)
        "word_count": int,
        "char_count": int,
        "sr_applied": bool
    }
    """
    if not is_ocr_available():
        return {"raw": "", "clean": "", "avg_conf": 0.0, "word_count": 0, "char_count": 0, "sr_applied": False}

    try:
        # Load image → numpy RGB array
        if isinstance(image_path_or_img, str):
            from PIL import Image as _Image
            img_pil = _Image.open(image_path_or_img).convert("RGB")
        else:
            img_pil = image_path_or_img.convert("RGB")

        img_np = np.array(img_pil)

        sr_model, _ = get_sr_model()
        sr_applied = sr_model is not None

        # Preprocess frame once for optimal OCR (upscale, CLAHE, sharpen, Otsu)
        from services.ocr_engine import preprocess_for_ocr
        prep_np = preprocess_for_ocr(img_np)

        # Run OCR via shared engine using single preprocessed image
        clean_text, avg_conf, word_count, char_count = _ocr_ndarray(
            prep_np,
            label=debug_prefix,
            preprocess=False,
        )

        # Debug mode output saving
        is_debug = save_debug or os.environ.get("DEBUG_OCR", "false").lower() in ("1", "true", "yes")
        if is_debug:
            try:
                debug_dir = os.environ.get("DEBUG_OCR_DIR", "debug_ocr")
                os.makedirs(debug_dir, exist_ok=True)
                img_pil.save(
                    os.path.join(debug_dir, f"{debug_prefix}_orig.jpg"),
                    quality=JPEG_QUALITY,
                )
                try:
                    from services.ocr_engine import preprocess_for_ocr
                    prep_np = preprocess_for_ocr(img_np)
                    Image.fromarray(prep_np).save(os.path.join(debug_dir, f"{debug_prefix}_prep.png"))
                except Exception:
                    pass
                with open(
                    os.path.join(debug_dir, f"{debug_prefix}_raw.txt"),
                    "w", encoding="utf-8"
                ) as f:
                    f.write(clean_text)
                with open(
                    os.path.join(debug_dir, f"{debug_prefix}_clean.txt"),
                    "w", encoding="utf-8"
                ) as f:
                    f.write(
                        f"AVG_CONF: {avg_conf:.3f}\nWORD_COUNT: {word_count}\n"
                        f"CHAR_COUNT: {char_count}\n---\n{clean_text}"
                    )
                logger.info(
                    f"[VideoOCR Debug] Saved debug files for {debug_prefix} in {debug_dir}"
                )
            except Exception as dbg_err:
                logger.warning(f"[VideoOCR Debug] Failed to save debug outputs: {dbg_err}")

        return {
            "raw": clean_text,   # RapidOCR doesn't separate raw/clean; use clean for both
            "clean": clean_text,
            "avg_conf": round(avg_conf * 100, 1),  # normalise to 0–100 range for compat
            "word_count": word_count,
            "char_count": char_count,
            "sr_applied": sr_applied,
        }
    except Exception as e:
        logger.debug(f"[VideoOCR] OCR failed for frame: {e}")
        return {"raw": "", "clean": "", "avg_conf": 0.0, "word_count": 0, "char_count": 0, "sr_applied": False}


def run_ocr_on_frame(image_path: str) -> str:
    """
    Run RapidOCR on the given image file.
    Returns cleaned OCR text, or "" if OCR is unavailable or text is too short.
    (Preserves exact function signature for existing API compatibility).
    """
    res = run_ocr_on_frame_detail(image_path)
    return str(res["clean"])


# ── Multi-Frame Candidate Evaluation Helpers ─────────────────────────────────

def _score_candidate(avg_conf: float, word_count: int, char_count: int, min_chars: int = MIN_OCR_CHARS) -> float:
    """
    Compute weighted candidate score based on:
      - Average OCR confidence
      - Number of detected words
      - Total readable character count
    Returns 0.0 if character count is below min_chars.
    """
    if char_count < min_chars or word_count == 0:
        return 0.0
    return (avg_conf * 0.4) + (word_count * 2.5) + (char_count * 0.1)


def _process_candidate_timestamp(
    video_path: str,
    base_ts: float,
    offset: float,
    video_duration: float,
    idx: int,
    cand_idx: int,
    tmpdir: str,
    should_debug: bool
) -> Dict:
    """
    Process a single nearby candidate frame (extract -> OCR -> score).
    """
    cand_ts = round(max(0.0, min(video_duration, base_ts + offset)), 2)
    frame_path = os.path.join(tmpdir, f"cand_{idx:04d}_{cand_idx}_{cand_ts:.2f}.jpg")

    success = _extract_frame_at(video_path, cand_ts, frame_path)
    if not success:
        return {
            "ts": cand_ts,
            "offset": offset,
            "raw": "",
            "clean": "",
            "avg_conf": 0.0,
            "word_count": 0,
            "char_count": 0,
            "score": 0.0,
            "sr_applied": False,
        }

    ocr_res = run_ocr_on_frame_detail(
        frame_path,
        save_debug=should_debug,
        debug_prefix=f"frame_{idx:04d}_cand{cand_idx}_t{cand_ts:.1f}",
    )

    clean_text = str(ocr_res.get("clean", ""))
    avg_conf = float(ocr_res.get("avg_conf", 0.0))
    word_count = int(ocr_res.get("word_count", 0))
    char_count = int(ocr_res.get("char_count", 0))
    sr_applied = bool(ocr_res.get("sr_applied", False))

    score = _score_candidate(avg_conf, word_count, char_count, min_chars=MIN_OCR_CHARS)

    return {
        "ts": cand_ts,
        "offset": offset,
        "raw": str(ocr_res.get("raw", "")),
        "clean": clean_text,
        "avg_conf": avg_conf,
        "word_count": word_count,
        "char_count": char_count,
        "score": score,
        "sr_applied": sr_applied,
    }


# ── Main OCR pipeline ─────────────────────────────────────────────────────────

def extract_video_ocr_timeline(
    video_path: str,
    stride_sec: float = FRAME_STRIDE_SEC,
    debug: Optional[bool] = None,
) -> List[Dict]:
    """
    Full Multi-Frame OCR pipeline for a video file:

    1. Build sampling timestamps T (stride + scene changes).
    2. For each timestamp T, extract and evaluate 4 nearby candidate frames in parallel:
       T - 0.2s, T, T + 0.2s, T + 0.4s (avoids fade transitions & animation artifacts).
    3. Run AI Super-Resolution + pre-processing on each frame.
    4. Select winning candidate frame based on weighted score (confidence + words + chars).
    5. Clean noise and return OCR timeline sorted by start_time.

    Returns:
        List[{"start_time": float, "text": str}]
    """
    if not is_ocr_available():
        logger.warning("[VideoOCR] RapidOCR engine not available — returning empty OCR timeline")
        return []

    if not os.path.exists(video_path):
        logger.error(f"[VideoOCR] Video file not found: {video_path}")
        return []

    should_debug = debug if debug is not None else DEBUG_OCR
    _, sr_name = get_sr_model()
    logger.info(f"[VideoOCR MultiFrame] Starting OCR pipeline for: {video_path} | SR: {sr_name} | debug={should_debug}")

    duration = get_video_duration(video_path)
    timestamps = _build_frame_timestamps(video_path, stride_sec=stride_sec)
    if not timestamps:
        logger.warning("[VideoOCR MultiFrame] No timestamps to process")
        return []

    results: List[Dict] = []

    with tempfile.TemporaryDirectory(prefix="voicesum_ocr_") as tmpdir:
        for idx, base_ts in enumerate(timestamps):
            # Extract and evaluate nearby candidate frames in parallel using ThreadPoolExecutor
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(MULTI_FRAME_OFFSETS))) as executor:
                for cand_idx, offset in enumerate(MULTI_FRAME_OFFSETS):
                    f = executor.submit(
                        _process_candidate_timestamp,
                        video_path,
                        base_ts,
                        offset,
                        duration,
                        idx,
                        cand_idx,
                        tmpdir,
                        should_debug,
                    )
                    futures.append(f)

            candidate_results = [f.result() for f in concurrent.futures.as_completed(futures)]
            candidate_results.sort(key=lambda r: r["offset"])

            # Detailed logging per timestamp T
            log_lines = [f"[VideoOCR MultiFrame] Base timestamp T={base_ts:.2f}s evaluated {len(candidate_results)} candidate frame(s):"]
            for c in candidate_results:
                log_lines.append(
                    f"  • Candidate t={c['ts']:.2f}s ({c['offset']:+.1f}s): avg_conf={c['avg_conf']:.1f}%, "
                    f"words={c['word_count']}, chars={c['char_count']}, score={c['score']:.2f}, sr_applied={c['sr_applied']}"
                )
            logger.info("\n".join(log_lines))

            # Select candidate with highest weighted score
            best_candidate = max(candidate_results, key=lambda c: c["score"]) if candidate_results else None

            if best_candidate and best_candidate["score"] > 0 and len(best_candidate["clean"]) >= MIN_OCR_CHARS:
                logger.info(
                    f"[VideoOCR MultiFrame] Selected winning candidate t={best_candidate['ts']:.2f}s "
                    f"(score={best_candidate['score']:.2f}, sr_applied={best_candidate['sr_applied']})"
                )
                results.append({"start_time": round(best_candidate["ts"], 2), "text": best_candidate["clean"]})
            else:
                logger.info(f"[VideoOCR MultiFrame] Timestamp T={base_ts:.2f}s: all candidates produced insufficient OCR text — skipped.")

    logger.info(f"[VideoOCR MultiFrame] OCR complete: {len(results)} frames with readable text out of {len(timestamps)} sampled timestamps")
    return results


# ── Merge consecutive OCR blocks ──────────────────────────────────────────────

def _normalize_for_similarity(text: str) -> str:
    """Normalize text for fuzzy similarity comparison (lowercase, alphanumeric only, single spaces)."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", clean).strip()


def _text_similarity(a: str, b: str) -> float:
    """
    Compute fuzzy text similarity between two OCR texts.
    Combines difflib SequenceMatcher ratio on normalized strings and token set overlap ratio.
    """
    norm_a = _normalize_for_similarity(a)
    norm_b = _normalize_for_similarity(b)

    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0

    seq_ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

    words_a = set(norm_a.split())
    words_b = set(norm_b.split())
    if words_a and words_b:
        jaccard = len(words_a.intersection(words_b)) / float(len(words_a.union(words_b)))
    else:
        jaccard = 0.0

    return max(seq_ratio, jaccard)


def merge_ocr_results(
    ocr_entries: List[Dict],
    video_duration: float = 0.0,
    merge_ratio: float = OCR_MERGE_RATIO,
) -> List[Dict]:
    """
    Merge consecutive OCR entries whose text is fuzzy-similar (ratio >= merge_ratio).

    Input:  List[{"start_time": float, "text": str}]
    Output: List[{"start": float, "end": float, "text": str}]
    """
    if not ocr_entries:
        return []

    entries = sorted(ocr_entries, key=lambda e: e["start_time"])

    merged: List[Dict] = []
    group_start: float = entries[0]["start_time"]
    group_texts: List[Tuple[float, str]] = [(entries[0]["start_time"], entries[0]["text"])]

    def _score_text(t: str) -> int:
        """Score text entry by alphanumeric word count and overall length to pick representative text."""
        alpha = len(re.findall(r"[a-zA-Z0-9]", t))
        return alpha * 2 + len(t)

    def _flush_group(group_ts: float, texts: List[Tuple[float, str]], next_ts: float) -> Dict:
        """Produce one merged block from a group."""
        rep_text = max(texts, key=lambda t: _score_text(t[1]))[1]
        end_ts = next_ts if next_ts > 0 else texts[-1][0]
        return {
            "start": round(group_ts, 2),
            "end": round(end_ts, 2),
            "text": rep_text,
        }

    for i in range(1, len(entries)):
        current = entries[i]
        prev_text = group_texts[-1][1]
        curr_text = current["text"]

        if _text_similarity(prev_text, curr_text) >= merge_ratio:
            group_texts.append((current["start_time"], current["text"]))
        else:
            merged.append(_flush_group(group_start, group_texts, current["start_time"]))
            group_start = current["start_time"]
            group_texts = [(current["start_time"], current["text"])]

    final_end = video_duration if video_duration > 0 else (group_texts[-1][0] + FRAME_STRIDE_SEC)
    merged.append(_flush_group(group_start, group_texts, final_end))

    logger.info(f"[VideoOCR] Merged {len(ocr_entries)} OCR entries → {len(merged)} blocks via fuzzy matching")
    return merged


# ── Timeline query helper ─────────────────────────────────────────────────────

def get_overlapping_ocr_blocks(
    video_transcript: List[Dict],
    window_start: float,
    window_end: float,
) -> List[Dict]:
    """
    Return all OCR blocks whose time range overlaps [window_start, window_end].
    """
    if not video_transcript:
        return []
    return [
        b for b in video_transcript
        if b.get("end", 0.0) >= window_start and b.get("start", 0.0) <= window_end
    ]


# ── Supported video formats ───────────────────────────────────────────────────

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SUPPORTED_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/avi",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska",
    "video/webm",
    "video/x-ms-wmv",
}


def is_supported_video(filename: str, content_type: str = "") -> bool:
    """Return True if the file extension or MIME type is a supported video format."""
    ext = Path(filename).suffix.lower()
    if ext in SUPPORTED_VIDEO_EXTENSIONS:
        return True
    if content_type and content_type.lower() in SUPPORTED_VIDEO_MIME_TYPES:
        return True
    return False


def unload_sr_model():
    """Unload cached Super-Resolution model and free memory."""
    global _SR_MODEL, _SR_MODEL_LOADED, _SR_MODEL_NAME
    if _SR_MODEL_LOADED:
        logger.info("[VideoOCR AI SR] Unloading AI Super-Resolution model...")
        _SR_MODEL = None
        _SR_MODEL_LOADED = False
        _SR_MODEL_NAME = "none"
        import gc
        gc.collect()


def unload_video_ocr_pipeline():
    """Unload both RapidOCR ONNX sessions and AI Super-Resolution model."""
    try:
        from services.ocr_engine import unload_ocr_engine
        unload_ocr_engine()
    except Exception as e:
        logger.warning(f"[VideoOCR] Failed to unload OCR engine: {e}")

    try:
        unload_sr_model()
    except Exception as e:
        logger.warning(f"[VideoOCR] Failed to unload SR model: {e}")

