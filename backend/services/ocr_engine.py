"""
ocr_engine.py — Shared OCR engine using PP-OCRv6 Medium + ONNX Runtime.

PP-OCRv6 Medium (34.5M parameters, released June 2026 by PaddlePaddle) is a state-of-the-art,
production-ready universal OCR model family. It features:
  • LCNetV4 MetaFormer-style backbone with structural reparameterization
  • RepLKFPN large-kernel detection neck
  • EncoderWithLightSVTR local-global attention recognition neck
  • Unified multilingual support covering 50 languages

Pipeline Architecture:
  Input Image → Preprocessing → Text Detection → Page Orientation / Deskew →
  Crop & Text-line Orientation → Batched Text Recognition → Reading Order / Layout → Final OCR Result

With structured output on (services/ocr_layout.py) the final text keeps tables as Markdown
rows and columns, wide gaps inside a line as a tab and paragraph breaks as a blank line.

Every stage follows the official PaddleX PP-OCRv6 pipeline (DetResizeForTest, DBPostProcess,
get_rotate_crop_image, OCRReisizeNormImg, CTCLabelDecode, sorted_boxes) and is driven by the
system-wide OCR settings in services.ocr_settings, so every OCR entry point (documents,
scanned PDFs, embedded images, video frames) uses the same saved configuration.

Offline Model Storage:
  All model weights and dictionary files are stored strictly inside the project's
  `download/ppocrv6/` directory:
    - PP-OCRv6_medium_det.onnx   (Detection model, ~62 MB)
    - PP-OCRv6_medium_rec.onnx   (Recognition model, ~76.6 MB)
    - ch_ppocr_mobile_v2.0_cls.onnx (Orientation classifier, ~0.6 MB)
    - ppocrv6_dict.txt           (50-language dictionary, ~18,700 tokens)

  100% Offline: The engine operates entirely locally. No network requests,
  external API calls, or automatic model downloads are ever performed at runtime.
"""
from __future__ import annotations

import logging
import math
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional dependency flags ─────────────────────────────────────────────────
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import pyclipper
    _PYCLIPPER_AVAILABLE = True
except ImportError:
    _PYCLIPPER_AVAILABLE = False

try:
    from shapely.geometry import Polygon
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    from services.ocr_settings import OcrSettings, get_ocr_settings
    from services.ocr_preprocessing import apply_preprocessing, ensure_rgb, load_image_rgb
except ImportError:
    from backend.services.ocr_settings import OcrSettings, get_ocr_settings  # type: ignore[no-redef]
    from backend.services.ocr_preprocessing import (  # type: ignore[no-redef]
        apply_preprocessing, ensure_rgb, load_image_rgb,
    )


# ── Fixed model constants (properties of the shipped ONNX models) ─────────────
MIN_FRAME_WIDTH: int = 0         # Legacy preprocess_for_ocr min width (0 = natural resolution)
REC_IMAGE_HEIGHT: int = 48       # PP-OCRv6 recognition input height
REC_BASE_WIDTH: int = 320        # RecResizeImg image_shape [3, 48, 320] (inference_rec.yml)
CLS_IMAGE_HEIGHT: int = 48       # Orientation classifier input [3, 48, 192]
CLS_IMAGE_WIDTH: int = 192
CLS_BATCH_SIZE: int = 6          # PaddleOCR cls_batch_num
DET_MIN_SIZE: int = 3            # DBPostProcess min_size
SORT_LINE_TOLERANCE: float = 10  # PaddleOCR sorted_boxes same-line tolerance (px)

_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Path resolution ───────────────────────────────────────────────────────────

def get_model_dir() -> Path:
    """
    Resolve the absolute path to the offline PaddleOCR model directory.
    Prioritizes `backend/runtime/paddleocr` first, then frozen bundles, then fallbacks.
    """
    # 1. Primary path: backend/runtime/paddleocr (relative to services/ocr_engine.py)
    backend_runtime = Path(__file__).resolve().parent.parent / "runtime" / "paddleocr"
    if backend_runtime.exists() and (backend_runtime / "PP-OCRv6_medium_det.onnx").exists():
        return backend_runtime

    # 1b. Check config.RUNTIME_DIR if available
    try:
        from config import RUNTIME_DIR
        cfg_runtime = RUNTIME_DIR / "paddleocr"
        if cfg_runtime.exists() and (cfg_runtime / "PP-OCRv6_medium_det.onnx").exists():
            return cfg_runtime
    except Exception:
        pass

    # 2. PyInstaller frozen application (_MEIPASS)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        for sub in ("runtime/paddleocr", "backend/runtime/paddleocr", "download/ppocrv6"):
            bundle_dir = Path(sys._MEIPASS) / sub
            if bundle_dir.exists() and (bundle_dir / "PP-OCRv6_medium_det.onnx").exists():
                return bundle_dir

    # 3. Workspace root relative to this file (backend/services/ocr_engine.py -> project_root)
    proj_root = Path(__file__).resolve().parent.parent.parent
    for candidate in (
        proj_root / "backend" / "runtime" / "paddleocr",
        proj_root / "download" / "ppocrv6",
    ):
        if candidate.exists() and (candidate / "PP-OCRv6_medium_det.onnx").exists():
            return candidate

    # 4. Current working directory fallback
    for cwd_sub in ("runtime/paddleocr", "backend/runtime/paddleocr", "download/ppocrv6"):
        cwd_dir = Path.cwd() / cwd_sub
        if cwd_dir.exists() and (cwd_dir / "PP-OCRv6_medium_det.onnx").exists():
            return cwd_dir

    return backend_runtime


# ── Geometry helpers (ported from PaddleOCR / PaddleX) ────────────────────────

def get_mini_boxes(contour: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Minimum-area rectangle of a contour with corners ordered
    [top-left, top-right, bottom-right, bottom-left]. Returns (box, shorter side).
    """
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda p: p[0])
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2
    box = np.array([points[index_1], points[index_2], points[index_3], points[index_4]], dtype=np.float32)
    return box, float(min(bounding_box[1]))


def _box_score(bitmap: np.ndarray, points: np.ndarray) -> float:
    """Mean probability inside a polygon, evaluated on its bounding rectangle only."""
    h, w = bitmap.shape[:2]
    pts = points.reshape(-1, 2).astype(np.float32).copy()
    xmin = int(np.clip(np.floor(pts[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(pts[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(pts[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(pts[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    pts[:, 0] -= xmin
    pts[:, 1] -= ymin
    cv2.fillPoly(mask, pts.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0])


def _unclip(box: np.ndarray, unclip_ratio: float) -> Optional[np.ndarray]:
    """Expand a polygon by area × ratio / perimeter (Vatti clipping)."""
    if _SHAPELY_AVAILABLE:
        poly = Polygon(box)
        if poly.length <= 0 or poly.area <= 0:
            return None
        distance = poly.area * unclip_ratio / poly.length
    else:
        area = cv2.contourArea(box.astype(np.float32))
        perimeter = cv2.arcLength(box.astype(np.float32), True)
        if perimeter <= 0 or area <= 0:
            return None
        distance = area * unclip_ratio / perimeter
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if not expanded:
        return None
    return np.concatenate([np.array(p) for p in expanded], axis=0)


def pad_box(box: np.ndarray, ratio: float, width: int, height: int) -> np.ndarray:
    """
    Grow an ordered quad by `ratio` × its line height on every side, along the box's own
    axes, clipped to the image. Detected boxes hug the glyphs; the recognizer keeps word
    boundaries better with a little margin around the line.
    """
    if ratio <= 0:
        return box
    b = np.asarray(box, dtype=np.float32)
    u, v = b[1] - b[0], b[3] - b[0]
    len_u, len_v = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if len_u < 1 or len_v < 1:
        return box
    margin = ratio * min(len_u, len_v)
    du, dv = u / len_u * margin, v / len_v * margin
    out = np.array([b[0] - du - dv, b[1] + du - dv, b[2] + du + dv, b[3] - du + dv], dtype=np.float32)
    out[:, 0] = np.clip(out[:, 0], 0, width)
    out[:, 1] = np.clip(out[:, 1], 0, height)
    return out


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> Optional[np.ndarray]:
    """Perspective-rectify a text box; tall crops (h/w ≥ 1.5) are rotated 90° CCW (official)."""
    points = points.astype(np.float32)
    crop_w = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    crop_h = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    if crop_w < 2 or crop_h < 2:
        return None
    pts_std = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst = cv2.warpPerspective(
        img, M, (crop_w, crop_h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )
    if dst.shape[0] * 1.0 / dst.shape[1] >= 1.5:
        dst = np.rot90(dst)
    return np.ascontiguousarray(dst)


def _box_extent(box: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    arr = np.asarray(box, dtype=np.float32)
    return float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())


def _box_sides(box: np.ndarray) -> Tuple[float, float]:
    """(width along the top edge, height along the left edge) of an ordered box."""
    return float(np.linalg.norm(box[0] - box[1])), float(np.linalg.norm(box[0] - box[3]))


def _rotate_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate counter-clockwise by angle_deg, expanding the canvas so nothing is clipped."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2.0 - center[0]
    M[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ── Reading order ─────────────────────────────────────────────────────────────

@dataclass
class OcrLine:
    box: List[List[float]]
    text: str
    score: float


def sort_boxes(items: List[OcrLine]) -> List[OcrLine]:
    """PaddleOCR `sorted_boxes`: top-to-bottom, left-to-right with a same-line tolerance."""
    ordered = sorted(items, key=lambda it: (it.box[0][1], it.box[0][0]))
    for i in range(len(ordered) - 1):
        for j in range(i, -1, -1):
            if (abs(ordered[j + 1].box[0][1] - ordered[j].box[0][1]) < SORT_LINE_TOLERANCE
                    and ordered[j + 1].box[0][0] < ordered[j].box[0][0]):
                ordered[j], ordered[j + 1] = ordered[j + 1], ordered[j]
            else:
                break
    return ordered


def _join_line(parts: List[OcrLine]) -> OcrLine:
    xs0, ys0, xs1, ys1 = zip(*(_box_extent(p.box) for p in parts))
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    weights = [max(1, len(p.text)) for p in parts]
    score = float(sum(p.score * wt for p, wt in zip(parts, weights)) / sum(weights))
    return OcrLine(
        box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        text=" ".join(p.text for p in parts),
        score=score,
    )


def merge_lines(items: List[OcrLine], max_gap: float) -> List[OcrLine]:
    """
    Group regions that share a visual line (vertical overlap ≥ 50% of the smaller
    height) and join them left-to-right. Regions further apart than `max_gap`
    line heights stay separate so neighbouring columns are not fused.
    """
    if not items:
        return []
    extents = [_box_extent(it.box) for it in items]
    order = sorted(range(len(items)), key=lambda i: (extents[i][1] + extents[i][3]) / 2.0)

    rows: List[List[int]] = []
    row_top = row_bot = 0.0
    for idx in order:
        _, y0, _, y1 = extents[idx]
        h = max(1.0, y1 - y0)
        if rows:
            overlap = min(y1, row_bot) - max(y0, row_top)
            if overlap >= 0.5 * min(h, max(1.0, row_bot - row_top)):
                rows[-1].append(idx)
                row_top, row_bot = min(row_top, y0), max(row_bot, y1)
                continue
        rows.append([idx])
        row_top, row_bot = y0, y1

    lines: List[OcrLine] = []
    for row in rows:
        row.sort(key=lambda i: extents[i][0])
        heights = [max(1.0, extents[i][3] - extents[i][1]) for i in row]
        row_h = float(np.median(heights))
        segment = [row[0]]
        for prev, cur in zip(row, row[1:]):
            gap = extents[cur][0] - extents[prev][2]
            if gap > max_gap * row_h:
                lines.append(_join_line([items[i] for i in segment]))
                segment = [cur]
            else:
                segment.append(cur)
        lines.append(_join_line([items[i] for i in segment]))
    return lines


def xy_cut_order(items: List[OcrLine]) -> List[OcrLine]:
    """
    Column-aware reading order (recursive XY-cut). Columns are split first
    (left to right), then horizontal bands (top to bottom).
    """
    if len(items) <= 1:
        return list(items)
    extents = np.array([_box_extent(it.box) for it in items], dtype=np.float32)
    med_h = float(np.median(np.maximum(1.0, extents[:, 3] - extents[:, 1])))
    x_gap_min, y_gap_min = 1.0 * med_h, 1.5 * med_h

    def _gaps(lo: np.ndarray, hi: np.ndarray, min_gap: float) -> List[float]:
        order = np.argsort(lo)
        cuts, reach = [], hi[order[0]]
        for k in order[1:]:
            if lo[k] - reach >= min_gap:
                cuts.append((reach + lo[k]) / 2.0)
            reach = max(reach, hi[k])
        return cuts

    def _split(idx: List[int], depth: int) -> List[int]:
        if len(idx) <= 1 or depth > 32:
            return idx
        sub = extents[idx]
        for axis, min_gap in ((0, x_gap_min), (1, y_gap_min)):
            cuts = _gaps(sub[:, axis], sub[:, axis + 2], min_gap)
            if cuts:
                bounds = [-math.inf] + cuts + [math.inf]
                out: List[int] = []
                for lo, hi in zip(bounds, bounds[1:]):
                    part = [i for i in idx if lo <= (extents[i][axis] + extents[i][axis + 2]) / 2.0 < hi]
                    out.extend(_split(part, depth + 1))
                return out
        return sorted(idx, key=lambda i: (extents[i][1], extents[i][0]))

    return [items[i] for i in _split(list(range(len(items))), 0)]


def order_lines(items: List[OcrLine], cfg: OcrSettings) -> List[OcrLine]:
    """Apply the configured reading-order strategy."""
    if cfg.merge_lines:
        ordered = merge_lines(items, cfg.line_merge_max_gap)  # already row-major, left to right
    else:
        ordered = sort_boxes(items)
    if cfg.column_detection:
        ordered = xy_cut_order(ordered)
    return ordered


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class OcrResult:
    lines: List[OcrLine] = field(default_factory=list)
    text: str = ""
    avg_conf: float = 0.0
    word_count: int = 0
    char_count: int = 0
    box_count: int = 0
    image_size: Tuple[int, int] = (0, 0)          # (w, h) of the image OCR ran on
    det_input_size: Tuple[int, int] = (0, 0)      # (w, h) fed to the detector
    rotation: int = 0                              # page rotation applied (degrees CCW)
    skew_angle: float = 0.0                        # deskew rotation applied (degrees)
    table_count: int = 0                           # tables rebuilt by the structured layout
    applied_steps: List[str] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    image: Optional[np.ndarray] = None             # processed RGB image (only when requested)


# ── PP-OCRv6 Native Engine ───────────────────────────────────────────────────

def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("out of memory", "failed to allocate", "bfcarena", "alloc_failed"))


class PPOCRv6Engine:
    """
    Native ONNX Runtime inference engine for PP-OCRv6 Medium.
    Executes detection, direction classification, and recognition 100% offline.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        if not _ORT_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed. Please install onnxruntime-gpu or onnxruntime.")
        if not _CV2_AVAILABLE:
            raise RuntimeError("opencv-python is not installed. Please install opencv-python-headless.")
        if not _PYCLIPPER_AVAILABLE:
            raise RuntimeError("pyclipper is not installed.")

        self.model_dir = model_dir or get_model_dir()
        self.det_path = self.model_dir / "PP-OCRv6_medium_det.onnx"
        self.rec_path = self.model_dir / "PP-OCRv6_medium_rec.onnx"
        self.cls_path = self.model_dir / "ch_ppocr_mobile_v2.0_cls.onnx"
        self.dict_path = self.model_dir / "ppocrv6_dict.txt"

        self._validate_model_files()
        self.vocab = self._load_dictionary()

        # Session configuration
        self._init_cuda_environment()
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            # Native-resolution inputs vary in shape: heuristic cuDNN algo search avoids
            # re-benchmarking every new shape, and exact-size arena growth limits VRAM waste.
            providers = [
                ("CUDAExecutionProvider", {
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "arena_extend_strategy": "kSameAsRequested",
                }),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        logger.info(f"[PPOCRv6] Initialising ONNX Runtime sessions from {self.model_dir}")
        self.det_session = ort.InferenceSession(str(self.det_path), self.sess_options, providers=providers)
        self.rec_session = ort.InferenceSession(str(self.rec_path), self.sess_options, providers=providers)
        self.cls_session = (
            ort.InferenceSession(str(self.cls_path), self.sess_options, providers=providers)
            if self.cls_path.exists()
            else None
        )
        self.provider = self.det_session.get_providers()[0]
        self.uses_gpu = self.provider == "CUDAExecutionProvider"
        # Detection activations dominate memory at native resolution — run one at a time.
        self._det_lock = threading.Lock()
        logger.info(
            f"[PPOCRv6] Sessions initialised successfully. "
            f"Active det provider: {self.provider}, "
            f"rec provider: {self.rec_session.get_providers()[0]}"
        )

    def _validate_model_files(self) -> None:
        """Verify all mandatory model files exist locally without making network calls."""
        missing = []
        for path in (self.det_path, self.rec_path, self.dict_path):
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                f"[PPOCRv6] Missing offline model files in {self.model_dir}: {missing}. "
                f"Please ensure models are stored in download/ppocrv6/."
            )

    def _load_dictionary(self) -> List[str]:
        """Load character dictionary from local text file."""
        with open(self.dict_path, "r", encoding="utf-8") as f:
            chars = [line.rstrip("\r\n") for line in f]
        # Vocabulary mapping for CTC decode:
        # index 0: 'blank', index 1..len(chars): characters, index len(chars)+1: ' ' (space)
        vocab = ["blank"] + chars + [" "]
        return vocab

    @staticmethod
    def _init_cuda_environment() -> None:
        """Register PyTorch CUDA DLL directory on Windows if available."""
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

    # ── Pipeline Stage 1 & 2: Text Detection ──────────────────────────────────

    @staticmethod
    def det_resize_shape(h: int, w: int, cfg: OcrSettings) -> Tuple[int, int]:
        """Official DetResizeForTest: returns the (resize_h, resize_w) fed to the detector."""
        limit = cfg.det_limit_side_len
        if cfg.det_limit_type == "max":
            ratio = float(limit) / max(h, w) if max(h, w) > limit else 1.0
        else:
            ratio = float(limit) / min(h, w) if min(h, w) < limit else 1.0
        resize_h, resize_w = int(h * ratio), int(w * ratio)
        if max(resize_h, resize_w) > cfg.det_max_side_limit:
            ratio = float(cfg.det_max_side_limit) / max(resize_h, resize_w)
            resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)
        resize_h = max(int(round(resize_h / 32.0) * 32), 32)
        resize_w = max(int(round(resize_w / 32.0) * 32), 32)
        return resize_h, resize_w

    def _det_forward(self, resized_bgr: np.ndarray) -> np.ndarray:
        norm = (resized_bgr.astype(np.float32) / 255.0 - _DET_MEAN) / _DET_STD
        chw = np.ascontiguousarray(np.transpose(norm, (2, 0, 1))[np.newaxis, :])
        with self._det_lock:
            det_out = self.det_session.run(None, {"x": chw})[0]
        return det_out[0, 0]

    def detect_text(
        self, bgr_img: np.ndarray, cfg: Optional[OcrSettings] = None
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Run DBNet text detection on a BGR image.
        Returns list of (ordered_box_4x2, detection_score) in original image coordinates.
        """
        boxes, _ = self._detect(bgr_img, cfg or get_ocr_settings())
        return boxes

    def _detect(
        self, bgr_img: np.ndarray, cfg: OcrSettings
    ) -> Tuple[List[Tuple[np.ndarray, float]], Tuple[int, int]]:
        orig_h, orig_w = bgr_img.shape[:2]
        if orig_h < 3 or orig_w < 3:
            return [], (0, 0)

        src = bgr_img
        if orig_h + orig_w < 64:  # official: pad tiny images to at least 32×32
            src = np.zeros((max(32, orig_h), max(32, orig_w), 3), dtype=np.uint8)
            src[:orig_h, :orig_w] = bgr_img
        src_h, src_w = src.shape[:2]

        resize_h, resize_w = self.det_resize_shape(src_h, src_w, cfg)
        resized = cv2.resize(src, (resize_w, resize_h))
        try:
            prob_map = self._det_forward(resized)
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            resize_h = max(32, (resize_h // 2) // 32 * 32)
            resize_w = max(32, (resize_w // 2) // 32 * 32)
            logger.warning(f"[PPOCRv6] Detection ran out of memory — retrying at {resize_w}x{resize_h}")
            prob_map = self._det_forward(cv2.resize(src, (resize_w, resize_h)))

        boxes = self._db_postprocess(prob_map, src_h, src_w, cfg)
        if (src_h, src_w) != (orig_h, orig_w):
            for box, _ in boxes:
                box[:, 0] = np.clip(box[:, 0], 0, orig_w)
                box[:, 1] = np.clip(box[:, 1], 0, orig_h)
        return boxes, (resize_w, resize_h)

    @staticmethod
    def _db_postprocess(
        prob_map: np.ndarray, src_h: int, src_w: int, cfg: OcrSettings
    ) -> List[Tuple[np.ndarray, float]]:
        """Official DBPostProcess (quad boxes)."""
        height, width = prob_map.shape
        mask = (prob_map > cfg.det_db_thresh).astype(np.uint8)
        if cfg.det_use_dilation:
            mask = cv2.dilate(mask, np.array([[1, 1], [1, 1]], dtype=np.uint8))
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Tuple[np.ndarray, float]] = []
        for contour in contours[: cfg.det_max_candidates]:
            points, sside = get_mini_boxes(contour)
            if sside < DET_MIN_SIZE:
                continue
            if cfg.det_score_mode == "slow":
                score = _box_score(prob_map, contour.reshape(-1, 2))
            else:
                score = _box_score(prob_map, points)
            if score < cfg.det_db_box_thresh:
                continue

            expanded = _unclip(points, cfg.det_db_unclip_ratio)
            if expanded is None or len(expanded) < 3:
                continue
            box, sside = get_mini_boxes(expanded.reshape(-1, 1, 2).astype(np.float32))
            if sside < DET_MIN_SIZE + 2:
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * src_w), 0, src_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * src_h), 0, src_h)
            boxes.append((box.astype(np.float32), score))
        return boxes

    # ── Pipeline Stage 3: Crop & Orientation ──────────────────────────────────

    def crop_and_orient(
        self, bgr_img: np.ndarray, box: np.ndarray, cfg: Optional[OcrSettings] = None
    ) -> Optional[np.ndarray]:
        """Extract a rectified text crop and correct 180° rotation (single crop)."""
        cfg = cfg or get_ocr_settings()
        h, w = bgr_img.shape[:2]
        crop = get_rotate_crop_image(bgr_img, pad_box(box, cfg.rec_crop_padding, w, h))
        if crop is None or not cfg.use_textline_orientation:
            return crop
        return self.apply_textline_orientation([crop], cfg)[0][0]

    def classify_orientation(self, crops: List[np.ndarray]) -> List[Tuple[int, float]]:
        """Batched 0°/180° classification. Returns (label, score) per crop (label 1 = 180°)."""
        if self.cls_session is None or not crops:
            return [(0, 1.0)] * len(crops)
        results: List[Tuple[int, float]] = [(0, 1.0)] * len(crops)
        order = np.argsort([c.shape[1] / float(max(1, c.shape[0])) for c in crops])
        for beg in range(0, len(crops), CLS_BATCH_SIZE):
            idxs = order[beg:beg + CLS_BATCH_SIZE]
            batch = np.zeros((len(idxs), 3, CLS_IMAGE_HEIGHT, CLS_IMAGE_WIDTH), dtype=np.float32)
            for k, i in enumerate(idxs):
                crop = crops[i]
                ratio = crop.shape[1] / float(max(1, crop.shape[0]))
                resized_w = min(CLS_IMAGE_WIDTH, max(1, int(math.ceil(CLS_IMAGE_HEIGHT * ratio))))
                resized = cv2.resize(crop, (resized_w, CLS_IMAGE_HEIGHT)).astype(np.float32)
                batch[k, :, :, :resized_w] = np.transpose((resized / 255.0 - 0.5) / 0.5, (2, 0, 1))
            probs = self.cls_session.run(None, {"x": batch})[0]
            for k, i in enumerate(idxs):
                label = int(np.argmax(probs[k]))
                results[i] = (label, float(probs[k][label]))
        return results

    def apply_textline_orientation(
        self, crops: List[np.ndarray], cfg: OcrSettings,
        labels: Optional[List[Tuple[int, float]]] = None,
    ) -> List[Tuple[np.ndarray, Tuple[int, float]]]:
        """Flip crops classified as 180° with confidence above cls_thresh."""
        labels = labels if labels is not None else self.classify_orientation(crops)
        out = []
        for crop, (label, score) in zip(crops, labels):
            if label == 1 and score > cfg.cls_thresh:
                crop = cv2.rotate(crop, cv2.ROTATE_180)
            out.append((crop, (label, score)))
        return out

    # ── Pipeline Stage 4 & 5: Text Recognition & CTC Decode ───────────────────

    def _ctc_decode(self, probs: np.ndarray) -> Tuple[str, float]:
        preds = np.argmax(probs, axis=1)
        maxp = np.max(probs, axis=1)
        vocab_len = len(self.vocab)
        chars: List[str] = []
        confs: List[float] = []
        last_idx = 0
        for idx, prob in zip(preds, maxp):
            if idx != 0 and idx != last_idx and idx < vocab_len:
                chars.append(self.vocab[idx])
                confs.append(float(prob))
            last_idx = idx
        text = "".join(chars).strip()
        return text, (float(np.mean(confs)) if confs else 0.0)

    @staticmethod
    def _rec_norm(crop: np.ndarray, target_w: int, max_width: int) -> Tuple[np.ndarray, int]:
        """Official resize_norm_img: keep aspect ratio at height 48 (squeeze above max_width)."""
        h, w = crop.shape[:2]
        resized_w = int(math.ceil(REC_IMAGE_HEIGHT * w / float(max(1, h))))
        resized_w = max(1, min(resized_w, target_w, max_width))
        resized = cv2.resize(crop, (resized_w, REC_IMAGE_HEIGHT)).astype(np.float32)
        norm = np.transpose((resized / 255.0 - 0.5) / 0.5, (2, 0, 1))
        return norm, resized_w

    def _rec_run(self, batch_crops: List[np.ndarray], max_width: int) -> List[Tuple[str, float]]:
        max_wh_ratio = REC_BASE_WIDTH / float(REC_IMAGE_HEIGHT)
        for crop in batch_crops:
            max_wh_ratio = max(max_wh_ratio, crop.shape[1] / float(max(1, crop.shape[0])))
        img_w = min(max_width, int(REC_IMAGE_HEIGHT * max_wh_ratio))
        batch = np.zeros((len(batch_crops), 3, REC_IMAGE_HEIGHT, img_w), dtype=np.float32)
        for k, crop in enumerate(batch_crops):
            norm, resized_w = self._rec_norm(crop, img_w, max_width)
            batch[k, :, :, :resized_w] = norm  # right-pad with zeros (official)
        out = self.rec_session.run(None, {"x": batch})[0]  # (N, T, vocab)
        return [self._ctc_decode(out[k]) for k in range(len(batch_crops))]

    def recognize_batch(
        self, crops: List[np.ndarray], cfg: Optional[OcrSettings] = None
    ) -> List[Tuple[str, float]]:
        """Recognise crops in aspect-ratio-sorted batches; falls back to one-by-one on error."""
        cfg = cfg or get_ocr_settings()
        results: List[Tuple[str, float]] = [("", 0.0)] * len(crops)
        valid = [i for i, c in enumerate(crops) if c is not None and c.shape[0] >= 2 and c.shape[1] >= 2]
        order = sorted(valid, key=lambda i: crops[i].shape[1] / float(crops[i].shape[0]))
        batch_size = max(1, cfg.rec_batch_size)
        for beg in range(0, len(order), batch_size):
            idxs = order[beg:beg + batch_size]
            try:
                decoded = self._rec_run([crops[i] for i in idxs], cfg.rec_max_width)
            except Exception as exc:
                if len(idxs) == 1:
                    logger.debug(f"[PPOCRv6] Recognition failed for one crop: {exc}")
                    continue
                logger.debug(f"[PPOCRv6] Batched recognition failed ({exc}); retrying one by one")
                decoded = []
                for i in idxs:
                    try:
                        decoded.extend(self._rec_run([crops[i]], cfg.rec_max_width))
                    except Exception:
                        decoded.append(("", 0.0))
            for i, res in zip(idxs, decoded):
                results[i] = res
        return results

    def recognize_text(self, crop_bgr: np.ndarray, cfg: Optional[OcrSettings] = None) -> Tuple[str, float]:
        """Run PP-OCRv6 recognition on one rectified text line patch."""
        return self.recognize_batch([crop_bgr], cfg)[0]

    # ── Page-level heuristics ──────────────────────────────────────────────────

    @staticmethod
    def _mostly_vertical(boxes: List[Tuple[np.ndarray, float]]) -> bool:
        """True when most elongated boxes are taller than wide (page rotated 90°/270°)."""
        elongated = vertical = 0
        for box, _ in boxes:
            w, h = _box_sides(box)
            if max(w, h) >= 2.0 * max(1.0, min(w, h)):
                elongated += 1
                vertical += h > w
        return elongated >= 3 and vertical / elongated >= 0.6

    @staticmethod
    def _estimate_skew(boxes: List[Tuple[np.ndarray, float]], max_angle: float) -> float:
        """Median angle (degrees, positive = sloping down to the right) of long text lines."""
        angles = []
        for box, _ in boxes:
            w, h = _box_sides(box)
            if w >= 3.0 * max(1.0, h):
                dx, dy = box[1][0] - box[0][0], box[1][1] - box[0][1]
                angles.append(math.degrees(math.atan2(dy, dx)))
        if len(angles) < 3:
            return 0.0
        angle = float(np.median(angles))
        return angle if 0.3 <= abs(angle) <= max_angle else 0.0

    # ── Full End-to-End Inference ─────────────────────────────────────────────

    def run(
        self,
        img_np: np.ndarray,
        cfg: Optional[OcrSettings] = None,
        *,
        preprocess: bool = True,
        keep_image: bool = False,
    ) -> OcrResult:
        """Run the complete configured pipeline on an RGB (or gray / RGBA) image."""
        cfg = cfg or get_ocr_settings()
        result = OcrResult()
        if img_np is None or img_np.size == 0:
            return result

        t0 = time.perf_counter()
        rgb = ensure_rgb(img_np)
        if preprocess:
            rgb, result.applied_steps = apply_preprocessing(rgb, cfg)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result.timings_ms["preprocess"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        boxes, det_size = self._detect(bgr, cfg)
        if cfg.auto_orient and len(boxes) >= 3 and self._mostly_vertical(boxes):
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            result.rotation = 90
            boxes, det_size = self._detect(bgr, cfg)
        if cfg.deskew and len(boxes) >= 3:
            angle = self._estimate_skew(boxes, cfg.deskew_max_angle)
            if angle:
                bgr = _rotate_bound(bgr, angle)
                result.skew_angle = round(angle, 2)
                boxes, det_size = self._detect(bgr, cfg)
        result.timings_ms["detection"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()

        def _crops() -> List[Optional[np.ndarray]]:
            h, w = bgr.shape[:2]
            return [get_rotate_crop_image(bgr, pad_box(box, cfg.rec_crop_padding, w, h)) for box, _ in boxes]

        crops = _crops()
        labels: Optional[List[Tuple[int, float]]] = None
        if (cfg.use_textline_orientation or cfg.auto_orient) and self.cls_session is not None:
            valid = [c for c in crops if c is not None]
            valid_labels = self.classify_orientation(valid)
            it = iter(valid_labels)
            labels = [next(it) if c is not None else (0, 1.0) for c in crops]
            if cfg.auto_orient and len(valid) >= 3:
                flipped = sum(1 for lab, score in valid_labels if lab == 1 and score >= 0.75)
                if flipped / len(valid) >= 0.6:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_180)
                    result.rotation = (result.rotation + 180) % 360
                    boxes, det_size = self._detect(bgr, cfg)
                    crops = _crops()
                    valid = [c for c in crops if c is not None]
                    it = iter(self.classify_orientation(valid))
                    labels = [next(it) if c is not None else (0, 1.0) for c in crops]
        # Lines the classifier calls upside down. It sees long lines squeezed to 192 px and
        # sometimes flags upright text with high confidence, so each flip is checked below.
        flagged: List[int] = []
        if cfg.use_textline_orientation and labels is not None:
            flagged = [i for i, crop in enumerate(crops)
                       if crop is not None and labels[i][0] == 1 and labels[i][1] > cfg.cls_thresh]
        result.timings_ms["orientation"] = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        recognised = self.recognize_batch(crops, cfg)
        if flagged:
            # Keep whichever orientation the recognizer reads more confidently.
            flipped_crops = [cv2.rotate(crops[i], cv2.ROTATE_180) for i in flagged]
            for i, crop, (text, score) in zip(flagged, flipped_crops, self.recognize_batch(flipped_crops, cfg)):
                if text and score > recognised[i][1]:
                    crops[i], recognised[i] = crop, (text, score)
        result.timings_ms["recognition"] = (time.perf_counter() - t3) * 1000

        items: List[OcrLine] = []
        for (box, _), crop, (text, score) in zip(boxes, crops, recognised):
            if crop is None or not text or score < cfg.rec_score_thresh:
                continue
            items.append(OcrLine(box=box.tolist(), text=text, score=score))

        if cfg.structured_output:
            try:  # lazy: ocr_layout imports this module
                from services.ocr_layout import build_layout
            except ImportError:
                from backend.services.ocr_layout import build_layout  # type: ignore[no-redef]
            layout = build_layout(items, cfg)
            result.lines, result.text, plain_text = layout.lines, layout.text, layout.plain_text
            result.table_count = layout.table_count
        else:
            result.lines = order_lines(items, cfg)
            raw_text = "\n".join(line.text for line in result.lines)
            result.text = _clean_ocr_text(raw_text) if cfg.text_cleanup else _normalise_text(raw_text)
            plain_text = result.text
        result.avg_conf = float(np.mean([it.score for it in items])) if items else 0.0
        result.word_count = len(re.findall(r"\S+", plain_text))
        result.char_count = sum(1 for ch in plain_text if ch.isalnum())
        result.box_count = len(boxes)
        result.image_size = (bgr.shape[1], bgr.shape[0])
        result.det_input_size = det_size
        result.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        if keep_image:
            result.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return result

    def __call__(
        self, img_np: np.ndarray, cfg: Optional[OcrSettings] = None
    ) -> Tuple[List[List[Union[List[List[float]], str, float]]], None]:
        """
        Run complete OCR pipeline on an input image (RGB).
        Returns:
          ([[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence], ...), None
        in reading order. Maintains output format compatible with downstream consumers.
        """
        res = self.run(img_np, cfg, preprocess=False)
        return [[line.box, line.text, line.score] for line in res.lines], None

    def close(self) -> None:
        """Release ONNX Runtime inference sessions."""
        self.det_session = None
        self.rec_session = None
        self.cls_session = None


# ── Singleton state ───────────────────────────────────────────────────────────
_OCR_ENGINE: Optional[PPOCRv6Engine] = None
_OCR_ENGINE_LOCK = threading.Lock()
_OCR_AVAILABLE: Optional[bool] = None

# Lease tracking: unload requests are deferred while inference is in flight.
_LEASE_LOCK = threading.RLock()
_LEASES: int = 0
_UNLOAD_PENDING: bool = False


# ── Engine initialisation ─────────────────────────────────────────────────────

def get_ocr_engine() -> Optional[PPOCRv6Engine]:
    """
    Return the singleton PP-OCRv6 Medium engine, initialising it on first call.

    Thread-safe. Returns None if model files are missing or initialization fails.
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
            logger.info("[OCREngine] Initialising PP-OCRv6 Medium OCR engine...")
            engine = PPOCRv6Engine()
            _OCR_ENGINE = engine
            _OCR_AVAILABLE = True
            logger.info("[OCREngine] PP-OCRv6 Medium engine initialised successfully.")
        except Exception as exc:
            _OCR_AVAILABLE = False
            logger.error(f"[OCREngine] Failed to initialise PP-OCRv6 engine: {exc}", exc_info=True)
            _OCR_ENGINE = None

    return _OCR_ENGINE


@contextmanager
def ocr_engine_lease() -> Iterator[Optional[PPOCRv6Engine]]:
    """
    Borrow the engine for one inference. While any lease is held, unload requests
    (after a document finishes, or when the embedder loads) are deferred instead of
    tearing down sessions that another thread is still using.
    """
    global _LEASES, _UNLOAD_PENDING
    with _LEASE_LOCK:
        engine = get_ocr_engine()
        if engine is not None:
            _LEASES += 1
    try:
        yield engine
    finally:
        if engine is not None:
            with _LEASE_LOCK:
                _LEASES -= 1
                if _LEASES == 0 and _UNLOAD_PENDING:
                    _UNLOAD_PENDING = False
                    _unload_now()


def _unload_now() -> None:
    """Tear down the engine. Caller must hold _LEASE_LOCK with no active leases."""
    global _OCR_ENGINE, _OCR_AVAILABLE
    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            logger.info("[OCREngine] Unloading PP-OCRv6 engine from memory...")
            try:
                _OCR_ENGINE.close()
            except Exception as exc:
                logger.warning(f"[OCREngine] Exception closing PP-OCRv6 sessions: {exc}")
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
            logger.info("[OCREngine] PP-OCRv6 engine successfully unloaded.")
        else:
            _OCR_AVAILABLE = None


def unload_ocr_engine() -> None:
    """
    Unload PP-OCRv6 engine from memory and run garbage collection.
    Releases ONNX Runtime sessions, PyTorch/CUDA memory, and Python references.
    If OCR is still running in another thread, the unload happens when it finishes.
    """
    global _UNLOAD_PENDING
    with _LEASE_LOCK:
        if _LEASES > 0:
            _UNLOAD_PENDING = True
            logger.info(f"[OCREngine] Unload deferred: {_LEASES} OCR task(s) still running.")
            return
        _unload_now()


def is_ocr_available() -> bool:
    """Return True if the PP-OCRv6 engine initialised successfully."""
    if _OCR_AVAILABLE is not None:
        return bool(_OCR_AVAILABLE)
    get_ocr_engine()
    return bool(_OCR_AVAILABLE)


def effective_ocr_workers(cfg: Optional[OcrSettings] = None) -> int:
    """Parallel OCR workers to use: the configured value, capped at 2 on CPU-only inference."""
    cfg = cfg or get_ocr_settings()
    engine = _OCR_ENGINE
    if engine is not None:
        on_gpu = engine.uses_gpu
    else:
        on_gpu = _ORT_AVAILABLE and "CUDAExecutionProvider" in ort.get_available_providers()
    return max(1, cfg.max_workers if on_gpu else min(cfg.max_workers, 2))


def get_ocr_status() -> Dict[str, object]:
    """Engine / model status for diagnostics. Never loads the engine."""
    model_dir = get_model_dir()
    files = {}
    for name in ("PP-OCRv6_medium_det.onnx", "PP-OCRv6_medium_rec.onnx",
                 "ch_ppocr_mobile_v2.0_cls.onnx", "ppocrv6_dict.txt"):
        path = model_dir / name
        files[name] = path.stat().st_size if path.is_file() else None
    engine = _OCR_ENGINE
    return {
        "model": "PP-OCRv6 Medium",
        "model_dir": str(model_dir),
        "files": files,
        "models_present": all(files[n] for n in ("PP-OCRv6_medium_det.onnx",
                                                  "PP-OCRv6_medium_rec.onnx", "ppocrv6_dict.txt")),
        "loaded": engine is not None,
        "active_provider": engine.provider if engine is not None else None,
        "available_providers": ort.get_available_providers() if _ORT_AVAILABLE else [],
        "gpu_available": _ORT_AVAILABLE and "CUDAExecutionProvider" in ort.get_available_providers(),
    }


# ── Image pre-processing ──────────────────────────────────────────────────────

def preprocess_for_ocr(
    img_np: np.ndarray, min_width: int = MIN_FRAME_WIDTH, cfg: Optional[OcrSettings] = None
) -> np.ndarray:
    """
    Prepare an image array for PP-OCRv6 inference.

    PP-OCRv6 uses continuous-tone images with sub-pixel font anti-aliasing, so by default
    nothing destructive is applied. Steps:
      1. Scale image if width < min_width preserving aspect ratio (legacy callers).
      2. Ensure 3-channel RGB.
      3. Apply the optional preprocessing steps enabled in the OCR settings.
    """
    if img_np is None or img_np.size == 0:
        return img_np

    h, w = img_np.shape[:2]
    if min_width and 0 < w < min_width:
        scale = min_width / float(w)
        new_w, new_h = min_width, max(1, int(h * scale))
        if _CV2_AVAILABLE:
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif _PIL_AVAILABLE:
            img_np = np.array(Image.fromarray(img_np).resize((new_w, new_h), Image.Resampling.LANCZOS))

    processed, _ = apply_preprocessing(ensure_rgb(img_np), cfg or get_ocr_settings())
    return processed


# ── Core OCR helpers ──────────────────────────────────────────────────────────

def _parse_ppocr_result(
    result: Optional[list],
    min_score: Optional[float] = None,
) -> Tuple[str, float, int, int]:
    """
    Parse raw PP-OCRv6 output into (text, avg_confidence, word_count, char_count).
    Result items: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], detected_text, confidence_score]
    """
    if not result:
        return "", 0.0, 0, 0
    if min_score is None:
        min_score = get_ocr_settings().rec_score_thresh

    lines: List[str] = []
    scores: List[float] = []

    for item in result:
        if not (isinstance(item, (list, tuple)) and len(item) >= 3):
            continue
        text = str(item[1]).strip()
        try:
            score = float(item[2])
        except (TypeError, ValueError):
            score = 0.0

        if text and score >= min_score:
            lines.append(text)
            scores.append(score)

    full_text = "\n".join(lines)
    avg_conf = float(np.mean(scores)) if scores else 0.0
    word_count = len(re.findall(r"\S+", full_text))
    char_count = sum(1 for ch in full_text if ch.isalnum())
    return full_text, avg_conf, word_count, char_count


# Runs of 3+ identical separator characters, optionally spaced: dot leaders ("....", ". . ."),
# rules ("___", "- - -") and borders read as text ("|||"). They stand between words or fields,
# so they become a tab rather than vanishing ("Agenda.....10:00" must not read "Agenda10:00").
SEPARATOR_RUN = re.compile(r"([|~_=\-^\\/@#$%*+.·•])(?: ?\1){2,}")


def squash_spaces(text: str) -> str:
    """Collapse runs of spaces; a tab (wide gap between fields) absorbs the spaces around it."""
    text = re.sub(r" *\t[ \t]*", "\t", text)
    return re.sub(r" {2,}", " ", text).strip()


def _normalise_text(raw: str) -> str:
    lines = [squash_spaces(ln) for ln in (raw or "").splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _clean_ocr_text(raw: str) -> str:
    """
    Post-process raw OCR output to remove noise.
    Removes short garbage lines, separator runs and consecutive duplicates.
    Letter/digit checks are Unicode-aware so non-Latin scripts are kept.
    """
    if not raw:
        return ""

    lines = raw.splitlines()
    cleaned: List[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        s = SEPARATOR_RUN.sub("\t", s).strip()
        if not s:
            continue
        alpha = sum(1 for ch in s if ch.isalnum())
        non_sp = len(re.sub(r"\s+", "", s))
        # Discard garbage lines (a single CJK character is a full word, so allow 1)
        min_alpha = 1 if any(ord(ch) > 0x2E7F for ch in s) else 2
        if alpha < min_alpha:
            continue
        if non_sp > 0 and alpha / non_sp < 0.35:
            continue
        cleaned.append(squash_spaces(s))

    # Deduplicate consecutive identical lines
    deduped: List[str] = []
    for line in cleaned:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    return "\n".join(deduped).strip()


def _run_ocr(
    img_np: np.ndarray,
    cfg: Optional[OcrSettings],
    *,
    preprocess: bool,
    label: str,
    keep_image: bool = False,
) -> Optional[OcrResult]:
    """Run the pipeline under an engine lease. Returns None if the engine is unavailable."""
    cfg = cfg or get_ocr_settings()
    with ocr_engine_lease() as engine:
        if engine is None:
            return None
        try:
            return engine.run(img_np, cfg, preprocess=preprocess, keep_image=keep_image)
        except Exception as exc:
            logger.warning(f"[OCREngine] {label}: inference error: {exc}")
            return OcrResult()


def _ocr_np(
    img_np: np.ndarray,
    preprocess: bool = True,
    label: str = "",
    dual_pass: bool = False,
    cfg: Optional[OcrSettings] = None,
) -> Tuple[str, float, int, int]:
    """
    Run PP-OCRv6 inference on a numpy array.
    """
    res = _run_ocr(img_np, cfg, preprocess=preprocess, label=label)
    if res is None:
        return "", 0.0, 0, 0
    return res.text, res.avg_conf, res.word_count, res.char_count


# ── Public API ────────────────────────────────────────────────────────────────

def ocr_image_bytes(
    image_bytes: bytes, *, label: str = "<bytes>", cfg: Optional[OcrSettings] = None
) -> str:
    """
    Run OCR on in-memory image bytes. Returns cleaned text or "" on failure.
    Images smaller than the configured minimum size (icons, bullets) are skipped.
    """
    if not image_bytes:
        return ""
    if not is_ocr_available():
        logger.warning(f"[OCREngine] OCR engine not available — skipping {label}")
        return ""
    cfg = cfg or get_ocr_settings()

    try:
        img_np = load_image_rgb(image_bytes)
    except Exception as img_err:
        logger.warning(f"[OCREngine] Could not open image bytes for {label}: {img_err}")
        return ""

    h, w = img_np.shape[:2]
    if w < cfg.min_image_side or h < cfg.min_image_side or (w * h) < cfg.min_image_area:
        logger.info(
            f"[OCREngine] Skipping small image {label} ({w}x{h}px < "
            f"{cfg.min_image_side}px / {cfg.min_image_area}px² threshold)"
        )
        return ""

    text, conf, wc, cc = _ocr_np(img_np, preprocess=True, label=label, cfg=cfg)
    if text:
        logger.info(f"[OCREngine] {label} ({w}x{h}px): OCR complete -> conf={conf:.2f}, words={wc}, chars={cc}")
    else:
        logger.info(f"[OCREngine] {label} ({w}x{h}px): OCR finished -> no text detected")
    return text


def ocr_image_file(path: str, cfg: Optional[OcrSettings] = None) -> str:
    """
    Run OCR on a standalone image file. Returns cleaned text or "".
    Used by doc_extractor.py for PNG/JPG/JPEG/WEBP files.
    """
    if not os.path.isfile(path):
        logger.warning(f"[OCREngine] Image file not found: {path}")
        return ""
    if not is_ocr_available():
        logger.warning(f"[OCREngine] OCR engine not available — skipping {os.path.basename(path)}")
        return ""

    try:
        img_np = load_image_rgb(path)
        text, conf, wc, cc = _ocr_np(img_np, preprocess=True, label=os.path.basename(path), cfg=cfg)
        logger.debug(f"[OCREngine] {os.path.basename(path)}: conf={conf:.2f}, words={wc}, chars={cc}")
        return text
    except Exception as exc:
        logger.warning(f"[OCREngine] OCR failed for {os.path.basename(path)}: {exc}")
        return ""


def ocr_image_ndarray(
    img_np: np.ndarray,
    label: str = "frame",
    preprocess: bool = True,
    cfg: Optional[OcrSettings] = None,
) -> Tuple[str, float, int, int]:
    """
    Run OCR on a uint8 numpy array (H × W × 3, RGB channel order).
    Returns (clean_text, avg_confidence, word_count, char_count).
    Used by video_processing_service.py for per-frame OCR and for scanned PDF pages.
    """
    if not is_ocr_available():
        return "", 0.0, 0, 0
    return _ocr_np(img_np, preprocess=preprocess, label=label, cfg=cfg)


def ocr_image_detailed(
    img_np: np.ndarray,
    cfg: Optional[OcrSettings] = None,
    *,
    preprocess: bool = True,
    keep_image: bool = False,
    label: str = "image",
) -> Optional[OcrResult]:
    """Run OCR and return the full structured result (lines, boxes, timings). None if unavailable."""
    if not is_ocr_available():
        return None
    return _run_ocr(img_np, cfg, preprocess=preprocess, label=label, keep_image=keep_image)
