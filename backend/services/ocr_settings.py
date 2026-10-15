"""
ocr_settings.py — System-wide OCR configuration for the PP-OCRv6 pipeline.

One saved configuration drives every OCR task: document extraction (PDF / DOCX /
PPTX / images), scanned-PDF page rendering, embedded-image OCR and video-frame OCR.

Storage:
  The configuration is a JSON blob in the single-row `ocr_settings` table
  (created in database.connect_db). New fields therefore never need a migration;
  stored JSON is merged onto the defaults and invalid stored values fall back to
  their defaults instead of breaking OCR.

Runtime access:
  OCR runs inside worker threads, so `get_ocr_settings()` is synchronous and
  cache-first with a sqlite3 fallback (same approach as prompt_service). A
  ContextVar override (`use_ocr_settings`) lets a caller pin a snapshot for one
  document, or test unsaved settings, without touching the saved configuration.

Defaults follow the official PaddleX PP-OCRv6 pipeline configuration
(paddlex/configs/pipelines/OCR.yaml, PaddleOCR 3.7) unless a field's help text
says otherwise.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _field(
    default: Any,
    *,
    group: str,
    label: str,
    help: str,
    ge: Optional[float] = None,
    le: Optional[float] = None,
    step: Optional[float] = None,
    unit: Optional[str] = None,
    options: Optional[List[Dict[str, str]]] = None,
    depends_on: Optional[Dict[str, Any]] = None,
    warning: Optional[str] = None,
) -> Any:
    """Declare a settings field together with the UI metadata served to the frontend."""
    extra: Dict[str, Any] = {"group": group, "label": label, "help": help}
    for key, value in (
        ("step", step), ("unit", unit), ("options", options),
        ("depends_on", depends_on), ("warning", warning),
    ):
        if value is not None:
            extra[key] = value
    return Field(default, ge=ge, le=le, json_schema_extra=extra)


# ── Settings model ────────────────────────────────────────────────────────────

class OcrSettings(BaseModel):
    """Validated OCR configuration. Unknown keys are ignored."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = SCHEMA_VERSION

    # ── Image preprocessing (all optional; official PP-OCRv6 feeds unmodified images)
    upscale_min_side: int = _field(
        0, group="preprocessing", label="Upscale small images",
        help="Upscale images whose shorter side is below this size (cubic, at most 4x). "
             "Helps tiny text in low-resolution screenshots. 0 disables.",
        ge=0, le=4000, step=50, unit="px",
    )
    grayscale: bool = _field(
        False, group="preprocessing", label="Convert to grayscale",
        help="Remove colour before OCR. Can help on coloured or patterned backgrounds; "
             "PP-OCRv6 is trained on colour images, so leave off unless needed.",
    )
    denoise: Literal["none", "median", "bilateral", "nlmeans"] = _field(
        "none", group="preprocessing", label="Noise reduction",
        help="Reduce scanner or compression noise before detection.",
        options=[
            {"value": "none", "label": "Off"},
            {"value": "median", "label": "Median (speckle / salt-and-pepper)"},
            {"value": "bilateral", "label": "Bilateral (edge-preserving)"},
            {"value": "nlmeans", "label": "Non-local means (strongest, slowest)"},
        ],
    )
    denoise_strength: int = _field(
        7, group="preprocessing", label="Noise reduction strength",
        help="Higher values remove more noise but can soften thin strokes.",
        ge=1, le=30, step=1,
        depends_on={"field": "denoise", "values": ["median", "bilateral", "nlmeans"]},
    )
    clahe: bool = _field(
        False, group="preprocessing", label="Contrast enhancement (CLAHE)",
        help="Adaptive histogram equalisation on the lightness channel. Useful for faded "
             "or low-contrast scans and unevenly lit photos.",
    )
    clahe_clip_limit: float = _field(
        2.0, group="preprocessing", label="CLAHE clip limit",
        help="Contrast amplification limit. Higher values boost contrast more aggressively.",
        ge=1.0, le=8.0, step=0.1, depends_on={"field": "clahe", "values": [True]},
    )
    clahe_tile_grid: int = _field(
        8, group="preprocessing", label="CLAHE tile grid",
        help="Number of tiles per side used for local equalisation.",
        ge=2, le=16, step=1, depends_on={"field": "clahe", "values": [True]},
    )
    sharpen: bool = _field(
        False, group="preprocessing", label="Sharpen",
        help="Unsharp mask to crisp up slightly blurred text (e.g. video frames, phone photos).",
    )
    sharpen_amount: float = _field(
        1.0, group="preprocessing", label="Sharpen amount",
        help="Strength of the unsharp mask.",
        ge=0.1, le=3.0, step=0.1, depends_on={"field": "sharpen", "values": [True]},
    )
    binarize: Literal["none", "otsu", "adaptive"] = _field(
        "none", group="preprocessing", label="Binarization",
        help="Convert to pure black and white.",
        options=[
            {"value": "none", "label": "Off (recommended)"},
            {"value": "otsu", "label": "Otsu (global threshold)"},
            {"value": "adaptive", "label": "Adaptive (local threshold)"},
        ],
        warning="Not recommended for PP-OCRv6: the models are trained on continuous-tone "
                "images and binarization removes anti-aliased stroke detail.",
    )
    binarize_block_size: int = _field(
        31, group="preprocessing", label="Adaptive block size",
        help="Neighbourhood size for adaptive thresholding (odd number).",
        ge=3, le=99, step=2, unit="px", depends_on={"field": "binarize", "values": ["adaptive"]},
    )
    binarize_c: int = _field(
        10, group="preprocessing", label="Adaptive offset (C)",
        help="Constant subtracted from the local mean. Higher values make more pixels white.",
        ge=0, le=30, step=1, depends_on={"field": "binarize", "values": ["adaptive"]},
    )
    deskew: bool = _field(
        False, group="preprocessing", label="Deskew pages",
        help="Straighten slightly rotated scans using the angle of detected text lines, "
             "then detect again. Improves line grouping and reading order on skewed scans.",
    )
    deskew_max_angle: float = _field(
        10.0, group="preprocessing", label="Maximum deskew angle",
        help="Skew angles larger than this are left untouched.",
        ge=0.5, le=20.0, step=0.5, unit="°", depends_on={"field": "deskew", "values": [True]},
    )
    auto_orient: bool = _field(
        True, group="preprocessing", label="Automatic page orientation",
        help="Detect pages rotated by 90°, 180° or 270° from the detected text lines and "
             "rotate them upright. Model-free stand-in for PP-OCRv6's document orientation "
             "module; only acts when most text lines agree.",
    )

    # ── Text detection (DBNet)
    det_limit_type: Literal["min", "max"] = _field(
        "min", group="detection", label="Resize mode",
        help="How the image is resized before detection.",
        options=[
            {"value": "min", "label": "Min side (official: keep native resolution)"},
            {"value": "max", "label": "Max side (downscale large images, faster)"},
        ],
    )
    det_limit_side_len: int = _field(
        64, group="detection", label="Side length limit",
        help="Min mode: the shorter side is upscaled to at least this. "
             "Max mode: the longer side is downscaled to at most this.",
        ge=32, le=4000, step=32, unit="px",
    )
    det_max_side_limit: int = _field(
        4000, group="detection", label="Maximum side length",
        help="Hard cap on the longer side fed to the detector, protecting memory on very large images.",
        ge=640, le=8000, step=32, unit="px",
    )
    det_db_thresh: float = _field(
        0.3, group="detection", label="Pixel threshold",
        help="Probability above which a pixel counts as text. Lower finds fainter text but more noise.",
        ge=0.05, le=0.9, step=0.01,
    )
    det_db_box_thresh: float = _field(
        0.6, group="detection", label="Box threshold",
        help="Minimum average score for a detected region to be kept.",
        ge=0.05, le=0.95, step=0.01,
    )
    det_db_unclip_ratio: float = _field(
        1.5, group="detection", label="Unclip ratio",
        help="How much detected regions are expanded. Raise if characters at line edges are cut off.",
        ge=1.0, le=4.0, step=0.05,
    )
    det_use_dilation: bool = _field(
        False, group="detection", label="Dilate text mask",
        help="Thicken the text mask before extracting boxes. Can help very thin fonts.",
    )
    det_score_mode: Literal["fast", "slow"] = _field(
        "fast", group="detection", label="Box scoring",
        help="Fast scores each box over its rectangle; slow scores the exact contour (more precise "
             "for curved or slanted text).",
        options=[
            {"value": "fast", "label": "Fast (official)"},
            {"value": "slow", "label": "Slow (contour polygon)"},
        ],
    )
    det_max_candidates: int = _field(
        3000, group="detection", label="Maximum text regions",
        help="Upper bound on regions processed per image (model config default 3000).",
        ge=100, le=10000, step=100,
    )

    # ── Text-line orientation
    use_textline_orientation: bool = _field(
        True, group="orientation", label="Text-line orientation correction",
        help="Classify every text line as upright or upside down and flip it before recognition.",
    )
    cls_thresh: float = _field(
        0.9, group="orientation", label="Flip confidence threshold",
        help="A line is flipped only when the classifier is at least this confident.",
        ge=0.5, le=1.0, step=0.01, depends_on={"field": "use_textline_orientation", "values": [True]},
    )

    # ── Text recognition
    rec_score_thresh: float = _field(
        0.5, group="recognition", label="Minimum line confidence",
        help="Lines recognised with lower confidence are dropped. Official pipeline keeps "
             "everything (0.0); 0.5 matches PaddleOCR's drop_score and keeps noise out of "
             "summaries and search.",
        ge=0.0, le=1.0, step=0.01,
    )
    rec_max_width: int = _field(
        3200, group="recognition", label="Maximum line width",
        help="Longest normalised text-line width (at 48 px height). Wider lines are squeezed.",
        ge=320, le=3200, step=32, unit="px",
    )
    rec_batch_size: int = _field(
        6, group="recognition", label="Recognition batch size",
        help="Text lines recognised per model call.",
        ge=1, le=8, step=1,
    )
    rec_crop_padding: float = _field(
        0.1, group="recognition", label="Text-line crop margin",
        help="Margin added around each detected text line before recognition, as a fraction of "
             "the line height. Detected boxes hug the glyphs; a small margin roughly halved "
             "merged and split words in testing. 0 matches the official pipeline.",
        ge=0.0, le=0.5, step=0.05, unit="× line height",
    )

    # ── Layout & output
    merge_lines: bool = _field(
        True, group="layout", label="Merge words into lines",
        help="Join text regions that sit on the same visual line into one output line.",
    )
    line_merge_max_gap: float = _field(
        2.0, group="layout", label="Maximum gap within a line",
        help="Regions further apart than this (in line heights) stay separate, so adjacent "
             "columns are not fused.",
        ge=0.5, le=10.0, step=0.1, unit="× line height",
        depends_on={"field": "merge_lines", "values": [True]},
    )
    column_detection: bool = _field(
        False, group="layout", label="Multi-column reading order",
        help="Detect columns (XY-cut) and read each column top to bottom before moving right.",
    )
    structured_output: bool = _field(
        True, group="layout", label="Preserve tables and layout",
        help="Rebuild tables from the positions of the detected text (Markdown rows and columns), "
             "keep wide gaps inside a line as a tab and separate paragraphs with a blank line. "
             "Off returns plain text lines.",
    )
    text_cleanup: bool = _field(
        True, group="layout", label="Remove noise lines",
        help="Drop lines that are mostly symbols and collapse repeated punctuation and duplicates.",
    )
    min_image_side: int = _field(
        100, group="layout", label="Skip images narrower than",
        help="Embedded images smaller than this on either side are not OCR'd (icons, bullets).",
        ge=0, le=2000, step=10, unit="px",
    )
    min_image_area: int = _field(
        10000, group="layout", label="Skip images smaller than",
        help="Embedded images with fewer pixels than this are not OCR'd.",
        ge=0, le=1_000_000, step=1000, unit="px²",
    )

    # ── Scanned PDFs
    pdf_ocr_mode: Literal["auto", "always", "never"] = _field(
        "auto", group="pdf", label="Page OCR mode",
        help="When PDF pages are rendered to images and OCR'd.",
        options=[
            {"value": "auto", "label": "Auto (pages without a usable text layer)"},
            {"value": "always", "label": "Always (ignore the text layer)"},
            {"value": "never", "label": "Never (text layer and embedded images only)"},
        ],
    )
    pdf_render_dpi: int = _field(
        200, group="pdf", label="Render resolution",
        help="DPI used to render scanned pages. 200 suits most documents; 300 for small print.",
        ge=72, le=600, step=10, unit="dpi",
    )
    pdf_scan_text_threshold: int = _field(
        50, group="pdf", label="Scanned page text threshold",
        help="Pages with fewer selectable characters than this are treated as scans.",
        ge=0, le=5000, step=10, unit="chars",
        depends_on={"field": "pdf_ocr_mode", "values": ["auto"]},
    )
    pdf_scan_image_coverage: float = _field(
        0.85, group="pdf", label="Scanned page image coverage",
        help="Pages mostly covered by images (this fraction or more) with little text are "
             "also treated as scans, e.g. scans with a stamped header.",
        ge=0.1, le=1.0, step=0.05, depends_on={"field": "pdf_ocr_mode", "values": ["auto"]},
    )

    # ── Performance
    max_workers: int = _field(
        4, group="performance", label="Parallel OCR workers",
        help="Images / pages processed concurrently. Detection is serialised to protect GPU "
             "memory; on CPU this is capped at 2.",
        ge=1, le=8, step=1,
    )

    @field_validator("binarize_block_size")
    @classmethod
    def _odd_block_size(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("must be an odd number")
        return value

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "OcrSettings":
        if self.det_limit_type == "max" and self.det_limit_side_len > self.det_max_side_limit:
            raise ValueError(
                "Side length limit cannot exceed the maximum side length in max resize mode"
            )
        return self


class OcrSettingsIn(OcrSettings):
    """Request body for saving settings: unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")


# ── Groups, presets & schema served to the UI ─────────────────────────────────

GROUPS: List[Dict[str, str]] = [
    {"id": "preprocessing", "title": "Image preprocessing",
     "description": "Optional image clean-up applied before detection. PP-OCRv6 is trained on "
                    "unmodified images, so these are off by default; enable them for difficult inputs."},
    {"id": "detection", "title": "Text detection",
     "description": "PP-OCRv6 DBNet detector resize and post-processing parameters."},
    {"id": "orientation", "title": "Text-line orientation",
     "description": "Per-line upside-down correction using the orientation classifier."},
    {"id": "recognition", "title": "Text recognition",
     "description": "PP-OCRv6 recogniser input size, batching and confidence filtering."},
    {"id": "layout", "title": "Layout & output",
     "description": "Reading order, line assembly and filtering of the final text."},
    {"id": "pdf", "title": "Scanned PDFs",
     "description": "How PDF pages without a usable text layer are rendered and OCR'd page by page."},
    {"id": "performance", "title": "Performance",
     "description": "Concurrency for multi-image documents and multi-page scans."},
]

_CROSS_FIELD_KEYS = ("det_limit_side_len", "det_max_side_limit", "det_limit_type")

PRESETS: Dict[str, Dict[str, Any]] = {
    "official": {
        "label": "PP-OCRv6 Official",
        "description": "Official PaddleX PP-OCRv6 pipeline values (recommended).",
        "overrides": {},
    },
    "fast": {
        "label": "Fast",
        "description": "Downscale to 960 px for detection. Best for CPU-only machines and video frames.",
        "overrides": {
            "det_limit_type": "max", "det_limit_side_len": 960,
            "auto_orient": False, "pdf_render_dpi": 150,
        },
    },
    "scanned": {
        "label": "Scanned documents",
        "description": "300 dpi page rendering, deskew and multi-column reading order.",
        "overrides": {
            "pdf_render_dpi": 300, "deskew": True, "column_detection": True, "auto_orient": True,
        },
    },
    "legacy": {
        "label": "Legacy",
        "description": "Parameters used before this update (960 px detection, looser thresholds).",
        "overrides": {
            "det_limit_type": "max", "det_limit_side_len": 960,
            "det_db_thresh": 0.2, "det_db_box_thresh": 0.45, "det_db_unclip_ratio": 1.4,
            "det_score_mode": "slow", "cls_thresh": 0.6, "rec_score_thresh": 0.4,
            "rec_max_width": 960, "merge_lines": False, "auto_orient": False,
            "pdf_render_dpi": 150, "pdf_scan_text_threshold": 20, "max_workers": 5,
            "rec_crop_padding": 0.0, "structured_output": False,
        },
    },
}


def preset_settings(name: str) -> OcrSettings:
    """Return the full settings for a named preset."""
    preset = PRESETS[name]
    return OcrSettings(**{**OcrSettings().model_dump(), **preset["overrides"]})


def presets_payload() -> List[Dict[str, Any]]:
    """Presets as full settings objects for the UI."""
    return [
        {"id": key, "label": p["label"], "description": p["description"],
         "settings": preset_settings(key).model_dump()}
        for key, p in PRESETS.items()
    ]


def settings_schema() -> Dict[str, Any]:
    """Field metadata (type, bounds, labels, grouping) so the UI never duplicates ranges."""
    defaults = OcrSettings()
    fields: List[Dict[str, Any]] = []
    for name, info in OcrSettings.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        if not extra.get("group"):
            continue
        entry: Dict[str, Any] = {"name": name, "default": getattr(defaults, name), **extra}
        annotation = info.annotation
        if annotation is bool:
            entry["type"] = "bool"
        elif annotation is int:
            entry["type"] = "int"
        elif annotation is float:
            entry["type"] = "float"
        elif get_origin(annotation) is Literal:
            entry["type"] = "enum"
            entry.setdefault("options", [{"value": v, "label": str(v)} for v in get_args(annotation)])
        for meta in info.metadata:
            if getattr(meta, "ge", None) is not None:
                entry["min"] = meta.ge
            if getattr(meta, "le", None) is not None:
                entry["max"] = meta.le
        fields.append(entry)
    return {"version": SCHEMA_VERSION, "groups": GROUPS, "fields": fields}


# ── Lenient loading ───────────────────────────────────────────────────────────

def load_lenient(raw: Any) -> OcrSettings:
    """
    Build settings from stored JSON, falling back to defaults for any field that is
    missing or invalid. Never raises: OCR must keep working with a damaged row.
    """
    if not isinstance(raw, dict):
        return OcrSettings()
    data = {k: v for k, v in raw.items() if k in OcrSettings.model_fields}
    data["schema_version"] = SCHEMA_VERSION

    for _ in range(len(data) + 2):
        try:
            return OcrSettings(**data)
        except ValidationError as exc:
            bad = {err["loc"][0] for err in exc.errors() if err.get("loc")}
            if not bad:
                bad = {k for k in _CROSS_FIELD_KEYS if k in data}
            if not bad:
                break
            logger.warning(f"[OCRSettings] Ignoring invalid stored values for: {sorted(bad)}")
            for key in bad:
                data.pop(key, None)
    return OcrSettings()


# ── Cache & persistence ───────────────────────────────────────────────────────

_cache: Optional[OcrSettings] = None
_cache_updated_at: Optional[str] = None
_cache_lock = threading.Lock()
_override: ContextVar[Optional[OcrSettings]] = ContextVar("ocr_settings_override", default=None)


def _sqlite_path() -> Optional[str]:
    """Resolve the SQLite file path from settings.DATABASE_URL."""
    try:
        from config import settings
        db_url = getattr(settings, "DATABASE_URL", "") or ""
    except Exception:
        return None
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if db_url.startswith(prefix):
            return db_url[len(prefix):]
    return db_url or None


def _load_sync() -> Tuple[OcrSettings, Optional[str], bool]:
    """Read the saved row with sqlite3. Returns (settings, updated_at, table_exists)."""
    import sqlite3

    db_path = _sqlite_path()
    if not db_path or not os.path.exists(db_path):
        return OcrSettings(), None, False
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ocr_settings'")
            if not cur.fetchone():
                return OcrSettings(), None, False
            cur.execute("SELECT config_json, updated_at FROM ocr_settings WHERE id = 1")
            row = cur.fetchone()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning(f"[OCRSettings] Sync load failed, using defaults: {exc}")
        return OcrSettings(), None, False

    if not row:
        return OcrSettings(), None, True
    try:
        raw = json.loads(row[0] or "{}")
    except Exception:
        logger.warning("[OCRSettings] Stored configuration is not valid JSON — using defaults.")
        raw = {}
    return load_lenient(raw), row[1], True


def _set_cache(cfg: OcrSettings, updated_at: Optional[str]) -> None:
    global _cache, _cache_updated_at
    with _cache_lock:
        _cache = cfg
        _cache_updated_at = updated_at


def get_ocr_settings() -> OcrSettings:
    """
    Return the active OCR settings: a ContextVar override if one is set for the
    current call chain, otherwise the saved configuration (cached).
    """
    override = _override.get()
    if override is not None:
        return override

    cached = _cache
    if cached is not None:
        return cached

    cfg, updated_at, table_exists = _load_sync()
    if table_exists:
        _set_cache(cfg, updated_at)
    return cfg


def get_settings_updated_at() -> Optional[str]:
    return _cache_updated_at


def invalidate_cache() -> None:
    global _cache, _cache_updated_at
    with _cache_lock:
        _cache = None
        _cache_updated_at = None


@contextmanager
def use_ocr_settings(cfg: Optional[OcrSettings]) -> Iterator[OcrSettings]:
    """
    Pin `cfg` as the active settings for the current thread / task context.
    Passing None pins a snapshot of the currently active settings.

    ThreadPoolExecutor workers do not inherit ContextVars, so code that fans out
    to worker threads must pass the settings object explicitly.
    """
    snapshot = cfg if cfg is not None else get_ocr_settings()
    token = _override.set(snapshot)
    try:
        yield snapshot
    finally:
        _override.reset(token)


async def load_ocr_settings(db) -> Tuple[OcrSettings, Optional[str]]:
    """Load the saved configuration through the async session and refresh the cache."""
    from sqlalchemy import text

    result = await db.execute(text("SELECT config_json, updated_at FROM ocr_settings WHERE id = 1"))
    row = result.mappings().fetchone()
    if row:
        try:
            raw = json.loads(row["config_json"] or "{}")
        except Exception:
            raw = {}
        cfg, updated_at = load_lenient(raw), row["updated_at"]
    else:
        cfg, updated_at = OcrSettings(), None
    _set_cache(cfg, updated_at)
    return cfg, updated_at


async def save_ocr_settings(db, cfg: OcrSettings, user_id: Optional[str] = None) -> str:
    """Persist the configuration (single row) and refresh the cache."""
    from sqlalchemy import text

    payload = OcrSettings(**cfg.model_dump())  # normalise (drops forbid-model subclass)
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        text("""
            INSERT INTO ocr_settings (id, config_json, updated_at, updated_by)
            VALUES (1, :config_json, :now, :user_id)
            ON CONFLICT(id) DO UPDATE SET
                config_json = excluded.config_json,
                updated_at  = excluded.updated_at,
                updated_by  = excluded.updated_by
        """),
        {"config_json": payload.model_dump_json(), "now": now, "user_id": user_id},
    )
    await db.commit()
    _set_cache(payload, now)
    logger.info(f"[OCRSettings] Saved OCR settings ({settings_fingerprint(payload)})")
    return now


async def reset_ocr_settings(db) -> OcrSettings:
    """Delete the saved configuration; every OCR task falls back to the defaults."""
    from sqlalchemy import text

    await db.execute(text("DELETE FROM ocr_settings WHERE id = 1"))
    await db.commit()
    defaults = OcrSettings()
    _set_cache(defaults, None)
    logger.info("[OCRSettings] OCR settings reset to defaults")
    return defaults


def settings_fingerprint(cfg: OcrSettings) -> str:
    """Short, log-friendly identifier of a configuration plus its key values."""
    digest = hashlib.sha1(cfg.model_dump_json().encode("utf-8")).hexdigest()[:8]
    return (
        f"cfg={digest} det={cfg.det_limit_type}/{cfg.det_limit_side_len}/{cfg.det_max_side_limit} "
        f"db={cfg.det_db_thresh:.2f}/{cfg.det_db_box_thresh:.2f}/{cfg.det_db_unclip_ratio:.2f} "
        f"rec>={cfg.rec_score_thresh:.2f} pdf={cfg.pdf_ocr_mode}@{cfg.pdf_render_dpi}dpi"
    )
