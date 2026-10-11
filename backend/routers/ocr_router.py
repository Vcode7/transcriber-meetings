"""
ocr_router.py — OCR Settings page API.

  GET  /ocr/settings          saved settings + defaults + presets + field schema
  PUT  /ocr/settings          validate & save (system-wide; applies to every OCR task)
  POST /ocr/settings/reset    restore defaults
  GET  /ocr/status            model files / engine state (never loads the engine)
  POST /ocr/test              run the production OCR path on an uploaded document,
                              optionally with unsaved settings

  Experimental (testing only, isolated from the production OCR pipeline):
  GET  /ocr/vlm/models        Ollama models with vision capability flags
  POST /ocr/vlm/test          Chandra OCR 2 via Ollama → HTML
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError
from sqlalchemy import text

from database import get_db_context
from routers.auth import get_current_user
from services.ocr_settings import (
    OcrSettings,
    OcrSettingsIn,
    load_ocr_settings,
    presets_payload,
    reset_ocr_settings,
    save_ocr_settings,
    settings_fingerprint,
    settings_schema,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["ocr"])

MAX_UPLOAD_MB = 25
MAX_TEST_PAGES = 10
PREVIEW_MAX_SIDE = 1280
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TEST_EXTS = IMAGE_EXTS | {".pdf", ".docx", ".pptx", ".doc", ".ppt"}
VLM_EXTS = IMAGE_EXTS | {".pdf"}

# One OCR test at a time so tests never compete with each other for the GPU.
_test_semaphore = asyncio.Semaphore(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _magic_ok(ext: str, data: bytes) -> bool:
    """Check the file signature matches its extension."""
    head = data[:16]
    if ext == ".pdf":
        return b"%PDF" in data[:1024]
    if ext == ".png":
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    if ext in (".jpg", ".jpeg"):
        return head.startswith(b"\xff\xd8\xff")
    if ext == ".webp":
        return head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    if ext in (".docx", ".pptx"):
        return head.startswith(b"PK\x03\x04")
    if ext in (".doc", ".ppt"):
        return head.startswith(b"\xd0\xcf\x11\xe0")
    return False


async def _read_upload(file: UploadFile, allowed: set) -> tuple[str, str, bytes]:
    filename = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(filename.lower())[1]
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or 'unknown'}'. Allowed: {', '.join(sorted(allowed))}",
        )
    data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB.")
    if not _magic_ok(ext, data):
        raise HTTPException(status_code=400, detail=f"The file content does not look like a valid {ext} file.")
    return filename, ext, data


def _parse_unsaved_settings(raw: Optional[str]) -> Optional[OcrSettings]:
    if not raw:
        return None
    try:
        return OcrSettingsIn.model_validate_json(raw)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False)
        for err in errors:
            err["loc"] = ["settings", *err.get("loc", ())]
        raise HTTPException(status_code=422, detail=errors)


def _preview_data_url(image, lines) -> Optional[str]:
    """Downscaled JPEG of the processed image with detected text lines outlined."""
    if image is None:
        return None
    import cv2
    import numpy as np

    canvas = image.copy()
    h, w = canvas.shape[:2]
    thickness = max(1, int(round(max(h, w) / 700)))
    for line in lines:
        pts = np.array(line.box, dtype=np.int32).reshape(-1, 1, 2)
        color = (22, 163, 74) if line.score >= 0.8 else (234, 140, 0)
        cv2.polylines(canvas, [pts], True, color, thickness)
    scale = min(1.0, PREVIEW_MAX_SIDE / float(max(h, w)))
    if scale < 1.0:
        canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _text_stats(value: str) -> Dict[str, int]:
    return {
        "char_count": len(value or ""),
        "word_count": len(re.findall(r"\S+", value or "")),
    }


def _page_payload(page: int, source: str, page_text: str, result=None, **extra) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"page": page, "source": source, "text": page_text, **_text_stats(page_text)}
    if result is not None:
        payload.update({
            "avg_confidence": round(result.avg_conf, 4),
            "line_count": len(result.lines),
            "box_count": result.box_count,
            "rotation": result.rotation,
            "skew_angle": result.skew_angle,
            "applied_steps": result.applied_steps,
            "image_size": list(result.image_size),
            "det_input_size": list(result.det_input_size),
            "timings_ms": {k: round(v) for k, v in result.timings_ms.items()},
            "lines": [{"text": ln.text, "score": round(ln.score, 4)} for ln in result.lines[:400]],
            "preview": _preview_data_url(result.image, result.lines),
        })
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def _run_test(data: bytes, filename: str, ext: str, cfg: OcrSettings, max_pages: int, show_boxes: bool) -> Dict:
    """Blocking OCR test run (executed in a worker thread)."""
    from services.doc_extractor import extract_text_from_bytes, ocr_pdf_pages_detailed
    from services.ocr_engine import get_ocr_status, is_ocr_available, ocr_image_detailed
    from services.ocr_preprocessing import load_image_rgb
    from services.ocr_settings import use_ocr_settings

    if not is_ocr_available():
        raise HTTPException(
            status_code=503,
            detail="The OCR engine is not available. Check that the PP-OCRv6 model files are installed.",
        )

    started = time.perf_counter()
    pages: List[Dict[str, Any]] = []
    with use_ocr_settings(cfg):
        if ext in IMAGE_EXTS:
            kind, page_count = "image", 1
            try:
                img = load_image_rgb(data)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Could not decode the image: {exc}")
            t0 = time.perf_counter()
            result = ocr_image_detailed(img, cfg, keep_image=show_boxes, label=filename)
            if result is None:
                raise HTTPException(status_code=503, detail="The OCR engine is not available.")
            full_text = result.text
            pages.append(_page_payload(1, "ocr", result.text, result,
                                       elapsed_ms=round((time.perf_counter() - t0) * 1000)))
        elif ext == ".pdf":
            kind = "pdf"
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                full_text, details, page_count = ocr_pdf_pages_detailed(
                    tmp_path, cfg, max_pages=max_pages, keep_images=show_boxes
                )
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Could not read the PDF: {exc}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            for page_idx in sorted(details):
                d = details[page_idx]
                pages.append(_page_payload(
                    page_idx, d["source"], d["text"], d.get("ocr_result"),
                    elapsed_ms=round(d["elapsed_ms"]) if d.get("elapsed_ms") is not None else None,
                    render_dpi=d.get("render_dpi"),
                    text_layer_chars=d.get("text_layer_chars"),
                    embedded_images=d.get("embedded_images"),
                    embedded_images_with_text=d.get("embedded_images_with_text"),
                ))
        else:
            kind, page_count = "document", 1
            full_text = extract_text_from_bytes(data, filename, ocr_settings=cfg)
            pages.append(_page_payload(1, "document", full_text))

    ocr_pages = [p for p in pages if p.get("avg_confidence") is not None and p.get("line_count")]
    status = get_ocr_status()
    return {
        "filename": filename,
        "kind": kind,
        "page_count": page_count,
        "pages_processed": len(pages),
        "pages": pages,
        "text": full_text,
        **_text_stats(full_text),
        "avg_confidence": (
            round(sum(p["avg_confidence"] for p in ocr_pages) / len(ocr_pages), 4) if ocr_pages else None
        ),
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "settings_fingerprint": settings_fingerprint(cfg),
        "provider": status.get("active_provider"),
    }


async def _user_ollama_url(user_id: str) -> str:
    from routers.arena_router import normalize_ollama_url

    async with get_db_context() as db:
        row = (await db.execute(
            text("SELECT ollama_server_url FROM user_settings WHERE user_id = :uid"), {"uid": user_id}
        )).mappings().fetchone()
    url = row["ollama_server_url"] if row and row.get("ollama_server_url") else "http://localhost:11434"
    return normalize_ollama_url(url)


# ── Settings ──────────────────────────────────────────────────────────────────

@router.get("/settings")
async def get_settings(current_user: dict = Depends(get_current_user)):
    async with get_db_context() as db:
        cfg, updated_at = await load_ocr_settings(db)
    return {
        "settings": cfg.model_dump(),
        "defaults": OcrSettings().model_dump(),
        "presets": presets_payload(),
        "schema": settings_schema(),
        "updated_at": updated_at,
    }


@router.put("/settings")
async def put_settings(body: OcrSettingsIn, current_user: dict = Depends(get_current_user)):
    async with get_db_context() as db:
        updated_at = await save_ocr_settings(db, body, user_id=current_user.get("id"))
    saved = OcrSettings(**body.model_dump())
    return {"settings": saved.model_dump(), "updated_at": updated_at,
            "fingerprint": settings_fingerprint(saved)}


@router.post("/settings/reset")
async def reset_settings(current_user: dict = Depends(get_current_user)):
    async with get_db_context() as db:
        defaults = await reset_ocr_settings(db)
    return {"settings": defaults.model_dump(), "updated_at": None,
            "fingerprint": settings_fingerprint(defaults)}


@router.get("/status")
async def ocr_status(current_user: dict = Depends(get_current_user)):
    from services.ocr_engine import get_ocr_status
    from services.ocr_settings import get_ocr_settings, get_settings_updated_at

    status = await asyncio.to_thread(get_ocr_status)
    cfg = get_ocr_settings()
    return {**status, "settings_fingerprint": settings_fingerprint(cfg),
            "settings_updated_at": get_settings_updated_at()}


# ── OCR test ──────────────────────────────────────────────────────────────────

@router.post("/test")
async def test_ocr(
    file: UploadFile = File(...),
    settings: Optional[str] = Form(None),
    max_pages: int = Form(5),
    show_boxes: bool = Form(True),
    current_user: dict = Depends(get_current_user),
):
    """
    OCR an uploaded document through the production code path. `settings` (JSON)
    lets the page test unsaved values; it never changes the saved configuration.
    """
    filename, ext, data = await _read_upload(file, TEST_EXTS)
    unsaved = _parse_unsaved_settings(settings)
    if unsaved is not None:
        cfg = OcrSettings(**unsaved.model_dump())
    else:
        async with get_db_context() as db:
            cfg, _ = await load_ocr_settings(db)
    pages = max(1, min(MAX_TEST_PAGES, int(max_pages)))

    async with _test_semaphore:
        try:
            result = await asyncio.to_thread(_run_test, data, filename, ext, cfg, pages, show_boxes)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(f"[OCRTest] OCR test failed for {filename}")
            raise HTTPException(status_code=500, detail=f"OCR test failed: {exc}")
    result["used_unsaved_settings"] = unsaved is not None
    return result


# ── Experimental VLM OCR (testing only) ───────────────────────────────────────

@router.get("/vlm/models")
async def vlm_models(current_user: dict = Depends(get_current_user)):
    from services import vlm_ocr_experimental as vlm

    server_url = await _user_ollama_url(current_user["id"])
    result = await vlm.list_models(server_url)
    result["busy"] = vlm.is_busy()
    return result


@router.post("/vlm/test")
async def vlm_test(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    page: int = Form(1),
    prompt_mode: str = Form("ocr_layout"),
    max_tokens: int = Form(8192),
    custom_prompt: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user),
):
    """EXPERIMENTAL: Chandra OCR 2 through Ollama. Not used by the production OCR pipeline."""
    from services import vlm_ocr_experimental as vlm

    model = (model or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="Select an Ollama model.")
    if prompt_mode not in vlm.PROMPT_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown prompt mode '{prompt_mode}'.")
    if not 256 <= max_tokens <= 16384:
        raise HTTPException(status_code=400, detail="Max output tokens must be between 256 and 16384.")
    if custom_prompt and len(custom_prompt) > 20000:
        raise HTTPException(status_code=400, detail="Custom prompt is too long (20,000 characters max).")
    if vlm.is_busy():
        raise HTTPException(status_code=409, detail="Another experimental VLM OCR run is in progress.")

    filename, ext, data = await _read_upload(file, VLM_EXTS)
    server_url = await _user_ollama_url(current_user["id"])

    try:
        image, page_count = await asyncio.to_thread(vlm.prepare_image, data, filename, page)
    except vlm.VlmOcrError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the document: {exc}")

    try:
        result = await vlm.run_vlm_ocr(
            server_url, model, image,
            prompt_mode=prompt_mode, max_tokens=max_tokens,
            custom_prompt=custom_prompt, is_disconnected=request.is_disconnected,
        )
    except vlm.VlmOcrError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except vlm.VlmOcrCancelled:
        logger.info("[VLM-OCR experimental] Client disconnected — generation aborted.")
        raise HTTPException(status_code=499, detail="Cancelled by the client.")

    result.update({
        "server_url": server_url,
        "filename": filename,
        "page": page if ext == ".pdf" else 1,
        "page_count": page_count,
        "image_size": [image.width, image.height],
        "source_preview": await asyncio.to_thread(vlm.thumbnail_data_url, image),
    })
    return result
