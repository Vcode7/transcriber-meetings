"""
rom_pipeline_router.py — ROM settings chosen on the Upload / Record / Tab Recording pages.

Endpoints (prefix /rom-pipeline):
  GET    /defaults                 — the user's saved ROM settings ("Save as default"), or defaults
  PUT    /defaults                 — save them
  POST   /context-files            — stage a meeting-context file (no recording exists yet)
  POST   /agenda-document          — stage an agenda document, extract it (PPT Testing extractor)
                                     and extract its agenda items (same LLM path as Create Agenda)
  POST   /agenda-document/{staged_id}/agendas
                                   — extract the agenda items again (e.g. "Extract Agenda from
                                     Tables" toggled) from the staged text, without re-reading it
  DELETE /staged/{staged_id}       — remove a staged file
  GET    /previous-meetings        — meetings usable as previous-meeting context (Stage 2)
  GET    /{recording_id}           — the meeting's ROM settings + automatic run status
  POST   /{recording_id}/start     — (re)start the automatic run from the stored settings
"""
import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import text

from database import dt_to_str, get_db, get_db_context
from models.rom_pipeline import DEFAULT_FIELDS, RomPipelineSettings
from routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rom-pipeline", tags=["rom-pipeline"])

MAX_FILE_SIZE_MB = 30
AGENDA_EXTENSIONS = {"ppt", "pptx", "pdf", "doc", "docx", "txt", "md", "png", "jpg", "jpeg", "webp", "xlsx", "xls", "csv"}
CONTEXT_EXTENSIONS = {"pdf", "docx", "pptx", "txt", "md", "png", "jpg", "jpeg", "webp", "xlsx", "xls", "csv"}


def _user_id(current_user) -> str:
    from routers.rom_router import _validate_user_id
    return _validate_user_id(current_user)


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""


async def _read_upload(file: UploadFile, allowed: set) -> tuple:
    filename = file.filename or "document"
    if _ext(filename) not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '.{_ext(filename)}'. Allowed: {', '.join(sorted(allowed))}")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File '{filename}' exceeds {MAX_FILE_SIZE_MB} MB limit.")
    return filename, content


# ── Defaults ──────────────────────────────────────────────────────────────────

@router.get("/defaults")
async def get_defaults(current_user: dict = Depends(get_current_user)):
    """Saved defaults, else the existing ROM settings (rom_* columns) and generate_mom_auto."""
    user_id = _user_id(current_user)
    async with get_db_context() as db:
        r = await db.execute(text("SELECT * FROM user_settings WHERE user_id = :uid"), {"uid": user_id})
        row = r.mappings().fetchone()
    row = dict(row) if row else {}

    saved = None
    if row.get("rom_pipeline_defaults"):
        try:
            saved = RomPipelineSettings.model_validate(json.loads(row["rom_pipeline_defaults"]))
        except Exception as exc:
            logger.warning(f"[ROMPipeline] Ignoring invalid saved defaults for {user_id}: {exc}")
    if saved is None:
        def _get(key, default):
            return row[key] if row.get(key) is not None else default
        generate_mom_auto = bool(_get("generate_mom_auto", 1))
        saved = RomPipelineSettings(
            output_type="mom" if generate_mom_auto else "off",
            transcript_window_minutes=max(0.5, min(30.0, float(_get("rom_transcript_window", 2.0)))),
            meeting_top_k=int(_get("rom_meeting_top_k", 5)),
            global_top_k=int(_get("rom_global_top_k", 3)),
            min_similarity_threshold=row.get("rom_min_similarity_threshold", 0.60),
            process_all_together=bool(_get("rom_stage2_process_all_together", 0)),
        )
        return {"defaults": saved.model_dump(include=set(DEFAULT_FIELDS)), "saved": False}
    return {"defaults": saved.model_dump(include=set(DEFAULT_FIELDS)), "saved": True}


@router.put("/defaults")
async def save_defaults(body: RomPipelineSettings, current_user: dict = Depends(get_current_user)):
    user_id = _user_id(current_user)
    payload = json.dumps(body.model_dump(include=set(DEFAULT_FIELDS)))
    now = dt_to_str(datetime.now(timezone.utc))
    async with get_db_context() as db:
        r = await db.execute(
            text("UPDATE user_settings SET rom_pipeline_defaults = :d, updated_at = :now WHERE user_id = :uid"),
            {"d": payload, "now": now, "uid": user_id},
        )
        if r.rowcount == 0:
            await db.execute(
                text("INSERT INTO user_settings (user_id, rom_pipeline_defaults, updated_at) VALUES (:uid, :d, :now)"),
                {"d": payload, "now": now, "uid": user_id},
            )
        await db.commit()
    return {"status": "success", "defaults": json.loads(payload)}


# ── Staged files ──────────────────────────────────────────────────────────────

@router.post("/context-files")
async def stage_context_file(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    from services.rom_staging import stage_file
    user_id = _user_id(current_user)
    filename, content = await _read_upload(file, CONTEXT_EXTENSIONS)
    staged_id = stage_file(user_id, filename, content)
    return {"staged_id": staged_id, "filename": filename}


def _extract_agendas(agenda_text: str, from_tables: bool) -> dict:
    """Agenda items of a staged agenda document's text (blocking: LLM)."""
    from services.ai_provider import get_provider
    from services.rom_service import extract_agendas
    provider = get_provider()
    try:
        return extract_agendas(provider, agenda_text, context="", from_tables=from_tables)
    finally:
        provider.unload_model()
        # The capture (and its transcription) usually starts right after this: free the GPU.
        provider.release_ollama_model()


class AgendaExtractionBody(BaseModel):
    from_tables: bool = True


@router.post("/agenda-document")
async def stage_agenda_document(
    file: UploadFile = File(...),
    from_tables: bool = Form(True),
    current_user: dict = Depends(get_current_user),
):
    """Stage the agenda document and extract its agenda items for the editable agenda list.
    ``from_tables``: "Extract Agenda from Tables" (see rom_service.extract_agendas)."""
    from services.agenda_doc_points import is_agenda_document
    from services.ppt_insights import extract_document_slides, slides_to_dicts
    from services.rom_staging import delete_staged, stage_file, write_extracted

    user_id = _user_id(current_user)
    filename, content = await _read_upload(file, AGENDA_EXTENSIONS)
    staged_id = stage_file(user_id, filename, content)
    loop = asyncio.get_event_loop()

    def _extract():
        if is_agenda_document(filename):
            return extract_document_slides(content, filename)
        from services.doc_extractor import extract_text_from_bytes
        return extract_text_from_bytes(content, filename), []

    try:
        agenda_text, slides = await loop.run_in_executor(None, _extract)
    except Exception as exc:
        delete_staged(user_id, staged_id)
        logger.error(f"[ROMPipeline] Agenda document extraction failed for '{filename}': {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to extract content from '{filename}': {exc}")
    if not agenda_text or not agenda_text.strip():
        delete_staged(user_id, staged_id)
        raise HTTPException(status_code=422, detail="No usable text could be extracted from the uploaded document.")
    write_extracted(user_id, staged_id, {"filename": filename, "extracted_text": agenda_text,
                                         "slides": slides_to_dicts(slides)})

    try:
        result = await loop.run_in_executor(None, lambda: _extract_agendas(agenda_text, from_tables))
    except Exception as exc:
        logger.error(f"[ROMPipeline] Agenda extraction failed for '{filename}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agenda extraction failed: {exc}")

    return {"staged_id": staged_id, "filename": filename, "agenda_text": agenda_text,
            "agendas": result["agendas"], "extraction_mode": result["extraction_mode"],
            "table_count": result["table_count"], "slide_count": len(slides)}


@router.post("/agenda-document/{staged_id}/agendas")
async def reextract_agenda_items(staged_id: str, body: AgendaExtractionBody,
                                 current_user: dict = Depends(get_current_user)):
    """Extract the agenda items of a staged agenda document again from its stored text."""
    from services.rom_staging import read_extracted

    extracted = read_extracted(_user_id(current_user), staged_id)
    agenda_text = (extracted or {}).get("extracted_text") or ""
    if not agenda_text.strip():
        raise HTTPException(status_code=404, detail="The agenda document is no longer staged. Upload it again.")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: _extract_agendas(agenda_text, body.from_tables))
    except Exception as exc:
        logger.error(f"[ROMPipeline] Agenda re-extraction failed for {staged_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agenda extraction failed: {exc}")
    return {"agendas": result["agendas"], "extraction_mode": result["extraction_mode"],
            "table_count": result["table_count"]}


@router.delete("/staged/{staged_id}")
async def delete_staged_file(staged_id: str, current_user: dict = Depends(get_current_user)):
    from services.rom_staging import delete_staged
    return {"deleted": delete_staged(_user_id(current_user), staged_id)}


@router.get("/previous-meetings")
async def previous_meetings(current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    """Meetings selectable as "previous meeting" context before the new meeting exists."""
    from routers.rom_router import get_previous_meetings_list
    return await get_previous_meetings_list("", current_user=current_user, db=db)


# ── Per meeting ───────────────────────────────────────────────────────────────

@router.get("/{recording_id}")
async def get_pipeline(recording_id: str, current_user: dict = Depends(get_current_user)):
    from services.rom_service import get_stage1_progress
    from tasks.pipeline import active_tasks
    from tasks.rom_auto_pipeline import is_running, read_state

    user_id = _user_id(current_user)
    state = await read_state(recording_id)
    if not state or state.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Recording not found")
    settings = state.get("settings")
    status = dict(state.get("status") or {}) or None
    if status:
        live = is_running(recording_id)
        rec_status = state.get("recording_status")
        if status.get("state") in ("queued", "running") and not live:
            status["state"] = "interrupted"
            status["error"] = status.get("error") or "The ROM pipeline stopped unexpectedly (e.g. the backend restarted)."
        elif status.get("state") == "waiting" and rec_status in ("error", "cancelled"):
            status["state"] = "error"
            status["error"] = f"Transcription did not finish (status: {rec_status})"
        elif (status.get("state") == "waiting" and rec_status == "done" and not live
              and recording_id not in active_tasks):
            status["state"] = "interrupted"
            status["error"] = "Transcription finished but the ROM pipeline did not start."
    stage1_progress = None
    if status and status.get("state") == "running" and status.get("stage") == "stage1":
        stage1_progress = get_stage1_progress(user_id, recording_id)
    return {
        "settings": settings.model_dump() if settings else None,
        "status": status,
        "recording_status": state.get("recording_status"),
        "stage1_progress": stage1_progress,
    }


@router.post("/{recording_id}/start")
async def start_pipeline(recording_id: str, current_user: dict = Depends(get_current_user)):
    """Run the stored ROM settings again (after an error or interruption)."""
    from tasks.rom_auto_pipeline import is_running, read_state, start, write_status

    user_id = _user_id(current_user)
    state = await read_state(recording_id)
    if not state or state.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Recording not found")
    settings = state.get("settings")
    if not settings or not settings.auto_run:
        raise HTTPException(status_code=400, detail="This meeting has no automatic ROM pipeline settings.")
    if state.get("recording_status") != "done":
        raise HTTPException(status_code=409, detail="Transcription has not finished yet.")
    if is_running(recording_id):
        raise HTTPException(status_code=409, detail="The ROM pipeline is already running for this meeting.")
    await write_status(recording_id, state="queued", error=None, stage=None, stage_label="Queued")
    start(recording_id, user_id)
    return {"status": "started"}
