"""
raw_mom_router.py — Shared utility endpoints used by the ROM pipeline.

This router exposes two endpoints that are consumed by the ROM page (RomPage.tsx).
The Raw MoM Lab feature has been removed; these endpoints remain because the ROM
feature depends on them.

Endpoints
---------
POST /raw-mom/{recording_id}/agenda             — Parse/generate agenda items for a recording
POST /raw-mom/{recording_id}/extract-file-text  — Extract plain text from an uploaded file
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from database import get_db, get_db_context, from_json
from routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/raw-mom", tags=["raw-mom"])


# -- Pydantic request models --------------------------------------------------

class AgendaItem(BaseModel):
    topic: str
    speaker: Optional[str] = None


class CreateAgendaRequest(BaseModel):
    force_reparse: bool = False


# -- Helpers ------------------------------------------------------------------

async def _get_recording_or_404(recording_id: str, user_id: str) -> dict:
    """Fetch recording row or raise 404."""
    async with get_db_context() as db:
        r = await db.execute(
            text("""
                SELECT id, filename, created_at, duration, speakers_detected,
                       transcript, transcript_embedded, meeting_context_embedded,
                       agenda_summary, context_summary
                FROM recordings
                WHERE id = :id AND user_id = :uid
            """),
            {"id": recording_id, "uid": user_id},
        )
        row = r.mappings().fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Recording not found")
    return dict(row)


# -- POST agenda (create/load agenda items only, no LLM if cached) ------------

@router.post("/{recording_id}/agenda")
async def create_agenda_endpoint(
    recording_id: str,
    body: CreateAgendaRequest = CreateAgendaRequest(),
    current_user: dict = Depends(get_current_user),
):
    """
    Generate or load the agenda items for a recording.

    If a cached parsed agenda already exists in the DB and force_reparse=False,
    returns it immediately (no LLM call). Otherwise runs get_or_create_agenda_items
    which will parse the raw agenda document or generate from the context summary.
    """
    user_id = current_user["id"]
    rec = await _get_recording_or_404(recording_id, user_id)

    transcript = from_json(rec.get("transcript"), [])
    agenda_text: Optional[str] = rec.get("agenda_summary") or None
    if not agenda_text:
        agenda_text = await _load_raw_agenda_text(recording_id, user_id)

    import asyncio
    _loop = asyncio.get_running_loop()

    try:
        agendas, source = await _loop.run_in_executor(
            None,
            lambda: _run_create_agenda(
                recording_id=recording_id,
                user_id=user_id,
                transcript=transcript,
                agenda_text=agenda_text,
                force_reparse=body.force_reparse,
            ),
        )
    except Exception as e:
        logger.error(f"[RawMoM] Agenda creation failed for {recording_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agenda creation failed: {str(e)}")

    return {"agendas": agendas, "source": source}


# -- POST extract-file-text (in-session file text extraction) -----------------

from fastapi import UploadFile, File as FastAPIFile
import tempfile
import os as _os


@router.post("/{recording_id}/extract-file-text")
async def extract_file_text_endpoint(
    recording_id: str,
    file: UploadFile = FastAPIFile(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Accept a multipart file upload and extract its plain text.

    No DB storage. Returns { "filename": str, "text": str, "char_count": int }
    """
    await _get_recording_or_404(recording_id, current_user["id"])

    filename = file.filename or "upload"
    content = await file.read()

    suffix = _os.path.splitext(filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from services.doc_extractor import extract_text_from_file
        import asyncio
        _loop = asyncio.get_running_loop()
        text_content = await _loop.run_in_executor(
            None, lambda: extract_text_from_file(tmp_path, filename)
        )
    except Exception as e:
        logger.error(f"[RawMoM] extract-file-text failed for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")
    finally:
        try:
            _os.unlink(tmp_path)
        except Exception:
            pass

    text_content = text_content.strip()
    return {
        "filename": filename,
        "text": text_content,
        "char_count": len(text_content),
    }


# -- Async helper -------------------------------------------------------------

async def _load_raw_agenda_text(recording_id: str, user_id: str) -> Optional[str]:
    """Load raw text from agenda attachments (fallback when agenda_summary is empty)."""
    try:
        from services.doc_extractor import extract_text_from_file
        async with get_db_context() as db:
            r = await db.execute(
                text(
                    "SELECT filename, file_path FROM recording_attachments "
                    "WHERE recording_id = :rid AND user_id = :uid AND type = 'agenda' "
                    "ORDER BY created_at ASC LIMIT 1"
                ),
                {"rid": recording_id, "uid": user_id},
            )
            att = r.mappings().fetchone()

        if not att:
            return None

        text_content = extract_text_from_file(att["file_path"], att["filename"])
        return text_content.strip() if text_content else None

    except Exception as e:
        logger.warning(f"[RawMoM] Could not load raw agenda text: {e}")
        return None


# -- Sync pipeline runners ----------------------------------------------------

def _run_create_agenda(
    recording_id: str,
    user_id: str,
    transcript: list,
    agenda_text: Optional[str],
    force_reparse: bool,
) -> tuple:
    """Synchronous agenda generation. Returns (agenda_items, source)."""
    from services.rag_pipeline import (
        get_or_create_agenda_items,
        _load_parsed_agenda,
        _save_parsed_agenda,
    )
    from services.text_embedding_service import unload_text_embedder

    try:
        if not force_reparse:
            cached = _load_parsed_agenda(recording_id)
            if cached:
                logger.info(f"[RawMoM] Returning cached agenda ({len(cached)} items)")
                return cached, "cached"

        if force_reparse:
            _save_parsed_agenda(recording_id, user_id, [])

        agenda_items = get_or_create_agenda_items(
            recording_id=recording_id,
            user_id=user_id,
            transcript=transcript,
            agenda_text=agenda_text,
        )
        source = "parsed" if agenda_text else "generated"
        return agenda_items, source
    finally:
        try:
            from services.ai_provider import QwenProvider
            QwenProvider.unload_model()
        except Exception:
            pass
        try:
            unload_text_embedder()
        except Exception:
            pass
