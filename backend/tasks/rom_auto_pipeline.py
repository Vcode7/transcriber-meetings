"""
rom_auto_pipeline.py — Runs the ROM pipeline by itself once transcription finishes.

The ROM settings chosen on the Upload / Record / Tab Recording pages are stored in
``recordings.rom_settings``. When the transcription pipeline reaches status 'done'
(tasks/pipeline.py calls ``maybe_start``), this module runs the same stages the ROM page
runs, by calling the existing ROM route handlers directly:

  mom: meeting context → Stage 1 → Stage 2 → MoM from Stage 2 points
  rom: meeting context → Stage 1 → Stage 2 → Create Agenda (user's agendas) → Final ROM (long,
       with agenda action points unless "Include action points" is off)
       → MoM from Stage 2 points when the user's "generate MoM automatically" setting is on

Progress is written to ``recordings.rom_pipeline_status`` (polled by the ROM page via
/rom-pipeline/{id}). Runs are serialised: one LLM-heavy ROM run at a time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import HTTPException
from sqlalchemy import text

from database import dt_to_str, get_db_context
from models.rom_pipeline import RomPipelineSettings, parse_rom_settings

logger = logging.getLogger(__name__)

STAGE_LABELS = {
    "context": "Preparing meeting context",
    "stage1": "Stage 1: Window extraction",
    "stage2": "Stage 2: RAG enhancement",
    "agenda": "Stage 3: Creating agenda",
    "final_rom": "Stage 3: Agenda matching & Final ROM",
    "mom": "Generating MoM from Stage 2 points",
}
ACTIVE_STATES = ("waiting", "queued", "running")

_TASKS: Dict[str, asyncio.Task] = {}
_RUN_LOCK: Optional[asyncio.Lock] = None


def _lock() -> asyncio.Lock:
    global _RUN_LOCK
    if _RUN_LOCK is None:
        _RUN_LOCK = asyncio.Lock()
    return _RUN_LOCK


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initial_status_json(settings: RomPipelineSettings) -> str:
    """rom_pipeline_status stored when the meeting is created."""
    if not settings.auto_run:
        return json.dumps({"state": "off", "output_type": settings.output_type})
    return json.dumps({
        "state": "waiting", "output_type": settings.output_type, "stage": None,
        "stage_label": "Waiting for transcription to finish", "message": "", "error": None,
    })


def is_running(recording_id: str) -> bool:
    task = _TASKS.get(recording_id)
    return bool(task and not task.done())


async def read_state(recording_id: str) -> Dict[str, Any]:
    """{status, recording_status, settings, user_id} of a recording (raw DB values parsed)."""
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT user_id, status, rom_settings, rom_pipeline_status FROM recordings WHERE id = :id"),
            {"id": recording_id},
        )
        row = r.mappings().fetchone()
    if not row:
        return {}
    try:
        status = json.loads(row["rom_pipeline_status"]) if row["rom_pipeline_status"] else None
    except Exception:
        status = None
    return {
        "user_id": row["user_id"],
        "recording_status": row["status"],
        "settings": parse_rom_settings(row["rom_settings"]),
        "status": status if isinstance(status, dict) else None,
    }


async def write_status(recording_id: str, **fields: Any) -> Dict[str, Any]:
    state = await read_state(recording_id)
    status = dict(state.get("status") or {})
    status.update(fields)
    status["updated_at"] = _now()
    async with get_db_context() as db:
        await db.execute(
            text("UPDATE recordings SET rom_pipeline_status = :s WHERE id = :id"),
            {"s": json.dumps(status), "id": recording_id},
        )
        await db.commit()
    return status


async def replaces_direct_mom(recording_id: str) -> bool:
    """True when the meeting's MoM comes from the ROM run (output MoM/ROM), so the transcription
    pipeline must skip its direct transcript-summary MoM."""
    try:
        settings = (await read_state(recording_id)).get("settings")
        return bool(settings and settings.auto_run)
    except Exception as exc:
        logger.warning(f"[ROMAuto] {recording_id} — could not read ROM settings: {exc}")
        return False


async def maybe_start(recording_id: str, user_id: str) -> bool:
    """Called when the transcription pipeline returns: start the ROM run if the meeting asked for one."""
    try:
        state = await read_state(recording_id)
        settings = state.get("settings")
        status = state.get("status") or {}
        if not settings or not settings.auto_run:
            return False
        if state.get("recording_status") != "done":
            logger.info(f"[ROMAuto] {recording_id} — transcription ended with status "
                        f"{state.get('recording_status')!r}; ROM pipeline not started")
            await write_status(recording_id, state="error", stage=None,
                               error=f"Transcription did not finish (status: {state.get('recording_status')})",
                               stage_label="Not started")
            return False
        if status.get("state") != "waiting":
            return False
        return start(recording_id, user_id)
    except Exception as exc:
        logger.error(f"[ROMAuto] {recording_id} — could not start ROM pipeline: {exc}", exc_info=True)
        return False


def start(recording_id: str, user_id: str) -> bool:
    """Schedule a run (no-op if one is already running for this recording)."""
    if is_running(recording_id):
        return False
    task = asyncio.create_task(run(recording_id, user_id))
    _TASKS[recording_id] = task
    task.add_done_callback(lambda _t: _TASKS.pop(recording_id, None) if _TASKS.get(recording_id) is _t else None)
    return True


# ── Steps ─────────────────────────────────────────────────────────────────────

async def _attach_context_files(recording_id: str, user_id: str, settings: RomPipelineSettings) -> int:
    """Move staged context files onto the recording and embed them (same as /attachments/{id}/process)."""
    from routers.attachments_router import _attachment_dir, _compute_hash, process_attachments
    from services.rom_staging import delete_staged, staged_file

    attached = 0
    for ref in settings.context_files:
        src = staged_file(user_id, ref.staged_id)
        if not src:
            logger.warning(f"[ROMAuto] {recording_id} — staged context file {ref.filename!r} is gone; skipped")
            continue
        with open(src, "rb") as fh:
            content = fh.read()
        filename = ref.filename or os.path.basename(src)
        dest = os.path.join(_attachment_dir(recording_id), f"{uuid.uuid4().hex}_{os.path.basename(src)}")
        shutil.copyfile(src, dest)
        async with get_db_context() as db:
            await db.execute(
                text("INSERT INTO recording_attachments (id, recording_id, user_id, type, filename, file_path, file_hash, created_at) "
                     "VALUES (:id, :rid, :uid, 'context', :filename, :path, :hash, :created_at)"),
                {"id": str(uuid.uuid4()), "rid": recording_id, "uid": user_id, "filename": filename,
                 "path": dest, "hash": _compute_hash(content), "created_at": dt_to_str(datetime.now(timezone.utc))},
            )
            await db.commit()
        delete_staged(user_id, ref.staged_id)
        attached += 1
    if attached:
        await process_attachments(recording_id, type="context", current_user={"id": user_id})
    return attached


async def _attach_agenda_document(recording_id: str, user_id: str, settings: RomPipelineSettings) -> bool:
    """Staged agenda document → rom_data.agenda_documents (used by Generate Final ROM)."""
    from routers.rom_router import _get_rom_data, _save_rom_data
    from services.agenda_doc_points import is_agenda_document
    from services.rom_staging import delete_staged, read_extracted

    ref = settings.agenda_doc
    if not ref:
        return False
    extracted = read_extracted(user_id, ref.staged_id)
    if not extracted:
        logger.warning(f"[ROMAuto] {recording_id} — staged agenda document {ref.filename!r} is gone; skipped")
        return False
    filename = extracted.get("filename") or ref.filename
    if is_agenda_document(filename) and extracted.get("slides"):
        async with get_db_context() as db:
            data = await _get_rom_data(recording_id, user_id, db)
            docs = [d for d in data.get("agenda_documents") or [] if isinstance(d, dict) and d.get("filename") != filename]
            docs.append({"filename": filename, "extracted_text": extracted.get("extracted_text", ""),
                         "slides": extracted["slides"]})
            data["agenda_documents"] = docs
            await _save_rom_data(recording_id, user_id, data, db)
    delete_staged(user_id, ref.staged_id)
    return True


async def _generate_mom_auto_enabled(user_id: str) -> bool:
    async with get_db_context() as db:
        r = await db.execute(text("SELECT generate_mom_auto FROM user_settings WHERE user_id = :uid"), {"uid": user_id})
        row = r.mappings().fetchone()
    return bool(row["generate_mom_auto"]) if row and row.get("generate_mom_auto") is not None else True


TRANSCRIPTION_POLL_SEC = 10


async def _wait_for_transcriptions(recording_id: str, next_stage: str) -> None:
    """The GPU cannot hold Whisper/diarization and the LLM at once (6 GB): before each stage, wait
    while another meeting is being transcribed so the ROM run never starves it of VRAM."""
    from tasks.pipeline import active_tasks

    async def busy() -> bool:
        if any(rid != recording_id and not t.done() for rid, t in list(active_tasks.items())):
            return True
        # 10-minute chunks of a live recording are transcribed in the background (not in active_tasks).
        # Old rows can be stuck in 'pending', so only recent chunks count.
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        async with get_db_context() as db:
            r = await db.execute(text("SELECT COUNT(*) FROM recording_chunks WHERE status IN ('pending', 'processing') "
                                      "AND created_at >= :cutoff"), {"cutoff": cutoff})
            return bool(r.scalar())

    announced = False
    while await busy():
        if not announced:
            logger.info(f"[ROMAuto] {recording_id} — waiting for another transcription before {next_stage}")
            await write_status(recording_id, state="running", stage=next_stage,
                               stage_label="Waiting for another transcription to finish",
                               message=f"Next: {STAGE_LABELS[next_stage]}")
            announced = True
        await asyncio.sleep(TRANSCRIPTION_POLL_SEC)


def _release_llm() -> None:
    try:
        from services.ai_provider import get_provider
        get_provider().release_ollama_model()
    except Exception as exc:
        logger.warning(f"[ROMAuto] Could not release the LLM: {exc}")


def _stage2_examples() -> list:
    try:
        from services.training.stage2_training_service import load_example_points
        return [p for p in load_example_points() if p and str(p).strip()]
    except Exception:
        return []


async def run(recording_id: str, user_id: str) -> None:
    from routers import rom_router as rr

    await write_status(recording_id, state="queued", stage=None, stage_label="Queued", error=None,
                       message="", finished_at=None)
    async with _lock():
        state = await read_state(recording_id)
        settings: Optional[RomPipelineSettings] = state.get("settings")
        if not settings or not settings.auto_run:
            await write_status(recording_id, state="off")
            return
        current = None

        async def stage(name: str, message: str = "") -> None:
            nonlocal current
            current = name
            await _wait_for_transcriptions(recording_id, name)
            logger.info(f"[ROMAuto] {recording_id} — {STAGE_LABELS[name]}")
            await write_status(recording_id, state="running", stage=name, stage_label=STAGE_LABELS[name],
                               message=message)

        await write_status(recording_id, state="running", output_type=settings.output_type, started_at=_now(),
                           finished_at=None, error=None)
        try:
            if settings.context_files:
                await stage("context")
                await _attach_context_files(recording_id, user_id, settings)

            await stage("stage1")
            async with get_db_context() as db:
                await rr.generate_stage1(
                    recording_id, rr.Stage1Request(transcript_window_minutes=settings.transcript_window_minutes),
                    current_user=user_id, db=db)

            await stage("stage2")
            previous_mode = settings.previous_meeting_mode
            if previous_mode == "select" and not settings.previous_meeting_id:
                previous_mode = "auto"
            async with get_db_context() as db:
                await rr.generate_stage2(recording_id, rr.Stage2Request(
                    meeting_context_top_k=settings.meeting_top_k,
                    global_context_top_k=settings.global_top_k,
                    discussion_window_size=settings.discussion_window_size,
                    min_similarity_threshold=settings.min_similarity_threshold,
                    process_all_together=settings.process_all_together,
                    reference_example_points=_stage2_examples() or None,
                    previous_meeting_mode=previous_mode,
                    previous_meeting_id=settings.previous_meeting_id if previous_mode == "select" else None,
                    previous_meeting_top_k=settings.previous_meeting_top_k if previous_mode != "off" else 0,
                ), current_user=user_id, db=db)

            make_mom = settings.output_type == "mom"
            if settings.output_type == "rom":
                await stage("agenda")
                await _attach_agenda_document(recording_id, user_id, settings)
                async with get_db_context() as db:
                    await rr.create_agenda(recording_id, rr.CreateAgendaRequest(
                        agenda_text=settings.agenda_text or None,
                        agendas=settings.agendas or None,
                        force_reextract=True,
                        agenda_from_tables=settings.agenda_from_tables,
                    ), current_user=user_id, db=db)

                await stage("final_rom")
                async with get_db_context() as db:
                    await rr.generate_final_rom_from_agendas(
                        recording_id,
                        rr.GenerateFinalRomRequest(include_action_points=settings.include_action_points),
                        current_user=user_id, db=db)
                make_mom = await _generate_mom_auto_enabled(user_id)

            if make_mom:
                await stage("mom")
                async with get_db_context() as db:
                    await rr.generate_mom_from_rom(recording_id, action_chunk_size=None, current_user=user_id, db=db)

            await write_status(recording_id, state="done", stage=None, stage_label="Complete", finished_at=_now(),
                               message="")
            logger.info(f"[ROMAuto] {recording_id} — ROM pipeline complete ({settings.output_type})")
        except asyncio.CancelledError:
            await write_status(recording_id, state="error", error="Cancelled", finished_at=_now())
            raise
        except Exception as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            logger.error(f"[ROMAuto] {recording_id} — failed at {current}: {detail}", exc_info=True)
            await write_status(recording_id, state="error", error=str(detail),
                               stage_label=f"Failed: {STAGE_LABELS.get(current, current or 'start')}",
                               finished_at=_now())
        finally:
            # Free the GPU for the next transcription (the Ollama server otherwise keeps the LLM loaded).
            await asyncio.get_event_loop().run_in_executor(None, _release_llm)
