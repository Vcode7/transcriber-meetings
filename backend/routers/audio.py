"""Audio upload/record router — accepts files, triggers pipeline."""
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Form
from sqlalchemy import text

from database import get_db, dt_to_str, to_json
from routers.auth import get_current_user
from utils.storage import save_upload, delete_file
from utils.audio_utils import validate_audio, convert_to_wav, get_duration
from tasks.pipeline import run_pipeline, run_finalize_pipeline
from tasks.chunk_pipeline import run_chunk_pipeline
from tasks.upload_chunk_pipeline import run_upload_chunk_pipeline, UPLOAD_CHUNK_THRESHOLD_SEC

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["audio"])


async def _create_recording_and_run(
    file_path: str,
    original_filename: str,
    user_id: str,
    background_tasks: BackgroundTasks,
    meeting_prompt: str = "",
    participant_voice_ids: Optional[List[str]] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
) -> str:
    """Create recording row and schedule pipeline."""
    try:
        duration = get_duration(file_path)
    except Exception:
        duration = 0.0

    recording_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    pv_json = to_json(participant_voice_ids or [])

    logger.info(f"[Audio] Creating recording {recording_id} for user {user_id}, file={file_path}, duration={duration:.1f}s")

    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO recordings (
                    id, user_id, filename, file_path, duration, status, progress,
                    transcript, raw_text, summary, key_points, action_items, speakers_detected,
                    language, error_message, created_at, processed_at,
                    meeting_prompt, participant_voice_ids, use_vocabulary
                )
                VALUES (
                    :id, :user_id, :filename, :file_path, :duration, 'pending', 'queued',
                    '[]', NULL, NULL, '[]', '[]', '[]', 'en', NULL, :created_at, NULL,
                    :meeting_prompt, :participant_voice_ids, :use_vocabulary
                )
            """),
            {
                "id": recording_id,
                "user_id": user_id,
                "filename": original_filename,
                "file_path": file_path,
                "duration": duration,
                "created_at": dt_to_str(now),
                "meeting_prompt": meeting_prompt or "",
                "participant_voice_ids": pv_json,
                "use_vocabulary": 1 if use_vocabulary else 0,
            },
        )
        await db.commit()

    logger.info(f"[Audio] Recording row created: {recording_id}. Scheduling pipeline...")
    import asyncio as _asyncio
    from tasks.pipeline import register_task
    task = _asyncio.create_task(
        run_pipeline(
            recording_id,
            file_path,
            user_id,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=participant_voice_ids or [],
            use_vocabulary=use_vocabulary,
            speaker_summary=speaker_summary,
        )
    )
    register_task(recording_id, task)
    logger.info(f"[Audio] Pipeline task scheduled for {recording_id}")
    return recording_id


async def _create_recording_and_run_chunked(
    file_path: str,
    original_filename: str,
    user_id: str,
    background_tasks: BackgroundTasks,
    duration: float,
    meeting_prompt: str = "",
    participant_voice_ids: Optional[List[str]] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
) -> str:
    """Create recording row and schedule chunked upload pipeline."""
    recording_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    pv_json = to_json(participant_voice_ids or [])

    logger.info(f"[Audio] Creating chunked recording {recording_id} for user {user_id}, duration={duration:.1f}s")

    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO recordings (
                    id, user_id, filename, file_path, duration, status, progress,
                    transcript, raw_text, summary, key_points, action_items, speakers_detected,
                    language, error_message, created_at, processed_at,
                    meeting_prompt, participant_voice_ids, use_vocabulary
                )
                VALUES (
                    :id, :user_id, :filename, :file_path, :duration, 'pending', 'queued',
                    '[]', NULL, NULL, '[]', '[]', '[]', 'en', NULL, :created_at, NULL,
                    :meeting_prompt, :participant_voice_ids, :use_vocabulary
                )
            """),
            {
                "id": recording_id,
                "user_id": user_id,
                "filename": original_filename,
                "file_path": file_path,
                "duration": duration,
                "created_at": dt_to_str(now),
                "meeting_prompt": meeting_prompt or "",
                "participant_voice_ids": pv_json,
                "use_vocabulary": 1 if use_vocabulary else 0,
            },
        )
        await db.commit()

    logger.info(f"[Audio] Recording row created: {recording_id}. Scheduling chunked pipeline...")
    import asyncio as _asyncio
    from tasks.pipeline import register_task
    task = _asyncio.create_task(
        run_upload_chunk_pipeline(
            recording_id,
            file_path,
            user_id,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=participant_voice_ids or [],
            use_vocabulary=use_vocabulary,
            speaker_summary=speaker_summary,
        )
    )
    register_task(recording_id, task)
    return recording_id


# ── Upload audio file ──────────────────────────────────────────
@router.post("/upload")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    meeting_prompt: Optional[str] = Form(default=""),
    participant_voice_ids: Optional[str] = Form(default="[]"),  # JSON array string
    use_vocabulary: Optional[bool] = Form(default=False),
    speaker_summary: Optional[bool] = Form(default=False),
    current_user: dict = Depends(get_current_user),
):
    """Upload a pre-recorded audio file for processing."""
    import json as _json
    user_id = current_user["id"]
    logger.info(f"[Audio] Upload request from user {user_id}, filename={file.filename}")

    # Parse participant_voice_ids JSON string
    try:
        pv_ids: List[str] = _json.loads(participant_voice_ids or "[]")
    except Exception:
        pv_ids = []

    raw_path = await save_upload(file, user_id, prefix="rec_")

    # Convert to WAV
    wav_path = raw_path.rsplit(".", 1)[0] + "_proc.wav"
    try:
        convert_to_wav(raw_path, wav_path)
        delete_file(raw_path)
    except Exception as e:
        logger.error(f"[Audio] Audio conversion failed: {e}", exc_info=True)
        delete_file(raw_path)
        raise HTTPException(status_code=422, detail=f"Audio conversion failed: {e}")

    valid, reason = validate_audio(wav_path)
    if not valid:
        logger.warning(f"[Audio] Audio validation failed: {reason}")
        delete_file(wav_path)
        raise HTTPException(status_code=422, detail=reason)

    logger.info(f"[Audio] Audio validated OK: {wav_path}")

    # Get duration (header-only read — no RAM spike regardless of file size)
    duration = get_duration(wav_path)
    logger.info(f"[Audio] Upload validated OK: {wav_path}, duration={duration:.1f}s")

    # ── Auto-route: large uploads use the chunked pipeline ────────────────
    if duration > UPLOAD_CHUNK_THRESHOLD_SEC:
        logger.info(
            f"[Audio] Upload duration {duration:.0f}s exceeds threshold "
            f"{UPLOAD_CHUNK_THRESHOLD_SEC:.0f}s — routing to chunked pipeline"
        )
        recording_id = await _create_recording_and_run_chunked(
            file_path=wav_path,
            original_filename=file.filename,
            user_id=user_id,
            background_tasks=background_tasks,
            duration=duration,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=pv_ids,
            use_vocabulary=use_vocabulary or False,
            speaker_summary=speaker_summary or False,
        )
    else:
        recording_id = await _create_recording_and_run(
            file_path=wav_path,
            original_filename=file.filename,
            user_id=user_id,
            background_tasks=background_tasks,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=pv_ids,
            use_vocabulary=use_vocabulary or False,
            speaker_summary=speaker_summary or False,
        )

    return {
        "recording_id": recording_id,
        "status": "pending",
        "message": "Audio uploaded. Processing started in background.",
    }


# ── Submit raw recorded audio (from browser MediaRecorder) ────
@router.post("/record")
async def submit_recording(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    meeting_prompt: Optional[str] = Form(default=""),
    participant_voice_ids: Optional[str] = Form(default="[]"),
    use_vocabulary: Optional[bool] = Form(default=False),
    speaker_summary: Optional[bool] = Form(default=False),
    current_user: dict = Depends(get_current_user),
):
    """
    Accept audio blob from browser MediaRecorder (webm/ogg/wav).
    Converts to 16kHz WAV and queues for processing.
    """
    import json as _json
    user_id = current_user["id"]
    logger.info(f"[Audio] Record submission from user {user_id}, content_type={file.content_type}")

    try:
        pv_ids: List[str] = _json.loads(participant_voice_ids or "[]")
    except Exception:
        pv_ids = []

    raw_path = await save_upload(file, user_id, prefix="live_")
    wav_path = raw_path.rsplit(".", 1)[0] + "_proc.wav"

    try:
        convert_to_wav(raw_path, wav_path)
        delete_file(raw_path)
    except Exception as e:
        logger.error(f"[Audio] Conversion failed: {e}", exc_info=True)
        delete_file(raw_path)
        raise HTTPException(status_code=422, detail=f"Conversion failed: {e}")

    valid, reason = validate_audio(wav_path)
    if not valid:
        logger.warning(f"[Audio] Audio validation failed: {reason}")
        delete_file(wav_path)
        raise HTTPException(status_code=422, detail=reason)

    # Get duration (header-only read — no RAM spike regardless of file size)
    duration = get_duration(wav_path)
    logger.info(f"[Audio] Recording validated OK: {wav_path}, duration={duration:.1f}s")

    # ── Auto-route: large recordings use the chunked pipeline ─────────────
    if duration > UPLOAD_CHUNK_THRESHOLD_SEC:
        logger.info(
            f"[Audio] Recording duration {duration:.0f}s exceeds threshold "
            f"{UPLOAD_CHUNK_THRESHOLD_SEC:.0f}s — routing to chunked pipeline"
        )
        recording_id = await _create_recording_and_run_chunked(
            file_path=wav_path,
            original_filename=file.filename or "live_recording.wav",
            user_id=user_id,
            background_tasks=background_tasks,
            duration=duration,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=pv_ids,
            use_vocabulary=use_vocabulary or False,
            speaker_summary=speaker_summary or False,
        )
    else:
        recording_id = await _create_recording_and_run(
            file_path=wav_path,
            original_filename=file.filename or "live_recording.wav",
            user_id=user_id,
            background_tasks=background_tasks,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=pv_ids,
            use_vocabulary=use_vocabulary or False,
            speaker_summary=speaker_summary or False,
        )

    return {
        "recording_id": recording_id,
        "status": "pending",
        "message": "Recording received. Processing started.",
    }


# ── Submit a background chunk during recording ────────────────
@router.post("/chunk")
async def submit_chunk(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_index: int = Form(...),
    chunk_start_sec: float = Form(default=0.0),
    chunk_end_sec: float = Form(default=0.0),
    meeting_prompt: Optional[str] = Form(default=""),
    use_vocabulary: Optional[bool] = Form(default=False),
    current_user: dict = Depends(get_current_user),
):
    """
    Accept a single 10-minute audio chunk submitted during a long recording.
    Saves the chunk, schedules transcription-only background processing,
    and returns a chunk_id for tracking.
    """
    user_id = current_user["id"]
    chunk_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    logger.info(
        f"[Audio] Chunk submission from user {user_id}, "
        f"chunk_index={chunk_index}, start={chunk_start_sec:.1f}s, end={chunk_end_sec:.1f}s"
    )

    raw_path = await save_upload(file, user_id, prefix=f"chunk_{chunk_index}_")
    wav_path = raw_path.rsplit(".", 1)[0] + "_proc.wav"

    try:
        convert_to_wav(raw_path, wav_path)
        delete_file(raw_path)
    except Exception as e:
        logger.error(f"[Audio] Chunk conversion failed: {e}", exc_info=True)
        delete_file(raw_path)
        raise HTTPException(status_code=422, detail=f"Chunk conversion failed: {e}")

    valid, reason = validate_audio(wav_path)
    if not valid:
        delete_file(wav_path)
        raise HTTPException(status_code=422, detail=reason)

    # Create chunk row in DB
    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO recording_chunks (
                    id, recording_id, chunk_index, chunk_start_sec, chunk_end_sec,
                    file_path, status, transcript, raw_text, aligned_result, created_at
                ) VALUES (
                    :id, NULL, :chunk_index, :chunk_start_sec, :chunk_end_sec,
                    :file_path, 'pending', '[]', NULL, NULL, :created_at
                )
            """),
            {
                "id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_start_sec": chunk_start_sec,
                "chunk_end_sec": chunk_end_sec,
                "file_path": wav_path,
                "created_at": dt_to_str(now),
            },
        )
        await db.commit()

    # Schedule transcription-only pipeline for this chunk
    background_tasks.add_task(
        run_chunk_pipeline,
        chunk_id=chunk_id,
        chunk_wav_path=wav_path,
        user_id=user_id,
        meeting_prompt=meeting_prompt or "",
        use_vocabulary=use_vocabulary or False,
    )

    logger.info(f"[Audio] Chunk {chunk_id} (index={chunk_index}) queued for background transcription")
    return {"chunk_id": chunk_id, "status": "pending"}


# ── Finalize a chunked recording (called when recording stops) ─
@router.post("/record-finalize")
async def finalize_recording(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_ids: Optional[str] = Form(default="[]"),         # JSON array of chunk IDs
    meeting_prompt: Optional[str] = Form(default=""),
    participant_voice_ids: Optional[str] = Form(default="[]"),
    use_vocabulary: Optional[bool] = Form(default=False),
    speaker_summary: Optional[bool] = Form(default=False),
    current_user: dict = Depends(get_current_user),
):
    """
    Called when the user stops a long recording (>= 10 min).
    Accepts the full audio blob + list of already-submitted chunk IDs.
    Saves the full audio, creates the recording row, then schedules
    the finalize pipeline which: waits for chunks, merges transcripts,
    runs diarization on full audio, assigns speakers, generates AI insights.
    """
    user_id = current_user["id"]
    logger.info(f"[Audio] Finalize request from user {user_id}")

    # Parse chunk_ids JSON
    try:
        parsed_chunk_ids: List[str] = json.loads(chunk_ids or "[]")
    except Exception:
        parsed_chunk_ids = []

    # Parse participant_voice_ids JSON
    try:
        pv_ids: List[str] = json.loads(participant_voice_ids or "[]")
    except Exception:
        pv_ids = []

    # Save the full audio blob
    raw_path = await save_upload(file, user_id, prefix="final_")
    wav_path = raw_path.rsplit(".", 1)[0] + "_proc.wav"

    try:
        convert_to_wav(raw_path, wav_path)
        delete_file(raw_path)
    except Exception as e:
        logger.error(f"[Audio] Finalize audio conversion failed: {e}", exc_info=True)
        delete_file(raw_path)
        raise HTTPException(status_code=422, detail=f"Audio conversion failed: {e}")

    valid, reason = validate_audio(wav_path)
    if not valid:
        delete_file(wav_path)
        raise HTTPException(status_code=422, detail=reason)

    # Get duration
    try:
        duration = get_duration(wav_path)
    except Exception:
        duration = 0.0

    # Create recording row
    recording_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO recordings (
                    id, user_id, filename, file_path, duration, status, progress,
                    transcript, raw_text, summary, key_points, action_items, speakers_detected,
                    language, error_message, created_at, processed_at,
                    meeting_prompt, participant_voice_ids, use_vocabulary,
                    chunk_ids, is_chunked
                ) VALUES (
                    :id, :user_id, :filename, :file_path, :duration, 'pending', 'queued',
                    '[]', NULL, NULL, '[]', '[]', '[]', 'en', NULL, :created_at, NULL,
                    :meeting_prompt, :participant_voice_ids, :use_vocabulary,
                    :chunk_ids, 1
                )
            """),
            {
                "id": recording_id,
                "user_id": user_id,
                "filename": file.filename or "recording.wav",
                "file_path": wav_path,
                "duration": duration,
                "created_at": dt_to_str(now),
                "meeting_prompt": meeting_prompt or "",
                "participant_voice_ids": json.dumps(pv_ids),
                "use_vocabulary": 1 if use_vocabulary else 0,
                "chunk_ids": json.dumps(parsed_chunk_ids),
            },
        )
        await db.commit()

    # Link chunks to this recording_id
    if parsed_chunk_ids:
        async with get_db() as db:
            for cid in parsed_chunk_ids:
                await db.execute(
                    text("UPDATE recording_chunks SET recording_id = :rid WHERE id = :cid"),
                    {"rid": recording_id, "cid": cid},
                )
            await db.commit()

    # Schedule finalize pipeline
    import asyncio as _asyncio
    from tasks.pipeline import register_task
    task = _asyncio.create_task(
        run_finalize_pipeline(
            recording_id=recording_id,
            full_wav_path=wav_path,
            chunk_ids=parsed_chunk_ids,
            user_id=user_id,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=pv_ids,
            use_vocabulary=use_vocabulary or False,
            speaker_summary=speaker_summary or False,
        )
    )
    register_task(recording_id, task)

    logger.info(
        f"[Audio] Finalize pipeline scheduled: recording={recording_id}, "
        f"chunks={len(parsed_chunk_ids)}, duration={duration:.1f}s"
    )
    return {
        "recording_id": recording_id,
        "status": "pending",
        "message": "Recording finalized. Processing started.",
    }


# ── Poll job status ───────────────────────────────────────────
@router.get("/jobs/{recording_id}")
async def get_job_status(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Poll the status of a recording job. Returns full result when done."""
    user_id = current_user["id"]

    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    import json
    resp = {
        "job_id": recording_id,
        "status": rec["status"],
        "progress": rec.get("progress", ""),
        "duration": rec.get("duration", 0),
        "created_at": rec["created_at"],
    }

    if rec["status"] == "done":
        resp["result"] = {
            "transcript": json.loads(rec["transcript"] or "[]"),
            "raw_text": rec.get("raw_text", ""),
            "summary": rec.get("summary", ""),
            "short_summary": rec.get("short_summary", ""),
            "detailed_summary": rec.get("detailed_summary", ""),
            "key_points": json.loads(rec["key_points"] or "[]"),
            "action_items": json.loads(rec["action_items"] or "[]"),
            "speakers_detected": json.loads(rec["speakers_detected"] or "[]"),
            "language": rec.get("language", "en"),
            "processed_at": rec.get("processed_at"),
            "speaker_summary": json.loads(rec["speaker_summary"]) if rec.get("speaker_summary") else None,
        }
    elif rec["status"] == "transcript_ready":
        # Phase 1 done — return transcript immediately while Qwen3 runs in background
        resp["ai_generating"] = True
        resp["result"] = {
            "transcript": json.loads(rec["transcript"] or "[]"),
            "raw_text": rec.get("raw_text", ""),
            "speakers_detected": json.loads(rec["speakers_detected"] or "[]"),
            "language": rec.get("language", "en"),
            # AI fields not yet available
            "summary": "",
            "short_summary": "",
            "detailed_summary": "",
            "key_points": [],
            "action_items": [],
        }
    elif rec["status"] == "error":
        resp["error"] = rec.get("error_message", "Unknown error.")
        logger.warning(f"[Audio] Job {recording_id} is in error state: {resp['error']}")

    return resp


@router.post("/jobs/{recording_id}/cancel")
async def cancel_job(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Cancel an active transcription/processing job.
    Terminates the background worker task, cleans up RAM/VRAM, and marks job cancelled.
    """
    user_id = current_user["id"]
    logger.info(f"[Audio] Cancel request received for recording={recording_id} from user={user_id}")

    async with get_db() as db:
        r = await db.execute(
            text("SELECT id, status, file_path FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    if rec["status"] in ("done", "error", "cancelled"):
        return {"status": rec["status"], "message": f"Job is already completed with status: {rec['status']}"}

    # Cancel the active task cooperatively
    from tasks.pipeline import cancel_task
    task_cancelled = await cancel_task(recording_id)

    # Force update the status in the database to 'cancelled' (in case the task wasn't running or unregistered)
    async with get_db() as db:
        await db.execute(
            text("UPDATE recordings SET status = 'cancelled', progress = NULL, error_message = 'Cancelled by user' WHERE id = :id"),
            {"id": recording_id}
        )
        await db.commit()

    # Unload models to free GPU/CPU memory
    from tasks.pipeline import unload_all_models
    unload_all_models()

    logger.info(f"[Audio] Cancel request completed for recording={recording_id}. Active task found/cancelled={task_cancelled}")
    return {"status": "cancelled", "message": "Processing cancelled successfully."}


# ── List all active jobs for user ─────────────────────────────
@router.get("/jobs")
async def list_active_jobs(
    current_user: dict = Depends(get_current_user),
):
    """
    Return all non-completed recordings for the authenticated user.
    Used by the frontend on startup to reconnect to in-flight jobs
    that were started before the current page session.

    Only returns recordings in status: pending, processing, transcript_ready.
    (Done/error recordings are accessible through the History endpoints.)
    """
    user_id = current_user["id"]

    async with get_db() as db:
        r = await db.execute(
            text("""
                SELECT id, filename, status, progress, duration, created_at, meeting_prompt
                FROM recordings
                WHERE user_id = :uid
                  AND status IN ('pending', 'processing', 'transcript_ready')
                ORDER BY created_at DESC
                LIMIT 20
            """),
            {"uid": user_id},
        )
        rows = r.mappings().fetchall()

    jobs = []
    for rec in rows:
        jobs.append({
            "job_id": rec["id"],
            "filename": rec.get("filename", ""),
            "status": rec["status"],
            "progress": rec.get("progress", ""),
            "duration": rec.get("duration", 0),
            "created_at": rec["created_at"],
        })

    logger.debug(f"[Audio] Active jobs for user {user_id}: {len(jobs)}")
    return {"jobs": jobs}
