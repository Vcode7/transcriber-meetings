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
from tasks.pipeline import run_pipeline

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
    background_tasks.add_task(
        run_pipeline,
        recording_id,
        file_path,
        user_id,
        meeting_prompt=meeting_prompt or "",
        participant_voice_ids=participant_voice_ids or [],
        use_vocabulary=use_vocabulary,
        speaker_summary=speaker_summary,
    )
    logger.info(f"[Audio] Pipeline task scheduled for {recording_id}")
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

    logger.info(f"[Audio] Recording validated OK: {wav_path}")
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
