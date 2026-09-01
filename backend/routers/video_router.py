"""
Video upload router — accepts video files, extracts audio, triggers existing pipeline,
and asynchronously runs the video OCR timeline extraction.

Design:
  - POST /video/upload  — main upload endpoint
  - GET  /video/jobs/{recording_id}  — job status (extends existing audio job status)
  - GET  /video/jobs/{recording_id}/ocr-status  — OCR-specific status
  - GET  /video/jobs/{recording_id}/video-transcript  — returns merged OCR transcript

The audio transcription pipeline is completely unchanged.
OCR runs as a separate asyncio task that writes video_transcript to the DB when done.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import text

from database import get_db, get_db_context, dt_to_str, to_json, from_json
from routers.auth import get_current_user
from utils.storage import save_upload, delete_file, get_user_dir
from utils.audio_utils import validate_audio, convert_to_wav, get_duration, trim_audio, process_audio_edit
from tasks.pipeline import run_pipeline, register_task
from tasks.upload_chunk_pipeline import run_upload_chunk_pipeline, UPLOAD_CHUNK_THRESHOLD_SEC
from services.video_processing_service import (
    extract_audio_from_video,
    extract_video_ocr_timeline,
    merge_ocr_results,
    get_video_duration,
    is_supported_video,
    SUPPORTED_VIDEO_EXTENSIONS,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/video", tags=["video"])

# ── OCR status tracking (in-memory, per recording) ───────────────────────────
# Maps recording_id → "pending" | "processing" | "done" | "skipped" | "error"
_ocr_status: dict[str, str] = {}


async def _validate_audio_for_user(wav_path: str, user_id: str, db) -> tuple[bool, str]:
    enabled = True
    min_dur = 2.0
    min_rms = 0.003
    try:
        r = await db.execute(
            text("SELECT enable_audio_validation, min_audio_duration_seconds, min_audio_rms_threshold FROM user_settings WHERE user_id = :uid"),
            {"uid": user_id}
        )
        row = r.mappings().fetchone()
        if row:
            if row.get("enable_audio_validation") is not None:
                enabled = bool(row["enable_audio_validation"])
            if row.get("min_audio_duration_seconds") is not None:
                min_dur = float(row["min_audio_duration_seconds"])
            if row.get("min_audio_rms_threshold") is not None:
                min_rms = float(row["min_audio_rms_threshold"])
    except Exception as e:
        logger.warning(f"Could not load audio validation settings: {e}")
    return validate_audio(wav_path, enabled=enabled, min_duration=min_dur, min_rms=min_rms)


async def _create_video_recording(
    video_path: str,
    wav_path: str,
    original_filename: str,
    user_id: str,
    duration: float,
    meeting_prompt: str = "",
    participant_voice_ids: Optional[List[str]] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
) -> str:
    """Insert a recording row for a video source and return the recording_id."""
    recording_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    pv_json = to_json(participant_voice_ids or [])

    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO recordings (
                    id, user_id, filename, file_path, duration, status, progress,
                    transcript, raw_text, summary, key_points, action_items, speakers_detected,
                    language, error_message, created_at, processed_at,
                    meeting_prompt, participant_voice_ids, use_vocabulary,
                    source_type
                )
                VALUES (
                    :id, :user_id, :filename, :file_path, :duration, 'pending', 'queued',
                    '[]', NULL, NULL, '[]', '[]', '[]', 'en', NULL, :created_at, NULL,
                    :meeting_prompt, :participant_voice_ids, :use_vocabulary,
                    'video'
                )
            """),
            {
                "id": recording_id,
                "user_id": user_id,
                "filename": original_filename,
                "file_path": video_path,   # store original video path
                "duration": duration,
                "created_at": dt_to_str(now),
                "meeting_prompt": meeting_prompt or "",
                "participant_voice_ids": pv_json,
                "use_vocabulary": 1 if use_vocabulary else 0,
            },
        )
        await db.commit()

    logger.info(f"[Video] Recording row created: {recording_id} (source_type=video)")
    return recording_id


import os

# Configurable delay (in seconds) between starting Whisper transcription and starting Video OCR
VIDEO_OCR_START_DELAY_SEC: float = float(os.environ.get("VIDEO_OCR_START_DELAY_SEC", "3.0"))


async def _run_ocr_task(recording_id: str, video_path: str, duration: float):
    """
    Async OCR task — runs in a thread executor to avoid blocking the event loop.
    Writes the merged video_transcript JSON to the DB when done.
    """
    _ocr_status[recording_id] = "processing"
    logger.info(f"[VideoOCR] Starting OCR task for recording={recording_id}, video={video_path}")

    try:
        loop = asyncio.get_event_loop()

        # Run synchronous OCR pipeline in a thread executor
        raw_ocr = await loop.run_in_executor(
            None,
            lambda: extract_video_ocr_timeline(video_path),
        )

        if not raw_ocr:
            logger.info(f"[VideoOCR] No readable OCR text found for recording={recording_id} — skipping")
            _ocr_status[recording_id] = "skipped"
            return

        # Merge consecutive identical blocks
        merged = await loop.run_in_executor(
            None,
            lambda: merge_ocr_results(raw_ocr, video_duration=duration),
        )

        # Persist to DB
        async with get_db_context() as db:
            await db.execute(
                text("UPDATE recordings SET video_transcript = :vt WHERE id = :id"),
                {"vt": to_json(merged), "id": recording_id},
            )
            await db.commit()

        _ocr_status[recording_id] = "done"
        logger.info(
            f"[VideoOCR] OCR complete for recording={recording_id}: "
            f"{len(raw_ocr)} raw frames → {len(merged)} merged blocks"
        )

    except asyncio.CancelledError:
        _ocr_status[recording_id] = "skipped"
        logger.info(f"[VideoOCR] OCR task cancelled for recording={recording_id}")
        raise
    except Exception as exc:
        _ocr_status[recording_id] = "error"
        logger.error(f"[VideoOCR] OCR task failed for recording={recording_id}: {exc}", exc_info=True)
    finally:
        try:
            from services.video_processing_service import unload_video_ocr_pipeline
            unload_video_ocr_pipeline()
        except Exception:
            pass


async def _run_synchronized_video_pipeline(
    recording_id: str,
    video_path: str,
    wav_path: str,
    user_id: str,
    duration: float,
    meeting_prompt: str = "",
    participant_voice_ids: Optional[List[str]] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
):
    """
    Synchronized Video Pipeline Orchestration:
      1. After audio extraction, immediately load Whisper model and start audio transcription.
      2. Wait a short configurable delay (default 3.0s) so Whisper model is fully initialized and VRAM stabilizes.
      3. Start Video OCR timeline extraction in parallel while Whisper continues transcribing.
      4. Both pipelines run concurrently and synchronize via asyncio.gather().
      5. After transcription finishes, explicitly unload Whisper model and free VRAM/RAM.
      6. After Video OCR finishes, explicitly unload RapidOCR / ONNX Runtime engine and free resources.
      7. Both pipelines synchronize before completing.
    """
    _ocr_status[recording_id] = "pending"
    logger.info(f"[VideoPipeline] Starting synchronized pipeline orchestration for recording={recording_id}")
    loop = asyncio.get_running_loop()

    # Task A: Audio Transcription
    async def _transcribe_task_fn():
        logger.info(f"[VideoPipeline] {recording_id} — Starting Whisper audio transcription...")
        try:
            if duration > UPLOAD_CHUNK_THRESHOLD_SEC:
                await run_upload_chunk_pipeline(
                    recording_id,
                    wav_path,
                    user_id,
                    meeting_prompt=meeting_prompt,
                    participant_voice_ids=participant_voice_ids,
                    use_vocabulary=use_vocabulary,
                    speaker_summary=speaker_summary,
                )
            else:
                await run_pipeline(
                    recording_id,
                    wav_path,
                    user_id,
                    meeting_prompt=meeting_prompt,
                    participant_voice_ids=participant_voice_ids,
                    use_vocabulary=use_vocabulary,
                    speaker_summary=speaker_summary,
                )
            logger.info(f"[VideoPipeline] {recording_id} — Audio transcription task finished ✓")
        finally:
            try:
                from services.transcription import unload_whisperx_model, unload_align_model
                unload_whisperx_model()
                unload_align_model()
            except Exception as e:
                logger.warning(f"[VideoPipeline] {recording_id} — Failed to unload Whisper models: {e}")

    # Task B: Video OCR (Delayed Start for VRAM stabilization)
    async def _ocr_task_fn():
        logger.info(f"[VideoPipeline] {recording_id} — Waiting {VIDEO_OCR_START_DELAY_SEC}s delay for VRAM stabilization...")
        await asyncio.sleep(VIDEO_OCR_START_DELAY_SEC)

        _ocr_status[recording_id] = "processing"
        logger.info(f"[VideoPipeline] {recording_id} — Starting Video OCR pipeline in parallel...")
        try:
            raw_ocr = await loop.run_in_executor(
                None,
                lambda: extract_video_ocr_timeline(video_path),
            )

            if not raw_ocr:
                logger.info(f"[VideoPipeline] {recording_id} — No readable OCR text found — skipping")
                _ocr_status[recording_id] = "skipped"
                return

            merged = await loop.run_in_executor(
                None,
                lambda: merge_ocr_results(raw_ocr, video_duration=duration),
            )

            async with get_db_context() as db:
                await db.execute(
                    text("UPDATE recordings SET video_transcript = :vt WHERE id = :id"),
                    {"vt": to_json(merged), "id": recording_id},
                )
                await db.commit()

            _ocr_status[recording_id] = "done"
            logger.info(
                f"[VideoPipeline] {recording_id} — OCR task complete: "
                f"{len(raw_ocr)} raw frames → {len(merged)} merged blocks ✓"
            )
        except asyncio.CancelledError:
            _ocr_status[recording_id] = "skipped"
            logger.info(f"[VideoPipeline] OCR task cancelled for recording={recording_id}")
            raise
        except Exception as exc:
            _ocr_status[recording_id] = "error"
            logger.error(f"[VideoPipeline] OCR task failed for recording={recording_id}: {exc}", exc_info=True)
        finally:
            try:
                from services.video_processing_service import unload_video_ocr_pipeline
                unload_video_ocr_pipeline()
            except Exception as e:
                logger.warning(f"[VideoPipeline] {recording_id} — Failed to unload Video OCR engine: {e}")

    # Launch concurrently and synchronize
    t_task = asyncio.create_task(_transcribe_task_fn())
    o_task = asyncio.create_task(_ocr_task_fn())

    try:
        await asyncio.gather(t_task, o_task)
        logger.info(f"[VideoPipeline] {recording_id} — Synchronized video pipeline complete ✓")
    except asyncio.CancelledError:
        t_task.cancel()
        o_task.cancel()
        raise
    except Exception as exc:
        logger.error(f"[VideoPipeline] {recording_id} — Orchestration error: {exc}", exc_info=True)


# ── Upload endpoint ───────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    meeting_prompt: Optional[str] = Form(default=""),
    participant_voice_ids: Optional[str] = Form(default="[]"),
    use_vocabulary: Optional[bool] = Form(default=False),
    speaker_summary: Optional[bool] = Form(default=False),
    trim_start_sec: Optional[float] = Form(default=None),
    trim_end_sec: Optional[float] = Form(default=None),
    cut_start_sec: Optional[float] = Form(default=None),
    cut_end_sec: Optional[float] = Form(default=None),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    """
    Accept a pre-recorded video file (MP4, AVI, MOV, MKV, WebM) for processing.

    Pipeline:
      1. Save the video file to disk.
      2. Validate the video format.
      3. Extract audio track to WAV using ffmpeg.
      4. Optionally trim/cut audio track if requested.
      5. Validate the extracted audio.
      6. Create a recording row (source_type='video').
      7. Schedule synchronized audio transcription & parallel OCR pipeline.
    """
    user_id = current_user["id"]
    logger.info(f"[Video] Upload request from user={user_id}, filename={file.filename}")

    # Validate format
    if not is_supported_video(file.filename or "", file.content_type or ""):
        exts = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video format. Accepted: {exts}",
        )

    # Parse participant_voice_ids
    try:
        pv_ids: List[str] = json.loads(participant_voice_ids or "[]")
    except Exception:
        pv_ids = []

    # ── 1. Save raw video ──────────────────────────────────────────────────────
    video_path = await save_upload(file, user_id, prefix="vid_")
    logger.info(f"[Video] Saved video to: {video_path}")

    # ── 2. Extract audio track ─────────────────────────────────────────────────
    base = video_path.rsplit(".", 1)[0]
    wav_path = base + "_audio.wav"
    try:
        extract_audio_from_video(video_path, wav_path)
    except RuntimeError as e:
        delete_file(video_path)
        logger.error(f"[Video] Audio extraction failed: {e}")
        raise HTTPException(status_code=422, detail=f"Audio extraction from video failed: {e}")

    # ── Optional trim / middle-cut ────────────────────────────────────────
    has_edit = (trim_start_sec is not None and trim_end_sec is not None) or (cut_start_sec is not None and cut_end_sec is not None)
    if has_edit:
        try:
            edited_path = process_audio_edit(
                wav_path,
                start_sec=trim_start_sec,
                end_sec=trim_end_sec,
                cut_start_sec=cut_start_sec,
                cut_end_sec=cut_end_sec,
            )
            if edited_path != wav_path:
                delete_file(wav_path)
                wav_path = edited_path
                logger.info(f"[Video] Edited extracted audio (trim: {trim_start_sec}-{trim_end_sec}, cut: {cut_start_sec}-{cut_end_sec}) → {wav_path}")
        except Exception as edit_err:
            logger.warning(f"[Video] Audio edit failed (non-fatal): {edit_err}; using original audio.")

    # ── 3. Validate extracted audio ───────────────────────────────────────────
    valid, reason = await _validate_audio_for_user(wav_path, user_id, db)
    if not valid:
        delete_file(video_path)
        delete_file(wav_path)
        raise HTTPException(status_code=422, detail=f"Extracted audio invalid: {reason}")

    # ── 4. Get duration ───────────────────────────────────────────────────────
    try:
        duration = get_duration(wav_path)
    except Exception:
        duration = get_video_duration(video_path) or 0.0

    logger.info(f"[Video] Validated OK. Duration={duration:.1f}s, video={video_path}, wav={wav_path}")

    # ── 5. Create recording row ───────────────────────────────────────────────
    recording_id = await _create_video_recording(
        video_path=video_path,
        wav_path=wav_path,
        original_filename=file.filename or "recording.mp4",
        user_id=user_id,
        duration=duration,
        meeting_prompt=meeting_prompt or "",
        participant_voice_ids=pv_ids,
        use_vocabulary=use_vocabulary or False,
        speaker_summary=speaker_summary or False,
    )

    # ── 6. Schedule synchronized video pipeline ────────────────────────────────
    task = asyncio.create_task(
        _run_synchronized_video_pipeline(
            recording_id=recording_id,
            video_path=video_path,
            wav_path=wav_path,
            user_id=user_id,
            duration=duration,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=pv_ids,
            use_vocabulary=use_vocabulary or False,
            speaker_summary=speaker_summary or False,
        )
    )
    register_task(recording_id, task)
    logger.info(f"[Video] Synchronized video pipeline scheduled for recording={recording_id}")

    return {
        "recording_id": recording_id,
        "status": "pending",
        "message": "Video uploaded. Audio transcription and OCR extraction started in background.",
        "source_type": "video",
    }


# ── Status endpoints ──────────────────────────────────────────────────────────

@router.get("/jobs/{recording_id}")
async def get_video_job_status(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Poll the combined status of a video recording job.
    Returns transcription status (same as /audio/jobs/{id}) plus OCR status.
    """
    user_id = current_user["id"]

    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT * FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    resp = {
        "job_id": recording_id,
        "status": rec["status"],
        "progress": rec.get("progress", ""),
        "duration": rec.get("duration", 0),
        "created_at": rec["created_at"],
        "source_type": rec.get("source_type", "video"),
        "ocr_status": _ocr_status.get(recording_id, "unknown" if rec.get("video_transcript") is None else "done"),
        "has_video_transcript": rec.get("video_transcript") is not None,
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
    elif rec["status"] == "error":
        resp["error"] = rec.get("error_message", "Unknown error.")

    return resp


@router.get("/jobs/{recording_id}/ocr-status")
async def get_ocr_status(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return the current OCR processing status for a video recording."""
    user_id = current_user["id"]

    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT id, video_transcript FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    vt = rec.get("video_transcript")
    status = _ocr_status.get(recording_id, "unknown" if vt is None else "done")
    blocks: list = from_json(vt) if vt else []

    return {
        "recording_id": recording_id,
        "ocr_status": status,
        "block_count": len(blocks),
    }


@router.get("/jobs/{recording_id}/video-transcript")
async def get_video_transcript(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return the merged video OCR transcript for a recording."""
    user_id = current_user["id"]

    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT id, source_type, video_transcript FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    vt = rec.get("video_transcript")
    blocks: list = from_json(vt) if vt else []
    ocr_status = _ocr_status.get(recording_id, "unknown" if vt is None else "done")

    return {
        "recording_id": recording_id,
        "source_type": rec.get("source_type", "audio"),
        "ocr_status": ocr_status,
        "video_transcript": blocks,
        "block_count": len(blocks),
    }


@router.post("/jobs/{recording_id}/rerun-ocr")
async def rerun_video_ocr(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Re-run the Video OCR timeline extraction and merging pipeline for an existing recording.

    - Only re-runs frame extraction, AI super-res preprocessing, RapidOCR, timeline generation, and block merging.
    - Does NOT re-run audio extraction, speech transcription, diarization, speaker ID, MoM generation, or AI insights.
    - Overwrites the `video_transcript` column in the database upon completion.
    """
    user_id = current_user["id"]

    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT id, source_type, file_path, duration FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    video_path = rec.get("file_path", "")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(
            status_code=400,
            detail="Video source file is no longer available on disk. Cannot re-run Video OCR.",
        )

    if _ocr_status.get(recording_id) == "processing":
        raise HTTPException(
            status_code=409,
            detail="Video OCR task is already running for this recording.",
        )

    duration = rec.get("duration", 0.0) or get_video_duration(video_path) or 0.0

    _ocr_status[recording_id] = "pending"
    asyncio.create_task(_run_ocr_task(recording_id, video_path, duration))
    logger.info(f"[VideoOCR] Re-run OCR task queued for recording={recording_id}")

    return {
        "recording_id": recording_id,
        "status": "processing",
        "ocr_status": "processing",
        "message": "Video OCR extraction re-run started in background.",
    }

