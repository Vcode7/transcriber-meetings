"""History router — list, view, and delete recordings."""
import hashlib
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy import text

from database import get_db, to_json, from_json
from routers.auth import get_current_user
from utils.storage import delete_file
from services.llm import (
    generate_short_summary,
    generate_detailed_summary,
    generate_key_points,
    generate_action_items,
    build_context_summary,
)
from config import settings
from tasks.pipeline import _filter_high_confidence_segments, _raw_text_hash

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/history", tags=["history"])


@router.patch("/{recording_id}/rename")
async def rename_recording(
    recording_id: str,
    filename: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user),
):
    """Rename a recording. Validates the new name and updates in DB."""
    name = filename.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name cannot be empty.")
    if len(name) > 200:
        raise HTTPException(status_code=422, detail="Name too long (max 200 chars).")

    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text("UPDATE recordings SET filename = :name WHERE id = :id AND user_id = :uid"),
            {"name": name, "id": recording_id, "uid": user_id},
        )
        await db.commit()
        if r.rowcount == 0:
            raise HTTPException(status_code=404, detail="Recording not found.")

    return {"id": recording_id, "filename": name}


@router.get("")
async def list_history(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text("""
                SELECT id, filename, duration, status, speakers_detected, summary, created_at, processed_at
                FROM recordings
                WHERE user_id = :uid
                ORDER BY created_at DESC
                LIMIT 100
            """),
            {"uid": user_id},
        )
        recordings = r.mappings().fetchall()

    return [
        {
            "id": rec["id"],
            "filename": rec.get("filename", ""),
            "duration": rec.get("duration", 0),
            "status": rec.get("status", "unknown"),
            "speakers_detected": json.loads(rec["speakers_detected"] or "[]"),
            "has_summary": bool(rec.get("summary")),
            "created_at": rec["created_at"],
            "processed_at": rec.get("processed_at"),
        }
        for rec in recordings
    ]


@router.get("/{recording_id}")
async def get_recording(recording_id: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    rec_dict = dict(rec)
    spk_mappings = from_json(rec_dict.get("speaker_mappings"), {})
    if spk_mappings and isinstance(spk_mappings, dict):
        from services.speaker_sync import apply_speaker_mappings_to_recording_dict
        rec_dict = apply_speaker_mappings_to_recording_dict(rec_dict, spk_mappings)

    return {
        "id": rec_dict["id"],
        "filename": rec_dict.get("filename", ""),
        "duration": rec_dict.get("duration", 0),
        "status": rec_dict.get("status"),
        "file_path": rec_dict.get("file_path"),
        "source_type": rec_dict.get("source_type", "audio"),
        "video_transcript": from_json(rec_dict["video_transcript"]) if rec_dict.get("video_transcript") else None,
        "transcript": from_json(rec_dict["transcript"], []) if isinstance(rec_dict["transcript"], str) else rec_dict.get("transcript", []),
        "raw_text": rec_dict.get("raw_text", ""),
        "summary": rec_dict.get("summary", ""),
        "short_summary": rec_dict.get("short_summary", "") or "",
        "detailed_summary": rec_dict.get("detailed_summary", "") or "",
        "key_points": from_json(rec_dict["key_points"], []) if isinstance(rec_dict["key_points"], str) else rec_dict.get("key_points", []),
        "action_items": from_json(rec_dict["action_items"], []) if isinstance(rec_dict["action_items"], str) else rec_dict.get("action_items", []),
        "speakers_detected": from_json(rec_dict["speakers_detected"], []) if isinstance(rec_dict["speakers_detected"], str) else rec_dict.get("speakers_detected", []),
        "language": rec_dict.get("language", "en"),
        "speaker_summary": from_json(rec_dict["speaker_summary"]) if isinstance(rec_dict.get("speaker_summary"), str) else rec_dict.get("speaker_summary"),
        "speaker_mappings": from_json(rec_dict.get("speaker_mappings"), {}) if isinstance(rec_dict.get("speaker_mappings"), str) else rec_dict.get("speaker_mappings", {}),
        "created_at": rec_dict["created_at"],
        "processed_at": rec_dict.get("processed_at"),
    }


@router.delete("/{recording_id}")
async def delete_recording(recording_id: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text("SELECT file_path FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()
        if not rec:
            raise HTTPException(status_code=404, detail="Recording not found.")

        if rec.get("file_path"):
            delete_file(rec["file_path"])

        # Remove from any collections
        await db.execute(
            text("DELETE FROM meeting_collection_items WHERE meeting_id = :id"),
            {"id": recording_id},
        )
        
        await db.execute(
            text("DELETE FROM recordings WHERE id = :id"),
            {"id": recording_id},
        )
        await db.commit()

    return {"message": "Recording deleted."}


@router.get("/{recording_id}/audio")
async def stream_audio(recording_id: str, current_user: dict = Depends(get_current_user)):
    """Return file path for audio playback (frontend fetches with auth)."""
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text("SELECT file_path FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Not found.")

    from fastapi.responses import FileResponse
    from pathlib import Path
    import os

    fp = rec.get("file_path", "")
    if not fp:
        raise HTTPException(status_code=404, detail="Audio file not found on disk.")

    target_path = Path(fp)
    if not target_path.is_absolute() or not target_path.exists():
        # Try resolving relative to RUNTIME_DIR
        candidate = settings.RUNTIME_DIR / fp
        if candidate.exists():
            target_path = candidate
        else:
            # Fallback search in user's upload directory
            upload_candidate = settings.UPLOAD_DIR / user_id / target_path.name
            if upload_candidate.exists():
                target_path = upload_candidate

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found on disk.")

    ext = target_path.suffix.lower()
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".webm": "audio/webm",
        ".flac": "audio/flac",
    }
    media_type = media_types.get(ext, "audio/wav")

    return FileResponse(str(target_path), media_type=media_type)



@router.post("/{recording_id}/regenerate-insights")
async def regenerate_insights(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Re-generate all AI summaries for an existing recording using the local Qwen3 4B model.
    Reads the existing transcript from the database — no re-transcription required.
    Updates: short_summary, detailed_summary, summary, key_points, action_items.
    """
    user_id = current_user["id"]

    # Fetch the recording (include context_summary for caching)
    async with get_db() as db:
        r = await db.execute(
            text("SELECT id, transcript, raw_text, status, context_summary, context_summary_hash "
                 "FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    if rec.get("status") not in ("done", "error", "transcript_ready"):
        raise HTTPException(
            status_code=400,
            detail="Recording is still being processed. Wait until transcription finishes.",
        )

    transcript = from_json(rec["transcript"], [])
    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="No transcript found. The recording may not have been transcribed yet.",
        )

    # Filter low-confidence segments before generating insights
    transcript = _filter_high_confidence_segments(transcript, settings.MIN_AVG_SEGMENT_CONFIDENCE)
    logger.info(f"[Regenerate] Using {len(transcript)} high-confidence segments for {recording_id}")

    # Resolve context_summary (reuse cached or build fresh)
    raw_text = rec.get("raw_text") or ""
    current_hash = _raw_text_hash(raw_text)
    context: str | None = None
    if (
        rec.get("context_summary")
        and rec.get("context_summary_hash") == current_hash
    ):
        context = rec["context_summary"]
        logger.info(f"[Regenerate] Reusing cached context_summary for {recording_id}")
    else:
        logger.info(f"[Regenerate] Building fresh context_summary for {recording_id}")
        import asyncio as _asyncio
        _loop = _asyncio.get_running_loop()
        try:
            context = await _loop.run_in_executor(None, build_context_summary, transcript)
            if context:
                async with get_db() as db:
                    await db.execute(
                        text("UPDATE recordings SET context_summary = :ctx, context_summary_hash = :h "
                             "WHERE id = :id AND user_id = :uid"),
                        {"ctx": context, "h": current_hash, "id": recording_id, "uid": user_id},
                    )
                    await db.commit()
                logger.info(f"[Regenerate] context_summary stored for {recording_id} ✓")
        except Exception as ctx_err:
            logger.warning(f"[Regenerate] context_summary build failed (non-fatal): {ctx_err}")
            context = None

    logger.info(f"[Regenerate] Starting Qwen3 4B re-summarization for {recording_id}")

    try:
        import asyncio as _asyncio
        _loop = _asyncio.get_running_loop()
        short_summary = await _loop.run_in_executor(
            None, lambda: generate_short_summary(transcript, context=context)
        )
        detailed_summary = await _loop.run_in_executor(
            None, lambda: generate_detailed_summary(transcript, context=context)
        )
        key_points = await _loop.run_in_executor(
            None, lambda: generate_key_points(transcript, context=context)
        )
        action_items = await _loop.run_in_executor(
            None, lambda: generate_action_items(transcript, context=context)
        )
    except Exception as e:
        logger.error(f"[Regenerate] Qwen3 inference failed for {recording_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
    finally:
        from services.ai_provider import QwenProvider
        QwenProvider.unload_model()

    # Save back to DB
    async with get_db() as db:
        await db.execute(
            text("""
                UPDATE recordings SET
                    short_summary = :short_summary,
                    detailed_summary = :detailed_summary,
                    summary = :summary,
                    key_points = :key_points,
                    action_items = :action_items
                WHERE id = :id AND user_id = :uid
            """),
            {
                "short_summary": short_summary,
                "detailed_summary": detailed_summary,
                "summary": short_summary,  # backward compat
                "key_points": to_json(key_points),
                "action_items": to_json(action_items),
                "id": recording_id,
                "uid": user_id,
            },
        )
        await db.commit()

    logger.info(f"[Regenerate] Qwen3 4B re-summarization complete for {recording_id} ✓")

    return {
        "status": "done",
        "recording_id": recording_id,
        "short_summary": short_summary,
        "detailed_summary": detailed_summary,
        "key_points": key_points,
        "action_items": action_items,
    }


@router.post("/{recording_id}/generate-insights")
async def generate_insights_selective(
    recording_id: str,
    body: dict = Body(default={}),
    current_user: dict = Depends(get_current_user),
):
    """
    Generate AI insights selectively for an existing recording.

    Body: { "tasks": ["short_summary", "detailed_summary", "key_points", "action_items"] }

    Only the requested tasks are run. Other fields are left unchanged in the DB.
    Returns the newly generated values for the requested tasks.
    """
    from services.llm import (
        generate_short_summary as _gen_short,
        generate_detailed_summary as _gen_detailed,
        generate_key_points as _gen_kp,
        generate_action_items as _gen_ai,
    )

    user_id = current_user["id"]
    tasks: list[str] = body.get("tasks", ["short_summary", "detailed_summary", "key_points", "action_items"])

    ALLOWED = {"short_summary", "detailed_summary", "key_points", "action_items"}
    tasks = [t for t in tasks if t in ALLOWED]
    if not tasks:
        raise HTTPException(status_code=400, detail="No valid tasks specified.")

    # Fetch recording (include context_summary for caching)
    async with get_db() as db:
        r = await db.execute(
            text("SELECT id, transcript, raw_text, status, context_summary, context_summary_hash "
                 "FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    if rec.get("status") not in ("done", "error", "transcript_ready"):
        raise HTTPException(
            status_code=400,
            detail="Recording is still being processed. Wait until transcription finishes.",
        )

    transcript = from_json(rec["transcript"], [])
    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="No transcript found. The recording may not have been transcribed yet.",
        )

    # Filter low-confidence segments before generating insights
    transcript = _filter_high_confidence_segments(transcript, settings.MIN_AVG_SEGMENT_CONFIDENCE)
    logger.info(f"[GenerateInsights] Using {len(transcript)} high-confidence segments for {recording_id}")

    # Resolve context_summary (reuse cached or build fresh)
    raw_text = rec.get("raw_text") or ""
    current_hash = _raw_text_hash(raw_text)
    context: str | None = None
    if (
        rec.get("context_summary")
        and rec.get("context_summary_hash") == current_hash
    ):
        context = rec["context_summary"]
        logger.info(f"[GenerateInsights] Reusing cached context_summary for {recording_id}")
    else:
        logger.info(f"[GenerateInsights] Building fresh context_summary for {recording_id}")
        import asyncio as _asyncio
        _loop_ctx = _asyncio.get_running_loop()
        try:
            context = await _loop_ctx.run_in_executor(None, build_context_summary, transcript)
            if context:
                async with get_db() as db:
                    await db.execute(
                        text("UPDATE recordings SET context_summary = :ctx, context_summary_hash = :h "
                             "WHERE id = :id AND user_id = :uid"),
                        {"ctx": context, "h": current_hash, "id": recording_id, "uid": user_id},
                    )
                    await db.commit()
                logger.info(f"[GenerateInsights] context_summary stored for {recording_id} ✓")
        except Exception as ctx_err:
            logger.warning(f"[GenerateInsights] context_summary build failed (non-fatal): {ctx_err}")
            context = None

    logger.info(f"[GenerateInsights] tasks={tasks} for recording={recording_id}")

    import asyncio as _asyncio
    _loop = _asyncio.get_running_loop()

    results: dict = {}
    try:
        if "short_summary" in tasks:
            results["short_summary"] = await _loop.run_in_executor(
                None, lambda: _gen_short(transcript, context=context)
            )
        if "detailed_summary" in tasks:
            results["detailed_summary"] = await _loop.run_in_executor(
                None, lambda: _gen_detailed(transcript, context=context)
            )
        if "key_points" in tasks:
            results["key_points"] = await _loop.run_in_executor(
                None, lambda: _gen_kp(transcript, context=context)
            )
        if "action_items" in tasks:
            results["action_items"] = await _loop.run_in_executor(
                None, lambda: _gen_ai(transcript, context=context)
            )
    except Exception as e:
        logger.error(f"[GenerateInsights] Inference failed for {recording_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
    finally:
        from services.ai_provider import QwenProvider
        QwenProvider.unload_model()

    # Build DB update — only update the columns that were requested
    set_clauses = []
    params: dict = {"id": recording_id, "uid": user_id}

    if "short_summary" in results:
        set_clauses.append("short_summary = :short_summary")
        set_clauses.append("summary = :short_summary")  # backward compat
        params["short_summary"] = results["short_summary"]
    if "detailed_summary" in results:
        set_clauses.append("detailed_summary = :detailed_summary")
        params["detailed_summary"] = results["detailed_summary"]
    if "key_points" in results:
        set_clauses.append("key_points = :key_points")
        params["key_points"] = to_json(results["key_points"])
    if "action_items" in results:
        set_clauses.append("action_items = :action_items")
        params["action_items"] = to_json(results["action_items"])

    if set_clauses:
        async with get_db() as db:
            await db.execute(
                text(f"UPDATE recordings SET {', '.join(set_clauses)} WHERE id = :id AND user_id = :uid"),
                params,
            )
            await db.commit()

    logger.info(f"[GenerateInsights] Complete for {recording_id} ✓  tasks={tasks}")

    return {
        "status": "done",
        "recording_id": recording_id,
        **results,
    }


@router.patch("/{recording_id}/transcript")
async def update_transcript(
    recording_id: str,
    segment_index: int = Body(...),
    text_val: str = Body(..., alias="text"),
    current_user: dict = Depends(get_current_user),
):
    """Update a single segment in the transcript and reconstruct the words array with 1.0 probability."""
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text("SELECT transcript FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        row = r.mappings().fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Recording not found.")

        transcript = from_json(row["transcript"], [])
        if not transcript or segment_index < 0 or segment_index >= len(transcript):
            raise HTTPException(status_code=400, detail="Invalid segment index.")

        seg = transcript[segment_index]
        seg["text"] = text_val

        # Reconstruct words array for the new edited text so they remain highlighted
        words = text_val.split()
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        duration = max(0.0, seg_end - seg_start)
        
        num_words = len(words)
        word_duration = duration / num_words if num_words > 0 else 0.0
        
        seg_words = []
        for wi, word in enumerate(words):
            w_start = seg_start + wi * word_duration
            w_end = w_start + word_duration
            seg_words.append({
                "word": word,
                "start": round(w_start, 3),
                "end": round(w_end, 3),
                "probability": 1.0,  # edited words get high confidence highlight (green)
            })
        seg["words"] = seg_words

        await db.execute(
            text("UPDATE recordings SET transcript = :transcript, "
                 "context_summary = NULL, context_summary_hash = NULL "
                 "WHERE id = :id"),
            {"transcript": to_json(transcript), "id": recording_id},
        )
        await db.commit()
    logger.info(
        f"[TranscriptEdit] Invalidated context_summary for {recording_id} "
        "(transcript changed) — will rebuild on next AI request."
    )
    return {"status": "success", "segment_index": segment_index, "segment": seg}


@router.post("/{recording_id}/reidentify-speakers")
async def reidentify_speakers(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Re-run diarization + speaker identification on a completed recording.

    * Does NOT re-transcribe or re-align the audio.
    * Replaces speaker labels in the stored transcript using current voice profiles.
    * Conditionally regenerates the MoM if speaker names change.
    * The frontend can poll GET /audio/jobs/{recording_id} for progress.

    Returns immediately with status="processing"; caller must poll for completion.
    """
    import asyncio as _asyncio
    import os as _os
    import json as _json
    from tasks.rereid_pipeline import run_reidentify_pipeline, register_reid_task, is_reid_running

    user_id = current_user["id"]

    # ── Fetch recording ───────────────────────────────────────────────────────
    async with get_db() as db:
        r = await db.execute(
            text(
                "SELECT id, status, file_path, transcript "
                "FROM recordings WHERE id = :id AND user_id = :uid"
            ),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    status = rec.get("status", "")
    if status not in ("done", "transcript_ready", "error"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Recording must be fully processed before re-running speaker identification. "
                f"Current status: {status!r}"
            ),
        )

    transcript = _json.loads(rec.get("transcript") or "[]")
    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="Recording has no transcript. Please wait for transcription to complete.",
        )

    file_path = rec.get("file_path", "")
    if not file_path or not _os.path.exists(file_path):
        raise HTTPException(
            status_code=400,
            detail="Audio file is no longer available on disk. Cannot re-run speaker identification.",
        )

    # ── Guard against duplicate reid runs ─────────────────────────────────────
    if is_reid_running(recording_id):
        raise HTTPException(
            status_code=409,
            detail="Speaker re-identification is already running for this recording.",
        )

    # ── Mark as processing immediately so the frontend can poll ───────────────
    async with get_db() as db:
        await db.execute(
            text(
                "UPDATE recordings SET status='processing', progress='diarizing' "
                "WHERE id = :id"
            ),
            {"id": recording_id},
        )
        await db.commit()

    # ── Schedule the background task ─────────────────────────────────────────
    task = _asyncio.create_task(
        run_reidentify_pipeline(recording_id, file_path, user_id)
    )
    register_reid_task(recording_id, task)
    logger.info(
        f"[ReID] Re-identification task scheduled for recording={recording_id} "
        f"user={user_id}"
    )

    return {
        "status": "processing",
        "recording_id": recording_id,
        "message": "Speaker re-identification started. Poll /audio/jobs/{recording_id} for progress.",
    }


@router.post("/{recording_id}/rerun")
async def rerun_recording_pipeline(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Re-run the entire processing pipeline for an existing recording.

    - Reuses the existing recording_id, source media file, uploaded attachments, and metadata.
    - Cancels any in-flight task for this recording.
    - Clears previous transcript, diarization, MOM, AI Insights, ROM, and Video OCR outputs.
    - Re-executes the processing pipeline using the user's current application settings.
    """
    import asyncio as _asyncio
    import os as _os
    from tasks.pipeline import run_pipeline, cancel_task, register_task
    from tasks.upload_chunk_pipeline import run_upload_chunk_pipeline, UPLOAD_CHUNK_THRESHOLD_SEC

    user_id = current_user["id"]

    # 1. Fetch recording details
    async with get_db() as db:
        r = await db.execute(
            text("""
                SELECT id, filename, file_path, duration, meeting_prompt,
                       participant_voice_ids, use_vocabulary, speaker_summary,
                       source_type
                FROM recordings
                WHERE id = :id AND user_id = :uid
            """),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    file_path = rec.get("file_path", "")
    if not file_path or not _os.path.exists(file_path):
        raise HTTPException(
            status_code=400,
            detail="Source media file is no longer available on disk. Cannot re-run processing pipeline.",
        )

    # 2. Cancel existing task if running
    await cancel_task(recording_id)

    # 3. Reset outputs in DB
    async with get_db() as db:
        await db.execute(
            text("""
                UPDATE recordings SET
                    status = 'pending',
                    progress = 'queued',
                    transcript = '[]',
                    raw_text = NULL,
                    summary = NULL,
                    short_summary = NULL,
                    detailed_summary = NULL,
                    key_points = '[]',
                    action_items = '[]',
                    speakers_detected = '[]',
                    speaker_summary = NULL,
                    context_summary = NULL,
                    context_summary_hash = NULL,
                    agenda_summary = NULL,
                    agenda_summary_hash = NULL,
                    reference_summary = NULL,
                    reference_summary_hash = NULL,
                    raw_mom = NULL,
                    rom_data = NULL,
                    video_transcript = NULL,
                    transcript_embedded = 0,
                    meeting_context_embedded = 0,
                    error_message = NULL,
                    processed_at = NULL
                WHERE id = :id AND user_id = :uid
            """),
            {"id": recording_id, "uid": user_id},
        )
        try:
            await db.execute(
                text("DELETE FROM minutes_of_meeting WHERE recording_id = :id"),
                {"id": recording_id},
            )
        except Exception:
            pass  # table might not exist
        await db.commit()

    # 4. Parse metadata
    meeting_prompt = rec.get("meeting_prompt") or ""
    pv_json = rec.get("participant_voice_ids")
    participant_voice_ids = from_json(pv_json, []) if pv_json else []
    use_vocabulary = bool(rec.get("use_vocabulary"))
    speaker_summary = bool(rec.get("speaker_summary"))
    duration = float(rec.get("duration") or 0.0)
    source_type = rec.get("source_type") or "audio"

    # 5. Launch existing pipeline task
    logger.info(f"[HistoryRerun] Rerunning pipeline for recording={recording_id} (source_type={source_type}, duration={duration:.1f}s)")

    if source_type == "video":
        wav_path = file_path
        from services.video_processing_service import is_supported_video, extract_audio_from_video, extract_video_ocr_timeline
        if is_supported_video(file_path):
            try:
                wav_path = extract_audio_from_video(file_path)
            except Exception as ve:
                logger.warning(f"[HistoryRerun] Audio extraction from video failed: {ve}")

        async def _run_video_pipeline():
            try:
                ocr_task = _asyncio.create_task(extract_video_ocr_timeline(recording_id, file_path))
                if duration > UPLOAD_CHUNK_THRESHOLD_SEC:
                    await run_upload_chunk_pipeline(
                        recording_id=recording_id,
                        file_path=wav_path,
                        user_id=user_id,
                        meeting_prompt=meeting_prompt,
                        participant_voice_ids=participant_voice_ids,
                        use_vocabulary=use_vocabulary,
                        speaker_summary=speaker_summary,
                    )
                else:
                    await run_pipeline(
                        recording_id=recording_id,
                        file_path=wav_path,
                        user_id=user_id,
                        meeting_prompt=meeting_prompt,
                        participant_voice_ids=participant_voice_ids,
                        use_vocabulary=use_vocabulary,
                        speaker_summary=speaker_summary,
                    )
                await ocr_task
            except Exception as e:
                logger.error(f"[HistoryRerun] Video pipeline failed for {recording_id}: {e}", exc_info=True)

        task = _asyncio.create_task(_run_video_pipeline())
    elif duration > UPLOAD_CHUNK_THRESHOLD_SEC:
        task = _asyncio.create_task(
            run_upload_chunk_pipeline(
                recording_id=recording_id,
                file_path=file_path,
                user_id=user_id,
                meeting_prompt=meeting_prompt,
                participant_voice_ids=participant_voice_ids,
                use_vocabulary=use_vocabulary,
                speaker_summary=speaker_summary,
            )
        )
    else:
        task = _asyncio.create_task(
            run_pipeline(
                recording_id=recording_id,
                file_path=file_path,
                user_id=user_id,
                meeting_prompt=meeting_prompt,
                participant_voice_ids=participant_voice_ids,
                use_vocabulary=use_vocabulary,
                speaker_summary=speaker_summary,
            )
        )

    register_task(recording_id, task)

    return {
        "status": "pending",
        "recording_id": recording_id,
        "message": "Pipeline rerun initiated. Poll /audio/jobs/{recording_id} for progress.",
    }

