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

    return {
        "id": rec["id"],
        "filename": rec.get("filename", ""),
        "duration": rec.get("duration", 0),
        "status": rec.get("status"),
        "file_path": rec.get("file_path"),
        "transcript": json.loads(rec["transcript"] or "[]"),
        "raw_text": rec.get("raw_text", ""),
        "summary": rec.get("summary", ""),
        "short_summary": rec.get("short_summary", "") or "",
        "detailed_summary": rec.get("detailed_summary", "") or "",
        "key_points": json.loads(rec["key_points"] or "[]"),
        "action_items": json.loads(rec["action_items"] or "[]"),
        "speakers_detected": json.loads(rec["speakers_detected"] or "[]"),
        "language": rec.get("language", "en"),
        "speaker_summary": json.loads(rec["speaker_summary"]) if rec.get("speaker_summary") else None,
        "created_at": rec["created_at"],
        "processed_at": rec.get("processed_at"),
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
    import os
    fp = rec.get("file_path", "")
    if not fp or not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="Audio file not found on disk.")
    return FileResponse(fp, media_type="audio/wav")


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

