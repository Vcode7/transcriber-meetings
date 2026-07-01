"""History router — list, view, and delete recordings."""
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
)

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

    # Fetch the recording
    async with get_db() as db:
        r = await db.execute(
            text("SELECT id, transcript, status FROM recordings WHERE id = :id AND user_id = :uid"),
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

    logger.info(f"[Regenerate] Starting Qwen3 4B re-summarization for {recording_id}")

    try:
        import asyncio as _asyncio
        _loop = _asyncio.get_event_loop()
        short_summary = await _loop.run_in_executor(None, generate_short_summary, transcript)
        detailed_summary = await _loop.run_in_executor(None, generate_detailed_summary, transcript)
        key_points = await _loop.run_in_executor(None, generate_key_points, transcript)
        action_items = await _loop.run_in_executor(None, generate_action_items, transcript)
    except Exception as e:
        logger.error(f"[Regenerate] Qwen3 inference failed for {recording_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

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
