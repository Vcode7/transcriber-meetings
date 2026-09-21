"""
correction_router.py - REST API for Post-Transcription Correction and Acronym Validation.

Endpoints:
  GET  /corrections/{recording_id}            -- Load saved proposals + decisions (no LLM call)
  POST /corrections/{recording_id}/run        -- Trigger detection (idempotent)
  PUT  /corrections/{recording_id}/decisions  -- Save user decisions
  POST /corrections/{recording_id}/apply      -- Apply decisions to recordings.transcript
  POST /corrections/{recording_id}/revert     -- Restore original_transcript
  GET  /corrections/acronym-dictionary        -- List user persistent dictionary
  POST /corrections/acronym-dictionary        -- Add entry to dictionary
  DELETE /corrections/acronym-dictionary/{acronym} -- Remove entry from dictionary
"""
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from sqlalchemy import text
from typing import List, Optional

from database import get_db, get_db_context, to_json, from_json
from routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/corrections", tags=["corrections"])


# =============================================================================
# Pydantic models
# =============================================================================

class CorrectionDecision(BaseModel):
    wrong: str
    correct: str
    user_correct: Optional[str] = None
    enabled: bool = True


class AcronymDecision(BaseModel):
    acronym: str
    full_form: Optional[str] = None
    user_full_form: Optional[str] = None
    is_known: bool = False
    enabled: bool = True
    not_an_acronym: bool = False
    first_occurrence: Optional[int] = None


class DecisionsPayload(BaseModel):
    correction_decisions: List[CorrectionDecision] = []
    acronym_decisions: List[AcronymDecision] = []


class DictionaryEntry(BaseModel):
    acronym: str
    full_form: str


# =============================================================================
# Helper
# =============================================================================

def _now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# GET /corrections/acronym-dictionary  (must be before /{recording_id} to avoid clash)
# =============================================================================

@router.get("/acronym-dictionary")
async def list_acronym_dictionary(
    current_user: dict = Depends(get_current_user),
):
    """Return the user persistent acronym->full-form dictionary."""
    user_id = current_user["id"]
    async with get_db_context() as db:
        r = await db.execute(
            text(
                "SELECT acronym, full_form, created_at, updated_at "
                "FROM acronym_dictionary WHERE user_id = :uid ORDER BY acronym"
            ),
            {"uid": user_id},
        )
        rows = r.mappings().fetchall()
    return [dict(row) for row in rows]


@router.post("/acronym-dictionary")
async def add_acronym_dictionary_entry(
    entry: DictionaryEntry,
    current_user: dict = Depends(get_current_user),
):
    """Add or update an acronym in the user persistent dictionary."""
    user_id = current_user["id"]
    acronym = entry.acronym.strip().upper()
    full_form = entry.full_form.strip()
    if not acronym or not full_form:
        raise HTTPException(status_code=422, detail="acronym and full_form are required.")

    now = _now_str()
    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO acronym_dictionary (id, user_id, acronym, full_form, created_at, updated_at)
                VALUES (:id, :uid, :acr, :ff, :now, :now)
                ON CONFLICT(user_id, acronym) DO UPDATE SET full_form = :ff, updated_at = :now
            """),
            {"id": str(uuid.uuid4()), "uid": user_id, "acr": acronym, "ff": full_form, "now": now},
        )
        await db.commit()
    logger.info(f"[CorrectionRouter] User {user_id} saved dictionary: {acronym} -> {full_form}")
    return {"acronym": acronym, "full_form": full_form}


@router.delete("/acronym-dictionary/{acronym}")
async def delete_acronym_dictionary_entry(
    acronym: str,
    current_user: dict = Depends(get_current_user),
):
    """Remove an acronym from the user persistent dictionary."""
    user_id = current_user["id"]
    async with get_db_context() as db:
        await db.execute(
            text("DELETE FROM acronym_dictionary WHERE user_id = :uid AND acronym = :acr"),
            {"uid": user_id, "acr": acronym.upper()},
        )
        await db.commit()
    return {"deleted": acronym}


# =============================================================================
# GET /corrections/{recording_id}
# =============================================================================

@router.get("/{recording_id}")
async def get_corrections(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Load saved correction proposals and acronym detections for a recording.
    Does NOT re-run the LLM.
    """
    user_id = current_user["id"]

    async with get_db_context() as db:
        # Verify recording belongs to this user
        rec_row = await db.execute(
            text("SELECT id, status FROM recordings WHERE id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        if not rec_row.fetchone():
            raise HTTPException(status_code=404, detail="Recording not found.")

        # Load corrections
        corr_row = await db.execute(
            text("SELECT * FROM transcript_corrections WHERE recording_id = :rid"),
            {"rid": recording_id},
        )
        corr = corr_row.mappings().fetchone()

        # Load acronyms
        acr_row = await db.execute(
            text("SELECT * FROM transcript_acronyms WHERE recording_id = :rid"),
            {"rid": recording_id},
        )
        acr = acr_row.mappings().fetchone()

    return {
        "recording_id": recording_id,
        "corrections": {
            "proposed": from_json(corr["proposed_corrections"], []) if corr else [],
            "decisions": from_json(corr["user_decisions"], []) if corr else [],
            "applied": bool(corr["applied"]) if corr else False,
            "has_data": corr is not None,
        },
        "acronyms": {
            "detected": from_json(acr["detected_acronyms"], []) if acr else [],
            "decisions": from_json(acr["user_decisions"], []) if acr else [],
            "applied": bool(acr["applied"]) if acr else False,
            "has_data": acr is not None,
        },
    }


# =============================================================================
# POST /corrections/{recording_id}/run
# =============================================================================

@router.post("/{recording_id}/run")
async def run_corrections(
    recording_id: str,
    force: bool = False,
    current_user: dict = Depends(get_current_user),
):
    """
    Trigger LLM correction detection + acronym detection for a recording.
    Idempotent: skips if already run unless force=True.
    """
    user_id = current_user["id"]

    async with get_db_context() as db:
        rec_row = await db.execute(
            text("SELECT id, status, transcript, raw_text FROM recordings WHERE id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        rec = rec_row.mappings().fetchone()
        if not rec:
            raise HTTPException(status_code=404, detail="Recording not found.")

        if rec["status"] not in ("done", "transcript_ready", "error"):
            raise HTTPException(
                status_code=400,
                detail="Recording is still processing. Wait until transcription completes.",
            )

        if not force:
            # Check if already run
            existing = await db.execute(
                text("SELECT id FROM transcript_corrections WHERE recording_id = :rid"),
                {"rid": recording_id},
            )
            if existing.fetchone():
                return {"message": "Correction detection already complete. Use force=true to re-run."}

    transcript = from_json(rec["transcript"], [])
    raw_text = rec.get("raw_text") or ""

    import asyncio
    loop = asyncio.get_running_loop()
    from services.correction_service import run_correction_pipeline

    try:
        await run_correction_pipeline(recording_id, user_id, raw_text, transcript, loop)
    except Exception as e:
        logger.error(f"[CorrectionRouter] run_corrections failed for {recording_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Correction detection failed: {str(e)}")

    return {"message": "Correction and acronym detection complete.", "recording_id": recording_id}


# =============================================================================
# PUT /corrections/{recording_id}/decisions
# =============================================================================

@router.put("/{recording_id}/decisions")
async def save_decisions(
    recording_id: str,
    payload: DecisionsPayload,
    current_user: dict = Depends(get_current_user),
):
    """Save user decisions (corrections + acronyms) without applying them yet."""
    user_id = current_user["id"]
    now = _now_str()

    async with get_db_context() as db:
        rec_row = await db.execute(
            text("SELECT id FROM recordings WHERE id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        if not rec_row.fetchone():
            raise HTTPException(status_code=404, detail="Recording not found.")

        corr_decisions_json = to_json([d.model_dump() for d in payload.correction_decisions])
        acr_decisions_json = to_json([d.model_dump() for d in payload.acronym_decisions])

        # Save correction decisions
        existing_corr = await db.execute(
            text("SELECT id FROM transcript_corrections WHERE recording_id = :rid"),
            {"rid": recording_id},
        )
        if existing_corr.fetchone():
            await db.execute(
                text("""
                    UPDATE transcript_corrections
                    SET user_decisions = :decisions, updated_at = :now
                    WHERE recording_id = :rid
                """),
                {"decisions": corr_decisions_json, "now": now, "rid": recording_id},
            )
        else:
            await db.execute(
                text("""
                    INSERT INTO transcript_corrections
                        (id, recording_id, user_id, proposed_corrections, user_decisions,
                         applied, original_transcript, created_at, updated_at)
                    VALUES (:id, :rid, :uid, '[]', :decisions, 0, NULL, :now, :now)
                """),
                {"id": str(uuid.uuid4()), "rid": recording_id, "uid": user_id,
                 "decisions": corr_decisions_json, "now": now},
            )

        # Save acronym decisions
        existing_acr = await db.execute(
            text("SELECT id FROM transcript_acronyms WHERE recording_id = :rid"),
            {"rid": recording_id},
        )
        if existing_acr.fetchone():
            await db.execute(
                text("""
                    UPDATE transcript_acronyms
                    SET user_decisions = :decisions, updated_at = :now
                    WHERE recording_id = :rid
                """),
                {"decisions": acr_decisions_json, "now": now, "rid": recording_id},
            )
        else:
            await db.execute(
                text("""
                    INSERT INTO transcript_acronyms
                        (id, recording_id, user_id, detected_acronyms, user_decisions,
                         applied, created_at, updated_at)
                    VALUES (:id, :rid, :uid, '[]', :decisions, 0, :now, :now)
                """),
                {"id": str(uuid.uuid4()), "rid": recording_id, "uid": user_id,
                 "decisions": acr_decisions_json, "now": now},
            )

        await db.commit()

    logger.info(
        f"[CorrectionRouter] Saved decisions for {recording_id}: "
        f"{len(payload.correction_decisions)} corrections, {len(payload.acronym_decisions)} acronyms."
    )
    return {"message": "Decisions saved.", "recording_id": recording_id}


# =============================================================================
# POST /corrections/{recording_id}/apply
# =============================================================================

@router.post("/{recording_id}/apply")
async def apply_corrections(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Apply enabled user decisions to recordings.transcript.
    Saves the original transcript in transcript_corrections.original_transcript for revert.
    """
    user_id = current_user["id"]
    now = _now_str()

    async with get_db_context() as db:
        rec_row = await db.execute(
            text("SELECT id, transcript FROM recordings WHERE id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        rec = rec_row.mappings().fetchone()
        if not rec:
            raise HTTPException(status_code=404, detail="Recording not found.")

        corr_row = await db.execute(
            text("SELECT * FROM transcript_corrections WHERE recording_id = :rid"),
            {"rid": recording_id},
        )
        corr = corr_row.mappings().fetchone()

        acr_row = await db.execute(
            text("SELECT * FROM transcript_acronyms WHERE recording_id = :rid"),
            {"rid": recording_id},
        )
        acr = acr_row.mappings().fetchone()

    # Load transcript
    transcript = from_json(rec["transcript"], [])
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript found for this recording.")

    original_json = rec["transcript"]  # raw JSON string -- save before modification

    # Load decisions
    corr_decisions = from_json(corr["user_decisions"] if corr else None, [])
    acr_decisions = from_json(acr["user_decisions"] if acr else None, [])

    # If no user_decisions, fall back to proposed_corrections with all enabled
    if not corr_decisions and corr:
        proposed = from_json(corr["proposed_corrections"], [])
        corr_decisions = [{"wrong": p["wrong"], "correct": p["correct"], "enabled": True} for p in proposed]

    from services.correction_service import (
        apply_corrections_to_transcript,
        apply_acronym_expansions_to_transcript,
    )

    # Apply spelling corrections
    updated_transcript, corr_count = apply_corrections_to_transcript(transcript, corr_decisions)

    # Apply acronym expansions
    updated_transcript = apply_acronym_expansions_to_transcript(updated_transcript, acr_decisions)

    updated_json = to_json(updated_transcript)

    async with get_db_context() as db:
        # Save updated transcript to recordings
        await db.execute(
            text("UPDATE recordings SET transcript = :t WHERE id = :rid"),
            {"t": updated_json, "rid": recording_id},
        )

        # Mark corrections as applied + save original for revert
        if corr:
            await db.execute(
                text("""
                    UPDATE transcript_corrections
                    SET applied = 1, original_transcript = :orig, updated_at = :now
                    WHERE recording_id = :rid
                """),
                {"orig": original_json, "now": now, "rid": recording_id},
            )
        if acr:
            await db.execute(
                text("""
                    UPDATE transcript_acronyms
                    SET applied = 1, updated_at = :now
                    WHERE recording_id = :rid
                """),
                {"now": now, "rid": recording_id},
            )

        await db.commit()

    logger.info(
        f"[CorrectionRouter] Applied corrections to {recording_id}: "
        f"{corr_count} correction(s), acronym expansions also applied."
    )
    return {
        "message": "Corrections applied to transcript.",
        "recording_id": recording_id,
        "corrections_applied": corr_count,
    }


# =============================================================================
# POST /corrections/{recording_id}/revert
# =============================================================================

@router.post("/{recording_id}/revert")
async def revert_corrections(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Restore the original transcript (before any correction was applied)."""
    user_id = current_user["id"]
    now = _now_str()

    async with get_db_context() as db:
        rec_row = await db.execute(
            text("SELECT id FROM recordings WHERE id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        if not rec_row.fetchone():
            raise HTTPException(status_code=404, detail="Recording not found.")

        corr_row = await db.execute(
            text("SELECT original_transcript, applied FROM transcript_corrections WHERE recording_id = :rid"),
            {"rid": recording_id},
        )
        corr = corr_row.mappings().fetchone()

        if not corr or not corr.get("original_transcript"):
            raise HTTPException(
                status_code=400,
                detail="No original transcript found to revert to. Apply corrections first.",
            )
        if not corr["applied"]:
            raise HTTPException(status_code=400, detail="Corrections have not been applied yet.")

        original_json = corr["original_transcript"]

        # Restore original
        await db.execute(
            text("UPDATE recordings SET transcript = :t WHERE id = :rid"),
            {"t": original_json, "rid": recording_id},
        )
        # Mark as not-applied
        await db.execute(
            text("""
                UPDATE transcript_corrections
                SET applied = 0, updated_at = :now
                WHERE recording_id = :rid
            """),
            {"now": now, "rid": recording_id},
        )
        await db.execute(
            text("""
                UPDATE transcript_acronyms
                SET applied = 0, updated_at = :now
                WHERE recording_id = :rid
            """),
            {"now": now, "rid": recording_id},
        )
        await db.commit()

    logger.info(f"[CorrectionRouter] Reverted corrections for {recording_id}.")
    return {"message": "Original transcript restored.", "recording_id": recording_id}
