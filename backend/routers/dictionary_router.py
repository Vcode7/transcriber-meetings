"""
Dictionary Router

Endpoints for:
  - Shortcut Dictionary CRUD + CSV import/export
  - Technical Vocabulary CRUD + bulk save
  - Document import (rule-based and AI-assisted extraction)
"""
import csv
import io
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from database import get_db
from routers.auth import get_current_user
from models.dictionary import (
    ShortcutCreate, ShortcutUpdate, VocabCreate, VocabBulkCreate
)
from services import dictionary_service as ds
from services.vocab_extractor import extract_rule_based, extract_ai_assisted

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dictionary", tags=["dictionary"])


# ═══════════════════════════════════════════════════════════════
# SHORTCUT DICTIONARY
# ═══════════════════════════════════════════════════════════════

@router.get("/shortcuts")
async def list_shortcuts(current_user: dict = Depends(get_current_user)):
    """List all shortcut dictionary entries for the current user."""
    user_id = current_user["id"]
    async with get_db() as db:
        items = await ds.list_shortcuts(db, user_id)
    return items


@router.post("/shortcuts", status_code=201)
async def create_shortcut(
    body: ShortcutCreate,
    current_user: dict = Depends(get_current_user),
):
    """Create a new shortcut entry."""
    user_id = current_user["id"]
    async with get_db() as db:
        item = await ds.create_shortcut(db, user_id, body.shortcut, body.full_form)
    return item


@router.put("/shortcuts/{shortcut_id}")
async def update_shortcut(
    shortcut_id: str,
    body: ShortcutUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Update an existing shortcut entry."""
    user_id = current_user["id"]
    async with get_db() as db:
        item = await ds.update_shortcut(
            db, shortcut_id, user_id,
            shortcut=body.shortcut,
            full_form=body.full_form,
        )
    if not item:
        raise HTTPException(status_code=404, detail="Shortcut not found.")
    return item


@router.delete("/shortcuts/{shortcut_id}", status_code=204)
async def delete_shortcut(
    shortcut_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a shortcut entry."""
    user_id = current_user["id"]
    async with get_db() as db:
        deleted = await ds.delete_shortcut(db, shortcut_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Shortcut not found.")


@router.get("/shortcuts/export")
async def export_shortcuts(current_user: dict = Depends(get_current_user)):
    """Export all shortcuts as a CSV file."""
    user_id = current_user["id"]
    async with get_db() as db:
        items = await ds.list_shortcuts(db, user_id)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["shortcut", "full_form"])
    writer.writeheader()
    for item in items:
        writer.writerow({"shortcut": item["shortcut"], "full_form": item["full_form"]})
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=shortcuts.csv"},
    )


@router.post("/shortcuts/import")
async def import_shortcuts(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Import shortcuts from a CSV file.
    Expected columns: shortcut, full_form (or short, full).
    Returns count of imported entries.
    """
    user_id = current_user["id"]
    data = await file.read()
    try:
        text = data.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        entries = list(reader)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CSV parse error: {e}")

    async with get_db() as db:
        count = await ds.bulk_create_shortcuts(db, user_id, entries)

    return {"imported": count, "total_in_file": len(entries)}


# ═══════════════════════════════════════════════════════════════
# TECHNICAL VOCABULARY
# ═══════════════════════════════════════════════════════════════

@router.get("/vocabulary")
async def list_vocabulary(current_user: dict = Depends(get_current_user)):
    """List all technical vocabulary words for the current user."""
    user_id = current_user["id"]
    async with get_db() as db:
        items = await ds.list_vocabulary(db, user_id)
    return items


@router.post("/vocabulary", status_code=201)
async def create_vocab(
    body: VocabCreate,
    current_user: dict = Depends(get_current_user),
):
    """Add a single technical vocabulary word."""
    user_id = current_user["id"]
    async with get_db() as db:
        item = await ds.create_vocab_word(db, user_id, body.word)
    if not item:
        raise HTTPException(status_code=409, detail="Word already exists in vocabulary.")
    return item


@router.delete("/vocabulary/{vocab_id}", status_code=204)
async def delete_vocab(
    vocab_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a technical vocabulary word."""
    user_id = current_user["id"]
    async with get_db() as db:
        deleted = await ds.delete_vocab_word(db, vocab_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Word not found.")


@router.post("/vocabulary/bulk", status_code=201)
async def bulk_create_vocab(
    body: VocabBulkCreate,
    current_user: dict = Depends(get_current_user),
):
    """Bulk add vocabulary words after user review."""
    user_id = current_user["id"]
    async with get_db() as db:
        count = await ds.bulk_create_vocabulary(db, user_id, body.words)
    return {"saved": count, "total": len(body.words)}


# ═══════════════════════════════════════════════════════════════
# DOCUMENT IMPORT
# ═══════════════════════════════════════════════════════════════

@router.post("/extract/rule-based")
async def extract_rule_based_endpoint(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a document and extract vocabulary using rule-based analysis.
    Returns extracted words for user review — does NOT save automatically.
    """
    data = await file.read()
    try:
        words = extract_rule_based(file.filename or "upload.txt", data)
    except Exception as e:
        logger.error(f"[DictionaryRouter] Rule-based extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    return {"words": words, "count": len(words)}


@router.post("/extract/ai")
async def extract_ai_endpoint(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a document and extract vocabulary using AI (QwenProvider).
    Returns {technical_words, shortcuts} for user review — does NOT save automatically.
    """
    data = await file.read()
    try:
        result = await extract_ai_assisted(file.filename or "upload.txt", data)
    except Exception as e:
        logger.error(f"[DictionaryRouter] AI extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI extraction failed: {e}")

    return result


# ═══════════════════════════════════════════════════════════════
# TRANSCRIPT EXPANSION (server-side helper)
# ═══════════════════════════════════════════════════════════════

@router.post("/expand")
async def expand_transcript(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """
    Expand abbreviations in a transcript using the user's shortcut dictionary.
    Body: {segments: [...]} — returns {segments: [...]} with expansions applied.
    """
    from services.transcript_normalizer import expand_transcript_segments

    user_id = current_user["id"]
    segments = body.get("segments", [])

    async with get_db() as db:
        shortcuts = await ds.list_shortcuts(db, user_id)

    expanded = expand_transcript_segments(segments, shortcuts)
    return {"segments": expanded}
