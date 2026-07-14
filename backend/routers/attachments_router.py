"""
attachments_router.py — Upload, list, delete, and process Agenda/Context files for MoM generation.

Endpoints:
  POST   /attachments/{recording_id}/upload   — upload one or more files (type=agenda|context)
  GET    /attachments/{recording_id}           — list all files for a recording
  DELETE /attachments/{recording_id}/{id}      — delete a specific file
  POST   /attachments/{recording_id}/process  — extract text + LLM compress -> store summaries
  GET    /attachments/{recording_id}/summaries — get current agenda/reference summaries
"""
import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import text

from database import get_db, dt_to_str, from_json
from routers.auth import get_current_user
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/attachments", tags=["attachments"])

ATTACHMENTS_SUBDIR = "attachments"
MAX_FILE_SIZE_MB = 30
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".webp",
}


def _attachment_dir(recording_id: str) -> str:
    path = os.path.join(settings.UPLOAD_DIR, ATTACHMENTS_SUBDIR, recording_id)
    os.makedirs(path, exist_ok=True)
    return path


def _compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _row_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "recording_id": row["recording_id"],
        "type": row["type"],
        "filename": row["filename"],
        "file_hash": row["file_hash"],
        "created_at": row["created_at"],
    }


# ── Upload ─────────────────────────────────────────────────────────────────────

@router.post("/{recording_id}/upload")
async def upload_attachments(
    recording_id: str,
    type: str = Form(..., description="'agenda' or 'context'"),
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload one or more agenda or context files for a recording."""
    user_id = current_user["id"]

    if type not in ("agenda", "context"):
        raise HTTPException(status_code=400, detail="type must be 'agenda' or 'context'")

    # Verify recording belongs to user
    async with get_db() as db:
        r = await db.execute(
            text("SELECT id FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        if not r.fetchone():
            raise HTTPException(status_code=404, detail="Recording not found")

    uploaded = []
    dest_dir = _attachment_dir(recording_id)

    for upload in files:
        filename = upload.filename or "upload"
        ext = os.path.splitext(filename.lower())[1]
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{ext}' for '{filename}'. "
                    f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                ),
            )

        data = await upload.read()
        if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File '{filename}' exceeds {MAX_FILE_SIZE_MB} MB limit.",
            )

        file_hash = _compute_hash(data)

        # Skip duplicate (same hash already stored for this recording+type)
        async with get_db() as db:
            r = await db.execute(
                text(
                    "SELECT id FROM recording_attachments "
                    "WHERE recording_id = :rid AND user_id = :uid "
                    "AND type = :type AND file_hash = :hash"
                ),
                {"rid": recording_id, "uid": user_id, "type": type, "hash": file_hash},
            )
            if r.fetchone():
                logger.info(f"[Attachments] Skipping duplicate file '{filename}' (hash={file_hash[:8]})")
                continue

        # Save to disk
        safe_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(dest_dir, safe_name)
        with open(file_path, "wb") as f:
            f.write(data)

        attachment_id = str(uuid.uuid4())
        now = dt_to_str(datetime.now(timezone.utc))

        async with get_db() as db:
            await db.execute(
                text(
                    "INSERT INTO recording_attachments "
                    "(id, recording_id, user_id, type, filename, file_path, file_hash, created_at) "
                    "VALUES (:id, :rid, :uid, :type, :filename, :file_path, :file_hash, :created_at)"
                ),
                {
                    "id": attachment_id,
                    "rid": recording_id,
                    "uid": user_id,
                    "type": type,
                    "filename": filename,
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "created_at": now,
                },
            )
            await db.commit()

        uploaded.append({"id": attachment_id, "filename": filename, "type": type})
        logger.info(f"[Attachments] Uploaded '{filename}' ({type}) for recording {recording_id}")

    return {"uploaded": uploaded, "skipped_duplicates": len(files) - len(uploaded)}


# ── List ───────────────────────────────────────────────────────────────────────

@router.get("/{recording_id}")
async def list_attachments(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """List all uploaded agenda/context files for a recording."""
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text(
                "SELECT * FROM recording_attachments "
                "WHERE recording_id = :rid AND user_id = :uid "
                "ORDER BY created_at ASC"
            ),
            {"rid": recording_id, "uid": user_id},
        )
        rows = r.mappings().fetchall()
    return {"files": [_row_to_dict(row) for row in rows]}


# ── Delete ─────────────────────────────────────────────────────────────────────

@router.delete("/{recording_id}/{attachment_id}")
async def delete_attachment(
    recording_id: str,
    attachment_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete an uploaded file and clear the related summary if no files remain."""
    user_id = current_user["id"]

    async with get_db() as db:
        r = await db.execute(
            text(
                "SELECT * FROM recording_attachments "
                "WHERE id = :id AND recording_id = :rid AND user_id = :uid"
            ),
            {"id": attachment_id, "rid": recording_id, "uid": user_id},
        )
        row = r.mappings().fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Attachment not found")

        att_type = row["type"]
        file_path = row["file_path"]

        # Delete DB row
        await db.execute(
            text("DELETE FROM recording_attachments WHERE id = :id"),
            {"id": attachment_id},
        )
        await db.commit()

    # Delete file from disk (non-fatal)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"[Attachments] Could not delete file {file_path}: {e}")

    # If no files of that type remain, clear the summary
    summary_col = "agenda_summary" if att_type == "agenda" else "reference_summary"
    hash_col = "agenda_summary_hash" if att_type == "agenda" else "reference_summary_hash"

    async with get_db() as db:
        r = await db.execute(
            text(
                "SELECT COUNT(*) as cnt FROM recording_attachments "
                "WHERE recording_id = :rid AND user_id = :uid AND type = :type"
            ),
            {"rid": recording_id, "uid": user_id, "type": att_type},
        )
        cnt = r.mappings().fetchone()["cnt"]
        if cnt == 0:
            await db.execute(
                text(
                    f"UPDATE recordings SET {summary_col} = NULL, {hash_col} = NULL "
                    "WHERE id = :id AND user_id = :uid"
                ),
                {"id": recording_id, "uid": user_id},
            )
            await db.commit()
            logger.info(f"[Attachments] Cleared {summary_col} for {recording_id} (no files remain)")

    return {"status": "deleted"}


# ── Process ────────────────────────────────────────────────────────────────────

@router.post("/{recording_id}/process")
async def process_attachments(
    recording_id: str,
    type: str = Form(..., description="'agenda' or 'context'"),
    current_user: dict = Depends(get_current_user),
):
    """
    Extract text from all uploaded files of the given type and run LLM compression.
    Stores the resulting summary in recordings.agenda_summary or recordings.reference_summary.
    Returns the generated summary.
    """
    user_id = current_user["id"]

    if type not in ("agenda", "context"):
        raise HTTPException(status_code=400, detail="type must be 'agenda' or 'context'")

    # Get all files of this type
    async with get_db() as db:
        r = await db.execute(
            text(
                "SELECT filename, file_path, file_hash FROM recording_attachments "
                "WHERE recording_id = :rid AND user_id = :uid AND type = :type "
                "ORDER BY created_at ASC"
            ),
            {"rid": recording_id, "uid": user_id, "type": type},
        )
        files = r.mappings().fetchall()

    if not files:
        raise HTTPException(
            status_code=400,
            detail=f"No {type} files uploaded. Please upload files first.",
        )

    # Build a combined hash of all files to detect changes
    combined_hash = hashlib.sha256(
        "|".join(row["file_hash"] for row in files).encode()
    ).hexdigest()

    # Check if we already have an up-to-date summary
    summary_col = "agenda_summary" if type == "agenda" else "reference_summary"
    hash_col = "agenda_summary_hash" if type == "agenda" else "reference_summary_hash"

    async with get_db() as db:
        r = await db.execute(
            text(f"SELECT {summary_col}, {hash_col} FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if rec and rec[summary_col] and rec[hash_col] == combined_hash:
        logger.info(f"[Attachments] Reusing cached {type} summary for {recording_id}")
        return {"summary": rec[summary_col], "cached": True}

    # Extract text from each file
    from services.doc_extractor import extract_text_from_file
    texts = []
    for row in files:
        text_content = extract_text_from_file(row["file_path"], row["filename"])
        if text_content.strip():
            texts.append(f"=== {row['filename']} ===\n{text_content}")
        else:
            logger.warning(f"[Attachments] No text extracted from '{row['filename']}'")

    if not texts:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Could not extract any text from the uploaded {type} files. "
                "Ensure the files contain readable text. "
                "For images, Tesseract-OCR must be installed on the system."
            ),
        )

    combined_text = "\n\n".join(texts)
    logger.info(
        f"[Attachments] Processing {type} files for {recording_id}: "
        f"{len(files)} file(s), {len(combined_text)} chars total"
    )

    # Run LLM compression in thread executor
    import asyncio as _asyncio
    _loop = _asyncio.get_running_loop()

    if type == "agenda":
        from services.llm import compress_agenda as _compress
    else:
        from services.llm import compress_reference as _compress

    try:
        summary = await _loop.run_in_executor(None, _compress, combined_text)
    except Exception as e:
        logger.error(f"[Attachments] LLM compression failed for {type}: {e}")
        raise HTTPException(status_code=500, detail=f"LLM compression failed: {e}")
    finally:
        from services.ai_provider import QwenProvider
        QwenProvider.unload_model()

    if not summary or not summary.strip():
        raise HTTPException(
            status_code=500,
            detail="LLM returned an empty summary. Please try again.",
        )

    # Store summary in DB
    async with get_db() as db:
        await db.execute(
            text(
                f"UPDATE recordings SET {summary_col} = :summary, {hash_col} = :hash "
                "WHERE id = :id AND user_id = :uid"
            ),
            {"summary": summary, "hash": combined_hash, "id": recording_id, "uid": user_id},
        )
        await db.commit()

    logger.info(f"[Attachments] Stored {type} summary ({len(summary)} chars) for {recording_id}")
    return {"summary": summary, "cached": False}


# ── Get summaries ──────────────────────────────────────────────────────────────

@router.get("/{recording_id}/summaries")
async def get_summaries(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return the current agenda_summary and reference_summary for a recording."""
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text(
                "SELECT agenda_summary, reference_summary FROM recordings "
                "WHERE id = :id AND user_id = :uid"
            ),
            {"id": recording_id, "uid": user_id},
        )
        row = r.mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Recording not found")

    return {
        "agenda_summary": row["agenda_summary"] or None,
        "reference_summary": row["reference_summary"] or None,
    }
