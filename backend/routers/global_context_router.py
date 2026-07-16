"""
global_context_router.py — Manage organization-wide knowledge base documents.

These documents are available to ALL meetings for a user, providing
organizational context (manuals, glossaries, technical specs, etc.)
that the LLM might not know.

Endpoints
---------
POST   /global-context/upload           — Upload org documents → embed → store
GET    /global-context/                 — List all documents (with embed status)
DELETE /global-context/{doc_id}         — Delete document + remove from FAISS
POST   /global-context/reindex          — Re-embed all documents (model change)
GET    /global-context/status           — Embedding model info + stats
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import text

from database import get_db, dt_to_str
from routers.auth import get_current_user
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/global-context", tags=["global-context"])

GLOBAL_CONTEXT_SUBDIR = "global_context"
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".webp",
    ".xlsx", ".xls", ".csv",
}


def _global_context_dir() -> str:
    """Return (and create) the directory for global context files."""
    path = os.path.join(settings.UPLOAD_DIR, GLOBAL_CONTEXT_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _row_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "filename": row["filename"],
        "file_hash": row["file_hash"],
        "embedded": bool(row["embedded"]),
        "chunk_count": row.get("chunk_count") or 0,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_global_context(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload one or more organizational documents.

    Documents are immediately extracted, chunked, embedded, and stored
    in the per-user FAISS global context index.
    """
    user_id = current_user["id"]
    dest_dir = _global_context_dir()
    uploaded = []
    skipped = 0

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

        # Skip exact duplicate (same user, same hash)
        async with get_db() as db:
            r = await db.execute(
                text(
                    "SELECT id FROM global_context_documents "
                    "WHERE user_id = :uid AND file_hash = :hash"
                ),
                {"uid": user_id, "hash": file_hash},
            )
            if r.fetchone():
                logger.info(f"[GlobalCtx] Skipping duplicate '{filename}' (hash={file_hash[:8]})")
                skipped += 1
                continue

        # Save file to disk
        safe_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(dest_dir, safe_name)
        with open(file_path, "wb") as f:
            f.write(data)

        doc_id = str(uuid.uuid4())
        now = dt_to_str(datetime.now(timezone.utc))

        async with get_db() as db:
            await db.execute(
                text(
                    "INSERT INTO global_context_documents "
                    "(id, user_id, filename, file_path, file_hash, embedded, chunk_count, created_at, updated_at) "
                    "VALUES (:id, :uid, :filename, :file_path, :file_hash, 0, 0, :now, :now)"
                ),
                {
                    "id": doc_id,
                    "uid": user_id,
                    "filename": filename,
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "now": now,
                },
            )
            await db.commit()

        # Embed in a thread executor (CPU/GPU intensive)
        import asyncio
        _loop = asyncio.get_running_loop()
        chunk_count = 0
        error_msg = None

        try:
            chunk_count = await _loop.run_in_executor(
                None,
                lambda: _embed_doc(doc_id, file_path, filename, user_id),
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[GlobalCtx] Embedding failed for '{filename}': {e}")

        # Update embedded status
        async with get_db() as db:
            await db.execute(
                text(
                    "UPDATE global_context_documents "
                    "SET embedded = :emb, chunk_count = :cc, updated_at = :now "
                    "WHERE id = :id"
                ),
                {
                    "emb": 1 if chunk_count > 0 else 0,
                    "cc": chunk_count,
                    "now": dt_to_str(datetime.now(timezone.utc)),
                    "id": doc_id,
                },
            )
            await db.commit()

        uploaded.append({
            "id": doc_id,
            "filename": filename,
            "embedded": chunk_count > 0,
            "chunk_count": chunk_count,
            "error": error_msg,
        })
        logger.info(f"[GlobalCtx] Uploaded and embedded '{filename}' ({chunk_count} chunks)")

    # Unload embedding model to free GPU memory
    if uploaded:
        import asyncio
        _loop = asyncio.get_running_loop()
        try:
            from services.text_embedding_service import unload_text_embedder
            await _loop.run_in_executor(None, unload_text_embedder)
        except Exception as e:
            logger.warning(f"[GlobalCtx] Failed to unload text embedder: {e}")

    return {"uploaded": uploaded, "skipped_duplicates": skipped}


def _embed_doc(doc_id: str, file_path: str, filename: str, user_id: str) -> int:
    """Synchronous helper: extract text → chunk → embed → store in FAISS."""
    from services.rag_pipeline import embed_global_context_doc
    return embed_global_context_doc(doc_id, file_path, filename, user_id)


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/")
async def list_global_context(
    current_user: dict = Depends(get_current_user),
):
    """List all global context documents for the current user."""
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text(
                "SELECT * FROM global_context_documents "
                "WHERE user_id = :uid ORDER BY created_at ASC"
            ),
            {"uid": user_id},
        )
        rows = r.mappings().fetchall()
    return {"documents": [_row_to_dict(row) for row in rows]}


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/{doc_id}")
async def delete_global_context_doc(
    doc_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a global context document and remove its vectors from FAISS."""
    user_id = current_user["id"]

    async with get_db() as db:
        r = await db.execute(
            text(
                "SELECT * FROM global_context_documents "
                "WHERE id = :id AND user_id = :uid"
            ),
            {"id": doc_id, "uid": user_id},
        )
        row = r.mappings().fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = row["file_path"]
        await db.execute(
            text("DELETE FROM global_context_documents WHERE id = :id"),
            {"id": doc_id},
        )
        await db.commit()

    # Remove from disk (non-fatal)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"[GlobalCtx] Could not delete file {file_path}: {e}")

    # Remove from FAISS index
    import asyncio
    _loop = asyncio.get_running_loop()
    try:
        await _loop.run_in_executor(
            None,
            lambda: _remove_doc_from_index(doc_id, user_id),
        )
    except Exception as e:
        logger.warning(f"[GlobalCtx] FAISS deletion failed for {doc_id}: {e}")

    return {"status": "deleted", "doc_id": doc_id}


def _remove_doc_from_index(doc_id: str, user_id: str) -> int:
    from services.rag_pipeline import remove_global_context_doc
    return remove_global_context_doc(doc_id, user_id)


# ── Re-index All ──────────────────────────────────────────────────────────────

@router.post("/reindex")
async def reindex_global_context(
    current_user: dict = Depends(get_current_user),
):
    """
    Re-embed all global context documents for the current user.

    Use this after changing the embedding model (QWEN_EMBEDDING_MODEL_NAME).
    All existing FAISS vectors for this user are cleared before re-embedding.
    """
    user_id = current_user["id"]

    async with get_db() as db:
        r = await db.execute(
            text(
                "SELECT id, filename, file_path FROM global_context_documents "
                "WHERE user_id = :uid ORDER BY created_at ASC"
            ),
            {"uid": user_id},
        )
        docs = r.mappings().fetchall()

    if not docs:
        return {"message": "No documents to re-index.", "processed": 0}

    # Clear the FAISS index first
    import asyncio
    _loop = asyncio.get_running_loop()
    try:
        await _loop.run_in_executor(None, lambda: _clear_user_index(user_id))
    except Exception as e:
        logger.error(f"[GlobalCtx] Failed to clear index before reindex: {e}")

    results = []
    for doc in docs:
        doc_id = doc["id"]
        file_path = doc["file_path"]
        filename = doc["filename"]

        if not os.path.exists(file_path):
            results.append({"id": doc_id, "filename": filename, "status": "file_missing"})
            continue

        chunk_count = 0
        try:
            chunk_count = await _loop.run_in_executor(
                None,
                lambda: _embed_doc(doc_id, file_path, filename, user_id),
            )
            now = dt_to_str(datetime.now(timezone.utc))
            async with get_db() as db:
                await db.execute(
                    text(
                        "UPDATE global_context_documents "
                        "SET embedded = :emb, chunk_count = :cc, updated_at = :now "
                        "WHERE id = :id"
                    ),
                    {"emb": 1 if chunk_count > 0 else 0, "cc": chunk_count, "now": now, "id": doc_id},
                )
                await db.commit()
            results.append({"id": doc_id, "filename": filename, "chunks": chunk_count, "status": "ok"})
        except Exception as e:
            results.append({"id": doc_id, "filename": filename, "status": f"error: {e}"})

    # Unload embedding model to free GPU memory
    if results:
        import asyncio
        _loop = asyncio.get_running_loop()
        try:
            from services.text_embedding_service import unload_text_embedder
            await _loop.run_in_executor(None, unload_text_embedder)
        except Exception as e:
            logger.warning(f"[GlobalCtx] Failed to unload text embedder after reindex: {e}")

    return {"processed": len(results), "results": results}


def _clear_user_index(user_id: str) -> None:
    """Clear the FAISS index for a user (called before full reindex)."""
    from services.text_embedding_service import get_text_embedder
    from services.vector_store import get_global_context_store
    embedder = get_text_embedder()
    embedder.load()
    dim = embedder.embedding_dim()
    store = get_global_context_store(user_id, dim)
    store.clear()


# ── Status ────────────────────────────────────────────────────────────────────

@router.get("/status")
async def global_context_status(
    current_user: dict = Depends(get_current_user),
):
    """Return embedding model info and document/chunk stats."""
    user_id = current_user["id"]

    async with get_db() as db:
        r = await db.execute(
            text(
                "SELECT COUNT(*) as total, "
                "SUM(embedded) as embedded_count, "
                "SUM(chunk_count) as total_chunks "
                "FROM global_context_documents WHERE user_id = :uid"
            ),
            {"uid": user_id},
        )
        row = r.mappings().fetchone()

    return {
        "embedding_model": settings.QWEN_EMBEDDING_MODEL_NAME,
        "embedding_model_dir": settings.QWEN_EMBEDDING_MODEL_DIR,
        "total_documents": row["total"] or 0,
        "embedded_documents": row["embedded_count"] or 0,
        "total_chunks": row["total_chunks"] or 0,
        "vector_store_dir": settings.VECTOR_STORE_DIR,
    }
