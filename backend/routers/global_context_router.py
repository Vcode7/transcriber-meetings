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
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import text

from database import get_db, get_db_context, dt_to_str
from routers.auth import get_current_user
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/global-context", tags=["global-context"])

GLOBAL_CONTEXT_SUBDIR = "global_context"
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md",
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
        "relative_path": row.get("relative_path") or row["filename"],
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
    relative_paths: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload one or more organizational documents or entire folder structure.

    Documents are immediately extracted, chunked, embedded, and stored
    in the per-user FAISS global context index.
    """
    user_id = current_user["id"]
    dest_dir = _global_context_dir()
    uploaded = []
    skipped_duplicates = 0
    skipped_unsupported = []

    rel_paths_list: list = []
    if relative_paths:
        try:
            rel_paths_list = json.loads(relative_paths)
        except Exception as e:
            logger.warning(f"[GlobalCtx] Failed to parse relative_paths: {e}")
            rel_paths_list = []
    else:
        logger.info(f"[GlobalCtx] No relative_paths provided in form data")

    for idx, upload in enumerate(files):
        filename = upload.filename or "upload"
        rel_path = rel_paths_list[idx] if idx < len(rel_paths_list) else filename
        ext = os.path.splitext(filename.lower())[1]

        if ext not in ALLOWED_EXTENSIONS:
            skipped_unsupported.append({
                "filename": filename,
                "relative_path": rel_path,
                "reason": f"Unsupported format '{ext}'"
            })
            logger.info(f"[GlobalCtx] Skipping unsupported file '{rel_path}'")
            continue

        data = await upload.read()
        if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
            skipped_unsupported.append({
                "filename": filename,
                "relative_path": rel_path,
                "reason": f"Exceeds {MAX_FILE_SIZE_MB}MB limit"
            })
            continue

        file_hash = _compute_hash(data)

        # Skip exact duplicate (same user, same hash)
        async with get_db_context() as db:
            r = await db.execute(
                text(
                    "SELECT id FROM global_context_documents "
                    "WHERE user_id = :uid AND file_hash = :hash"
                ),
                {"uid": user_id, "hash": file_hash},
            )
            if r.fetchone():
                logger.info(f"[GlobalCtx] Skipping duplicate '{rel_path}' (hash={file_hash[:8]})")
                skipped_duplicates += 1
                continue

        # Save file to disk.
        # When uploading a folder the browser may send upload.filename as a
        # relative path (e.g. "SubFolder/file.pdf" or "SubFolder\file.pdf").
        # Using that as-is in os.path.join() creates a path with subdirectories
        # that don't exist yet, causing FileNotFoundError.
        # We extract only the basename for the physical on-disk name; the full
        # relative path is already stored separately in rel_path for display.
        basename = os.path.basename(filename.replace("\\", "/"))
        if not basename:
            basename = filename.replace("\\", "/").replace("/", "_") or "upload"
        safe_name = f"{uuid.uuid4().hex}_{basename}"
        file_path = os.path.join(dest_dir, safe_name)
        with open(file_path, "wb") as f:
            f.write(data)

        doc_id = str(uuid.uuid4())
        now = dt_to_str(datetime.now(timezone.utc))

        async with get_db_context() as db:
            await db.execute(
                text(
                    "INSERT INTO global_context_documents "
                    "(id, user_id, filename, relative_path, file_path, file_hash, embedded, chunk_count, created_at, updated_at) "
                    "VALUES (:id, :uid, :filename, :rel_path, :file_path, :file_hash, 0, 0, :now, :now)"
                ),
                {
                    "id": doc_id,
                    "uid": user_id,
                    "filename": basename,
                    "rel_path": rel_path,
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
                lambda: _embed_doc(doc_id, file_path, basename, user_id, rel_path),
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[GlobalCtx] Embedding failed for '{rel_path}': {e}")

        # Update embedded status
        async with get_db_context() as db:
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
            "filename": basename,
            "relative_path": rel_path,
            "embedded": chunk_count > 0,
            "chunk_count": chunk_count,
            "error": error_msg,
        })
        logger.info(f"[GlobalCtx] Uploaded and embedded '{rel_path}' ({chunk_count} chunks)")

    # Unload embedding model to free GPU memory
    if uploaded:
        import asyncio
        _loop = asyncio.get_running_loop()
        try:
            from services.text_embedding_service import unload_text_embedder
            await _loop.run_in_executor(None, unload_text_embedder)
        except Exception as e:
            logger.warning(f"[GlobalCtx] Failed to unload text embedder: {e}")

    return {
        "uploaded": uploaded,
        "skipped_duplicates": skipped_duplicates,
        "skipped_unsupported": skipped_unsupported,
    }


def _embed_doc(doc_id: str, file_path: str, filename: str, user_id: str, relative_path: Optional[str] = None) -> int:
    """Synchronous helper: extract text → chunk → embed → store in FAISS."""
    from services.rag_pipeline import embed_global_context_doc
    return embed_global_context_doc(doc_id, file_path, filename, user_id, relative_path=relative_path)


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/")
async def list_global_context(
    current_user: dict = Depends(get_current_user),
):
    """List all global context documents for the current user."""
    user_id = current_user["id"]
    async with get_db_context() as db:
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

    async with get_db_context() as db:
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

    async with get_db_context() as db:
        r = await db.execute(
            text(
                "SELECT id, filename, relative_path, file_path FROM global_context_documents "
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
        rel_path = doc.get("relative_path") or filename

        if not os.path.exists(file_path):
            results.append({"id": doc_id, "filename": filename, "status": "file_missing"})
            continue

        chunk_count = 0
        try:
            chunk_count = await _loop.run_in_executor(
                None,
                lambda d_id=doc_id, f_p=file_path, f_n=filename, r_p=rel_path: _embed_doc(d_id, f_p, f_n, user_id, relative_path=r_p),
            )
            now = dt_to_str(datetime.now(timezone.utc))
            async with get_db_context() as db:
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

    async with get_db_context() as db:
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


# ── Document Detail Inspection ────────────────────────────────────────────────

@router.get("/doc/{doc_id}")
async def get_global_context_doc_detail(
    doc_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Return comprehensive inspection details for a single global context document,
    including file metadata, extracted raw text, chunks, context summary,
    keywords, and technical entities.
    """
    user_id = current_user["id"]

    async with get_db_context() as db:
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

    doc_dict = _row_to_dict(row)
    file_path = row["file_path"]
    filename = row["filename"]

    # File size
    file_size = 0
    file_exists = False
    if os.path.exists(file_path):
        file_exists = True
        file_size = os.path.getsize(file_path)

    # 1. Text extraction is deferred to on-demand request (click 'Extract Full Text')
    extracted_text_clean = ""
    extracted_preview = ""

    # 2. Retrieve stored chunks & metadata directly from ChromaDB vector store
    chunks_list = []
    unique_keywords = set()
    unique_entities = set()
    unique_acronyms = set()
    block_stats = {"paragraphs": 0, "headings": 0, "tables": 0, "lists": 0}

    try:
        from services.vector_store import get_global_context_store
        store = get_global_context_store(user_id)

        all_metas = getattr(store, "_meta", [])
        for m in all_metas:
            m_doc_id = m.get("doc_id")
            m_fn = m.get("filename")
            if m_doc_id == doc_id or (m_fn and m_fn == filename):
                txt = m.get("_text") or m.get("text") or ""
                b_type = m.get("block_type", "paragraph")
                if b_type in block_stats:
                    block_stats[b_type] += 1
                else:
                    block_stats["paragraphs"] += 1

                kw_list = m.get("keywords", [])
                if isinstance(kw_list, list):
                    for k in kw_list:
                        if k and str(k).strip():
                            unique_keywords.add(str(k).strip())

                ent_list = m.get("technical_entities", [])
                if isinstance(ent_list, list):
                    for e in ent_list:
                        if e and str(e).strip():
                            unique_entities.add(str(e).strip())

                acr_list = m.get("acronyms", [])
                if isinstance(acr_list, list):
                    for a in acr_list:
                        if a and str(a).strip():
                            unique_acronyms.add(str(a).strip())

                chunks_list.append({
                    "chunk_index": m.get("chunk_index", len(chunks_list)),
                    "total_chunks": m.get("total_chunks", 0),
                    "text": txt[:600],
                    "block_type": b_type,
                    "heading": m.get("heading"),
                    "section": m.get("section"),
                    "chapter": m.get("chapter"),
                    "page_number": m.get("page_number"),
                    "keywords": kw_list[:10] if isinstance(kw_list, list) else [],
                    "technical_entities": ent_list[:10] if isinstance(ent_list, list) else [],
                    "acronyms": acr_list[:10] if isinstance(acr_list, list) else [],
                    "dates": m.get("dates", [])[:5] if isinstance(m.get("dates"), list) else [],
                    "project_names": m.get("project_names", [])[:5] if isinstance(m.get("project_names"), list) else [],
                    "entities": m.get("entities", [])[:5] if isinstance(m.get("entities"), list) else [],
                    "numbers": m.get("numbers", [])[:5] if isinstance(m.get("numbers"), list) else [],
                    "important_terms": m.get("important_terms", [])[:10] if isinstance(m.get("important_terms"), list) else [],
                })
    except Exception as e:
        logger.warning(f"[GlobalCtx] Failed to load vector store chunks for {filename}: {e}")

    # Build context summary from first 3 stored ChromaDB chunks
    context_summary = ""
    if chunks_list:
        chunk_texts = [c["text"].strip() for c in chunks_list[:3] if c.get("text")]
        context_summary = "\n\n".join(chunk_texts)
        if len(context_summary) > 800:
            context_summary = context_summary[:800] + "..."

    return {
        "document": doc_dict,
        "file_exists": file_exists,
        "file_size": file_size,
        "extracted_text": extracted_text_clean,
        "extracted_preview": extracted_preview,
        "context_summary": context_summary,
        "keywords": sorted(list(unique_keywords)),
        "technical_entities": sorted(list(unique_entities)),
        "acronyms": sorted(list(unique_acronyms)),
        "structure_stats": block_stats,
        "chunk_count": len(chunks_list) or doc_dict.get("chunk_count", 0),
        "chunks": chunks_list,
        "pipeline_steps": [
            {"step": "1. Upload & Storage", "status": "completed" if file_exists else "failed", "detail": f"Saved at {row['relative_path'] or filename}"},
            {"step": "2. Text Extraction", "status": "completed" if doc_dict.get("embedded") else "on-demand", "detail": "Available on demand (click 'Extract Full Text')"},
            {"step": "3. Structure-Aware Chunking", "status": "completed" if (chunks_list or doc_dict.get("chunk_count", 0) > 0) else "pending", "detail": f"{len(chunks_list) or doc_dict.get('chunk_count', 0)} chunks stored in ChromaDB"},
            {"step": "4. Vector Store Embedding", "status": "completed" if doc_dict.get("embedded") else "pending", "detail": "Indexed into ChromaDB collection"},
        ]
    }


@router.post("/doc/{doc_id}/extract-text")
async def extract_global_context_doc_text(
    doc_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Perform on-demand text extraction for a global context document.
    Executed ONLY when the user explicitly clicks 'Extract Full Text'.
    """
    user_id = current_user["id"]

    async with get_db_context() as db:
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
    filename = row["filename"]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File no longer exists on disk")

    import asyncio
    _loop = asyncio.get_running_loop()

    def _do_extract():
        from services.doc_extractor import extract_text_from_file
        from services.ocr_engine import unload_ocr_engine
        txt = extract_text_from_file(file_path, filename) or ""
        try:
            unload_ocr_engine()
        except Exception:
            pass
        return txt

    extracted_text = await _loop.run_in_executor(None, _do_extract)
    clean_text = extracted_text.strip()

    return {
        "doc_id": doc_id,
        "extracted_text": clean_text,
        "extracted_preview": clean_text[:1200],
        "character_count": len(clean_text),
    }


# ── Meeting Context Hierarchy ─────────────────────────────────────────────────

@router.get("/meeting-context")
async def list_meeting_context_hierarchy(
    current_user: dict = Depends(get_current_user),
):
    """
    Return all meetings that have uploaded context files/attachments,
    structured in an expandable hierarchy:
    Meeting -> Files -> Context -> Keywords / Structured Data
    """
    user_id = current_user["id"]

    async with get_db_context() as db:
        # Fetch recordings that have attachments
        r_recs = await db.execute(
            text(
                "SELECT DISTINCT r.id, r.filename, r.created_at "
                "FROM recordings r "
                "JOIN recording_attachments a ON r.id = a.recording_id "
                "WHERE r.user_id = :uid "
                "ORDER BY r.created_at DESC"
            ),
            {"uid": user_id},
        )
        rec_rows = r_recs.mappings().fetchall()

        # Fetch all attachments for this user
        r_atts = await db.execute(
            text(
                "SELECT * FROM recording_attachments "
                "WHERE user_id = :uid ORDER BY created_at ASC"
            ),
            {"uid": user_id},
        )
        att_rows = r_atts.mappings().fetchall()

    # Group attachments by recording_id
    atts_by_recording: dict = {}
    for att in att_rows:
        rid = att["recording_id"]
        if rid not in atts_by_recording:
            atts_by_recording[rid] = []
        atts_by_recording[rid].append(att)

    from services.doc_extractor import extract_text_from_file

    meetings_list = []
    for rec in rec_rows:
        rid = rec["id"]
        rec_title = rec.get("title") or rec.get("filename") or f"Meeting {rid[:8]}"
        attachments = atts_by_recording.get(rid, [])

        files_data = []
        for att in attachments:
            file_id = att["id"]
            filename = att["filename"]
            file_path = att["file_path"]
            att_type = att["type"]

            file_exists = os.path.exists(file_path) if file_path else False
            file_size = os.path.getsize(file_path) if file_exists else 0

            # On-demand text extraction deferred to explicit request
            extracted_text = ""
            extracted_clean = ""
            summary = ""

            # Retrieve stored meeting vector chunks & metadata directly from ChromaDB
            chunks_list = []
            keywords_set = set()
            entities_set = set()
            acronyms_set = set()
            block_stats = {"paragraphs": 0, "headings": 0, "tables": 0, "lists": 0}

            try:
                from services.vector_store import get_meeting_context_store
                m_store = get_meeting_context_store(rid)

                for m in getattr(m_store, "_meta", []):
                    m_fn = m.get("filename") or m.get("document_name")
                    if not m_fn or m_fn == filename:
                        txt = m.get("_text") or m.get("text") or ""
                        b_type = m.get("block_type", "paragraph")
                        if b_type in block_stats:
                            block_stats[b_type] += 1
                        else:
                            block_stats["paragraphs"] += 1

                        for k in m.get("keywords", []):
                            if k and str(k).strip(): keywords_set.add(str(k).strip())
                        for e in m.get("technical_entities", []):
                            if e and str(e).strip(): entities_set.add(str(e).strip())
                        for a in m.get("acronyms", []):
                            if a and str(a).strip(): acronyms_set.add(str(a).strip())

                        chunks_list.append({
                            "chunk_index": m.get("chunk_index", len(chunks_list)),
                            "total_chunks": m.get("total_chunks", 0),
                            "text": txt[:400],
                            "block_type": b_type,
                            "heading": m.get("heading"),
                            "section": m.get("section"),
                            "keywords": m.get("keywords", [])[:5] if isinstance(m.get("keywords"), list) else [],
                            "technical_entities": m.get("technical_entities", [])[:5] if isinstance(m.get("technical_entities"), list) else [],
                            "acronyms": m.get("acronyms", [])[:5] if isinstance(m.get("acronyms"), list) else [],
                            "dates": m.get("dates", [])[:5] if isinstance(m.get("dates"), list) else [],
                            "project_names": m.get("project_names", [])[:5] if isinstance(m.get("project_names"), list) else [],
                        })
            except Exception:
                pass

            if not chunks_list and extracted_clean:
                paras = [p for p in extracted_clean.split("\n\n") if p.strip()]
                block_stats["paragraphs"] = len(paras)

            files_data.append({
                "id": file_id,
                "filename": filename,
                "type": att_type,
                "file_size": file_size,
                "file_exists": file_exists,
                "created_at": att["created_at"],
                "extracted_text_preview": extracted_clean[:800],
                "extracted_text_full": extracted_clean,
                "summary": summary,
                "keywords": sorted(list(keywords_set)),
                "technical_entities": sorted(list(entities_set)),
                "acronyms": sorted(list(acronyms_set)),
                "structure_stats": block_stats,
                "chunk_count": len(chunks_list),
                "chunks": chunks_list,
            })

        meetings_list.append({
            "recording_id": rid,
            "recording_title": rec_title,
            "source_type": rec.get("source_type") or "audio",
            "created_at": rec["created_at"],
            "file_count": len(files_data),
            "files": files_data,
        })

    return {"meetings": meetings_list}
