"""
embedding_training_router.py — FastAPI Router for Embedding Model Domain Adaptation and Variant Management.
Prefix: /api/embedding-training
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import text

from database import get_db
from routers.auth import get_current_user
from services.embedding_training.embedding_dataset_service import (
    process_and_prepare_corpus,
    load_corpus_chunks,
)
from services.embedding_training.embedding_trainer import (
    start_domain_training,
    get_job_status,
    cancel_training_job,
)
from services.embedding_training.variant_manager import (
    discover_base_models,
    save_trained_variant,
    list_variants,
    select_variant,
    delete_variant,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/embedding-training", tags=["embedding-training"])


def _uid(current_user: dict) -> str:
    uid = current_user.get("id") or current_user.get("user_id")
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return uid


# ── Pydantic Request Schemas ──────────────────────────────────────────────────

class StartTrainingRequest(BaseModel):
    base_model_name: str
    variant_name: str
    session_id: str
    epochs: int = Field(default=3, ge=1, le=20)
    batch_size: int = Field(default=4, ge=1, le=32)
    learning_rate: float = Field(default=2e-5, ge=1e-6, le=1e-3)
    temperature: float = Field(default=0.05, ge=0.01, le=1.0)


class SaveVariantRequest(BaseModel):
    variant_name: str
    base_model_name: str
    staged_model_path: str
    document_count: int = 0
    document_names: List[str] = []
    chunk_count: int = 0
    metrics: Dict[str, Any] = {}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/base-models")
async def get_base_models(current_user: dict = Depends(get_current_user)):
    """Return list of locally available base embedding models."""
    _uid(current_user)
    models = discover_base_models()
    return {"models": models, "count": len(models)}


@router.post("/extract-documents")
async def extract_documents(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Accept multiple uploaded documents (PDF, DOC/DOCX, PPT/PPTX, TXT, MD),
    extract and normalize text, remove empty/duplicates, and return statistics.
    """
    _uid(current_user)
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    uploaded_data = []
    for f in files:
        content = await f.read()
        uploaded_data.append((f.filename or "document.txt", content, f.content_type or ""))

    session_id = str(uuid.uuid4())
    try:
        results = process_and_prepare_corpus(uploaded_data, session_id)
        return results
    except Exception as exc:
        logger.exception(f"[EmbeddingTrainingRouter] Document processing error: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {exc}")


@router.post("/train")
async def trigger_training(
    req: StartTrainingRequest,
    current_user: dict = Depends(get_current_user),
):
    """Start unsupervised domain adaptation training in the background."""
    _uid(current_user)

    # 1. Validate base model
    base_models = discover_base_models()
    selected_model = next((m for m in base_models if m["name"] == req.base_model_name), None)
    if not selected_model:
        raise HTTPException(
            status_code=404,
            detail=f"Base model '{req.base_model_name}' not found on server."
        )

    # 2. Load corpus chunks
    try:
        chunks = load_corpus_chunks(req.session_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="Corpus session not found or expired. Please re-upload documents."
        )

    if not chunks or len(chunks) < 2:
        raise HTTPException(
            status_code=400,
            detail="Extracted corpus must contain at least 2 text passages for domain training."
        )

    # 3. Launch background trainer
    try:
        job_id = start_domain_training(
            base_model_name=req.base_model_name,
            base_model_path=selected_model["path"],
            variant_name=req.variant_name.strip(),
            chunks=chunks,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            temperature=req.temperature,
        )
        return {
            "status": "started",
            "job_id": job_id,
            "variant_name": req.variant_name,
            "base_model": req.base_model_name,
            "total_chunks": len(chunks),
        }
    except Exception as exc:
        logger.exception(f"[EmbeddingTrainingRouter] Failed to start training: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not start training: {exc}")


@router.get("/train/status/{job_id}")
async def check_training_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Poll progress, current loss, logs, and status of an active training job."""
    _uid(current_user)
    status = get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")
    return status


@router.post("/train/cancel/{job_id}")
async def cancel_training(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Cancel an ongoing training job."""
    _uid(current_user)
    success = cancel_training_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Training job not found or already ended.")
    return {"status": "cancelling", "job_id": job_id}


@router.post("/variants/save")
async def save_variant(
    req: SaveVariantRequest,
    current_user: dict = Depends(get_current_user),
):
    """Save a completed trained model as an isolated embedding variant."""
    user_id = _uid(current_user)
    try:
        result = await save_trained_variant(
            user_id=user_id,
            variant_name=req.variant_name,
            base_model_name=req.base_model_name,
            staged_model_path=req.staged_model_path,
            document_count=req.document_count,
            document_names=req.document_names,
            chunk_count=req.chunk_count,
            metrics=req.metrics,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        logger.exception(f"[EmbeddingTrainingRouter] Failed to save variant: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not save variant: {exc}")


@router.get("/variants")
async def get_all_variants(current_user: dict = Depends(get_current_user)):
    """List all previously trained embedding variants."""
    user_id = _uid(current_user)
    variants = await list_variants(user_id)
    return {"variants": variants, "count": len(variants)}


@router.post("/variants/{variant_id}/select")
async def select_active_variant(
    variant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Activate an embedding variant for the RAG pipeline."""
    user_id = _uid(current_user)
    try:
        result = await select_variant(user_id, variant_id)
        return result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as exc:
        logger.exception(f"[EmbeddingTrainingRouter] Error selecting variant: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not select variant: {exc}")


@router.delete("/variants/{variant_id}")
async def delete_saved_variant(
    variant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a saved variant and remove its model files from disk."""
    user_id = _uid(current_user)
    try:
        result = await delete_variant(user_id, variant_id)
        return result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as exc:
        logger.exception(f"[EmbeddingTrainingRouter] Error deleting variant: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not delete variant: {exc}")
