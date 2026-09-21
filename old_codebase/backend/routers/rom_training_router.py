"""
ROM Training Router
Endpoints for the ROM Training feature (Data Creation, Database, Training, History).
Completely separate from the existing Training & Optimization system.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import text

from database import get_db
from routers.auth import get_current_user
from services.rom_training.rom_training_models import (
    SaveTrainingDataRequest, StartRomTrainingRequest,
)
from services.rom_training import rom_training_dataset_service as ds
from services.rom_training import rom_training_service as ts
from config import settings as app_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rom-training", tags=["rom-training"])


def _uid(current_user: dict) -> str:
    uid = current_user.get("id") or current_user.get("user_id")
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return uid


# ── Meetings ─────────────────────────────────────────────────────────────────

@router.get("/meetings")
async def list_meetings(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """List all meetings with their ROM availability status."""
    user_id = _uid(current_user)
    meetings = await ds.get_meetings_with_rom_status(user_id, db)
    return {"meetings": meetings}


@router.get("/meetings/{recording_id}/long-rom-points")
async def get_long_rom_points(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get Long ROM points (per agenda) for a meeting."""
    user_id = _uid(current_user)
    data = await ds.get_meeting_long_rom_points(recording_id, user_id, db)
    if data is None:
        return {"available": False, "agendas": []}
    return {"available": True, **data}


# ── Data Creation ─────────────────────────────────────────────────────────────

@router.post("/extract-mom")
async def extract_mom(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a manual MoM file (PDF/DOCX/DOC/TXT).
    Extracts agenda sections and points exactly as written.
    """
    user_id = _uid(current_user)

    content = await file.read()
    filename = file.filename or "uploaded_mom"

    import tempfile, os, asyncio
    suffix = os.path.splitext(filename.lower())[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    text_content = ""
    try:
        from services.doc_extractor import extract_text_from_file
        loop = asyncio.get_running_loop()
        text_content = await loop.run_in_executor(
            None, lambda: extract_text_from_file(tmp_path, filename)
        )
    except Exception as e:
        logger.error(f"[RomTraining] extract_text_from_file failed for {filename}: {e}", exc_info=True)
        if suffix in (".txt", ".md", ".csv", ".json"):
            try:
                text_content = content.decode("utf-8", errors="replace")
            except Exception:
                raise HTTPException(status_code=400, detail=f"Could not extract text from file: {e}")
        else:
            raise HTTPException(status_code=400, detail=f"Failed to extract text from {suffix.upper() or 'uploaded'} file: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    text_content = (text_content or "").strip()
    if not text_content:
        raise HTTPException(
            status_code=400,
            detail=f"No readable text could be extracted from '{filename}'. Please make sure the file contains text or clear scanned pages.",
        )

    logger.info(f"[RomTraining] Extracted {len(text_content)} chars from '{filename}'. Extracting structure with LLM...")

    # Extract MoM structure using LLM in threadpool to avoid blocking event loop
    loop = asyncio.get_running_loop()
    agendas = await loop.run_in_executor(
        None, lambda: ds.extract_mom_points_from_text(text_content, user_id)
    )

    return {
        "agendas": agendas,
        "total_agendas": len(agendas),
        "raw_text_length": len(text_content),
    }


@router.post("/match-agendas")
async def match_agendas(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """
    Fuzzy-match extracted MoM agendas to meeting agendas.
    body: {mom_agendas: [...], meeting_agendas: [...]}
    """
    _uid(current_user)
    mom_agendas = body.get("mom_agendas", [])
    meeting_agendas = body.get("meeting_agendas", [])

    matches = ds.match_agendas(mom_agendas, meeting_agendas)
    return {"matches": matches}


@router.post("/data")
async def save_training_data(
    body: SaveTrainingDataRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Save matched training data pairs to the database."""
    user_id = _uid(current_user)
    saved_ids = await ds.save_training_data(
        user_id=user_id,
        recording_id=body.recording_id,
        recording_title=body.recording_title,
        pairs=body.pairs,
        db=db,
    )
    return {"saved": len(saved_ids), "ids": saved_ids}


@router.get("/data/meeting/{recording_id}")
async def get_data_for_meeting(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get all training data entries for a specific meeting."""
    user_id = _uid(current_user)
    data = await ds.get_training_data_for_meeting(user_id, recording_id, db)
    return {"data": data, "count": len(data)}


# ── Database Tab ──────────────────────────────────────────────────────────────

@router.get("/data")
async def list_training_data(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """List all training data in the database."""
    user_id = _uid(current_user)
    data = await ds.list_training_data(user_id, db)
    return {"data": data, "total": len(data)}


@router.delete("/data/{data_id}")
async def delete_training_data(
    data_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Delete a training data entry."""
    user_id = _uid(current_user)
    deleted = await ds.delete_training_data(data_id, user_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Training data not found.")
    return {"deleted": True}


# ── Training ──────────────────────────────────────────────────────────────────

@router.post("/variants")
async def start_training(
    body: StartRomTrainingRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Start a new ROM training job."""
    user_id = _uid(current_user)

    # Validate dataset availability
    all_data = await ds.list_training_data(user_id, db)
    if len(all_data) < body.dataset_size + body.test_size:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough training data. Have {len(all_data)} entries, need {body.dataset_size + body.test_size}.",
        )

    # Sample train + test (non-overlapping)
    import random
    shuffled = all_data.copy()
    random.shuffle(shuffled)
    test_data = shuffled[:body.test_size]
    train_data = shuffled[body.test_size:body.test_size + body.dataset_size]

    lora_config = body.lora_config.model_dump()

    # Create variant record
    variant_id = await ts.create_variant_record(
        user_id=user_id,
        name=body.name,
        description=body.description,
        base_model=body.model_path,
        training_type=body.training_type.value,
        parent_variant_id=body.parent_variant_id,
        config=lora_config,
        dataset_size=len(train_data),
        test_size=len(test_data),
        db=db,
    )

    # Start background training
    await ts.start_training(
        variant_id=variant_id,
        user_id=user_id,
        base_model=body.model_path,
        training_type=body.training_type.value,
        parent_variant_id=body.parent_variant_id,
        lora_config=lora_config,
        train_data=train_data,
        test_data=test_data,
        db_url=app_settings.DATABASE_URL,
    )

    return {
        "variant_id": variant_id,
        "status": "training",
        "train_samples": len(train_data),
        "test_samples": len(test_data),
    }


@router.get("/variants")
async def list_variants(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """List all trained variants."""
    user_id = _uid(current_user)
    variants = await ts.list_variants(user_id, db)
    return {"variants": variants}


@router.get("/variants/{variant_id}")
async def get_variant(
    variant_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get a variant's details and status."""
    user_id = _uid(current_user)
    variant = await ts.get_variant(variant_id, user_id, db)
    if not variant:
        raise HTTPException(status_code=404, detail="Variant not found.")
    return variant


@router.delete("/variants/{variant_id}")
async def delete_variant(
    variant_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Delete a variant and its adapter files."""
    user_id = _uid(current_user)
    deleted = await ts.delete_variant(variant_id, user_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Variant not found.")
    return {"deleted": True}
