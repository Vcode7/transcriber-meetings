from __future__ import annotations

import io
import json
import logging
import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from database import get_db, dt_to_str, from_json, to_json
from routers.auth import get_current_user
from docx import Document as DocxDocument
from docx.shared import Inches

from services.rom_service import rom_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rom", tags=["rom"])

def set_fixed_table_column_widths(table, widths):
    """
    Disables table auto-fit and applies fixed column widths consistently
    across table columns and all cells in every row (header and data rows).
    """
    table.autofit = False
    table.allow_autofit = False
    for i, w in enumerate(widths):
        if i < len(table.columns):
            table.columns[i].width = w
    for row in table.rows:
        for i, w in enumerate(widths):
            if i < len(row.cells):
                row.cells[i].width = w

def add_bold_heading(doc, text: str, level: int = 0):
    """Adds a heading with bold styling explicitly set on all runs."""
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.bold = True
    return h

def add_bold_label(container, label: str, value: str):
    """
    Adds a paragraph with bold key label (e.g. 'Speakers: ') and normal text value.
    Supports both cell and doc containers.
    """
    if not value or not str(value).strip():
        return None
    para = container.add_paragraph()
    run_lbl = para.add_run(f"{label}: ")
    run_lbl.bold = True
    para.add_run(str(value).strip())
    return para

def style_table_header_bold(table):
    """Bolds all text runs in the table header row."""
    if not table.rows:
        return
    hdr_row = table.rows[0]
    for cell in hdr_row.cells:
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True

# ── Pydantic request models ───────────────────────────────────────────────────

class Stage1Request(BaseModel):
    transcript_window_minutes: float = Field(default=2.0, ge=0.5, le=30.0)

class Stage2Request(BaseModel):
    meeting_context_top_k: int = Field(default=5, ge=0, le=30)
    global_context_top_k: int = Field(default=3, ge=0, le=30)
    discussion_window_size: int = Field(default=5, ge=3, le=50)

class Stage3Request(BaseModel):
    agenda_text: Optional[str] = None
    meeting_context_top_k: int = Field(default=5, ge=0, le=30)
    global_context_top_k: int = Field(default=3, ge=0, le=30)
    force_reextract: bool = False
    previous_mom_texts: Optional[List[str]] = Field(default=None, description="Extracted text content from previous meeting MoM files (optional)")
    previous_mom_char_limit: int = Field(default=20000, ge=1000, le=100000)
    batch_size: int = Field(default=20, ge=5, le=100, description="Number of discussion points per LLM assignment batch")

class CreateAgendaRequest(BaseModel):
    agenda_text: Optional[str] = None
    meeting_context_top_k: int = Field(default=5, ge=0, le=30)
    global_context_top_k: int = Field(default=3, ge=0, le=30)
    force_reextract: bool = False
    previous_mom_texts: Optional[List[str]] = Field(default=None)
    previous_mom_char_limit: int = Field(default=20000, ge=1000, le=100000)

class GenerateFinalRomRequest(BaseModel):
    meeting_context_top_k: int = Field(default=5, ge=0, le=30)
    global_context_top_k: int = Field(default=3, ge=0, le=30)
    batch_size: int = Field(default=20, ge=5, le=100)
    include_agenda_doc_points: bool = Field(default=False)
    agendas: Optional[List[dict]] = Field(default=None, description="Optional edited agenda items from Stage 3")

class UpdateStage3AgendasRequest(BaseModel):
    agendas: List[dict]

class UpdateFinalRomRequest(BaseModel):
    final_rom: dict

class UpdateSpeakerMappingsRequest(BaseModel):
    speaker_mappings: Dict[str, str]


class GenerateAdvancedMomRequest(BaseModel):
    custom_prompt: str
    regenerate_title: bool = False
    regenerate_intro: bool = False
    regenerate_conclusion: bool = False


# ── Defensive Auth Validation Helper ──────────────────────────────────────────

def _validate_user_id(user_obj: Any) -> str:
    """
    Validates and extracts a primitive string user_id from any incoming value
    (string, user dict, Pydantic model, or user instance).
    Raises HTTPException 400 if user_id cannot be extracted or is not a non-empty string.
    Defends SQLite against 'type dict is not supported' parameter binding errors.
    """
    if user_obj is None:
        raise HTTPException(status_code=401, detail="User authentication context is missing")

    if isinstance(user_obj, str):
        uid = user_obj.strip()
    elif isinstance(user_obj, dict):
        uid = user_obj.get("id") or user_obj.get("user_id") or user_obj.get("sub")
        if isinstance(uid, str):
            uid = uid.strip()
    elif hasattr(user_obj, "id"):
        uid = getattr(user_obj, "id")
        if isinstance(uid, str):
            uid = uid.strip()
    else:
        uid = None

    if not uid or not isinstance(uid, str):
        logger.error(f"[AuthValidation] Invalid user_id parameter: type={type(user_obj).__name__}, value={user_obj}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid user_id parameter: expected string UUID, got {type(user_obj).__name__}"
        )
    return uid


# ── DB Helpers ────────────────────────────────────────────────────────────────

async def _get_recording_or_404(recording_id: str, user_id_param: Any, db) -> dict:
    user_id = _validate_user_id(user_id_param)
    result = await db.execute(
        text("SELECT * FROM recordings WHERE id = :id AND user_id = :uid"),
        {"id": recording_id, "uid": user_id}
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Recording not found")
    return dict(row._mapping)

async def _get_rom_data(recording_id: str, user_id_param: Any, db) -> dict:
    user_id = _validate_user_id(user_id_param)
    row = await _get_recording_or_404(recording_id, user_id, db)
    if row.get("rom_data"):
        return from_json(row["rom_data"])
    return {}

async def _save_rom_data(recording_id: str, user_id_param: Any, rom_data: dict, db):
    user_id = _validate_user_id(user_id_param)
    await db.execute(
        text("UPDATE recordings SET rom_data = :data WHERE id = :id AND user_id = :uid"),
        {"data": to_json(rom_data), "id": recording_id, "uid": user_id}
    )
    await db.commit()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/{recording_id}")
async def get_rom_data(recording_id: str, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    return {
        "status": "success",
        "rom_data": data
    }

@router.get("/{recording_id}/status")
async def get_rom_status(recording_id: str, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    return {
        "stage1": data.get("stage1", {}).get("status", "pending"),
        "stage2": data.get("stage2", {}).get("status", "pending"),
        "stage3": data.get("stage3", {}).get("status", "pending"),
        "final_rom": "done" if data.get("final_rom") else "pending"
    }

@router.post("/{recording_id}/stage1/generate")
async def generate_stage1(recording_id: str, req: Stage1Request, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    row = await _get_recording_or_404(recording_id, user_id, db)
    transcript = from_json(row.get("transcript") or "[]")
    
    if not transcript:
        raise HTTPException(status_code=400, detail="Recording has no transcript")

    # Fetch video OCR / frame transcript data if present
    source_type = row.get("source_type") or "audio"
    video_transcript = None
    raw_vt = row.get("video_transcript")
    if raw_vt:
        video_transcript = from_json(raw_vt)
        logger.info(
            f"[ROM Router] Stage 1: video transcript loaded, "
            f"{len(video_transcript)} OCR blocks loaded for recording={recording_id}"
        )
    elif source_type == "video":
        logger.info(
            f"[ROM Router] Stage 1: video recording but no OCR data yet for recording={recording_id}"
        )
        
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        lambda: rom_service.extract_discussion_points(
            transcript,
            req.transcript_window_minutes,
            user_id,
            video_transcript=video_transcript,
            source_type=source_type,
        )
    )
    
    data = await _get_rom_data(recording_id, user_id, db)
    data["stage1"] = {
        "status": "done",
        "transcript_window_minutes": req.transcript_window_minutes,
        "discussion_points": result.get("discussion_points", []),
        "windows_processed": result.get("windows_processed", 0),
        "video_ocr_blocks_used": len(video_transcript) if video_transcript else 0,
        "source_type": source_type,
    }
    
    # Reset subsequent stages
    if "stage2" in data: del data["stage2"]
    if "stage3" in data: del data["stage3"]
    if "final_rom" in data: del data["final_rom"]
    
    await _save_rom_data(recording_id, user_id, data, db)
    return {
        "status": "success",
        "stage1": data["stage1"],
        "rom_data": data
    }


def _extract_transcript_snippet(transcript: list, start_time: float, end_time: float) -> str:
    """Reconstruct transcript text snippet for a given timeline range [start_time, end_time]."""
    lines = []
    for seg in transcript:
        s = float(seg.get("start", 0.0) or 0.0)
        e = float(seg.get("end", 0.0) or 0.0)
        if (end_time <= 0 and start_time <= 0) or (s <= end_time and e >= start_time):
            spk = seg.get("speaker", "Unknown")
            txt = seg.get("text", "").strip()
            lines.append(f"[{s:.1f}s-{e:.1f}s] {spk}: {txt}")
    return "\n".join(lines)

@router.get("/{recording_id}/stage1/download/docx")
async def download_stage1_docx(recording_id: str, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    row = await _get_recording_or_404(recording_id, user_id, db)
    transcript = from_json(row.get("transcript") or "[]")
    data = await _get_rom_data(recording_id, user_id, db)
    points = data.get("stage1", {}).get("discussion_points", [])
    
    doc = DocxDocument()
    add_bold_heading(doc, "ROM Stage 1: Extracted Discussion Points & Source Transcripts", 0)
    
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Original Transcript'
    hdr_cells[1].text = 'Extracted Discussion Point & Fields'
    style_table_header_bold(table)

    from collections import OrderedDict
    grouped_points = OrderedDict()

    for p in points:
        raw_text = p.get("raw_transcript_text")

        if not raw_text and transcript:
            t_start = float(p.get("timeline_start", 0.0) or 0.0)
            t_end = float(p.get("timeline_end", 0.0) or 0.0)
            raw_text = _extract_transcript_snippet(transcript, t_start, t_end)

        video_context = p.get("video_transcript_context")

        if raw_text:
            group_key = raw_text.strip()
        else:
            group_key = f"{p.get('timeline_start', 0.0)}-{p.get('timeline_end', 0.0)}"

        if group_key not in grouped_points:
            grouped_points[group_key] = {
                "raw_text": raw_text,
                "video_context": video_context,
                "timeline_start": p.get("timeline_start"),
                "timeline_end": p.get("timeline_end"),
                "points": []
            }

        grouped_points[group_key]["points"].append(p)

    for group in grouped_points.values():
        row_cells = table.add_row().cells

        cell0 = row_cells[0]
        cell0.text = ""
        if group["raw_text"]:
            add_bold_label(cell0, "[Audio Transcript]", group['raw_text'].strip())
        if group["video_context"]:
            add_bold_label(cell0, "[Video Transcript / Frame OCR]", group['video_context'].strip())
        if not cell0.text.strip():
            add_bold_label(cell0, "Timeline", f"{group['timeline_start']}s - {group['timeline_end']}s")

        cell1 = row_cells[1]
        cell1.text = ""

        field_labels = {
            "speakers": "Speakers",
            "action_owner": "Action Owner",
            "decisions": "Decisions",
            "action_items": "Action Items",
            "technical_terms": "Technical Terms",
            "dates": "Dates",
            "numbers": "Numbers/Quantities",
            "project_names": "Project Names",
            "references": "References",
            "questions": "Questions Raised",
            "required_information": "Required Information"
        }

        for idx, point in enumerate(group["points"], start=1):
            if idx > 1:
                cell1.add_paragraph()  # Blank line between points

            if point.get("discussion_point"):
                add_bold_label(cell1, f"Discussion Point {idx}", point["discussion_point"])

            for key, label in field_labels.items():
                val = point.get(key)
                if not val:
                    continue
                if isinstance(val, list):
                    if len(val) == 0: continue
                    value = ", ".join(str(v) for v in val)
                else:
                    value = str(val).strip()
                    if not value: continue
                add_bold_label(cell1, label, value)

    f = io.BytesIO()
    doc.save(f)
    f.seek(0)
    
    return StreamingResponse(
        f,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename=rom_stage1_{recording_id}.docx"}
    )

@router.post("/{recording_id}/stage2/generate")
async def generate_stage2(recording_id: str, req: Stage2Request, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    points = data.get("stage1", {}).get("discussion_points", [])
    
    if not points:
        raise HTTPException(status_code=400, detail="Stage 1 must be completed first")
        
    loop = asyncio.get_event_loop()
    polished = await loop.run_in_executor(
        None,
        lambda: rom_service.enhance_discussion_points(
            points, recording_id, user_id,
            req.meeting_context_top_k, req.global_context_top_k,
            req.discussion_window_size
        )
    )
    
    data["stage2"] = {
        "status": "done",
        "polished_points": polished
    }
    
    # Reset subsequent stages
    if "stage3" in data: del data["stage3"]
    if "final_rom" in data: del data["final_rom"]
    
    await _save_rom_data(recording_id, user_id, data, db)
    return {
        "status": "success",
        "stage2": data["stage2"],
        "rom_data": data
    }

@router.get("/{recording_id}/stage2/download/docx")
async def download_stage2_docx(recording_id: str, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    
    stage1_points = data.get("stage1", {}).get("discussion_points", [])
    stage1_map = {p.get("id"): p for p in stage1_points}
    points = data.get("stage2", {}).get("polished_points", [])
    
    doc = DocxDocument()
    add_bold_heading(doc, "ROM Stage 2: Enhanced Discussion Points & RAG Context", 0)
    
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Input Discussion Point (Stage 1)'
    hdr_cells[1].text = 'Retrieved Context (RAG)'
    hdr_cells[2].text = 'Enhanced Discussion Point (Stage 2)'
    style_table_header_bold(table)
    
    for p in points:
        row_cells = table.add_row().cells

        # Col 0: Input Stage 1 Point(s)
        col0_cell = row_cells[0]
        col0_cell.text = ""

        raw_ids = p.get("original_point_ids")
        if isinstance(raw_ids, str):
            orig_ids = [raw_ids]
        elif isinstance(raw_ids, list):
            orig_ids = raw_ids
        else:
            orig_ids = [p.get("original_point_id")] if p.get("original_point_id") else []
            
        matched_orig = [stage1_map[oid] for oid in orig_ids if oid in stage1_map]
        
        if matched_orig:
            for idx, orig_p in enumerate(matched_orig, 1):
                if len(matched_orig) > 1:
                    add_bold_label(col0_cell, "Input Point", f"#{idx}")
                add_bold_label(col0_cell, "Discussion Point", orig_p.get('discussion_point', ''))
                add_bold_label(col0_cell, "Timeline", f"{orig_p.get('timeline_start', 0.0)}s - {orig_p.get('timeline_end', 0.0)}s")
                if orig_p.get("speakers"):
                    spk_val = ", ".join(orig_p['speakers']) if isinstance(orig_p['speakers'], list) else str(orig_p['speakers'])
                    add_bold_label(col0_cell, "Speakers", spk_val)
                if orig_p.get("video_transcript_context"):
                    add_bold_label(col0_cell, "Video Transcript / OCR Context", orig_p['video_transcript_context'])
        else:
            add_bold_label(col0_cell, "Original Point IDs", ", ".join(orig_ids) if orig_ids else "N/A")
            add_bold_label(col0_cell, "Timeline", f"{p.get('timeline_start', 0.0)}s - {p.get('timeline_end', 0.0)}s")
            if p.get("speakers"):
                spk_val = ", ".join(p['speakers']) if isinstance(p['speakers'], list) else str(p['speakers'])
                add_bold_label(col0_cell, "Speakers", spk_val)

        # Col 1: Context Retrieved
        col1_cell = row_cells[1]
        col1_cell.text = ""
        retrieved = p.get("retrieved_context", {})
        meeting_chunks = retrieved.get("meeting_chunks", []) if isinstance(retrieved, dict) else []
        global_chunks = retrieved.get("global_chunks", []) if isinstance(retrieved, dict) else []

        if meeting_chunks:
            m_texts = [ (c.get("text") or c.get("content") or c.get("chunk") or "").strip() for c in meeting_chunks ]
            m_texts = [ t for t in m_texts if t ]
            if m_texts:
                add_bold_label(col1_cell, "Meeting Context", "\n\n".join(m_texts))

        if global_chunks:
            g_texts = [ (c.get("text") or c.get("content") or c.get("chunk") or "").strip() for c in global_chunks ]
            g_texts = [ t for t in g_texts if t ]
            if g_texts:
                add_bold_label(col1_cell, "Global Context", "\n\n".join(g_texts))

        if not col1_cell.text.strip():
            col1_cell.text = "No additional context retrieved."

        # Col 2: Enhanced Discussion Point (Stage 2)
        col2_cell = row_cells[2]
        col2_cell.text = ""
        add_bold_label(col2_cell, "Enhanced Point", p.get('polished_text', ''))

        field_labels = {
            "speakers": "Speakers",
            "action_owner": "Action Owner",
            "decisions": "Decisions",
            "action_items": "Action Items",
            "technical_terms": "Technical Terms",
            "dates": "Dates",
            "numbers": "Numbers/Quantities",
            "references": "References"
        }

        for key, label in field_labels.items():
            val = p.get(key)
            if val:
                if isinstance(val, list) and len(val) > 0:
                    add_bold_label(col2_cell, label, ", ".join(str(v) for v in val))
                elif isinstance(val, str) and val.strip():
                    add_bold_label(col2_cell, label, val)

    f = io.BytesIO()
    doc.save(f)
    f.seek(0)
    
    return StreamingResponse(
        f,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename=rom_stage2_{recording_id}.docx"}
    )

@router.post("/{recording_id}/stage3/upload-agenda")
async def upload_agenda(recording_id: str, file: UploadFile = File(...), current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    await _get_recording_or_404(recording_id, user_id, db)
    content = await file.read()
    text_content = content.decode("utf-8", errors="ignore")
    return {"agenda_text": text_content}

@router.post("/{recording_id}/stage3/upload-previous-mom")
async def upload_previous_mom(recording_id: str, file: UploadFile = File(...), current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    """Upload a previous meeting MoM file. Returns extracted text. Call separately for each MoM file."""
    user_id = _validate_user_id(current_user)
    await _get_recording_or_404(recording_id, user_id, db)
    
    # Size check: Max 10MB limit
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size exceeds the maximum limit of 10MB.")

    # Extract text content dynamically using doc_extractor
    import tempfile
    import os as _os
    from services.doc_extractor import extract_text_from_file

    filename = file.filename or "upload"
    suffix = _os.path.splitext(filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import asyncio
        _loop = asyncio.get_running_loop()
        text_content = await _loop.run_in_executor(
            None, lambda: extract_text_from_file(tmp_path, filename)
        )
    except Exception as e:
        logger.error(f"[ROM Router] Previous MoM text extraction failed for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")
    finally:
        try:
            _os.unlink(tmp_path)
        except Exception:
            pass

    return {"mom_text": text_content, "filename": filename}


@router.post("/{recording_id}/stage3/create-agenda")
async def create_agenda(recording_id: str, req: CreateAgendaRequest, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    """Step 1 of Stage 3: Generate agendas only. Does NOT map discussion points."""
    user_id = _validate_user_id(current_user)
    row = await _get_recording_or_404(recording_id, user_id, db)
    transcript = from_json(row.get("transcript") or "[]")
    data = await _get_rom_data(recording_id, user_id, db)

    if req.previous_mom_texts and len(req.previous_mom_texts) > 5:
        raise HTTPException(status_code=400, detail="Maximum of 5 previous MoM files can be uploaded.")

    previous_mom_texts = []
    if req.previous_mom_texts:
        limit = req.previous_mom_char_limit
        previous_mom_texts = [t[:limit] for t in req.previous_mom_texts]

    if not data.get("stage2", {}).get("polished_points"):
        raise HTTPException(status_code=400, detail="Stage 2 must be completed before creating agendas")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: rom_service.generate_agendas_only(
            agenda_text=req.agenda_text,
            recording_id=recording_id,
            user_id=user_id,
            meeting_context_top_k=req.meeting_context_top_k,
            global_context_top_k=req.global_context_top_k,
            force_reextract=req.force_reextract,
            transcript=transcript,
            previous_mom_texts=previous_mom_texts,
        )
    )

    # Store agendas in stage3 (preserve point mappings if they exist)
    existing_s3 = data.get("stage3", {})
    data["stage3"] = {
        "status": "agendas_ready",
        "agendas": result.get("agendas", []),
        "expanded_agendas": result.get("expanded_agendas", []),
        "retrieved_context_chunks": result.get("retrieved_context_chunks", {}),
        # Preserve existing doc points if any
        "agenda_doc_points": existing_s3.get("agenda_doc_points", {}),
        # Reset mapping data since agendas changed
        "candidate_results": [],
        "batch_assignments": [],
        "point_mappings": {},
        "agenda_groups": {},
        "similarity_matrix": [],
    }
    # Reset final_rom since agendas changed
    if "final_rom" in data:
        del data["final_rom"]

    await _save_rom_data(recording_id, user_id, data, db)
    return {
        "status": "success",
        "stage3": data["stage3"],
        "rom_data": data
    }


@router.post("/{recording_id}/stage3/agenda/{agenda_id}/upload-document")
async def upload_agenda_document(
    recording_id: str,
    agenda_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Upload a supporting document for a specific agenda item. Extracts 2-5 factual points via LLM."""
    user_id = _validate_user_id(current_user)
    await _get_recording_or_404(recording_id, user_id, db)
    data = await _get_rom_data(recording_id, user_id, db)

    agendas = data.get("stage3", {}).get("agendas", [])
    agenda = next((a for a in agendas if a.get("agenda_id") == agenda_id), None)
    if not agenda:
        raise HTTPException(status_code=404, detail=f"Agenda {agenda_id} not found. Please create agendas first.")

    # Extract file text
    try:
        content = await file.read()
        # Try UTF-8 decode for plain text; for binary formats, use raw-mom extract endpoint
        try:
            text_content = content.decode("utf-8", errors="replace")
        except Exception:
            text_content = content.decode("latin-1", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if not text_content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file appears to be empty or binary-only. Please upload a text-extractable document.")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: rom_service.extract_agenda_document_points(
            agenda_id=agenda_id,
            agenda_title=agenda.get("title", ""),
            agenda_description=agenda.get("description", ""),
            document_text=text_content,
        )
    )

    # Save doc points into stage3.agenda_doc_points
    doc_points_store = data.get("stage3", {}).get("agenda_doc_points", {})
    presenter = result.get("presenter")
    doc_points_store[agenda_id] = {
        "doc_name": file.filename,
        "points": result.get("points", []),
        "presenter": presenter,
    }
    if "stage3" not in data:
        data["stage3"] = {}
    data["stage3"]["agenda_doc_points"] = doc_points_store

    # Also update presenter/speaker on the agenda item if found
    if presenter:
        for a in data["stage3"].get("agendas", []):
            if a.get("agenda_id") == agenda_id:
                a["presenter"] = presenter
                a["speaker"] = presenter

    await _save_rom_data(recording_id, user_id, data, db)
    return {
        "status": "success",
        "agenda_id": agenda_id,
        "doc_name": file.filename,
        "points": result.get("points", []),
        "points_count": len(result.get("points", [])),
        "presenter": presenter,
    }


@router.put("/{recording_id}/stage3/agendas")
async def update_stage3_agendas(
    recording_id: str,
    req: UpdateStage3AgendasRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Save edited Stage 3 agendas."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    if "stage3" not in data:
        data["stage3"] = {}
    data["stage3"]["agendas"] = req.agendas
    await _save_rom_data(recording_id, user_id, data, db)

    # Synchronize edited agendas to parsed_agenda_json in the recordings table
    from services.rag_pipeline import _save_parsed_agenda
    parsed_items = []
    for a in req.agendas:
        parsed_items.append({
            "topic": a.get("title") or a.get("topic") or "",
            "speaker": a.get("speaker") or a.get("presenter"),
            "details": a.get("description") or a.get("details") or "",
            "keywords": a.get("keywords") or [],
            "related_concepts": a.get("related_concepts") or [],
            "alternative_terminology": a.get("alternative_terminology") or [],
            "expected_themes": a.get("expected_themes") or [],
        })
    _save_parsed_agenda(recording_id, user_id, parsed_items)

    return {"status": "success", "agendas": req.agendas}


@router.post("/{recording_id}/stage3/generate-final-rom")
async def generate_final_rom_from_agendas(
    recording_id: str,
    req: GenerateFinalRomRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Step 2 of Stage 3: Map discussion points to agendas and generate Final ROM.
    Requires agendas to be created first via /stage3/create-agenda.
    Optionally includes agenda document points."""
    user_id = _validate_user_id(current_user)
    await _get_recording_or_404(recording_id, user_id, db)
    data = await _get_rom_data(recording_id, user_id, db)

    polished_points = data.get("stage2", {}).get("polished_points", [])
    if not polished_points:
        raise HTTPException(status_code=400, detail="Stage 2 must be completed first")

    stage3_data = data.get("stage3", {})
    if req.agendas:
        stage3_data["agendas"] = req.agendas
        data["stage3"]["agendas"] = req.agendas

    agendas = stage3_data.get("agendas", [])
    if not agendas:
        raise HTTPException(status_code=400, detail="Agendas must be created first. Run /stage3/create-agenda first.")

    expanded_agendas = stage3_data.get("expanded_agendas", [])
    agenda_doc_points = stage3_data.get("agenda_doc_points", {}) if req.include_agenda_doc_points else {}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: rom_service.map_points_to_agendas(
            polished_points=polished_points,
            agendas=agendas,
            expanded_agendas=expanded_agendas,
            recording_id=recording_id,
            user_id=user_id,
            meeting_context_top_k=req.meeting_context_top_k,
            global_context_top_k=req.global_context_top_k,
            batch_size=req.batch_size,
            include_agenda_doc_points=req.include_agenda_doc_points,
            agenda_doc_points=agenda_doc_points,
        )
    )

    # Update stage3 with mapping results
    data["stage3"].update({
        "status": "done",
        "candidate_results": result.get("candidate_results", []),
        "batch_assignments": result.get("batch_assignments", []),
        "point_mappings": result.get("point_mappings", {}),
        "agenda_groups": result.get("agenda_groups", {}),
        "similarity_matrix": result.get("similarity_matrix", []),
    })

    from services.rom_service import apply_speaker_mappings_to_final_rom

    raw_final = {
        "agendas": result.get("final_rom_agendas", []),
        "include_agenda_doc_points": req.include_agenda_doc_points,
        "speaker_mappings": data.get("final_rom", {}).get("speaker_mappings", {}),
    }
    data["final_rom"] = apply_speaker_mappings_to_final_rom(raw_final)

    await _save_rom_data(recording_id, user_id, data, db)
    return {
        "status": "success",
        "stage3": data["stage3"],
        "final_rom": data["final_rom"],
        "rom_data": data
    }

@router.post("/{recording_id}/stage3/generate")
async def generate_stage3(recording_id: str, req: Stage3Request, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    row = await _get_recording_or_404(recording_id, user_id, db)
    transcript = from_json(row.get("transcript") or "[]")
    data = await _get_rom_data(recording_id, user_id, db)
    polished_points = data.get("stage2", {}).get("polished_points", [])

    if req.previous_mom_texts and len(req.previous_mom_texts) > 5:
        raise HTTPException(status_code=400, detail="Maximum of 5 previous MoM files can be uploaded.")

    previous_mom_texts = []
    if req.previous_mom_texts:
        limit = req.previous_mom_char_limit
        previous_mom_texts = [t[:limit] for t in req.previous_mom_texts]

    if not polished_points:
        raise HTTPException(status_code=400, detail="Stage 2 must be completed first")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: rom_service.generate_agendas_and_map(
            polished_points=polished_points,
            agenda_text=req.agenda_text,
            recording_id=recording_id,
            user_id=user_id,
            meeting_context_top_k=req.meeting_context_top_k,
            global_context_top_k=req.global_context_top_k,
            force_reextract=req.force_reextract,
            transcript=transcript,
            previous_mom_texts=previous_mom_texts,
            batch_size=req.batch_size,
        )
    )

    data["stage3"] = {
        "status": "done",
        "agendas": result.get("agendas", []),
        "expanded_agendas": result.get("expanded_agendas", []),
        "candidate_results": result.get("candidate_results", []),
        "batch_assignments": result.get("batch_assignments", []),
        "point_mappings": result.get("point_mappings", {}),
        "agenda_groups": result.get("agenda_groups", {}),
        "similarity_matrix": result.get("similarity_matrix", []),
    }

    # Auto-generate draft final ROM (uses preserved point_mappings — backward compat)
    final_agendas = []
    mappings = result.get("point_mappings", {})
    # Build assignment lookup for is_probable flag
    assign_map: dict = {a["point_id"]: a for a in result.get("batch_assignments", []) if a.get("point_id")}

    for a in result.get("agendas", []):
        agenda_points = []
        for p in polished_points:
            if mappings.get(p.get("id")) == a.get("agenda_id"):
                p_copy = dict(p)
                p_copy["text"] = p.get("polished_text") or p.get("text", "")
                assignment = assign_map.get(p.get("id"), {})
                p_copy["assignment_confidence"] = assignment.get("confidence", "medium")
                p_copy["assignment_reason"]     = assignment.get("reason", "")
                p_copy["is_probable"]           = assignment.get("is_probable", False)
                agenda_points.append(p_copy)

        if agenda_points:
            agenda_copy = dict(a)
            agenda_copy["discussion_points"] = agenda_points
            final_agendas.append(agenda_copy)

    data["final_rom"] = {"agendas": final_agendas}

    await _save_rom_data(recording_id, user_id, data, db)
    return {
        "status": "success",
        "stage3": data["stage3"],
        "final_rom": data["final_rom"],
        "rom_data": data
    }

@router.get("/{recording_id}/stage3/download/docx")
async def download_stage3_docx(recording_id: str, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    
    stage2_points = data.get("stage2", {}).get("polished_points", [])
    stage3_data = data.get("stage3", {})
    agendas = stage3_data.get("agendas", [])
    matrix = stage3_data.get("similarity_matrix", [])
    mappings = stage3_data.get("point_mappings", {})
    
    doc = DocxDocument()
    add_bold_heading(doc, "ROM Stage 3: Point ↔ Agenda Similarity Matrix", 0)
    
    if stage2_points and agendas and matrix:
        num_agendas = len(agendas)
        num_cols = 2 + num_agendas  # Code (P1..PN), A1..AN, Best Match

        table = doc.add_table(rows=1, cols=num_cols)
        table.style = 'Table Grid'

        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Point Code'
        for j in range(num_agendas):
            hdr_cells[1 + j].text = f"A{j+1}"
        hdr_cells[1 + num_agendas].text = 'Best Match Agenda'
        style_table_header_bold(table)

        for i, p in enumerate(stage2_points):
            row_cells = table.add_row().cells
            row_cells[0].text = f"P{i+1}"

            p_id = p.get("id", "")
            best_a_id = mappings.get(p_id)

            row_scores = matrix[i] if i < len(matrix) else []
            best_idx = 0
            best_score = -1.0

            for j in range(num_agendas):
                score = float(row_scores[j]) if j < len(row_scores) else 0.0
                a_id = agendas[j].get("agenda_id", f"A{j+1}")
                is_match = (a_id == best_a_id)

                if score > best_score:
                    best_score = score
                    best_idx = j

                cell_text = f"{score:.3f}"
                if is_match:
                    cell_text += " [MATCH]"
                row_cells[1 + j].text = cell_text

            row_cells[1 + num_agendas].text = f"A{best_idx+1} ({best_score:.3f})"

        # Page 2 Onwards: Reference Guide
        doc.add_page_break()
        add_bold_heading(doc, "Agenda & Discussion Point Reference Guide", level=1)
        
        add_bold_heading(doc, "Agenda Definitions (A1, A2...)", level=2)
        for j, a in enumerate(agendas):
            a_code = f"A{j+1}"
            a_title = a.get('title', 'Untitled Agenda')
            p_head = doc.add_paragraph()
            r_code = p_head.add_run(f"{a_code} = {a_title}")
            r_code.bold = True
            if a.get('description'):
                add_bold_label(doc, "Description", a['description'])
            if a.get('keywords'):
                kw_val = ", ".join(a['keywords']) if isinstance(a['keywords'], list) else str(a['keywords'])
                add_bold_label(doc, "Keywords", kw_val)

        add_bold_heading(doc, "Discussion Point Reference List (P1, P2...)", level=2)
        for i, p in enumerate(stage2_points):
            p_code = f"P{i+1}"
            p_para = doc.add_paragraph()
            r_pcode = p_para.add_run(f"{p_code}: ")
            r_pcode.bold = True
            p_para.add_run(p.get('polished_text', ''))
            if p.get('speakers'):
                spk_val = ", ".join(p['speakers']) if isinstance(p['speakers'], list) else str(p['speakers'])
                add_bold_label(doc, "Speakers", spk_val)
    else:
        doc.add_paragraph("No Stage 3 similarity matrix data available.")
        
    f = io.BytesIO()
    doc.save(f)
    f.seek(0)
    
    return StreamingResponse(
        f,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename=rom_stage3_similarity_{recording_id}.docx"}
    )

@router.put("/{recording_id}/speaker-mappings")
async def update_speaker_mappings(
    recording_id: str,
    req: UpdateSpeakerMappingsRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    user_id = _validate_user_id(current_user)
    from services.speaker_sync import sync_global_speaker_rename
    updated_rec = await sync_global_speaker_rename(db, recording_id, user_id, req.speaker_mappings, replace_all=True)
    return {"status": "success", "speaker_mappings": updated_rec.get("speaker_mappings", {})}

@router.put("/{recording_id}/final")
async def update_final_rom(recording_id: str, req: UpdateFinalRomRequest, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    final_rom = req.final_rom or {}
    spk_mappings = final_rom.get("speaker_mappings", {})
    if isinstance(spk_mappings, dict):
        from services.speaker_sync import sync_global_speaker_rename
        await sync_global_speaker_rename(db, recording_id, user_id, spk_mappings, replace_all=True)
    data = await _get_rom_data(recording_id, user_id, db)
    from services.rom_service import apply_speaker_mappings_to_final_rom
    data["final_rom"] = apply_speaker_mappings_to_final_rom(final_rom)
    await _save_rom_data(recording_id, user_id, data, db)
    return {"status": "success", "final_rom": data["final_rom"]}

@router.get("/{recording_id}/final/download/docx")
async def download_final_docx(recording_id: str, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    from services.rom_service import apply_speaker_mappings_to_final_rom
    if data.get("final_rom"):
        data["final_rom"] = apply_speaker_mappings_to_final_rom(data["final_rom"])
    raw_agendas = data.get("final_rom", {}).get("agendas", [])

    # Hide agendas with zero points in final output
    agendas = [a for a in raw_agendas if a.get("discussion_points")]

    # Check if this meeting uses default single "General Discussion" / "No Agenda"
    is_single_default = False
    if len(raw_agendas) == 1:
        stitle = (raw_agendas[0].get("title") or "").strip().lower()
        if raw_agendas[0].get("is_default_agenda") or stitle in ("general discussion", "no agenda", "general meeting discussion"):
            is_single_default = True

    doc = DocxDocument()
    add_bold_heading(doc, "Final Record of Meeting (ROM)", 0)

    if agendas or (is_single_default and raw_agendas):
        display_agendas = agendas if agendas else raw_agendas
        if is_single_default:
            # 3-column format (No Agenda / General Discussion): ID (5%), Discussion Points (65%), Action / Speaker (30%)
            col_widths = [Inches(0.325), Inches(4.225), Inches(1.95)]
            table = doc.add_table(rows=1, cols=3)
            table.style = 'Table Grid'
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'ID'
            hdr_cells[1].text = 'Discussion Points'
            hdr_cells[2].text = 'Action / Speaker'
            style_table_header_bold(table)

            global_p_counter = 1
            for a in display_agendas:
                pts = a.get("discussion_points", [])
                for pt in pts:
                    p_code = f"P{global_p_counter}"
                    global_p_counter += 1
                    pt_text = (pt.get("text") or pt.get("polished_text") or pt.get("discussion_point") or "").strip()
                    spk = pt.get("speaker")
                    if not spk and pt.get("speakers"):
                        spk = ", ".join(pt["speakers"]) if isinstance(pt["speakers"], list) else str(pt["speakers"])
                    spk_str = str(spk).strip() if spk else "-"

                    row_cells = table.add_row().cells
                    row_cells[0].text = p_code
                    
                    dp_cell = row_cells[1]
                    dp_cell.text = ""
                    add_bold_label(dp_cell, p_code, pt_text if pt_text else p_code)

                    row_cells[2].text = spk_str
            set_fixed_table_column_widths(table, col_widths)
        else:
            # 4-column format: ID (5%), Agenda (15%), Discussion Points (50%), Action / Speaker (30%)
            col_widths = [Inches(0.325), Inches(1.95), Inches(3.25), Inches(0.975)]
            table = doc.add_table(rows=1, cols=4)
            table.style = 'Table Grid'

            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'ID'
            hdr_cells[1].text = 'Agenda'
            hdr_cells[2].text = 'Discussion Points'
            hdr_cells[3].text = 'Action / Speaker'
            style_table_header_bold(table)

            global_p_counter = 1

            for a_idx, a in enumerate(display_agendas, 1):
                a_code = a.get("agenda_id") or f"A{a_idx}"
                a_title = a.get("title", "")
                agenda_label = f"{a_code} – {a_title}" if a_title else a_code
                pts = a.get("discussion_points", [])

                for p_idx, pt in enumerate(pts):
                    p_code = f"P{global_p_counter}"
                    global_p_counter += 1
                    pt_text = (pt.get("text") or pt.get("polished_text") or pt.get("discussion_point") or "").strip()

                    spk = pt.get("speaker")
                    if not spk and pt.get("speakers"):
                        speakers_list = pt.get("speakers")
                        spk = ", ".join(speakers_list) if isinstance(speakers_list, list) else str(speakers_list)
                    spk_str = str(spk).strip() if spk else "-"

                    row_cells = table.add_row().cells
                    row_cells[0].text = str(a_idx) if p_idx == 0 else ""

                    # Agenda Column: FULLY BOLD content
                    agenda_cell = row_cells[1]
                    agenda_cell.text = ""
                    if p_idx == 0 and agenda_label:
                        p_ag = agenda_cell.add_paragraph()
                        r_ag = p_ag.add_run(agenda_label)
                        r_ag.bold = True

                    # Discussion Points Column: Key-label bold formatting
                    dp_cell = row_cells[2]
                    dp_cell.text = ""
                    add_bold_label(dp_cell, p_code, pt_text if pt_text else p_code)

                    row_cells[3].text = spk_str
            set_fixed_table_column_widths(table, col_widths)

    else:
        doc.add_paragraph("No Final ROM data available.")

    f = io.BytesIO()
    doc.save(f)
    f.seek(0)

    return StreamingResponse(
        f,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename=rom_final_{recording_id}.docx"}
    )

def format_precise_point_text(pt: dict, p_code: str = "") -> str:
    from services.rom_service import clean_calendar_dates

    raw_decisions = pt.get("decisions") or []
    if isinstance(raw_decisions, str): raw_decisions = [raw_decisions]
    decisions = [str(d).strip() for d in raw_decisions if str(d).strip() and str(d).strip().lower() not in ("none", "n/a", "null")]

    raw_actions = pt.get("action_items") or []
    if isinstance(raw_actions, str): raw_actions = [raw_actions]
    actions = [str(a).strip() for a in raw_actions if str(a).strip() and str(a).strip().lower() not in ("none", "n/a", "null")]

    valid_dates = clean_calendar_dates(pt.get("dates") or [])

    content_parts = []
    if decisions:
        content_parts.extend(decisions)
    if actions:
        content_parts.extend(actions)

    if not content_parts:
        pt_text = (pt.get("text") or pt.get("polished_text") or pt.get("discussion_point") or "").strip()
        if pt_text:
            content_parts.append(pt_text)

    if not content_parts:
        main_text = "Discussion noted."
    else:
        joined = ". ".join(p.rstrip(".") for p in content_parts)
        main_text = f"{joined}."

    prefix = f"{p_code}: " if p_code else ""
    if valid_dates:
        date_str = ", ".join(valid_dates)
        return f"{prefix}{main_text} ({date_str})"
    else:
        return f"{prefix}{main_text}"

@router.get("/{recording_id}/precise/download/docx")
async def download_precise_docx(recording_id: str, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    from services.rom_service import apply_speaker_mappings_to_final_rom
    if data.get("final_rom"):
        data["final_rom"] = apply_speaker_mappings_to_final_rom(data["final_rom"])
    agendas = data.get("final_rom", {}).get("agendas", [])

    doc = DocxDocument()
    doc.add_heading("Precise Record of Meeting (ROM)", 0)

    if agendas:
        col_widths = [Inches(0.325), Inches(0.975), Inches(3.25), Inches(1.95)]
        table = doc.add_table(rows=1, cols=4)
        table.style = 'Table Grid'

        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'ID'
        hdr_cells[1].text = 'Agenda'
        hdr_cells[2].text = 'Discussion Points'
        hdr_cells[3].text = 'Action / Speaker'

        global_p_counter = 1

        for a_idx, a in enumerate(agendas, 1):
            a_code = a.get("agenda_id") or f"A{a_idx}"
            a_title = a.get("title", "")
            agenda_label = f"{a_code} – {a_title}" if a_title else a_code

            pts = a.get("discussion_points", [])

            if not pts:
                row_cells = table.add_row().cells
                row_cells[0].text = str(a_idx)
                row_cells[1].text = agenda_label
                row_cells[2].text = "No discussion points mapped"
                row_cells[3].text = "-"
            else:
                for p_idx, pt in enumerate(pts):
                    p_code = f"P{global_p_counter}"
                    global_p_counter += 1

                    point_str = format_precise_point_text(pt, p_code)

                    spk = pt.get("speaker")
                    if not spk and pt.get("speakers"):
                        speakers_list = pt.get("speakers")
                        spk = ", ".join(speakers_list) if isinstance(speakers_list, list) else str(speakers_list)
                    spk_str = str(spk).strip() if spk else "-"

                    row_cells = table.add_row().cells
                    row_cells[0].text = str(a_idx) if p_idx == 0 else ""
                    row_cells[1].text = agenda_label if p_idx == 0 else ""
                    row_cells[2].text = point_str
                    row_cells[3].text = spk_str
        set_fixed_table_column_widths(table, col_widths)

    else:
        doc.add_paragraph("No Precise ROM data available.")

    f = io.BytesIO()
    doc.save(f)
    f.seek(0)

    return StreamingResponse(
        f,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename=rom_precise_{recording_id}.docx"}
    )


@router.get("/{recording_id}/agenda-transcript/download/docx")
async def download_agenda_transcript_docx(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Download Agenda-mapped Transcript as a Word (.docx) table."""
    user_id = _validate_user_id(current_user)
    rec_row = await _get_recording_or_404(recording_id, user_id, db)
    data = await _get_rom_data(recording_id, user_id, db)

    from services.rom_service import apply_speaker_mappings_to_final_rom, _format_time_hhmm
    if data.get("final_rom"):
        data["final_rom"] = apply_speaker_mappings_to_final_rom(data["final_rom"])

    final_rom = data.get("final_rom")
    if not final_rom or not final_rom.get("agendas"):
        raise HTTPException(status_code=404, detail="Final ROM not generated yet. Please generate Final ROM first.")

    raw_transcript = []
    if rec_row.get("transcript"):
        try:
            raw_transcript = json.loads(rec_row["transcript"]) if isinstance(rec_row["transcript"], str) else rec_row["transcript"]
        except Exception:
            raw_transcript = []

    speaker_mappings = final_rom.get("speaker_mappings") or {}
    agendas = final_rom.get("agendas", [])

    # Check if single default agenda (General Discussion / No Agenda)
    is_single_default = False
    if len(agendas) <= 1:
        if not agendas:
            is_single_default = True
        else:
            stitle = (agendas[0].get("title") or "").strip().lower()
            if agendas[0].get("is_default_agenda") or stitle in ("general discussion", "no agenda", "general meeting discussion"):
                is_single_default = True

    # Build mapping from segment index to assigned agenda
    segments_info = []
    for idx, seg in enumerate(raw_transcript):
        spk_raw = str(seg.get("speaker") or "Unknown").strip()
        spk = speaker_mappings.get(spk_raw, spk_raw)
        start_t = float(seg.get("start") or 0.0)
        end_t = float(seg.get("end") or 0.0)
        txt = str(seg.get("text") or "").strip()
        segments_info.append({
            "index": idx,
            "speaker": spk,
            "start": start_t,
            "end": end_t,
            "text": txt,
            "agenda_id": None,
            "agenda_label": None
        })

    segment_to_agenda = {}  # seg_idx -> (agenda_id, agenda_label)
    
    for a_idx, a in enumerate(agendas, 1):
        a_id = a.get("agenda_id") or f"A{a_idx}"
        a_title = (a.get("title") or "").strip()
        agenda_label = f"{a_id} – {a_title}" if a_title else a_id

        pts = a.get("discussion_points") or []
        for pt in pts:
            t_start = pt.get("timeline_start")
            t_end = pt.get("timeline_end")

            if t_start is not None and t_end is not None and float(t_end) > float(t_start):
                f_start = float(t_start)
                f_end = float(t_end)
                for seg in segments_info:
                    s_idx = seg["index"]
                    if s_idx not in segment_to_agenda:
                        if seg["start"] < f_end and seg["end"] > f_start:
                            segment_to_agenda[s_idx] = (a_id, agenda_label)

    default_agenda_id = agendas[0].get("agenda_id") if agendas else "A1"
    default_agenda_title = (agendas[0].get("title") or "").strip() if agendas else "General Discussion"
    default_agenda_label = f"{default_agenda_id} – {default_agenda_title}" if default_agenda_title else default_agenda_id

    for seg in segments_info:
        s_idx = seg["index"]
        if s_idx in segment_to_agenda:
            seg["agenda_id"], seg["agenda_label"] = segment_to_agenda[s_idx]
        else:
            seg["agenda_id"] = default_agenda_id
            seg["agenda_label"] = default_agenda_label

    # Merge adjacent or duplicate transcript segments that have SAME agenda AND SAME speaker
    from collections import OrderedDict

    # ------------------------------------------------------------------
    # Group transcript segments by agenda while preserving transcript order
    # ------------------------------------------------------------------
    agenda_groups = OrderedDict()

    # Create groups in agenda order
    for a_idx, agenda in enumerate(agendas, 1):
        aid = agenda.get("agenda_id") or f"A{a_idx}"
        title = (agenda.get("title") or "").strip()
        label = f"{aid} – {title}" if title else aid

        agenda_groups[aid] = {
            "agenda_id": aid,
            "label": label,
            "segments": []
        }

    # Assign transcript segments to their agenda
    for seg in segments_info:
        aid = seg["agenda_id"]

        if aid not in agenda_groups:
            agenda_groups[aid] = {
                "agenda_id": aid,
                "label": seg["agenda_label"],
                "segments": []
            }

        agenda_groups[aid]["segments"].append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        })

    # ------------------------------------------------------------------
    # Merge consecutive transcript segments from the same speaker
    # (within each agenda only)
    # ------------------------------------------------------------------
    for agenda in agenda_groups.values():
        merged = []

        for seg in agenda["segments"]:

            if (
                merged
                and merged[-1]["speaker"] == seg["speaker"]
            ):
                merged[-1]["end"] = max(merged[-1]["end"], seg["end"])

                cur_text = seg["text"].strip()

                if cur_text and cur_text.lower() not in merged[-1]["text"].lower():
                    if merged[-1]["text"]:
                        merged[-1]["text"] += " " + cur_text
                    else:
                        merged[-1]["text"] = cur_text

            else:
                merged.append({
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                })

        agenda["segments"] = merged

    # ------------------------------------------------------------------
    # Flatten back into merged_items (agenda grouped)
    # ------------------------------------------------------------------
    merged_items = []

    for agenda in agenda_groups.values():
        for seg in agenda["segments"]:
            merged_items.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "agenda_id": agenda["agenda_id"],
                "agenda_label": agenda["label"]
            })
    doc = DocxDocument()
    add_bold_heading(doc, "Agenda Transcript Mapping", 0)

    if not merged_items:
        doc.add_paragraph("No transcript data available.")
    else:
        if is_single_default:
            # 2-column table: ID, Transcript Segment
            col_widths = [Inches(0.4), Inches(6.1)]
            table = doc.add_table(rows=1, cols=2)
            table.style = 'Table Grid'

            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'ID'
            hdr_cells[1].text = 'Transcript Segment'
            style_table_header_bold(table)

            for t_idx, item in enumerate(merged_items, 1):
                t_code = f"T{t_idx}"
                time_str = f"{_format_time_hhmm(item['start'])}–{_format_time_hhmm(item['end'])}"
                seg_header = f"{item['speaker']} ({time_str}): "

                row_cells = table.add_row().cells
                row_cells[0].text = t_code

                dp_cell = row_cells[1]
                dp_cell.text = ""
                p = dp_cell.add_paragraph()
                r_spk = p.add_run(seg_header)
                r_spk.bold = True
                p.add_run(item["text"])

            set_fixed_table_column_widths(table, col_widths)

        else:
            # 3-column table: ID, Transcript Segment, Agenda
            col_widths = [Inches(0.4), Inches(4.35), Inches(1.75)]
            table = doc.add_table(rows=1, cols=3)
            table.style = 'Table Grid'

            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'ID'
            hdr_cells[1].text = 'Transcript Segment'
            hdr_cells[2].text = 'Agenda'
            style_table_header_bold(table)

            previous_agenda = None
            previous_agenda_cell = None

            for t_idx, item in enumerate(merged_items, 1):
                t_code = f"T{t_idx}"
                time_str = f"{_format_time_hhmm(item['start'])}–{_format_time_hhmm(item['end'])}"
                seg_header = f"{item['speaker']} ({time_str}): "

                row_cells = table.add_row().cells

                # ID
                row_cells[0].text = t_code

                # Transcript
                dp_cell = row_cells[1]
                dp_cell.text = ""
                p = dp_cell.add_paragraph()
                r_spk = p.add_run(seg_header)
                r_spk.bold = True
                p.add_run(item["text"])

                # Agenda (merge with previous if same)
                if previous_agenda == item["agenda_label"] and previous_agenda_cell is not None:
                    previous_agenda_cell = previous_agenda_cell.merge(row_cells[2])
                else:
                    ag_cell = row_cells[2]
                    ag_cell.text = ""
                    p_ag = ag_cell.add_paragraph()
                    r_ag = p_ag.add_run(item["agenda_label"])
                    r_ag.bold = True

                    previous_agenda = item["agenda_label"]
                    previous_agenda_cell = ag_cell

            set_fixed_table_column_widths(table, col_widths)

    f = io.BytesIO()
    doc.save(f)
    f.seek(0)

    return StreamingResponse(
        f,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename=rom_agenda_transcript_{recording_id}.docx"}
    )


# ── Agenda & Point Deletion Endpoints ─────────────────────────────────────────

@router.delete("/{recording_id}/stage3/agenda/{agenda_id}")
async def delete_stage3_agenda(
    recording_id: str,
    agenda_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Delete an agenda item. Reassigns all points mapped to it to 'General Discussion'."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    stage3 = data.get("stage3", {})
    agendas = stage3.get("agendas", [])

    target_agenda = next((a for a in agendas if a.get("agenda_id") == agenda_id), None)
    if not target_agenda:
        raise HTTPException(status_code=404, detail=f"Agenda {agenda_id} not found")

    # Remove target agenda
    agendas = [a for a in agendas if a.get("agenda_id") != agenda_id]

    # Ensure "General Discussion" agenda exists as fallback
    gen_discussion_id = "A1"
    gen_agenda = next((a for a in agendas if a.get("agenda_id") == gen_discussion_id or a.get("title", "").strip().lower() in ("general discussion", "no agenda")), None)
    if not gen_agenda:
        gen_agenda = {
            "agenda_id": gen_discussion_id,
            "title": "General Discussion",
            "description": "General meeting discussion points",
            "is_default_agenda": True
        }
        agendas.insert(0, gen_agenda)
    else:
        gen_discussion_id = gen_agenda.get("agenda_id", "A1")

    # Reassign points mapped to deleted agenda to General Discussion
    point_mappings = stage3.get("point_mappings", {})
    agenda_groups = stage3.get("agenda_groups", {})

    reassigned_pids = [pid for pid, aid in point_mappings.items() if aid == agenda_id]
    for pid in reassigned_pids:
        point_mappings[pid] = gen_discussion_id

    # Update agenda_groups
    agenda_groups.pop(agenda_id, None)
    gen_list = agenda_groups.get(gen_discussion_id, [])
    for pid in reassigned_pids:
        if pid not in gen_list:
            gen_list.append(pid)
    agenda_groups[gen_discussion_id] = gen_list

    stage3["agendas"] = agendas
    stage3["point_mappings"] = point_mappings
    stage3["agenda_groups"] = agenda_groups
    data["stage3"] = stage3

    # Update final_rom agendas
    if "final_rom" in data and isinstance(data["final_rom"].get("agendas"), list):
        final_agendas = data["final_rom"]["agendas"]
        final_agendas = [fa for fa in final_agendas if fa.get("agenda_id") != agenda_id]
        data["final_rom"]["agendas"] = final_agendas

    await _save_rom_data(recording_id, user_id, data, db)

    # Synchronize updated agendas (after deletion) to parsed_agenda_json in the recordings table
    from services.rag_pipeline import _save_parsed_agenda
    parsed_items = []
    for a in agendas:
        parsed_items.append({
            "topic": a.get("title") or a.get("topic") or "",
            "speaker": a.get("speaker") or a.get("presenter"),
            "details": a.get("description") or a.get("details") or "",
            "keywords": a.get("keywords") or [],
            "related_concepts": a.get("related_concepts") or [],
            "alternative_terminology": a.get("alternative_terminology") or [],
            "expected_themes": a.get("expected_themes") or [],
        })
    _save_parsed_agenda(recording_id, user_id, parsed_items)

    return {"status": "success", "rom_data": data, "stage3": stage3}


@router.delete("/{recording_id}/stage3/point/{point_id}")
async def delete_discussion_point(
    recording_id: str,
    point_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Delete an individual enhanced discussion point from Stage 2, Stage 3, and Final ROM."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)

    # 1. Remove from stage2.polished_points
    if "stage2" in data and isinstance(data["stage2"].get("polished_points"), list):
        data["stage2"]["polished_points"] = [
            p for p in data["stage2"]["polished_points"] if p.get("id") != point_id
        ]

    # 2. Remove from stage3 mappings
    if "stage3" in data:
        stage3 = data["stage3"]
        if "point_mappings" in stage3 and isinstance(stage3["point_mappings"], dict):
            stage3["point_mappings"].pop(point_id, None)
        if "batch_assignments" in stage3 and isinstance(stage3["batch_assignments"], list):
            stage3["batch_assignments"] = [
                ba for ba in stage3["batch_assignments"] if ba.get("point_id") != point_id
            ]
        if "candidate_results" in stage3 and isinstance(stage3["candidate_results"], list):
            stage3["candidate_results"] = [
                cr for cr in stage3["candidate_results"] if cr.get("point_id") != point_id
            ]
        if "agenda_groups" in stage3 and isinstance(stage3["agenda_groups"], dict):
            for aid, pids in stage3["agenda_groups"].items():
                if isinstance(pids, list) and point_id in pids:
                    stage3["agenda_groups"][aid] = [p for p in pids if p != point_id]

    # 3. Remove from final_rom agendas
    if "final_rom" in data and isinstance(data["final_rom"].get("agendas"), list):
        for fa in data["final_rom"]["agendas"]:
            if "discussion_points" in fa and isinstance(fa["discussion_points"], list):
                fa["discussion_points"] = [
                    pt for pt in fa["discussion_points"] if pt.get("id") != point_id
                ]

    await _save_rom_data(recording_id, user_id, data, db)
    return {"status": "success", "rom_data": data}


# ── MOM Generation from Enhanced ROM ─────────────────────────────────────────

@router.post("/{recording_id}/stage3/generate-mom-from-rom")
async def generate_mom_from_rom(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Generate Minutes of Meeting (MOM) using Stage 2/Stage 3 Enhanced Discussion Points."""
    user_id = _validate_user_id(current_user)
    row = await _get_recording_or_404(recording_id, user_id, db)
    data = await _get_rom_data(recording_id, user_id, db)

    polished_points = data.get("stage2", {}).get("polished_points", [])
    if not polished_points:
        raise HTTPException(status_code=400, detail="Stage 2 enhanced discussion points must exist before generating MOM")

    rec_meta = {
        "filename": row.get("filename") or f"Recording {recording_id[:8]}",
        "created_at": row.get("created_at"),
        "duration": row.get("duration"),
        "speakers_detected": row.get("speakers_detected") or [],
    }

    loop = asyncio.get_event_loop()
    enhanced_mom = await loop.run_in_executor(
        None,
        lambda: rom_service.generate_mom_from_enhanced_rom(
            polished_points=polished_points,
            recording_meta=rec_meta,
            recording_id=recording_id,
            user_id=user_id,
        )
    )

    if "stage3" not in data:
        data["stage3"] = {}
    data["stage3"]["enhanced_mom"] = enhanced_mom

    await _save_rom_data(recording_id, user_id, data, db)

    # ── OVERWRITE / UPDATE MAIN MOM IN minutes_of_meeting TABLE ─────────
    # Apply stored speaker_mappings if present
    spk_map_raw = row.get("speaker_mappings")
    spk_mappings = from_json(spk_map_raw, {}) if isinstance(spk_map_raw, str) else (spk_map_raw or {})
    if spk_mappings and isinstance(spk_mappings, dict):
        from services.speaker_sync import apply_speaker_mappings_to_mom_dict
        enhanced_mom = apply_speaker_mappings_to_mom_dict(enhanced_mom, spk_mappings)

    r_mom = await db.execute(
        text("SELECT id, versions FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
        {"rid": recording_id, "uid": user_id},
    )
    existing_mom = r_mom.fetchone()

    now = datetime.now(timezone.utc)
    now_str = dt_to_str(now)

    title = enhanced_mom.get("title") or row.get("filename") or "Minutes of Meeting"
    date_val = enhanced_mom.get("date") or ""
    duration = enhanced_mom.get("duration") or row.get("duration") or 0
    planned_start = enhanced_mom.get("planned_start_time") or ""
    actual_start = enhanced_mom.get("actual_start_time") or ""
    participants = enhanced_mom.get("participants", [])
    intro = enhanced_mom.get("introduction", "")
    pts_discussed = enhanced_mom.get("points_discussed", [])
    actions = enhanced_mom.get("action_items", [])
    conclusion = enhanced_mom.get("conclusion", "")

    # Standardize action items list for MoM table
    norm_actions = []
    for item in actions:
        if isinstance(item, dict):
            task_str = str(item.get("task") or item.get("item") or item.get("description") or "").strip()
            owner_str = str(item.get("owner", "Unassigned") or "Unassigned").strip()
            deadline_str = str(item.get("deadline", "ASAP") or "ASAP").strip()
            if task_str:
                norm_actions.append({
                    "task": task_str,
                    "owner": owner_str,
                    "deadline": deadline_str,
                    "status": item.get("status", "open")
                })
        elif isinstance(item, str) and item.strip():
            norm_actions.append({"task": item.strip(), "owner": "Unassigned", "deadline": "ASAP", "status": "open"})

    if existing_mom:
        existing_versions = from_json(existing_mom._mapping.get("versions"), [])
        v_num = len(existing_versions) + 1
        new_version = {"version": v_num, "data": enhanced_mom, "saved_at": now_str}
        updated_versions = existing_versions + [new_version]

        await db.execute(
            text("""
                UPDATE minutes_of_meeting SET
                    title = :title, date = :date, duration = :duration,
                    planned_start_time = :planned_start_time,
                    actual_start_time = :actual_start_time,
                    participants = :participants,
                    introduction = :introduction,
                    points_discussed = :points_discussed,
                    action_items = :action_items,
                    conclusion = :conclusion,
                    versions = :versions, is_draft = 0, updated_at = :updated_at
                WHERE recording_id = :rid AND user_id = :uid
            """),
            {
                "title": title,
                "date": date_val,
                "duration": duration,
                "planned_start_time": planned_start,
                "actual_start_time": actual_start,
                "participants": to_json(participants),
                "introduction": intro,
                "points_discussed": to_json(pts_discussed),
                "action_items": to_json(norm_actions),
                "conclusion": conclusion,
                "versions": to_json(updated_versions),
                "updated_at": now_str,
                "rid": recording_id,
                "uid": user_id,
            }
        )
    else:
        mom_id = str(uuid.uuid4())
        initial_version = [{"version": 1, "data": enhanced_mom, "saved_at": now_str}]
        await db.execute(
            text("""
                INSERT INTO minutes_of_meeting (
                    id, recording_id, user_id, title, date, duration,
                    planned_start_time, actual_start_time, participants,
                    introduction, points_discussed, action_items, conclusion,
                    versions, is_draft, created_at, updated_at
                ) VALUES (
                    :id, :rid, :uid, :title, :date, :duration,
                    :planned_start_time, :actual_start_time, :participants,
                    :introduction, :points_discussed, :action_items, :conclusion,
                    :versions, 0, :created_at, :updated_at
                )
            """),
            {
                "id": mom_id,
                "rid": recording_id,
                "uid": user_id,
                "title": title,
                "date": date_val,
                "duration": duration,
                "planned_start_time": planned_start,
                "actual_start_time": actual_start,
                "participants": to_json(participants),
                "introduction": intro,
                "points_discussed": to_json(pts_discussed),
                "action_items": to_json(norm_actions),
                "conclusion": conclusion,
                "versions": to_json(initial_version),
                "created_at": now_str,
                "updated_at": now_str,
            }
        )

    # Synchronize and update text summary fields in the recordings table
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
            "short_summary": intro,
            "detailed_summary": conclusion,
            "summary": intro,
            "key_points": to_json(pts_discussed),
            "action_items": to_json(norm_actions),
            "id": recording_id,
            "uid": user_id,
        }
    )
    await db.commit()

    return {
        "status": "success",
        "mom": enhanced_mom,
        "enhanced_mom": enhanced_mom,
        "rom_data": data
    }


@router.post("/{recording_id}/stage3/generate-advanced-mom")
async def generate_advanced_mom(
    recording_id: str,
    req: GenerateAdvancedMomRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Generate Advanced MoM by refining, enhancing, organizing, and enriching action points using custom prompt instructions.
    Optionally regenerates Title, Introduction, or Conclusion if their respective flags are True.
    Updates minutes_of_meeting table with refined action items.
    """
    user_id = _validate_user_id(current_user)
    row = await _get_recording_or_404(recording_id, user_id, db)
    data = await _get_rom_data(recording_id, user_id, db)

    final_rom = data.get("final_rom") or {}
    if not final_rom or not final_rom.get("agendas"):
        raise HTTPException(status_code=400, detail="Final ROM must exist before generating Advanced MoM.")

    r_mom = await db.execute(
        text("SELECT * FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
        {"rid": recording_id, "uid": user_id},
    )
    existing_mom_row = r_mom.fetchone()
    existing_mom_dict = dict(existing_mom_row._mapping) if existing_mom_row else {}

    rec_meta = {
        "filename": row.get("filename") or f"Recording {recording_id[:8]}",
        "created_at": row.get("created_at"),
        "duration": row.get("duration"),
        "speakers_detected": row.get("speakers_detected") or [],
    }

    loop = asyncio.get_event_loop()
    advanced_mom = await loop.run_in_executor(
        None,
        lambda: rom_service.generate_advanced_mom(
            final_rom=final_rom,
            existing_mom=existing_mom_dict,
            custom_prompt=req.custom_prompt,
            regenerate_title=req.regenerate_title,
            regenerate_intro=req.regenerate_intro,
            regenerate_conclusion=req.regenerate_conclusion,
            recording_meta=rec_meta,
            recording_id=recording_id,
            user_id=user_id,
        )
    )

    if advanced_mom.get("agendas"):
        data["final_rom"]["agendas"] = advanced_mom["agendas"]

    if "stage3" not in data:
        data["stage3"] = {}
    data["stage3"]["enhanced_mom"] = advanced_mom

    await _save_rom_data(recording_id, user_id, data, db)

    actions = advanced_mom.get("action_items") or []
    norm_actions = []
    for item in actions:
        if isinstance(item, dict):
            task_str = str(item.get("task") or item.get("item") or item.get("description") or "").strip()
            owner_str = str(item.get("owner", "Unassigned") or "Unassigned").strip()
            deadline_str = str(item.get("deadline", "ASAP") or "ASAP").strip()
            if task_str:
                norm_actions.append({
                    "task": task_str,
                    "owner": owner_str,
                    "deadline": deadline_str,
                    "status": item.get("status", "open")
                })
        elif isinstance(item, str) and item.strip():
            norm_actions.append({"task": item.strip(), "owner": "Unassigned", "deadline": "ASAP", "status": "open"})

    title = advanced_mom.get("title") or existing_mom_dict.get("title") or row.get("filename") or "Minutes of Meeting"
    intro = advanced_mom.get("introduction") or existing_mom_dict.get("introduction") or ""
    conclusion = advanced_mom.get("conclusion") or existing_mom_dict.get("conclusion") or ""

    now = datetime.now(timezone.utc)
    now_str = dt_to_str(now)

    if existing_mom_row:
        existing_versions = from_json(existing_mom_dict.get("versions"), [])
        v_num = len(existing_versions) + 1
        new_version = {"version": v_num, "data": advanced_mom, "saved_at": now_str}
        updated_versions = existing_versions + [new_version]

        await db.execute(
            text("""
                UPDATE minutes_of_meeting SET
                    title = :title,
                    introduction = :introduction,
                    action_items = :action_items,
                    conclusion = :conclusion,
                    versions = :versions, is_draft = 0, updated_at = :updated_at
                WHERE recording_id = :rid AND user_id = :uid
            """),
            {
                "title": title,
                "introduction": intro,
                "action_items": to_json(norm_actions),
                "conclusion": conclusion,
                "versions": to_json(updated_versions),
                "updated_at": now_str,
                "rid": recording_id,
                "uid": user_id,
            }
        )
    else:
        mom_id = str(uuid.uuid4())
        initial_version = [{"version": 1, "data": advanced_mom, "saved_at": now_str}]
        await db.execute(
            text("""
                INSERT INTO minutes_of_meeting (
                    id, recording_id, user_id, title, date, duration,
                    planned_start_time, actual_start_time, participants,
                    introduction, points_discussed, action_items, conclusion,
                    versions, is_draft, created_at, updated_at
                ) VALUES (
                    :id, :rid, :uid, :title, :date, :duration,
                    :planned_start_time, :actual_start_time, :participants,
                    :introduction, :points_discussed, :action_items, :conclusion,
                    :versions, 0, :created_at, :updated_at
                )
            """),
            {
                "id": mom_id,
                "rid": recording_id,
                "uid": user_id,
                "title": title,
                "date": str(row.get("created_at") or ""),
                "duration": row.get("duration") or 0,
                "planned_start_time": "",
                "actual_start_time": "",
                "participants": to_json(row.get("speakers_detected") or []),
                "introduction": intro,
                "points_discussed": to_json([]),
                "action_items": to_json(norm_actions),
                "conclusion": conclusion,
                "versions": to_json(initial_version),
                "created_at": now_str,
                "updated_at": now_str,
            }
        )

    await db.commit()

    return {
        "status": "success",
        "mom": advanced_mom,
        "enhanced_mom": advanced_mom,
        "rom_data": data
    }


@router.get("/{recording_id}/stage3/enhanced-mom/download/docx")
async def download_enhanced_mom_docx(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Download the main MOM as a Word (.docx) document."""
    user_id = _validate_user_id(current_user)
    row = await _get_recording_or_404(recording_id, user_id, db)
    data = await _get_rom_data(recording_id, user_id, db)

    # Fetch main MoM from minutes_of_meeting table
    r_mom = await db.execute(
        text("SELECT * FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
        {"rid": recording_id, "uid": user_id},
    )
    mom_row = r_mom.mappings().fetchone()
    if mom_row:
        from routers.mom_router import _mom_row_to_dict
        mom_data = _mom_row_to_dict(mom_row)
    else:
        mom_data = data.get("stage3", {}).get("enhanced_mom", {})

    if not mom_data:
        raise HTTPException(status_code=404, detail="No MOM generated yet. Call /stage3/generate-mom-from-rom first.")

    spk_map_raw = row.get("speaker_mappings")
    spk_mappings = from_json(spk_map_raw, {}) if isinstance(spk_map_raw, str) else (spk_map_raw or {})
    if spk_mappings and isinstance(spk_mappings, dict):
        from services.speaker_sync import apply_speaker_mappings_to_mom_dict
        mom_data = apply_speaker_mappings_to_mom_dict(mom_data, spk_mappings)

    doc = DocxDocument()
    title = mom_data.get("title") or row.get("filename") or "Minutes of Meeting"
    add_bold_heading(doc, title, 0)

    # Metadata section
    if mom_data.get("date"):
        add_bold_label(doc, "Date", mom_data['date'])
    if mom_data.get("duration"):
        add_bold_label(doc, "Duration", str(mom_data['duration']))
    participants = mom_data.get("participants", [])
    if participants:
        part_str = ", ".join(str(p) for p in participants)
        add_bold_label(doc, "Participants", part_str)

    # Introduction
    if mom_data.get("introduction"):
        add_bold_heading(doc, "1. Introduction & Overview", level=1)
        doc.add_paragraph(mom_data["introduction"])

    # Points Discussed
    pts = mom_data.get("points_discussed", [])
    if pts:
        add_bold_heading(doc, "2. Key Discussion Points", level=1)
        for i, pt in enumerate(pts, 1):
            if isinstance(pt, dict):
                add_bold_heading(doc, f"2.{i} {pt.get('topic', 'Discussion')}", level=2)
                if pt.get("summary"):
                    doc.add_paragraph(pt["summary"])
            else:
                doc.add_paragraph(f"• {str(pt)}")

    # Action Items
    actions = mom_data.get("action_items", [])
    if actions:
        add_bold_heading(doc, "3. Action Items", level=1)
        tbl = doc.add_table(rows=1, cols=4)
        tbl.style = 'Table Grid'
        hdr = tbl.rows[0].cells
        hdr[0].text = "#"
        hdr[1].text = "Action Item"
        hdr[2].text = "Owner"
        hdr[3].text = "Deadline"
        style_table_header_bold(tbl)

        for idx, act in enumerate(actions, 1):
            r = tbl.add_row().cells
            r[0].text = str(idx)
            r[1].text = act.get("task") or act.get("item") or act.get("description") or "-"
            r[2].text = act.get("owner") or "Unassigned"
            r[3].text = act.get("deadline") or "ASAP"

    # Conclusion
    if mom_data.get("conclusion"):
        add_bold_heading(doc, "4. Conclusion", level=1)
        doc.add_paragraph(mom_data["conclusion"])

    f = io.BytesIO()
    doc.save(f)
    f.seek(0)

    return StreamingResponse(
        f,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename=mom_{recording_id}.docx"}
    )


@router.delete("/{recording_id}")
async def delete_rom_data(recording_id: str, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    await db.execute(
        text("UPDATE recordings SET rom_data = NULL WHERE id = :id AND user_id = :uid"),
        {"id": recording_id, "uid": user_id}
    )
    await db.commit()
    return {"status": "success"}
