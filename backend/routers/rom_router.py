from __future__ import annotations

import io
import json
import logging
import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from database import get_db, dt_to_str, from_json, to_json
from routers.auth import get_current_user
from docx import Document as DocxDocument
from docx.shared import Inches

from services.rom_service import rom_service, get_stage1_progress as _get_stage1_progress

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

class RerunStage1WindowRequest(BaseModel):
    window_index: int
    user_feedback: str = ""
    transcript_window_minutes: float = Field(default=2.0, ge=0.5, le=30.0)

class AcceptRerunStage1WindowRequest(BaseModel):
    window_index: int
    user_feedback: str = ""
    original_points: List[dict] = []
    corrected_points: List[dict] = []
    transcript_window: Optional[str] = None

class Stage2Request(BaseModel):
    meeting_context_top_k: int = Field(default=5, ge=0, le=30)
    global_context_top_k: int = Field(default=3, ge=0, le=30)
    discussion_window_size: int = Field(default=5, ge=3, le=50)
    min_similarity_threshold: Optional[float] = Field(default=0.80, ge=0.0, le=1.0)
    process_all_together: Optional[bool] = Field(default=None, description="If True, process all discussion points together in a single call")
    reference_example_points: Optional[List[str]] = Field(default=None, description="Style-only reference example points")
    previous_meeting_mode: str = Field(default="auto", description="Dual mode: 'auto' | 'select' | 'off'")
    previous_meeting_id: Optional[str] = Field(default=None, description="Specific meeting ID when previous_meeting_mode is 'select'")
    previous_meeting_top_k: int = Field(default=3, ge=0, le=20, description="Number of previous Stage 2 points to retrieve")

class Stage2ExamplePointsBody(BaseModel):
    points: List[str]

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
    discussion_order: Optional[List[str]] = Field(default=None, description="Expected agenda discussion order, e.g. ['A1', 'A3', 'A2']")
    agenda_timeline: Optional[Dict[str, Dict[str, float]]] = Field(default=None, description="Approximate agenda timeline ranges with start_sec and end_sec")
    enable_discussion_order: bool = Field(default=False, description="If True, pass discussion order guidance to LLM. If False, omit.")
    enable_agenda_timeline: bool = Field(default=False, description="If True, pass timeline guidance to LLM. If False, omit.")
    skipped_agendas: Optional[Dict[str, dict]] = Field(default=None, description="Agenda IDs to skip from point mapping, e.g. {'A2': {'note': 'Forward to next meeting'}}")
    include_action_points: bool = Field(default=False, description="If True, include action points mapped to discussion points in the Final ROM")

class UpdateStage3AgendasRequest(BaseModel):
    agendas: List[dict]

class CreateAgendaItemRequest(BaseModel):
    title: str
    description: Optional[str] = ""
    presenter: Optional[str] = None
    agenda_id: Optional[str] = None

class UpdateFinalRomRequest(BaseModel):
    final_rom: dict
    version: Optional[str] = None

class ToggleActionPointsRequest(BaseModel):
    include_action_points: bool = True

class SelectRomVersionRequest(BaseModel):
    version: str = Field(..., description="'long', 'short', or 'medium'")

class UpdateSpeakerMappingsRequest(BaseModel):
    speaker_mappings: Dict[str, str]


class GenerateAdvancedMomRequest(BaseModel):
    custom_prompt: str
    regenerate_title: bool = False
    regenerate_intro: bool = False
    regenerate_conclusion: bool = False


DEFAULT_REWRITE_INSTRUCTION = (
    "Rewrite the ROM in a formal, professional writing style. "
    "Improve grammar, sentence structure, readability, and formatting only. "
    "Do not change, add, remove, or reinterpret any facts, discussion points, decisions, "
    "action items, speakers, or context. "
    "Preserve the exact meaning and structure while presenting it in polished formal language."
)


class RewriteRomRequest(BaseModel):
    rewrite_instruction: str = DEFAULT_REWRITE_INSTRUCTION
    mode: str = Field(default="window", description="'window', 'complete', or 'reference'")
    window_size: int = Field(default=3, ge=1, le=10, description="Points per window (window/reference modes)")
    writing_rules: str = Field(default="", description="Writing rules extracted from a reference document (reference mode)")


class GenerateRomVersionRequest(BaseModel):
    version: str = Field(default="short", description="'short', 'medium', or 'long'")
    writing_rules: str = Field(default="", description="Optional writing rules from a reference document")
    base_final_rom: Optional[Dict] = Field(default=None, description="Base final ROM with original Stage 2 points")


class ExtractWritingRulesRequest(BaseModel):
    reference_text: str = Field(..., description="Extracted text of the reference MoM/ROM document")


class MergePointsRequest(BaseModel):
    point_ids: List[str] = Field(..., min_length=2)

class FindReplaceRequest(BaseModel):
    find_text: str = Field(..., min_length=1)
    replace_text: str = ""

class FindPreviewRequest(BaseModel):
    find_text: str = Field(..., min_length=1)

class SplitPointRequest(BaseModel):
    point_id: str
    selected_text: str = Field(..., min_length=1)

class DeleteTextRequest(BaseModel):
    point_id: str
    text_to_delete: str = Field(..., min_length=1)

class GenerateEditTrainingRequest(BaseModel):
    selected_change_ids: List[str] = Field(..., min_length=1)

class ManualEditPointRequest(BaseModel):
    polished_text: str = Field(..., min_length=1)


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
    try:
        res = await db.execute(
            text("SELECT rom_data FROM rom_metadata WHERE recording_id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id}
        )
        rm_row = res.fetchone()
        if rm_row and rm_row[0]:
            parsed = from_json(rm_row[0], {})
            if isinstance(parsed, dict):
                return parsed
    except Exception:
        pass
    row = await _get_recording_or_404(recording_id, user_id, db)
    if row.get("rom_data"):
        parsed = from_json(row["rom_data"], {})
        if isinstance(parsed, dict):
            return parsed
    return {}


async def _save_rom_data(recording_id: str, user_id_param: Any, rom_data: dict, db):
    user_id = _validate_user_id(user_id_param)
    rom_json = to_json(rom_data)
    now_str = datetime.now(timezone.utc).isoformat()
    await db.execute(
        text("UPDATE recordings SET rom_data = :data WHERE id = :id AND user_id = :uid"),
        {"data": rom_json, "id": recording_id, "uid": user_id}
    )
    try:
        await db.execute(
            text("""
                INSERT INTO rom_metadata (id, recording_id, user_id, rom_data, created_at, updated_at)
                VALUES (:id, :rid, :uid, :data, :cat, :uat)
                ON CONFLICT(recording_id) DO UPDATE SET
                    rom_data = :data,
                    updated_at = :uat
            """),
            {
                "id": str(uuid.uuid4()),
                "rid": recording_id,
                "uid": user_id,
                "data": rom_json,
                "cat": now_str,
                "uat": now_str,
            }
        )
    except Exception as e:
        logger.warning(f"[ROM Router] Failed to upsert rom_metadata for {recording_id}: {e}")
    await db.commit()

async def _record_edit_history(
    db, recording_id: str, user_id: str, change_type: str,
    point_ids_affected: list, before_state: list, after_state: list,
    metadata: dict = None
) -> str:
    """Record a Stage 2 edit in the history table."""
    change_id = str(uuid.uuid4())
    await db.execute(
        text(
            "INSERT INTO stage2_edit_history "
            "(id, recording_id, user_id, change_type, point_ids_affected, "
            "before_state, after_state, metadata, is_reverted, created_at) "
            "VALUES (:id, :rid, :uid, :ct, :pids, :bs, :as_, :meta, 0, :cat)"
        ),
        {
            "id": change_id,
            "rid": recording_id,
            "uid": user_id,
            "ct": change_type,
            "pids": to_json(point_ids_affected),
            "bs": to_json(before_state),
            "as_": to_json(after_state),
            "meta": to_json(metadata or {}),
            "cat": datetime.utcnow().isoformat(),
        },
    )
    await db.commit()
    return change_id


async def _clear_stage2_edit_history(db, recording_id: str, user_id: str):
    """Delete all recorded Stage 2 edit history entries when Stage 2 is regenerated."""
    try:
        await db.execute(
            text("DELETE FROM stage2_edit_history WHERE recording_id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        await db.commit()
    except Exception as e:
        logger.warning(f"[ROM Router] Failed to clear stage2_edit_history for {recording_id}: {e}")



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
        
    r = await db.execute(
        text("SELECT rom_parallel_window_processing, rom_separate_action_extraction FROM user_settings WHERE user_id = :uid"),
        {"uid": user_id},
    )
    us_row = r.fetchone()
    parallel_concurrency = us_row[0] if us_row and us_row[0] is not None else 2
    separate_action_extraction = bool(us_row[1]) if us_row and us_row[1] is not None else False

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        lambda: rom_service.extract_discussion_points(
            transcript,
            req.transcript_window_minutes,
            user_id,
            recording_id=recording_id,
            video_transcript=video_transcript,
            source_type=source_type,
            parallel_window_processing=parallel_concurrency,
            separate_action_extraction=separate_action_extraction,
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
    
    # Preserve downstream stages and mark as outdated with warnings
    data.setdefault("outdated_warnings", {})
    if data.get("stage2"):
        data["outdated_warnings"]["stage2"] = "Stage 1 discussion points were regenerated. Existing Stage 2 points may be outdated."
    if data.get("stage3"):
        data["outdated_warnings"]["stage3"] = "Stage 1 discussion points were regenerated. Existing agenda data may be outdated."
    if data.get("final_rom"):
        data["outdated_warnings"]["final_rom"] = "Stage 1 discussion points were regenerated. Existing Final ROM may be outdated."
    
    await _save_rom_data(recording_id, user_id, data, db)
    return {
        "status": "success",
        "stage1": data["stage1"],
        "rom_data": data
    }


@router.get("/{recording_id}/stage1/progress")
async def get_stage1_progress_endpoint(recording_id: str, current_user: dict = Depends(get_current_user)):
    """Returns current Stage 1 generation progress for live frontend updates.
    
    Polls an in-memory progress tracker updated by rom_service during extraction.
    Returns windows_completed, windows_total, eta_seconds, and concurrency.
    """
    user_id = _validate_user_id(current_user)
    progress = _get_stage1_progress(user_id, recording_id)
    return progress


@router.post("/{recording_id}/stage1/rerun-window")
async def rerun_stage1_window_endpoint(
    recording_id: str,
    req: RerunStage1WindowRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Stage 1: Re-run extraction for a single transcript window incorporating user feedback."""
    user_id = _validate_user_id(current_user)
    rec_row = await _get_recording_or_404(recording_id, user_id, db)
    transcript = from_json(rec_row.get("transcript") or "[]")
    
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty")

    data = await _get_rom_data(recording_id, user_id, db)
    all_points = data.get("stage1", {}).get("discussion_points", [])
    window_points = [p for p in all_points if p.get("window_index") == req.window_index]

    video_transcript = from_json(rec_row.get("video_transcript") or "[]")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: rom_service.rerun_single_stage1_window(
            transcript=transcript,
            window_index=req.window_index,
            user_feedback=req.user_feedback,
            window_minutes=req.transcript_window_minutes,
            video_transcript=video_transcript,
            existing_points=window_points,
        )
    )

    return {
        "status": "success",
        "result": result
    }


@router.post("/{recording_id}/stage1/rerun-window/accept")
async def accept_rerun_stage1_window_endpoint(
    recording_id: str,
    req: AcceptRerunStage1WindowRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Stage 1: Accept regenerated points for a window, update ROM data, and persist for DSPy training."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)

    if "stage1" not in data or "discussion_points" not in data["stage1"]:
        raise HTTPException(status_code=400, detail="Stage 1 points not found")

    old_points = data["stage1"]["discussion_points"]
    # Replace points for window_index with corrected_points
    new_points = []
    replaced = False
    for p in old_points:
        if p.get("window_index") == req.window_index:
            if not replaced:
                new_points.extend(req.corrected_points)
                replaced = True
        else:
            new_points.append(p)

    if not replaced:
        new_points.extend(req.corrected_points)

    data["stage1"]["discussion_points"] = new_points

    # Preserve downstream stages and mark as outdated with warnings
    data.setdefault("outdated_warnings", {})
    if data.get("stage2"):
        data["outdated_warnings"]["stage2"] = "Stage 1 window was re-run and accepted. Existing Stage 2 points may be outdated."
    if data.get("stage3"):
        data["outdated_warnings"]["stage3"] = "Stage 1 window was re-run and accepted. Existing agenda data may be outdated."
    if data.get("final_rom"):
        data["outdated_warnings"]["final_rom"] = "Stage 1 window was re-run and accepted. Existing Final ROM may be outdated."

    await _save_rom_data(recording_id, user_id, data, db)

    # Persist feedback record for DSPy training
    feedback_id = None
    try:
        from services.training.stage1_training_service import save_feedback
        feedback_id = save_feedback({
            "user_id": user_id,
            "meeting_id": recording_id,
            "window_index": req.window_index,
            "transcript_window": req.transcript_window or "",
            "model_output": json.dumps(req.original_points, ensure_ascii=False),
            "corrected_output": json.dumps(req.corrected_points, ensure_ascii=False),
            "comment": req.user_feedback,
            "categories": ["re_run_correction"],
            "source": "main_stage1_page",
        })
    except Exception as err:
        logger.warning(f"[ROM Router] Failed to save DSPy feedback: {err}")

    return {
        "status": "success",
        "stage1": data["stage1"],
        "rom_data": data,
        "feedback_id": feedback_id
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
            "technical_terms": "Technical Terms",
            "dates": "Dates",
            "numbers": "Numbers/Quantities",
            "references": "References",
            "action_items": "Action Items",
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

@router.get("/{recording_id}/previous-meetings")
async def get_previous_meetings_list(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Return a list of other meetings for the current user that have generated Stage 2 points,
    along with metadata (meeting name, date, stage2 points count).
    """
    user_id = _validate_user_id(current_user)
    query = text("""
        SELECT r.id, r.filename, r.title, r.created_at, COALESCE(rm.rom_data, r.rom_data) AS rom_data
        FROM recordings r
        LEFT JOIN rom_metadata rm ON r.id = rm.recording_id
        WHERE r.user_id = :uid AND r.id != :cur_id
        ORDER BY r.created_at DESC
    """)
    rows = (await db.execute(query, {"uid": user_id, "cur_id": recording_id})).fetchall()

    meetings = []
    for row in rows:
        rec_id = row[0]
        filename = row[1]
        title = row[2]
        created_at = row[3]
        rom_data_raw = row[4]

        stage2_count = 0
        if rom_data_raw:
            try:
                rom_data = json.loads(rom_data_raw) if isinstance(rom_data_raw, str) else rom_data_raw
                stage2_pts = rom_data.get("stage2", {}).get("polished_points", [])
                stage2_count = len(stage2_pts)
            except Exception:
                pass

        meeting_name = title or filename or f"Meeting {rec_id[:8]}"
        date_str = str(created_at)[:10] if created_at else ""

        meetings.append({
            "id": rec_id,
            "name": meeting_name,
            "title": title or filename or "",
            "date": date_str,
            "stage2_count": stage2_count,
            "has_stage2": stage2_count > 0,
        })

    return {"meetings": meetings}


@router.post("/{recording_id}/stage2/generate")
async def generate_stage2(recording_id: str, req: Stage2Request, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    points = data.get("stage1", {}).get("discussion_points", [])
    
    if not points:
        raise HTTPException(status_code=400, detail="Stage 1 must be completed first")

    # Fetch transcript from DB to enable Stage 2 Validation Pass
    rec_row = await _get_recording_or_404(recording_id, user_id, db)
    transcript_raw = rec_row.get("transcript") or "[]"
    transcript_for_validation = from_json(transcript_raw) if transcript_raw else []
    r = await db.execute(
        text("SELECT rom_parallel_window_processing, rom_min_similarity_threshold, rom_stage2_process_all_together, rom_separate_action_extraction, rom_pipeline_mode FROM user_settings WHERE user_id = :uid"),
        {"uid": user_id},
    )
    us_row = r.fetchone()
    parallel_concurrency = us_row[0] if us_row and us_row[0] is not None else 2
    
    # Priority: explicitly passed in request body -> DB user_settings -> default 0.80
    min_sim_thresh = req.min_similarity_threshold
    if min_sim_thresh is None and us_row and us_row[1] is not None:
        min_sim_thresh = us_row[1]

    proc_all_together = req.process_all_together
    if proc_all_together is None and us_row and us_row[2] is not None:
        proc_all_together = bool(us_row[2])
    elif proc_all_together is None:
        proc_all_together = False

    sep_action_extraction = bool(us_row[3]) if us_row and us_row[3] is not None else False

    # rom_pipeline_mode: "base" (default) uses base prompts; "dspy" enables trained variant check
    pipeline_mode = str(us_row[4] or "base") if us_row and us_row[4] is not None else "base"
    use_dspy_mode = (pipeline_mode == "dspy")
    logger.info(f"[ROM Router] Stage 2: rom_pipeline_mode={pipeline_mode!r}, use_dspy_mode={use_dspy_mode}")

    meeting_name = rec_row.get("title") or rec_row.get("filename") or "Meeting"
    created_at_val = rec_row.get("created_at")
    meeting_date = str(created_at_val)[:10] if created_at_val else ""

    loop = asyncio.get_event_loop()
    polished = await loop.run_in_executor(
        None,
        lambda: rom_service.enhance_discussion_points(
            points, recording_id, user_id,
            req.meeting_context_top_k, req.global_context_top_k,
            req.discussion_window_size,
            parallel_window_processing=parallel_concurrency,
            min_similarity_threshold=min_sim_thresh,
            transcript=transcript_for_validation,
            process_all_together=proc_all_together,
            separate_action_extraction=sep_action_extraction,
            reference_example_points=req.reference_example_points,
            previous_meeting_mode=req.previous_meeting_mode,
            previous_meeting_id=req.previous_meeting_id,
            previous_meeting_top_k=req.previous_meeting_top_k,
            meeting_name=meeting_name,
            meeting_date=meeting_date,
            use_dspy_mode=use_dspy_mode,
        )
    )
    
    # Clear Stage 2 edit history when Stage 2 is regenerated (fresh version)
    await _clear_stage2_edit_history(db, recording_id, user_id)

    data["stage2"] = {
        "status": "done",
        "polished_points": polished,
        "edit_history": None,
    }
    
    # Clear Stage 2 outdated warning since Stage 2 was just regenerated
    data.setdefault("outdated_warnings", {})
    data["outdated_warnings"].pop("stage2", None)

    # Preserve downstream stages and mark as outdated with warnings
    if data.get("stage3"):
        data["outdated_warnings"]["stage3"] = "Stage 2 discussion points were regenerated. Existing agenda data may be outdated."
    if data.get("final_rom"):
        data["outdated_warnings"]["final_rom"] = "Stage 2 discussion points were regenerated. Existing Final ROM may be outdated."
    
    await _save_rom_data(recording_id, user_id, data, db)
    return {
        "status": "success",
        "stage2": data["stage2"],
        "rom_data": data
    }


@router.post("/{recording_id}/stage2/preview-context")
async def preview_stage2_context(
    recording_id: str,
    req: Stage2Request,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Preview context retrieved for all Stage 2 point groups without running LLM enhancement or altering points."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    points = data.get("stage1", {}).get("discussion_points", [])

    if not points:
        raise HTTPException(status_code=400, detail="Stage 1 must be completed first to preview context")

    r = await db.execute(
        text("SELECT rom_min_similarity_threshold, rom_stage2_process_all_together FROM user_settings WHERE user_id = :uid"),
        {"uid": user_id},
    )
    us_row = r.fetchone()

    min_sim_thresh = req.min_similarity_threshold
    if min_sim_thresh is None and us_row and us_row[0] is not None:
        min_sim_thresh = us_row[0]

    proc_all_together = req.process_all_together
    if proc_all_together is None and us_row and us_row[1] is not None:
        proc_all_together = bool(us_row[1])
    elif proc_all_together is None:
        proc_all_together = False

    loop = asyncio.get_event_loop()
    preview_result = await loop.run_in_executor(
        None,
        lambda: rom_service.preview_stage2_context(
            discussion_points=points,
            recording_id=recording_id,
            user_id=user_id,
            meeting_top_k=req.meeting_context_top_k,
            global_top_k=req.global_context_top_k,
            discussion_window_size=req.discussion_window_size,
            min_similarity_threshold=min_sim_thresh,
            process_all_together=proc_all_together,
            previous_meeting_mode=req.previous_meeting_mode,
            previous_meeting_id=req.previous_meeting_id,
            previous_meeting_top_k=req.previous_meeting_top_k,
            max_preview_groups=3,
        )
    )

    return {
        "status": "success",
        "groups": preview_result.get("groups", []),
        "total_groups": preview_result.get("total_groups", 0),
        "preview_groups_count": preview_result.get("preview_groups_count", len(preview_result.get("groups", []))),
        "total_points": preview_result.get("total_points", 0),
    }


@router.get("/stage2/example-points")
async def get_stage2_example_points_endpoint(current_user: dict = Depends(get_current_user)):
    """Get persisted style-only Stage 2 reference example points."""
    _validate_user_id(current_user)
    try:
        from services.training.stage2_training_service import load_example_points
        pts = load_example_points()
    except Exception:
        pts = []
    return {"points": pts, "count": len(pts)}


@router.post("/stage2/example-points")
async def save_stage2_example_points_endpoint(
    req: Stage2ExamplePointsBody,
    current_user: dict = Depends(get_current_user),
):
    """Save style-only Stage 2 reference example points."""
    _validate_user_id(current_user)
    try:
        from services.training.stage2_training_service import save_example_points
        save_example_points(req.points)
    except Exception as e:
        logger.warning(f"[ROM Router] Failed to save example points: {e}")
    return {"saved": True, "count": len(req.points)}

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
        if not isinstance(retrieved, dict):
            retrieved = {}
        meeting_chunks = retrieved.get("meeting_chunks", [])
        global_chunks = retrieved.get("global_chunks", [])
        previous_chunks = retrieved.get("previous_meeting_chunks", [])
        cur_report = retrieved.get("context_usage_report", p.get("context_usage_report", {}))
        if not isinstance(cur_report, dict):
            cur_report = {}

        context_added = False

        if meeting_chunks:
            m_texts = [
                (c.get("_text") or c.get("text") or c.get("content") or c.get("chunk") or "").strip()
                for c in meeting_chunks
            ]
            m_texts = [t for t in m_texts if t]
            if m_texts:
                add_bold_label(col1_cell, "Meeting Context", "\n\n".join(m_texts))
                context_added = True

        if global_chunks:
            g_texts = [
                (c.get("_text") or c.get("text") or c.get("content") or c.get("chunk") or "").strip()
                for c in global_chunks
            ]
            g_texts = [t for t in g_texts if t]
            if g_texts:
                add_bold_label(col1_cell, "Global Context", "\n\n".join(g_texts))
                context_added = True

        if previous_chunks:
            p_texts = [
                f"[Previous: {c.get('meeting_name', 'Meeting')}] {(c.get('_text') or c.get('text') or '').strip()}"
                for c in previous_chunks
            ]
            p_texts = [t for t in p_texts if t.strip()]
            if p_texts:
                add_bold_label(col1_cell, "Previous Meeting Stage 2 Context", "\n\n".join(p_texts))
                context_added = True

        # Show which documents were referenced
        docs_used = cur_report.get("documents", [])
        if docs_used:
            add_bold_label(col1_cell, "Source Documents", ", ".join(str(d) for d in docs_used))
            context_added = True

        # Show context usage flags if context was retrieved but no text chunks
        if not context_added:
            m_used = cur_report.get("meeting_context_used", False)
            g_used = cur_report.get("global_context_used", False)
            p_used = cur_report.get("previous_meeting_context_used", False)
            c_used = cur_report.get("context_added", False)
            if m_used or g_used or p_used or c_used:
                status_parts = []
                if m_used:
                    status_parts.append("Meeting context retrieved")
                if g_used:
                    status_parts.append("Global context retrieved")
                if p_used:
                    status_parts.append("Previous meeting Stage 2 context retrieved")
                if c_used:
                    status_parts.append("Context applied to point")
                add_bold_label(col1_cell, "Context Status", "; ".join(status_parts))
                context_added = True

        if not context_added:
            col1_cell.paragraphs[0].add_run("No additional context retrieved.")

        # Col 2: Enhanced Discussion Point (Stage 2)
        col2_cell = row_cells[2]
        col2_cell.text = ""
        add_bold_label(col2_cell, "Enhanced Point", p.get('polished_text', ''))

        field_labels = {
            "speakers": "Speakers",
            "technical_terms": "Technical Terms",
            "dates": "Dates",
            "numbers": "Numbers/Quantities",
            "references": "References",
            "action_items": "Action Items",
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
    # Clear Stage 3 outdated warning and mark Final ROM as outdated instead of deleting
    data.setdefault("outdated_warnings", {})
    data["outdated_warnings"].pop("stage3", None)
    if data.get("final_rom"):
        data["outdated_warnings"]["final_rom"] = "Stage 3 agendas were regenerated. Existing Final ROM may be outdated."

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

    # Extract file text supporting DOCX, PDF, TXT, OCR, etc.
    try:
        content = await file.read()
        filename = file.filename or "upload.txt"
        from services.doc_extractor import extract_text_from_file
        import tempfile
        import os
        ext = os.path.splitext(filename.lower())[1] or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            text_content = extract_text_from_file(tmp_path, filename)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        if not text_content or not text_content.strip():
            try:
                text_content = content.decode("utf-8", errors="replace")
            except Exception:
                text_content = content.decode("latin-1", errors="replace")
    except Exception as e:
        logger.error(f"[ROM UploadDoc] Failed to extract text from '{file.filename}': {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if not text_content or not text_content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file appears to be empty or unreadable. Please upload a valid text, PDF, or Word document.")

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

    # Synchronize agendas into final_rom and final_rom_versions while preserving existing discussion points
    if "final_rom" in data and isinstance(data["final_rom"].get("agendas"), list):
        final_agendas = data["final_rom"]["agendas"]
        existing_by_id = {fa.get("agenda_id"): fa for fa in final_agendas}
        synced_final = []
        for a in req.agendas:
            aid = a.get("agenda_id")
            if aid in existing_by_id:
                fa = existing_by_id[aid]
                fa["title"] = a.get("title", fa.get("title"))
                fa["description"] = a.get("description", fa.get("description"))
                if a.get("presenter") or a.get("speaker"):
                    fa["presenter"] = a.get("presenter") or a.get("speaker")
                    fa["speaker"] = fa["presenter"]
                synced_final.append(fa)
            else:
                new_fa = dict(a)
                new_fa["discussion_points"] = []
                synced_final.append(new_fa)
        data["final_rom"]["agendas"] = synced_final

        if "final_rom_versions" in data and isinstance(data["final_rom_versions"], dict):
            for v_name, v_rom in data["final_rom_versions"].items():
                if isinstance(v_rom, dict) and isinstance(v_rom.get("agendas"), list):
                    v_by_id = {fa.get("agenda_id"): fa for fa in v_rom["agendas"]}
                    synced_v = []
                    for a in req.agendas:
                        aid = a.get("agenda_id")
                        if aid in v_by_id:
                            fa = v_by_id[aid]
                            fa["title"] = a.get("title", fa.get("title"))
                            fa["description"] = a.get("description", fa.get("description"))
                            if a.get("presenter") or a.get("speaker"):
                                fa["presenter"] = a.get("presenter") or a.get("speaker")
                                fa["speaker"] = fa["presenter"]
                            synced_v.append(fa)
                        else:
                            new_fa = dict(a)
                            new_fa["discussion_points"] = []
                            synced_v.append(new_fa)
    if data.get("final_rom"):
        data.setdefault("outdated_warnings", {})["final_rom"] = "Stage 3 agendas were modified. Existing Final ROM mapping may be outdated."

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

    return {"status": "success", "agendas": req.agendas, "rom_data": data}



async def _populate_action_points_for_agendas(
    agendas: List[dict],
    polished_points: List[dict],
    stage1_points: List[dict],
    recording_id: str = "",
) -> List[dict]:
    """Collect or generate action points for discussion points within agendas.
    1. First checks for existing action_items on Stage 2 polished points and Stage 1 discussion points.
    2. If no action items exist anywhere, falls back to running provider.extract_actions_from_enhanced_points
       to extract action items from polished points in chunks and map them to points by source_point_id.
    """
    from services.rom_service import normalize_action_item, format_action_point_display_text

    polished_by_id = {p.get("id"): p for p in polished_points if p.get("id")}
    stage1_by_id = {p.get("id"): p for p in stage1_points if p.get("id")}

    for agenda in agendas:
        if agenda.get("skipped"):
            continue
        for dp in agenda.get("discussion_points", []):
            dp_id = dp.get("id", "")
            # Prefer Stage 2 polished point's action_items, fall back to Stage 1
            src_point = polished_by_id.get(dp_id) or stage1_by_id.get(dp_id)
            existing_actions = dp.get("action_items") or []
            if not existing_actions and src_point:
                existing_actions = src_point.get("action_items") or []
            # Also check original_point_id(s) for merged points
            if not existing_actions and src_point:
                orig_ids = []
                if src_point.get("original_point_id"):
                    orig_ids.append(src_point["original_point_id"])
                if src_point.get("original_point_ids"):
                    orig_ids.extend(src_point["original_point_ids"])
                for oid in orig_ids:
                    s1pt = stage1_by_id.get(oid)
                    if s1pt and s1pt.get("action_items"):
                        existing_actions = s1pt["action_items"]
                        break

            # Normalize and format action points
            action_points = []
            seen_tasks = set()
            if existing_actions and isinstance(existing_actions, list):
                for act in existing_actions:
                    norm = normalize_action_item(act)
                    if not norm.get("task"):
                        continue
                    display_text = format_action_point_display_text(norm)
                    task_key = (display_text or norm["task"]).strip().lower()
                    if task_key in seen_tasks:
                        continue
                    seen_tasks.add(task_key)
                    action_points.append({
                        "task": display_text or norm["task"],
                        "assignee": norm.get("assignee"),
                        "deadline": norm.get("deadline"),
                        "source_point_id": dp_id,
                    })
            dp["action_points"] = action_points

    # Check if total action points found is 0 across all agendas that have discussion points
    has_non_skipped_points = any(
        bool(dp) for a in agendas if not a.get("skipped") for dp in a.get("discussion_points", [])
    )
    total_actions = sum(len(dp.get("action_points", [])) for a in agendas for dp in a.get("discussion_points", []))
    if total_actions == 0 and polished_points and has_non_skipped_points:
        logger.info(f"[{recording_id}] No prior action items found. Running provider.extract_actions_from_enhanced_points fallback...")
        from services.ai_provider import get_provider
        provider = get_provider()
        loop = asyncio.get_event_loop()
        try:
            raw_extracted = await loop.run_in_executor(
                None,
                lambda: provider.extract_actions_from_enhanced_points(polished_points)
            )
        except Exception as _ex:
            logger.warning(f"[{recording_id}] Fallback action extraction failed ({_ex}).")
            raw_extracted = []
        finally:
            provider.unload_model()

        if raw_extracted:
            actions_by_point_id = {}
            unassigned_actions = []
            for act in raw_extracted:
                norm = normalize_action_item(act)
                if not norm.get("task"):
                    continue
                display_text = format_action_point_display_text(norm)
                src_id = act.get("source_point_id") or act.get("id")
                ap_item = {
                    "task": display_text or norm["task"],
                    "assignee": norm.get("assignee") or norm.get("owner"),
                    "deadline": norm.get("deadline"),
                    "source_point_id": src_id,
                }
                if src_id:
                    actions_by_point_id.setdefault(src_id, []).append(ap_item)
                else:
                    unassigned_actions.append(ap_item)

            for agenda in agendas:
                if agenda.get("skipped"):
                    continue
                for dp in agenda.get("discussion_points", []):
                    dp_id = dp.get("id", "")
                    if dp_id in actions_by_point_id:
                        dp["action_points"] = actions_by_point_id[dp_id]

            # If there were unassigned actions and some discussion points exist, assign to the first available point
            if unassigned_actions:
                for agenda in agendas:
                    if agenda.get("skipped"):
                        continue
                    dps = agenda.get("discussion_points", [])
                    if dps:
                        dps[0].setdefault("action_points", []).extend(unassigned_actions)
                        break

    return agendas


@router.post("/{recording_id}/stage3/generate-final-rom")
async def generate_final_rom_from_agendas(
    recording_id: str,
    req: GenerateFinalRomRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Step 2 of Stage 3: Map discussion points to agendas and generate Final ROM.
    Requires agendas to be created first via /stage3/create-agenda.
    Optionally includes agenda document points.
    Supports skipping agendas from point mapping and including action points."""
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

    # Determine effective order and timeline based on enable flags
    effective_order = req.discussion_order if req.enable_discussion_order else None
    effective_timeline = req.agenda_timeline if req.enable_agenda_timeline else None

    # Build set of skipped agenda IDs
    skipped_agendas_map = req.skipped_agendas or {}
    skipped_agenda_ids = set(skipped_agendas_map.keys()) if skipped_agendas_map else set()

    # Filter out skipped agendas from the agendas sent to point mapping
    mapping_agendas = [a for a in agendas if a.get("agenda_id", "") not in skipped_agenda_ids]
    mapping_expanded = [ea for ea in expanded_agendas if ea.get("agenda_id", "") not in skipped_agenda_ids]

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: rom_service.map_points_to_agendas(
            polished_points=polished_points,
            agendas=mapping_agendas,
            expanded_agendas=mapping_expanded,
            recording_id=recording_id,
            user_id=user_id,
            meeting_context_top_k=req.meeting_context_top_k,
            global_context_top_k=req.global_context_top_k,
            batch_size=req.batch_size,
            include_agenda_doc_points=req.include_agenda_doc_points,
            agenda_doc_points=agenda_doc_points,
            discussion_order=effective_order,
            agenda_timeline=effective_timeline,
        )
    )

    # Update stage3 with mapping results and guidance parameters
    stage3_updates = {
        "status": "done",
        "candidate_results": result.get("candidate_results", []),
        "batch_assignments": result.get("batch_assignments", []),
        "point_mappings": result.get("point_mappings", {}),
        "agenda_groups": result.get("agenda_groups", {}),
        "similarity_matrix": result.get("similarity_matrix", []),
        "enable_agenda_order": req.enable_discussion_order,
        "enable_agenda_timeline": req.enable_agenda_timeline,
    }
    if req.discussion_order is not None:
        stage3_updates["discussion_order"] = req.discussion_order
    if req.agenda_timeline is not None:
        stage3_updates["agenda_timeline"] = req.agenda_timeline
    if skipped_agendas_map:
        stage3_updates["skipped_agendas"] = skipped_agendas_map

    data["stage3"].update(stage3_updates)

    from services.rom_service import apply_speaker_mappings_to_final_rom

    # Build final ROM agendas: mapped agendas from result + re-insert skipped agendas at original positions
    mapped_agendas = result.get("final_rom_agendas", [])
    mapped_by_id = {a.get("agenda_id", ""): a for a in mapped_agendas}

    final_agendas_ordered = []
    for a in agendas:
        aid = a.get("agenda_id", "")
        if aid in skipped_agenda_ids:
            # Re-insert skipped agenda with skip metadata and no discussion points
            skip_info = skipped_agendas_map.get(aid, {})
            skipped_entry = dict(a)
            skipped_entry["discussion_points"] = []
            skipped_entry["skipped"] = True
            skipped_entry["skip_note"] = skip_info.get("note", "Keep this agenda if forward to next meeting")
            final_agendas_ordered.append(skipped_entry)
        elif aid in mapped_by_id:
            final_agendas_ordered.append(mapped_by_id[aid])
        else:
            # Agenda not in mapping result (shouldn't happen, but be safe)
            fallback = dict(a)
            fallback["discussion_points"] = []
            final_agendas_ordered.append(fallback)

    # Collect or generate action points for each agenda's discussion points when requested
    if req.include_action_points:
        final_agendas_ordered = await _populate_action_points_for_agendas(
            agendas=final_agendas_ordered,
            polished_points=polished_points,
            stage1_points=data.get("stage1", {}).get("discussion_points", []) if data.get("stage1") else [],
            recording_id=recording_id,
        )

    existing_final = data.get("final_rom") or {}
    raw_final = {
        **existing_final,
        "agendas": final_agendas_ordered,
        "include_agenda_doc_points": req.include_agenda_doc_points,
        "include_action_points": req.include_action_points,
        "skipped_agendas": skipped_agendas_map if skipped_agendas_map else None,
        "speaker_mappings": existing_final.get("speaker_mappings", {}),
    }
    import copy
    data["final_rom"] = apply_speaker_mappings_to_final_rom(raw_final)
    data.setdefault("final_rom_versions", {})
    data["final_rom_versions"]["long"] = copy.deepcopy(data["final_rom"])
    data["final_rom_active_version"] = "long"

    # Clear Final ROM outdated warning now that fresh Final ROM is generated
    if "outdated_warnings" in data and isinstance(data["outdated_warnings"], dict):
        data["outdated_warnings"].pop("final_rom", None)

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

        agenda_copy = dict(a)
        agenda_copy["discussion_points"] = agenda_points
        final_agendas.append(agenda_copy)

    import copy
    existing_final = data.get("final_rom") or {}
    data["final_rom"] = {**existing_final, "agendas": final_agendas}
    data.setdefault("final_rom_versions", {})
    data["final_rom_versions"]["long"] = copy.deepcopy(data["final_rom"])
    data["final_rom_active_version"] = "long"

    # Clear Final ROM outdated warning now that fresh Final ROM is generated
    if "outdated_warnings" in data and isinstance(data["outdated_warnings"], dict):
        data["outdated_warnings"].pop("final_rom", None)

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
    if isinstance(spk_mappings, dict) and spk_mappings:
        from services.speaker_sync import sync_global_speaker_rename
        await sync_global_speaker_rename(db, recording_id, user_id, spk_mappings, replace_all=False)
    data = await _get_rom_data(recording_id, user_id, db)
    from services.rom_service import apply_speaker_mappings_to_final_rom
    data["final_rom"] = apply_speaker_mappings_to_final_rom(final_rom)

    # Also update in final_rom_versions
    data.setdefault("final_rom_versions", {})
    active_ver = (req.version or data.get("final_rom_active_version") or "long").strip().lower()
    data["final_rom_versions"][active_ver] = data["final_rom"]
    data["final_rom_active_version"] = active_ver

    await _save_rom_data(recording_id, user_id, data, db)
    return {
        "status": "success",
        "final_rom": data["final_rom"],
        "final_rom_versions": data["final_rom_versions"],
        "active_version": active_ver
    }


@router.post("/{recording_id}/final/action-points")
async def toggle_final_rom_action_points(
    recording_id: str,
    req: ToggleActionPointsRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Toggle Include Action Points in Final ROM.
    When enabled:
      - Reuses existing action points from Stage 1/2 discussion points if present.
      - If no action points were previously extracted, extracts them using provider.extract_actions_from_enhanced_points
        and maps them to discussion points.
      - Sets include_action_points = True on final_rom and saves to db.
    When disabled:
      - Sets include_action_points = False on final_rom and saves to db.
    """
    user_id = _validate_user_id(current_user)
    await _get_recording_or_404(recording_id, user_id, db)
    data = await _get_rom_data(recording_id, user_id, db)

    final_rom = data.get("final_rom") or {}
    agendas = final_rom.get("agendas", [])
    if not agendas:
        raise HTTPException(status_code=400, detail="Final ROM has not been generated yet.")

    final_rom["include_action_points"] = req.include_action_points

    if req.include_action_points:
        # Check if any discussion points already have action_points populated
        has_action_points = any(
            bool(dp.get("action_points"))
            for a in agendas
            for dp in a.get("discussion_points", [])
        )
        if not has_action_points:
            polished_points = data.get("stage2", {}).get("polished_points", [])
            stage1_points = data.get("stage1", {}).get("discussion_points", []) if data.get("stage1") else []
            final_rom["agendas"] = await _populate_action_points_for_agendas(
                agendas=agendas,
                polished_points=polished_points,
                stage1_points=stage1_points,
                recording_id=recording_id,
            )

    # Also update across final_rom_versions
    data["final_rom"] = final_rom
    if "final_rom_versions" in data and isinstance(data["final_rom_versions"], dict):
        for ver_name, ver_rom in data["final_rom_versions"].items():
            if isinstance(ver_rom, dict):
                ver_rom["include_action_points"] = req.include_action_points
                if req.include_action_points and ver_name == "long":
                    ver_rom["agendas"] = final_rom.get("agendas", [])

    await _save_rom_data(recording_id, user_id, data, db)
    return {
        "status": "success",
        "include_action_points": req.include_action_points,
        "final_rom": data["final_rom"],
        "final_rom_versions": data.get("final_rom_versions", {}),
        "rom_data": data
    }


@router.post("/{recording_id}/final/extract-writing-rules")
async def extract_writing_rules(
    recording_id: str,
    req: ExtractWritingRulesRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Analyze a reference MoM/ROM document's writing style and return structured Writing Rules.
    The rules describe style, tone, formatting, and structure — never any content from the document.
    """
    user_id = _validate_user_id(current_user)
    await _get_recording_or_404(recording_id, user_id, db)  # Access control

    reference_text = (req.reference_text or "").strip()
    if not reference_text:
        raise HTTPException(status_code=400, detail="reference_text must not be empty.")

    loop = asyncio.get_event_loop()
    writing_rules = await loop.run_in_executor(
        None,
        lambda: rom_service.extract_writing_style_rules(reference_text=reference_text)
    )

    return {
        "status": "success",
        "writing_rules": writing_rules,
    }


@router.post("/{recording_id}/final/rewrite")
async def rewrite_final_rom(
    recording_id: str,
    req: RewriteRomRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Rewrite the writing style of the Final ROM discussion points using an LLM.
    Modes: 'window' (default), 'complete', 'reference' (uses writing_rules from uploaded doc).
    The rewritten result is returned but NOT saved; the user must explicitly save or revert.
    """
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    final_rom = data.get("final_rom") or {}
    if not final_rom or not final_rom.get("agendas"):
        raise HTTPException(
            status_code=400,
            detail="Final ROM must be generated before it can be rewritten."
        )

    instruction = (req.rewrite_instruction or "").strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="rewrite_instruction must not be empty.")

    valid_modes = {"window", "complete", "reference"}
    mode = (req.mode or "window").strip().lower()
    if mode not in valid_modes:
        mode = "window"

    loop = asyncio.get_event_loop()
    rewritten_final_rom = await loop.run_in_executor(
        None,
        lambda: rom_service.rewrite_final_rom(
            final_rom=final_rom,
            rewrite_instruction=instruction,
            mode=mode,
            window_size=req.window_size,
            writing_rules=req.writing_rules or "",
        )
    )

    return {
        "status": "success",
        "rewritten_final_rom": rewritten_final_rom,
    }


@router.post("/{recording_id}/final/generate-version")
async def generate_rom_version(
    recording_id: str,
    req: GenerateRomVersionRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Generate a Short or Medium condensed version of the Final ROM.
    Points are processed agenda-wise: all points for each agenda are passed together to the LLM.
    Correctly persists and saves the selected version without overwriting other versions.
    """
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    import copy

    # Initialize final_rom_versions dictionary
    data.setdefault("final_rom_versions", {})
    if "long" not in data["final_rom_versions"] and data.get("final_rom"):
        data["final_rom_versions"]["long"] = copy.deepcopy(data["final_rom"])

    # Base ROM is always the Long version with original Stage 2 points
    base_rom = (
        (req.base_final_rom if req.base_final_rom and req.base_final_rom.get("agendas") else None)
        or data.get("final_rom_versions", {}).get("long")
        or data.get("final_rom")
        or {}
    )
    if not base_rom or not base_rom.get("agendas"):
        raise HTTPException(
            status_code=400,
            detail="Final ROM must be generated before a version can be produced."
        )

    version = (req.version or "short").strip().lower()
    if version not in ("short", "medium", "long"):
        raise HTTPException(status_code=400, detail="version must be 'short', 'medium', or 'long'.")

    loop = asyncio.get_event_loop()
    rewritten_final_rom = await loop.run_in_executor(
        None,
        lambda: rom_service.generate_rom_version(
            final_rom=base_rom,
            version=version,
            writing_rules=req.writing_rules or "",
        )
    )

    # Save to data and persist to database
    data["final_rom_versions"][version] = rewritten_final_rom
    data["final_rom_active_version"] = version
    data["final_rom"] = rewritten_final_rom
    await _save_rom_data(recording_id, user_id, data, db)

    return {
        "status": "success",
        "version": version,
        "rewritten_final_rom": rewritten_final_rom,
        "final_rom": rewritten_final_rom,
        "final_rom_versions": data["final_rom_versions"],
        "rom_data": data,
    }


@router.post("/{recording_id}/final/select-version")
async def select_rom_version(
    recording_id: str,
    req: SelectRomVersionRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Select an active ROM version (long, short, medium).
    """
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    import copy

    version = (req.version or "long").strip().lower()
    if version not in ("long", "short", "medium"):
        raise HTTPException(status_code=400, detail="version must be 'long', 'short', or 'medium'.")

    versions = data.get("final_rom_versions") or {}
    if "long" not in versions and data.get("final_rom"):
        versions["long"] = copy.deepcopy(data["final_rom"])
        data["final_rom_versions"] = versions

    if version not in versions:
        raise HTTPException(status_code=404, detail=f"ROM version '{version}' has not been generated yet.")

    data["final_rom_active_version"] = version
    data["final_rom"] = versions[version]
    await _save_rom_data(recording_id, user_id, data, db)

    return {
        "status": "success",
        "version": version,
        "final_rom": versions[version],
        "final_rom_versions": versions,
        "rom_data": data,
    }


@router.get("/{recording_id}/final/download/docx")
async def download_final_docx(
    recording_id: str,
    version: Optional[str] = Query(None),
    include_action_points: Optional[bool] = Query(None),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    from services.rom_service import apply_speaker_mappings_to_final_rom, normalize_action_item, format_action_point_display_text

    target_rom = None
    if version and data.get("final_rom_versions", {}).get(version.strip().lower()):
        target_rom = data["final_rom_versions"][version.strip().lower()]
    if not target_rom:
        target_rom = data.get("final_rom") or {}

    target_rom = apply_speaker_mappings_to_final_rom(target_rom)
    raw_agendas = target_rom.get("agendas", [])
    effective_include_action_points = include_action_points if include_action_points is not None else target_rom.get("include_action_points", False)

    # Include agendas with discussion points OR skipped agendas (show skip note)
    agendas = [a for a in raw_agendas if a.get("discussion_points") or a.get("skipped")]

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

            for a in display_agendas:
                if a.get("skipped"):
                    # Show skipped agenda note in single-default format
                    row_cells = table.add_row().cells
                    row_cells[0].text = "—"
                    skip_note = a.get("skip_note", "Skipped")
                    dp_cell = row_cells[1]
                    dp_cell.text = ""
                    p_skip = dp_cell.add_paragraph()
                    r_skip = p_skip.add_run(f"[Skipped] {a.get('title', '')} — {skip_note}")
                    r_skip.italic = True
                    row_cells[2].text = "—"
                    continue

                pts = a.get("discussion_points", [])

                # Collect all action points belonging to this agenda
                agenda_action_points = []
                seen_action_keys = set()
                if isinstance(a.get("action_points"), list):
                    for ap in a["action_points"]:
                        task = ap.get("task") or ap.get("item") or ap.get("description") or ""
                        key = task.strip().lower()
                        if key and key not in seen_action_keys:
                            seen_action_keys.add(key)
                            agenda_action_points.append(ap)

                for pt in pts:
                    action_pts = list(pt.get("action_points") or [])
                    if not action_pts and pt.get("action_items"):
                        for ai in pt["action_items"]:
                            norm = normalize_action_item(ai)
                            if norm.get("task"):
                                action_pts.append({
                                    "task": format_action_point_display_text(norm) or norm["task"],
                                    "assignee": norm.get("assignee") or norm.get("owner"),
                                    "deadline": norm.get("deadline")
                                })
                    for ap in action_pts:
                        task = ap.get("task") or ap.get("item") or ap.get("description") or ""
                        key = task.strip().lower()
                        if key and key not in seen_action_keys:
                            seen_action_keys.add(key)
                            agenda_action_points.append(ap)

                # First: List all agenda discussion points
                for p_idx, pt in enumerate(pts):
                    pt_text = (pt.get("text") or pt.get("polished_text") or pt.get("discussion_point") or "").strip()
                    spk = pt.get("speaker")
                    if not spk and pt.get("speakers"):
                        spk = ", ".join(pt["speakers"]) if isinstance(pt["speakers"], list) else str(pt["speakers"])
                    spk_str = str(spk).strip() if spk else "-"

                    row_cells = table.add_row().cells
                    row_cells[0].text = str(p_idx + 1)
                    
                    dp_cell = row_cells[1]
                    dp_cell.text = ""
                    p_dp = dp_cell.add_paragraph()
                    p_dp.add_run(f"• {pt_text}" if pt_text else "•")

                    row_cells[2].text = spk_str

                # Then: Action Points for this agenda
                if effective_include_action_points and agenda_action_points:
                    # Add heading: Action Points
                    head_row_cells = table.add_row().cells
                    head_row_cells[0].text = ""
                    head_dp_cell = head_row_cells[1]
                    head_dp_cell.text = ""
                    p_head = head_dp_cell.add_paragraph()
                    r_head = p_head.add_run("Action Points")
                    r_head.bold = True
                    head_row_cells[2].text = ""

                    # List all action points belonging to that agenda underneath the heading
                    for ap in agenda_action_points:
                        ap_text = ap.get("task") or ap.get("item") or ap.get("description") or ""
                        ap_assignee = ap.get("assignee") or ap.get("owner") or ""
                        ap_deadline = ap.get("deadline") or ""
                        ap_owner_str = ap_assignee if ap_assignee else "-"
                        if ap_deadline and str(ap_deadline).lower() not in ("none", "n/a", "null", "asap", ""):
                            ap_owner_str += f" [Due: {ap_deadline}]"

                        ap_row_cells = table.add_row().cells
                        ap_row_cells[0].text = ""
                        ap_dp_cell = ap_row_cells[1]
                        ap_dp_cell.text = ""
                        p_ap = ap_dp_cell.add_paragraph()
                        p_ap.add_run(f"• {ap_text}")
                        ap_row_cells[2].text = ap_owner_str

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

            for a_idx, a in enumerate(display_agendas, 1):
                a_code = a.get("agenda_id") or f"A{a_idx}"
                a_title = a.get("title", "")
                agenda_label = f"{a_code} – {a_title}" if a_title else a_code

                # Handle skipped agendas: show agenda with skip note
                if a.get("skipped"):
                    skip_note = a.get("skip_note", "Skipped")
                    row_cells = table.add_row().cells
                    row_cells[0].text = str(a_idx)

                    agenda_cell = row_cells[1]
                    agenda_cell.text = ""
                    p_ag = agenda_cell.add_paragraph()
                    r_ag = p_ag.add_run(agenda_label)
                    r_ag.bold = True

                    dp_cell = row_cells[2]
                    dp_cell.text = ""
                    p_skip = dp_cell.add_paragraph()
                    r_skip = p_skip.add_run(f"[Skipped] {skip_note}")
                    r_skip.italic = True

                    row_cells[3].text = "—"
                    continue

                pts = a.get("discussion_points", [])

                # Collect all action points belonging to this agenda
                agenda_action_points = []
                seen_action_keys = set()
                if isinstance(a.get("action_points"), list):
                    for ap in a["action_points"]:
                        task = ap.get("task") or ap.get("item") or ap.get("description") or ""
                        key = task.strip().lower()
                        if key and key not in seen_action_keys:
                            seen_action_keys.add(key)
                            agenda_action_points.append(ap)

                for pt in pts:
                    action_pts = list(pt.get("action_points") or [])
                    if not action_pts and pt.get("action_items"):
                        for ai in pt["action_items"]:
                            norm = normalize_action_item(ai)
                            if norm.get("task"):
                                action_pts.append({
                                    "task": format_action_point_display_text(norm) or norm["task"],
                                    "assignee": norm.get("assignee") or norm.get("owner"),
                                    "deadline": norm.get("deadline")
                                })
                    for ap in action_pts:
                        task = ap.get("task") or ap.get("item") or ap.get("description") or ""
                        key = task.strip().lower()
                        if key and key not in seen_action_keys:
                            seen_action_keys.add(key)
                            agenda_action_points.append(ap)

                # First: List all agenda discussion points
                if pts:
                    for p_idx, pt in enumerate(pts):
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

                        # Discussion Points Column: clean bullet text (no point IDs)
                        dp_cell = row_cells[2]
                        dp_cell.text = ""
                        p_dp = dp_cell.add_paragraph()
                        p_dp.add_run(f"• {pt_text}" if pt_text else "•")

                        row_cells[3].text = spk_str
                else:
                    # Agenda with no discussion points
                    row_cells = table.add_row().cells
                    row_cells[0].text = str(a_idx)
                    agenda_cell = row_cells[1]
                    agenda_cell.text = ""
                    if agenda_label:
                        p_ag = agenda_cell.add_paragraph()
                        r_ag = p_ag.add_run(agenda_label)
                        r_ag.bold = True
                    row_cells[2].text = "—"
                    row_cells[3].text = "—"

                # Then: Action Points for this agenda
                if effective_include_action_points and agenda_action_points:
                    # Add heading: Action Points
                    head_row_cells = table.add_row().cells
                    head_row_cells[0].text = ""
                    head_row_cells[1].text = ""
                    dp_head_cell = head_row_cells[2]
                    dp_head_cell.text = ""
                    p_head = dp_head_cell.add_paragraph()
                    r_head = p_head.add_run("Action Points")
                    r_head.bold = True
                    head_row_cells[3].text = ""

                    # List all action points belonging to that agenda underneath the heading
                    for ap in agenda_action_points:
                        ap_text = ap.get("task") or ap.get("item") or ap.get("description") or ""
                        ap_assignee = ap.get("assignee") or ap.get("owner") or ""
                        ap_deadline = ap.get("deadline") or ""
                        ap_owner_str = ap_assignee if ap_assignee else "-"
                        if ap_deadline and str(ap_deadline).lower() not in ("none", "n/a", "null", "asap", ""):
                            ap_owner_str += f" [Due: {ap_deadline}]"

                        ap_row_cells = table.add_row().cells
                        ap_row_cells[0].text = ""
                        ap_row_cells[1].text = ""
                        ap_dp_cell = ap_row_cells[2]
                        ap_dp_cell.text = ""
                        p_ap = ap_dp_cell.add_paragraph()
                        p_ap.add_run(f"• {ap_text}")
                        ap_row_cells[3].text = ap_owner_str

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


# ── Agenda Management Endpoints ─────────────────────────────────────────

@router.post("/{recording_id}/stage3/agenda")
async def create_stage3_agenda(
    recording_id: str,
    req: CreateAgendaItemRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db)
):
    """Create a new agenda item in Stage 3 and reflect it across Final ROM."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    if "stage3" not in data:
        data["stage3"] = {}
    agendas = data["stage3"].get("agendas", [])
    if not isinstance(agendas, list):
        agendas = []

    # Determine unique agenda_id if not provided
    if req.agenda_id and req.agenda_id.strip():
        new_id = req.agenda_id.strip()
    else:
        max_num = 0
        for a in agendas:
            aid = str(a.get("agenda_id", "")).strip()
            if aid.startswith("A") and aid[1:].isdigit():
                max_num = max(max_num, int(aid[1:]))
        new_id = f"A{max_num + 1 if max_num > 0 else len(agendas) + 1}"

    presenter_val = req.presenter.strip() if req.presenter and req.presenter.strip() else None
    new_agenda = {
        "agenda_id": new_id,
        "title": req.title.strip(),
        "description": (req.description or "").strip(),
        "presenter": presenter_val,
        "speaker": presenter_val,
        "keywords": [],
        "discussion_points": [],
    }
    agendas.append(new_agenda)
    data["stage3"]["agendas"] = agendas

    # Also reflect in final_rom if it exists
    if "final_rom" in data and isinstance(data["final_rom"].get("agendas"), list):
        final_agendas = data["final_rom"]["agendas"]
        if not any(fa.get("agenda_id") == new_id for fa in final_agendas):
            final_agendas.append(dict(new_agenda))
            data["final_rom"]["agendas"] = final_agendas

    # Also reflect in final_rom_versions if it exists
    if "final_rom_versions" in data and isinstance(data["final_rom_versions"], dict):
        for v_name, v_rom in data["final_rom_versions"].items():
            if isinstance(v_rom, dict) and isinstance(v_rom.get("agendas"), list):
                if not any(fa.get("agenda_id") == new_id for fa in v_rom["agendas"]):
                    v_rom["agendas"].append(dict(new_agenda))

    if data.get("final_rom"):
        data.setdefault("outdated_warnings", {})["final_rom"] = "Stage 3 agendas were modified. Existing Final ROM mapping may be outdated."

    await _save_rom_data(recording_id, user_id, data, db)

    # Sync parsed_agenda_json in recordings table
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

    return {"status": "success", "agenda": new_agenda, "rom_data": data, "stage3": data["stage3"]}


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
        deleted_fa = next((fa for fa in final_agendas if fa.get("agenda_id") == agenda_id), None)
        final_agendas = [fa for fa in final_agendas if fa.get("agenda_id") != agenda_id]
        if deleted_fa and deleted_fa.get("discussion_points"):
            gen_fa = next((fa for fa in final_agendas if fa.get("agenda_id") == gen_discussion_id), None)
            if gen_fa:
                gen_fa.setdefault("discussion_points", []).extend(deleted_fa["discussion_points"])
        data["final_rom"]["agendas"] = final_agendas

    # Also update final_rom_versions
    if "final_rom_versions" in data and isinstance(data["final_rom_versions"], dict):
        for v_name, v_rom in data["final_rom_versions"].items():
            if isinstance(v_rom, dict) and isinstance(v_rom.get("agendas"), list):
                deleted_v = next((fa for fa in v_rom["agendas"] if fa.get("agenda_id") == agenda_id), None)
                v_agendas = [fa for fa in v_rom["agendas"] if fa.get("agenda_id") != agenda_id]
                if deleted_v and deleted_v.get("discussion_points"):
                    gen_v = next((fa for fa in v_agendas if fa.get("agenda_id") == gen_discussion_id), None)
                    if gen_v:
                        gen_v.setdefault("discussion_points", []).extend(deleted_v["discussion_points"])
                v_rom["agendas"] = v_agendas

    if data.get("final_rom"):
        data.setdefault("outdated_warnings", {})["final_rom"] = "Stage 3 agendas were modified. Existing Final ROM mapping may be outdated."

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

    before_s2_pts = list(data.get("stage2", {}).get("polished_points", []))

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

    # Record in edit history
    deleted_point = next((p for p in before_s2_pts if p.get("id") == point_id), None)
    if deleted_point:
        await _record_edit_history(
            db, recording_id, user_id, "delete",
            [point_id], [deleted_point], [],
            {"deletion_source": "manual"}
        )

    await _save_rom_data(recording_id, user_id, data, db)
    return {"status": "success", "rom_data": data}


# ── MOM Generation from Enhanced ROM ─────────────────────────────────────────

@router.post("/{recording_id}/stage3/generate-mom-from-rom")
async def generate_mom_from_rom(
    recording_id: str,
    action_chunk_size: Optional[int] = None,
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

    # Read separate_action_extraction & rom_action_generation_chunk_size from user_settings
    r_us = await db.execute(
        text("SELECT rom_separate_action_extraction, rom_action_generation_chunk_size FROM user_settings WHERE user_id = :uid"),
        {"uid": user_id},
    )
    us_row = r_us.mappings().fetchone() if r_us else None
    separate_action_extraction = bool(us_row["rom_separate_action_extraction"]) if us_row and us_row.get("rom_separate_action_extraction") is not None else False
    saved_chunk_size = int(us_row["rom_action_generation_chunk_size"]) if us_row and us_row.get("rom_action_generation_chunk_size") is not None else 10
    effective_chunk_size = action_chunk_size if (action_chunk_size is not None and action_chunk_size >= 1) else saved_chunk_size

    loop = asyncio.get_event_loop()
    enhanced_mom = await loop.run_in_executor(
        None,
        lambda: rom_service.generate_mom_from_enhanced_rom(
            polished_points=polished_points,
            recording_meta=rec_meta,
            recording_id=recording_id,
            user_id=user_id,
            separate_action_extraction=separate_action_extraction,
            action_chunk_size=effective_chunk_size,
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
                act_entry = {
                    "task": task_str,
                    "owner": owner_str,
                    "deadline": deadline_str,
                    "status": item.get("status", "open")
                }
                if "raw_json" in item:
                    act_entry["raw_json"] = item["raw_json"]
                norm_actions.append(act_entry)
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
                act_entry = {
                    "task": task_str,
                    "owner": owner_str,
                    "deadline": deadline_str,
                    "status": item.get("status", "open")
                }
                if "raw_json" in item:
                    act_entry["raw_json"] = item["raw_json"]
                norm_actions.append(act_entry)
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


@router.post("/{recording_id}/stage2/merge-points")
async def merge_stage2_points(
    recording_id: str,
    req: MergePointsRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Merge multiple Stage 2 discussion points into one using LLM."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    pts = data.get("stage2", {}).get("polished_points", [])
    
    # Collect points to merge, strictly preserving their original chronological order
    selected_set = set(req.point_ids)
    selected = [p for p in pts if p.get("id") in selected_set]
    if len(selected) < 2:
        raise HTTPException(400, f"Need at least 2 valid points to merge, found {len(selected)}")
    
    # Sort selected points by timeline_start and original array position
    selected = sorted(selected, key=lambda x: (x.get("timeline_start", 0), pts.index(x)))
    before_state = [dict(p) for p in selected]
    
    # Call LLM to merge
    from services.ai_provider import get_provider
    provider = get_provider()
    loop = asyncio.get_event_loop()
    merged_result = await loop.run_in_executor(
        None, lambda: provider.merge_discussion_points(selected)
    )
    
    polished_text = ""
    logger.info(f"[merge_stage2_points] Raw LLM result for {len(selected)} points: {merged_result}")
    if isinstance(merged_result, dict):
        raw_val = (
            merged_result.get("polished_text") or 
            merged_result.get("text") or 
            merged_result.get("discussion_point") or
            merged_result.get("summary") or ""
        )
        polished_text = provider._extract_clean_text_value(raw_val)
        logger.info(f"[merge_stage2_points] Extracted clean text (len={len(polished_text)}): {polished_text[:120]}...")
    
    if not polished_text:
        logger.warning("[merge_stage2_points] LLM output unparseable or empty. Using combined text fallback.")
        polished_text = " ".join(p.get("polished_text", "") for p in selected if p.get("polished_text"))
        logger.info(f"[merge_stage2_points] Fallback combined text: {polished_text[:120]}...")
    
    # Build merged point - inherit timeline from earliest to latest
    first_pt = next(p for p in pts if p["id"] == req.point_ids[0])
    first_idx = pts.index(first_pt)
    
    merged_point = {
        "id": str(uuid.uuid4()),
        "original_point_ids": [pid for p in selected for pid in (p.get("original_point_ids") or [p.get("original_point_id", p["id"])])],
        "original_point_id": selected[0].get("original_point_id", selected[0]["id"]),
        "polished_text": polished_text,
        "timeline_start": min(p.get("timeline_start", 0) for p in selected),
        "timeline_end": max(p.get("timeline_end", 0) for p in selected),
        "speakers": list(set(s for p in selected for s in (p.get("speakers") or []))),
        "technical_terms": (merged_result.get("technical_terms") if isinstance(merged_result, dict) else None) or list(set(t for p in selected for t in (p.get("technical_terms") or []))),
        "dates": (merged_result.get("dates") if isinstance(merged_result, dict) else None) or list(set(d for p in selected for d in (p.get("dates") or []))),
        "numbers": (merged_result.get("numbers") if isinstance(merged_result, dict) else None) or list(set(n for p in selected for n in (p.get("numbers") or []))),
        "references": (merged_result.get("references") if isinstance(merged_result, dict) else None) or list(set(r for p in selected for r in (p.get("references") or []))),
        "action_items": (merged_result.get("action_items") if isinstance(merged_result, dict) else None) or [ai for p in selected for ai in (p.get("action_items") or [])],
    }
    
    # Replace: remove selected points, insert merged at first position
    new_pts = [p for p in pts if p["id"] not in req.point_ids]
    new_pts.insert(first_idx, merged_point)
    data["stage2"]["polished_points"] = new_pts
    
    # Preserve downstream stages and mark as outdated with warnings
    if data.get("stage3"):
        data.setdefault("outdated_warnings", {})["stage3"] = "Stage 2 discussion points were merged/edited. Existing agenda mapping may be outdated."
    if data.get("final_rom"):
        data.setdefault("outdated_warnings", {})["final_rom"] = "Stage 2 discussion points were merged/edited. Existing Final ROM may be outdated."
    
    await _save_rom_data(recording_id, user_id, data, db)
    
    change_id = await _record_edit_history(
        db, recording_id, user_id, "merge",
        req.point_ids, before_state, [merged_point],
        {"merged_count": len(selected)}
    )
    
    return {"status": "success", "merged_point": merged_point, "change_id": change_id, "rom_data": data}

@router.post("/{recording_id}/stage2/find-replace/preview")
async def find_replace_preview(
    recording_id: str,
    req: FindPreviewRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Preview find occurrences across Stage 2 points."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    pts = data.get("stage2", {}).get("polished_points", [])
    
    total = 0
    points_affected = []
    for i, pt in enumerate(pts):
        text_content = pt.get("polished_text", "")
        count = text_content.count(req.find_text)
        if count > 0:
            total += count
            points_affected.append({"point_id": pt["id"], "count": count, "point_number": i + 1})
    
    return {"total_occurrences": total, "points_affected": points_affected}

@router.post("/{recording_id}/stage2/find-replace")
async def find_replace_apply(
    recording_id: str,
    req: FindReplaceRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Apply find and replace across Stage 2 points."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    pts = data.get("stage2", {}).get("polished_points", [])
    
    before_state = []
    after_state = []
    affected_ids = []
    total_replacements = 0
    
    for pt in pts:
        old_text = pt.get("polished_text", "")
        count = old_text.count(req.find_text)
        if count > 0:
            before_state.append(dict(pt))
            new_text = old_text.replace(req.find_text, req.replace_text)
            pt["polished_text"] = new_text
            after_state.append(dict(pt))
            affected_ids.append(pt["id"])
            total_replacements += count
    
    if total_replacements == 0:
        return {"status": "success", "affected_count": 0, "change_id": None, "rom_data": data}
    
    # Preserve downstream stages and mark as outdated with warnings
    if data.get("stage3"):
        data.setdefault("outdated_warnings", {})["stage3"] = "Stage 2 discussion points were edited. Existing agenda mapping may be outdated."
    if data.get("final_rom"):
        data.setdefault("outdated_warnings", {})["final_rom"] = "Stage 2 discussion points were edited. Existing Final ROM may be outdated."
    await _save_rom_data(recording_id, user_id, data, db)
    
    change_id = await _record_edit_history(
        db, recording_id, user_id, "find_replace",
        affected_ids, before_state, after_state,
        {"find_text": req.find_text, "replace_text": req.replace_text, "total_replacements": total_replacements}
    )
    
    return {"status": "success", "affected_count": total_replacements, "change_id": change_id, "rom_data": data}

@router.post("/{recording_id}/stage2/split-point")
async def split_stage2_point(
    recording_id: str,
    req: SplitPointRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Split a Stage 2 discussion point into two using LLM."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    pts = data.get("stage2", {}).get("polished_points", [])
    
    original = next((p for p in pts if p.get("id") == req.point_id), None)
    if not original:
        raise HTTPException(404, "Point not found")
    
    before_state = [dict(original)]
    original_idx = pts.index(original)
    
    from services.ai_provider import get_provider
    provider = get_provider()
    loop = asyncio.get_event_loop()
    split_result = await loop.run_in_executor(
        None, lambda: provider.split_discussion_point(to_json(original), req.selected_text)
    )
    
    split_points_raw = split_result.get("points", []) if isinstance(split_result, dict) else []
    if not split_points_raw or len(split_points_raw) < 2:
        logger.warning("[split_stage2_point] LLM split did not produce 2 points. Using text-based split fallback.")
        orig_text = original.get("polished_text", "")
        sel_text = req.selected_text.strip()
        rem_text = orig_text.replace(sel_text, "").strip()
        if not rem_text:
            rem_text = orig_text
        split_points_raw = [
            {"polished_text": rem_text, "speakers": original.get("speakers", [])},
            {"polished_text": sel_text, "speakers": original.get("speakers", [])}
        ]
    
    new_points = []
    for sp in split_points_raw[:2]:
        new_pt = {
            "id": str(uuid.uuid4()),
            "original_point_ids": original.get("original_point_ids", [original.get("original_point_id", original["id"])]),
            "original_point_id": original.get("original_point_id", original["id"]),
            "polished_text": provider._extract_clean_text_value(sp.get("polished_text", "")),
            "timeline_start": original.get("timeline_start", 0),
            "timeline_end": original.get("timeline_end", 0),
            "speakers": sp.get("speakers", original.get("speakers", [])),
            "technical_terms": sp.get("technical_terms", []),
            "dates": sp.get("dates", []),
            "numbers": sp.get("numbers", []),
            "references": sp.get("references", []),
            "action_items": sp.get("action_items", []),
        }
        new_points.append(new_pt)
    
    # Replace original with two new points
    pts.pop(original_idx)
    for i, np in enumerate(new_points):
        pts.insert(original_idx + i, np)
    
    data["stage2"]["polished_points"] = pts
    # Preserve downstream stages and mark as outdated with warnings
    if data.get("stage3"):
        data.setdefault("outdated_warnings", {})["stage3"] = "Stage 2 discussion points were edited. Existing agenda mapping may be outdated."
    if data.get("final_rom"):
        data.setdefault("outdated_warnings", {})["final_rom"] = "Stage 2 discussion points were edited. Existing Final ROM may be outdated."
    await _save_rom_data(recording_id, user_id, data, db)
    
    change_id = await _record_edit_history(
        db, recording_id, user_id, "split",
        [req.point_id], before_state, new_points,
        {"selected_text": req.selected_text}
    )
    
    return {"status": "success", "new_points": new_points, "change_id": change_id, "rom_data": data}

@router.post("/{recording_id}/stage2/delete-text")
async def delete_text_from_point(
    recording_id: str,
    req: DeleteTextRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Delete selected text from a Stage 2 point without LLM."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    pts = data.get("stage2", {}).get("polished_points", [])
    
    pt = next((p for p in pts if p.get("id") == req.point_id), None)
    if not pt:
        raise HTTPException(404, "Point not found")
    
    old_text = pt.get("polished_text", "")
    if req.text_to_delete not in old_text:
        raise HTTPException(400, "Selected text not found in point")
    
    before_state = [dict(pt)]
    pt["polished_text"] = old_text.replace(req.text_to_delete, "", 1).strip()
    after_state = [dict(pt)]
    
    # Preserve downstream stages and mark as outdated with warnings
    if data.get("stage3"):
        data.setdefault("outdated_warnings", {})["stage3"] = "Stage 2 discussion points were edited. Existing agenda mapping may be outdated."
    if data.get("final_rom"):
        data.setdefault("outdated_warnings", {})["final_rom"] = "Stage 2 discussion points were edited. Existing Final ROM may be outdated."
    await _save_rom_data(recording_id, user_id, data, db)
    
    change_id = await _record_edit_history(
        db, recording_id, user_id, "delete_text",
        [req.point_id], before_state, after_state,
        {"deleted_text": req.text_to_delete}
    )
    
    return {"status": "success", "updated_point": pt, "change_id": change_id, "rom_data": data}

@router.get("/{recording_id}/stage2/edit-history")
async def get_stage2_edit_history(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get Stage 2 edit history for a recording."""
    user_id = _validate_user_id(current_user)
    result = await db.execute(
        text(
            "SELECT * FROM stage2_edit_history "
            "WHERE recording_id = :rid AND user_id = :uid "
            "ORDER BY created_at DESC"
        ),
        {"rid": recording_id, "uid": user_id},
    )
    rows = result.fetchall()
    changes = []
    for row in rows:
        r = dict(row._mapping)
        r["point_ids_affected"] = from_json(r.get("point_ids_affected", "[]"))
        r["before_state"] = from_json(r.get("before_state", "[]"))
        r["after_state"] = from_json(r.get("after_state", "[]"))
        r["metadata"] = from_json(r.get("metadata", "{}"))
        changes.append(r)
    return {"changes": changes}

@router.post("/{recording_id}/stage2/edit-history/{change_id}/revert")
async def revert_stage2_change(
    recording_id: str,
    change_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Revert a Stage 2 edit change."""
    user_id = _validate_user_id(current_user)
    
    # Fetch the change
    result = await db.execute(
        text(
            "SELECT * FROM stage2_edit_history "
            "WHERE id = :cid AND recording_id = :rid AND user_id = :uid"
        ),
        {"cid": change_id, "rid": recording_id, "uid": user_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(404, "Change not found")
    
    change = dict(row._mapping)
    if change.get("is_reverted"):
        raise HTTPException(400, "Change already reverted")
    
    before_state = from_json(change.get("before_state", "[]"))
    after_state = from_json(change.get("after_state", "[]"))
    change_type = change.get("change_type")
    
    data = await _get_rom_data(recording_id, user_id, db)
    pts = data.get("stage2", {}).get("polished_points", [])
    
    # Get IDs of after-state points (points created by this change)
    after_ids = {p["id"] for p in after_state if isinstance(p, dict) and "id" in p}
    # Get IDs of before-state points (points that existed before this change)
    before_ids = {p["id"] for p in before_state if isinstance(p, dict) and "id" in p}
    
    if change_type == "merge":
        # Remove merged point(s), re-insert originals
        insert_idx = None
        for i, p in enumerate(pts):
            if p.get("id") in after_ids:
                insert_idx = i
                break
        new_pts = [p for p in pts if p.get("id") not in after_ids]
        if insert_idx is not None:
            for j, bp in enumerate(before_state):
                new_pts.insert(insert_idx + j, bp)
        else:
            new_pts.extend(before_state)
        pts = new_pts
    elif change_type == "split":
        # Remove split points, re-insert original
        insert_idx = None
        for i, p in enumerate(pts):
            if p.get("id") in after_ids:
                insert_idx = i
                break
        new_pts = [p for p in pts if p.get("id") not in after_ids]
        if insert_idx is not None and before_state:
            new_pts.insert(insert_idx, before_state[0])
        elif before_state:
            new_pts.append(before_state[0])
        pts = new_pts
    elif change_type == "delete":
        if before_state:
            pts.extend(before_state)
    elif change_type in ("find_replace", "delete_text", "manual_edit"):
        # Restore before_state text for affected points
        before_map = {p["id"]: p for p in before_state if isinstance(p, dict) and "id" in p}
        for i, p in enumerate(pts):
            if p.get("id") in before_map:
                pts[i] = before_map[p["id"]]
    
    # Preserve downstream stages and mark as outdated with warnings
    if data.get("stage3"):
        data.setdefault("outdated_warnings", {})["stage3"] = "Stage 2 discussion points were reverted/edited. Existing agenda mapping may be outdated."
    if data.get("final_rom"):
        data.setdefault("outdated_warnings", {})["final_rom"] = "Stage 2 discussion points were reverted/edited. Existing Final ROM may be outdated."
    await _save_rom_data(recording_id, user_id, data, db)
    
    # Mark as reverted
    await db.execute(
        text(
            "UPDATE stage2_edit_history SET is_reverted = 1, reverted_at = :rat "
            "WHERE id = :cid"
        ),
        {"cid": change_id, "rat": datetime.utcnow().isoformat()},
    )
    await db.commit()
    
    return {"status": "success", "reverted_change_id": change_id, "rom_data": data}

@router.post("/{recording_id}/stage2/edit-history/{change_id}/redo")
async def redo_stage2_change(
    recording_id: str,
    change_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Redo a previously reverted Stage 2 edit change."""
    user_id = _validate_user_id(current_user)
    
    result = await db.execute(
        text(
            "SELECT * FROM stage2_edit_history "
            "WHERE id = :cid AND recording_id = :rid AND user_id = :uid"
        ),
        {"cid": change_id, "rid": recording_id, "uid": user_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(404, "Change not found")
    
    change = dict(row._mapping)
    if not change.get("is_reverted"):
        raise HTTPException(400, "Change is not reverted; cannot redo")
    
    before_state = from_json(change.get("before_state", "[]"))
    after_state = from_json(change.get("after_state", "[]"))
    change_type = change.get("change_type")
    
    data = await _get_rom_data(recording_id, user_id, db)
    pts = data.get("stage2", {}).get("polished_points", [])
    
    before_ids = {p["id"] for p in before_state if isinstance(p, dict) and "id" in p}
    after_ids = {p["id"] for p in after_state if isinstance(p, dict) and "id" in p}
    
    if change_type == "merge":
        insert_idx = None
        for i, p in enumerate(pts):
            if p.get("id") in before_ids:
                insert_idx = i
                break
        new_pts = [p for p in pts if p.get("id") not in before_ids]
        if insert_idx is not None:
            for j, ap in enumerate(after_state):
                new_pts.insert(insert_idx + j, ap)
        else:
            new_pts.extend(after_state)
        pts = new_pts
    elif change_type == "split":
        insert_idx = None
        for i, p in enumerate(pts):
            if p.get("id") in before_ids:
                insert_idx = i
                break
        new_pts = [p for p in pts if p.get("id") not in before_ids]
        if insert_idx is not None:
            for j, ap in enumerate(after_state):
                new_pts.insert(insert_idx + j, ap)
        else:
            new_pts.extend(after_state)
        pts = new_pts
    elif change_type == "delete":
        pts = [p for p in pts if p.get("id") not in before_ids]
    elif change_type in ("find_replace", "delete_text", "manual_edit"):
        after_map = {p["id"]: p for p in after_state if isinstance(p, dict) and "id" in p}
        for i, p in enumerate(pts):
            if p.get("id") in after_map:
                pts[i] = after_map[p["id"]]
    
    data["stage2"]["polished_points"] = pts
    # Preserve downstream stages and mark as outdated with warnings
    if data.get("stage3"):
        data.setdefault("outdated_warnings", {})["stage3"] = "Stage 2 discussion points were redone/edited. Existing agenda mapping may be outdated."
    if data.get("final_rom"):
        data.setdefault("outdated_warnings", {})["final_rom"] = "Stage 2 discussion points were redone/edited. Existing Final ROM may be outdated."
    await _save_rom_data(recording_id, user_id, data, db)
    
    await db.execute(
        text(
            "UPDATE stage2_edit_history SET is_reverted = 0, reverted_at = NULL "
            "WHERE id = :cid"
        ),
        {"cid": change_id},
    )
    await db.commit()
    
    return {"status": "success", "redone_change_id": change_id, "rom_data": data}

@router.post("/{recording_id}/stage2/point/{point_id}/manual-edit")
async def manual_edit_stage2_point(
    recording_id: str,
    point_id: str,
    req: ManualEditPointRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Manually edit a Stage 2 discussion point."""
    user_id = _validate_user_id(current_user)
    data = await _get_rom_data(recording_id, user_id, db)
    pts = data.get("stage2", {}).get("polished_points", [])
    
    pt = next((p for p in pts if p.get("id") == point_id), None)
    if not pt:
        raise HTTPException(404, "Point not found")
    
    old_text = pt.get("polished_text", "")
    from services.ai_provider import get_provider
    provider = get_provider()
    new_text = provider._extract_clean_text_value(req.polished_text.strip())
    
    if old_text == new_text:
        return {"status": "success", "message": "No changes made", "rom_data": data}
    
    before_state = [dict(pt)]
    pt["polished_text"] = new_text
    after_state = [dict(pt)]
    
    # Preserve downstream stages and mark as outdated with warnings
    if data.get("stage3"):
        data.setdefault("outdated_warnings", {})["stage3"] = "Stage 2 discussion points were manually edited. Existing agenda mapping may be outdated."
    if data.get("final_rom"):
        data.setdefault("outdated_warnings", {})["final_rom"] = "Stage 2 discussion points were manually edited. Existing Final ROM may be outdated."
    await _save_rom_data(recording_id, user_id, data, db)
    
    change_id = await _record_edit_history(
        db, recording_id, user_id, "manual_edit",
        [point_id], before_state, after_state,
        {"old_text": old_text[:200], "new_text": new_text[:200]}
    )
    
    return {"status": "success", "updated_point": pt, "change_id": change_id, "rom_data": data}

@router.get("/{recording_id}/stage2/training-edits")
async def get_training_edits(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get non-reverted Stage 2 edits suitable for training."""
    user_id = _validate_user_id(current_user)
    result = await db.execute(
        text(
            "SELECT * FROM stage2_edit_history "
            "WHERE recording_id = :rid AND user_id = :uid AND is_reverted = 0 "
            "ORDER BY created_at ASC"
        ),
        {"rid": recording_id, "uid": user_id},
    )
    rows = result.fetchall()
    edits = []
    for row in rows:
        r = dict(row._mapping)
        before = from_json(r.get("before_state", "[]"))
        after = from_json(r.get("after_state", "[]"))
        original_texts = [p.get("polished_text", "") for p in before if isinstance(p, dict)]
        final_texts = [p.get("polished_text", "") for p in after if isinstance(p, dict)]
        edits.append({
            "change_id": r["id"],
            "change_type": r["change_type"],
            "original_text": "\n---\n".join(original_texts),
            "final_text": "\n---\n".join(final_texts),
            "metadata": from_json(r.get("metadata", "{}")),
            "created_at": r["created_at"],
        })
    return {"edits": edits}

@router.post("/{recording_id}/stage2/generate-training-from-edits")
async def generate_training_from_edits(
    recording_id: str,
    req: GenerateEditTrainingRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Generate training dataset from Stage 2 edit history."""
    user_id = _validate_user_id(current_user)
    
    # Fetch selected changes
    placeholders = ",".join(f":cid{i}" for i in range(len(req.selected_change_ids)))
    params = {"rid": recording_id, "uid": user_id}
    params.update({f"cid{i}": cid for i, cid in enumerate(req.selected_change_ids)})
    
    result = await db.execute(
        text(
            f"SELECT * FROM stage2_edit_history "
            f"WHERE recording_id = :rid AND user_id = :uid AND is_reverted = 0 "
            f"AND id IN ({placeholders}) "
            f"ORDER BY created_at ASC"
        ),
        params,
    )
    rows = result.fetchall()
    
    if not rows:
        raise HTTPException(404, "No valid changes found")
    
    samples = []
    for row in rows:
        r = dict(row._mapping)
        before = from_json(r.get("before_state", "[]"))
        after = from_json(r.get("after_state", "[]"))
        change_type = r["change_type"]
        
        original_texts = [p.get("polished_text", "") for p in before if isinstance(p, dict)]
        final_texts = [p.get("polished_text", "") for p in after if isinstance(p, dict)]
        
        input_text = f"[Change Type: {change_type}]\n" + "\n---\n".join(original_texts)
        output_text = "\n---\n".join(final_texts)
        
        if input_text.strip() and output_text.strip():
            samples.append({
                "inputs": {"stage": "stage_2", "change_type": change_type, "text": input_text},
                "target": output_text,
                "change_id": r["id"],
            })
    
    # Save as a dataset file
    dataset_id = str(uuid.uuid4())
    dataset = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "stages": ["stage_2"],
        "source_type": "edit_history",
        "source_meeting_id": recording_id,
        "total_samples": len(samples),
        "samples_by_stage": {"stage_2": len(samples)},
        "samples": samples,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    from services.training.training_storage_service import save_dataset
    save_dataset(dataset_id, dataset)
    
    # Also store in training_datasets table
    await db.execute(
        text(
            "INSERT INTO training_datasets "
            "(dataset_id, user_id, stages, source_type, source_meeting_id, total_samples, samples_by_stage, created_at) "
            "VALUES (:did, :uid, :stages, :st, :smid, :ts, :sbs, :cat)"
        ),
        {
            "did": dataset_id,
            "uid": user_id,
            "stages": to_json(["stage_2"]),
            "st": "edit_history",
            "smid": recording_id,
            "ts": len(samples),
            "sbs": to_json({"stage_2": len(samples)}),
            "cat": datetime.utcnow().isoformat(),
        },
    )
    await db.commit()
    
    return {
        "status": "success",
        "dataset_id": dataset_id,
        "sample_count": len(samples),
        "samples_preview": samples[:5],
    }

@router.delete("/{recording_id}")
async def delete_rom_data(recording_id: str, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    user_id = _validate_user_id(current_user)
    await db.execute(
        text("UPDATE recordings SET rom_data = NULL WHERE id = :id AND user_id = :uid"),
        {"id": recording_id, "uid": user_id}
    )
    await db.commit()
    return {"status": "success"}
