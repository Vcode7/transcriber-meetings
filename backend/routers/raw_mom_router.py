"""
raw_mom_router.py — RAG-based Raw MoM extraction endpoints.

This router is COMPLETELY INDEPENDENT of the existing /mom router.
It does NOT modify or read from the minutes_of_meeting table.
Raw MoM data is stored in the recordings.raw_mom column.

Endpoints
---------
POST /raw-mom/{recording_id}/retrieve        — Retrieve evidence only (no LLM)
POST /raw-mom/{recording_id}/generate        — Accept pre-assembled evidence, call LLM
POST /raw-mom/{recording_id}/generate-full   — Run the full RAG+LLM pipeline (legacy)
GET  /raw-mom/{recording_id}                 — Fetch stored raw_mom JSON
DELETE /raw-mom/{recording_id}               — Clear raw_mom for a recording
GET  /raw-mom/{recording_id}/status          — Pipeline status (embedded flags, etc.)
GET  /raw-mom/{recording_id}/download/docx   — Download raw_mom as DOCX
"""
from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from database import get_db, dt_to_str, from_json
from routers.auth import get_current_user
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/raw-mom", tags=["raw-mom"])


# ── Pydantic request models ───────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    k_transcript: int = Field(default=8, ge=0, le=30)
    k_meeting: int = Field(default=4, ge=0, le=30)
    k_global: int = Field(default=2, ge=0, le=30)
    relative_similarity_cutoff: float = Field(default=0.01, ge=0.0, le=1.0)
    char_limit: int = Field(default=15000, ge=1000, le=100000)
    force_reembed: bool = False


class AgendaEvidenceItem(BaseModel):
    topic: str
    speaker: Optional[str] = None
    evidence: str  # pre-assembled evidence string from frontend


class GenerateFromEvidenceRequest(BaseModel):
    agendas: List[AgendaEvidenceItem]


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_recording_or_404(recording_id: str, user_id: str) -> dict:
    """Fetch recording row or raise 404."""
    async with get_db() as db:
        r = await db.execute(
            text("""
                SELECT id, filename, created_at, duration, speakers_detected,
                       transcript, raw_mom, transcript_embedded, meeting_context_embedded,
                       agenda_summary
                FROM recordings
                WHERE id = :id AND user_id = :uid
            """),
            {"id": recording_id, "uid": user_id},
        )
        row = r.mappings().fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Recording not found")
    return dict(row)


# ── GET status ────────────────────────────────────────────────────────────────

@router.get("/{recording_id}/status")
async def get_raw_mom_status(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Return the RAG pipeline status for a recording:
    - Whether raw_mom has been generated
    - Embedding flags (transcript, meeting context)
    - Embedding model info
    """
    user_id = current_user["id"]
    rec = await _get_recording_or_404(recording_id, user_id)

    has_raw_mom = bool(rec.get("raw_mom"))
    raw_mom_summary = None
    if has_raw_mom:
        try:
            data = json.loads(rec["raw_mom"])
            agendas = data.get("meeting", {}).get("agendas", [])
            raw_mom_summary = {
                "agenda_count": len(agendas),
                "total_discussion_entries": sum(
                    len(a.get("discussion", [])) for a in agendas
                ),
            }
        except Exception:
            pass

    return {
        "recording_id": recording_id,
        "has_raw_mom": has_raw_mom,
        "raw_mom_summary": raw_mom_summary,
        "transcript_embedded": bool(rec.get("transcript_embedded")),
        "meeting_context_embedded": bool(rec.get("meeting_context_embedded")),
        "embedding_model": settings.QWEN_EMBEDDING_MODEL_NAME,
        "has_agenda": bool(rec.get("agenda_summary")),
    }


# ── GET raw_mom ───────────────────────────────────────────────────────────────

@router.get("/{recording_id}")
async def get_raw_mom(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Fetch the stored raw_mom JSON for a recording.

    Returns 404 if raw MoM has not been generated yet.
    """
    user_id = current_user["id"]
    rec = await _get_recording_or_404(recording_id, user_id)

    if not rec.get("raw_mom"):
        raise HTTPException(
            status_code=404,
            detail="Raw MoM has not been generated for this recording. "
                   "Use POST /raw-mom/{id}/generate to generate it.",
        )

    try:
        raw_mom_data = json.loads(rec["raw_mom"])
    except Exception:
        raise HTTPException(status_code=500, detail="Raw MoM data is corrupted")

    return raw_mom_data


# ── POST retrieve (retrieval only, no LLM) ───────────────────────────────────

@router.post("/{recording_id}/retrieve")
async def retrieve_evidence_endpoint(
    recording_id: str,
    body: RetrieveRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieve evidence chunks for all agenda items without running any LLM.

    This is the first step of the interactive Raw MoM Lab workflow:
    1. Parse agenda (from DB cache or re-parse)
    2. Embed transcript/meeting context if needed
    3. Run FAISS retrieval per agenda item
    4. Apply relative similarity filtering
    5. Return raw chunks with full metadata

    No LLM call is made — callers can inspect, filter, and reorder chunks
    before calling POST /generate with pre-assembled evidence.
    """
    user_id = current_user["id"]
    rec = await _get_recording_or_404(recording_id, user_id)

    transcript = from_json(rec.get("transcript"), [])
    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="No transcript available. Run transcription first.",
        )

    agenda_text: Optional[str] = rec.get("agenda_summary") or None
    if not agenda_text:
        agenda_text = await _load_raw_agenda_text(recording_id, user_id)

    import asyncio
    _loop = asyncio.get_running_loop()

    try:
        result = await _loop.run_in_executor(
            None,
            lambda: _run_retrieve_pipeline(
                recording_id=recording_id,
                user_id=user_id,
                transcript=transcript,
                agenda_text=agenda_text,
                force_reembed=body.force_reembed,
                k_transcript=body.k_transcript,
                k_meeting=body.k_meeting,
                k_global=body.k_global,
                relative_cutoff=body.relative_similarity_cutoff,
                char_limit=body.char_limit,
            ),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"[RawMoM] Retrieve pipeline failed for {recording_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    return result


# ── POST generate (from pre-assembled evidence) ───────────────────────────────

@router.post("/{recording_id}/generate")
async def generate_from_evidence_endpoint(
    recording_id: str,
    body: GenerateFromEvidenceRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Generate Raw MoM using pre-assembled evidence provided by the caller.

    This is the second step of the interactive Raw MoM Lab workflow.
    The caller provides the evidence string per agenda item (assembled from
    approved chunks selected in the UI). No FAISS retrieval is performed here.

    The result is saved to recordings.raw_mom and returned.
    """
    user_id = current_user["id"]
    await _get_recording_or_404(recording_id, user_id)

    if not body.agendas:
        raise HTTPException(status_code=400, detail="No agenda items provided.")

    import asyncio
    _loop = asyncio.get_running_loop()

    try:
        raw_mom = await _loop.run_in_executor(
            None,
            lambda: _run_generate_from_evidence(body.agendas),
        )
    except Exception as e:
        logger.error(f"[RawMoM] Generate-from-evidence failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Persist to DB
    raw_mom_json = json.dumps(raw_mom, ensure_ascii=False, default=str)
    async with get_db() as db:
        await db.execute(
            text("UPDATE recordings SET raw_mom = :raw_mom WHERE id = :id AND user_id = :uid"),
            {"raw_mom": raw_mom_json, "id": recording_id, "uid": user_id},
        )
        await db.commit()

    logger.info(f"[RawMoM] Saved evidence-based raw_mom for recording {recording_id}")
    return raw_mom


# ── POST generate-full (legacy full pipeline) ─────────────────────────────────

@router.post("/{recording_id}/generate-full")
async def generate_raw_mom_endpoint(
    recording_id: str,
    force_reembed: bool = False,
    current_user: dict = Depends(get_current_user),
):
    """
    Run the full RAG-based Raw MoM extraction pipeline for a recording.

    Pipeline
    --------
    1. Load transcript from DB
    2. Load agenda text from DB (agenda_summary column)
    3. Embed transcript → FAISS (if not already embedded)
    4. Embed meeting context attachments → FAISS (if not already embedded)
    5. Parse agenda → [{topic, speaker}]
    6. For each agenda item: retrieve evidence → LLM extraction
    7. Persist raw_mom JSON to recordings.raw_mom

    Parameters
    ----------
    force_reembed : If true, re-embed transcript and meeting context even if
                    they were previously embedded (useful after transcript updates).

    Returns
    -------
    The generated raw_mom JSON.
    """
    user_id = current_user["id"]
    rec = await _get_recording_or_404(recording_id, user_id)

    transcript = from_json(rec.get("transcript"), [])
    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="No transcript available. Run transcription first before generating Raw MoM.",
        )

    # Load agenda text — use the agenda_summary stored after attachment processing.
    # This is the raw/compressed text of agenda attachments, loaded via the
    # existing attachments pipeline (type='agenda').
    agenda_text: Optional[str] = rec.get("agenda_summary") or None

    # If no agenda_summary in recordings, try to load raw agenda attachment text directly
    if not agenda_text:
        agenda_text = await _load_raw_agenda_text(recording_id, user_id)

    logger.info(
        f"[RawMoM] Starting generation: recording={recording_id}, "
        f"transcript_segs={len(transcript)}, has_agenda={bool(agenda_text)}"
    )

    # Run the pipeline in a thread executor (CPU/GPU intensive)
    import asyncio
    _loop = asyncio.get_running_loop()

    try:
        raw_mom = await _loop.run_in_executor(
            None,
            lambda: _run_rag_pipeline(
                recording_id=recording_id,
                user_id=user_id,
                transcript=transcript,
                agenda_text=agenda_text,
                force_reembed_transcript=force_reembed,
                force_reembed_meeting=force_reembed,
            ),
        )
    except FileNotFoundError as e:
        # Embedding model not found
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"[RawMoM] Pipeline failed for {recording_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Raw MoM generation failed: {str(e)}",
        )

    # Persist to DB
    raw_mom_json = json.dumps(raw_mom, ensure_ascii=False, default=str)
    now = dt_to_str(datetime.now(timezone.utc))

    async with get_db() as db:
        await db.execute(
            text("UPDATE recordings SET raw_mom = :raw_mom WHERE id = :id AND user_id = :uid"),
            {"raw_mom": raw_mom_json, "id": recording_id, "uid": user_id},
        )
        await db.commit()

    logger.info(f"[RawMoM] Saved raw_mom for recording {recording_id}")
    return raw_mom


def _run_rag_pipeline(
    recording_id: str,
    user_id: str,
    transcript: list,
    agenda_text: Optional[str],
    force_reembed_transcript: bool,
    force_reembed_meeting: bool,
) -> dict:
    """Synchronous wrapper for the async-hostile RAG pipeline."""
    from services.rag_pipeline import generate_raw_mom
    return generate_raw_mom(
        recording_id=recording_id,
        user_id=user_id,
        transcript=transcript,
        agenda_text=agenda_text,
        force_reembed_transcript=force_reembed_transcript,
        force_reembed_meeting=force_reembed_meeting,
    )


async def _load_raw_agenda_text(recording_id: str, user_id: str) -> Optional[str]:
    """
    Load raw text from agenda attachments directly (fallback when agenda_summary is empty).
    Uses the doc_extractor to get text from the first agenda file.
    """
    try:
        from services.doc_extractor import extract_text_from_file
        async with get_db() as db:
            r = await db.execute(
                text(
                    "SELECT filename, file_path FROM recording_attachments "
                    "WHERE recording_id = :rid AND user_id = :uid AND type = 'agenda' "
                    "ORDER BY created_at ASC LIMIT 1"
                ),
                {"rid": recording_id, "uid": user_id},
            )
            att = r.mappings().fetchone()

        if not att:
            return None

        text_content = extract_text_from_file(att["file_path"], att["filename"])
        return text_content.strip() if text_content else None

    except Exception as e:
        logger.warning(f"[RawMoM] Could not load raw agenda text: {e}")
        return None


# ── GET download/docx ────────────────────────────────────────────────────────

@router.get("/{recording_id}/download/docx")
async def download_raw_mom_docx(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Generate and download a DOCX version of the Raw MoM as a formatted table.
    Columns: # | Agenda | Speaker | Discussion Points | Actions | Deadline | Owner
    """
    user_id = current_user["id"]
    rec = await _get_recording_or_404(recording_id, user_id)

    if not rec.get("raw_mom"):
        raise HTTPException(status_code=404, detail="Raw MoM has not been generated yet.")

    try:
        raw_mom_data = json.loads(rec["raw_mom"])
    except Exception:
        raise HTTPException(status_code=500, detail="Raw MoM data is corrupted")

    try:
        docx_bytes = _build_docx(raw_mom_data, rec.get("filename", "Recording"))
    except Exception as e:
        logger.error(f"[RawMoM] DOCX generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DOCX generation failed: {str(e)}")

    safe_name = rec.get("filename", "raw_mom").replace(" ", "_").rsplit(".", 1)[0]
    filename = f"{safe_name}_raw_mom.docx"

    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── DELETE raw_mom ────────────────────────────────────────────────────────────

@router.delete("/{recording_id}")
async def delete_raw_mom(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Clear the raw_mom data for a recording.

    This does NOT delete the FAISS embeddings (transcript/meeting context).
    Use this to re-run the pipeline cleanly.
    """
    user_id = current_user["id"]
    await _get_recording_or_404(recording_id, user_id)

    async with get_db() as db:
        await db.execute(
            text("UPDATE recordings SET raw_mom = NULL WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        await db.commit()

    return {"status": "cleared", "recording_id": recording_id}


# ── Sync pipeline runners ─────────────────────────────────────────────────────

def _run_retrieve_pipeline(
    recording_id: str,
    user_id: str,
    transcript: list,
    agenda_text: Optional[str],
    force_reembed: bool,
    k_transcript: int,
    k_meeting: int,
    k_global: int,
    relative_cutoff: float,
    char_limit: int,
) -> dict:
    """Synchronous retrieval-only pipeline (no LLM)."""
    from services.rag_pipeline import (
        embed_transcript, embed_meeting_context, parse_agenda_items,
        retrieve_evidence_raw, _transcript_embedded, _meeting_context_embedded,
        _mark_transcript_embedded, _mark_meeting_context_embedded,
        _load_parsed_agenda, _save_parsed_agenda,
    )
    from services.text_embedding_service import unload_text_embedder

    try:
        # Step 1: Ensure transcript is embedded
        if transcript and (force_reembed or not _transcript_embedded(recording_id)):
            count = embed_transcript(recording_id, transcript)
            if count > 0:
                _mark_transcript_embedded(recording_id, user_id)

        # Step 2: Ensure meeting context is embedded
        if force_reembed or not _meeting_context_embedded(recording_id):
            count = embed_meeting_context(recording_id, user_id)
            if count > 0:
                _mark_meeting_context_embedded(recording_id, user_id)

        # Step 3: Parse agenda (from cache or LLM)
        agenda_items = _load_parsed_agenda(recording_id)
        if not agenda_items:
            if agenda_text and agenda_text.strip():
                from services.ai_provider import get_provider
                provider = get_provider()
                agenda_items = provider.parse_agenda_items(agenda_text)
                if agenda_items:
                    _save_parsed_agenda(recording_id, user_id, agenda_items)
        if not agenda_items:
            agenda_items = [{"topic": "General Meeting Discussion", "speaker": None}]

        # Step 4: Retrieve raw chunks per agenda item
        agendas_with_chunks = []
        for item in agenda_items:
            topic = item.get("topic", "")
            if not topic:
                continue
            chunks = retrieve_evidence_raw(
                agenda_topic=topic,
                recording_id=recording_id,
                user_id=user_id,
                k_global=k_global,
                k_meeting=k_meeting,
                k_transcript=k_transcript,
                relative_cutoff=relative_cutoff,
                char_limit=char_limit,
            )
            agendas_with_chunks.append({
                "topic": topic,
                "speaker": item.get("speaker"),
                "is_procedural": chunks["is_procedural"],
                "transcript_chunks": chunks["transcript"],
                "meeting_chunks": chunks["meeting"],
                "global_chunks": chunks["global"],
            })

        return {
            "agendas": agendas_with_chunks,
            "retrieval_params": {
                "k_transcript": k_transcript,
                "k_meeting": k_meeting,
                "k_global": k_global,
                "relative_similarity_cutoff": relative_cutoff,
                "char_limit": char_limit,
            },
        }
    finally:
        # Unload embedding model to free VRAM
        try:
            unload_text_embedder()
        except Exception as e:
            logger.warning(f"[RawMoM] Failed to unload embedder: {e}")


def _run_generate_from_evidence(agendas: list) -> dict:
    """Synchronous LLM generation using pre-assembled evidence (no retrieval)."""
    import gc
    import torch
    from services.ai_provider import get_provider, QwenProvider

    provider = get_provider()
    processed_agendas = []

    try:
        for item in agendas:
            topic = item.topic
            speaker = item.speaker
            evidence = item.evidence

            try:
                result = provider.extract_raw_mom_for_agenda(
                    agenda_topic=topic,
                    agenda_speaker=speaker,
                    evidence=evidence,
                )
                processed_agendas.append(result)
                logger.info(
                    f"[RawMoM] '{topic[:40]}': "
                    f"{len(result.get('discussion', []))} discussion entries"
                )
            except Exception as e:
                logger.error(f"[RawMoM] Extraction failed for '{topic}': {e}")
                processed_agendas.append({
                    "agenda_topic": topic,
                    "agenda_speaker": speaker,
                    "discussion": [],
                })
            finally:
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        return {"meeting": {"agendas": processed_agendas}}
    finally:
        try:
            QwenProvider.unload_model()
        except Exception as e:
            logger.warning(f"[RawMoM] Failed to unload LLM: {e}")


def _build_docx(raw_mom_data: dict, recording_name: str) -> bytes:
    """Build a DOCX file from raw_mom JSON and return bytes."""
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
    except ImportError:
        raise RuntimeError(
            "python-docx is required for DOCX export. "
            "Install it with: pip install python-docx"
        )

    doc = Document()

    # Title
    title = doc.add_heading(f"Raw Minutes of Meeting", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"Recording: {recording_name}").alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}").alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("")

    agendas = raw_mom_data.get("meeting", {}).get("agendas", [])
    if not agendas:
        doc.add_paragraph("No agenda items found.")
    else:
        # Table headers
        table = doc.add_table(rows=1, cols=6)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        for i, header in enumerate(["#", "Agenda", "Speaker", "Discussion Points", "Actions", "Deadline / Owner"]):
            hdr[i].text = header
            run = hdr[i].paragraphs[0].runs[0]
            run.bold = True

        row_num = 1
        for agenda in agendas:
            topic = agenda.get("agenda_topic", "")
            speaker = agenda.get("agenda_speaker") or ""
            discussion = agenda.get("discussion", [])

            if not discussion:
                row = table.add_row().cells
                row[0].text = str(row_num)
                row[1].text = topic
                row[2].text = speaker
                row[3].text = ""
                row[4].text = ""
                row[5].text = ""
                row_num += 1
            else:
                for entry in discussion:
                    row = table.add_row().cells
                    row[0].text = str(row_num)
                    row[1].text = topic
                    row[2].text = str(entry.get("speaker") or speaker or "")
                    row[3].text = str(entry.get("point", ""))
                    action = entry.get("action", {}) or {}
                    action_desc = str(action.get("description") or "")
                    row[4].text = action_desc
                    deadline = str(action.get("deadline") or "")
                    owner = str(action.get("owner") or "")
                    row[5].text = f"{owner} / {deadline}".strip(" /") if (owner or deadline) else ""
                    row_num += 1

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()
