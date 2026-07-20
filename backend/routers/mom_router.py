"""Minutes of Meeting (MoM) router — generate, fetch, update, and export MoMs."""
import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text

from database import get_db, dt_to_str, to_json, from_json
from routers.auth import get_current_user
from services.llm import generate_mom
from config import settings
from tasks.pipeline import _filter_high_confidence_segments
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mom", tags=["mom"])


class ActionItem(BaseModel):
    task: str = ""
    owner: str = ""
    deadline: str = ""


class MoMData(BaseModel):
    title: str = ""
    date: str = ""
    duration: float = 0.0
    planned_start_time: str = ""
    actual_start_time: str = ""
    participants: List[str] = []
    introduction: str = ""
    points_discussed: List[str] = []
    action_items: List[ActionItem] = []
    conclusion: str = ""


def _normalize_action_items(items: list) -> list:
    """Ensure each action item is a {task, owner, deadline} dict."""
    result = []
    for item in items:
        if isinstance(item, dict):
            result.append({
                "task": str(item.get("task", "") or ""),
                "owner": str(item.get("owner", "Unassigned") or "Unassigned"),
                "deadline": str(item.get("deadline", "ASAP") or "ASAP"),
            })
        elif isinstance(item, str) and item.strip():
            result.append({"task": item.strip(), "owner": "Unassigned", "deadline": "ASAP"})
    return result


def _mom_row_to_dict(mom) -> dict:
    """Convert a SQLite row for MoM into a clean dict with JSON fields deserialized."""
    raw_action_items = from_json(mom["action_items"], [])
    return {
        "id": mom["id"],
        "recording_id": mom["recording_id"],
        "user_id": mom["user_id"],
        "title": mom.get("title") or "",
        "date": mom.get("date") or "",
        "duration": mom.get("duration") or 0,
        "planned_start_time": mom.get("planned_start_time") or "",
        "actual_start_time": mom.get("actual_start_time") or "",
        "participants": from_json(mom["participants"], []) or [],
        "introduction": mom.get("introduction") or "",
        "points_discussed": from_json(mom["points_discussed"], []) or [],
        "action_items": _normalize_action_items(raw_action_items),
        "conclusion": mom.get("conclusion") or "",
        "is_draft": bool(mom.get("is_draft", False)),
        "created_at": mom.get("created_at"),
        "updated_at": mom.get("updated_at"),
    }


@router.get("/{recording_id}")
async def get_mom(recording_id: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        mom = r.mappings().fetchone()

    if not mom:
        raise HTTPException(status_code=404, detail="MoM not found")

    return _mom_row_to_dict(mom)


@router.post("/{recording_id}/generate")
async def generate_mom_endpoint(recording_id: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]

    try:
        async with get_db() as db:
            r = await db.execute(
                text("""
                    SELECT transcript, raw_text, filename, created_at, duration,
                           speakers_detected, context_summary, context_summary_hash,
                           agenda_summary, reference_summary
                    FROM recordings WHERE id = :id AND user_id = :uid
                """),
                {"id": recording_id, "uid": user_id},
            )
            rec = r.mappings().fetchone()

        if not rec:
            raise HTTPException(status_code=404, detail="Recording not found")

        transcript = from_json(rec["transcript"], [])
        if not transcript:
            raise HTTPException(status_code=400, detail="No transcript available to summarize")

        # Filter low-confidence segments before generating MoM
        filtered_transcript = _filter_high_confidence_segments(transcript, settings.MIN_AVG_SEGMENT_CONFIDENCE)
        logger.info(
            f"[MoM] generate: using {len(filtered_transcript)}/{len(transcript)} high-confidence segments "
            f"for recording={recording_id}"
        )

        meta = {
            "filename": rec.get("filename", "Meeting Notes"),
            "created_at": rec.get("created_at", ""),
            "duration": rec.get("duration", 0),
            "speakers_detected": from_json(rec["speakers_detected"], []),
        }

        # Resolve context_summary (reuse cached or build fresh)
        import asyncio as _asyncio
        _loop = _asyncio.get_running_loop()
        from tasks.pipeline import _raw_text_hash
        from services.llm import build_context_summary as _build_ctx
        raw_text = rec.get("raw_text") or ""
        current_hash = _raw_text_hash(raw_text)
        ctx: str | None = None
        if rec.get("context_summary") and rec.get("context_summary_hash") == current_hash:
            ctx = rec["context_summary"]
            logger.info(f"[MoM] Reusing cached context_summary for {recording_id} ({len(ctx.split()):,} words)")
        else:
            logger.info(f"[MoM] Building fresh context_summary for {recording_id} ({len(raw_text)} chars)")
            try:
                ctx = await _loop.run_in_executor(None, _build_ctx, filtered_transcript)
                if ctx:
                    async with get_db() as db:
                        await db.execute(
                            text("UPDATE recordings SET context_summary = :ctx, context_summary_hash = :h "
                                 "WHERE id = :id AND user_id = :uid"),
                            {"ctx": ctx, "h": current_hash, "id": recording_id, "uid": user_id},
                        )
                        await db.commit()
            except Exception as ctx_err:
                logger.warning(f"[MoM] context_summary build failed (non-fatal): {ctx_err}")
                ctx = None

        # Retrieve stored agenda / reference summaries (uploaded via /attachments)
        agenda_summary: str | None = rec.get("agenda_summary") or None
        reference_summary: str | None = rec.get("reference_summary") or None
        if agenda_summary:
            logger.info(f"[MoM] Injecting agenda_summary ({len(agenda_summary)} chars) for {recording_id}")
        if reference_summary:
            logger.info(f"[MoM] Injecting reference_summary ({len(reference_summary)} chars) for {recording_id}")

        mom_data = await _loop.run_in_executor(
            None,
            lambda: generate_mom(
                filtered_transcript,
                meta,
                context=ctx or None,
                agenda_summary=agenda_summary,
                reference_summary=reference_summary,
            ),
        )

        now = datetime.now(timezone.utc)
        mom_id = str(uuid.uuid4())
        initial_version = [{"version": 1, "data": mom_data, "saved_at": dt_to_str(now)}]

        async with get_db() as db:
            # Check if MoM already exists (upsert pattern)
            r = await db.execute(
                text("SELECT id FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
                {"rid": recording_id, "uid": user_id},
            )
            existing = r.fetchone()

            if existing:
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
                        "title": mom_data.get("title", ""),
                        "date": mom_data.get("date", ""),
                        "duration": mom_data.get("duration", 0),
                        "planned_start_time": mom_data.get("planned_start_time", ""),
                        "actual_start_time": mom_data.get("actual_start_time", ""),
                        "participants": to_json(mom_data.get("participants", [])),
                        "introduction": mom_data.get("introduction", ""),
                        "points_discussed": to_json(mom_data.get("points_discussed", [])),
                        "action_items": to_json(mom_data.get("action_items", [])),
                        "conclusion": mom_data.get("conclusion", ""),
                        "versions": to_json(initial_version),
                        "updated_at": dt_to_str(now),
                        "rid": recording_id,
                        "uid": user_id,
                    },
                )
            else:
                await db.execute(
                    text("""
                        INSERT INTO minutes_of_meeting (
                            id, recording_id, user_id, title, date, duration,
                            planned_start_time, actual_start_time,
                            participants, introduction, points_discussed,
                            action_items, conclusion,
                            versions, is_draft, created_at, updated_at
                        )
                        VALUES (
                            :id, :rid, :uid, :title, :date, :duration,
                            :planned_start_time, :actual_start_time,
                            :participants, :introduction, :points_discussed,
                            :action_items, :conclusion,
                            :versions, 0, :created_at, :updated_at
                        )
                    """),
                    {
                        "id": mom_id,
                        "rid": recording_id,
                        "uid": user_id,
                        "title": mom_data.get("title", ""),
                        "date": mom_data.get("date", ""),
                        "duration": mom_data.get("duration", 0),
                        "planned_start_time": mom_data.get("planned_start_time", ""),
                        "actual_start_time": mom_data.get("actual_start_time", ""),
                        "participants": to_json(mom_data.get("participants", [])),
                        "introduction": mom_data.get("introduction", ""),
                        "points_discussed": to_json(mom_data.get("points_discussed", [])),
                        "action_items": to_json(mom_data.get("action_items", [])),
                        "conclusion": mom_data.get("conclusion", ""),
                        "versions": to_json(initial_version),
                        "created_at": dt_to_str(now),
                        "updated_at": dt_to_str(now),
                    },
                )
            await db.commit()

            # Fetch the saved record
            r2 = await db.execute(
                text("SELECT * FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
                {"rid": recording_id, "uid": user_id},
            )
            saved = r2.mappings().fetchone()

        return _mom_row_to_dict(saved)
    finally:
        from services.ai_provider import QwenProvider
        QwenProvider.unload_model()



@router.patch("/{recording_id}")
async def update_mom(recording_id: str, data: MoMData, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]

    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        mom = r.mappings().fetchone()
        if not mom:
            raise HTTPException(status_code=404, detail="MoM not found")

        now = datetime.now(timezone.utc)
        update_data = data.dict()

        # Version tracking: push a version if last version is older than 5 min
        versions = from_json(mom["versions"], [])
        last_version = versions[-1] if versions else None
        push_version = False
        if last_version:
            last_saved_str = last_version.get("saved_at")
            if last_saved_str:
                try:
                    last_saved = datetime.fromisoformat(last_saved_str)
                    if last_saved.tzinfo is None:
                        last_saved = last_saved.replace(tzinfo=timezone.utc)
                    if (now - last_saved).total_seconds() > 300:
                        push_version = True
                except Exception:
                    pass

        if push_version:
            new_version = {
                "version": len(versions) + 1,
                "data": update_data,
                "saved_at": dt_to_str(now),
            }
            versions.append(new_version)

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
                    versions = :versions, is_draft = 1, updated_at = :updated_at
                WHERE recording_id = :rid AND user_id = :uid
            """),
            {
                "title": update_data.get("title", ""),
                "date": update_data.get("date", ""),
                "duration": update_data.get("duration", 0),
                "planned_start_time": update_data.get("planned_start_time", ""),
                "actual_start_time": update_data.get("actual_start_time", ""),
                "participants": to_json(update_data.get("participants", [])),
                "introduction": update_data.get("introduction", ""),
                "points_discussed": to_json(update_data.get("points_discussed", [])),
                "action_items": to_json(update_data.get("action_items", [])),
                "conclusion": update_data.get("conclusion", ""),
                "versions": to_json(versions),
                "updated_at": dt_to_str(now),
                "rid": recording_id,
                "uid": user_id,
            },
        )
        await db.commit()

    return {"status": "success", "updated_at": dt_to_str(now)}


@router.get("/{recording_id}/versions")
async def get_mom_versions(recording_id: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text("SELECT versions FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        row = r.mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="MoM not found")

    return {"versions": from_json(row["versions"], [])}


@router.post("/{recording_id}/pdf")
async def export_mom_pdf(
    recording_id: str,
    data: Optional[MoMData] = None,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["id"]

    # Check if the input data is missing or effectively empty
    is_empty = True
    if data:
        d = data.dict()
        # If any of the main content fields are present, we consider it non-empty
        if (d.get("title") or d.get("introduction") or d.get("conclusion") or
            d.get("points_discussed") or d.get("action_items") or d.get("participants")):
            is_empty = False

    if is_empty:
        # Load MoM from database (which also verifies ownership)
        async with get_db() as db:
            r = await db.execute(
                text("SELECT * FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
                {"rid": recording_id, "uid": user_id},
            )
            mom_row = r.mappings().fetchone()

        if not mom_row:
            raise HTTPException(status_code=404, detail="MoM not found")
        mom = _mom_row_to_dict(mom_row)
    else:
        # Verify ownership / existence
        async with get_db() as db:
            r = await db.execute(
                text("SELECT id FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
                {"rid": recording_id, "uid": user_id},
            )
            exists = r.fetchone()

        if not exists:
            raise HTTPException(status_code=404, detail="MoM not found")

        # Use the caller-supplied data (live editor state)
        mom = data.dict()

    mom["action_items"] = _normalize_action_items(mom.get("action_items", []))

    # Build PDF in memory to avoid disk accumulation
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'],
        fontSize=22, textColor=colors.HexColor("#0f172a"),
        spaceAfter=8, fontName="Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle', parent=styles['Normal'],
        fontSize=11, textColor=colors.HexColor("#334155"),
        spaceAfter=4, fontName="Helvetica"
    )
    header_style = ParagraphStyle(
        'HeaderStyle', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor("#1e293b"),
        spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold",
    )
    normal_style = ParagraphStyle(
        'NormalStyle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor("#334155"),
        spaceAfter=6, fontName="Helvetica", leading=15
    )
    meta_style = ParagraphStyle(
        'MetaStyle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor("#64748b"),
        spaceAfter=4, fontName="Helvetica-Oblique"
    )

    def _hr():
        from reportlab.platypus import HRFlowable
        return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=6, spaceBefore=2)

    story = []

    # ── Title Block ──────────────────────────────────────────────────
    story.append(Paragraph(escape(mom.get("title", "Minutes of Meeting")), title_style))
    story.append(_hr())

    # ── Meeting Meta ──────────────────────────────────────────────────
    story.append(Paragraph(f"<b>Date:</b> {escape(mom.get('date', 'Unknown'))}", subtitle_style))
    members = ", ".join(mom.get("participants", []))
    story.append(Paragraph(f"<b>Members:</b> {escape(members) if members else 'N/A'}", subtitle_style))
    if mom.get("planned_start_time"):
        story.append(Paragraph(f"<b>Planned Starting Time:</b> {escape(mom['planned_start_time'])}", subtitle_style))
    if mom.get("actual_start_time"):
        story.append(Paragraph(f"<b>Actual Starting Time:</b> {escape(mom['actual_start_time'])}", subtitle_style))
    story.append(Spacer(1, 14))

    # ── Introduction ──────────────────────────────────────────────────
    if mom.get("introduction"):
        story.append(Paragraph("INTRODUCTION", header_style))
        story.append(_hr())
        story.append(Paragraph(escape(mom["introduction"]), normal_style))
        story.append(Spacer(1, 8))

    # ── Points Discussed ──────────────────────────────────────────────
    if mom.get("points_discussed"):
        story.append(Paragraph("POINTS DISCUSSED", header_style))
        story.append(_hr())
        items = [
            ListItem(Paragraph(escape(str(pt)), normal_style), leftIndent=12)
            for pt in mom["points_discussed"]
        ]
        story.append(ListFlowable(items, bulletType='bullet', start='bulletchar',
                                  bulletColor=colors.HexColor("#3b82f6"), leftIndent=6))
        story.append(Spacer(1, 8))

    # ── Action Points ─────────────────────────────────────────────────
    if mom.get("action_items"):
        story.append(Paragraph("ACTION POINTS", header_style))
        story.append(_hr())

        # Group by owner for speaker-based display
        from collections import defaultdict
        by_owner: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ai in mom["action_items"]:
            owner = ai.get("owner", "Unassigned") if isinstance(ai, dict) else "Unassigned"
            by_owner[owner].append(ai)

        general = by_owner.pop("Unassigned", [])

        # Speaker-based action items first
        for owner, owner_items in sorted(by_owner.items()):
            story.append(Paragraph(f"<b>{escape(owner)}</b>", normal_style))
            items = []
            for ai in owner_items:
                t = ai.get('task', '') if isinstance(ai, dict) else str(ai)
                d = ai.get('deadline', 'ASAP') if isinstance(ai, dict) else 'ASAP'
                items.append(ListItem(
                    Paragraph(f"{escape(t)} <i>(Due: {escape(d)})</i>", normal_style),
                    leftIndent=12
                ))
            story.append(ListFlowable(items, bulletType='bullet',
                                      bulletColor=colors.HexColor("#7c3aed"), leftIndent=16))

        # General action items
        if general:
            story.append(Paragraph("<b>General</b>", normal_style))
            items = []
            for ai in general:
                t = ai.get('task', '') if isinstance(ai, dict) else str(ai)
                d = ai.get('deadline', 'ASAP') if isinstance(ai, dict) else 'ASAP'
                items.append(ListItem(
                    Paragraph(f"{escape(t)} <i>(Due: {escape(d)})</i>", normal_style),
                    leftIndent=12
                ))
            story.append(ListFlowable(items, bulletType='bullet',
                                      bulletColor=colors.HexColor("#ef4444"), leftIndent=16))
        story.append(Spacer(1, 8))

    # ── Conclusion ────────────────────────────────────────────────────
    if mom.get("conclusion"):
        story.append(Paragraph("CONCLUSION", header_style))
        story.append(_hr())
        story.append(Paragraph(escape(mom["conclusion"]), normal_style))

    try:
        doc.build(story)
    except Exception as e:
        logger.error(f"Failed to build MoM PDF: {e}")
        raise HTTPException(status_code=500, detail="PDF generation failed.")

    buffer.seek(0)
    safe_title = str(mom.get("title") or "Meeting").replace(" ", "_").replace("/", "-")
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="MoM_{safe_title}.pdf"'},
    )
