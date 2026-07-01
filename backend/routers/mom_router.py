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
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mom", tags=["mom"])


class ActionItem(BaseModel):
    task: str
    owner: str
    deadline: str


class MoMData(BaseModel):
    title: str
    date: str
    duration: float
    participants: List[str]
    agenda_items: List[str]
    discussion_summary: str
    decisions: List[str]
    action_items: List[ActionItem]
    risks_concerns: List[str]
    next_steps: List[str]
    next_meeting_date: Optional[str]


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
        "participants": from_json(mom["participants"], []) or [],
        "agenda_items": from_json(mom["agenda_items"], []) or [],
        "discussion_summary": mom.get("discussion_summary") or "",
        "decisions": from_json(mom["decisions"], []) or [],
        "action_items": _normalize_action_items(raw_action_items),
        "risks_concerns": from_json(mom["risks_concerns"], []) or [],
        "next_steps": from_json(mom["next_steps"], []) or [],
        "next_meeting_date": mom.get("next_meeting_date") or None,
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

    async with get_db() as db:
        r = await db.execute(
            text("""
                SELECT transcript, filename, created_at, duration, speakers_detected
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

    meta = {
        "filename": rec.get("filename", "Meeting Notes"),
        "created_at": rec.get("created_at", ""),
        "duration": rec.get("duration", 0),
        "speakers_detected": from_json(rec["speakers_detected"], []),
    }
    import asyncio as _asyncio
    _loop = _asyncio.get_event_loop()
    mom_data = await _loop.run_in_executor(None, lambda: generate_mom(transcript, meta))

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
                        participants = :participants, agenda_items = :agenda_items,
                        discussion_summary = :discussion_summary, decisions = :decisions,
                        action_items = :action_items, risks_concerns = :risks_concerns,
                        next_steps = :next_steps, next_meeting_date = :next_meeting_date,
                        versions = :versions, is_draft = 0, updated_at = :updated_at
                    WHERE recording_id = :rid AND user_id = :uid
                """),
                {
                    "title": mom_data.get("title", ""),
                    "date": mom_data.get("date", ""),
                    "duration": mom_data.get("duration", 0),
                    "participants": to_json(mom_data.get("participants", [])),
                    "agenda_items": to_json(mom_data.get("agenda_items", [])),
                    "discussion_summary": mom_data.get("discussion_summary", ""),
                    "decisions": to_json(mom_data.get("decisions", [])),
                    "action_items": to_json(mom_data.get("action_items", [])),
                    "risks_concerns": to_json(mom_data.get("risks_concerns", [])),
                    "next_steps": to_json(mom_data.get("next_steps", [])),
                    "next_meeting_date": mom_data.get("next_meeting_date", ""),
                    "versions": to_json(initial_version),
                    "updated_at": dt_to_str(now),
                    "rid": recording_id,
                    "uid": user_id,
                },
            )
        else:
            await db.execute(
                text("""
                    INSERT INTO minutes_of_meeting (id, recording_id, user_id, title, date, duration,
                        participants, agenda_items, discussion_summary, decisions, action_items,
                        risks_concerns, next_steps, next_meeting_date, versions, is_draft,
                        created_at, updated_at)
                    VALUES (:id, :rid, :uid, :title, :date, :duration,
                        :participants, :agenda_items, :discussion_summary, :decisions, :action_items,
                        :risks_concerns, :next_steps, :next_meeting_date, :versions, 0,
                        :created_at, :updated_at)
                """),
                {
                    "id": mom_id,
                    "rid": recording_id,
                    "uid": user_id,
                    "title": mom_data.get("title", ""),
                    "date": mom_data.get("date", ""),
                    "duration": mom_data.get("duration", 0),
                    "participants": to_json(mom_data.get("participants", [])),
                    "agenda_items": to_json(mom_data.get("agenda_items", [])),
                    "discussion_summary": mom_data.get("discussion_summary", ""),
                    "decisions": to_json(mom_data.get("decisions", [])),
                    "action_items": to_json(mom_data.get("action_items", [])),
                    "risks_concerns": to_json(mom_data.get("risks_concerns", [])),
                    "next_steps": to_json(mom_data.get("next_steps", [])),
                    "next_meeting_date": mom_data.get("next_meeting_date", ""),
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
                    participants = :participants, agenda_items = :agenda_items,
                    discussion_summary = :discussion_summary, decisions = :decisions,
                    action_items = :action_items, risks_concerns = :risks_concerns,
                    next_steps = :next_steps, next_meeting_date = :next_meeting_date,
                    versions = :versions, is_draft = 1, updated_at = :updated_at
                WHERE recording_id = :rid AND user_id = :uid
            """),
            {
                "title": update_data.get("title", ""),
                "date": update_data.get("date", ""),
                "duration": update_data.get("duration", 0),
                "participants": to_json(update_data.get("participants", [])),
                "agenda_items": to_json(update_data.get("agenda_items", [])),
                "discussion_summary": update_data.get("discussion_summary", ""),
                "decisions": to_json(update_data.get("decisions", [])),
                "action_items": to_json(update_data.get("action_items", [])),
                "risks_concerns": to_json(update_data.get("risks_concerns", [])),
                "next_steps": to_json(update_data.get("next_steps", [])),
                "next_meeting_date": update_data.get("next_meeting_date", ""),
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
async def export_mom_pdf(recording_id: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        mom_row = r.mappings().fetchone()

    if not mom_row:
        raise HTTPException(status_code=404, detail="MoM not found")

    mom = _mom_row_to_dict(mom_row)

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
        spaceAfter=12, fontName="Helvetica-Bold"
    )
    header_style = ParagraphStyle(
        'HeaderStyle', parent=styles['Heading2'],
        fontSize=14, textColor=colors.HexColor("#1e293b"),
        spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold",
        borderPadding=(0, 0, 4, 0), borderColor=colors.HexColor("#e2e8f0"),
        borderWidth=1, borderRadius=0
    )
    normal_style = ParagraphStyle(
        'NormalStyle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor("#334155"),
        spaceAfter=6, fontName="Helvetica", leading=14
    )
    meta_style = ParagraphStyle(
        'MetaStyle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor("#64748b"),
        spaceAfter=4, fontName="Helvetica-Oblique"
    )

    story = []
    story.append(Paragraph(mom.get("title", "Minutes of Meeting"), title_style))
    story.append(Paragraph(f"Date: {mom.get('date', 'Unknown')}", meta_style))
    story.append(Paragraph(f"Duration: {mom.get('duration', 0)} seconds", meta_style))
    story.append(Spacer(1, 20))

    def _add_section(title, content_items, is_list=False):
        if not content_items:
            return
        story.append(Paragraph(escape(title.upper()), header_style))
        if is_list:
            items = [ListItem(Paragraph(escape(str(item)), normal_style)) for item in content_items]
            story.append(ListFlowable(items, bulletType='bullet', start='bulletchar',
                                      bulletColor=colors.HexColor("#3b82f6")))
        else:
            story.append(Paragraph(escape(str(content_items)), normal_style))
        story.append(Spacer(1, 10))

    _add_section("Participants", mom.get("participants"), True)
    _add_section("Agenda Items", mom.get("agenda_items"), True)
    _add_section("Discussion Summary", mom.get("discussion_summary"), False)
    _add_section("Decisions Taken", mom.get("decisions"), True)

    if mom.get("action_items"):
        story.append(Paragraph("ACTION ITEMS", header_style))
        items = []
        for ai in mom["action_items"]:
            t = ai.get('task', '') if isinstance(ai, dict) else str(ai)
            o = ai.get('owner', 'Unassigned') if isinstance(ai, dict) else ''
            d = ai.get('deadline', 'No deadline') if isinstance(ai, dict) else ''
            text_str = f"<b>{escape(t)}</b> (Owner: {escape(o)}, Due: {escape(d)})"
            items.append(ListItem(Paragraph(text_str, normal_style)))
        story.append(ListFlowable(items, bulletType='bullet', bulletColor=colors.HexColor("#ef4444")))
        story.append(Spacer(1, 10))

    _add_section("Risks / Concerns", mom.get("risks_concerns"), True)
    _add_section("Next Steps", mom.get("next_steps"), True)

    if mom.get("next_meeting_date"):
        _add_section("Next Meeting", mom.get("next_meeting_date"), False)

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
