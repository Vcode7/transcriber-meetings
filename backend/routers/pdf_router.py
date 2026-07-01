"""
PDF Report Router — Generate professional meeting reports.
POST /pdf/{recording_id} → streams a PDF file to the client.
"""
import io
import logging
from datetime import datetime
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File , Form
from fastapi.responses import StreamingResponse
import json
from xml.sax.saxutils import escape

from database import get_db, from_json
from sqlalchemy import text
from routers.auth import get_current_user
from services.llm import generate_executive_summary, generate_key_decisions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pdf", tags=["pdf"])

# ── Brand colors (match frontend CSS vars) ────────────────────
NAVY       = (0.082, 0.149, 0.263)   # hsl(215 60% 12%) dark paper
NAVY_CARD  = (0.090, 0.165, 0.294)   # hsl(215 55% 16%) card
ACCENT     = (0.941, 0.290, 0.102)   # hsl(14 90% 56%)  orange-red
SUCCESS    = (0.196, 0.647, 0.275)   # hsl(130 60% 42%) green
PENCIL     = (0.545, 0.600, 0.667)   # hsl(200 30% 65%) muted text
WHITE      = (1, 1, 1)
OFF_WHITE  = (0.937, 0.929, 0.910)   # hsl(42 38% 96%) cream

SPEAKER_COLORS_RGB = [
    (0.941, 0.400, 0.220),  # orange-red
    (0.216, 0.659, 0.906),  # blue
    (0.216, 0.647, 0.271),  # green
    (0.573, 0.341, 0.824),  # purple
    (0.976, 0.761, 0.176),  # yellow
    (0.906, 0.278, 0.533),  # pink
]


def _fmt_duration(seconds: float) -> str:
    if not seconds or seconds <= 0:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _speaker_analytics(transcript: List[Dict]) -> List[Dict]:
    """Compute speaking time & percentage per speaker."""
    totals: Dict[str, float] = {}
    for seg in transcript:
        label = seg.get("speaker_label", "Unknown")
        duration = max(0.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
        totals[label] = totals.get(label, 0.0) + duration

    grand_total = sum(totals.values()) or 1.0
    result = []
    for label, t in sorted(totals.items(), key=lambda x: -x[1]):
        result.append({
            "speaker": label,
            "seconds": t,
            "formatted": _fmt_duration(t),
            "pct": round(t / grand_total * 100, 1),
        })
    return result


def _parse_action_items(raw_items: List[str]) -> List[Dict[str, str]]:
    """
    Robust action item parser supporting:
      1. "Task — Owner: Name — Deadline: Date" (em-dash format)
      2. "Task (Owner: Name, Deadline: Date)" (parentheses format)
      3. "Task - Owner by Deadline" (legacy fallback)
    """
    import re
    parsed = []
    for item in raw_items:
        # Clean up leading numbers, bullets, brackets
        item_clean = re.sub(r'^\s*[-*•\d\.\s\[\]\(\)]+\s*', '', item).strip()
        
        task = item_clean
        owner = "—"
        deadline = "—"
        
        # Normalize dashes to easily identify fields
        normalized = item_clean.replace("—", " - ").replace("–", " - ")
        
        # Regex search for Owner and Deadline labels (case-insensitive)
        owner_match = re.search(r'(?i)(?:owner|assignee|assigned\s+to)[:\s]+([^-\|,;\)\]]+)', normalized)
        deadline_match = re.search(r'(?i)(?:deadline|due\s+date|due)[:\s]+([^-\|,;\)\]]+)', normalized)
        
        if owner_match:
            owner = owner_match.group(1).strip()
            owner = re.sub(r'[\)\],]+$', '', owner).strip()
            
        if deadline_match:
            deadline = deadline_match.group(1).strip()
            deadline = re.sub(r'[\)\],]+$', '', deadline).strip()
            
        # Extract task description by splitting at dash markers or parentheses
        if owner_match or deadline_match:
            parts = re.split(r'\s+[-–—]\s+|\s+\(', item_clean)
            if parts and parts[0].strip():
                task = parts[0].strip()
        else:
            # Fallback to simple split heuristics
            if " by " in item_clean.lower():
                parts = item_clean.lower().split(" by ", 1)
                deadline = parts[1].strip().capitalize()
                task = parts[0].strip()
            if " - " in task:
                parts = task.split(" - ", 1)
                task = parts[0].strip()
                owner = parts[1].strip()
                
        # Strip wrapping brackets/braces from task description
        task = task.strip().strip("[]()")
        
        if owner and owner != "—":
            owner = owner.title()
            
        parsed.append({
            "task": task,
            "owner": owner or "—",
            "deadline": deadline or "—"
        })
    return parsed


# ── ReportLab PDF builder ──────────────────────────────────────

def _build_pdf(
    rec: Dict,
    exec_summary: dict,
    decisions: List[str],
    ref_image_data: List[bytes] = None,
    ref_image_names: List[str] = None,
    ref_doc_pages: List[bytes] = None,
    ref_doc_texts: List[tuple] = None,
    include_transcription: bool = True,
    speaker_summary_data: Dict = None,
) -> bytes:
    """Build the full PDF and return as bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, KeepTogether
    )
    from reportlab.lib import colors
    from reportlab.lib.colors import Color, HexColor

    # ── Color helpers
    def rgb(r, g, b): return Color(r, g, b)

    C_NAVY       = rgb(*NAVY)
    C_NAVY_CARD  = rgb(*NAVY_CARD)
    C_ACCENT     = rgb(*ACCENT)
    C_SUCCESS    = rgb(*SUCCESS)
    C_PENCIL     = rgb(*PENCIL)
    C_WHITE      = rgb(*WHITE)
    C_OFF_WHITE  = rgb(*OFF_WHITE)
    C_LIGHT_BG   = Color(0.118, 0.200, 0.345)   # slightly lighter navy for row alt

    # ── Page setup
    buf = io.BytesIO()
    W, H = A4

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20*mm,
        rightMargin=20*mm,
        topMargin=22*mm,
        bottomMargin=22*mm,
        title=f"Meeting Report — {rec.get('filename', 'Meeting')}",
        author="VoiceSum",
    )

    story = []

    # ─────────────────────────────────────────────────────────────
    # PAGE 1: COVER / HEADER
    # ─────────────────────────────────────────────────────────────

    # Logo + brand name row
    story.append(Paragraph(
        '<font color="#f04a1a" size="22"><b>🎙 VoiceSum</b></font>',
        ParagraphStyle("logo", fontName="Helvetica-Bold", fontSize=22, leading=28,
                       textColor=C_ACCENT)
    ))
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=2, color=C_ACCENT, spaceAfter=6*mm))

    # Report title
    story.append(Paragraph(
        "MEETING REPORT",
        ParagraphStyle("report_title", fontName="Helvetica-Bold", fontSize=28, leading=34,
                       textColor=C_WHITE, spaceAfter=2*mm)
    ))

    # Meeting name
    meeting_name = rec.get("filename", "Untitled Meeting")
    story.append(Paragraph(
        meeting_name,
        ParagraphStyle("meeting_name", fontName="Helvetica-Bold", fontSize=16, leading=20,
                       textColor=C_ACCENT, spaceAfter=8*mm)
    ))

    # Meta info grid
    created_at = rec.get("created_at", "")
    if isinstance(created_at, str):
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            date_str = dt.strftime("%B %d, %Y  %H:%M UTC")
        except Exception:
            date_str = created_at[:19].replace("T", "  ")
    else:
        date_str = str(created_at)

    duration_str = _fmt_duration(rec.get("duration", 0))
    speakers = rec.get("speakers_detected", [])
    speakers_str = ", ".join(speakers) if speakers else "Unknown"
    num_speakers = len(speakers) if speakers else 0
    segments = rec.get("transcript", [])

    meta_data = [
        ["Date", date_str],
        ["Duration", duration_str],
        ["Participants", f"{num_speakers} speaker{'s' if num_speakers != 1 else ''} — {speakers_str}"],
        ["Transcript Segments", str(len(segments))],
        ["Status", rec.get("status", "done").upper()],
    ]

    meta_style = TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), C_ACCENT),
        ("BACKGROUND", (1, 0), (1, -1), C_NAVY_CARD),
        ("TEXTCOLOR", (0, 0), (0, -1), C_WHITE),
        ("TEXTCOLOR", (1, 0), (1, -1), C_OFF_WHITE),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [C_NAVY_CARD, C_LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.3, C_PENCIL),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])

    meta_table = Table(
        meta_data,
        colWidths=[45*mm, W - 40*mm - 45*mm],
        style=meta_style,
        hAlign="LEFT",
    )
    story.append(meta_table)
    story.append(Spacer(1, 10*mm))

    # ── Section header helper
    def section_header(number: str, title: str):
        return KeepTogether([
            Spacer(1, 6*mm),
            Table(
                [[f"  {number}", f"  {title}"]],
                colWidths=[14*mm, W - 40*mm - 14*mm],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (0, 0), C_ACCENT),
                    ("BACKGROUND", (1, 0), (1, 0), C_NAVY_CARD),
                    ("TEXTCOLOR", (0, 0), (0, 0), C_WHITE),
                    ("TEXTCOLOR", (1, 0), (1, 0), C_WHITE),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 11),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ]),
            ),
            Spacer(1, 4*mm),
        ])

    # ── Sub-label helper
    def sub_label(text: str):
        return Paragraph(
            text.upper(),
            ParagraphStyle("sub_label", fontName="Helvetica-Bold", fontSize=7.5,
                           textColor=C_ACCENT, leading=12, spaceAfter=2*mm,
                           letterSpacing=1.0)
        )

    # ── Body text style
    body_style = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=9, leading=14,
        textColor=C_OFF_WHITE, spaceAfter=3*mm
    )

    bullet_style = ParagraphStyle(
        "bullet", fontName="Helvetica", fontSize=9, leading=14,
        textColor=C_OFF_WHITE, leftIndent=12, spaceAfter=2*mm,
        bulletIndent=0
    )

    # ─────────────────────────────────────────────────────────────
    # SECTION 1: EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────────────────────
    story.append(section_header("01", "Executive Summary"))

    summary_items = [
        ("Meeting Purpose", exec_summary.get("purpose", "—"), None),
        ("Main Discussion Points", None, exec_summary.get("discussion_points", [])),
        ("Outcomes", None, exec_summary.get("outcomes", [])),
        ("Next Steps", None, exec_summary.get("next_steps", [])),
    ]

    for label, paragraph, bullets in summary_items:
        block = [sub_label(label)]
        if paragraph:
            block.append(Paragraph(escape(paragraph), body_style))
        if bullets:
            for b in bullets:
                block.append(Paragraph(f"• {escape(b)}", bullet_style))
        block.append(Spacer(1, 3*mm))
        story.append(KeepTogether(block))

    # ─────────────────────────────────────────────────────────────
    # SECTION 2: SHORT SUMMARY
    # ─────────────────────────────────────────────────────────────
    short_summary_text = rec.get("short_summary", "") or rec.get("summary", "")
    story.append(section_header("02", "Short Summary"))
    if short_summary_text:
        story.append(Paragraph(escape(short_summary_text), body_style))
    else:
        story.append(Paragraph("No short summary available.", body_style))
    story.append(Spacer(1, 2*mm))

    # ─────────────────────────────────────────────────────────────
    # SECTION 3: DETAILED SUMMARY
    # ─────────────────────────────────────────────────────────────
    detailed_summary_text = rec.get("detailed_summary", "") or ""
    story.append(section_header("03", "Detailed Summary"))
    if detailed_summary_text:
        # Render markdown-style bullet lines
        for line in detailed_summary_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("## ") or line.startswith("### "):
                heading = line.lstrip("#").strip()
                story.append(Paragraph(escape(heading), ParagraphStyle(
                    "ds_h", fontName="Helvetica-Bold", fontSize=10, leading=14,
                    textColor=C_ACCENT, spaceAfter=2*mm, spaceBefore=3*mm,
                )))
            elif line.startswith(("- ", "* ", "• ")):
                text = line.lstrip("-* •").strip()
                story.append(Paragraph(f"• {escape(text)}", bullet_style))
            else:
                story.append(Paragraph(escape(line), body_style))
    else:
        story.append(Paragraph("No detailed summary available.", body_style))
    story.append(Spacer(1, 2*mm))

    # ─────────────────────────────────────────────────────────────
    # SECTION 4: KEY DISCUSSION POINTS
    # ─────────────────────────────────────────────────────────────
    key_points_list = rec.get("key_points", [])
    story.append(section_header("04", "Key Discussion Points"))
    if key_points_list:
        for kp in key_points_list:
            story.append(Paragraph(f"• {escape(str(kp))}", bullet_style))
    else:
        story.append(Paragraph("No key discussion points identified.", body_style))
    story.append(Spacer(1, 2*mm))

    # ─────────────────────────────────────────────────────────────
    # SECTION 5: KEY DECISIONS
    # ─────────────────────────────────────────────────────────────
    story.append(section_header("05", "Key Decisions"))

    if decisions:
        for dec in decisions:
            row_block = Table(
                [["✓", Paragraph(escape(dec), body_style)]],
                colWidths=[8*mm, W - 40*mm - 8*mm],
                style=TableStyle([
                    ("TEXTCOLOR", (0, 0), (0, 0), C_SUCCESS),
                    ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (0, 0), 11),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ])
            )
            story.append(row_block)
            story.append(Spacer(1, 2*mm))
    else:
        story.append(Paragraph("No key decisions were explicitly identified in this meeting.", body_style))

    story.append(Spacer(1, 2*mm))

    # ─────────────────────────────────────────────────────────────
    # SECTION 6: ACTION ITEMS
    # ─────────────────────────────────────────────────────────────
    story.append(section_header("06", "Action Items"))

    action_items_raw = rec.get("action_items", [])
    if action_items_raw:
        parsed_actions = _parse_action_items(action_items_raw)
        action_header = ["#", "Task", "Owner", "Deadline"]
        action_data = [action_header]
        for i, item in enumerate(parsed_actions, 1):
            action_data.append([
                str(i),
                Paragraph(escape(item["task"]), ParagraphStyle("at", fontName="Helvetica", fontSize=8.5, leading=12, textColor=C_OFF_WHITE)),
                item["owner"],
                item["deadline"],
            ])

        action_table = Table(
            action_data,
            colWidths=[8*mm, W - 40*mm - 8*mm - 35*mm - 30*mm, 35*mm, 30*mm],
            repeatRows=1,
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_NAVY_CARD, C_LIGHT_BG]),
                ("TEXTCOLOR", (0, 1), (-1, -1), C_OFF_WHITE),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.3, C_PENCIL),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ]),
        )
        story.append(action_table)
    else:
        story.append(Paragraph("No specific action items were identified in this meeting.", body_style))

    story.append(Spacer(1, 2*mm))

    # ─────────────────────────────────────────────────────────────
    # SECTION 7: PER-SPEAKER SUMMARIES (optional — only when data exists)
    # ─────────────────────────────────────────────────────────────
    if speaker_summary_data:
        story.append(section_header("07", "Per-Speaker Summaries"))
        speaker_colors = list(speaker_color_map.values()) or [C_ACCENT]
        for sp_idx, (speaker_name, sp_data) in enumerate(speaker_summary_data.items()):
            sp_color = speaker_colors[sp_idx % len(speaker_colors)]

            # Speaker name sub-header
            story.append(KeepTogether([
                Spacer(1, 3*mm),
                Paragraph(escape(speaker_name), ParagraphStyle(
                    f"sp_name_{sp_idx}", fontName="Helvetica-Bold", fontSize=10.5,
                    leading=13, textColor=sp_color, spaceAfter=2*mm,
                )),
                HRFlowable(width="100%", thickness=0.5, color=sp_color, spaceAfter=3*mm),
            ]))

            sp_summary = sp_data.get("summary", "")
            sp_kp = sp_data.get("key_points", [])
            sp_ai = sp_data.get("action_items", [])

            block = []
            if sp_summary:
                block.append(sub_label("Summary"))
                block.append(Paragraph(escape(sp_summary), body_style))
                block.append(Spacer(1, 2*mm))
            if sp_kp:
                block.append(sub_label("Key Points"))
                for kp in sp_kp:
                    block.append(Paragraph(f"• {escape(str(kp))}", bullet_style))
                block.append(Spacer(1, 2*mm))
            if sp_ai:
                block.append(sub_label("Action Items"))
                for ai_item in sp_ai:
                    block.append(Paragraph(f"• {escape(str(ai_item))}", bullet_style))
                block.append(Spacer(1, 2*mm))

            if block:
                story.append(KeepTogether(block))

        story.append(Spacer(1, 2*mm))

    # ─────────────────────────────────────────────────────────────
    # SECTION 8 (renumbered): PARTICIPANT ANALYTICS
    # ─────────────────────────────────────────────────────────────
    analytics_section_num = "08" if not speaker_summary_data else "08"
    story.append(section_header(analytics_section_num, "Participant Analytics"))

    analytics = _speaker_analytics(segments)

    # Build color map consistent with frontend
    speaker_color_map: Dict[str, Any] = {}
    for idx, row in enumerate(analytics):
        color_rgb = SPEAKER_COLORS_RGB[idx % len(SPEAKER_COLORS_RGB)]
        speaker_color_map[row["speaker"]] = Color(*color_rgb)

    if analytics:
        # Stats table
        analytics_header = ["Speaker", "Speaking Time", "Contribution", "Visual"]
        analytics_data = [analytics_header]
        for row in analytics:
            pct = row["pct"]
            bar_len = max(1, int(pct / 100 * 40))
            bar = "█" * bar_len
            analytics_data.append([
                row["speaker"],
                row["formatted"],
                f"{pct}%",
                bar,
            ])

        col_widths = [45*mm, 35*mm, 22*mm, W - 40*mm - 45*mm - 35*mm - 22*mm]
        analytics_table_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_NAVY_CARD, C_LIGHT_BG]),
            ("TEXTCOLOR", (0, 1), (-1, -1), C_OFF_WHITE),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.3, C_PENCIL),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
        # Color the bar column and speaker column per speaker
        for i, row in enumerate(analytics, 1):
            c = speaker_color_map.get(row["speaker"], C_ACCENT)
            analytics_table_style.add("TEXTCOLOR", (3, i), (3, i), c)
            analytics_table_style.add("TEXTCOLOR", (0, i), (0, i), c)
            analytics_table_style.add("FONTNAME", (0, i), (0, i), "Helvetica-Bold")

        analytics_table = Table(analytics_data, colWidths=col_widths, style=analytics_table_style)
        story.append(analytics_table)
    else:
        story.append(Paragraph("No participant data available.", body_style))

    story.append(Spacer(1, 2*mm))

    # ─────────────────────────────────────────────────────────────
    # SECTION 9: SPEAKER TIMELINE (conditional on include_transcription)
    # ─────────────────────────────────────────────────────────────
    if include_transcription:
        story.append(section_header("09", "Speaker Timeline"))

        if segments:
            tl_header = ["Start", "End", "Speaker", "Content"]
            tl_data = [tl_header]
            for seg in segments:
                label = seg.get("speaker_label", "Unknown")
                text = seg.get("text", "").strip()
                # Truncate long segments in timeline view
                if len(text) > 200:
                    text = text[:197] + "…"
                tl_data.append([
                    _fmt_time(seg.get("start", 0)),
                    _fmt_time(seg.get("end", 0)),
                    label,
                    Paragraph(escape(text), ParagraphStyle("tl", fontName="Helvetica", fontSize=8, leading=11, textColor=C_OFF_WHITE)),
                ])

            tl_col_widths = [14*mm, 14*mm, 35*mm, W - 40*mm - 14*mm - 14*mm - 35*mm]
            tl_style = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_NAVY_CARD, C_LIGHT_BG]),
                ("TEXTCOLOR", (0, 1), (-1, -1), C_PENCIL),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.2, C_PENCIL),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (1, -1), "CENTER"),
            ])
            # Color speaker names
            for i, seg in enumerate(segments, 1):
                label = seg.get("speaker_label", "Unknown")
                c = speaker_color_map.get(label, C_ACCENT)
                tl_style.add("TEXTCOLOR", (2, i), (2, i), c)
                tl_style.add("FONTNAME", (2, i), (2, i), "Helvetica-Bold")

            tl_table = Table(tl_data, colWidths=tl_col_widths, repeatRows=1, style=tl_style)
            story.append(tl_table)
        else:
            story.append(Paragraph("No timeline data available.", body_style))

        story.append(Spacer(1, 2*mm))

    # ─────────────────────────────────────────────────────────────
    # SECTION 10: FULL TRANSCRIPT (conditional on include_transcription)
    # ─────────────────────────────────────────────────────────────
    if include_transcription:
        story.append(PageBreak())
        story.append(section_header("10", "Full Transcript"))

        ts_name_style = ParagraphStyle(
            "ts_name", fontName="Helvetica-Bold", fontSize=9, leading=12,
            spaceAfter=1*mm
        )
        ts_meta_style = ParagraphStyle(
            "ts_meta", fontName="Helvetica", fontSize=7.5, leading=10,
            textColor=C_PENCIL, spaceAfter=1.5*mm
        )
        ts_text_style = ParagraphStyle(
            "ts_text", fontName="Helvetica", fontSize=9, leading=14,
            textColor=C_OFF_WHITE, leftIndent=0, spaceAfter=5*mm
        )

        if segments:
            for i, seg in enumerate(segments):
                label = seg.get("speaker_label", "Unknown")
                text = seg.get("text", "").strip()
                start = _fmt_time(seg.get("start", 0))
                end = _fmt_time(seg.get("end", 0))
                color = speaker_color_map.get(label, C_ACCENT)

                name_para = Paragraph(escape(label), ParagraphStyle(
                    f"ts_name_{i}", fontName="Helvetica-Bold", fontSize=9.5,
                    leading=12, textColor=color, spaceAfter=1*mm
                ))
                meta_para = Paragraph(escape(f"{start} → {end}"), ts_meta_style)
                text_para = Paragraph(escape(text), ts_text_style)

                seg_block = KeepTogether([name_para, meta_para, text_para])
                story.append(seg_block)
        else:
            story.append(Paragraph("No transcript segments available.", body_style))

    # ─────────────────────────────────────────────────────────────
    # SECTION 11: REFERENCE IMAGES (optional)
    # ─────────────────────────────────────────────────────────────
    if ref_image_data:
        from reportlab.platypus import Image as RLImage
        story.append(PageBreak())
        story.append(section_header("11", "Reference Images"))
        for idx, (img_bytes, img_name) in enumerate(zip(ref_image_data, ref_image_names or [])):
            try:
                img_buf = io.BytesIO(img_bytes)
                rl_img = RLImage(img_buf, width=W - 40*mm, height=None)
                # Scale to fit page width while maintaining aspect ratio
                if rl_img.imageWidth > 0:
                    scale = (W - 40*mm) / rl_img.imageWidth
                    rl_img.drawWidth = W - 40*mm
                    rl_img.drawHeight = rl_img.imageHeight * scale
                    if rl_img.drawHeight > 200*mm:
                        rl_img.drawHeight = 200*mm
                        rl_img.drawWidth = rl_img.imageWidth * (200*mm / rl_img.imageHeight)
                story.append(Paragraph(
                    escape(img_name or f"Image {idx + 1}"),
                    ParagraphStyle("img_cap", fontName="Helvetica", fontSize=8, leading=10,
                                   textColor=C_PENCIL, spaceAfter=2*mm)
                ))
                story.append(rl_img)
                story.append(Spacer(1, 6*mm))
            except Exception as e:
                logger.warning(f"[PDF] Could not embed image {img_name}: {e}")
                story.append(Paragraph(f"[Image could not be embedded: {escape(str(img_name))}]", body_style))

    # ─────────────────────────────────────────────────────────────
    # SECTION 12: REFERENCE DOCUMENTS (optional)
    # ─────────────────────────────────────────────────────────────
    if ref_doc_texts:
        story.append(PageBreak())
        story.append(section_header("12", "Reference Documents"))
        for (doc_name, doc_text) in ref_doc_texts:
            story.append(sub_label(doc_name))
            for line in doc_text.split("\n"):
                line = line.strip()
                if line:
                    story.append(Paragraph(escape(line), body_style))
            story.append(Spacer(1, 4*mm))

    # ─────────────────────────────────────────────────────────────
    # Build PDF
    # ─────────────────────────────────────────────────────────────

    # Dark background on all pages
    def _on_page_bg(canvas, doc):
        canvas.saveState()
        # Dark navy background
        canvas.setFillColor(rgb(*NAVY))
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        # Top accent bar
        canvas.setFillColor(rgb(*ACCENT))
        canvas.rect(0, H - 8*mm, W, 8*mm, fill=1, stroke=0)
        # Footer
        canvas.setFillColor(rgb(*PENCIL))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(20*mm, 8*mm, "VoiceSum — Confidential Meeting Report")
        canvas.drawRightString(W - 20*mm, 8*mm, f"Page {doc.page}")
        # Footer rule
        canvas.setStrokeColor(rgb(*PENCIL))
        canvas.setLineWidth(0.3)
        canvas.line(20*mm, 13*mm, W - 20*mm, 13*mm)
        canvas.restoreState()

    doc.build(story, onFirstPage=_on_page_bg, onLaterPages=_on_page_bg)
    return buf.getvalue()


# ── Endpoint ───────────────────────────────────────────────────

@router.post("/{recording_id}")
async def generate_pdf_report(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
    images: List[UploadFile] = File(default=[]),
    documents: List[UploadFile] = File(default=[]),
    include_transcription: bool = Form(default=True),
):
    """
    Generate and stream a professional PDF meeting report.
    Optionally attach reference images (PNG/JPG) and documents (PDF/DOCX).
    """
    user_id = current_user["id"]

    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec_row = r.mappings().fetchone()

    if not rec_row:
        raise HTTPException(status_code=404, detail="Recording not found.")

    rec = dict(rec_row)

    if rec.get("status") not in ("done", "transcript_ready"):
        raise HTTPException(
            status_code=400,
            detail="Recording is still being processed. Please wait until transcription is complete."
        )

    transcript = from_json(rec.get("transcript"), [])

    # Generate LLM content for PDF (may take 2-5s) — run in executor to avoid blocking event loop
    try:
        import asyncio as _asyncio
        _loop = _asyncio.get_event_loop()
        logger.info(f"[PDF] Generating executive summary for {recording_id}")
        exec_summary = await _loop.run_in_executor(None, generate_executive_summary, transcript)
        decisions = await _loop.run_in_executor(None, generate_key_decisions, transcript)
    except RuntimeError as e:
        logger.error(f"[PDF] LLM error: {e}")
        exec_summary = {
            "purpose": "LLM unavailable — see full transcript for details.",
            "discussion_points": [],
            "outcomes": [],
            "next_steps": [],
        }
        decisions = []

    # Read uploaded files
    ref_image_data: List[bytes] = []
    ref_image_names: List[str] = []
    ref_doc_pages: List[bytes] = []    # flattened PDF pages as individual PDFs
    ref_doc_texts: List[tuple] = []    # (filename, text_content) for DOCX

    for img_file in images:
        if img_file.filename:
            data = await img_file.read()
            if data:
                ref_image_data.append(data)
                ref_image_names.append(img_file.filename)

    for doc_file in documents:
        if not doc_file.filename:
            continue
        data = await doc_file.read()
        if not data:
            continue
        fname = doc_file.filename.lower()
        if fname.endswith(".pdf"):
            # Extract individual pages as separate PDFs using PyMuPDF
            try:
                import fitz  # PyMuPDF
                src_doc = fitz.open(stream=data, filetype="pdf")
                for page_num in range(len(src_doc)):
                    page_doc = fitz.open()
                    page_doc.insert_pdf(src_doc, from_page=page_num, to_page=page_num)
                    ref_doc_pages.append(page_doc.tobytes())
                    page_doc.close()
                src_doc.close()
            except Exception as e:
                logger.warning(f"[PDF] Could not process PDF attachment {doc_file.filename}: {e}")
        elif fname.endswith(".docx"):
            # Extract text from DOCX using python-docx
            try:
                import io as _io
                from docx import Document
                doc_obj = Document(_io.BytesIO(data))
                text_parts = [para.text for para in doc_obj.paragraphs if para.text.strip()]
                ref_doc_texts.append((doc_file.filename, "\n".join(text_parts)))
            except Exception as e:
                logger.warning(f"[PDF] Could not process DOCX attachment {doc_file.filename}: {e}")

    rec_plain = {
        "filename": rec.get("filename", "meeting.pdf"),
        "duration": rec.get("duration", 0),
        "status": rec.get("status", "done"),
        "transcript": transcript,
        "summary": rec.get("summary", ""),
        "short_summary": rec.get("short_summary", "") or rec.get("summary", ""),
        "detailed_summary": rec.get("detailed_summary", "") or "",
        "key_points": from_json(rec.get("key_points"), []),
        "action_items": from_json(rec.get("action_items"), []),
        "speakers_detected": from_json(rec.get("speakers_detected"), []),
        "created_at": rec.get("created_at", ""),
    }

    # Load per-speaker summary data if available
    speaker_summary_data = from_json(rec.get("speaker_summary"), None)

    # Build PDF
    try:
        logger.info(f"[PDF] Building PDF for {recording_id} (include_transcription={include_transcription})")
        pdf_bytes = _build_pdf(
            rec_plain, exec_summary, decisions,
            ref_image_data=ref_image_data,
            ref_image_names=ref_image_names,
            ref_doc_pages=ref_doc_pages,
            ref_doc_texts=ref_doc_texts,
            include_transcription=include_transcription,
            speaker_summary_data=speaker_summary_data,
        )
        
        # If there are attached PDF pages, append them directly using PyMuPDF (fitz)
        if ref_doc_pages:
            try:
                import fitz  # PyMuPDF
                main_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page_bytes in ref_doc_pages:
                    page_doc = fitz.open(stream=page_bytes, filetype="pdf")
                    main_doc.insert_pdf(page_doc)
                    page_doc.close()
                pdf_bytes = main_doc.tobytes()
                main_doc.close()
                logger.info(f"[PDF] Merged {len(ref_doc_pages)} attached PDF pages into the report successfully.")
            except Exception as e:
                logger.warning(f"[PDF] Failed to merge PDF attachments: {e}")
    except Exception as e:
        logger.exception(f"[PDF] Build error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    # Derive safe filename
    raw_name = str(rec_plain.get("filename") or "meeting")
    for ext in [".wav", ".mp3", ".webm", ".mp4", ".m4a", ".ogg", ".flac"]:
        raw_name = raw_name.replace(ext, "")
    safe_name = raw_name.replace(" ", "_").replace("/", "_")[:60] or "meeting"
    pdf_filename = f"VoiceSum_Report_{safe_name}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{pdf_filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )
