"""
manual_mom_service.py — Manual MoM (Minutes of Meeting) Processing & Ingestion Service.

Enables users to upload manual MoM documents (.pdf, .docx, .doc, .txt, .md, .pptx, .xlsx, .csv),
extracts content, infers structured meeting metadata (meeting name, date, speakers, agendas,
action items, decisions, summary), chunks the text, and ingests vectors directly into
ChromaDB collection `final_rom_{user_id}` for seamless, high-priority AI Chat retrieval.
"""
from __future__ import annotations

import os
import re
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from database import get_db_context, to_json, from_json
from config import settings
from services.doc_extractor import extract_text_from_file

logger = logging.getLogger(__name__)


# ── Metadata & Section Parser ────────────────────────────────────────────────

def _clean_str(val: Optional[str]) -> str:
    return (val or "").strip()


def extract_metadata_and_sections(raw_text: str, filename: str) -> Dict[str, Any]:
    """
    Extract structured meeting metadata and sections from raw extracted MoM text.
    Handles various standard MoM formats, markdown, plain text, and informal notes.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    full_text = "\n".join(lines)

    # 1. Meeting Title / Name Extraction
    title = ""
    # Look for explicit Title/Meeting Name tags
    title_match = re.search(
        r"^(?:meeting\s*(?:name|title)|title|subject|meeting)\s*[:\-–]\s*(.+)$",
        raw_text,
        re.IGNORECASE | re.MULTILINE,
    )
    if title_match:
        cand = title_match.group(1).strip()
        # Clean common trailing artifacts
        cand = re.sub(r"^(?:minutes of meeting|mom|meeting notes)\s*[:\-–]?\s*", "", cand, flags=re.IGNORECASE).strip()
        if len(cand) >= 3:
            title = cand

    # If no explicit tag, check top 5 lines for markdown header (# Title) or bold title (**Title**)
    if not title:
        for line in lines[:5]:
            h_match = re.match(r"^#{1,3}\s+(.+)$", line)
            if h_match:
                cand = h_match.group(1).strip()
                if not re.match(r"^(?:minutes of meeting|mom|notes|meeting)$", cand, re.IGNORECASE):
                    title = cand
                    break
            bold_match = re.match(r"^\*\*(.+?)\*\*$", line)
            if bold_match:
                cand = bold_match.group(1).strip()
                if not re.match(r"^(?:minutes of meeting|mom|notes|meeting)$", cand, re.IGNORECASE):
                    title = cand
                    break

    # Fallback to filename
    if not title:
        base = os.path.splitext(os.path.basename(filename))[0]
        # Replace underscores and hyphens with spaces
        title = re.sub(r"[-_]+", " ", base).strip()
        # Capitalize nicely
        title = " ".join(word.capitalize() for word in title.split())
        if not title:
            title = "Manual MoM Meeting"

    # 2. Meeting Date Extraction
    meeting_date = ""
    # Explicit date label
    date_label_match = re.search(
        r"^(?:meeting\s*date|date|held\s*on|meeting\s*time\s*&?\s*date)\s*[:\-–]\s*([^\n\r]+)$",
        raw_text,
        re.IGNORECASE | re.MULTILINE,
    )
    if date_label_match:
        date_str = date_label_match.group(1).strip()
        # Try finding standard date patterns inside the string
        d_search = re.search(r"\b(20\d\d[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01]))\b", date_str)
        if d_search:
            meeting_date = d_search.group(1).replace("/", "-").replace(".", "-")
        else:
            # Check for dd-mm-yyyy or dd/mm/yyyy
            d_search_rev = re.search(r"\b((?:0[1-9]|[12]\d|3[01])[-/.](?:0[1-9]|1[0-2])[-/.](20\d\d))\b", date_str)
            if d_search_rev:
                day, month, year = re.split(r"[-/.]", d_search_rev.group(1))
                meeting_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            else:
                meeting_date = date_str[:30]

    if not meeting_date:
        # Search anywhere in first 2000 chars for standard ISO or written date
        head = raw_text[:2000]
        iso_m = re.search(r"\b(20\d\d[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01]))\b", head)
        if iso_m:
            meeting_date = iso_m.group(1).replace("/", "-").replace(".", "-")
        else:
            # e.g., "October 8, 2026" or "8 October 2026"
            written_m = re.search(
                r"\b(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+20\d\d)\b",
                head,
                re.IGNORECASE,
            )
            if written_m:
                meeting_date = written_m.group(0)

    if not meeting_date:
        meeting_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 3. Speakers / Attendees Extraction
    speakers: List[str] = []
    attendees_sec = re.search(
        r"(?:attendees|participants|present|members|attendees\s*list|meeting\s*attendees|speakers)\s*[:\-–]\s*\n?([^\n#]+(?:\n(?!\s*(?:agenda|action|decision|summary|overview|topics|date))[^\n#]+)*)",
        raw_text,
        re.IGNORECASE,
    )
    if attendees_sec:
        block = attendees_sec.group(1)
        # Split on commas, semicolons, bullets, or newlines
        raw_names = re.split(r"[,;\n\r•\-\*]+", block)
        for rn in raw_names:
            name = rn.strip()
            # Remove parentheses like (Chair), (Product Manager), email addresses
            name = re.sub(r"\(.*?\)", "", name).strip()
            name = re.sub(r"<.*?>", "", name).strip()
            name = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "", name).strip()
            if len(name) >= 2 and len(name) <= 40 and not re.match(r"^(?:and|none|tbd|all|optional)$", name, re.IGNORECASE):
                if name not in speakers:
                    speakers.append(name)

    # Also detect inline dialogue prefixes like "Alice: We need to deploy" or "Bob (Tech Lead): Agree"
    dialogue_speakers = set(re.findall(r"^([A-Z][a-zA-Z\s]{1,25})(?:\s*\([^)]*\))?\s*:\s+", raw_text, re.MULTILINE))
    for ds in dialogue_speakers:
        ds_clean = ds.strip()
        if ds_clean.lower() not in {"note", "notes", "agenda", "action", "decision", "summary", "status", "date", "time", "topic", "item"}:
            if len(ds_clean) >= 2 and ds_clean not in speakers:
                speakers.append(ds_clean)

    # 4. Action Items Extraction
    action_items: List[Dict[str, Any]] = []
    actions_sec = re.search(
        r"(?:action\s*items?|next\s*steps?|follow\s*ups?|tasks?|to\s*do|action\s*points?)\s*[:\-–]?\s*\n?((?:[^\n#]+\n?)*?)(?=\n\s*#{1,3}|\n\s*(?:decisions?|summary|agenda|notes?:)|\Z)",
        raw_text,
        re.IGNORECASE,
    )
    if actions_sec:
        act_block = actions_sec.group(1)
        for act_line in act_block.splitlines():
            line = act_line.strip()
            if not line:
                continue
            # Remove bullet prefixes
            clean_act = re.sub(r"^[\s•\-\*\d\.\(\)\[\]xX]+\s*", "", line).strip()
            if len(clean_act) >= 5:
                # Try finding owner e.g. "(@John)", "(Owner: Alice)", "Bob to review..."
                owner = ""
                owner_m = re.search(r"\((?:owner|assigned(?:\s*to)?|lead)?\s*[:\-–]?\s*@?([A-Za-z\s]{2,25})\)", clean_act, re.IGNORECASE)
                if owner_m:
                    owner = owner_m.group(1).strip()
                elif "@" in clean_act:
                    at_m = re.search(r"@([A-Za-z0-9_\-]+)", clean_act)
                    if at_m:
                        owner = at_m.group(1)

                due_date = ""
                due_m = re.search(r"\((?:due|by|deadline)\s*[:\-–]?\s*([^)]+)\)", clean_act, re.IGNORECASE)
                if due_m:
                    due_date = due_m.group(1).strip()

                action_items.append({
                    "task": clean_act,
                    "owner": owner or "Unassigned",
                    "due_date": due_date,
                    "status": "pending",
                })

    # 5. Key Decisions Extraction
    decisions: List[str] = []
    dec_sec = re.search(
        r"(?:decisions?\s*made|key\s*decisions?|decisions?|conclusions?|resolutions?)\s*[:\-–]?\s*\n?((?:[^\n#]+\n?)*?)(?=\n\s*#{1,3}|\n\s*(?:action|next steps|summary|agenda)|\Z)",
        raw_text,
        re.IGNORECASE,
    )
    if dec_sec:
        dec_block = dec_sec.group(1)
        for dec_line in dec_block.splitlines():
            line = dec_line.strip()
            if not line:
                continue
            clean_dec = re.sub(r"^[\s•\-\*\d\.\(\)]+\s*", "", line).strip()
            if len(clean_dec) >= 5 and clean_dec not in decisions:
                decisions.append(clean_dec)

    # 6. Agendas & Discussion Sections
    agendas: List[Dict[str, Any]] = []
    # Try finding markdown sections (e.g. ## Agenda 1: Backend Architecture)
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    header_matches = list(header_pattern.finditer(raw_text))

    if len(header_matches) >= 2:
        for idx, match in enumerate(header_matches):
            h_title = match.group(2).strip()
            # Ignore headers that are purely action items / decisions / attendees if already extracted
            if re.match(r"^(?:attendees|participants|action items|next steps|decisions|summary)$", h_title, re.IGNORECASE):
                continue

            start_pos = match.end()
            end_pos = header_matches[idx + 1].start() if idx + 1 < len(header_matches) else len(raw_text)
            section_body = raw_text[start_pos:end_pos].strip()

            pts = []
            for bline in section_body.splitlines():
                bl = bline.strip()
                if bl and len(bl) >= 4:
                    pts.append(re.sub(r"^[\s•\-\*\d\.\(\)]+\s*", "", bl).strip())

            if section_body:
                agendas.append({
                    "agenda_id": f"agenda_{len(agendas) + 1}",
                    "title": h_title,
                    "discussion_points": pts if pts else [section_body[:500]],
                    "content": section_body,
                })

    # If no headers found, check for numbered Agendas like "1. Project Status", "2. Technical Review"
    if not agendas:
        numbered_agendas = re.finditer(r"(?:^|\n)(?:Agenda\s*)?(\d+)[\.\)]\s*([^\n]+)\n((?:(?!\n(?:\d+[\.\)]|Agenda\s*\d+))[^\n]+\n*)*)", raw_text, re.MULTILINE)
        for na in numbered_agendas:
            a_idx = na.group(1)
            a_title = na.group(2).strip()
            a_body = na.group(3).strip()
            if len(a_title) >= 3 and not re.match(r"^(?:action|attendee|decision|date)", a_title, re.IGNORECASE):
                pts = [re.sub(r"^[\s•\-\*\d\.\(\)]+\s*", "", l).strip() for l in a_body.splitlines() if len(l.strip()) >= 4]
                agendas.append({
                    "agenda_id": f"agenda_{a_idx}",
                    "title": a_title,
                    "discussion_points": pts if pts else ([a_body] if a_body else [a_title]),
                    "content": a_body or a_title,
                })

    # Default fallback agenda if none structured
    if not agendas:
        agendas.append({
            "agenda_id": "agenda_1",
            "title": "General Discussion & Meeting Minutes",
            "discussion_points": [line for line in lines[:20] if len(line) >= 10],
            "content": full_text[:2000],
        })

    # 7. Summary
    summary = ""
    summary_match = re.search(
        r"(?:executive\s*summary|summary|overview|meeting\s*recap|background)\s*[:\-–]?\s*\n?([^\n#]+(?:\n(?!\s*(?:agenda|action|decision|attendees|topics))[^\n#]+)*)",
        raw_text,
        re.IGNORECASE,
    )
    if summary_match:
        summary = summary_match.group(1).strip()
    else:
        # First 2-3 substantive lines of text
        substantive = [l for l in lines if len(l) > 30 and not re.match(r"^(?:date|title|meeting|attendees|participants)", l, re.IGNORECASE)]
        summary = " ".join(substantive[:3]) if substantive else full_text[:400]

    return {
        "title": title,
        "meeting_date": meeting_date,
        "speakers": speakers,
        "agendas": agendas,
        "action_items": action_items,
        "decisions": decisions,
        "summary": summary,
    }


# ── Semantic Chunking ────────────────────────────────────────────────────────

def chunk_manual_mom(raw_text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Produce rich, high-quality retrieval chunks with accurate metadata for vector indexing.
    Chunks are organized by section, agenda, action items, decisions, and overview.
    """
    chunks: List[Dict[str, Any]] = []
    chunk_index = 0
    all_speakers = meta.get("speakers") or []

    # 1. Summary Chunk
    summary = meta.get("summary", "").strip()
    if summary and len(summary) >= 40:
        chunks.append({
            "chunk_index": chunk_index,
            "agenda_title": "Executive Summary",
            "speakers": all_speakers,
            "text": f"[{meta.get('title', 'Meeting')} - Executive Summary]\n{summary}",
        })
        chunk_index += 1

    # 2. Agenda / Section Chunks
    for ag in meta.get("agendas", []):
        ag_title = ag.get("title") or "Discussion Point"
        body = ag.get("content") or "\n".join(ag.get("discussion_points") or [])
        body = body.strip()
        if not body:
            continue

        # Subdivide long sections (> 700 chars) into manageable paragraphs/blocks
        if len(body) > 700:
            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            if len(paragraphs) <= 1:
                # Split by individual lines or bullets
                paragraphs = [l.strip() for l in body.splitlines() if len(l.strip()) >= 10]

            curr_block = []
            curr_len = 0
            for p in paragraphs:
                curr_block.append(p)
                curr_len += len(p)
                if curr_len >= 500:
                    chunk_text = f"[{ag_title}]\n" + "\n".join(curr_block)
                    chunks.append({
                        "chunk_index": chunk_index,
                        "agenda_title": ag_title,
                        "speakers": all_speakers,
                        "text": chunk_text,
                    })
                    chunk_index += 1
                    curr_block = []
                    curr_len = 0

            if curr_block:
                chunk_text = f"[{ag_title}]\n" + "\n".join(curr_block)
                chunks.append({
                    "chunk_index": chunk_index,
                    "agenda_title": ag_title,
                    "speakers": all_speakers,
                    "text": chunk_text,
                })
                chunk_index += 1
        else:
            chunk_text = f"[{ag_title}]\n{body}"
            chunks.append({
                "chunk_index": chunk_index,
                "agenda_title": ag_title,
                "speakers": all_speakers,
                "text": chunk_text,
            })
            chunk_index += 1

    # 3. Key Decisions Chunk
    decisions = meta.get("decisions") or []
    if decisions:
        dec_lines = [f"• {d}" for d in decisions]
        chunks.append({
            "chunk_index": chunk_index,
            "agenda_title": "Key Decisions",
            "speakers": all_speakers,
            "text": f"[{meta.get('title', 'Meeting')} - Key Decisions]\n" + "\n".join(dec_lines),
        })
        chunk_index += 1

    # 4. Action Items Chunk
    actions = meta.get("action_items") or []
    if actions:
        act_lines = []
        for a in actions:
            owner_tag = f" (Owner: {a.get('owner')})" if a.get("owner") and a.get("owner") != "Unassigned" else ""
            due_tag = f" (Due: {a.get('due_date')})" if a.get("due_date") else ""
            act_lines.append(f"• {a.get('task')}{owner_tag}{due_tag}")

        chunks.append({
            "chunk_index": chunk_index,
            "agenda_title": "Action Items",
            "speakers": all_speakers,
            "text": f"[{meta.get('title', 'Meeting')} - Action Items]\n" + "\n".join(act_lines),
        })
        chunk_index += 1

    # Fallback if no chunks were formed: chunk raw text into 500-char segments
    if not chunks and raw_text:
        text_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        cur_buf = []
        cur_chars = 0
        for l in text_lines:
            cur_buf.append(l)
            cur_chars += len(l)
            if cur_chars >= 500:
                chunks.append({
                    "chunk_index": chunk_index,
                    "agenda_title": "Meeting Content",
                    "speakers": all_speakers,
                    "text": "\n".join(cur_buf),
                })
                chunk_index += 1
                cur_buf = []
                cur_chars = 0
        if cur_buf:
            chunks.append({
                "chunk_index": chunk_index,
                "agenda_title": "Meeting Content",
                "speakers": all_speakers,
                "text": "\n".join(cur_buf),
            })

    return chunks


# ── File Processing & Database Ingestion ─────────────────────────────────────

async def process_manual_mom_file(
    user_id: str,
    file_bytes: bytes,
    filename: str,
) -> Dict[str, Any]:
    """
    Process an uploaded Manual MoM file:
    1. Saves file to workspace uploads/manual_moms.
    2. Extracts clean text via doc_extractor (offline PP-OCRv6, PDF, DOCX, etc.).
    3. Infers meeting title, date, speakers, agendas, action items, decisions, summary.
    4. Chunks text into structured retrieval segments.
    5. Embeds chunks and adds to ChromaDB `final_rom_{user_id}` for AI Chat Priority 1 retrieval.
    6. Stores in `manual_moms` SQLite table.
    """
    from services.vector_store import get_final_rom_points_store
    from services.text_embedding_service import get_text_embedder

    mom_id = f"mom_{uuid.uuid4().hex[:12]}"
    clean_filename = os.path.basename(filename)
    file_ext = os.path.splitext(clean_filename)[1].lower()
    file_size = len(file_bytes)

    # 1. Save file to disk
    user_upload_dir = Path(settings.UPLOAD_DIR) / "manual_moms" / str(user_id)
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = user_upload_dir / f"{mom_id}_{clean_filename}"
    saved_path.write_bytes(file_bytes)
    str_path = str(saved_path)

    # 2. Extract Text
    logger.info(f"[Manual MoM] Extracting text for {clean_filename} ({file_size} bytes)...")
    raw_text = extract_text_from_file(str_path, clean_filename)
    if not raw_text or not raw_text.strip():
        # Fallback to UTF-8 decoding if text-based
        try:
            raw_text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            raw_text = ""

    if not raw_text or not raw_text.strip():
        raw_text = f"(No readable text could be extracted from {clean_filename})"

    # 3. Parse Metadata & Sections
    meta = extract_metadata_and_sections(raw_text, clean_filename)
    title = meta["title"]
    meeting_date = meta["meeting_date"]
    speakers = meta["speakers"]
    agendas = meta["agendas"]
    action_items = meta["action_items"]
    decisions = meta["decisions"]
    summary = meta["summary"]

    # 4. Chunk
    chunks = chunk_manual_mom(raw_text, meta)
    logger.info(f"[Manual MoM] Generated {len(chunks)} chunks for '{title}' (ID: {mom_id})")

    # 5. Embed and Ingest into ChromaDB final_rom_{user_id}
    embedder = get_text_embedder()
    dim = embedder.embedding_dim()
    embedder.load()

    chunk_texts = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        metadatas.append({
            "doc_id": mom_id,
            "meeting_id": mom_id,
            "recording_id": mom_id,
            "meeting_name": title,
            "date": meeting_date,
            "source": "manual_mom",
            "is_manual_mom": "true",
            "agenda_title": c.get("agenda_title", "Discussion"),
            "speakers": ",".join(c.get("speakers") or []),
            "clean_text": c["text"],
            "chunk_index": c["chunk_index"],
        })

    if chunk_texts:
        embeddings = embedder.encode(chunk_texts)
        final_store = get_final_rom_points_store(user_id, dim)
        final_store.load_or_create()
        final_store.add(texts=chunk_texts, metadatas=metadatas, embeddings=embeddings)
        logger.info(f"[Manual MoM] Ingested {len(chunk_texts)} vectors into ChromaDB for meeting {mom_id}")

    # 6. Database Storage
    now_iso = datetime.now(timezone.utc).isoformat()
    # Format agendas for ROM compatibility
    rom_agendas = []
    for ag in agendas:
        pts = ag.get("discussion_points") or [ag.get("content", "")]
        formatted_pts = []
        for p_idx, pt in enumerate(pts):
            formatted_pts.append({
                "id": f"{ag.get('agenda_id')}_pt_{p_idx}",
                "text": pt,
                "clean_text": pt,
                "speakers": speakers,
            })
        rom_agendas.append({
            "agenda_id": ag.get("agenda_id"),
            "title": ag.get("title"),
            "discussion_points": formatted_pts,
        })

    rom_data_dict = {
        "is_manual_mom": True,
        "source_type": "manual_mom",
        "final_rom": {
            "meeting_info": {
                "title": title,
                "date": meeting_date,
                "participants": speakers,
            },
            "agendas": rom_agendas,
            "action_items": action_items,
            "decisions": decisions,
            "summary": summary,
        },
    }
    rom_data_json = json.dumps(rom_data_dict)

    async with get_db_context() as session:
        # Insert into manual_moms table
        await session.execute(
            text("""
                INSERT OR REPLACE INTO manual_moms (
                    id, user_id, filename, title, meeting_date, file_path, file_size,
                    file_type, raw_text, summary, speakers, agendas, action_items,
                    decisions, metadata_json, chunks_json, created_at, updated_at
                ) VALUES (
                    :id, :user_id, :filename, :title, :meeting_date, :file_path, :file_size,
                    :file_type, :raw_text, :summary, :speakers, :agendas, :action_items,
                    :decisions, :metadata_json, :chunks_json, :created_at, :updated_at
                )
            """),
            {
                "id": mom_id,
                "user_id": user_id,
                "filename": clean_filename,
                "title": title,
                "meeting_date": meeting_date,
                "file_path": str_path,
                "file_size": file_size,
                "file_type": file_ext,
                "raw_text": raw_text,
                "summary": summary,
                "speakers": json.dumps(speakers),
                "agendas": json.dumps(agendas),
                "action_items": json.dumps(action_items),
                "decisions": json.dumps(decisions),
                "metadata_json": json.dumps(meta),
                "chunks_json": json.dumps(chunks),
                "created_at": now_iso,
                "updated_at": now_iso,
            },
        )


        # Upsert rom_metadata
        await session.execute(
            text("""
                INSERT OR REPLACE INTO rom_metadata (
                    id, recording_id, user_id, rom_data, created_at, updated_at
                ) VALUES (
                    :id, :recording_id, :user_id, :rom_data, :created_at, :updated_at
                )
            """),
            {
                "id": mom_id,
                "recording_id": mom_id,
                "user_id": user_id,
                "rom_data": rom_data_json,
                "created_at": now_iso,
                "updated_at": now_iso,
            },
        )
        await session.commit()

    logger.info(f"[Manual MoM] Successfully processed and stored MoM '{title}' ({mom_id})")

    return {
        "id": mom_id,
        "filename": clean_filename,
        "title": title,
        "meeting_date": meeting_date,
        "file_size": file_size,
        "file_type": file_ext,
        "chunks_count": len(chunks),
        "speakers_count": len(speakers),
        "agendas_count": len(agendas),
        "action_items_count": len(action_items),
        "decisions_count": len(decisions),
        "summary": summary,
        "created_at": now_iso,
    }


async def list_manual_moms(user_id: str) -> List[Dict[str, Any]]:
    """List all manual MoM uploads for the specified user."""
    async with get_db_context() as session:
        rows = (
            await session.execute(
                text("""
                    SELECT id, filename, title, meeting_date, file_size, file_type,
                           speakers, agendas, action_items, decisions, chunks_json,
                           created_at, updated_at
                    FROM manual_moms
                    WHERE user_id = :uid
                    ORDER BY created_at DESC
                """),
                {"uid": user_id},
            )
        ).mappings().fetchall()

        result = []
        for r in rows:
            speakers = from_json(r["speakers"], [])
            agendas = from_json(r["agendas"], [])
            action_items = from_json(r["action_items"], [])
            decisions = from_json(r["decisions"], [])
            chunks = from_json(r["chunks_json"], [])

            result.append({
                "id": r["id"],
                "filename": r["filename"],
                "title": r["title"],
                "meeting_date": r["meeting_date"] or "",
                "file_size": r["file_size"],
                "file_type": r["file_type"],
                "speakers_count": len(speakers),
                "agendas_count": len(agendas),
                "action_items_count": len(action_items),
                "decisions_count": len(decisions),
                "chunks_count": len(chunks),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })
        return result


async def get_manual_mom_detail(user_id: str, mom_id: str) -> Optional[Dict[str, Any]]:
    """Get full details of a specific manual MoM upload."""
    async with get_db_context() as session:
        row = (
            await session.execute(
                text("""
                    SELECT id, user_id, filename, title, meeting_date, file_path,
                           file_size, file_type, raw_text, summary, speakers,
                           agendas, action_items, decisions, metadata_json,
                           chunks_json, created_at, updated_at
                    FROM manual_moms
                    WHERE id = :mid AND user_id = :uid
                """),
                {"mid": mom_id, "uid": user_id},
            )
        ).mappings().fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "filename": row["filename"],
            "title": row["title"],
            "meeting_date": row["meeting_date"] or "",
            "file_path": row["file_path"],
            "file_size": row["file_size"],
            "file_type": row["file_type"],
            "raw_text": row["raw_text"],
            "summary": row["summary"] or "",
            "speakers": from_json(row["speakers"], []),
            "agendas": from_json(row["agendas"], []),
            "action_items": from_json(row["action_items"], []),
            "decisions": from_json(row["decisions"], []),
            "metadata_json": from_json(row["metadata_json"], {}),
            "chunks": from_json(row["chunks_json"], []),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


async def delete_manual_mom(user_id: str, mom_id: str) -> bool:
    """Delete a manual MoM from SQLite DB, ChromaDB vector store, and filesystem."""
    from services.vector_store import get_final_rom_points_store

    file_path = None
    async with get_db_context() as session:
        row = (
            await session.execute(
                text("SELECT file_path FROM manual_moms WHERE id = :mid AND user_id = :uid"),
                {"mid": mom_id, "uid": user_id},
            )
        ).mappings().fetchone()

        if row:
            file_path = row["file_path"]

        # Delete from manual_moms table
        await session.execute(
            text("DELETE FROM manual_moms WHERE id = :mid AND user_id = :uid"),
            {"mid": mom_id, "uid": user_id},
        )
        # Delete from recordings table
        await session.execute(
            text("DELETE FROM recordings WHERE id = :mid AND user_id = :uid"),
            {"mid": mom_id, "uid": user_id},
        )
        # Delete from rom_metadata table
        await session.execute(
            text("DELETE FROM rom_metadata WHERE recording_id = :mid AND user_id = :uid"),
            {"mid": mom_id, "uid": user_id},
        )
        await session.commit()

    # Remove from ChromaDB vector store
    try:
        final_store = get_final_rom_points_store(user_id)
        if final_store.exists():
            final_store.load_or_create()
            final_store.delete_by_filter("meeting_id", mom_id)
            logger.info(f"[Manual MoM] Purged vectors for {mom_id} from final_rom_{user_id}")
    except Exception as e:
        logger.warning(f"[Manual MoM] Error purging vectors from ChromaDB: {e}")

    # Remove file on disk
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"[Manual MoM] Error deleting file {file_path}: {e}")

    return True


async def reindex_manual_mom(user_id: str, mom_id: str) -> Dict[str, Any]:
    """Re-chunk and re-embed an existing manual MoM into ChromaDB."""
    from services.vector_store import get_final_rom_points_store
    from services.text_embedding_service import get_text_embedder

    detail = await get_manual_mom_detail(user_id, mom_id)
    if not detail:
        raise ValueError(f"Manual MoM '{mom_id}' not found")

    raw_text = detail["raw_text"]
    meta = extract_metadata_and_sections(raw_text, detail["filename"])
    chunks = chunk_manual_mom(raw_text, meta)

    # Re-embed
    embedder = get_text_embedder()
    dim = embedder.embedding_dim()
    embedder.load()

    final_store = get_final_rom_points_store(user_id, dim)
    final_store.load_or_create()
    final_store.delete_by_filter("meeting_id", mom_id)

    chunk_texts = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        metadatas.append({
            "doc_id": mom_id,
            "meeting_id": mom_id,
            "recording_id": mom_id,
            "meeting_name": meta["title"],
            "date": meta["meeting_date"],
            "source": "manual_mom",
            "is_manual_mom": "true",
            "agenda_title": c.get("agenda_title", "Discussion"),
            "speakers": ",".join(c.get("speakers") or []),
            "clean_text": c["text"],
            "chunk_index": c["chunk_index"],
        })

    if chunk_texts:
        embeddings = embedder.encode(chunk_texts)
        final_store.add(texts=chunk_texts, metadatas=metadatas, embeddings=embeddings)

    # Update chunks in DB
    now_iso = datetime.now(timezone.utc).isoformat()
    async with get_db_context() as session:
        await session.execute(
            text("""
                UPDATE manual_moms
                SET chunks_json = :chunks_json, updated_at = :now
                WHERE id = :mid AND user_id = :uid
            """),
            {
                "chunks_json": json.dumps(chunks),
                "now": now_iso,
                "mid": mom_id,
                "uid": user_id,
            },
        )
        await session.commit()

    return {
        "status": "success",
        "mom_id": mom_id,
        "chunks_indexed": len(chunks),
    }
