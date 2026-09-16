"""
ROM Training Dataset Service
Handles: fetching Long ROM points, extracting manual MoM structure, agenda matching, DB persistence.
"""
import json
import logging
import uuid
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from database import to_json, from_json

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    """Simple word-overlap Jaccard similarity for agenda title matching."""
    a_words = set(re.sub(r'[^a-z0-9 ]', '', a.lower()).split())
    b_words = set(re.sub(r'[^a-z0-9 ]', '', b.lower()).split())
    if not a_words or not b_words:
        return 0.0
    intersection = a_words & b_words
    union = a_words | b_words
    return len(intersection) / union


def _safe_parse_rom_data(raw_data: Any) -> dict:
    """
    Safely parse rom_data from database, handling JSON strings, double-encoded JSON,
    literal 'null', invalid JSON, and non-dict values without raising exceptions.
    """
    if not raw_data:
        return {}
    if isinstance(raw_data, dict):
        return raw_data
    try:
        parsed = from_json(raw_data, default={})
        # If double-encoded as JSON string, attempt a second parse
        if isinstance(parsed, str):
            parsed_str = parsed.strip()
            if parsed_str.lower() in ("null", "none", ""):
                return {}
            try:
                parsed = json.loads(parsed_str)
            except Exception:
                return {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        logger.warning(f"[RomTrainingDataset] Failed to parse rom_data: {e}")
        return {}


# ── Long ROM Points ─────────────────────────────────────────────────────────

async def get_meeting_long_rom_points(
    recording_id: str,
    user_id: str,
    db: AsyncSession,
) -> Optional[Dict[str, Any]]:
    """
    Returns the Long ROM points for a meeting, organized by agenda.
    Checks both rom_metadata and recordings tables.
    Returns None if no ROM data or no final ROM.
    """
    r = await db.execute(
        text("""
            SELECT COALESCE(rm.rom_data, r.rom_data)
            FROM recordings r
            LEFT JOIN rom_metadata rm ON rm.recording_id = r.id
            WHERE r.id = :rid AND r.user_id = :uid
        """),
        {"rid": recording_id, "uid": user_id},
    )
    row = r.fetchone()
    if not row or not row[0]:
        return None

    rom_data = _safe_parse_rom_data(row[0])
    if not rom_data:
        return None

    versions = rom_data.get("final_rom_versions") if isinstance(rom_data, dict) else {}
    if not isinstance(versions, dict):
        versions = {}

    # Prefer long version, then fallback to final_rom/medium/short
    long_rom = (
        versions.get("long")
        or (rom_data.get("final_rom") if isinstance(rom_data, dict) else None)
        or versions.get("medium")
        or versions.get("short")
    )
    if not isinstance(long_rom, dict):
        return None

    agendas = long_rom.get("agendas", [])
    if not agendas or not isinstance(agendas, list):
        return None

    result = []
    for ag in agendas:
        if isinstance(ag, dict):
            result.append({
                "agenda_id": ag.get("agenda_id", ""),
                "agenda_title": ag.get("title", "General"),
                "discussion_points": ag.get("discussion_points", []),
            })

    if not result:
        return None

    return {"agendas": result}


async def get_meetings_with_rom_status(user_id: str, db: AsyncSession) -> List[Dict[str, Any]]:
    """List all meetings for the user with their ROM completion status."""
    r = await db.execute(
        text("""
            SELECT r.id, r.title, r.filename, r.created_at,
                   rm.id as has_rm_id, COALESCE(rm.rom_data, r.rom_data) as rom_data
            FROM recordings r
            LEFT JOIN rom_metadata rm ON rm.recording_id = r.id
            WHERE r.user_id = :uid
            ORDER BY r.created_at DESC
        """),
        {"uid": user_id},
    )
    rows = r.fetchall()
    result = []
    for row in rows:
        try:
            rom_data = _safe_parse_rom_data(row[5])

            versions = rom_data.get("final_rom_versions") if isinstance(rom_data, dict) else {}
            if not isinstance(versions, dict):
                versions = {}

            # Detect final_rom_versions.long and fallback appropriately
            final_rom = (
                versions.get("long")
                or (rom_data.get("final_rom") if isinstance(rom_data, dict) else None)
                or versions.get("medium")
                or versions.get("short")
            )

            has_long_rom = bool(
                isinstance(final_rom, dict)
                and final_rom.get("agendas")
                and len(final_rom["agendas"]) > 0
            )

            # has_rom indicates any valid ROM generation exists
            has_rom = bool(
                row[4]
                or has_long_rom
                or (isinstance(rom_data, dict) and bool(rom_data.get("stage1") or rom_data.get("stage2") or rom_data.get("stage3")))
            )

            result.append({
                "id": row[0],
                "title": row[1] or row[2] or "Untitled",
                "created_at": row[3],
                "has_rom": has_rom,
                "has_long_rom": has_long_rom,
            })
        except Exception as err:
            logger.error(f"[RomTrainingDataset] Error processing meeting row {row[0]}: {err}")
            result.append({
                "id": row[0],
                "title": row[1] or row[2] or "Untitled",
                "created_at": row[3],
                "has_rom": bool(row[4]),
                "has_long_rom": False,
            })
    return result



# ── MoM Extraction ──────────────────────────────────────────────────────────

def extract_mom_points_from_text(text_content: str, user_id: str) -> List[Dict[str, Any]]:
    """
    Extract agenda items and their points from manually written MoM text.
    Uses LLM to identify the structure (headings and bullet points).
    Points are extracted EXACTLY as written — no modification, summarization, or merging.
    Returns list of {agenda_title, points: [str]}.
    """
    from services.ai_provider import get_provider
    from services.prompt_service import get_prompt_sync

    provider = get_provider()

    prompt = f"""You are a structured data extractor. Extract the agenda sections and their bullet points from the following Minutes of Meeting document.

RULES (STRICTLY FOLLOW):
1. Extract EVERY agenda/section heading as a separate entry.
2. Under each heading, extract ALL bullet points EXACTLY as written — do NOT add, modify, summarize, split, merge, or alter any text.
3. Preserve the complete original text of every point.
4. If there is no clear section heading, use "General" as the agenda title.
5. Return ONLY a JSON array. Each element: {{"agenda_title": "...", "points": ["...", "...", ...]}}
6. Do NOT include any explanation or markdown fences.

Minutes of Meeting:
{text_content[:8000]}

JSON array:"""

    try:
        if hasattr(provider, 'query'):
            raw = provider.query(prompt, max_tokens=4096, temperature=0.0)
        else:
            raw = provider._infer(prompt, max_new_tokens=4096)

        # Strip think tags
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        # Strip markdown fences
        raw = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()

        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception as e:
        logger.warning(f"[RomTrainingDataset] MoM extraction failed: {e}. Falling back to regex.")
        return _extract_mom_regex(text_content)


def _extract_mom_regex(text_content: str) -> List[Dict[str, Any]]:
    """Fallback regex-based MoM extraction."""
    lines = text_content.splitlines()
    agendas = []
    current_agenda = None
    current_points = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Detect heading: short line without leading bullet, or ALL CAPS / ends with colon
        is_heading = (
            (not re.match(r'^[\-\*•\d]', stripped) and len(stripped) < 80 and stripped.endswith(':'))
            or re.match(r'^[A-Z][A-Z ]{5,}$', stripped)
        )
        if is_heading:
            if current_agenda is not None:
                agendas.append({"agenda_title": current_agenda, "points": current_points})
            current_agenda = stripped.rstrip(':')
            current_points = []
        elif re.match(r'^[\-\*•]\s+', stripped):
            point = re.sub(r'^[\-\*•]\s+', '', stripped)
            current_points.append(point)
        elif current_agenda and stripped:
            current_points.append(stripped)

    if current_agenda is not None and current_points:
        agendas.append({"agenda_title": current_agenda, "points": current_points})
    elif not agendas and current_points:
        agendas.append({"agenda_title": "General", "points": current_points})

    return agendas


# ── Agenda Matching ──────────────────────────────────────────────────────────

def match_agendas(
    mom_agendas: List[Dict],
    meeting_agendas: List[Dict],
) -> List[Dict[str, Any]]:
    """
    Fuzzy-match MoM agendas to meeting agendas by title similarity.
    Returns list of AgendaMatchItem dicts.
    Each mom agenda gets its best match (or None if similarity < 0.2).
    """
    results = []
    for mom_idx, mom_ag in enumerate(mom_agendas):
        mom_title = mom_ag.get("agenda_title", "")
        best_match = None
        best_score = 0.0

        for meeting_ag in meeting_agendas:
            m_title = meeting_ag.get("agenda_title", meeting_ag.get("title", ""))
            score = _similarity(mom_title, m_title)
            if score > best_score:
                best_score = score
                best_match = meeting_ag

        results.append({
            "mom_agenda_title": mom_title,
            "mom_agenda_idx": mom_idx,
            "meeting_agenda_id": best_match.get("agenda_id") if best_match and best_score >= 0.2 else None,
            "meeting_agenda_title": best_match.get("agenda_title", best_match.get("title")) if best_match and best_score >= 0.2 else None,
            "confidence": round(best_score, 3),
            "manual": False,
        })
    return results


# ── DB Persistence ───────────────────────────────────────────────────────────

async def save_training_data(
    user_id: str,
    recording_id: str,
    recording_title: Optional[str],
    pairs: List[Dict[str, Any]],
    db: AsyncSession,
) -> List[str]:
    """
    Save matched agenda pairs to rom_training_data.
    Each pair: {agenda_id, agenda_title, long_rom_points, manual_mom_points}.
    Upserts by (user_id, recording_id, agenda_id).
    Returns list of saved IDs.
    """
    now = datetime.now(timezone.utc).isoformat()
    saved_ids = []

    for pair in pairs:
        # Check if exists
        r = await db.execute(
            text("""
                SELECT id FROM rom_training_data
                WHERE user_id = :uid AND recording_id = :rid AND agenda_id = :aid
            """),
            {"uid": user_id, "rid": recording_id, "aid": pair["agenda_id"]},
        )
        existing = r.fetchone()

        if existing:
            row_id = existing[0]
            await db.execute(
                text("""
                    UPDATE rom_training_data
                    SET recording_title = :rtitle,
                        agenda_title = :atitle,
                        long_rom_points = :lrp,
                        manual_mom_points = :mmp
                    WHERE id = :id
                """),
                {
                    "rtitle": recording_title,
                    "atitle": pair["agenda_title"],
                    "lrp": to_json(pair.get("long_rom_points", [])),
                    "mmp": to_json(pair.get("manual_mom_points", [])),
                    "id": row_id,
                },
            )
        else:
            row_id = str(uuid.uuid4())
            await db.execute(
                text("""
                    INSERT INTO rom_training_data
                    (id, user_id, recording_id, recording_title, agenda_id, agenda_title,
                     long_rom_points, manual_mom_points, created_at)
                    VALUES (:id, :uid, :rid, :rtitle, :aid, :atitle, :lrp, :mmp, :now)
                """),
                {
                    "id": row_id,
                    "uid": user_id,
                    "rid": recording_id,
                    "rtitle": recording_title,
                    "aid": pair["agenda_id"],
                    "atitle": pair["agenda_title"],
                    "lrp": to_json(pair.get("long_rom_points", [])),
                    "mmp": to_json(pair.get("manual_mom_points", [])),
                    "now": now,
                },
            )
        saved_ids.append(row_id)

    await db.commit()
    return saved_ids


async def list_training_data(user_id: str, db: AsyncSession) -> List[Dict[str, Any]]:
    """List all training data rows for a user."""
    r = await db.execute(
        text("""
            SELECT id, recording_id, recording_title, agenda_id, agenda_title,
                   long_rom_points, manual_mom_points, created_at
            FROM rom_training_data
            WHERE user_id = :uid
            ORDER BY created_at DESC
        """),
        {"uid": user_id},
    )
    rows = r.fetchall()
    result = []
    for row in rows:
        result.append({
            "id": row[0],
            "recording_id": row[1],
            "recording_title": row[2],
            "agenda_id": row[3],
            "agenda_title": row[4],
            "long_rom_points": from_json(row[5], default=[]),
            "manual_mom_points": from_json(row[6], default=[]),
            "created_at": row[7],
        })
    return result


async def get_training_data_for_meeting(user_id: str, recording_id: str, db: AsyncSession) -> List[Dict[str, Any]]:
    """Get all training data rows for a specific meeting."""
    r = await db.execute(
        text("""
            SELECT id, agenda_id, agenda_title, long_rom_points, manual_mom_points, created_at
            FROM rom_training_data
            WHERE user_id = :uid AND recording_id = :rid
            ORDER BY created_at ASC
        """),
        {"uid": user_id, "rid": recording_id},
    )
    rows = r.fetchall()
    return [{
        "id": row[0],
        "agenda_id": row[1],
        "agenda_title": row[2],
        "long_rom_points": from_json(row[3], default=[]),
        "manual_mom_points": from_json(row[4], default=[]),
        "created_at": row[5],
    } for row in rows]


async def delete_training_data(data_id: str, user_id: str, db: AsyncSession) -> bool:
    """Delete a training data entry."""
    r = await db.execute(
        text("DELETE FROM rom_training_data WHERE id = :id AND user_id = :uid"),
        {"id": data_id, "uid": user_id},
    )
    await db.commit()
    return r.rowcount > 0


async def sample_training_data(
    user_id: str,
    n: int,
    db: AsyncSession,
    exclude_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Randomly sample n training data rows from DB."""
    import random
    all_data = await list_training_data(user_id, db)
    if exclude_ids:
        all_data = [d for d in all_data if d["id"] not in exclude_ids]
    random.shuffle(all_data)
    return all_data[:n]
