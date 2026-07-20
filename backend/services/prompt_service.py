"""
Prompt Template Service — system-wide, shared by all users.

Architecture
------------
- Defaults come from the hardcoded constants in services.ai_provider (never removed).
- Custom overrides are stored in the SQLite `prompt_templates` table.
- An in-process LRU-style dict (_cache) is invalidated on every write,
  so changes take effect on the next AI call without a restart.
- If a prompt is missing, blank, or invalid, the hardcoded default is returned.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)

# ── In-process cache: key → custom template string ─────────────────────────
# Invalidated on write; populated lazily from DB on first read.
_cache: dict[str, str] = {}
_cache_loaded = False          # True after we have done the first full DB load


def _invalidate(key: str | None = None) -> None:
    """Invalidate one key, or the whole cache if key is None."""
    global _cache_loaded
    if key is None:
        _cache.clear()
        _cache_loaded = False
    else:
        _cache.pop(key, None)


# ── Prompt metadata (drives the Settings UI) ────────────────────────────────
PROMPT_META: list[dict] = [
    # ── MoM pipeline ──────────────────────────────────────────────────────
    {
        "key": "mom",
        "name": "Minutes of Meeting",
        "category": "MoM",
        "description": "Main prompt used to generate the final Minutes of Meeting from a meeting transcript.",
        "variables": ["{transcript}", "{agenda_section}", "{reference_section}"],
    },
    {
        "key": "mom_merge",
        "name": "MoM Section Merge",
        "category": "MoM",
        "description": "Merges multiple partial MoMs (generated section-by-section) into one consolidated document.",
        "variables": ["{partial_moms_json}"],
    },
    {
        "key": "raw_mom_to_mom",
        "name": "Raw MoM → Final MoM",
        "category": "MoM",
        "description": "Converts structured Raw MoM JSON (from the Raw MoM Lab) into a polished final MoM.",
        "variables": ["{raw_mom_text}"],
    },
    # ── Raw MoM pipeline ──────────────────────────────────────────────────
    {
        "key": "raw_mom_extraction",
        "name": "Raw MoM Extraction",
        "category": "Raw MoM",
        "description": "Per-agenda RAG extraction prompt. Extracts structured discussion entries from retrieved evidence for one agenda topic.",
        "variables": ["{agenda_topic}", "{agenda_speaker}", "{evidence}"],
    },
    {
        "key": "agenda_compress",
        "name": "Agenda Document Compression",
        "category": "Raw MoM",
        "description": "Parses and extracts agenda items from an uploaded document (PDF, DOCX, etc.).",
        "variables": ["{text}"],
    },
    {
        "key": "reference_compress",
        "name": "Reference Document Compression",
        "category": "Raw MoM",
        "description": "Extracts key facts from uploaded reference/context documents used as background knowledge.",
        "variables": ["{text}"],
    },
    {
        "key": "agenda_from_summary",
        "name": "Agenda Reconstructed from Summary",
        "category": "Raw MoM",
        "description": "Generates a list of structured agenda items from a meeting's transcription summary when no agenda file is provided.",
        "variables": ["{summary}"],
    },
    # ── Summaries ─────────────────────────────────────────────────────────
    {
        "key": "executive_summary",
        "name": "Executive Summary",
        "category": "Summaries",
        "description": "Generates a structured executive summary (Purpose, Discussion Points, Outcomes, Next Steps) for the meeting report.",
        "variables": ["{transcript}"],
    },
    {
        "key": "short_summary",
        "name": "Short Summary",
        "category": "Summaries",
        "description": "Generates a 120-word single-paragraph summary of the meeting.",
        "variables": ["{transcript}"],
    },
    {
        "key": "detailed_summary",
        "name": "Detailed Summary",
        "category": "Summaries",
        "description": "Generates a detailed multi-section report covering each major discussion topic.",
        "variables": ["{transcript}"],
    },
    {
        "key": "chunk_summary",
        "name": "Chunk Hierarchical Summary",
        "category": "Summaries",
        "description": "Summarizes a single 10-minute transcript chunk into 3–7 sentences. Used for hierarchical context compression.",
        "variables": ["{chunk}"],
    },
    # ── Analysis ─────────────────────────────────────────────────────────
    {
        "key": "key_points",
        "name": "Key Points",
        "category": "Analysis",
        "description": "Extracts the main discussion topics and key points from the meeting transcript.",
        "variables": ["{transcript}"],
    },
    {
        "key": "action_items",
        "name": "Action Items",
        "category": "Analysis",
        "description": "Extracts all action items from the transcript, grouped by speaker.",
        "variables": ["{transcript}"],
    },
    {
        "key": "key_decisions",
        "name": "Key Decisions",
        "category": "Analysis",
        "description": "Extracts all concrete decisions agreed upon during the meeting.",
        "variables": ["{transcript}"],
    },
    # ── Speaker ──────────────────────────────────────────────────────────
    {
        "key": "speaker_summary",
        "name": "Speaker Summary",
        "category": "Speaker",
        "description": "Summarizes a specific speaker's contributions from their transcript lines.",
        "variables": ["{speaker}", "{transcript}"],
    },
    {
        "key": "speaker_key_points",
        "name": "Speaker Key Points",
        "category": "Speaker",
        "description": "Extracts 3–6 key points from a specific speaker's contributions.",
        "variables": ["{speaker}", "{transcript}"],
    },
    {
        "key": "speaker_action_items",
        "name": "Speaker Action Items",
        "category": "Speaker",
        "description": "Extracts action items assigned to or committed by a specific speaker.",
        "variables": ["{speaker}", "{transcript}"],
    },
    {
        "key": "raw_mom_repair",
        "name": "Raw MoM JSON Repair",
        "category": "Raw MoM",
        "description": "Repairs malformed JSON generated by the raw MoM extraction pipeline.",
        "variables": ["{raw_json}"],
    },
]

VALID_KEYS: frozenset[str] = frozenset(m["key"] for m in PROMPT_META)


def _defaults() -> dict[str, str]:
    """
    Lazily import defaults from ai_provider.py constants.
    Deferred import avoids a circular dependency since ai_provider.py imports
    this module after the constants are defined.
    """
    from services.ai_provider import (
        MOM_PROMPT,
        MOM_MERGE_PROMPT,
        RAW_MOM_TO_MOM_PROMPT,
        RAW_MOM_EXTRACTION_PROMPT,
        AGENDA_COMPRESS_PROMPT,
        REFERENCE_COMPRESS_PROMPT,
        AGENDA_FROM_SUMMARY_PROMPT,
        EXECUTIVE_SUMMARY_PROMPT,
        SHORT_SUMMARY_PROMPT,
        DETAILED_SUMMARY_PROMPT,
        CHUNK_SUMMARY_PROMPT,
        KEY_POINTS_PROMPT,
        ACTION_ITEMS_PROMPT,
        KEY_DECISIONS_PROMPT,
        SPEAKER_SUMMARY_PROMPT,
        SPEAKER_KEY_POINTS_PROMPT,
        SPEAKER_ACTION_ITEMS_PROMPT,
        RAW_MOM_REPAIR_PROMPT,
    )
    return {
        "mom":                MOM_PROMPT,
        "mom_merge":          MOM_MERGE_PROMPT,
        "raw_mom_to_mom":     RAW_MOM_TO_MOM_PROMPT,
        "raw_mom_extraction": RAW_MOM_EXTRACTION_PROMPT,
        "agenda_compress":    AGENDA_COMPRESS_PROMPT,
        "reference_compress": REFERENCE_COMPRESS_PROMPT,
        "agenda_from_summary": AGENDA_FROM_SUMMARY_PROMPT,
        "executive_summary":  EXECUTIVE_SUMMARY_PROMPT,
        "short_summary":      SHORT_SUMMARY_PROMPT,
        "detailed_summary":   DETAILED_SUMMARY_PROMPT,
        "chunk_summary":      CHUNK_SUMMARY_PROMPT,
        "key_points":         KEY_POINTS_PROMPT,
        "action_items":       ACTION_ITEMS_PROMPT,
        "key_decisions":      KEY_DECISIONS_PROMPT,
        "speaker_summary":    SPEAKER_SUMMARY_PROMPT,
        "speaker_key_points": SPEAKER_KEY_POINTS_PROMPT,
        "speaker_action_items": SPEAKER_ACTION_ITEMS_PROMPT,
        "raw_mom_repair":     RAW_MOM_REPAIR_PROMPT,
    }


# ── Core async API ─────────────────────────────────────────────────────────


async def _ensure_cache_loaded(db: AsyncSession) -> None:
    """Load all custom templates from DB into _cache on first call."""
    global _cache_loaded
    if _cache_loaded:
        return
    rows = await db.execute(text("SELECT key, template FROM prompt_templates"))
    for row in rows.mappings():
        if row["key"] in VALID_KEYS and row["template"].strip():
            _cache[row["key"]] = row["template"]
    _cache_loaded = True


async def get_prompt(db: AsyncSession, key: str) -> str:
    """Return the active template for key (custom or default)."""
    await _ensure_cache_loaded(db)
    if key in _cache and _cache[key].strip():
        return _cache[key]
    return _defaults().get(key, "")


def get_prompt_sync(key: str) -> str:
    """
    Synchronous cache-only lookup — used inside AI pipeline threads.

    If the key is in the in-process cache, return it. Otherwise return
    the hardcoded default. This never hits the DB, so it is safe to call
    from synchronous worker threads.
    """
    if key in _cache and _cache[key].strip():
        return _cache[key]
    return _defaults().get(key, "")


async def set_prompt(db: AsyncSession, key: str, template: str) -> None:
    """Save a custom template and invalidate the cache entry."""
    if key not in VALID_KEYS:
        raise ValueError(f"Unknown prompt key: {key!r}")
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        text("""
            INSERT INTO prompt_templates (key, template, updated_at)
            VALUES (:key, :template, :now)
            ON CONFLICT(key) DO UPDATE SET template = excluded.template, updated_at = excluded.updated_at
        """),
        {"key": key, "template": template, "now": now},
    )
    await db.commit()
    _cache[key] = template
    logger.info(f"[PromptService] Template '{key}' saved ({len(template)} chars)")


async def reset_prompt(db: AsyncSession, key: str) -> None:
    """Delete a custom template; future calls will use the hardcoded default."""
    if key not in VALID_KEYS:
        raise ValueError(f"Unknown prompt key: {key!r}")
    await db.execute(text("DELETE FROM prompt_templates WHERE key = :key"), {"key": key})
    await db.commit()
    _cache.pop(key, None)
    logger.info(f"[PromptService] Template '{key}' reset to default")


async def reset_all_prompts(db: AsyncSession) -> None:
    """Delete ALL custom templates; all fall back to hardcoded defaults."""
    await db.execute(text("DELETE FROM prompt_templates"))
    await db.commit()
    _invalidate()
    logger.info("[PromptService] All templates reset to defaults")


async def list_prompts(db: AsyncSession) -> list[dict]:
    """Return metadata + current value for every prompt."""
    await _ensure_cache_loaded(db)
    defaults = _defaults()
    result = []
    # Load DB row timestamps for display
    rows = await db.execute(text("SELECT key, updated_at FROM prompt_templates"))
    ts_map: dict[str, str] = {r["key"]: r["updated_at"] for r in rows.mappings()}

    for meta in PROMPT_META:
        key = meta["key"]
        is_custom = key in _cache and _cache[key].strip()
        result.append({
            **meta,
            "template": _cache[key] if is_custom else defaults.get(key, ""),
            "default_template": defaults.get(key, ""),
            "is_modified": bool(is_custom),
            "updated_at": ts_map.get(key),
        })
    return result


async def export_prompts(db: AsyncSession) -> dict:
    """Export all active templates (custom + defaults) as a JSON-serialisable dict."""
    await _ensure_cache_loaded(db)
    defaults = _defaults()
    return {
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "templates": {
            key: _cache.get(key) or defaults.get(key, "")
            for key in VALID_KEYS
        },
    }


async def import_prompts(db: AsyncSession, data: dict) -> dict[str, Any]:
    """
    Import templates from a JSON export dict.
    Only valid keys with non-empty strings are imported.
    Returns a summary of imported / skipped counts.
    """
    if not isinstance(data, dict) or "templates" not in data:
        raise ValueError("Invalid import format — expected {\"version\": 1, \"templates\": {...}}")

    templates = data["templates"]
    if not isinstance(templates, dict):
        raise ValueError("'templates' must be a dict")

    imported, skipped = 0, 0
    for key, template in templates.items():
        if key not in VALID_KEYS:
            skipped += 1
            continue
        if not isinstance(template, str) or not template.strip():
            skipped += 1
            continue
        await set_prompt(db, key, template)
        imported += 1

    logger.info(f"[PromptService] Import: {imported} imported, {skipped} skipped")
    return {"imported": imported, "skipped": skipped}
