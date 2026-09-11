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
        "key": "agenda_compress_with_context",
        "name": "Agenda Document Compression with Context",
        "category": "Raw MoM",
        "description": "Extracts agenda topics from document using reference context.",
        "variables": ["{text}"],
    },
    # ── Collection AI ─────────────────────────────────────────────────────
    {
        "key": "collection_planning",
        "name": "Collection Chat Triage & Retrieval Planner",
        "category": "Collection AI",
        "description": "Evaluates user query and plans context retrieval from collection vector store.",
        "variables": ["{conversation_history}", "{question}"],
    },
    {
        "key": "collection_chat",
        "name": "Collection Chat Response Generation",
        "category": "Collection AI",
        "description": "Generates answers to user questions using retrieved multi-meeting collection context.",
        "variables": ["{conversation_history}", "{context}", "{question}"],
    },
    {
        "key": "collection_compare",
        "name": "Collection Meeting Comparison Report",
        "category": "Collection AI",
        "description": "Generates structured comparative report between two meetings in a collection.",
        "variables": ["{meeting_a_name}", "{meeting_a_date}", "{meeting_a_context}", "{meeting_b_name}", "{meeting_b_date}", "{meeting_b_context}"],
    },
    {
        "key": "collection_topic_growth",
        "name": "Collection Topic Growth Tracking Report",
        "category": "Collection AI",
        "description": "Tracks chronological evolution of a specific topic across meetings in a collection.",
        "variables": ["{topic}", "{meetings_context}"],
    },
    # ── ROM Pipeline ─────────────────────────────────────────────────────
    {
        "key": "rom_discussion",
        "name": "Stage 1: Extract Discussion Points",
        "category": "ROM",
        "description": "Extracts Stage 1 discussion points preserving exact transcript meaning (who said what to whom) and explicit assigner/assignee action items.",
        "variables": ["{previous_context}", "{video_context_section}", "{window_text}"],
    },
    {
        "key": "rom_discussion_embedded",
        "name": "Stage 1: Extract Discussion Points with Embedded Actions",
        "category": "ROM",
        "description": "Default Stage 1 prompt. Extracts structured discussion points weaving actions, commitments, and explicit action_owner directly into points.",
        "variables": ["{previous_context}", "{video_context_section}", "{window_text}"],
    },
    {
        "key": "rom_discussion_no_actions",
        "name": "Stage 1: Extract Discussion Points Only (No Actions)",
        "category": "ROM",
        "description": "Stage 1 prompt used when separate action extraction is enabled. Extracts structured discussion points while skipping action items.",
        "variables": ["{previous_context}", "{video_context_section}", "{window_text}"],
    },
    {
        "key": "rom_action_extraction",
        "name": "Stage 1: Separate Action Item Extraction",
        "category": "ROM",
        "description": "Stage 1 prompt used when separate action extraction is enabled. Focuses exclusively on extracting complete, self-contained action items with assigner/assignee/deadlines.",
        "variables": ["{previous_context}", "{video_context_section}", "{window_text}"],
    },
    {
        "key": "stage1_json_repair",
        "name": "Stage 1: Repair Malformed/Truncated JSON",
        "category": "ROM",
        "description": "Repairs malformed or truncated JSON output from Stage 1 discussion or action extraction calls without modifying content.",
        "variables": ["{invalid_json}"],
    },
    {
        "key": "mom_action_regen",
        "name": "View MoM Action Point Regeneration",
        "category": "MoM",
        "description": "Regenerates self-contained action items directly from window transcript when user requests action point regeneration in MoM view.",
        "variables": ["{window_text}"],
    },
    {
        "key": "mom_action_dedup",
        "name": "MoM Action Point Deduplication",
        "category": "MoM",
        "description": "Deduplicates and merges action points across meeting transcript windows.",
        "variables": ["{action_items_json}"],
    },
    {
        "key": "mom_extract_actions",
        "name": "Generate Action Point from Enhanced Point",
        "category": "MoM",
        "description": "Extracts action items, commitments, and deadlines from enhanced discussion points during MoM generation.",
        "variables": ["{points_json}"],
    },
    {
        "key": "rom_polish",
        "name": "Stage 2: Polish & Merge Points",
        "category": "ROM",
        "description": "Polishes discussion points, merges duplicates, and enriches points using retrieved context.",
        "variables": ["{original_points}", "{retrieved_context}"],
    },
    {
        "key": "rom_enhance_window",
        "name": "Stage 2: Enhance Window with Context",
        "category": "ROM",
        "description": "Enhances a window of points using independently retrieved Meeting and Global Context.",
        "variables": ["{window_json}", "{meeting_context}", "{global_context}"],
    },
    {
        "key": "rom_enhance_all_together",
        "name": "Stage 2: Enhance All Points Together",
        "category": "ROM",
        "description": "Enhances all discussion points from a meeting together in a single call using retrieved Meeting and Global Context.",
        "variables": ["{points_json}", "{meeting_context}", "{global_context}"],
    },
    {
        "key": "rom_deduplicate",
        "name": "Stage 2: Deduplicate High Similarity Points",
        "category": "ROM",
        "description": "Evaluates points with high semantic similarity (>=0.90) into duplicate, complementary, or different.",
        "variables": ["{candidate_points}"],
    },
    {
        "key": "rom_agenda",
        "name": "Stage 3: Generate ROM Agendas",
        "category": "ROM",
        "description": "Generates enriched agenda representations with keywords, concepts, and expected themes.",
        "variables": ["{agenda_input}", "{context}", "{previous_mom}"],
    },
    {
        "key": "rom_mom_expansion",
        "name": "Stage 3: Expand Agendas with Previous MoM",
        "category": "ROM",
        "description": "Expands current meeting agendas using Previous Meeting MoM documents.",
        "variables": ["{agenda_list}", "{previous_mom_docs}"],
    },
    {
        "key": "rom_agenda_assign_batch",
        "name": "Stage 3: Batch Agenda Assignment",
        "category": "ROM",
        "description": "Assigns each discussion point in a batch directly to the most appropriate agenda from the complete agenda list.",
        "variables": ["{agenda_reference}", "{discussion_order_guidance}", "{timeline_guidance}", "{batch_json}"],
    },
    {
        "key": "rom_agenda_doc_points",
        "name": "Stage 3: Extract Supporting Doc Points",
        "category": "ROM",
        "description": "Extracts 2–5 factual, agenda-specific points from an uploaded supporting document.",
        "variables": ["{agenda_title}", "{agenda_description}", "{document_text}"],
    },
    {
        "key": "rom_version_short",
        "name": "Short ROM Rewrite",
        "category": "ROM",
        "description": "Generates a Short ROM version per agenda. Processes all points for each agenda together. Focuses on action points and decisions only. Input fields: Point ID, Speaker, Discussion, Action Owner.",
        "variables": ["{agenda_title}", "{points_json}", "{rules_section}"],
    },
    {
        "key": "rom_version_medium",
        "name": "Medium ROM Rewrite",
        "category": "ROM",
        "description": "Generates a Medium ROM version per agenda. Processes all points for each agenda together. Produces an aggregated version with key discussion details and action points. Input fields: Point ID, Speaker, Discussion, Action Owner.",
        "variables": ["{agenda_title}", "{points_json}", "{rules_section}"],
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
        AGENDA_COMPRESS_WITH_CONTEXT_PROMPT,
        COLLECTION_PLANNING_PROMPT,
        COLLECTION_CHAT_PROMPT,
        COLLECTION_COMPARE_PROMPT,
        COLLECTION_TOPIC_GROWTH_PROMPT,
        ROM_DISCUSSION_EXTRACTION_PROMPT,
        ROM_DISCUSSION_EMBEDDED_PROMPT,
        ROM_DISCUSSION_NO_ACTION_ITEMS_PROMPT,
        ROM_ACTION_EXTRACTION_PROMPT,
        STAGE1_JSON_REPAIR_PROMPT,
        MOM_REGENERATE_ACTION_POINTS_PROMPT,
        MOM_EXTRACT_ACTIONS_FROM_POINTS_PROMPT,
        ROM_POLISH_PROMPT,
        ROM_ENHANCE_WINDOW_PROMPT,
        ROM_ENHANCE_ALL_TOGETHER_PROMPT,
        ROM_DEDUPLICATION_PROMPT,
        ROM_AGENDA_GENERATION_PROMPT,
        ROM_MOM_EXPANSION_PROMPT,
        ROM_AGENDA_ASSIGN_BATCH_PROMPT,
        ROM_AGENDA_DOC_POINTS_PROMPT,
        MOM_DEDUPLICATE_ACTION_POINTS_PROMPT,
        ROM_VERSION_SHORT_PROMPT,
        ROM_VERSION_MEDIUM_PROMPT,
    )
    return {
        "mom":                            MOM_PROMPT,
        "mom_merge":                      MOM_MERGE_PROMPT,
        "mom_action_dedup":               MOM_DEDUPLICATE_ACTION_POINTS_PROMPT,
        "mom_action_regen":               MOM_REGENERATE_ACTION_POINTS_PROMPT,
        "mom_extract_actions":            MOM_EXTRACT_ACTIONS_FROM_POINTS_PROMPT,
        "agenda_compress":                AGENDA_COMPRESS_PROMPT,
        "agenda_compress_with_context":   AGENDA_COMPRESS_WITH_CONTEXT_PROMPT,
        "reference_compress":             REFERENCE_COMPRESS_PROMPT,
        "agenda_from_summary":             AGENDA_FROM_SUMMARY_PROMPT,
        "executive_summary":              EXECUTIVE_SUMMARY_PROMPT,
        "short_summary":                  SHORT_SUMMARY_PROMPT,
        "detailed_summary":               DETAILED_SUMMARY_PROMPT,
        "chunk_summary":                  CHUNK_SUMMARY_PROMPT,
        "key_points":                     KEY_POINTS_PROMPT,
        "action_items":                   ACTION_ITEMS_PROMPT,
        "key_decisions":                  KEY_DECISIONS_PROMPT,
        "speaker_summary":                SPEAKER_SUMMARY_PROMPT,
        "speaker_key_points":             SPEAKER_KEY_POINTS_PROMPT,
        "speaker_action_items":           SPEAKER_ACTION_ITEMS_PROMPT,
        "collection_planning":            COLLECTION_PLANNING_PROMPT,
        "collection_chat":                COLLECTION_CHAT_PROMPT,
        "collection_compare":             COLLECTION_COMPARE_PROMPT,
        "collection_topic_growth":        COLLECTION_TOPIC_GROWTH_PROMPT,
        "rom_discussion":                 ROM_DISCUSSION_EXTRACTION_PROMPT,
        "rom_discussion_embedded":        ROM_DISCUSSION_EMBEDDED_PROMPT,
        "rom_discussion_no_actions":      ROM_DISCUSSION_NO_ACTION_ITEMS_PROMPT,
        "rom_action_extraction":         ROM_ACTION_EXTRACTION_PROMPT,
        "stage1_json_repair":             STAGE1_JSON_REPAIR_PROMPT,
        "rom_polish":                     ROM_POLISH_PROMPT,
        "rom_enhance_window":             ROM_ENHANCE_WINDOW_PROMPT,
        "rom_enhance_all_together":       ROM_ENHANCE_ALL_TOGETHER_PROMPT,
        "rom_deduplicate":                ROM_DEDUPLICATION_PROMPT,
        "rom_agenda":                     ROM_AGENDA_GENERATION_PROMPT,
        "rom_mom_expansion":              ROM_MOM_EXPANSION_PROMPT,
        "rom_agenda_assign_batch":        ROM_AGENDA_ASSIGN_BATCH_PROMPT,
        "rom_agenda_doc_points":          ROM_AGENDA_DOC_POINTS_PROMPT,
        "rom_version_short":              ROM_VERSION_SHORT_PROMPT,
        "rom_version_medium":             ROM_VERSION_MEDIUM_PROMPT,
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


def _load_cache_sync() -> None:
    """Synchronously load all custom templates from SQLite into _cache if not yet loaded."""
    global _cache_loaded
    if _cache_loaded:
        return
    import os
    import sqlite3
    try:
        from config import settings
        db_url = getattr(settings, "DATABASE_URL", "")
        db_path = None
        if db_url.startswith("sqlite+aiosqlite:///"):
            db_path = db_url[len("sqlite+aiosqlite:///"):]
        elif db_url.startswith("sqlite:///"):
            db_path = db_url[len("sqlite:///"):]
        elif db_url:
            db_path = db_url

        if db_path and os.path.exists(db_path):
            conn = sqlite3.connect(db_path, timeout=5.0)
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_templates'")
                if cursor.fetchone():
                    cursor.execute("SELECT key, template FROM prompt_templates")
                    rows = cursor.fetchall()
                    for r in rows:
                        k = r["key"]
                        t = r["template"]
                        if k in VALID_KEYS and t and t.strip():
                            _cache[k] = t
            finally:
                conn.close()
    except Exception as e:
        logger.warning(f"[PromptService] Sync cache load failed: {e}")
    _cache_loaded = True


async def get_prompt(db: AsyncSession, key: str) -> str:
    """Return the active template for key (custom or default)."""
    await _ensure_cache_loaded(db)
    if key in _cache and _cache[key].strip():
        return _cache[key]
    return _defaults().get(key, "")


def get_prompt_sync(key: str) -> str:
    """
    Synchronous cache-first lookup with sync DB fallback — used inside AI pipeline threads.

    If the in-process cache has not been loaded from the database yet, it will load
    all custom templates synchronously via sqlite3 so saved settings are always respected,
    even before any async endpoint has run.
    """
    if not _cache_loaded:
        _load_cache_sync()
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
