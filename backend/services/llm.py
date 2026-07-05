"""
LLM service — thin shim over the QwenProvider AI layer.

All AI features (summarization, key points, action items, decisions,
executive summary, detailed summary, MoM) use the local Qwen3 4B
Instruct model. No cloud services. No internet required.

Callers (pipeline.py, routers) are unchanged — this shim preserves
the public API surface.

Context-caching pattern
------------------------
To avoid redundant _hierarchical_summarize() calls, callers should:
  1. Call ``build_context_summary(transcript)`` once.
  2. Store the result (e.g. in the DB as ``context_summary``).
  3. Pass it as ``context=...`` to all downstream generate_* calls.

When ``context`` is None every function falls back to computing the
hierarchical summary internally (backward-compatible behaviour).
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional

from services.ai_provider import get_provider

logger = logging.getLogger(__name__)


# ── Context builder (call once, cache, reuse) ─────────────────

def build_context_summary(transcript: List[Dict]) -> str:
    """
    Build the compressed hierarchical context summary for a transcript.

    Call this ONCE after the transcript is finalised, store the result
    in the DB (recordings.context_summary), then pass it as ``context``
    to every generate_* function below to skip all redundant LLM work.
    """
    return get_provider().build_context_summary(transcript)


# ── Public API (used by pipeline.py + routers) ────────────────

def generate_summary(transcript: List[Dict], context: Optional[str] = None) -> str:
    """Generate meeting summary (offline, Qwen3 4B)."""
    return get_provider().generate_summary(transcript, context=context)


def generate_key_points(transcript: List[Dict], context: Optional[str] = None) -> List[str]:
    """Extract key points (offline, Qwen3 4B)."""
    return get_provider().generate_key_points(transcript, context=context)


def generate_action_items(transcript: List[Dict], context: Optional[str] = None) -> List[str]:
    """Extract action items (offline, Qwen3 4B)."""
    return get_provider().generate_action_items(transcript, context=context)


def generate_key_decisions(transcript: List[Dict], context: Optional[str] = None) -> List[str]:
    """Extract key decisions (offline, Qwen3 4B)."""
    return get_provider().generate_key_decisions(transcript, context=context)


def generate_mom(transcript: List[Dict], recording_meta: dict, context: Optional[str] = None) -> dict:
    """Generate Minutes of Meeting (offline, Qwen3 4B)."""
    return get_provider().generate_mom(transcript, recording_meta, context=context)


def generate_executive_summary(transcript: List[Dict], context: Optional[str] = None) -> dict:
    """Generate executive summary for PDF report (offline, Qwen3 4B)."""
    return get_provider().generate_executive_summary(transcript, context=context)


def generate_short_summary(transcript: List[Dict], context: Optional[str] = None) -> str:
    """Generate a concise ~120-word meeting summary (offline, Qwen3 4B)."""
    return get_provider().generate_short_summary(transcript, context=context)


def generate_detailed_summary(transcript: List[Dict], context: Optional[str] = None) -> str:
    """Generate a comprehensive detailed meeting report (offline, Qwen3 4B)."""
    return get_provider().generate_detailed_summary(transcript, context=context)


def generate_speaker_summaries(transcript: List[Dict]) -> dict:
    """Generate per-speaker summaries, key points, and action items (offline, Qwen3 4B)."""
    return get_provider().generate_speaker_summaries(transcript)
