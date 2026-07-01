"""
LLM service — thin shim over the QwenProvider AI layer.

All AI features (summarization, key points, action items, decisions,
executive summary, detailed summary, MoM) use the local Qwen3 4B
Instruct model. No cloud services. No internet required.

Callers (pipeline.py, routers) are unchanged — this shim preserves
the public API surface.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from services.ai_provider import get_provider

logger = logging.getLogger(__name__)


# ── Public API (used by pipeline.py + routers) ────────────────

def generate_summary(transcript: List[Dict]) -> str:
    """Generate meeting summary (offline, Qwen3 4B)."""
    return get_provider().generate_summary(transcript)


def generate_key_points(transcript: List[Dict]) -> List[str]:
    """Extract key points (offline, Qwen3 4B)."""
    return get_provider().generate_key_points(transcript)


def generate_action_items(transcript: List[Dict]) -> List[str]:
    """Extract action items (offline, Qwen3 4B)."""
    return get_provider().generate_action_items(transcript)


def generate_key_decisions(transcript: List[Dict]) -> List[str]:
    """Extract key decisions (offline, Qwen3 4B)."""
    return get_provider().generate_key_decisions(transcript)


def generate_mom(transcript: List[Dict], recording_meta: dict) -> dict:
    """Generate Minutes of Meeting (offline, Qwen3 4B)."""
    return get_provider().generate_mom(transcript, recording_meta)


def generate_executive_summary(transcript: List[Dict]) -> dict:
    """Generate executive summary for PDF report (offline, Qwen3 4B)."""
    return get_provider().generate_executive_summary(transcript)


def generate_short_summary(transcript: List[Dict]) -> str:
    """Generate a concise ~60-word meeting summary (offline, Qwen3 4B)."""
    return get_provider().generate_short_summary(transcript)


def generate_detailed_summary(transcript: List[Dict]) -> str:
    """Generate a comprehensive detailed meeting report (offline, Qwen3 4B)."""
    return get_provider().generate_detailed_summary(transcript)


def generate_speaker_summaries(transcript: List[Dict]) -> dict:
    """Generate per-speaker summaries, key points, and action items (offline, Qwen3 4B)."""
    return get_provider().generate_speaker_summaries(transcript)
