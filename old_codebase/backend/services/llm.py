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


def compress_agenda(text: str) -> str:
    """Compress raw agenda/objectives document text into a structured numbered list (offline, Qwen3 4B)."""
    return get_provider().compress_agenda(text)


def compress_reference(text: str) -> str:
    """Compress reference/context document text into a concise knowledge summary (offline, Qwen3 4B)."""
    return get_provider().compress_reference(text)


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


def generate_mom(
    transcript: List[Dict],
    recording_meta: dict,
    context: Optional[str] = None,
    agenda_summary: Optional[str] = None,
    reference_summary: Optional[str] = None,
) -> dict:
    """Generate Minutes of Meeting (offline, Qwen3 4B)."""
    return get_provider().generate_mom(
        transcript,
        recording_meta,
        context=context,
        agenda_summary=agenda_summary,
        reference_summary=reference_summary,
    )




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


def generate_agenda_from_summary(summary: str) -> List[Dict]:
    """Reconstruct structured agenda items from a transcription summary (offline, Qwen3 4B)."""
    return get_provider().generate_agenda_from_summary(summary)


# ── Rewrite helpers ───────────────────────────────────────────

def analyze_writing_style(document_text: str, section_type: str) -> str:
    """
    Analyze a reference document and extract writing style rules for a given section type.

    Args:
        document_text: Combined text from reference document(s).
        section_type: 'discussion_points' or 'action_items'.

    Returns:
        A newline-separated list of style/formatting rules as a string.
    """
    if section_type == "discussion_points":
        section_label = "discussion points / points discussed"
        guidance = (
            "Focus on: tone (formal/informal), structure (topic-then-detail vs free-form), "
            "level of detail per point, use of bullet sub-points, length of each point, "
            "use of technical vs plain language, tense, and any consistent formatting patterns."
        )
    else:
        section_label = "action items / tasks"
        guidance = (
            "Focus on: phrasing style (imperative vs descriptive), how tasks are described, "
            "specificity of deadlines, how ownership is expressed, formatting conventions, "
            "verbosity, and any consistent structural patterns."
        )

    prompt = f"""You are an expert writing analyst. Study the following document carefully and extract a concise set of writing style rules that describe HOW the {section_label} are written in this document.

DOCUMENT:
{document_text}

{guidance}

Output ONLY a numbered list of clear, actionable writing style rules (5–15 rules). Each rule should be a single sentence that could be used as an instruction to rewrite content in the same style.

Do NOT include rules about what topics to cover. Focus purely on HOW things are written (style, tone, structure, formatting, length, language).

Writing style rules:"""

    return get_provider()._infer(prompt, max_new_tokens=1024)


def rewrite_section_content(
    content: str,
    rules: str,
    custom_prompt: str,
    section_type: str,
) -> str:
    """
    Rewrite a piece of MoM content according to the provided style rules.

    Args:
        content: The raw content to rewrite (stringified for the LLM).
        rules: Style/formatting rules generated from reference documents.
        custom_prompt: Additional user-supplied instructions.
        section_type: 'discussion_points', 'action_items', 'introduction', or 'conclusion'.

    Returns:
        The rewritten content as a string (caller parses back to list/text as needed).
    """
    section_labels = {
        "discussion_points": "discussion points",
        "action_items": "action item task descriptions",
        "introduction": "meeting introduction paragraph",
        "conclusion": "meeting conclusion paragraph",
    }
    label = section_labels.get(section_type, "content")

    rules_block = ""
    if rules and rules.strip():
        rules_block = f"""STYLE RULES TO FOLLOW:
{rules.strip()}

"""

    custom_block = ""
    if custom_prompt and custom_prompt.strip():
        custom_block = f"""ADDITIONAL INSTRUCTIONS:
{custom_prompt.strip()}

"""

    prompt = f"""You are an expert meeting minutes editor. Rewrite the following {label} according to the style rules below.

CRITICAL REQUIREMENTS:
- Preserve ALL factual information, decisions, outcomes, questions, and task assignments exactly — do not add or remove facts.
- Only improve writing quality, tone, structure, and formatting.
- Return ONLY the rewritten {label} in exactly the same format as the input (numbered list for points/items, plain paragraph for intro/conclusion).
- Do NOT add explanations, comments, or preamble.

{rules_block}{custom_block}CONTENT TO REWRITE:
{content}

Rewritten {label}:"""

    logger.info(
        f"[MoM Rewrite Step 1/6 - LLM Request] Section: '{section_type}', "
        f"Input chars: {len(content)}, Rules chars: {len(rules)}, Custom prompt chars: {len(custom_prompt)}"
    )
    logger.debug(f"[MoM Rewrite Step 1/6 - Prompt Sent to LLM]:\n{prompt}")

    raw_response = get_provider()._infer(prompt, max_new_tokens=4096)
    
    logger.info(
        f"[MoM Rewrite Step 1/6 - Raw LLM Response Received] Section: '{section_type}', "
        f"Response chars: {len(raw_response or '')}, Raw lines: {len((raw_response or '').splitlines())}"
    )
    logger.info(f"[MoM Rewrite Step 1/6 - Raw LLM Content]:\n{raw_response}")

    return raw_response or ""


def summarize_long_discussion_point(point: str, threshold: int) -> str:
    """
    Condense a single long discussion point to be concise (target under threshold words)
    while preserving all essential factual information, decisions, key details, numbers, and outcomes.
    Preserves Topic: prefix if present.
    """
    prompt = f"""You are an expert meeting minutes editor. Condense the following discussion point to be concise (under {threshold} words) while preserving ALL important factual information, decisions, key details, numbers, and outcomes.

CRITICAL REQUIREMENTS:
- Preserve all facts, decisions, names, metrics, and outcomes — do NOT lose any key details.
- Condense the text so it is under {threshold} words.
- Return ONLY the condensed discussion point without explanations, intro, preamble, quotes, or markdown list numbering.

DISCUSSION POINT:
{point}

CONDENSED DISCUSSION POINT:"""

    logger.info(f"[Summarize Long Point] Condensing point ({len(point.split())} words) with threshold={threshold}")
    condensed = get_provider()._infer(prompt, max_new_tokens=1024)
    if condensed and condensed.strip():
        cleaned = condensed.strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
        import re
        cleaned = re.sub(r'^(\d+[\.\)\-]\s*|[\-*•]\s*)', '', cleaned).strip()
        return cleaned or point
    return point


def summarize_long_points(points: List[str], threshold: int = 70) -> tuple[List[str], int]:
    """
    Process a list of discussion points. Any point whose word count exceeds `threshold`
    is sent to the LLM to be condensed. Points at or below threshold remain unchanged.

    Returns:
        (updated_points_list, count_of_condensed_points)
    """
    updated_points = []
    condensed_count = 0

    for pt in points:
        pt_str = str(pt or "").strip()
        word_count = len(pt_str.split())
        if word_count > threshold:
            condensed_pt = summarize_long_discussion_point(pt_str, threshold)
            updated_points.append(condensed_pt)
            condensed_count += 1
        else:
            updated_points.append(pt_str)

    return updated_points, condensed_count



