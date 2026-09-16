"""
correction_service.py - Post-Transcription Correction and Acronym Validation

Pipeline stage that runs ONCE after the full transcription pipeline completes:
  Transcription -> Diarization -> Speaker ID -> Final Transcript
    -> Correction and Acronym Detection (this service) -> Save to DB

Features:
  1. ASR Spelling Correction: LLM detects obvious ASR errors, returns JSON proposals.
     Only plain transcript text is sent (no timestamps, confidence, metadata).
     LLM must return: [{"wrong": "...", "correct": "..."}]
     Application applies replacements itself; LLM never rewrites the transcript.

  2. Acronym Detection: Regex-based, no LLM needed.
     Detects ALL-CAPS tokens (2+ chars) and common acronym patterns.
     Checks against persistent per-user acronym_dictionary.
     Unknown acronyms surfaced to user for manual full-form entry.

  3. Apply / Revert:
     Word-aware replacements (\bword\b) -- no uncontrolled global string replace.
     Original transcript saved before apply for safe revert.

Safety guarantees:
  LLM/JSON failure -> [] corrections returned, transcript untouched.
  wrong token validated to exist in transcript before saving.
  All Stage 9 errors are non-fatal.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Match ALL-CAPS contiguous sequences of 2+ letters, optionally followed by digits (e.g. LLM, API, GPT4, IB)
_CONTIGUOUS_ACRONYM_PATTERN = re.compile(r'\b[A-Z]{2,}[0-9]*\b')
_ACRONYM_PATTERN = _CONTIGUOUS_ACRONYM_PATTERN  # Backwards compatibility alias

# Match spaced single-letter uppercase sequences of 2 to 8 letters (e.g. L L M, I B, A I, N L P)
_SPACED_ACRONYM_PATTERN = re.compile(r'\b[A-Z](?:\s+[A-Z]){1,7}\b')

# Match dotted uppercase sequences of 2 to 8 letters (e.g. U.S.A., A.I., A.I)
_DOTTED_ACRONYM_PATTERN = re.compile(r'\b[A-Z](?:\.[A-Z]){1,7}\.?')

# Common English words that happen to appear all-caps or spaced in transcripts -- skip these
_COMMON_WORDS_SKIP = frozenset({
    "I", "A", "AN", "THE", "AND", "OR", "NOT", "BUT", "FOR", "SO", "IS",
    "ARE", "WAS", "BE", "BY", "ON", "IN", "AT", "TO", "OF", "IF", "IT",
    "OK", "HI", "NO", "YES", "DO", "GO", "CAN", "MAY", "WILL", "SHALL",
    "UP", "US", "WE", "ME", "MY", "AM", "HE", "AS", "HAD", "HAS", "HER",
    "HIM", "HIS", "HOW", "ITS", "NOW", "OUR", "OUT", "SEE", "SHE", "TOO",
    "WHO", "WHY", "ALL", "ANY", "DID", "GET", "LET", "SAY",
})


# =============================================================================
# 1. Plain-text extraction -- ONLY text, no metadata
# =============================================================================

def extract_plain_text(segments: List[Dict[str, Any]]) -> str:
    """
    Extract only the spoken text from transcript segments.
    Never includes timestamps, confidence scores, speaker info, or any metadata.
    """
    texts = []
    for seg in segments:
        t = (seg.get("text") or "").strip()
        if t:
            texts.append(t)
    return " ".join(texts)


# =============================================================================
# 2. LLM-based ASR correction detection
# =============================================================================

_ASR_CORRECTION_PROMPT = """\
You are a speech recognition error detector. Your ONLY job is to find obvious ASR (automatic speech recognition) spelling mistakes -- words that were mis-transcribed due to how they sound.

CRITICAL RULES:
- Do NOT rewrite, paraphrase, summarize, improve grammar, or change meaning.
- Do NOT correct punctuation, capitalization, or style.
- Do NOT change proper nouns or technical terms unless clearly misspelled.
- ONLY detect clear ASR word errors where the "wrong" token is exactly as it appears in the transcript.
- The "wrong" field MUST exactly match a token that appears verbatim in the TRANSCRIPT below.
- If nothing is obviously wrong, return [].

Return ONLY a JSON array (no markdown, no explanation):
[{"wrong": "aplication", "correct": "application"}, {"wrong": "transciption", "correct": "transcription"}]

TRANSCRIPT:
{transcript}

JSON corrections:"""


def detect_asr_corrections(plain_text: str) -> List[Dict[str, str]]:
    """
    Send only the plain transcript text to the LLM to detect obvious ASR errors.

    Returns:
        List of {"wrong": "...", "correct": "..."} dicts.
        Returns [] on LLM failure, JSON parse failure, or empty transcript.
    """
    if not plain_text or not plain_text.strip():
        logger.info("[Correction] Empty transcript -- skipping LLM correction detection.")
        return []

    # Truncate very long transcripts to avoid context overflow (~12k chars ~= ~3k tokens)
    max_chars = 12000
    truncated = plain_text[:max_chars]
    if len(plain_text) > max_chars:
        logger.info(
            f"[Correction] Transcript truncated from {len(plain_text)} to {max_chars} chars for LLM."
        )

    prompt = _ASR_CORRECTION_PROMPT.format(transcript=truncated)

    try:
        from services.ai_provider import get_provider
        provider = get_provider()
        logger.info(f"[Correction] Sending {len(truncated)} chars to LLM for ASR correction detection.")
        raw = provider._infer(prompt, max_new_tokens=512)
        logger.info(f"[Correction] LLM raw response ({len(raw or '')} chars): {repr((raw or '')[:200])}")
    except Exception as e:
        logger.warning(f"[Correction] LLM inference failed (non-fatal): {e}")
        return []

    return _parse_and_validate_corrections(raw, plain_text)


def _parse_and_validate_corrections(
    raw: str,
    plain_text: str,
) -> List[Dict[str, str]]:
    """
    Parse the LLM JSON response and validate each entry:
    - Must have "wrong" and "correct" keys (strings)
    - "wrong" must exactly appear in the plain_text
    - "wrong" != "correct"
    - Not an empty string
    """
    if not raw or not raw.strip():
        logger.info("[Correction] LLM returned empty response -- no corrections.")
        return []

    # Extract JSON array from the response (handle cases where LLM adds extra text)
    text_to_parse = raw.strip()
    match = re.search(r'\[.*\]', text_to_parse, re.DOTALL)
    if match:
        text_to_parse = match.group(0)

    try:
        parsed = json.loads(text_to_parse)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[Correction] Failed to parse LLM JSON response: {e}. Raw: {repr(text_to_parse[:300])}")
        return []

    if not isinstance(parsed, list):
        logger.warning(f"[Correction] LLM returned non-list JSON: {type(parsed)}. Ignoring.")
        return []

    valid = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        wrong = item.get("wrong", "")
        correct = item.get("correct", "")
        if not isinstance(wrong, str) or not isinstance(correct, str):
            continue
        wrong = wrong.strip()
        correct = correct.strip()
        if not wrong or not correct or wrong == correct:
            continue
        # Validate: "wrong" must actually appear in the transcript
        if not re.search(r'\b' + re.escape(wrong) + r'\b', plain_text, re.IGNORECASE):
            logger.debug(f"[Correction] Skipping '{wrong}' -- not found in transcript.")
            continue
        valid.append({"wrong": wrong, "correct": correct})

    logger.info(f"[Correction] {len(valid)} valid correction(s) detected by LLM.")
    return valid


# =============================================================================
# 3. Acronym detection (regex + dictionary lookup)
# =============================================================================

async def detect_acronyms(
    plain_text: str,
    user_id: str,
    db,
) -> List[Dict[str, Any]]:
    """
    Detect acronyms in the transcript text (contiguous, spaced, or dotted).

    Steps:
      1. Regex scan for contiguous (LLM, IB), spaced (L L M, I B), and dotted (U.S.A., A.I.) patterns
      2. Canonicalize to unspaced uppercase form (e.g. "L L M" -> "LLM")
      3. Deduplicate; record first occurrence offset
      4. Lookup each canonical acronym in user's acronym_dictionary
      5. Return [{acronym, full_form, is_known, first_occurrence}]
    """
    if not plain_text or not plain_text.strip():
        return []

    # Gather all candidate matches across contiguous, spaced, and dotted patterns
    raw_candidates: List[tuple[str, int]] = []
    for m in _CONTIGUOUS_ACRONYM_PATTERN.finditer(plain_text):
        raw_candidates.append((m.group(0), m.start()))
    for m in _SPACED_ACRONYM_PATTERN.finditer(plain_text):
        raw_candidates.append((m.group(0), m.start()))
    for m in _DOTTED_ACRONYM_PATTERN.finditer(plain_text):
        raw_candidates.append((m.group(0), m.start()))

    # Sort candidates by character position
    raw_candidates.sort(key=lambda x: x[1])

    # Find canonical acronyms with their first occurrence position
    found: Dict[str, int] = {}  # canonical acronym -> first char offset
    for word, pos in raw_candidates:
        canonical = re.sub(r'[\s.]', '', word).upper()
        if len(canonical) < 2:
            continue
        if canonical in _COMMON_WORDS_SKIP:
            continue
        if canonical not in found:
            found[canonical] = pos

    if not found:
        logger.info("[Acronym] No acronyms detected in transcript.")
        return []

    logger.info(f"[Acronym] {len(found)} unique acronym candidate(s) detected: {list(found.keys())[:20]}")

    # Lookup all in dictionary (single query)
    acronym_list = list(found.keys())
    try:
        # SQLite IN clause with tuple
        placeholders = ",".join([f":a{i}" for i in range(len(acronym_list))])
        params = {f"a{i}": acr for i, acr in enumerate(acronym_list)}
        params["uid"] = user_id
        rows = await db.execute(
            text(
                f"SELECT acronym, full_form FROM acronym_dictionary "
                f"WHERE user_id = :uid AND acronym IN ({placeholders})"
            ),
            params,
        )
        known = {row[0]: row[1] for row in rows.fetchall()}
    except Exception as e:
        logger.warning(f"[Acronym] Dictionary lookup failed (non-fatal): {e}")
        known = {}

    result = []
    for acr, pos in sorted(found.items(), key=lambda x: x[1]):
        is_known = acr in known
        result.append({
            "acronym": acr,
            "full_form": known.get(acr),
            "is_known": is_known,
            "first_occurrence": pos,
        })

    logger.info(
        f"[Acronym] {sum(1 for r in result if r['is_known'])} known, "
        f"{sum(1 for r in result if not r['is_known'])} unknown."
    )
    return result


# =============================================================================
# 4. Apply corrections to transcript (word-aware, non-destructive)
# =============================================================================

def apply_corrections_to_transcript(
    transcript: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
) -> tuple:
    """
    Apply user-approved corrections to the transcript.

    For each decision where enabled=True:
      - Uses word-aware regex \bwrong\b to replace in seg["text"] and word["word"]
      - Applies user_correct if provided, else falls back to LLM's correct

    Returns:
        (new_transcript, total_replacements_count)
        On exception: returns (original_transcript, 0) unchanged
    """
    if not decisions:
        return transcript, 0

    enabled = [
        d for d in decisions
        if d.get("enabled", True) and d.get("wrong", "").strip()
    ]
    if not enabled:
        return transcript, 0

    try:
        new_transcript = copy.deepcopy(transcript)
        total_replaced = 0

        for decision in enabled:
            wrong = decision.get("wrong", "").strip()
            # user_correct takes priority over LLM's correct
            correct = (decision.get("user_correct") or decision.get("correct") or "").strip()
            if not wrong or not correct or wrong == correct:
                continue

            pattern = re.compile(r'\b' + re.escape(wrong) + r'\b')

            for seg in new_transcript:
                # Apply to segment text
                old_text = seg.get("text", "")
                new_text, n = pattern.subn(correct, old_text)
                if n:
                    seg["text"] = new_text
                    total_replaced += n

                # Apply to individual word tokens
                for word_obj in seg.get("words", []):
                    w = word_obj.get("word", "")
                    new_w, wn = pattern.subn(correct, w)
                    if wn:
                        word_obj["word"] = new_w

        logger.info(
            f"[Correction] Applied {len(enabled)} correction rule(s); "
            f"{total_replaced} total replacement(s) in transcript."
        )
        return new_transcript, total_replaced

    except Exception as e:
        logger.error(
            f"[Correction] apply_corrections_to_transcript failed -- original transcript preserved: {e}",
            exc_info=True,
        )
        return transcript, 0


# =============================================================================
# 5. Apply acronym expansions (first-occurrence only)
# =============================================================================

def _build_acronym_pattern(acronym: str) -> re.Pattern:
    """
    Build a regex pattern that matches the acronym in all common forms:
      - Contiguous: LLM, llm
      - Spaced: L L M, l l m
      - Dotted: L.L.M., L.L.M, l.l.m
    Uses lookarounds to avoid matching within larger alphanumeric words.
    """
    clean = re.sub(r'[\s.]', '', acronym)
    letters = list(clean.upper())
    if not letters:
        return re.compile(r"(?!)")
    SEP = r"[\s.\-,]*"
    letter_pats = [re.escape(c) + r"\.?" for c in letters]
    inner = SEP.join(letter_pats)
    pattern = r"(?<![A-Za-z0-9])" + inner + r"(?![A-Za-z0-9])"
    return re.compile(pattern, re.IGNORECASE)


def apply_acronym_expansions_to_transcript(
    transcript: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Expand known acronyms on their FIRST occurrence only.
    Format: "LLM" or "L L M" -> "LLM (Large Language Model)"
    Only enabled decisions with a full_form are applied.
    Returns modified copy; original unchanged on exception.
    """
    enabled = [
        d for d in decisions
        if d.get("enabled", True)
        and not d.get("not_an_acronym", False)
        and (d.get("user_full_form") or d.get("full_form"))
    ]
    if not enabled:
        return transcript

    try:
        new_transcript = copy.deepcopy(transcript)
        expanded: set = set()

        for seg in new_transcript:
            text_val = seg.get("text", "")
            for dec in enabled:
                acr = dec.get("acronym", "").strip()
                if not acr or acr in expanded:
                    continue
                full = (dec.get("user_full_form") or dec.get("full_form") or "").strip()
                if not full:
                    continue
                expansion = f"{acr} ({full})"
                pattern = _build_acronym_pattern(acr)
                new_text, n = pattern.subn(expansion, text_val, count=1)
                if n:
                    seg["text"] = new_text
                    text_val = new_text
                    expanded.add(acr)

        logger.info(f"[Acronym] Expanded {len(expanded)} acronym(s) in transcript.")
        return new_transcript

    except Exception as e:
        logger.error(
            f"[Acronym] apply_acronym_expansions failed -- original transcript preserved: {e}",
            exc_info=True,
        )
        return transcript


# =============================================================================
# 6. Pipeline orchestrator -- runs as Stage 9
# =============================================================================

async def run_correction_pipeline(
    recording_id: str,
    user_id: str,
    raw_text: str,
    final_segments: List[Dict[str, Any]],
    loop,
) -> None:
    """
    Orchestrates ASR correction detection + acronym detection and persists
    results to transcript_corrections and transcript_acronyms tables.

    Called ONCE after the pipeline completes (Stage 9).
    Never modifies recordings.transcript directly.
    All failures are logged; none propagate.
    """
    from database import get_db_context, to_json

    logger.info(f"[Correction] {recording_id} -- Stage 9 correction pipeline starting.")

    plain_text = extract_plain_text(final_segments)
    if not plain_text.strip():
        logger.warning(f"[Correction] {recording_id} -- Empty transcript text; skipping correction pipeline.")
        return

    # Step 1: LLM ASR correction detection
    corrections: List[Dict[str, str]] = []
    try:
        corrections = await loop.run_in_executor(
            None, lambda: detect_asr_corrections(plain_text)
        )
    except Exception as e:
        logger.warning(f"[Correction] {recording_id} -- LLM correction detection failed (non-fatal): {e}")

    # Step 2: Acronym detection (regex + dictionary)
    acronyms: List[Dict[str, Any]] = []
    try:
        async with get_db_context() as db:
            acronyms = await detect_acronyms(plain_text, user_id, db)
    except Exception as e:
        logger.warning(f"[Correction] {recording_id} -- Acronym detection failed (non-fatal): {e}")

    now = datetime.now(timezone.utc).isoformat()

    # Step 3a: Persist corrections
    try:
        async with get_db_context() as db:
            existing = await db.execute(
                text("SELECT id FROM transcript_corrections WHERE recording_id = :rid"),
                {"rid": recording_id},
            )
            row = existing.fetchone()
            if row:
                await db.execute(
                    text("""
                        UPDATE transcript_corrections
                        SET proposed_corrections = :proposals,
                            user_decisions = '[]',
                            applied = 0,
                            original_transcript = NULL,
                            updated_at = :now
                        WHERE recording_id = :rid
                    """),
                    {"proposals": to_json(corrections), "now": now, "rid": recording_id},
                )
            else:
                await db.execute(
                    text("""
                        INSERT INTO transcript_corrections
                            (id, recording_id, user_id, proposed_corrections, user_decisions,
                             applied, original_transcript, created_at, updated_at)
                        VALUES (:id, :rid, :uid, :proposals, '[]', 0, NULL, :now, :now)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "rid": recording_id,
                        "uid": user_id,
                        "proposals": to_json(corrections),
                        "now": now,
                    },
                )
            await db.commit()
        logger.info(f"[Correction] {recording_id} -- {len(corrections)} correction proposal(s) saved.")
    except Exception as e:
        logger.error(f"[Correction] {recording_id} -- Failed to save corrections to DB: {e}", exc_info=True)

    # Step 3b: Persist acronyms
    try:
        async with get_db_context() as db:
            existing = await db.execute(
                text("SELECT id FROM transcript_acronyms WHERE recording_id = :rid"),
                {"rid": recording_id},
            )
            row = existing.fetchone()
            if row:
                await db.execute(
                    text("""
                        UPDATE transcript_acronyms
                        SET detected_acronyms = :acrs,
                            user_decisions = '[]',
                            applied = 0,
                            updated_at = :now
                        WHERE recording_id = :rid
                    """),
                    {"acrs": to_json(acronyms), "now": now, "rid": recording_id},
                )
            else:
                await db.execute(
                    text("""
                        INSERT INTO transcript_acronyms
                            (id, recording_id, user_id, detected_acronyms, user_decisions,
                             applied, created_at, updated_at)
                        VALUES (:id, :rid, :uid, :acrs, '[]', 0, :now, :now)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "rid": recording_id,
                        "uid": user_id,
                        "acrs": to_json(acronyms),
                        "now": now,
                    },
                )
            await db.commit()
        logger.info(f"[Correction] {recording_id} -- {len(acronyms)} acronym(s) saved.")
    except Exception as e:
        logger.error(f"[Correction] {recording_id} -- Failed to save acronyms to DB: {e}", exc_info=True)

    logger.info(f"[Correction] {recording_id} -- Stage 9 correction pipeline complete.")
