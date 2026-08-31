"""
Prompt Builder Service

Combines Dictionary Shortcuts + Technical Vocabulary + Global Prompt + Meeting Prompt
into a single Whisper initial_prompt, respecting the ~224-token / ~896-char budget.

Priority / truncation order (highest-value first):
  1. Shortcut dictionary  — always included; teaches Whisper "SR -> Sprint Review" etc.
  2. Technical vocabulary — always included regardless of the use_vocabulary flag
  3. Global prompt        — user's standing instructions
  4. Meeting prompt       — per-meeting context / custom instruction (appended last)

If the user supplies a meeting_prompt, it is appended AFTER the dictionary terms
rather than replacing them.  Dictionary terms are always included by default.
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Whisper's initial_prompt is prepended to the first decoding window.
# Hard-limit to avoid degrading transcription quality.
_MAX_CHARS = 800


def build_whisper_prompt(
    global_prompt: str = "",
    meeting_prompt: str = "",
    participant_names: List[str] = None,
    vocabulary: List[str] = None,
    use_vocabulary: bool = False,
    shortcuts: List[Dict[str, str]] = None,
) -> str:
    """
    Combine all prompt components into a single Whisper initial_prompt string.

    ``shortcuts`` — list of dicts with ``shortcut`` and ``full_form`` keys,
    fetched from the ``shortcut_dictionary`` table.  Included unconditionally
    (no flag required) so Whisper always learns domain-specific abbreviations.

    ``vocabulary`` — list of bare words from ``technical_vocabulary``.
    Also included unconditionally so proper nouns / jargon are recognised.

    The user's ``global_prompt`` and ``meeting_prompt`` are *appended* after
    the dictionary block so they can add context without overwriting term hints.
    """
    participant_names = participant_names or []
    vocabulary = vocabulary or []
    shortcuts = shortcuts or []

    parts: List[str] = []
    budget = _MAX_CHARS

    # ── Block 1: Shortcut dictionary (always included) ───────────────────────
    # Format: "SR: Sprint Review, MoM: Minutes of Meeting, ..."
    # Keeps each entry as "abbr: full form" so Whisper learns the expansion.
    if shortcuts:
        shortcut_tokens: List[str] = []
        running = len("Dict: ")
        for item in shortcuts:
            sc = (item.get("shortcut") or "").strip()
            if not sc:
                continue
            token = f"{sc}"
            addition = len(token) + 2  # token + ", "
            if running + addition > budget * 0.45:  # cap at 45 % of budget
                break
            shortcut_tokens.append(token)
            running += addition
        if shortcut_tokens:
            sc_line = "Dict: " + ", ".join(shortcut_tokens) + "."
            parts.append(sc_line)
            budget -= len(sc_line) + 1

    # ── Block 2: Technical vocabulary (always included) ──────────────────────
    if vocabulary:
        vocab_tokens: List[str] = []
        running = len("Terms: ")
        remaining = min(budget * 0.30, budget - sum(len(p) + 1 for p in parts) - len("Terms: "))
        for word in vocabulary:
            addition = len(word) + 2  # word + ", "
            if running + addition > remaining:
                break
            vocab_tokens.append(word)
            running += addition
        if vocab_tokens:
            vocab_line = "Terms: " + ", ".join(vocab_tokens) + "."
            parts.append(vocab_line)

    # ── Block 3: Participant names ────────────────────────────────────────────
    if participant_names:
        clean_names = [n.strip() for n in participant_names if n.strip()]
        if clean_names:
            name_line = "Speakers: " + ", ".join(clean_names) + "."
            current_len = sum(len(p) + 1 for p in parts)
            if current_len + len(name_line) + 1 <= _MAX_CHARS:
                parts.append(name_line)

    # ── Block 4: Global prompt (user's standing instructions) ────────────────
    if global_prompt and global_prompt.strip():
        gp = global_prompt.strip()
        current_len = sum(len(p) + 1 for p in parts)
        if current_len + len(gp) + 1 <= _MAX_CHARS:
            parts.append(gp)
        else:
            # Truncate global prompt to fit
            remaining = _MAX_CHARS - current_len - 1
            if remaining > 20:
                parts.append(gp[:remaining].rsplit(" ", 1)[0])

    # ── Block 5: Meeting prompt (per-meeting custom context, appended last) ───
    if meeting_prompt and meeting_prompt.strip():
        mp = meeting_prompt.strip()
        current_len = sum(len(p) + 1 for p in parts)
        if current_len + len(mp) + 1 <= _MAX_CHARS:
            parts.append(mp)
        else:
            remaining = _MAX_CHARS - current_len - 1
            if remaining > 20:
                parts.append(mp[:remaining].rsplit(" ", 1)[0])

    result = "\n".join(parts)

    # Hard truncation guard
    if len(result) > _MAX_CHARS:
        result = result[:_MAX_CHARS].rsplit(" ", 1)[0]
        logger.warning(
            f"[PromptBuilder] Prompt truncated to {len(result)} chars "
            f"(budget={_MAX_CHARS})"
        )

    logger.debug(f"[PromptBuilder] Built prompt ({len(result)} chars): {result[:120]}...")
    return result
