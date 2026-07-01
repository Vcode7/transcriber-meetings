"""
Prompt Builder Service

Combines Global Prompt + Meeting Prompt + Participant Names + Technical Vocabulary
into a single Whisper initial_prompt, respecting the ~224-token / ~896-char budget.
"""
import logging
from typing import List

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
) -> str:
    """
    Combine all prompt components into a single Whisper initial_prompt string.

    Priority / truncation order (highest to lowest):
      1. Global prompt       — always included if set
      2. Meeting prompt      — included next
      3. Participant names   — compact line
      4. Vocabulary terms    — appended last, truncated first if budget exceeded
    """
    participant_names = participant_names or []
    vocabulary = vocabulary or []

    parts: List[str] = []

    if global_prompt and global_prompt.strip():
        parts.append(global_prompt.strip())

    if meeting_prompt and meeting_prompt.strip():
        parts.append(meeting_prompt.strip())

    if participant_names:
        clean_names = [n.strip() for n in participant_names if n.strip()]
        if clean_names:
            parts.append("Speakers: " + ", ".join(clean_names) + ".")

    if use_vocabulary and vocabulary:
        # Build vocab string, truncating to fit budget
        current_len = sum(len(p) + 1 for p in parts)
        remaining = _MAX_CHARS - current_len - len("Terms: ")
        if remaining > 20:
            vocab_terms: List[str] = []
            running = 0
            for word in vocabulary:
                addition = len(word) + 2  # word + ", "
                if running + addition > remaining:
                    break
                vocab_terms.append(word)
                running += addition
            if vocab_terms:
                parts.append("Terms: " + ", ".join(vocab_terms) + ".")

    result = "\n".join(parts)

    # Hard truncation guard
    if len(result) > _MAX_CHARS:
        result = result[:_MAX_CHARS].rsplit(" ", 1)[0]
        logger.warning(
            f"[PromptBuilder] Prompt truncated to {len(result)} chars "
            f"(budget={_MAX_CHARS})"
        )

    logger.debug(f"[PromptBuilder] Built prompt ({len(result)} chars): {result[:100]}...")
    return result
