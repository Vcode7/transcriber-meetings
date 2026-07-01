"""
Transcript Normalizer Service

Expands abbreviations from the Shortcut Dictionary in transcript text.

Handles all common variant forms of an abbreviation, e.g. for "LLM":
  - LLM, llm, Llm
  - L L M, L.L.M, L-L-M, L,L,M, l l m
  - LL M, L LM, etc.

Uses word-boundary assertions to avoid replacing inside larger words.
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def _build_pattern(shortcut: str) -> re.Pattern:
    """
    Build a regex that matches the shortcut in all normalised forms:
      - Contiguous: LLM, llm
      - Spaced: L L M, l l m
      - Dotted: L.L.M
      - Hyphenated: L-L-M
      - Comma-separated: L,L,M
      - Mixed: L. L. M.
    The pattern uses word boundaries so it never matches inside words.
    """
    letters = list(shortcut.upper())
    if not letters:
        return re.compile(r"(?!)")  # never-match

    # Between each letter allow optional separator: space, dot, hyphen, comma
    SEP = r"[\s.\-,]*"
    # Pattern for each letter: case-insensitive letter + optional trailing dot
    letter_pats = [re.escape(c) + r"\.?" for c in letters]
    inner = SEP.join(letter_pats)

    # Use word boundaries — but since the shortcut may end with dot we use lookahead
    pattern = r"(?<![A-Za-z0-9])" + inner + r"(?![A-Za-z0-9])"
    return re.compile(pattern, re.IGNORECASE)


def expand_shortcuts(
    text: str,
    shortcuts: List[Dict[str, str]],
) -> str:
    """
    Replace all abbreviation variants in *text* with their full forms.

    Args:
        text: The transcript text to expand.
        shortcuts: List of dicts with keys "shortcut" and "full_form".

    Returns:
        Expanded text.
    """
    if not text or not shortcuts:
        return text

    for entry in shortcuts:
        shortcut = entry.get("shortcut", "").strip()
        full_form = entry.get("full_form", "").strip()
        if not shortcut or not full_form:
            continue
        try:
            pat = _build_pattern(shortcut)
            text = pat.sub(full_form, text)
        except Exception as e:
            logger.warning(f"[Normalizer] Pattern failed for '{shortcut}': {e}")

    return text


def expand_transcript_segments(
    segments: List[Dict],
    shortcuts: List[Dict[str, str]],
) -> List[Dict]:
    """
    Apply expand_shortcuts to the text field of each transcript segment.
    Returns a new list (original segments are not mutated).
    """
    result = []
    for seg in segments:
        new_seg = dict(seg)
        new_seg["text"] = expand_shortcuts(seg.get("text", ""), shortcuts)
        # Also expand word-level text if present
        if "words" in seg and seg["words"]:
            new_words = []
            for w in seg["words"]:
                new_w = dict(w)
                new_w["word"] = expand_shortcuts(w.get("word", ""), shortcuts)
                new_words.append(new_w)
            new_seg["words"] = new_words
        result.append(new_seg)
    return result
