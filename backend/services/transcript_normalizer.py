"""
Transcript Normalizer Service

Provides two complementary normalization modes:

1. **Shortcut expansion** (expand_shortcuts / expand_transcript_segments)
   Expands abbreviations from the Shortcut Dictionary in transcript text.
   Handles all common variant forms of an abbreviation, e.g. for "LLM":
     - LLM, llm, Llm
     - L L M, L.L.M, L-L-M, L,L,M, l l m
     - LL M, L LM, etc.
   Uses word-boundary assertions to avoid replacing inside larger words.

2. **Alignment normalization** (normalize_text_for_alignment / normalize_segments_for_alignment)
   A lightweight, deterministic text normalization pass designed to run on raw
   Whisper transcription segments BEFORE whisperx.align() is called.

   WhisperX's CTC-based forced alignment works best when the input text closely
   matches what the acoustic model was trained on.  Numbers as digits, uncommon
   punctuation, and compressed acronyms all hurt alignment quality.

   Steps applied (in order, all pure Python — no new dependencies):
     a. Contraction expansion  — "don't" → "do not", "we're" → "we are"
     b. Number expansion       — "3" → "three", "100" → "one hundred"
     c. Acronym spacing        — "AI" → "A I", "NLP" → "N L P"
     d. Punctuation cleanup    — strip chars that wav2vec2 can't align to
     e. Whitespace collapse    — normalize multiple spaces / line breaks
"""
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Shortcut / abbreviation expansion
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Alignment-focused normalization
# ─────────────────────────────────────────────────────────────────────────────

# ── 2a. Contraction table ─────────────────────────────────────────────────────
# fmt: off
_CONTRACTIONS: Dict[str, str] = {
    "don't": "do not",        "doesn't": "does not",
    "didn't": "did not",      "won't": "will not",
    "wouldn't": "would not",  "couldn't": "could not",
    "shouldn't": "should not","can't": "cannot",
    "isn't": "is not",        "aren't": "are not",
    "wasn't": "was not",      "weren't": "were not",
    "haven't": "have not",    "hasn't": "has not",
    "hadn't": "had not",      "it's": "it is",
    "that's": "that is",      "there's": "there is",
    "here's": "here is",      "what's": "what is",
    "who's": "who is",        "he's": "he is",
    "she's": "she is",        "we're": "we are",
    "they're": "they are",    "you're": "you are",
    "i'm": "i am",            "we've": "we have",
    "they've": "they have",   "you've": "you have",
    "i've": "i have",         "he'd": "he would",
    "she'd": "she would",     "we'd": "we would",
    "they'd": "they would",   "you'd": "you would",
    "i'd": "i would",         "he'll": "he will",
    "she'll": "she will",     "we'll": "we will",
    "they'll": "they will",   "you'll": "you will",
    "i'll": "i will",         "let's": "let us",
    "that'll": "that will",   "there'll": "there will",
    "it'll": "it will",       "who'll": "who will",
    "what'll": "what will",   "there'd": "there would",
    "that'd": "that would",   "it'd": "it would",
    "'cause": "because",      "gotta": "got to",
    "gonna": "going to",      "wanna": "want to",
    "kinda": "kind of",       "sorta": "sort of",
}
# fmt: on

# Build case-insensitive contraction regex once at module load.
# Sorted longest-first to avoid partial matches.
_CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _expand_contractions(text: str) -> str:
    """Expand English contractions to their full forms."""
    def _replace(m: re.Match) -> str:
        token = m.group(0)
        expanded = _CONTRACTIONS.get(token.lower(), token)
        # Preserve initial capitalisation
        if token[0].isupper():
            return expanded[0].upper() + expanded[1:]
        return expanded

    return _CONTRACTION_PATTERN.sub(_replace, text)


# ── 2b. Number expansion ──────────────────────────────────────────────────────
_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
]


def _int_to_words(n: int) -> str:
    """Convert a non-negative integer ≤ 999,999 to English words."""
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n == 0:
        return "zero"
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (" " + _ONES[ones] if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + (" " + _int_to_words(rest) if rest else "")
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        return _int_to_words(thousands) + " thousand" + (" " + _int_to_words(rest) if rest else "")
    # Larger numbers: just return as string (alignment will handle digits)
    return str(n)


def _expand_numbers(text: str) -> str:
    """
    Replace standalone integers (1–999,999) with English words.
    Leaves decimals, years (e.g. 2024), phone numbers, and large numbers alone
    to avoid producing unnatural text that hurts alignment more than it helps.
    """
    def _replace_num(m: re.Match) -> str:
        raw = m.group(0)
        try:
            val = int(raw)
            # Skip 4-digit years (1900–2099) — spoken as digits in context
            if 1900 <= val <= 2099:
                return raw
            # Skip very large numbers — too risky to expand
            if val >= 1_000_000:
                return raw
            return _int_to_words(val)
        except ValueError:
            return raw

    # Match standalone integers not preceded/followed by digits, dots, or slashes
    return re.sub(r"(?<![.\d/])\b([0-9]{1,6})\b(?![.\d/])", _replace_num, text)


# ── 2c. Acronym spacing ───────────────────────────────────────────────────────
# An acronym is 2–6 uppercase letters not followed by a lowercase letter.
# e.g. "AI", "NLP", "CEO", "HTTP" — but NOT "Mr", "Dr", "I" (single letter)
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")


def _space_acronyms(text: str) -> str:
    """Insert spaces between letters of uppercase acronyms: AI → A I"""
    def _space(m: re.Match) -> str:
        return " ".join(m.group(1))

    return _ACRONYM_RE.sub(_space, text)


# ── 2d. Punctuation cleanup ───────────────────────────────────────────────────
# Characters wav2vec2 cannot align to — strip them (don't replace with space
# to avoid double-spacing; whitespace collapse in step 2e handles that).
_ALIGNMENT_HOSTILE_CHARS = re.compile(r"[\"#$%&()*+/<=>@\[\\\]^_{|}~`]")
# Repeated punctuation: "..." → " " , "---" → " "
_REPEATED_PUNCT = re.compile(r"[.\-!?,:;]{3,}")


def _clean_punctuation(text: str) -> str:
    """Strip punctuation that wav2vec2 CTC alignment cannot handle."""
    text = _ALIGNMENT_HOSTILE_CHARS.sub("", text)
    text = _REPEATED_PUNCT.sub(" ", text)
    return text


# ── 2e. Whitespace collapse ───────────────────────────────────────────────────
def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ── Vocabulary/domain term normalizer ────────────────────────────────────────
def _ensure_vocab_terms(text: str, vocab_terms: Optional[List[str]]) -> str:
    """
    Ensure domain-specific vocabulary terms appear in their canonical forms.
    For each term in the vocab list, if the text contains a lowercase/mixed-case
    version, replace with the canonical form.  This is intentionally conservative —
    only replaces whole-word matches.
    """
    if not vocab_terms:
        return text
    for term in vocab_terms:
        if not term or len(term) < 2:
            continue
        try:
            pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            text = pat.sub(term, text)
        except Exception:
            pass
    return text


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_text_for_alignment(
    text: str,
    vocab_terms: Optional[List[str]] = None,
) -> str:
    """
    Normalize a transcript text string for WhisperX forced alignment.

    Applies, in order:
      1. Contraction expansion   ("don't" → "do not")
      2. Number expansion        ("3" → "three")
      3. Acronym spacing         ("AI" → "A I")
      4. Punctuation cleanup     (remove alignment-hostile chars)
      5. Vocab term canonicalization (reuse existing vocab system)
      6. Whitespace collapse

    This is deterministic and meaning-preserving.  It does NOT:
      - Translate
      - Correct grammar
      - Alter proper nouns (only acronym spacing)

    Args:
        text: Raw transcript text (from Whisper output).
        vocab_terms: Optional list of domain-specific terms from the
                     existing vocabulary/prompt system.  These are
                     canonicalized (whole-word, case-insensitive match).

    Returns:
        Normalized text string, safe for wav2vec2 CTC alignment.
    """
    if not text:
        return text

    original = text
    steps_applied = []

    # Step 1: Contractions
    t = _expand_contractions(text)
    if t != text:
        steps_applied.append("contractions")
    text = t

    # Step 2: Numbers
    t = _expand_numbers(text)
    if t != text:
        steps_applied.append("numbers")
    text = t

    # Step 3: Acronyms
    t = _space_acronyms(text)
    if t != text:
        steps_applied.append("acronyms")
    text = t

    # Step 4: Punctuation
    t = _clean_punctuation(text)
    if t != text:
        steps_applied.append("punctuation")
    text = t

    # Step 5: Vocabulary terms
    if vocab_terms:
        t = _ensure_vocab_terms(text, vocab_terms)
        if t != text:
            steps_applied.append("vocab_terms")
        text = t

    # Step 6: Whitespace
    text = _collapse_whitespace(text)

    if steps_applied:
        logger.debug(
            f"[Normalizer] Alignment normalization applied: {steps_applied} | "
            f"'{original[:60]}' → '{text[:60]}'"
        )

    return text


def normalize_segments_for_alignment(
    segments: List[Dict],
    vocab_terms: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Apply normalize_text_for_alignment() to the text field of each segment.

    Operates on segment-level text only (not word-level tokens), since
    word-level text is re-generated by the aligner.

    Returns a new list — originals are not mutated.
    """
    if not segments:
        return segments

    result = []
    changed = 0
    for seg in segments:
        new_seg = dict(seg)
        original_text = seg.get("text", "")
        normalized = normalize_text_for_alignment(original_text, vocab_terms)
        new_seg["text"] = normalized
        if normalized != original_text:
            changed += 1
        result.append(new_seg)

    if changed:
        logger.info(
            f"[Normalizer] normalize_segments_for_alignment: "
            f"{changed}/{len(segments)} segments modified"
        )
    else:
        logger.debug("[Normalizer] normalize_segments_for_alignment: no changes needed")

    return result
