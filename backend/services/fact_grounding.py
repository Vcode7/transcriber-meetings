"""
Deterministic checks that generated text only uses facts found in its sources.

"Facts" are the details an LLM must never invent in a meeting record: numbers
(amounts, quantities, days), month and weekday names, and acronyms or codes
(e.g. "SAP-MM", "Q3", "FY26"). A generated sentence is grounded when every fact
token it contains also appears in the source discussion points.
"""
import re
from typing import Any, Iterable, List, Set

# Digit groups joined by "-" or "/" (ordinance numbers, dates, codes) form one number, so
# "2026-0108" and "20260108" are the same fact.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*(?:[-/]\d+)*")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")
_MONTHS = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr", "may": "may", "june": "jun",
    "july": "jul", "august": "aug", "september": "sep", "sept": "sep", "october": "oct", "november": "nov",
    "december": "dec", "jan": "jan", "feb": "feb", "mar": "mar", "apr": "apr", "jun": "jun", "jul": "jul",
    "aug": "aug", "sep": "sep", "oct": "oct", "nov": "nov", "dec": "dec",
}
# "may" and "mar" are also ordinary words; only treat them as months when capitalised.
_AMBIGUOUS_MONTHS = {"may", "mar"}
_WEEKDAYS = {
    "monday": "mon", "tuesday": "tue", "wednesday": "wed", "thursday": "thu", "friday": "fri",
    "saturday": "sat", "sunday": "sun",
}
_CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "will", "from", "into", "onto", "before", "after", "are",
    "was", "were", "has", "have", "had", "been", "being", "its", "their", "them", "they", "his", "her",
    "our", "your", "who", "what", "when", "which", "while", "shall", "should", "would", "could", "can",
    "must", "need", "needs", "also", "then", "than", "there", "here", "such", "per", "via", "all", "any",
    "each", "not", "only", "about", "over", "under", "between", "within", "upon", "both", "other",
}


def _normalize_number(token: str) -> str:
    token = re.sub(r"[-/]", "", token).replace(",", "")
    if "." in token:
        token = token.rstrip("0").rstrip(".") or "0"
    return token


def _is_acronym(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    upper = sum(1 for c in letters if c.isupper())
    has_digit = any(c.isdigit() for c in word)
    if not letters:
        return False
    if upper >= 2 and upper == len(letters):
        return True
    # Mixed letter/digit codes such as "Q3", "FY26", "ISO27001".
    return has_digit and upper >= 1 and letters[0].isupper()


def fact_tokens(text: Any) -> Set[str]:
    """Normalised fact tokens (numbers, months, weekdays, acronyms/codes) in ``text``."""
    s = str(text or "")
    facts: Set[str] = set()
    for num in _NUMBER_RE.findall(s):
        facts.add("#" + _normalize_number(num))
    for word in _WORD_RE.findall(s):
        low = word.lower()
        if low in _MONTHS and (low not in _AMBIGUOUS_MONTHS or word[0].isupper()):
            facts.add("@" + _MONTHS[low])
        elif low in _WEEKDAYS:
            facts.add("@" + _WEEKDAYS[low])
        elif _is_acronym(word):
            facts.add("$" + word.upper())
            # "SAP-MM" also grounds a later mention of "SAP" and "MM".
            for part in re.split(r"[-/]", word):
                if len(part) >= 2 and _is_acronym(part):
                    facts.add("$" + part.upper())
    return facts


def source_facts(texts: Iterable[Any]) -> Set[str]:
    facts: Set[str] = set()
    for text in texts:
        if isinstance(text, (list, tuple, set)):
            facts |= source_facts(text)
        elif text is not None:
            facts |= fact_tokens(text)
    return facts


def unsupported_facts(text: Any, facts: Set[str]) -> List[str]:
    """Fact tokens of ``text`` that do not appear in ``facts`` (readable form)."""
    return sorted(tok[1:] for tok in fact_tokens(text) - facts)


def content_tokens(text: Any) -> Set[str]:
    return set(_CONTENT_TOKEN_RE.findall(str(text or "").lower())) - _STOP_WORDS


def token_overlap(text: Any, source: Any) -> float:
    """Share of ``text``'s content tokens that also occur in ``source`` (0..1)."""
    tokens = content_tokens(text)
    if not tokens:
        return 0.0
    return len(tokens & content_tokens(source)) / len(tokens)


def mentions(text: Any, needle: Any) -> bool:
    needle_s = str(needle or "").strip().rstrip(".")
    return bool(needle_s) and re.search(
        r"(?<!\w)" + re.escape(needle_s) + r"(?!\w)", str(text or ""), re.IGNORECASE
    ) is not None


def split_sentences(text: Any) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return [p.strip() for p in parts if p.strip()]
