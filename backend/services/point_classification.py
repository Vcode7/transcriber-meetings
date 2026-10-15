"""
Discussion-point type and action flag (Stage 2).

Every Stage 2 point records what kind of discussion it is (``type``) and whether it states a
task someone must do after the meeting (``include_action_point``). The enhancement LLM proposes
both; when it does not (custom prompt, trained variant, failed window) they are taken from the
source points, then from simple text cues. Stage 2 never extracts the action itself: action
points are generated later from the flagged points only (services/agenda_action_points.py), and
the Short ROM condenses each point according to its type (services/rom_condense.py).
"""
import re
from typing import Any, Dict, Iterable, List, Optional

# Order = precedence when a point mixes several kinds (the outcome that matters most for the record).
DISCUSSION_TYPES = ("decision", "debate", "question_answer", "issue", "update", "discussion")
_PRECEDENCE = {t: rank for rank, t in enumerate(DISCUSSION_TYPES)}
# Sources whose type/flag came from a model or a person (not from the text cues below).
TRUSTED_SOURCES = ("llm", "user")

_TYPE_ALIASES = {
    "decision": "decision", "decisions": "decision", "decided": "decision", "agreement": "decision",
    "agreed": "decision", "approval": "decision", "resolution": "decision",
    "debate": "debate", "disagreement": "debate", "argument": "debate", "deliberation": "debate",
    "dispute": "debate",
    "question_answer": "question_answer", "question": "question_answer", "questions": "question_answer",
    "q&a": "question_answer", "qa": "question_answer", "q_a": "question_answer",
    "question_and_answer": "question_answer", "clarification": "question_answer", "open_question": "question_answer",
    "issue": "issue", "issues": "issue", "risk": "issue", "risks": "issue", "concern": "issue", "blocker": "issue",
    "problem": "issue", "delay": "issue", "dependency": "issue",
    "update": "update", "status": "update", "status_update": "update", "progress": "update",
    "progress_update": "update", "information": "update", "info": "update", "report": "update",
    "announcement": "update",
    "discussion": "discussion", "general_discussion": "discussion", "general": "discussion", "other": "discussion",
}

DECISION_RE = re.compile(
    r"\b(decided|decision|agreed|approved|approval\s+(?:was\s+)?(?:given|granted)|finali[sz]ed|confirmed|"
    r"resolved|concluded|rejected|signed\s+off|sign-off)\b", re.IGNORECASE)
ACTION_RE = re.compile(
    r"\b(will\s+(?:send|share|prepare|submit|update|review|complete|arrange|provide|check|coordinate|schedule|draft|"
    r"finali[sz]e|follow)|to\s+be\s+(?:done|completed|shared|sent|prepared|submitted|reviewed)|assigned|"
    r"responsible\s+for|needs?\s+to|has\s+to|have\s+to|action\s+item|was\s+asked\s+to|committed\s+to)\b", re.IGNORECASE)
RISK_RE = re.compile(
    r"\b(risks?|concerns?|issues?|delay(?:s|ed)?|blockers?|blocked|dependenc(?:y|ies)|bottlenecks?|shortages?|overruns?|"
    r"escalat\w*|problems?|challenges?|threats?|non-?compliance)\b", re.IGNORECASE)
QUESTION_RE = re.compile(
    r"\?|\b(open\s+question|unclear|to\s+be\s+clarified|clarification|unresolved|pending\s+decision|whether)\b",
    re.IGNORECASE)
_DEBATE_RE = re.compile(
    r"\b(disagreed|disagreement|argued|objected|objection|countered|pushed\s+back|debated|opposed|"
    r"on\s+the\s+other\s+hand|differing\s+views?|in\s+contrast)\b", re.IGNORECASE)
_QA_RE = re.compile(
    r"\?|\b(asked|enquired|inquired|queried|questioned|sought\s+clarification|clarified|responded|replied)\b",
    re.IGNORECASE)
_UPDATE_RE = re.compile(
    r"\b(status|progress|updated?|reported|on\s+track|completed|milestone|percent)\b|%", re.IGNORECASE)

_TRUE_STRINGS = {"true", "yes", "y", "1"}
_FALSE_STRINGS = {"false", "no", "n", "0"}


def point_text(point: Dict[str, Any]) -> str:
    return str(point.get("polished_text") or point.get("text") or point.get("discussion_point") or "").strip()


def normalize_discussion_type(value: Any) -> Optional[str]:
    if not value:
        return None
    key = re.sub(r"\s*&\s*", "&", str(value).strip().lower())
    key = re.sub(r"[\s\-/]+", "_", key)
    return _TYPE_ALIASES.get(key)


def normalize_action_flag(value: Any) -> Optional[bool]:
    """True/False for a stated flag, None when the flag is missing or unreadable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in _TRUE_STRINGS:
            return True
        if low in _FALSE_STRINGS:
            return False
    return None


def dominant_type(types: Iterable[Any]) -> Optional[str]:
    known = [t for t in (normalize_discussion_type(v) for v in types) if t]
    return min(known, key=_PRECEDENCE.get) if known else None


def heuristic_discussion_type(text: str) -> str:
    if DECISION_RE.search(text):
        return "decision"
    if _DEBATE_RE.search(text):
        return "debate"
    if _QA_RE.search(text):
        return "question_answer"
    if RISK_RE.search(text):
        return "issue"
    if _UPDATE_RE.search(text):
        return "update"
    return "discussion"


def has_action_cue(text: str) -> bool:
    return bool(ACTION_RE.search(text))


def _has_action_items(point: Dict[str, Any]) -> bool:
    items = point.get("action_items")
    return bool(items) and isinstance(items, list)


def _has_validated_owner(point: Dict[str, Any]) -> bool:
    """A transcript-validated action owner: owner evidence (embedded Stage 1 mode, where points
    carry no action_items) or an owner whose validation confidence is high or medium."""
    if point.get("owner_evidence"):
        return True
    return bool(point.get("action_owner")) and point.get("owner_confidence") in ("high", "medium")


def classify_stage2_point(ep: Dict[str, Any], sources: Iterable[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """Set ``type``/``type_source`` and ``include_action_point``/``action_flag_source`` on ``ep``.

    ``ep`` holds the LLM's proposal (if any); ``sources`` are the points it was written from
    (Stage 1 points, or Stage 2 points when merging). Type: the LLM's, else the strongest type
    of the sources, else text cues. Flag: the LLM's, else any source's, else text cues; Stage 1
    action items or a validated action owner (transcript-validated) always set it.
    """
    sources = [s for s in sources or [] if isinstance(s, dict)]
    text = point_text(ep)

    llm_type = normalize_discussion_type(ep.get("type"))
    if llm_type:
        ep["type"], ep["type_source"] = llm_type, ep.get("type_source") or "llm"
    else:
        typed = [(normalize_discussion_type(s.get("type")), s) for s in sources]
        typed = [(t, s) for t, s in typed if t]
        if typed:
            best_type, best_source = min(typed, key=lambda ts: _PRECEDENCE[ts[0]])
            ep["type"], ep["type_source"] = best_type, best_source.get("type_source") or "heuristic"
        else:
            ep["type"], ep["type_source"] = heuristic_discussion_type(text), "heuristic"

    flag = normalize_action_flag(ep.get("include_action_point"))
    flag_source = ep.get("action_flag_source") or "llm"
    if flag is None:
        source_flags = [normalize_action_flag(s.get("include_action_point")) for s in sources]
        if any(f is not None for f in source_flags):
            flag = any(bool(f) for f in source_flags)
            flagged = [s for s, f in zip(sources, source_flags) if f]
            flag_source = (flagged[0].get("action_flag_source") if flagged else None) or "heuristic"
        else:
            flag, flag_source = has_action_cue(text), "heuristic"
    # Stage 1 evidence of an action (validated action items or a validated owner) always sets the
    # flag: one wrong "no action" from the enhancement LLM must not lose the action point.
    # A flag a person switched off stays off.
    if not flag and flag_source != "user" and any(
        _has_action_items(p) or _has_validated_owner(p)
        for p in [ep] + sources if p.get("action_flag_source") != "user"
    ):
        flag, flag_source = True, "stage1"
    ep["include_action_point"] = bool(flag)
    ep["action_flag_source"] = flag_source
    return ep


# Stage 3 discussion analysis uses its own taxonomy (services/agenda_analysis.py).
_ANALYSIS_TYPE = {
    "decision": "decision", "question_answer": "question", "issue": "risk",
    "debate": "discussion", "update": "discussion", "discussion": "discussion",
}


def analysis_type_from_stage2(point: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Stage 3 classification derived from a model- or user-set Stage 2 type, else None."""
    stage2_type = normalize_discussion_type(point.get("type"))
    if not stage2_type or point.get("type_source") not in TRUSTED_SOURCES:
        return None
    mapped = _ANALYSIS_TYPE[stage2_type]
    flagged = bool(normalize_action_flag(point.get("include_action_point")))
    primary = "decision" if mapped == "decision" else ("action_item" if flagged else mapped)
    types: List[str] = list(dict.fromkeys([primary, mapped] + (["action_item"] if flagged else [])))
    if len(types) > 1:
        types = [t for t in types if t != "discussion"]
    return {"primary_type": primary, "types": types, "evidence": "", "source": "stage2"}
