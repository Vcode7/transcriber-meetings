"""
Deterministic validation of ROM action-item owners against speaker turns.

The LLM proposes an owner together with the basis for it (self commitment,
direct assignment, ...) and the transcript turns it relied on. These helpers
check that proposal against the turns themselves, canonicalize names to the
meeting's speaker roster, correct clear contradictions (e.g. the requester
named as owner of their own request), and attach a confidence level and an
evidence quote to every action item. No LLM calls happen here.
"""
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

VALID_BASES = ("self_commitment", "direct_assignment", "accepted_request", "named_responsibility", "unclear")
CONFIDENCE_LEVELS = ("low", "medium", "high")
_CONF_RANK = {level: rank for rank, level in enumerate(CONFIDENCE_LEVELS)}

VAGUE_OWNERS = {
    "my team", "our team", "you", "they", "everyone", "we", "someone", "somebody", "anybody", "team", "us",
    "i", "he", "she", "it", "null", "none", "n/a", "unknown", "undefined", "unassigned", "",
    "for information", "from document", "all", "all participants", "participants",
}

FUZZY_MATCH_THRESHOLD = 0.88
DUPLICATE_TASK_THRESHOLD = 0.85
EVIDENCE_QUOTE_MAX_CHARS = 200

_FIRST_PERSON_COMMIT_RE = re.compile(
    r"\b(?:i\s*['’]?\s*ll\b(?<!\bill)|i will\b(?!\s+not)|i shall\b|i['’]m going to\b|i am going to\b|"
    r"i['’]m gonna\b|let me\b(?!\s+know)|i can\b(?!['’]?t|not)|i could\b(?!n['’]?t)|i['’]d be happy to\b|"
    r"leave it (?:with|to) me\b|i['’]ll take\b|i will take\b)",
    re.IGNORECASE,
)
_STRONG_ACCEPT_RE = re.compile(
    r"\b(will do|on it|consider it done|no problem|i['’]ll do (?:it|that)|i will do (?:it|that)|absolutely|definitely)\b",
    re.IGNORECASE,
)
_WEAK_ACCEPT_RE = re.compile(
    r"\b(sure|okay|ok|yes|yeah|yep|got it|alright|all right|fine|agreed|of course|noted)\b",
    re.IGNORECASE,
)
_REQUEST_RE = re.compile(
    r"\?|\b(can you|could you|would you|will you|please|you need to|you have to|you should|you['’]ll|you will|"
    r"make sure you|i want you to|i['’]d like you to|take care of|assign(?:ed|ing)? (?:to|this|that))\b",
    re.IGNORECASE,
)
_TURN_ID_RE = re.compile(r"[Tt]?\s*(\d+)")
_NOTE_RE = re.compile(r"\s*[\(\[][^)\]]*[\)\]]\s*")
_SPLIT_RE = re.compile(r"\s*(?:,|;|/|&|\band\b)\s*", re.IGNORECASE)
_STOP_TOKENS = {
    "the", "and", "for", "with", "that", "this", "will", "from", "into", "onto", "before", "after", "complete",
    "completed", "task", "assigned", "should", "must", "need", "needs", "also", "then", "their", "them",
}


# ── Names ─────────────────────────────────────────────────────────────────────

def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _clean_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = _NOTE_RE.sub(" ", str(value)).strip().strip("\"'`").strip()
    s = re.sub(r"['’]s$", "", s).strip().rstrip(".,:;").strip()
    if s.lower() in VAGUE_OWNERS:
        return None
    return s or None


def is_person_label(label: Any) -> bool:
    return _clean_name(label) is not None


def canonicalize_owner(name: Any, roster: Optional[List[Dict[str, Any]]]) -> Tuple[Optional[str], bool]:
    """Map a free-text name to a roster label.

    Returns (name, resolved). ``resolved`` is True only when the name maps to
    exactly one roster participant; otherwise the cleaned name is returned as-is
    (an external person, or an ambiguous alias).
    """
    cleaned = _clean_name(name)
    if cleaned is None:
        return None, False
    if not roster:
        return cleaned, False

    key = _norm_key(cleaned)
    matches = {entry["label"] for entry in roster for alias in entry["aliases"] if _norm_key(alias) == key}
    if len(matches) == 1:
        return matches.pop(), True
    if len(matches) > 1:
        return cleaned, False

    # Fuzzy matching only for digit-free names: "Speaker 1" vs "Speaker 2" must never match.
    if not re.search(r"\d", cleaned):
        scored = []
        for entry in roster:
            best = max(
                (SequenceMatcher(None, cleaned.lower(), alias.lower()).ratio()
                 for alias in entry["aliases"] if not re.search(r"\d", alias)),
                default=0.0,
            )
            scored.append((best, entry["label"]))
        scored.sort(reverse=True)
        if scored and scored[0][0] >= FUZZY_MATCH_THRESHOLD and (len(scored) == 1 or scored[1][0] < scored[0][0]):
            return scored[0][1], True
    return cleaned, False


def split_owner_names(raw: Any) -> List[str]:
    if raw is None:
        return []
    parts: Iterable[Any] = raw if isinstance(raw, (list, tuple)) else _SPLIT_RE.split(str(raw))
    return [p for p in (str(x).strip() for x in parts if x is not None) if p]


def canonicalize_owner_list(raw: Any, roster: Optional[List[Dict[str, Any]]]) -> List[Tuple[str, bool]]:
    result: List[Tuple[str, bool]] = []
    seen = set()
    for part in split_owner_names(raw):
        name, resolved = canonicalize_owner(part, roster)
        if name and name.lower() not in seen:
            seen.add(name.lower())
            result.append((name, resolved))
    return result


def _participants_mentioned(text: str, roster: Optional[List[Dict[str, Any]]]) -> List[str]:
    found = []
    for entry in roster or []:
        for alias in entry["aliases"]:
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text or "", re.IGNORECASE):
                found.append(entry["label"])
                break
    return found


# ── Turn evidence ─────────────────────────────────────────────────────────────

def parse_turn_ids(raw: Any) -> List[str]:
    if raw is None:
        return []
    tokens = raw if isinstance(raw, (list, tuple)) else [raw]
    ids: List[str] = []
    for token in tokens:
        for num in _TURN_ID_RE.findall(str(token)):
            tid = f"T{int(num)}"
            if tid not in ids:
                ids.append(tid)
    return ids


def _raw_evidence_ids(item: Dict[str, Any]) -> Any:
    if item.get("evidence_turn_ids") is not None:
        return item.get("evidence_turn_ids")
    evidence = item.get("evidence")
    if isinstance(evidence, dict):
        return evidence.get("turn_ids")
    return None


def _build_evidence(turns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not turns:
        return None
    quote = " / ".join(f"{t['speaker']}: {t['text']}" for t in turns)
    if len(quote) > EVIDENCE_QUOTE_MAX_CHARS:
        quote = quote[:EVIDENCE_QUOTE_MAX_CHARS - 1].rstrip() + "…"
    return {
        "turn_ids": [t["turn_id"] for t in turns],
        "start": round(min(t["start"] for t in turns), 2),
        "end": round(max(t["end"] for t in turns), 2),
        "quote": quote,
    }


def _normalize_confidence(value: Any, default: str = "medium") -> str:
    s = str(value or "").strip().lower()
    return s if s in _CONF_RANK else default


def min_confidence(levels: Iterable[Optional[str]]) -> Optional[str]:
    ranked = [lvl for lvl in levels if lvl in _CONF_RANK]
    return min(ranked, key=_CONF_RANK.get) if ranked else None


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(v for v in values if v))


# ── Validation ────────────────────────────────────────────────────────────────

def validate_action(
    item: Dict[str, Any],
    turns_by_id: Dict[str, Dict[str, Any]],
    roster: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Validate one action item's owner against its evidence turns.

    Input keys used: task, assigner, assignee, assignee_basis, owner_confidence,
    evidence_turn_ids (or evidence.turn_ids). Returns a new dict with canonical
    assigner/assignee plus assignee_basis, owner_confidence, owner_resolved and
    evidence ({turn_ids, start, end, quote} or None). Corrections are applied
    only on clear contradictions with the turns; otherwise the LLM's owner is
    kept and the confidence reflects how well the turns support it.
    """
    out = dict(item)
    out.pop("evidence_turn_ids", None)

    basis = str(item.get("assignee_basis") or item.get("basis") or "").strip().lower()
    if basis not in VALID_BASES:
        basis = "unclear"

    assignees = canonicalize_owner_list(item.get("assignee"), roster)
    names = [name for name, _ in assignees]
    resolved_by_name = {name: resolved for name, resolved in assignees}
    roster_labels = {entry["label"] for entry in roster or []}
    assigner, _ = canonicalize_owner(item.get("assigner"), roster)

    ev_ids = [tid for tid in parse_turn_ids(_raw_evidence_ids(item)) if tid in turns_by_id]
    ev_turns = sorted((turns_by_id[tid] for tid in ev_ids), key=lambda t: t["start"])
    deciding_turns: List[Dict[str, Any]] = list(ev_turns)

    confirmed = False
    corrected_from: Optional[str] = None
    contradiction = False

    def _correct(new_names: List[str], reason: str) -> None:
        nonlocal names, corrected_from
        if new_names != names:
            corrected_from = ", ".join(names) or None
            logger.info(
                f"[OwnerValidator] owner_corrected: '{corrected_from}' -> '{', '.join(new_names)}' "
                f"({reason}; turns={ev_ids}) task={str(item.get('task') or '')[:80]!r}"
            )
            for n in new_names:
                resolved_by_name.setdefault(n, n in roster_labels)
            names = new_names

    if ev_turns and basis == "self_commitment":
        commit_turns = [t for t in ev_turns if _FIRST_PERSON_COMMIT_RE.search(t["text"]) and is_person_label(t["speaker"])]
        commit_speakers = _unique(t["speaker"] for t in commit_turns)
        if commit_turns:
            deciding_turns = commit_turns
            kept = [n for n in names if n in commit_speakers]
            if kept:
                names, confirmed = kept, True
            elif len(commit_speakers) == 1:
                _correct(commit_speakers, "self_commitment spoken by a different speaker")
            else:
                contradiction = True

    elif ev_turns and basis in ("direct_assignment", "accepted_request"):
        req_idx = next((i for i, t in enumerate(ev_turns) if _REQUEST_RE.search(t["text"])), None)
        if req_idx is not None:
            request_turn = ev_turns[req_idx]
            requester = request_turn["speaker"] if is_person_label(request_turn["speaker"]) else None
            if requester and assigner != requester:
                if assigner:
                    logger.info(f"[OwnerValidator] assigner_corrected: '{assigner}' -> '{requester}' (turn {request_turn['turn_id']})")
                assigner = requester
            addressed = [p for p in _participants_mentioned(request_turn["text"], roster) if p != requester]
            later = [t for t in ev_turns[req_idx + 1:] if t["speaker"] != requester and is_person_label(t["speaker"])]
            strong_turns = [t for t in later if _FIRST_PERSON_COMMIT_RE.search(t["text"]) or _STRONG_ACCEPT_RE.search(t["text"])]
            weak_turns = [t for t in later if t not in strong_turns and _WEAK_ACCEPT_RE.search(t["text"])]
            strong_acceptors = _unique(t["speaker"] for t in strong_turns)
            acceptors = _unique(t["speaker"] for t in strong_turns + weak_turns)
            deciding_turns = [request_turn] + strong_turns + weak_turns

            kept = [n for n in names if n in addressed or n in acceptors]
            if kept:
                names, confirmed = kept, True
            else:
                # Correct only clear contradictions: owner missing, owner == requester,
                # or a roster participant who was neither addressed nor accepted.
                owner_contradicts = (
                    not names
                    or (requester is not None and requester in names)
                    or all(resolved_by_name.get(n) for n in names)
                )
                replacement = addressed if len(addressed) == 1 else (strong_acceptors if len(strong_acceptors) == 1 else [])
                if owner_contradicts and replacement:
                    _correct(replacement, "owner not addressed by / accepting the request")
                elif requester is not None and requester in names:
                    contradiction = True

    elif ev_turns and basis == "named_responsibility":
        text = " ".join(t["text"] for t in ev_turns)
        mentioned = _participants_mentioned(text, roster)
        if names and all(
            n in mentioned or re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text, re.IGNORECASE)
            for n in names
        ):
            confirmed = True

    if assigner and names == [assigner] and basis in ("direct_assignment", "accepted_request"):
        contradiction = True

    # ── Confidence ──
    # Unverified or corrected owners are never "high"; missing evidence, a
    # contradiction or an uncertain deciding turn makes the owner "low".
    confidence = _normalize_confidence(item.get("owner_confidence"))
    if confirmed:
        confidence = "high"
    elif names:
        confidence = min_confidence([confidence, "medium"])
    if names and (not ev_turns or contradiction or basis == "unclear"):
        confidence = "low"
    if any(t.get("uncertain") or t.get("is_overlap") or t.get("overlap_speakers") for t in deciding_turns):
        confidence = "low"

    out["assigner"] = assigner
    out["assignee"] = ", ".join(names) if names else None
    out["assignee_basis"] = basis
    out["owner_confidence"] = confidence
    out["owner_resolved"] = bool(names) and all(resolved_by_name.get(n, n in roster_labels) for n in names)
    # Keep the human name the LLM used when it was replaced by a roster label
    # (a hint for renaming "Speaker N"); generic labels carry no information.
    if corrected_from and not resolved_by_name.get(corrected_from, True) and not re.search(r"\d", corrected_from):
        out["assignee_mentioned_as"] = corrected_from
    out["evidence"] = _build_evidence(ev_turns)
    return out


def evidence_outside_focus(action: Dict[str, Any], focus_turn_ids: Iterable[str]) -> bool:
    """True when an action cites evidence turns and none of them are in the focus window."""
    evidence = action.get("evidence") or {}
    ids = evidence.get("turn_ids") or []
    focus = set(focus_turn_ids)
    return bool(ids) and not any(tid in focus for tid in ids)


def validate_point_owners(
    point: Dict[str, Any],
    turns_by_id: Dict[str, Dict[str, Any]],
    roster: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Embedded-mode counterpart of validate_action for a discussion point.

    Validates each ``owner_evidence`` entry ({owner, basis, turn_ids}) and
    rewrites ``action_owner`` as the comma-joined validated owners, with a
    point-level ``owner_confidence``.
    """
    entries = point.get("owner_evidence") or []
    if isinstance(entries, dict):
        entries = [entries]

    owners: List[str] = []
    confidences: List[str] = []
    cleaned: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        v = validate_action(
            {
                "task": point.get("discussion_point", ""),
                "assignee": entry.get("owner") or entry.get("assignee"),
                "assigner": entry.get("assigner"),
                "assignee_basis": entry.get("basis") or entry.get("assignee_basis"),
                "evidence_turn_ids": entry.get("turn_ids") or entry.get("evidence_turn_ids"),
                "owner_confidence": entry.get("confidence") or entry.get("owner_confidence"),
            },
            turns_by_id,
            roster,
        )
        if not v["assignee"]:
            continue
        owners.extend(split_owner_names(v["assignee"]))
        confidences.append(v["owner_confidence"])
        cleaned.append({
            "owner": v["assignee"],
            "basis": v["assignee_basis"],
            "turn_ids": (v.get("evidence") or {}).get("turn_ids", []),
            "confidence": v["owner_confidence"],
        })

    if owners:
        point["action_owner"] = ", ".join(_unique(owners))
        point["owner_confidence"] = min_confidence(confidences)
    else:
        names = [n for n, _ in canonicalize_owner_list(point.get("action_owner"), roster)]
        point["action_owner"] = ", ".join(names) if names else None
        if names:
            point["owner_confidence"] = "low"
    point["owner_evidence"] = cleaned
    return point


def verify_owner_against_candidates(
    owner_raw: Any,
    owner_candidates: Iterable[str],
    transcript_excerpt: str,
    roster: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[str], Optional[str]]:
    """Post-check an owner extracted from polished prose (Stage 3).

    Owners must be one of the point's validated ``owner_candidates`` or be
    supported by the point's transcript excerpt. Returns (owner, confidence);
    confidence is None when there was nothing to verify against.
    """
    names = canonicalize_owner_list(owner_raw, roster)
    if not names:
        return None, None
    candidates = {c.lower() for c in owner_candidates or [] if c}
    excerpt = transcript_excerpt or ""
    if not candidates and not excerpt:
        return ", ".join(n for n, _ in names), None

    kept, all_candidates = [], True
    for name, _ in names:
        if name.lower() in candidates:
            kept.append(name)
            continue
        all_candidates = False
        spoke = re.search(r"\]\s*" + re.escape(name) + r":", excerpt)
        mentioned = re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", excerpt, re.IGNORECASE)
        if spoke or mentioned:
            kept.append(name)
    if not kept:
        logger.info(f"[OwnerValidator] owner_rejected: '{owner_raw}' not in candidates {sorted(candidates)} nor in excerpt")
        return None, "low"
    return ", ".join(kept), "high" if all_candidates else "medium"


# ── Duplicates ────────────────────────────────────────────────────────────────

def _task_tokens(task: Any) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", str(task or "").lower())) - _STOP_TOKENS


def task_similarity(a: Any, b: Any) -> float:
    """Overlap coefficient of the content tokens of two task descriptions (0..1)."""
    ta, tb = _task_tokens(a), _task_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def actions_are_duplicates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    owner_a = str(a.get("assignee") or "").lower()
    owner_b = str(b.get("assignee") or "").lower()
    if owner_a and owner_b and owner_a != owner_b:
        return False
    overlap = task_similarity(a.get("task"), b.get("task"))
    if overlap == 0.0:
        return False
    ids_a = set((a.get("evidence") or {}).get("turn_ids") or [])
    ids_b = set((b.get("evidence") or {}).get("turn_ids") or [])
    threshold = 0.6 if ids_a & ids_b else DUPLICATE_TASK_THRESHOLD
    return overlap >= threshold


def merge_duplicate_action(keep: Dict[str, Any], dup: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("assignee", "assigner", "deadline"):
        if not keep.get(key) and dup.get(key):
            keep[key] = dup[key]
    if len(str(dup.get("task") or "")) > len(str(keep.get("task") or "")):
        keep["task"] = dup["task"]
    if _CONF_RANK.get(dup.get("owner_confidence"), -1) > _CONF_RANK.get(keep.get("owner_confidence"), -1):
        keep["owner_confidence"] = dup["owner_confidence"]
    if not keep.get("evidence") and dup.get("evidence"):
        keep["evidence"] = dup["evidence"]
    return keep


def dedupe_point_actions(points: List[Dict[str, Any]]) -> int:
    """Remove duplicate action items across points (in place). Returns number removed."""
    kept: List[Dict[str, Any]] = []
    removed = 0
    for point in points:
        unique_items = []
        for act in point.get("action_items") or []:
            if not isinstance(act, dict):
                unique_items.append(act)
                continue
            match = next((k for k in kept if actions_are_duplicates(k, act)), None)
            if match is not None:
                merge_duplicate_action(match, act)
                removed += 1
                continue
            kept.append(act)
            unique_items.append(act)
        point["action_items"] = unique_items
    return removed
