import logging
import json
import uuid
import re
import gc
import time
import threading
from typing import List, Dict, Optional, Any, Union
import numpy as np

import math
from services.text_embedding_service import unload_text_embedder

logger = logging.getLogger(__name__)

# ── Stage 1 In-Memory Progress Tracker ─────────────────────────────────────
# Key: f"{user_id}:{recording_id}"
# Thread-safe updates via _stage1_progress_lock
_stage1_progress: Dict[str, dict] = {}
_stage1_progress_lock = threading.Lock()


def _progress_key(user_id: str, recording_id: str) -> str:
    return f"{user_id}:{recording_id}"


def init_stage1_progress(user_id: str, recording_id: str, total_windows: int, concurrency: int) -> None:
    """Initialize Stage 1 progress tracking before extraction starts."""
    key = _progress_key(user_id, recording_id)
    with _stage1_progress_lock:
        _stage1_progress[key] = {
            "status": "processing",
            "windows_completed": 0,
            "windows_total": total_windows,
            "concurrency": concurrency,
            "window_elapsed_times": [],  # elapsed seconds per completed window
            "started_at": time.monotonic(),
        }


def update_stage1_progress(user_id: str, recording_id: str, window_elapsed: float) -> None:
    """Record a completed window's elapsed time and update progress."""
    key = _progress_key(user_id, recording_id)
    with _stage1_progress_lock:
        rec = _stage1_progress.get(key)
        if rec is None:
            return
        rec["window_elapsed_times"].append(window_elapsed)
        rec["windows_completed"] = len(rec["window_elapsed_times"])


def finish_stage1_progress(user_id: str, recording_id: str) -> None:
    """Mark Stage 1 progress as done."""
    key = _progress_key(user_id, recording_id)
    with _stage1_progress_lock:
        rec = _stage1_progress.get(key)
        if rec:
            rec["status"] = "done"


def get_stage1_progress(user_id: str, recording_id: str) -> dict:
    """Return current Stage 1 progress snapshot with computed ETA."""
    key = _progress_key(user_id, recording_id)
    with _stage1_progress_lock:
        rec = _stage1_progress.get(key)
        if rec is None:
            return {"status": "idle", "windows_completed": 0, "windows_total": 0,
                    "concurrency": 1, "eta_seconds": None, "elapsed_seconds": 0.0}

        elapsed = time.monotonic() - rec.get("started_at", time.monotonic())
        completed = rec["windows_completed"]
        total = rec["windows_total"]
        concurrency = rec["concurrency"]
        times = rec["window_elapsed_times"]

        eta = None
        if times and completed < total:
            avg_window_time = sum(times) / len(times)
            remaining_windows = total - completed
            # With parallel processing, divide by concurrency
            eta = (avg_window_time * remaining_windows) / max(1, concurrency)

        return {
            "status": rec["status"],
            "windows_completed": completed,
            "windows_total": total,
            "concurrency": concurrency,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
        }


def clear_stage1_progress(user_id: str, recording_id: str) -> None:
    """Remove Stage 1 progress record from memory."""
    key = _progress_key(user_id, recording_id)
    with _stage1_progress_lock:
        _stage1_progress.pop(key, None)


def clean_calendar_dates(dates_list) -> list[str]:
    """
    Filter and sanitize dates array to ensure ONLY actual calendar dates remain.
    Strips out transcript timestamps, float offsets, audio timelines, and window ranges.
    """
    if not dates_list:
        return []

    if isinstance(dates_list, str):
        dates_list = [dates_list]

    clean_dates = []
    for d in dates_list:
        if isinstance(d, dict):
            val = str(d.get("value") or d.get("date") or "").strip()
        else:
            val = str(d).strip()

        if not val:
            continue

        if re.search(r'^\d+(\.\d+)?\s*[\-â€“â€”]\s*\d+(\.\d+)?$', val):
            continue
        if re.search(r'^\d{1,2}:\d{2}(:\d{2})?\s*[\-â€“â€”]\s*\d{1,2}:\d{2}(:\d{2})?$', val):
            continue
        if re.search(r'^\d{4,}\.\d+$', val) or re.search(r'^\d+\.\d{2,}$', val):
            continue
        if re.search(r'\b(timeline|window|timestamp|seconds?|offset)\b', val, re.IGNORECASE):
            continue

        has_month = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b', val, re.IGNORECASE)
        has_year = re.search(r'\b(19|20)\d{2}\b', val)
        has_date_fmt = re.search(r'\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}\b', val)
        has_day_spec = re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|yesterday|next week|end of month|q[1-4])\b', val, re.IGNORECASE)

        if has_month or has_year or has_date_fmt or has_day_spec or len(val) >= 4:
            if not re.match(r'^[\d\s.:\-â€“â€”]+$', val) or has_date_fmt or has_year:
                clean_dates.append(val)

    return clean_dates

def _extract_chunk_text(res: Any) -> str:
    if isinstance(res, dict):
        txt = (
            res.get("_text") or res.get("text") or res.get("content") or res.get("chunk") or
            res.get("page_content") or res.get("snippet") or res.get("raw_text") or ""
        )
    else:
        txt = (
            getattr(res, "_text", None) or getattr(res, "text", None) or getattr(res, "content", None) or
            getattr(res, "chunk", None) or getattr(res, "page_content", None) or ""
        )
    return str(txt).strip()

def _format_time_hhmm(seconds: float) -> str:
    """Format seconds into HH:MM (or MM:SS if under 1 hour)."""
    total_sec = int(round(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

ACTION_EVIDENCE_KEYS = (
    "assignee_basis", "owner_confidence", "owner_resolved", "evidence", "evidence_turn_ids", "assignee_mentioned_as",
)


def normalize_action_item(item: Any) -> Dict[str, Any]:
    """
    Normalizes an action item to a structured JSON dictionary:
    {"assigner": str|None, "assignee": str|None, "task": str, "deadline": str|None}
    """
    if isinstance(item, dict):
        raw_task = str(item.get("task") or item.get("action_point") or item.get("description") or item.get("item") or "").strip()
        assigner = item.get("assigner")
        assignee = item.get("assignee") or item.get("owner") or item.get("action_owner")
        deadline = item.get("deadline") or item.get("date") or item.get("due")

        VAGUE_ASSIGNEES = {
            "my team", "our team", "you", "they", "everyone", "we", "someone", "somebody", "anybody", "team", "us"
        }

        def _clean(val):
            if val is None:
                return None
            s = str(val).strip()
            if s.lower() in ("null", "none", "n/a", "unknown", "undefined", "unassigned", ""):
                return None
            if s.lower() in VAGUE_ASSIGNEES:
                return None
            return s

        normalized = {
            "assigner": _clean(assigner),
            "assignee": _clean(assignee),
            "task": raw_task,
            "deadline": _clean(deadline)
        }
        # Owner-validation metadata (see services/action_owner_validator.py) is
        # carried through untouched when present.
        for key in ACTION_EVIDENCE_KEYS:
            if item.get(key) is not None:
                normalized[key] = item[key]
        return normalized
    elif isinstance(item, str) and item.strip():
        return {
            "assigner": None,
            "assignee": None,
            "task": item.strip(),
            "deadline": None
        }
    return {
        "assigner": None,
        "assignee": None,
        "task": "",
        "deadline": None
    }


def normalize_stage2_action_item(item: Any) -> Dict[str, Any]:
    """
    Normalizes a Stage 2 action item to:
    {"task": str, "assignee": str|None, "deadline": str|None}
    """
    norm = normalize_action_item(item)
    stage2 = {
        "task": norm["task"],
        "assignee": norm["assignee"],
        "deadline": norm["deadline"],
    }
    for key in ("assigner",) + ACTION_EVIDENCE_KEYS:
        if norm.get(key) is not None:
            stage2[key] = norm[key]
    return stage2


def normalize_action_owner(raw_owner: Any, point_text: str = "", action_items: list = None) -> Optional[str]:
    """
    Extracts and normalizes the action owner for a Stage 1 discussion point.
    Handles strings, lists, action_items fallback, and text patterns like (Owner: Name).
    """
    VAGUE_ASSIGNEES = {
        "my team", "our team", "you", "they", "everyone", "we", "someone", "somebody", "anybody", "team", "us",
        "i", "he", "she", "it", "null", "none", "n/a", "unknown", "undefined", "unassigned", ""
    }

    def _is_valid(val: Any) -> bool:
        if not val:
            return False
        s = str(val).strip()
        return s.lower() not in VAGUE_ASSIGNEES

    if raw_owner:
        if isinstance(raw_owner, list):
            valid_owners = [str(o).strip() for o in raw_owner if _is_valid(o)]
            if valid_owners:
                return ", ".join(dict.fromkeys(valid_owners))
        elif isinstance(raw_owner, str) and _is_valid(raw_owner):
            return raw_owner.strip()

    # Fallback 1: inspect action_items list if present
    if action_items and isinstance(action_items, list):
        act_assignees = []
        for a in action_items:
            if isinstance(a, dict):
                assignee = a.get("assignee") or a.get("owner") or a.get("action_owner")
                if _is_valid(assignee):
                    act_assignees.append(str(assignee).strip())
            elif isinstance(a, str):
                m_act = re.search(r'\((?:Owner|Assignee|Action Owner):\s*([^)]+)\)', a, re.IGNORECASE)
                if m_act and _is_valid(m_act.group(1)):
                    act_assignees.append(m_act.group(1).strip())
        if act_assignees:
            return ", ".join(dict.fromkeys(act_assignees))

    # Fallback 2: regex extract from point_text
    if point_text and isinstance(point_text, str):
        m = re.search(r'\((?:Owner|Assignee|Action Owner):\s*([^)]+)\)', point_text, re.IGNORECASE)
        if not m:
            m = re.search(r'\[(?:Owner|Assignee|Action Owner):\s*([^\]]+)\]', point_text, re.IGNORECASE)
        if m and _is_valid(m.group(1)):
            return m.group(1).strip()

    return None

def format_action_point_display_text(item: Dict[str, Any]) -> str:
    """
    Generates displayed Action Point text from a structured JSON dict:
    e.g. "{assigner} assigned {assignee} to complete {task} before {deadline}"
    """
    if not isinstance(item, dict):
        return str(item).strip()

    task = str(item.get("task") or "").strip()
    if not task:
        return ""

    assigner = item.get("assigner")
    assignee = item.get("assignee")
    deadline = item.get("deadline")

    if assigner and str(assigner).lower() in ("null", "none", "n/a", "unknown", "unassigned"):
        assigner = None
    if assignee and str(assignee).lower() in ("null", "none", "n/a", "unknown", "unassigned"):
        assignee = None
    if deadline and str(deadline).lower() in ("null", "none", "n/a", "unknown", "asap"):
        deadline = None

    def _mentions(haystack: str, needle: Any) -> bool:
        needle_s = str(needle).strip().rstrip(".")
        return bool(needle_s) and re.search(
            r"(?<!\w)" + re.escape(needle_s) + r"(?!\w)", haystack, re.IGNORECASE
        ) is not None

    text = task

    # LLM-written tasks often already read "Bob will ... by Friday." — never repeat
    # the deadline or prefix an owner the sentence already names.
    if deadline and not _mentions(text, deadline):
        text = f"{text.rstrip().rstrip('.')} before {deadline}"

    if assignee:
        from services.action_owner_validator import split_owner_names
        if all(_mentions(text, name) for name in split_owner_names(assignee)):
            return text
    elif assigner and _mentions(text, assigner):
        return text

    if assigner and assignee:
        prefix = f"{assigner} assigned {assignee} to complete "
        if not text.startswith(prefix):
            text = f"{prefix}{text}"
    elif assigner:
        prefix = f"{assigner} assigned to complete "
        if not text.startswith(prefix):
            text = f"{prefix}{text}"
    elif assignee:
        prefix = f"{assignee} to complete "
        if not text.startswith(prefix):
            text = f"{prefix}{text}"

    return text


def _validate_user_id(user_obj: Any) -> str:
    """
    Validates and extracts a primitive string user_id from any incoming value
    (string, user dict, Pydantic model, or user instance).
    Defends SQLite and FAISS stores against dict parameter binding errors.
    """
    if isinstance(user_obj, str):
        uid = user_obj.strip()
    elif isinstance(user_obj, dict):
        uid = user_obj.get("id") or user_obj.get("user_id") or user_obj.get("sub")
        if isinstance(uid, str):
            uid = uid.strip()
    elif hasattr(user_obj, "id"):
        uid = getattr(user_obj, "id")
        if isinstance(uid, str):
            uid = uid.strip()
    else:
        uid = None

    if not uid or not isinstance(uid, str):
        logger.error(f"[RomService] Invalid user_id parameter: type={type(user_obj).__name__}, value={user_obj}")
        raise ValueError(f"Invalid user_id parameter: expected string UUID, got {type(user_obj).__name__}")
    return uid


def _merge_action_items_into_points(discussion_points: List[Dict], action_extractions: List[Dict]) -> List[Dict]:
    """Helper for merging action extractions into discussion points."""
    if not discussion_points:
        return []
    pts = [dict(p) for p in discussion_points]
    for p in pts:
        p.setdefault("action_items", [])
        p.setdefault("action_owner", None)
    if not action_extractions:
        return pts
    for ext in action_extractions:
        ref = ext.get("source_point_ref")
        owner = ext.get("action_owner") or ext.get("assignee")
        items = ext.get("action_items") or [ext]
        matched_point = None
        if ref:
            for p in pts:
                pt_text = str(p.get("discussion_point") or p.get("text") or "")
                if str(p.get("id")) == str(ref) or str(p.get("window_index")) == str(ref) or (ref and str(ref).lower() in pt_text.lower()):
                    matched_point = p
                    break
        if not matched_point and pts:
            matched_point = pts[-1]

        if matched_point:
            for item in items:
                norm = normalize_action_item(item)
                if norm["task"]:
                    matched_point["action_items"].append(norm)
            if owner:
                matched_point["action_owner"] = owner
            elif not matched_point.get("action_owner"):
                assignees = [
                    a.get("assignee") for a in matched_point.get("action_items", [])
                    if isinstance(a, dict) and a.get("assignee") and str(a.get("assignee")).lower() not in ("none", "n/a", "null", "unassigned", "")
                ]
                if assignees:
                    matched_point["action_owner"] = ", ".join(dict.fromkeys(assignees))
    return pts


def _attach_point_turns_and_speakers(
    points: List[Dict],
    window_turn_ids: List[str],
    turns_by_id: Dict[str, Dict],
    roster: List[Dict],
) -> None:
    """Keep only the point's turn IDs that belong to this window, and derive the
    point's speakers from those turns plus LLM-listed speakers that are real
    participants (hallucinated names are dropped; "For Information" OCR points stay)."""
    from services.action_owner_validator import parse_turn_ids, canonicalize_owner

    window_ids = set(window_turn_ids)
    for p in points:
        p["turn_ids"] = [tid for tid in parse_turn_ids(p.get("turn_ids")) if tid in window_ids]
        turn_speakers = [turns_by_id[tid]["speaker"] for tid in p["turn_ids"]]
        llm_speakers = p.get("speakers") or []
        if isinstance(llm_speakers, str):
            llm_speakers = [s.strip() for s in llm_speakers.split(",") if s.strip()]
        kept = []
        for spk in llm_speakers:
            if str(spk).strip().lower() == "for information":
                kept.append("For Information")
                continue
            name, resolved = canonicalize_owner(spk, roster)
            if name and (resolved or not roster):
                kept.append(name)
        merged = list(dict.fromkeys(turn_speakers + kept))
        if merged:
            p["speakers"] = merged


def _best_point_for_action(action: Dict, points: List[Dict]) -> Dict:
    """Pick the discussion point an action item belongs to: the point covering its
    evidence turns, else the point nearest to them, else the best task-text overlap."""
    from services.action_owner_validator import task_similarity

    ev_ids = set((action.get("evidence") or {}).get("turn_ids") or [])
    if ev_ids:
        overlaps = [(len(ev_ids & set(p.get("turn_ids") or [])), idx) for idx, p in enumerate(points)]
        best_overlap, best_idx = max(overlaps, key=lambda x: (x[0], -x[1]))
        if best_overlap > 0:
            return points[best_idx]
        ev_nums = [int(t[1:]) for t in ev_ids]
        with_turns = [p for p in points if p.get("turn_ids")]
        if with_turns:
            return min(
                with_turns,
                key=lambda p: min(abs(a - int(t[1:])) for a in ev_nums for t in p["turn_ids"]),
            )
    task = action.get("task") or ""
    scored = [(task_similarity(task, p.get("discussion_point") or ""), -idx) for idx, p in enumerate(points)]
    return points[-max(scored)[1]]


OWNER_EXCERPT_MAX_CHARS = 1200
_TURN_LINE_RE = re.compile(r"^\[(T\d+) \|")


def _owner_transcript_excerpt(sources: List[Dict], evidence_ids: List[str], turn_ids: List[str]) -> Optional[str]:
    """Transcript lines (from the sources' Stage 1 window text) for the turns that
    prove ownership first, then the point's other turns, capped in length."""
    lines: Dict[str, str] = {}
    for m in sources:
        raw = m.get("owner_transcript_excerpt") or m.get("raw_transcript_text") or ""
        for line in str(raw).splitlines():
            line = line.strip()
            match = _TURN_LINE_RE.match(line)
            if match:
                lines.setdefault(match.group(1), line)

    by_num = lambda tid: int(tid[1:])
    ordered = sorted(dict.fromkeys(evidence_ids), key=by_num)
    ordered += sorted((t for t in dict.fromkeys(turn_ids) if t not in set(ordered)), key=by_num)
    picked, total = [], 0
    for tid in ordered:
        line = lines.get(tid)
        if not line:
            continue
        if total + len(line) > OWNER_EXCERPT_MAX_CHARS and picked:
            break
        picked.append(tid)
        total += len(line) + 1
    return "\n".join(lines[t] for t in sorted(picked, key=by_num)) or None


def _carry_stage1_attribution(
    ep: Dict,
    sources: List[Dict],
    allow_llm_actions: bool = False,
    carry_actions: bool = True,
) -> Dict:
    """Carry speaker and owner attribution from source points into an enhanced or
    merged Stage 2 point as data. The enhancement LLM never sees the transcript, so
    it may narrow the speaker list but never add speakers or owners of its own."""
    from services.action_owner_validator import split_owner_names, min_confidence, actions_are_duplicates

    # Speakers: the sources' (turn-derived) speakers, narrowed by the LLM when it lists a subset.
    source_speakers = list(dict.fromkeys(spk for m in sources for spk in (m.get("speakers") or []) if spk))
    llm_speakers = ep.get("speakers") or []
    if isinstance(llm_speakers, str):
        llm_speakers = [s.strip() for s in llm_speakers.split(",") if s.strip()]
    if source_speakers:
        by_lower = {str(s).lower(): s for s in source_speakers}
        kept = list(dict.fromkeys(by_lower[str(s).lower()] for s in llm_speakers if str(s).lower() in by_lower))
        ep["speakers"] = kept or source_speakers
    else:
        ep["speakers"] = list(llm_speakers)

    # Owners: union of all sources' validated owners (not just the first match).
    owners = list(dict.fromkeys(n for m in sources for n in split_owner_names(m.get("action_owner"))))
    if owners:
        ep["action_owner"] = ", ".join(owners)
        confidence = min_confidence(m.get("owner_confidence") for m in sources if m.get("action_owner"))
        if confidence:
            ep["owner_confidence"] = confidence
    else:
        llm_owner = normalize_action_owner(ep.get("action_owner"))
        ep["action_owner"] = llm_owner
        if llm_owner:
            # Owner introduced during enhancement without transcript support.
            ep["owner_confidence"] = "low"

    if carry_actions:
        inherited: List[Dict] = []
        for m in sources:
            for a in m.get("action_items") or []:
                norm = normalize_stage2_action_item(a)
                if norm["task"] and not any(actions_are_duplicates(x, norm) for x in inherited):
                    inherited.append(norm)
        if not inherited and allow_llm_actions:
            raw_acts = ep.get("action_items") or []
            if not isinstance(raw_acts, list):
                raw_acts = [raw_acts]
            inherited = [n for n in (normalize_stage2_action_item(a) for a in raw_acts) if n["task"]]
        ep["action_items"] = inherited
    else:
        ep["action_items"] = []

    turn_ids = list(dict.fromkeys(
        t for m in sources for t in (m.get("stage1_turn_ids") or m.get("turn_ids") or [])
    ))
    if turn_ids:
        ep["stage1_turn_ids"] = turn_ids
    owner_evidence = [e for m in sources for e in (m.get("owner_evidence") or []) if isinstance(e, dict)]
    if owner_evidence:
        ep["owner_evidence"] = owner_evidence
    evidence_ids = [t for e in owner_evidence for t in (e.get("turn_ids") or [])]
    evidence_ids += [
        t for a in ep["action_items"] if isinstance(a, dict)
        for t in ((a.get("evidence") or {}).get("turn_ids") or [])
    ]
    excerpt = _owner_transcript_excerpt(sources, evidence_ids, turn_ids)
    if excerpt:
        ep["owner_transcript_excerpt"] = excerpt
    return ep


def _unenhanced_stage2_point(p: Dict, separate_action_extraction: bool) -> Dict:
    """Stage 1 point carried into Stage 2 unchanged (used when a Stage 2 window fails)."""
    ep = {
        "id": str(uuid.uuid4()),
        "original_point_ids": [p.get("id")] if p.get("id") else [],
        "original_point_id": p.get("id") or "N/A",
        "polished_text": p.get("discussion_point", ""),
        "discussion_point": p.get("discussion_point", ""),
        "timeline_start": p.get("timeline_start", 0.0),
        "timeline_end": p.get("timeline_end", 0.0),
        "technical_terms": p.get("technical_terms", []),
        "dates": [],
    }
    return _carry_stage1_attribution(ep, [p], carry_actions=separate_action_extraction)


def _refresh_point_owner_from_actions(point: Dict) -> None:
    """Set a point's action_owner/owner_confidence from its own validated action items."""
    from services.action_owner_validator import split_owner_names, min_confidence

    owned = [a for a in point.get("action_items") or [] if isinstance(a, dict) and a.get("assignee")]
    owners = list(dict.fromkeys(name for a in owned for name in split_owner_names(a["assignee"])))
    point["action_owner"] = ", ".join(owners) if owners else None
    if owners:
        point["owner_confidence"] = min_confidence(a.get("owner_confidence") for a in owned)
    else:
        point.pop("owner_confidence", None)


# ── Stage 3: point ↔ agenda semantic similarity ──────────────────────────────

# Contextual facets (supporting documents, previous MoMs) are broader than the agenda's
# own wording, so they count slightly less when picking a point's best-matching detail.
_CONTEXTUAL_FACET_WEIGHT = 0.9
_CONTEXTUAL_FACETS = {"previous_meeting", "supporting_context", "document_points"}
# Low-confidence LLM assignments are corrected when similarity strongly prefers another agenda.
_SIMILARITY_OVERRIDE_MARGIN = 0.10


# Bookkeeping keys of a previous-meeting expansion that carry no subject matter.
_NON_TEXT_DETAIL_KEYS = {"agenda_id", "found_in_previous_meeting"}


def _join_detail(value: Any) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(s for s in (_join_detail(v) for v in value if v is not None) if s)
    if isinstance(value, dict):
        return " ".join(
            s for s in (_join_detail(v) for k, v in value.items() if v and k not in _NON_TEXT_DETAIL_KEYS) if s
        )
    return str(value).strip() if value is not None else ""


_AGENDA_LIST_FIELDS = ("keywords", "related_concepts", "alternative_terminology", "expected_themes")


def normalize_agendas(raw_agendas: Any, merge_duplicates: bool = True) -> List[Dict[str, Any]]:
    """Make agendas from the LLM or the cache safe to use: every agenda gets a title, a
    unique agenda_id (missing ids become the next free A<n>), list fields and a
    speaker/presenter. With ``merge_duplicates`` an agenda repeating an earlier title is
    merged into it."""
    agendas: List[Dict[str, Any]] = []
    by_title: Dict[str, Dict[str, Any]] = {}
    used_ids = set()
    for item in raw_agendas or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("topic") or "").strip()
        if not title:
            continue
        title_key = re.sub(r"\W+", " ", title.lower()).strip()
        if merge_duplicates and title_key in by_title:
            existing = by_title[title_key]
            for field in _AGENDA_LIST_FIELDS:
                extra = [v for v in item.get(field) or [] if v not in existing[field]]
                existing[field] = existing[field] + extra
            if not existing["description"] and (item.get("description") or item.get("details")):
                existing["description"] = str(item.get("description") or item.get("details")).strip()
            continue
        agenda = dict(item)
        agenda["title"] = title
        agenda["description"] = str(item.get("description") or item.get("details") or "").strip()
        speaker = item.get("speaker") or item.get("presenter")
        agenda["speaker"] = speaker
        agenda["presenter"] = speaker
        for field in _AGENDA_LIST_FIELDS:
            values = item.get(field)
            agenda[field] = [v for v in values if v] if isinstance(values, list) else []
        agenda_id = str(item.get("agenda_id") or "").strip()
        agenda["agenda_id"] = agenda_id if agenda_id and agenda_id not in used_ids else ""
        if agenda["agenda_id"]:
            used_ids.add(agenda["agenda_id"])
        agendas.append(agenda)
        by_title.setdefault(title_key, agenda)
    next_num = 1
    for agenda in agendas:
        if not agenda["agenda_id"]:
            while f"A{next_num}" in used_ids:
                next_num += 1
            agenda["agenda_id"] = f"A{next_num}"
            used_ids.add(agenda["agenda_id"])
    return agendas


def _agenda_similarity_facets(
    agenda: Dict[str, Any],
    expanded: Optional[Dict[str, Any]] = None,
    doc_points: Optional[List[str]] = None,
) -> List[tuple]:
    """
    Every agenda detail a discussion point is compared against, as (facet, text).
    "profile" combines the agenda's own fields; the others are scored individually so
    a point matching one specific detail (e.g. an alternative term) is still recognised.
    """
    title = _join_detail(agenda.get("title"))
    description = _join_detail(agenda.get("description") or agenda.get("details"))
    own_fields = [
        ("title", title),
        ("description", description),
        ("keywords", _join_detail(agenda.get("keywords"))),
        ("related_concepts", _join_detail(agenda.get("related_concepts"))),
        ("alternative_terminology", _join_detail(agenda.get("alternative_terminology"))),
        ("expected_themes", _join_detail(agenda.get("expected_themes"))),
    ]
    profile = ". ".join(text for _, text in own_fields if text)
    facets = [("profile", profile or title)]
    facets.extend((name, text) for name, text in own_fields if text)

    expanded = expanded or {}
    previous = _join_detail(expanded.get("previous_meeting_summary"))
    if previous:
        facets.append(("previous_meeting", previous[:1500]))
    context = _join_detail(expanded.get("meeting_context_snippet"))
    if context:
        facets.append(("supporting_context", context[:1500]))
    doc_text = _join_detail(doc_points)
    if doc_text:
        facets.append(("document_points", doc_text[:1500]))
    return facets


def compute_point_agenda_similarity(
    points: List[Dict[str, Any]],
    agendas: List[Dict[str, Any]],
    embedder,
    expanded_agendas: Optional[List[Dict[str, Any]]] = None,
    agenda_doc_points: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Score every discussion point against every agenda using the active embedding
    model (the selected trained variant, if any).

    Per agenda:  score = 0.5 * sim(point, agenda profile)
                       + 0.5 * max over individual agenda details (title, description,
                         keywords, related concepts, alternative terminology, expected
                         themes, previous-meeting/supporting context, document points)

    Returns {"agenda_ids", "matrix" (points × agendas), "candidates" (top_k per point),
             "scores" ({point_id: {agenda_id: score}})}.
    """
    expanded_by_id = {e.get("agenda_id"): e for e in (expanded_agendas or []) if isinstance(e, dict)}
    agenda_ids = [a.get("agenda_id") or f"A{i + 1}" for i, a in enumerate(agendas)]

    facet_texts: List[str] = []
    facet_owner: List[tuple] = []  # (agenda index, facet name)
    for a_idx, agenda in enumerate(agendas):
        doc_entry = (agenda_doc_points or {}).get(agenda_ids[a_idx])
        doc_pts = doc_entry.get("points") if isinstance(doc_entry, dict) else None
        for name, text in _agenda_similarity_facets(agenda, expanded_by_id.get(agenda_ids[a_idx]), doc_pts):
            facet_texts.append(text)
            facet_owner.append((a_idx, name))

    point_texts = [
        (p.get("polished_text") or p.get("text") or p.get("discussion_point") or "").strip()
        for p in points
    ]

    vecs = np.asarray(embedder.encode_batch(facet_texts + point_texts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms < 1e-9, 1.0, norms)
    facet_vecs, point_vecs = vecs[:len(facet_texts)], vecs[len(facet_texts):]
    facet_sims = point_vecs @ facet_vecs.T  # (points, facets)

    n_points, n_agendas = len(points), len(agendas)
    profile = np.zeros((n_points, n_agendas), dtype=np.float32)
    best_detail = np.full((n_points, n_agendas), -np.inf, dtype=np.float32)
    best_facet = [[("profile", 0.0)] * n_agendas for _ in range(n_points)]
    for f_idx, (a_idx, name) in enumerate(facet_owner):
        col = facet_sims[:, f_idx]
        if name == "profile":
            profile[:, a_idx] = col
            continue
        weighted = col * (_CONTEXTUAL_FACET_WEIGHT if name in _CONTEXTUAL_FACETS else 1.0)
        for p_idx in range(n_points):
            if weighted[p_idx] > best_detail[p_idx, a_idx]:
                best_detail[p_idx, a_idx] = weighted[p_idx]
                best_facet[p_idx][a_idx] = (name, float(col[p_idx]))
    best_detail = np.where(np.isneginf(best_detail), profile, best_detail)
    matrix = 0.5 * profile + 0.5 * best_detail

    candidates = []
    scores: Dict[str, Dict[str, float]] = {}
    for p_idx, point in enumerate(points):
        pid = str(point.get("id") or f"pt_{p_idx + 1}")
        row = matrix[p_idx]
        if not point_texts[p_idx]:
            # Nothing to compare: no candidates, so callers keep their non-similarity fallback.
            matrix[p_idx] = 0.0
            candidates.append({"point_id": pid, "candidates": [], "margin": 0.0})
            continue
        scores[pid] = {agenda_ids[a]: round(float(row[a]), 4) for a in range(n_agendas)}
        order = np.argsort(row)[::-1][:max(1, top_k)]
        ranked = [
            {
                "agenda_id": agenda_ids[a],
                "agenda_title": agendas[a].get("title", ""),
                "score": round(float(row[a]), 4),
                "best_facet": best_facet[p_idx][a][0],
                "best_facet_score": round(best_facet[p_idx][a][1], 4),
            }
            for a in order
        ]
        margin = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else ranked[0]["score"]
        candidates.append({"point_id": pid, "candidates": ranked, "margin": round(float(margin), 4)})

    return {
        "agenda_ids": agenda_ids,
        "matrix": [[round(float(v), 4) for v in row] for row in matrix],
        "candidates": candidates,
        "scores": scores,
    }


def _agenda_reference_entry(agenda: Dict[str, Any]) -> Dict[str, Any]:
    """Agenda details shown to the LLM: everything relevant, not just title/description."""
    entry = {
        "agenda_id": agenda["agenda_id"],
        "title": agenda.get("title", ""),
        "description": agenda.get("description", ""),
    }
    for key in ("keywords", "related_concepts", "alternative_terminology", "expected_themes"):
        values = agenda.get(key)
        if isinstance(values, list):
            cleaned = [str(v).strip() for v in values if v is not None and str(v).strip()]
            if cleaned:
                entry[key] = cleaned[:15]
    return entry


SIMILARITY_GUIDANCE_TEXT = """SEMANTIC SIMILARITY REFERENCE (secondary evidence only):
Each discussion point includes "similarity_reference": the top agenda candidates ranked by embedding similarity between the point and ALL details of each agenda (title, description, keywords, related concepts, alternative terminology, expected themes and supporting context).
- Scores are relative: compare them between agendas for the same point; do not read them as probabilities.
- Use them as a tie-breaker and sanity check, never as proof on their own.
- When the discussion content clearly identifies an agenda, follow the content even if another agenda scores higher.
- When the content is ambiguous, prefer the higher-scoring candidate; a clear gap between the top two candidates is stronger evidence than a narrow one.
- If you choose an agenda that is not the top-scoring candidate, the reason must name the specific content that justifies it."""


class RomService:
    def extract_discussion_points(
        self,
        transcript: List[Dict],
        window_minutes: float = 2.0,
        user_id: str = None,
        recording_id: str = None,
        video_transcript: Optional[List[Dict]] = None,
        source_type: str = "audio",
        parallel_window_processing: Optional[int] = None,
        separate_action_extraction: bool = False,
        speaker_mappings: Optional[Dict[str, Any]] = None,
        verify_action_owners: bool = False,
    ) -> Dict:
        """Stage 1: Process transcript in sliding windows to extract discussion points.

        Args:
            separate_action_extraction: When True, each window runs two parallel LLM calls —
                one for discussion points (action_items omitted) and one for action items
                only.  Each action item is validated against the turns it cites and attached
                to the discussion point covering those turns; action_owner is derived from
                the validated items.  When False (default), a single embedded call is used and
                the point-level action_owner is validated against its owner_evidence.
            speaker_mappings: Recording speaker renames, used as aliases in the participant roster.
            verify_action_owners: When True, owners whose validation confidence is not "high"
                get one extra focused LLM check over the turns around their evidence.
        """
        from services.ai_provider import get_provider
        from services.video_processing_service import get_overlapping_ocr_blocks
        from services.rom_transcript_format import (
            build_turns, split_turns_into_windows, render_window, build_roster, format_roster_block,
        )
        from services.action_owner_validator import (
            validate_action, validate_point_owners, evidence_outside_focus, dedupe_point_actions,
        )
        from config import settings
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if user_id:
            user_id = _validate_user_id(user_id)
            
        if not transcript:
            return {"discussion_points": [], "windows_processed": 0}
            
        provider = get_provider()
        
        # Speaker-aware turns with recording-global IDs (T1..Tn). Windows are built
        # over turns so a question and its reply stay together, and every action
        # item can cite the turns that prove who owns it.
        turns = build_turns(transcript)
        turns_by_id = {t["turn_id"]: t for t in turns}
        windows = split_turns_into_windows(turns, window_minutes)
        roster = build_roster(transcript, speaker_mappings)
        participants_block = format_roster_block(roster)

        total_windows = len(windows)
        has_video_ocr = bool(video_transcript)

        concurrency = parallel_window_processing
        if concurrency is None:
            concurrency = getattr(settings, "ROM_PARALLEL_WINDOW_PROCESSING", 2)
        try:
            concurrency = max(1, min(5, int(concurrency)))
        except (ValueError, TypeError):
            concurrency = 2

        logger.info(
            f"[ROM Service] Stage 1 starting: {total_windows} time window(s) to process"
            f" | turns={len(turns)} | participants={[r['label'] for r in roster]}"
            f" | video_ocr={'enabled' if has_video_ocr else 'disabled'}"
            f" | parallel_concurrency={concurrency}"
        )

        def _verify_owner(task: str, turn_ids: List[str]) -> Optional[Dict]:
            """Optional focused LLM check of one owner; the answer is accepted only if
            it resolves to a participant and passes deterministic validation."""
            nums = sorted(int(t[1:]) for t in turn_ids if t in turns_by_id)
            if not nums:
                return None
            context = turns[max(0, nums[0] - 3): nums[-1] + 2]  # evidence turns ±2 (T<n> is turns[n-1])
            try:
                answer = provider.verify_action_owner(task, render_window(context), participants_block)
            except Exception as v_err:
                logger.warning(f"[ROM Service] Owner verification call failed: {v_err}")
                return None
            assignee = answer.get("assignee") if isinstance(answer, dict) else None
            if not assignee or str(assignee).strip().lower() == "unclear":
                return None
            candidate = validate_action(
                {
                    "task": task,
                    "assignee": assignee,
                    "assigner": answer.get("assigner"),
                    "assignee_basis": answer.get("basis"),
                    "evidence_turn_ids": answer.get("evidence_turn_ids") or turn_ids,
                },
                turns_by_id,
                roster,
            )
            if candidate["owner_resolved"] and candidate["owner_confidence"] in ("high", "medium"):
                return candidate
            return None

        def _verify_window_owners(points: List[Dict]) -> None:
            for p in points:
                if separate_action_extraction:
                    for act in p.get("action_items") or []:
                        if not act.get("assignee") or act.get("owner_confidence") == "high":
                            continue
                        ev_ids = (act.get("evidence") or {}).get("turn_ids") or p.get("turn_ids") or []
                        verified = _verify_owner(act.get("task", ""), ev_ids)
                        if verified:
                            logger.info(f"[ROM Service] owner_verified: '{act.get('assignee')}' -> '{verified['assignee']}' ({verified['owner_confidence']})")
                            for key in ("assignee", "assigner", "assignee_basis", "owner_confidence", "owner_resolved", "evidence"):
                                act[key] = verified.get(key)
                    _refresh_point_owner_from_actions(p)
                elif p.get("action_owner") and p.get("owner_confidence") != "high" and "," not in p["action_owner"]:
                    ev_ids = [t for e in p.get("owner_evidence") or [] for t in (e.get("turn_ids") or [])] or p.get("turn_ids") or []
                    verified = _verify_owner(p.get("discussion_point", ""), ev_ids)
                    if verified:
                        logger.info(f"[ROM Service] owner_verified: '{p['action_owner']}' -> '{verified['assignee']}' ({verified['owner_confidence']})")
                        p["action_owner"] = verified["assignee"]
                        p["owner_confidence"] = verified["owner_confidence"]
                        p["owner_evidence"] = [{
                            "owner": verified["assignee"],
                            "basis": verified["assignee_basis"],
                            "turn_ids": (verified.get("evidence") or {}).get("turn_ids", []),
                            "confidence": verified["owner_confidence"],
                        }]

        def _ocr_context(w_from: float, w_to: float) -> str:
            blocks = get_overlapping_ocr_blocks(video_transcript, w_from, w_to)
            return "\n\n".join(
                f"[{_format_time_hhmm(b['start'])} -> {_format_time_hhmm(b['end'])}]\n{b['text']}" for b in blocks
            )

        def _process_single_window(i_and_win):
            i, window = i_and_win
            _win_start_time = time.monotonic()
            w_start = round(window[0]["start"], 2) if window else 0.0
            w_end = round(window[-1]["end"], 2) if window else 0.0
            window_turn_ids = [t["turn_id"] for t in window]

            t_range = f"{_format_time_hhmm(w_start)}-{_format_time_hhmm(w_end)}"
            logger.info(f"[ROM Service] Processing Window {i+1}/{total_windows} ({t_range}) [Parallel limit={concurrency}]")

            # Turn lines use the resolved speaker label (never raw Pyannote IDs).
            window_text = render_window(window)
            video_context = _ocr_context(w_start, w_end) if has_video_ocr else ""
            prev_window_text = render_window(windows[i - 1]) if i > 0 else ""

            if separate_action_extraction:
                # ── Split-call path: run both LLM calls in parallel ──────────────────────
                # Call 1: Discussion points (no action_items) via no-actions prompt variant.
                # Call 2: Action items only via dedicated action extraction prompt. It also
                #         sees the next window as LOOK-AHEAD so an acceptance or deadline
                #         that lands after the window boundary is not lost.
                from concurrent.futures import ThreadPoolExecutor as _TPE

                double_window_text = window_text
                double_video_context = video_context
                if i + 1 < len(windows):
                    next_win = windows[i + 1]
                    double_window_text += (
                        "\n--- LOOK-AHEAD: next window (use only to confirm the acceptance, owner or deadline "
                        "of actions raised above; do not extract actions that start here) ---\n"
                        + render_window(next_win)
                    )
                    if has_video_ocr:
                        next_ocr = _ocr_context(round(next_win[0]["start"], 2), round(next_win[-1]["end"], 2))
                        if next_ocr:
                            double_video_context = f"{double_video_context}\n\n{next_ocr}" if double_video_context else next_ocr
                    logger.info(
                        f"[ROM Service] Window {i+1} action extraction using 2× window "
                        f"(current + look-ahead window {i+2})"
                    )

                def _call_discussion():
                    return provider.extract_rom_discussion_points(
                        window_text, None, video_context=video_context,
                        skip_action_items=True, separate_action_extraction=True,
                        previous_transcript=prev_window_text or None, participants=participants_block,
                    )

                def _call_actions():
                    return provider.extract_rom_action_points(
                        double_window_text, None, video_context=double_video_context,
                        previous_transcript=prev_window_text or None, participants=participants_block,
                    )

                with _TPE(max_workers=2) as _inner_exec:
                    fut_disc = _inner_exec.submit(_call_discussion)
                    fut_acts = _inner_exec.submit(_call_actions)

                    # Retrieve discussion result (errors propagate to outer try/except)
                    disc_result = fut_disc.result()
                    points = disc_result.get("discussion_points", [])
                    parse_error = bool(disc_result.get("parse_error"))
                    error_reason = disc_result.get("error_reason", "")

                    # Retrieve action result — failures are soft (continue with empty list)
                    validated_actions = []
                    try:
                        action_result = fut_acts.result()
                        raw_acts = action_result.get("action_items") or []
                        if not isinstance(raw_acts, list):
                            raw_acts = [raw_acts]
                        outside_focus = 0
                        for a in raw_acts:
                            norm = normalize_action_item(a)
                            if not norm["task"]:
                                continue
                            validated = validate_action(norm, turns_by_id, roster)
                            # Actions first raised in the look-ahead are extracted by the next window.
                            if evidence_outside_focus(validated, window_turn_ids):
                                outside_focus += 1
                                continue
                            validated_actions.append(validated)
                        if outside_focus:
                            logger.info(f"[ROM Service] Window {i+1}: skipped {outside_focus} action(s) raised in the look-ahead window")
                    except Exception as act_err:
                        logger.warning(
                            f"[ROM Service] Separate action extraction Call 2 failed for window {i+1}: {act_err}. "
                            "Continuing with empty action_items."
                        )
                        validated_actions = []

                _attach_point_turns_and_speakers(points, window_turn_ids, turns_by_id, roster)
                for p in points:
                    p["action_items"] = []

                if points:
                    for act in validated_actions:
                        _best_point_for_action(act, points)["action_items"].append(act)

                # action_owner = the validated owners of the point's own action items
                for p in points:
                    _refresh_point_owner_from_actions(p)

                logger.info(
                    f"[ROM Service] Window {i+1} separate action extraction (2× window): "
                    f"{len(validated_actions)} validated action item(s) mapped across {len(points)} discussion point(s)."
                )

            else:
                # ── Default embedded-action path (separate_action_extraction=False) ─────────
                # Uses ROM_DISCUSSION_EMBEDDED_PROMPT: actions are woven into discussion_point text.
                # action_owner comes with owner_evidence (basis + turn IDs) and is validated
                # against the cited turns.
                result = provider.extract_rom_discussion_points(
                    window_text, None, video_context=video_context,
                    separate_action_extraction=False,
                    previous_transcript=prev_window_text or None, participants=participants_block,
                )
                points = result.get("discussion_points", [])
                parse_error = bool(result.get("parse_error"))
                error_reason = result.get("error_reason", "")

                _attach_point_turns_and_speakers(points, window_turn_ids, turns_by_id, roster)
                for p in points:
                    raw_owner = p.get("action_owner") or p.get("action_owners") or p.get("owner") or p.get("assignee")
                    p["action_owner"] = normalize_action_owner(
                        raw_owner,
                        p.get("discussion_point", ""),
                        p.get("action_items", [])
                    )
                    validate_point_owners(p, turns_by_id, roster)
                    if not isinstance(p.get("action_items"), list):
                        p["action_items"] = []

            if verify_action_owners:
                _verify_window_owners(points)

            # Stamp every point with window metadata (both paths)
            for p in points:
                p["id"] = str(uuid.uuid4())
                p["window_index"] = i
                p["timeline_start"] = w_start
                p["timeline_end"] = w_end
                p["raw_transcript_text"] = window_text.strip()
                p["video_transcript_context"] = video_context

            meta = {
                "t_range": t_range,
                "seg_cnt": len(window),
                "char_cnt": len(window_text.strip()),
                "parse_error": parse_error,
                "error_reason": error_reason,
            }
            window_elapsed_secs = time.monotonic() - _win_start_time
            return i, points, meta, window_elapsed_secs

        window_results: Dict[int, List[Dict]] = {}
        window_errors: Dict[int, str] = {}
        window_meta: Dict[int, Dict] = {}

        # ── Init progress tracking (user_id may be None in tests) ──────────────
        _progress_user = user_id or "__anonymous__"
        _recording_id_str = str(recording_id) if recording_id else "__unknown__"
        if user_id:
            init_stage1_progress(_progress_user, _recording_id_str, total_windows, concurrency)

        try:
            if concurrency == 1 or total_windows <= 1:
                for i, window in enumerate(windows):
                    try:
                        _win_start = time.monotonic()
                        idx, pts, meta, win_elapsed = _process_single_window((i, window))
                        if user_id:
                            update_stage1_progress(_progress_user, _recording_id_str, win_elapsed)
                        window_results[idx] = pts
                        window_meta[idx] = meta
                    except Exception as w_err:
                        logger.error(f"[ROM Service] Stage 1 Window {i+1} failed ({w_err}). Continuing with remaining tasks...", exc_info=True)
                        if user_id:
                            update_stage1_progress(_progress_user, _recording_id_str, time.monotonic() - _win_start)
                        window_errors[i] = str(w_err)
                        window_meta[i] = {
                            "t_range": f"Window {i+1}",
                            "seg_cnt": len(window),
                            "char_cnt": sum(len(s.get('text', '')) for s in window),
                        }
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_to_idx = {
                        executor.submit(_process_single_window, (i, window)): i
                        for i, window in enumerate(windows)
                    }
                    for future in as_completed(future_to_idx):
                        win_idx = future_to_idx[future]
                        try:
                            idx, pts, meta, win_elapsed = future.result()
                            if user_id:
                                update_stage1_progress(_progress_user, _recording_id_str, win_elapsed)
                            window_results[idx] = pts
                            window_meta[idx] = meta
                        except Exception as w_err:
                            logger.error(f"[ROM Service] Stage 1 Window {win_idx+1} failed ({w_err}). Continuing with remaining tasks...", exc_info=True)
                            if user_id:
                                update_stage1_progress(_progress_user, _recording_id_str, 0.0)
                            window_errors[win_idx] = str(w_err)
                            window_meta[win_idx] = {
                                "t_range": f"Window {win_idx+1}",
                                "seg_cnt": len(windows[win_idx]),
                                "char_cnt": sum(len(s.get('text', '')) for s in windows[win_idx]),
                            }
        finally:
            provider.unload_model()
            gc.collect()
            if user_id:
                finish_stage1_progress(_progress_user, _recording_id_str)

        all_points = []
        for i in range(total_windows):
            pts = window_results.get(i, [])
            all_points.extend(pts)

        if separate_action_extraction:
            removed = dedupe_point_actions(all_points)
            if removed:
                for p in all_points:
                    _refresh_point_owner_from_actions(p)
                logger.info(f"[ROM Service] Stage 1 removed {removed} duplicate action item(s) across windows")

        # ── Comprehensive Stage 1 Window Log Summary ───────────────────────────
        summary_lines = [
            "================================================================================",
            f"[ROM Service Stage 1 Completion Summary] Total Discussion Points Extracted: {len(all_points)} across {total_windows} window(s)",
            "--------------------------------------------------------------------------------"
        ]

        for i in range(total_windows):
            pts = window_results.get(i, [])
            w_meta = window_meta.get(i, {})
            w_err = window_errors.get(i)
            t_range = w_meta.get("t_range", f"Window {i+1}")
            seg_cnt = w_meta.get("seg_cnt", 0)
            char_cnt = w_meta.get("char_cnt", 0)
            parse_err = w_meta.get("parse_error")
            err_reason = w_meta.get("error_reason", "")

            pt_cnt = len(pts)
            if pt_cnt > 0:
                summary_lines.append(f"  • Window {i+1}/{total_windows} ({t_range}): {pt_cnt} discussion point(s) extracted")
            else:
                if w_err:
                    cause_desc = f"[CAUSE: EXCEPTION ERROR] Processing error: {w_err}"
                elif char_cnt == 0:
                    cause_desc = f"[CAUSE: EMPTY CONTENT] Transcript window had no spoken text (silence / unvoiced segment)"
                elif parse_err:
                    cause_desc = f"[CAUSE: JSON PARSE ERROR] LLM output could not be parsed into valid JSON ({err_reason})"
                else:
                    cause_desc = f"[CAUSE: NO POINTS IN CONTENT] Spoken content was present ({seg_cnt} segment(s), {char_cnt} char(s)), but LLM extracted 0 points"

                summary_lines.append(f"  • Window {i+1}/{total_windows} ({t_range}): 0 points -> {cause_desc}")

        summary_lines.append("================================================================================")
        logger.info("\n".join(summary_lines))

        return {
            "discussion_points": all_points,
            "windows_processed": len(windows)
        }

    def rerun_single_stage1_window(
        self,
        transcript: List[Dict],
        window_index: int,
        user_feedback: str,
        window_minutes: float = 2.0,
        video_transcript: Optional[List[Dict]] = None,
        existing_points: Optional[List[Dict]] = None,
        speaker_mappings: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Stage 1 Re-run Window: Re-extracts discussion points for a single window
        using the window's transcript, previous output, and user correction feedback.
        Windows are built exactly as in extract_discussion_points (over speaker turns).
        """
        from services.ai_provider import get_provider
        from services.video_processing_service import get_overlapping_ocr_blocks
        from services.rom_transcript_format import (
            build_turns, split_turns_into_windows, render_window, build_roster, format_roster_block,
        )
        from services.action_owner_validator import validate_point_owners

        if not transcript:
            return {"discussion_points": [], "window_index": window_index, "transcript_text": ""}

        provider = get_provider()

        turns = build_turns(transcript)
        turns_by_id = {t["turn_id"]: t for t in turns}
        windows = split_turns_into_windows(turns, window_minutes)
        roster = build_roster(transcript, speaker_mappings)

        # Target index is 1-based (window_index), map to 0-based
        target_idx = max(0, min(len(windows) - 1, window_index - 1))
        target_window = windows[target_idx] if windows else []

        w_start = round(target_window[0]["start"], 2) if target_window else 0.0
        w_end = round(target_window[-1]["end"], 2) if target_window else 0.0

        window_text = render_window(target_window) + "\n" if target_window else ""

        video_context = ""
        if video_transcript:
            relevant_blocks = get_overlapping_ocr_blocks(video_transcript, w_start, w_end)
            if relevant_blocks:
                lines = []
                for b in relevant_blocks:
                    ts = f"{_format_time_hhmm(b['start'])} -> {_format_time_hhmm(b['end'])}"
                    lines.append(f"[{ts}]\n{b['text']}")
                video_context = "\n\n".join(lines)

        prev_window_text = render_window(windows[target_idx - 1]) if target_idx > 0 else ""

        existing_json = json.dumps(existing_points, ensure_ascii=False) if existing_points else ""

        res = provider.extract_rom_discussion_points_with_feedback(
            window_text=window_text,
            existing_points_json=existing_json,
            user_feedback=user_feedback,
            video_context=video_context,
            previous_transcript=prev_window_text or None,
            participants=format_roster_block(roster),
        )

        raw_points = res.get("discussion_points", [])
        _attach_point_turns_and_speakers(raw_points, [t["turn_id"] for t in target_window], turns_by_id, roster)
        regenerated_points = []
        for pt in raw_points:
            pt_id = str(uuid.uuid4())
            pt["id"] = pt_id
            pt["window_index"] = window_index
            pt["timeline_start"] = w_start
            pt["timeline_end"] = w_end
            pt["raw_transcript_text"] = window_text
            if video_context:
                pt["video_transcript_context"] = video_context
            raw_owner = pt.get("action_owner") or pt.get("action_owners") or pt.get("owner") or pt.get("assignee")
            pt["action_owner"] = normalize_action_owner(
                raw_owner,
                pt.get("discussion_point", ""),
                pt.get("action_items", [])
            )
            validate_point_owners(pt, turns_by_id, roster)
            if not isinstance(pt.get("action_items"), list):
                pt["action_items"] = []
            regenerated_points.append(pt)

        return {
            "window_index": window_index,
            "timeline_start": w_start,
            "timeline_end": w_end,
            "transcript_text": window_text,
            "video_context": video_context,
            "original_points": existing_points or [],
            "regenerated_points": regenerated_points,
        }

    def index_stage2_points_in_chromadb(
        self,
        recording_id: str,
        user_id: str,
        points: List[Dict],
        meeting_name: str = "",
        meeting_date: str = "",
        only_missing: bool = False,
        return_details: bool = False,
    ) -> Union[int, Dict[str, int]]:
        """
        Store generated Stage 2 points in ChromaDB collection `stage2_points_<user_id>`
        with relevant metadata (meeting_id, meeting_name, date, stage2_identifier, etc.).

        Parameters
        ----------
        recording_id   : Unique meeting/recording identifier.
        user_id        : Owner user ID.
        points         : List of Stage 2 point dicts.
        meeting_name   : Human-readable title of the meeting.
        meeting_date   : Creation date/time of the meeting.
        only_missing   : If True, do not overwrite existing points; only index new ones.
        return_details : If True, return a dict with detailed counts (total_found,
                         already_indexed, newly_indexed, skipped_failed).
                         If False, return newly_indexed count as int.
        """
        from services.vector_store import get_stage2_points_store
        from services.text_embedding_service import get_text_embedder

        if not user_id or not recording_id:
            if return_details:
                return {
                    "total_found": 0,
                    "already_indexed": 0,
                    "newly_indexed": 0,
                    "skipped_failed": 0,
                }
            return 0

        user_id = _validate_user_id(user_id)

        if not points:
            if not only_missing:
                try:
                    store = get_stage2_points_store(user_id)
                    if store.exists():
                        store.delete_by_filter("meeting_id", recording_id)
                except Exception as e:
                    logger.warning(f"[ROM Service] Failed to purge Stage 2 vectors for {recording_id}: {e}")
            if return_details:
                return {
                    "total_found": 0,
                    "already_indexed": 0,
                    "newly_indexed": 0,
                    "skipped_failed": 0,
                }
            return 0

        embedder = get_text_embedder()
        embedder.load()
        dim = getattr(embedder, "_dim", 1024)
        store = get_stage2_points_store(user_id, dim)
        store.load_or_create()

        already_indexed = 0
        skipped_failed = 0

        existing_point_ids = set()
        existing_stage2_ids = set()
        existing_chunk_ids = set()
        existing_texts = set()

        if only_missing:
            # Check what Stage 2 points already exist in ChromaDB for this meeting
            try:
                if store._collection is not None:
                    existing = store._collection.get(
                        where={"meeting_id": recording_id},
                        include=["metadatas", "documents"],
                    )
                    if existing:
                        for ex_id in (existing.get("ids") or []):
                            if ex_id:
                                existing_chunk_ids.add(ex_id)
                        for meta in (existing.get("metadatas") or []):
                            if meta:
                                pid = meta.get("point_id")
                                if pid:
                                    existing_point_ids.add(str(pid))
                                s2_id = meta.get("stage2_identifier")
                                if s2_id:
                                    existing_stage2_ids.add(str(s2_id))
                        for doc in (existing.get("documents") or []):
                            if doc:
                                existing_texts.add(doc.strip())
            except Exception as e:
                logger.warning(f"[ROM Service] Failed to check existing Stage 2 vectors for {recording_id}: {e}")
        else:
            # 1. Remove any previous entries for this meeting to prevent duplicates
            try:
                store.delete_by_filter("meeting_id", recording_id)
            except Exception as e:
                logger.warning(f"[ROM Service] Failed to delete previous Stage 2 vectors for {recording_id}: {e}")

        # 2. Extract texts and build rich metadata for each Stage 2 point
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, p in enumerate(points):
            if isinstance(p, dict):
                pt_text = (p.get("polished_text") or p.get("discussion_point") or "").strip()
                point_id = str(p.get("id") or "")
                spk_val = p.get("speakers")
                if isinstance(spk_val, list):
                    spk_str = ", ".join(str(s) for s in spk_val if s)
                else:
                    spk_str = str(spk_val or "")
                action_owner = p.get("action_owner") or ""
            elif isinstance(p, str):
                pt_text = p.strip()
                point_id = ""
                spk_str = ""
                action_owner = ""
            else:
                pt_text = ""
                point_id = ""
                spk_str = ""
                action_owner = ""

            if not pt_text:
                skipped_failed += 1
                continue

            if not point_id:
                point_id = str(uuid.uuid4())

            stage2_id = f"stage2_{recording_id}_{point_id}"
            chunk_id = f"stage2_point_{recording_id}_chunk_{idx}"

            if only_missing:
                is_already = (
                    (point_id and point_id in existing_point_ids)
                    or (stage2_id in existing_stage2_ids)
                    or (chunk_id in existing_chunk_ids)
                    or (pt_text in existing_texts)
                )
                if is_already:
                    already_indexed += 1
                    continue

            meta = {
                "user_id": user_id,
                "meeting_id": recording_id,
                "recording_id": recording_id,
                "doc_id": recording_id,
                "chunk_index": idx,
                "meeting_name": meeting_name or "Meeting",
                "date": meeting_date or "",
                "point_id": point_id,
                "stage": "stage2",
                "stage2_identifier": stage2_id,
                "speakers": spk_str,
                "action_owner": action_owner,
                "source": "stage2",
            }
            texts.append(pt_text)
            metadatas.append(meta)

        newly_indexed = 0
        if texts:
            try:
                vecs = embedder.encode_batch(texts)
                added_count = store.add(texts=texts, metadatas=metadatas, embeddings=vecs)
                newly_indexed = added_count
                logger.info(
                    f"[ROM Service] Indexed {added_count} Stage 2 points for meeting '{meeting_name}' "
                    f"({recording_id}) in ChromaDB collection '{store._collection_name}'"
                )
            except Exception as e:
                logger.error(f"[ROM Service] Failed to index Stage 2 points in ChromaDB: {e}", exc_info=True)
                skipped_failed += len(texts)

        if return_details:
            return {
                "total_found": len(points),
                "already_indexed": already_indexed,
                "newly_indexed": newly_indexed,
                "skipped_failed": skipped_failed,
            }
        return newly_indexed

    def index_final_rom_points_in_chromadb(
        self,
        recording_id: str,
        user_id: str,
        final_rom: Dict[str, Any],
        meeting_name: str = "",
        meeting_date: str = "",
        version: str = "long",
        only_missing: bool = False,
        return_details: bool = False,
    ) -> Union[int, Dict[str, int]]:
        """
        Store generated/edited Final ROM discussion points in ChromaDB collection `final_rom_<user_id>`
        with rich metadata (meeting_id, meeting_name, date, agenda_id, agenda_title, point_id,
        source='final_roim', version, speakers, action_owner, etc.).

        Guarantees that previous ChromaDB entries for this meeting are replaced/updated so there
        are never stale versions or duplicate entries.
        """
        from services.vector_store import get_final_rom_points_store
        from services.text_embedding_service import get_text_embedder

        agendas = []
        if isinstance(final_rom, dict):
            if "agendas" in final_rom:
                agendas = final_rom.get("agendas") or []
            elif "final_rom" in final_rom and isinstance(final_rom["final_rom"], dict):
                agendas = final_rom["final_rom"].get("agendas") or []
            elif "stage3" in final_rom and isinstance(final_rom["stage3"], dict):
                agendas = final_rom["stage3"].get("agendas") or []

        flat_points = []
        for a_idx, ag in enumerate(agendas):
            if not isinstance(ag, dict):
                continue
            ag_id = str(ag.get("agenda_id") or f"agenda_{a_idx}")
            ag_title = str(ag.get("title") or ag.get("agenda_title") or "").strip()
            pts = ag.get("discussion_points") or []
            for p in pts:
                flat_points.append({
                    "agenda_id": ag_id,
                    "agenda_title": ag_title,
                    "point": p,
                })

        if not user_id or not recording_id:
            if return_details:
                return {
                    "total_found": 0,
                    "already_indexed": 0,
                    "newly_indexed": 0,
                    "skipped_failed": 0,
                }
            return 0

        user_id = _validate_user_id(user_id)

        if not flat_points:
            if not only_missing:
                try:
                    store = get_final_rom_points_store(user_id)
                    if store.exists():
                        store.delete_by_filter("meeting_id", recording_id)
                except Exception as e:
                    logger.warning(f"[ROM Service] Failed to purge Final ROM vectors for {recording_id}: {e}")
            if return_details:
                return {
                    "total_found": 0,
                    "already_indexed": 0,
                    "newly_indexed": 0,
                    "skipped_failed": 0,
                }
            return 0

        embedder = get_text_embedder()
        embedder.load()
        dim = getattr(embedder, "_dim", 1024)
        store = get_final_rom_points_store(user_id, dim)
        store.load_or_create()

        already_indexed = 0
        skipped_failed = 0

        existing_point_ids = set()
        existing_final_ids = set()
        existing_chunk_ids = set()
        existing_texts = set()

        if only_missing:
            try:
                if store._collection is not None:
                    existing = store._collection.get(
                        where={"meeting_id": recording_id},
                        include=["metadatas", "documents"],
                    )
                    if existing:
                        for ex_id in (existing.get("ids") or []):
                            if ex_id:
                                existing_chunk_ids.add(ex_id)
                        for meta in (existing.get("metadatas") or []):
                            if meta:
                                pid = meta.get("point_id")
                                if pid:
                                    existing_point_ids.add(str(pid))
                                fid = meta.get("final_rom_identifier")
                                if fid:
                                    existing_final_ids.add(str(fid))
                        for doc in (existing.get("documents") or []):
                            if doc:
                                existing_texts.add(doc.strip())
            except Exception as e:
                logger.warning(f"[ROM Service] Failed to check existing Final ROM vectors for {recording_id}: {e}")
        else:
            try:
                store.delete_by_filter("meeting_id", recording_id)
            except Exception as e:
                logger.warning(f"[ROM Service] Failed to delete previous Final ROM vectors for {recording_id}: {e}")

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, item in enumerate(flat_points):
            ag_id = item["agenda_id"]
            ag_title = item["agenda_title"]
            p = item["point"]

            if isinstance(p, dict):
                pt_text = (p.get("text") or p.get("polished_text") or p.get("discussion_point") or "").strip()
                point_id = str(p.get("id") or "")
                spk_val = p.get("speakers") or p.get("speaker")
                if isinstance(spk_val, list):
                    spk_str = ", ".join(str(s) for s in spk_val if s)
                else:
                    spk_str = str(spk_val or "")
                action_owner = p.get("action_owner") or ""
            elif isinstance(p, str):
                pt_text = p.strip()
                point_id = ""
                spk_str = ""
                action_owner = ""
            else:
                pt_text = ""
                point_id = ""
                spk_str = ""
                action_owner = ""

            if not pt_text:
                skipped_failed += 1
                continue

            if not point_id:
                point_id = str(uuid.uuid4())

            final_id = f"final_{recording_id}_{point_id}"
            chunk_id = f"final_rom_{recording_id}_chunk_{idx}"
            doc_text = f"[{ag_title}] {pt_text}" if ag_title else pt_text

            if only_missing:
                is_already = (
                    (point_id and point_id in existing_point_ids)
                    or (final_id in existing_final_ids)
                    or (chunk_id in existing_chunk_ids)
                    or (doc_text in existing_texts)
                    or (pt_text in existing_texts)
                )
                if is_already:
                    already_indexed += 1
                    continue

            meta = {
                "user_id": user_id,
                "meeting_id": recording_id,
                "recording_id": recording_id,
                "doc_id": recording_id,
                "chunk_index": idx,
                "meeting_name": meeting_name or "Meeting",
                "date": meeting_date or "",
                "point_id": point_id,
                "agenda_id": ag_id,
                "agenda_title": ag_title,
                "stage": "final_roim",
                "source": "final_roim",
                "version": version or "long",
                "final_rom_identifier": final_id,
                "speakers": spk_str,
                "action_owner": action_owner,
                "clean_text": pt_text,
            }
            texts.append(doc_text)
            metadatas.append(meta)

        newly_indexed = 0
        if texts:
            try:
                vecs = embedder.encode_batch(texts)
                added_count = store.add(texts=texts, metadatas=metadatas, embeddings=vecs)
                newly_indexed = added_count
                logger.info(
                    f"[ROM Service] Indexed {added_count} Final ROM points for meeting '{meeting_name}' "
                    f"({recording_id}) in ChromaDB collection '{store._collection_name}'"
                )
            except Exception as e:
                logger.error(f"[ROM Service] Failed to index Final ROM points in ChromaDB: {e}", exc_info=True)
                skipped_failed += len(texts)

        if return_details:
            return {
                "total_found": len(flat_points),
                "already_indexed": already_indexed,
                "newly_indexed": newly_indexed,
                "skipped_failed": skipped_failed,
            }
        return newly_indexed


    def retrieve_previous_stage2_context(
        self,
        query: str,
        query_vec: np.ndarray,
        user_id: str,
        current_recording_id: str,
        previous_meeting_id: Optional[str] = None,
        top_k: int = 3,
        min_similarity_threshold: Optional[float] = 0.60,
        group_label: str = "",
    ) -> List[Dict]:
        """
        Retrieve previous Stage 2 points using ChromaDB embeddings + metadata filtering.
        - If previous_meeting_id is provided: retrieve only Stage 2 points from that meeting.
        - If previous_meeting_id is None (Auto Retrieve): retrieve top-K most similar Stage 2 points
          across all other meetings (meeting_id != current_recording_id).
        Ensure ONLY Stage 2 content is retrieved.
        """
        from services.vector_store import get_stage2_points_store
        from services.text_embedding_service import get_text_embedder

        if top_k <= 0 or not user_id:
            return []

        user_id = _validate_user_id(user_id)
        embedder = get_text_embedder()
        embedder.load()
        dim = getattr(embedder, "_dim", 1024)
        store = get_stage2_points_store(user_id, dim)

        fetch_k = max(top_k * 3, 10)
        score_thresh = min_similarity_threshold if (min_similarity_threshold is not None and min_similarity_threshold > 0.0) else 0.0
        prefix = f"[{group_label}] " if group_label else ""

        try:
            if previous_meeting_id and previous_meeting_id.strip():
                # Specific meeting filter
                target_mid = previous_meeting_id.strip()
                where_filter = {"meeting_id": target_mid}
                raw_results = store.search(query_vec, k=fetch_k, score_threshold=score_thresh, where=where_filter)
                logger.info(
                    f"[ROM Service] {prefix}Previous Stage 2 Context (Select Meeting '{target_mid}'): "
                    f"retrieved {len(raw_results)} candidate(s) (score_thresh={score_thresh})"
                )
            else:
                # Auto-retrieve across all other meetings
                where_filter = {"stage": "stage2"}
                raw_results = store.search(query_vec, k=fetch_k, score_threshold=score_thresh, where=where_filter)
                # Exclude current meeting points
                raw_results = [r for r in raw_results if r.get("meeting_id") != current_recording_id]
                logger.info(
                    f"[ROM Service] {prefix}Previous Stage 2 Context (Auto Retrieve): "
                    f"retrieved {len(raw_results)} candidate(s) across other meetings"
                )

            # Ensure only stage2 content is returned
            stage2_only = [
                r for r in raw_results
                if r.get("stage") == "stage2" or r.get("source") == "stage2_point" or "stage2" in str(r.get("stage2_identifier", ""))
            ]

            final_results = stage2_only[:top_k]
            return final_results
        except Exception as e:
            logger.warning(f"[ROM Service] {prefix}retrieve_previous_stage2_context failed: {e}", exc_info=True)
            return []

    def _format_previous_stage2_context(self, results: List[Dict]) -> str:
        """
        Format retrieved previous Stage 2 points into clear, structured context.
        """
        if not results:
            return ""
        blocks = []
        for r in results:
            m_name = r.get("meeting_name") or r.get("filename") or "Previous Meeting"
            m_date = r.get("date") or "Previous Date"
            txt = (r.get("_text") or r.get("text") or "").strip()
            if not txt:
                continue
            speakers = r.get("speakers")
            action_owner = r.get("action_owner")
            extra_info = []
            if speakers:
                extra_info.append(f"Speakers: {speakers}")
            if action_owner:
                extra_info.append(f"Action: {action_owner}")
            meta_suffix = f" ({', '.join(extra_info)})" if extra_info else ""
            blocks.append(f"[Previous Meeting: {m_name} | Date: {m_date}]\n- Stage 2 Point: {txt}{meta_suffix}")
        return "\n\n".join(blocks)

    def preview_stage2_context(
        self,
        discussion_points: List[Dict],
        recording_id: str,
        user_id: str,
        meeting_top_k: int = 5,
        global_top_k: int = 3,
        discussion_window_size: int = 5,
        min_similarity_threshold: Optional[float] = 0.80,
        process_all_together: bool = False,
        previous_meeting_mode: str = "auto",
        previous_meeting_id: Optional[str] = None,
        previous_meeting_top_k: int = 3,
        max_preview_groups: int = 3,
    ) -> Dict[str, Any]:
        """
        Stage 2: Preview the context retrieved for the initial discussion point groups without invoking LLM or mutating state.
        Returns a list of groups with their points, meeting context chunks, global context chunks, and previous meeting context chunks.
        """
        from services.text_embedding_service import get_text_embedder
        from services.vector_store import get_meeting_context_store, get_global_context_store
        from services.bm25 import BM25Index, _tokenize

        user_id = _validate_user_id(user_id)
        if not discussion_points:
            return {"groups": [], "total_groups": 0, "preview_groups_count": 0, "total_points": 0}

        embedder = get_text_embedder()
        embedder.load()

        dim = getattr(embedder, "_dim", 1024)
        meeting_store = get_meeting_context_store(recording_id, dim)
        global_store = get_global_context_store(user_id, dim)

        def _build_bm25(store) -> Optional[BM25Index]:
            try:
                store.load_or_create()
                metas = store._meta
                if not metas:
                    return None
                texts = [m.get("_text") or m.get("text") or "" for m in metas]
                return BM25Index.from_documents(texts, metas)
            except Exception as e:
                logger.warning(f"[ROM Service Preview] BM25 build failed: {e}")
                return None

        meeting_bm25 = _build_bm25(meeting_store)
        global_bm25 = _build_bm25(global_store)

        def _rrf_merge(result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
            rrf_scores: Dict[str, float] = {}
            best_entry: Dict[str, Dict] = {}
            for ranked in result_lists:
                for rank, entry in enumerate(ranked, start=1):
                    key = (entry.get("_text") or entry.get("text") or "").strip()
                    if not key:
                        continue
                    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                    if key not in best_entry or entry.get("score", 0) > best_entry[key].get("score", 0):
                        best_entry[key] = entry
            return sorted(best_entry.values(), key=lambda e: rrf_scores.get((e.get("_text") or e.get("text") or "").strip(), 0.0), reverse=True)

        def _diversity_dedup(results: List[Dict], sim_threshold: float = 0.92) -> List[Dict]:
            if len(results) <= 1:
                return results
            try:
                texts = [r.get("_text") or r.get("text") or "" for r in results]
                vecs = embedder.encode(texts)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1.0, norms)
                vecs = vecs / norms
                kept = []
                for i, (res, vec) in enumerate(zip(results, vecs)):
                    is_dup = False
                    for j in kept:
                        if np.dot(vec, vecs[j]) >= sim_threshold:
                            is_dup = True
                            break
                    if not is_dup:
                        kept.append(i)
                return [results[i] for i in kept]
            except Exception:
                seen: set = set()
                deduped = []
                for r in results:
                    txt = (r.get("_text") or r.get("text") or "").strip()
                    if txt and txt not in seen:
                        seen.add(txt)
                        deduped.append(r)
                return deduped

        def _hybrid_retrieve(query: str, query_vec: np.ndarray, store, bm25_idx: Optional[BM25Index], top_k: int, min_score: Optional[float] = None, store_name: str = "Context Store") -> List[Dict]:
            fetch_k = max(top_k * 3, 15)
            semantic_results = []
            try:
                score_thresh = min_score if (min_score is not None and min_score > 0.0) else 0.0
                if hasattr(store, "search_hybrid"):
                    semantic_results = store.search_hybrid(query_vec, query_text=query, k=fetch_k, score_threshold=score_thresh, expand_neighbors=True)
                else:
                    semantic_results = store.search(query_vec, k=fetch_k, score_threshold=score_thresh)
                for r in semantic_results:
                    r["_similarity_score"] = float(r.get("score", 0.0))
            except Exception as e:
                logger.warning(f"[ROM Service Preview] Semantic search failed for {store_name}: {e}")

            keyword_results: List[Dict] = []
            if bm25_idx is not None:
                try:
                    keyword_results = bm25_idx.search(query, k=fetch_k)
                except Exception as e:
                    logger.warning(f"[ROM Service Preview] BM25 search failed for {store_name}: {e}")

            metadata_results: List[Dict] = []
            try:
                query_tokens = _tokenize(query)
                all_metas = store._meta if hasattr(store, "_meta") else []
                metadata_results = BM25Index.metadata_search(query_tokens, all_metas, k=fetch_k)
            except Exception as e:
                logger.warning(f"[ROM Service Preview] Metadata search failed for {store_name}: {e}")

            fused = _rrf_merge([semantic_results, keyword_results, metadata_results])

            query_toks_set = set(_tokenize(query))
            for r in fused:
                boost = 0.0
                proj_names = r.get("project_names") or r.get("project_name") or []
                if isinstance(proj_names, str):
                    proj_names = [p.strip() for p in proj_names.split(",") if p.strip()]
                if any(p.lower() in query.lower() for p in proj_names if p):
                    boost += 0.10

                tech_terms = r.get("technical_terms") or r.get("technical_entities") or []
                if isinstance(tech_terms, str):
                    tech_terms = [t.strip() for t in tech_terms.split(",") if t.strip()]
                matches = sum(1 for t in tech_terms if t and t.lower() in query_toks_set)
                boost += min(0.15, matches * 0.05)

                heading_text = " ".join(filter(None, [
                    str(r.get("chapter") or r.get("main_topic") or ""),
                    str(r.get("section") or r.get("sub_topic") or ""),
                    str(r.get("heading") or r.get("topic") or ""),
                ])).lower()
                if heading_text:
                    h_toks = set(_tokenize(heading_text))
                    if h_toks.intersection(query_toks_set):
                        boost += 0.08

                doc_name = str(r.get("document_name") or r.get("filename") or "").lower()
                if doc_name and doc_name in query.lower():
                    boost += 0.10

                dates = r.get("dates") or r.get("date") or []
                if isinstance(dates, str):
                    dates = [d.strip() for d in dates.split(",") if d.strip()]
                if any(d.lower() in query.lower() for d in dates if d):
                    boost += 0.05

                r["_boost"] = boost
                if "_similarity_score" in r and r["_similarity_score"] is not None:
                    r["_similarity_score"] = float(r["_similarity_score"]) + boost

            fused = sorted(fused, key=lambda e: e.get("score", 0.0) + e.get("_boost", 0.0), reverse=True)
            diverse = _diversity_dedup(fused)

            selected_chunks = diverse
            if min_score is not None and min_score > 0.0:
                filtered = []
                for r in diverse:
                    sim_score = r.get("_similarity_score")
                    if sim_score is None:
                        txt = _extract_chunk_text(r)
                        if txt:
                            try:
                                c_vec = embedder.encode([txt])[0]
                                c_norm = np.linalg.norm(c_vec)
                                if c_norm > 1e-9:
                                    c_vec = c_vec / c_norm
                                sim_score = float(np.dot(query_vec, c_vec)) + r.get("_boost", 0.0)
                            except Exception:
                                sim_score = 0.0
                        else:
                            sim_score = 0.0
                        r["_similarity_score"] = sim_score
                        r["score"] = sim_score

                    if sim_score >= min_score:
                        filtered.append(r)
                selected_chunks = filtered

            return selected_chunks[:top_k]

        groups: List[Dict[str, Any]] = []

        if process_all_together:
            all_groups = [(0, discussion_points)]
        else:
            win_size = max(1, discussion_window_size)
            all_groups = []
            for w_idx, start_idx in enumerate(range(0, len(discussion_points), win_size)):
                all_groups.append((w_idx, discussion_points[start_idx : start_idx + win_size]))

        total_groups = len(all_groups)
        preview_limit = max_preview_groups if (max_preview_groups and max_preview_groups > 0) else total_groups
        preview_groups = all_groups[:preview_limit]

        logger.info(
            f"[ROM Service Preview] Starting Context Preview for initial {len(preview_groups)} of {total_groups} group(s) "
            f"({len(discussion_points)} total points, window_size={discussion_window_size if not process_all_together else 'All'}, "
            f"meeting_top_k={meeting_top_k}, global_top_k={global_top_k}, previous_mode='{previous_meeting_mode}')"
        )

        for w_idx, window in preview_groups:
            group_num = w_idx + 1
            logger.info(
                f"[ROM Service Preview] [Group {group_num}/{total_groups}] "
                f"Retrieving context for {len(window)} discussion point(s)..."
            )

            combined_query_parts = []
            for p in window:
                pt_text = str(p.get("discussion_point") or "")
                tech_terms = p.get("technical_terms")
                if isinstance(tech_terms, list):
                    clean_terms = [str(t).strip() for t in tech_terms if t is not None and str(t).strip()]
                    if clean_terms:
                        pt_text += " " + " ".join(clean_terms)
                combined_query_parts.append(pt_text)
            combined_query = " ".join(combined_query_parts).strip()

            query_vecs = embedder.encode_batch([combined_query])
            norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            query_vec = (query_vecs / norms)[0]

            meeting_results: List[Dict] = []
            global_results: List[Dict] = []
            previous_meeting_results: List[Dict] = []

            if meeting_top_k > 0:
                meeting_results = _hybrid_retrieve(
                    combined_query, query_vec, meeting_store, meeting_bm25, meeting_top_k,
                    min_score=min_similarity_threshold, store_name=f"Meeting Context (Group {group_num}/{total_groups})"
                )
            if global_top_k > 0:
                global_results = _hybrid_retrieve(
                    combined_query, query_vec, global_store, global_bm25, global_top_k,
                    min_score=min_similarity_threshold, store_name=f"Global Context (Group {group_num}/{total_groups})"
                )
            if previous_meeting_mode != "off" and previous_meeting_top_k > 0:
                previous_meeting_results = self.retrieve_previous_stage2_context(
                    query=combined_query,
                    query_vec=query_vec,
                    user_id=user_id,
                    current_recording_id=recording_id,
                    previous_meeting_id=previous_meeting_id if previous_meeting_mode == "select" else None,
                    top_k=previous_meeting_top_k,
                    min_similarity_threshold=min_similarity_threshold,
                    group_label=f"Group {group_num}/{total_groups}",
                )

            logger.info(
                f"[ROM Service Preview] [Group {group_num}/{total_groups}] "
                f"Done: matched {len(meeting_results)} meeting chunk(s), "
                f"{len(global_results)} global chunk(s), "
                f"{len(previous_meeting_results)} previous meeting point(s)"
            )

            groups.append({
                "group_index": group_num,
                "points_count": len(window),
                "query_snippet": combined_query[:200] + ("..." if len(combined_query) > 200 else ""),
                "points": [
                    {
                        "id": p.get("id"),
                        "discussion_point": p.get("discussion_point") or p.get("text", ""),
                        "timeline_start": p.get("timeline_start", 0),
                        "timeline_end": p.get("timeline_end", 0),
                        "speakers": p.get("speakers", []),
                        "action_owner": p.get("action_owner") or None,
                        "technical_terms": p.get("technical_terms", []),
                    }
                    for p in window
                ],
                "meeting_context": [
                    {
                        "filename": r.get("filename") or r.get("document_name") or "Meeting Document",
                        "text": _extract_chunk_text(r),
                        "score": round(float(r.get("_similarity_score") or r.get("score") or 0.0), 4),
                        "section": r.get("section") or r.get("heading") or None,
                    }
                    for r in meeting_results if _extract_chunk_text(r)
                ],
                "global_context": [
                    {
                        "filename": r.get("filename") or r.get("document_name") or "Global Document",
                        "text": _extract_chunk_text(r),
                        "score": round(float(r.get("_similarity_score") or r.get("score") or 0.0), 4),
                        "section": r.get("section") or r.get("heading") or None,
                    }
                    for r in global_results if _extract_chunk_text(r)
                ],
                "previous_meeting_context": [
                    {
                        "meeting_name": r.get("meeting_name") or r.get("filename") or "Previous Meeting",
                        "date": r.get("date") or "",
                        "text": (r.get("_text") or r.get("text") or "").strip(),
                        "score": round(float(r.get("score") or 0.0), 4),
                        "speakers": r.get("speakers") or "",
                        "action_owner": r.get("action_owner") or "",
                    }
                    for r in previous_meeting_results if (r.get("_text") or r.get("text") or "").strip()
                ],
            })

        logger.info(
            f"[ROM Service Preview] Completed Context Retrieval Preview for initial {len(groups)} group(s) "
            f"(out of {total_groups} total groups, {len(discussion_points)} total discussion points)"
        )

        return {
            "groups": groups,
            "total_groups": total_groups,
            "preview_groups_count": len(groups),
            "total_points": len(discussion_points),
        }

    def enhance_discussion_points(
        self,
        discussion_points: List[Dict],
        recording_id: str,
        user_id: str,
        meeting_top_k: int = 5,
        global_top_k: int = 3,
        discussion_window_size: int = 5,
        parallel_window_processing: Optional[int] = None,
        min_similarity_threshold: Optional[float] = 0.80,
        transcript: Optional[List[Dict]] = None,
        process_all_together: bool = False,
        separate_action_extraction: bool = False,
        reference_example_points: Optional[List[str]] = None,
        previous_meeting_mode: str = "auto",
        previous_meeting_id: Optional[str] = None,
        previous_meeting_top_k: int = 3,
        meeting_name: str = "",
        meeting_date: str = "",
        use_dspy_mode: bool = False,
    ) -> List[Dict]:
        """
        Stage 2: Enhance discussion points using hybrid retrieval (Semantic + BM25 + Metadata)
        with Reciprocal Rank Fusion, independent Meeting / Global context sources, and
        Previous Meeting Stage 2 Context (Dual-mode: Auto Retrieve or Select Meeting).
        """
        from services.ai_provider import get_provider
        from services.text_embedding_service import get_text_embedder, unload_text_embedder
        from services.vector_store import get_meeting_context_store, get_global_context_store
        from services.bm25 import BM25Index, _tokenize
        from config import settings
        from concurrent.futures import ThreadPoolExecutor, as_completed

        user_id = _validate_user_id(user_id)

        if not discussion_points:
            return []

        provider = get_provider()
        embedder = get_text_embedder()
        embedder.load()

        dim = getattr(embedder, "_dim", 1024)
        meeting_store = get_meeting_context_store(recording_id, dim)
        global_store = get_global_context_store(user_id, dim)

        # Build BM25 indexes lazily from the metadata sidecars
        def _build_bm25(store) -> Optional[BM25Index]:
            """Build a BM25 index over the FAISS store's metadata sidecar."""
            try:
                store.load_or_create()
                metas = store._meta  # list of dicts with _text key
                if not metas:
                    return None
                texts = [m.get("_text") or m.get("text") or "" for m in metas]
                return BM25Index.from_documents(texts, metas)
            except Exception as e:
                logger.warning(f"[ROM Service] BM25 index build failed: {e}")
                return None

        meeting_bm25 = _build_bm25(meeting_store)
        global_bm25 = _build_bm25(global_store)

        # RRF merge helper
        def _rrf_merge(
            result_lists: List[List[Dict]],
            k: int = 60,
        ) -> List[Dict]:
            rrf_scores: Dict[str, float] = {}
            best_entry: Dict[str, Dict] = {}

            for ranked in result_lists:
                for rank, entry in enumerate(ranked, start=1):
                    key = (entry.get("_text") or entry.get("text") or "").strip()
                    if not key:
                        continue
                    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                    if key not in best_entry or entry.get("score", 0) > best_entry[key].get("score", 0):
                        best_entry[key] = entry

            merged = sorted(
                best_entry.values(),
                key=lambda e: rrf_scores.get((e.get("_text") or e.get("text") or "").strip(), 0.0),
                reverse=True,
            )
            return merged

        # â”€â”€ Diversity deduplication (cosine sim â‰¥ 0.92 â†’ drop) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        def _diversity_dedup(
            results: List[Dict],
            sim_threshold: float = 0.92,
        ) -> List[Dict]:
            if len(results) <= 1:
                return results
            try:
                texts = [r.get("_text") or r.get("text") or "" for r in results]
                vecs = embedder.encode(texts)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1.0, norms)
                vecs = vecs / norms

                kept = []
                for i, (res, vec) in enumerate(zip(results, vecs)):
                    is_dup = False
                    for j in kept:
                        if np.dot(vec, vecs[j]) >= sim_threshold:
                            is_dup = True
                            break
                    if not is_dup:
                        kept.append(i)
                return [results[i] for i in kept]
            except Exception:
                seen: set = set()
                deduped = []
                for r in results:
                    txt = (r.get("_text") or r.get("text") or "").strip()
                    if txt and txt not in seen:
                        seen.add(txt)
                        deduped.append(r)
                return deduped

        # ── Retrieval helper for one store ─────────────────────────────────
        def _hybrid_retrieve(
            query: str,
            query_vec: np.ndarray,
            store,
            bm25_idx: Optional[BM25Index],
            top_k: int,
            min_score: Optional[float] = None,
            store_name: str = "Context Store",
        ) -> List[Dict]:
            fetch_k = max(top_k * 3, 15)
            store_dir = str(getattr(store, "_dir", "unknown"))
            total_vecs = getattr(getattr(store, "_index", None), "ntotal", 0)

            logger.info(
                f"[ROM Service Retrieval] {store_name}: Searching store at '{store_dir}' "
                f"(total_vectors_in_store={total_vecs}, top_k={top_k}, min_similarity_threshold={min_score})"
            )

            if total_vecs == 0:
                logger.warning(
                    f"[ROM Service Retrieval] {store_name} returned 0 chunks → REASON: "
                    f"Vector store is EMPTY (0 vectors loaded on disk at '{store_dir}'). "
                    f"No documents have been ingested into this vector store repository for user_id='{user_id}'."
                )
                return []

            semantic_results = []
            try:
                score_thresh = min_score if (min_score is not None and min_score > 0.0) else 0.0
                if hasattr(store, "search_hybrid"):
                    semantic_results = store.search_hybrid(query_vec, query_text=query, k=fetch_k, score_threshold=score_thresh, expand_neighbors=True)
                else:
                    semantic_results = store.search(query_vec, k=fetch_k, score_threshold=score_thresh)
                for r in semantic_results:
                    r["_similarity_score"] = float(r.get("score", 0.0))
            except Exception as e:
                logger.warning(f"[ROM Service] Semantic search failed for {store_name}: {e}")

            keyword_results: List[Dict] = []
            if bm25_idx is not None:
                try:
                    keyword_results = bm25_idx.search(query, k=fetch_k)
                except Exception as e:
                    logger.warning(f"[ROM Service] BM25 search failed for {store_name}: {e}")

            metadata_results: List[Dict] = []
            try:
                query_tokens = _tokenize(query)
                all_metas = store._meta if hasattr(store, "_meta") else []
                metadata_results = BM25Index.metadata_search(query_tokens, all_metas, k=fetch_k)
            except Exception as e:
                logger.warning(f"[ROM Service] Metadata search failed for {store_name}: {e}")

            fused = _rrf_merge([semantic_results, keyword_results, metadata_results])

            # Apply metadata relevance boosting to fused candidates
            query_toks_set = set(_tokenize(query))
            for r in fused:
                boost = 0.0

                # 1. Project name match (+0.10)
                proj_names = r.get("project_names") or r.get("project_name") or []
                if isinstance(proj_names, str):
                    proj_names = [p.strip() for p in proj_names.split(",") if p.strip()]
                if any(p.lower() in query.lower() for p in proj_names if p):
                    boost += 0.10

                # 2. Technical terms / entities overlap (+0.05 per match, max +0.15)
                tech_terms = r.get("technical_terms") or r.get("technical_entities") or []
                if isinstance(tech_terms, str):
                    tech_terms = [t.strip() for t in tech_terms.split(",") if t.strip()]
                matches = sum(1 for t in tech_terms if t and t.lower() in query_toks_set)
                boost += min(0.15, matches * 0.05)

                # 3. Topic / section / heading overlap (+0.08)
                heading_text = " ".join(filter(None, [
                    str(r.get("chapter") or r.get("main_topic") or ""),
                    str(r.get("section") or r.get("sub_topic") or ""),
                    str(r.get("heading") or r.get("topic") or ""),
                ])).lower()
                if heading_text:
                    h_toks = set(_tokenize(heading_text))
                    if h_toks.intersection(query_toks_set):
                        boost += 0.08

                # 4. Document name match (+0.10)
                doc_name = str(r.get("document_name") or r.get("filename") or "").lower()
                if doc_name and doc_name in query.lower():
                    boost += 0.10

                # 5. Date match (+0.05)
                dates = r.get("dates") or r.get("date") or []
                if isinstance(dates, str):
                    dates = [d.strip() for d in dates.split(",") if d.strip()]
                if any(d.lower() in query.lower() for d in dates if d):
                    boost += 0.05

                r["_boost"] = boost
                if "_similarity_score" in r and r["_similarity_score"] is not None:
                    r["_similarity_score"] = float(r["_similarity_score"]) + boost

            fused = sorted(fused, key=lambda e: e.get("score", 0.0) + e.get("_boost", 0.0), reverse=True)
            diverse = _diversity_dedup(fused)

            # Apply minimum similarity threshold filtering if configured (min_score is not None)
            selected_chunks = diverse
            if min_score is not None and min_score > 0.0:
                filtered = []
                for r in diverse:
                    sim_score = r.get("_similarity_score")
                    if sim_score is None:
                        txt = _extract_chunk_text(r)
                        if txt:
                            try:
                                c_vec = embedder.encode([txt])[0]
                                c_norm = np.linalg.norm(c_vec)
                                if c_norm > 1e-9:
                                    c_vec = c_vec / c_norm
                                sim_score = float(np.dot(query_vec, c_vec)) + r.get("_boost", 0.0)
                            except Exception:
                                sim_score = 0.0
                        else:
                            sim_score = 0.0
                        r["_similarity_score"] = sim_score
                        r["score"] = sim_score

                    if sim_score >= min_score:
                        filtered.append(r)
                selected_chunks = filtered

            final_results = selected_chunks[:top_k]

            # Diagnostic summary logging
            cand_scores = [round(r.get("_similarity_score", r.get("score", 0.0)), 4) for r in diverse]
            selected_scores = [round(r.get("_similarity_score", r.get("score", 0.0)), 4) for r in final_results]

            if not final_results:
                if not diverse:
                    logger.warning(
                        f"[ROM Service Retrieval] {store_name} returned 0 chunks → REASON: "
                        f"ChromaDB and BM25 found 0 matching candidates for query (store has {total_vecs} vectors at '{store_dir}')."
                    )
                else:
                    max_sc = max(cand_scores) if cand_scores else 0.0
                    logger.warning(
                        f"[ROM Service Retrieval] {store_name} returned 0 chunks → REASON: "
                        f"{len(diverse)} candidate chunk(s) found with scores {cand_scores}, but ALL were filtered out "
                        f"because they were below min_similarity_threshold={min_score} (highest score was {max_sc}). "
                        f"Try lowering min_similarity_threshold in settings."
                    )
            else:
                logger.info(
                    f"[ROM Service Retrieval] {store_name} returned {len(final_results)} chunk(s) "
                    f"out of {len(diverse)} candidate(s). Scores: {selected_scores}"
                )

            return final_results

        try:
            total_points = len(discussion_points)

            # ── "Process All Together" path ──────────────────────────────────────────
            if process_all_together:
                logger.info(
                    f"[ROM Service] Stage 2 'Process All Together' mode: {total_points} point(s) "
                    f"in a single LLM call"
                    f" | meeting_store_vecs={getattr(getattr(meeting_store, '_index', None), 'ntotal', 0)}"
                    f" | global_store_vecs={getattr(getattr(global_store, '_index', None), 'ntotal', 0)}"
                )

                # 1. Build combined query from ALL points
                all_query_parts = []
                for p in discussion_points:
                    pt_text = str(p.get("discussion_point") or "")
                    tech_terms = p.get("technical_terms")
                    if isinstance(tech_terms, list):
                        clean_terms = [str(t).strip() for t in tech_terms if t is not None and str(t).strip()]
                        if clean_terms:
                            pt_text += " " + " ".join(clean_terms)
                    all_query_parts.append(pt_text)
                combined_query = " ".join(all_query_parts).strip()

                query_vecs = embedder.encode_batch([combined_query])
                norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1.0, norms)
                query_vec = (query_vecs / norms)[0]

                # 2. Single retrieval pass
                meeting_results: List[Dict] = []
                global_results: List[Dict] = []
                previous_meeting_results: List[Dict] = []

                if meeting_top_k > 0:
                    meeting_results = _hybrid_retrieve(
                        combined_query, query_vec, meeting_store, meeting_bm25, meeting_top_k,
                        min_score=min_similarity_threshold, store_name="Meeting Context (All Together)"
                    )
                if global_top_k > 0:
                    global_results = _hybrid_retrieve(
                        combined_query, query_vec, global_store, global_bm25, global_top_k,
                        min_score=min_similarity_threshold, store_name="Global Context (All Together)"
                    )
                if previous_meeting_mode != "off" and previous_meeting_top_k > 0:
                    previous_meeting_results = self.retrieve_previous_stage2_context(
                        query=combined_query,
                        query_vec=query_vec,
                        user_id=user_id,
                        current_recording_id=recording_id,
                        previous_meeting_id=previous_meeting_id if previous_meeting_mode == "select" else None,
                        top_k=previous_meeting_top_k,
                        min_similarity_threshold=min_similarity_threshold,
                    )

                logger.info(
                    f"[ROM Service] Stage 2 'All Together' context retrieval: "
                    f"meeting_chunks={len(meeting_results)}, global_chunks={len(global_results)}, "
                    f"previous_meeting_chunks={len(previous_meeting_results)}"
                )

                # 3. Build context strings
                meeting_context_parts: List[str] = []
                for res in meeting_results:
                    txt = _extract_chunk_text(res)
                    fn = res.get("filename") or res.get("document_name") or "Meeting Document"
                    if txt:
                        meeting_context_parts.append(f"[{fn}]\n{txt}")
                global_context_parts: List[str] = []
                for res in global_results:
                    txt = _extract_chunk_text(res)
                    fn = res.get("filename") or res.get("document_name") or "Global Document"
                    if txt:
                        global_context_parts.append(f"[{fn}]\n{txt}")
                meeting_context_str = "\n\n".join(meeting_context_parts)
                global_context_str = "\n\n".join(global_context_parts)
                previous_meeting_context_str = self._format_previous_stage2_context(previous_meeting_results)

                # 4. Prepare all points JSON (streamlined: only id, discussion_point, speakers, action_owner)
                points_for_llm = json.dumps(
                    [
                        {
                            "id": p.get("id"),
                            "discussion_point": p.get("discussion_point"),
                            "speakers": p.get("speakers", []),
                            "action_owner": p.get("action_owner") or None,
                        }
                        for p in discussion_points
                    ],
                    ensure_ascii=False,
                )
                logger.info(
                    f"[Stage1\u2192Stage2] 'All Together': sending {total_points} point(s) to LLM "
                    f"(fields: id, discussion_point, speakers, action_owner)"
                )

                # 5. Single LLM call
                result = provider.enhance_rom_all_points_together(
                    points_json=points_for_llm,
                    meeting_context=meeting_context_str,
                    global_context=global_context_str,
                    previous_meeting_context=previous_meeting_context_str,
                    reference_example_points=reference_example_points,
                )
                enhanced_batch = result.get("enhanced_points", [])
                logger.info(
                    f"[Stage2→Schema] 'All Together': LLM returned {len(enhanced_batch)} point(s) "
                    f"from {total_points} input (fields: {list(enhanced_batch[0].keys()) if enhanced_batch else 'none'})"
                )

                # 6. Post-process: set polished_text from LLM's discussion_point, map metadata from Stage 1
                batch_by_id = {p.get("id"): p for p in discussion_points}
                polished_points: List[Dict] = []

                for ep in enhanced_batch:
                    ep["id"] = str(uuid.uuid4())

                    orig_ids = ep.get("original_point_ids")
                    if isinstance(orig_ids, str):
                        orig_ids = [orig_ids]
                    elif not isinstance(orig_ids, list):
                        orig_ids = []
                    if not orig_ids:
                        orig_ids = [p.get("id") for p in discussion_points if p.get("id")]

                    matched_points = [batch_by_id[oid] for oid in orig_ids if oid in batch_by_id]
                    if not matched_points:
                        matched_points = discussion_points
                        orig_ids = [p.get("id") for p in discussion_points if p.get("id")]

                    t_start = min(m.get("timeline_start", 0.0) for m in matched_points)
                    t_end = max(m.get("timeline_end", 0.0) for m in matched_points)

                    ep["original_point_ids"] = orig_ids
                    ep["original_point_id"] = orig_ids[0] if orig_ids else "N/A"
                    ep["timeline_start"] = t_start
                    ep["timeline_end"] = t_end

                    # ── Normalise text field: LLM now returns discussion_point; fall back for compat ──
                    ep_text = (
                        ep.get("discussion_point")
                        or ep.get("polished_text")
                        or ep.get("enhanced_text")
                        or ""
                    )
                    ep["polished_text"] = ep_text          # downstream field used by Stage 3
                    ep["discussion_point"] = ep_text        # keep alias for consistency
                    ep.pop("enhanced_text", None)           # remove old alias if present

                    # ── Reconstruct metadata from matched Stage 1 points (never from LLM) ──
                    ep.setdefault("technical_terms", list(dict.fromkeys(
                        t for m in matched_points for t in (m.get("technical_terms") or []) if t
                    )))
                    ep["dates"] = clean_calendar_dates(list(dict.fromkeys(
                        d for m in matched_points for d in (m.get("dates") or []) if d
                    )))
                    ep.setdefault("numbers", list(dict.fromkeys(
                        n for m in matched_points for n in (m.get("numbers") or []) if n
                    )))
                    ep.setdefault("references", list(dict.fromkeys(
                        r for m in matched_points for r in (m.get("references") or []) if r
                    )))

                    # Speakers, validated owners, action items and turn evidence come from
                    # the matched Stage 1 points (both modes), never from the LLM.
                    _carry_stage1_attribution(ep, matched_points, allow_llm_actions=separate_action_extraction)

                    ep["context_usage_report"] = {
                        "meeting_context_used": bool(meeting_context_str),
                        "global_context_used": bool(global_context_str),
                        "previous_meeting_context_used": bool(previous_meeting_context_str),
                        "context_added": bool(meeting_context_str or global_context_str or previous_meeting_context_str),
                        "documents": [],
                    }
                    ep["retrieved_context"] = {
                        "meeting_chunks": [
                            {
                                "text": _extract_chunk_text(r),
                                "score": float(r.get("score", 0.0)),
                                "filename": r.get("filename") or "Meeting Document",
                            }
                            for r in meeting_results
                        ],
                        "global_chunks": [
                            {
                                "text": _extract_chunk_text(r),
                                "score": float(r.get("score", 0.0)),
                                "filename": r.get("filename") or "Global Document",
                            }
                            for r in global_results
                        ],
                        "previous_meeting_chunks": [
                            {
                                "text": (r.get("_text") or r.get("text") or "").strip(),
                                "score": float(r.get("score", 0.0)),
                                "meeting_id": r.get("meeting_id") or "",
                                "meeting_name": r.get("meeting_name") or "Previous Meeting",
                                "date": r.get("date") or "",
                                "speakers": r.get("speakers") or "",
                                "action_owner": r.get("action_owner") or "",
                            }
                            for r in previous_meeting_results
                        ],
                        "context_usage_report": ep["context_usage_report"],
                    }

                    polished_points.append(ep)

                # ── Separate action-point processing for "All Together" mode ──────────
                if separate_action_extraction and polished_points:
                    polished_points = self._enhance_action_points_separately(
                        polished_points=polished_points,
                        discussion_points=discussion_points,
                        discussion_window_size=discussion_window_size,
                        provider=provider,
                        embedder=embedder,
                        meeting_store=meeting_store,
                        global_store=global_store,
                        meeting_bm25=meeting_bm25,
                        global_bm25=global_bm25,
                        meeting_top_k=meeting_top_k,
                        global_top_k=global_top_k,
                        min_similarity_threshold=min_similarity_threshold,
                        _hybrid_retrieve=_hybrid_retrieve,
                    )

            # Continue to validation and deduplication (same as windowed path)
            # — handled below after the windowed path block

            # ── Existing windowed path ───────────────────────────────────────────────
            else:
                total_windows = math.ceil(total_points / discussion_window_size) if total_points > 0 else 0

                # Determine parallel concurrency limit (default: 2, min: 1, max: 5)
                concurrency = parallel_window_processing
                if concurrency is None:
                    concurrency = getattr(settings, "ROM_PARALLEL_WINDOW_PROCESSING", 2)
                try:
                    concurrency = max(1, min(5, int(concurrency)))
                except (ValueError, TypeError):
                    concurrency = 2

                logger.info(
                    f"[ROM Service] Stage 2 starting: {total_points} point(s) across "
                    f"{total_windows} window(s) of size {discussion_window_size}"
                    f" | parallel_concurrency={concurrency}"
                    f" | meeting_store_vecs={getattr(getattr(meeting_store, '_index', None), 'ntotal', 0)}"
                    f" | global_store_vecs={getattr(getattr(global_store, '_index', None), 'ntotal', 0)}"
                )

                def _process_single_enhance_window(win_idx_and_window):
                    win_idx, window = win_idx_and_window
                    win_num = win_idx + 1
                    logger.info(
                        f"[ROM Service] Processing Stage 2 Window {win_num}/{total_windows} "
                        f"({len(window)} points) [Parallel limit={concurrency}]"
                    )

                    # 1. Combine window → single query
                    combined_query_parts = []
                    for p in window:
                        pt_text = str(p.get("discussion_point") or "")
                        tech_terms = p.get("technical_terms")
                        if isinstance(tech_terms, list):
                            clean_terms = [str(t).strip() for t in tech_terms if t is not None and str(t).strip()]
                            if clean_terms:
                                pt_text += " " + " ".join(clean_terms)
                        combined_query_parts.append(pt_text)

                    combined_query = " ".join(combined_query_parts).strip()

                    query_vecs = embedder.encode_batch([combined_query])
                    norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-9, 1.0, norms)
                    query_vec = (query_vecs / norms)[0]

                    # 2 & 3. Independent hybrid retrieval
                    meeting_results: List[Dict] = []
                    global_results: List[Dict] = []
                    previous_meeting_results: List[Dict] = []

                    if meeting_top_k > 0:
                        meeting_results = _hybrid_retrieve(
                            combined_query, query_vec, meeting_store, meeting_bm25, meeting_top_k,
                            min_score=min_similarity_threshold, store_name="Meeting Context"
                        )
                    else:
                        logger.info("[ROM Service Retrieval] Meeting Context: skipped (meeting_top_k=0)")

                    if global_top_k > 0:
                        global_results = _hybrid_retrieve(
                            combined_query, query_vec, global_store, global_bm25, global_top_k,
                            min_score=min_similarity_threshold, store_name="Global Context"
                        )
                    else:
                        logger.info("[ROM Service Retrieval] Global Context: skipped (global_top_k=0)")

                    if previous_meeting_mode != "off" and previous_meeting_top_k > 0:
                        previous_meeting_results = self.retrieve_previous_stage2_context(
                            query=combined_query,
                            query_vec=query_vec,
                            user_id=user_id,
                            current_recording_id=recording_id,
                            previous_meeting_id=previous_meeting_id if previous_meeting_mode == "select" else None,
                            top_k=previous_meeting_top_k,
                            min_similarity_threshold=min_similarity_threshold,
                        )

                    logger.info(
                        f"[ROM Service] Stage 2 Window {win_num}/{total_windows} context retrieval: "
                        f"meeting_chunks={len(meeting_results)}, global_chunks={len(global_results)}, "
                        f"previous_meeting_chunks={len(previous_meeting_results)} "
                        f"(meeting_top_k={meeting_top_k}, global_top_k={global_top_k}, "
                        f"min_similarity={min_similarity_threshold})"
                    )

                    # 4. Build context strings
                    meeting_context_parts: List[str] = []
                    meeting_filenames: List[str] = []
                    for res in meeting_results:
                        txt = _extract_chunk_text(res)
                        fn = res.get("filename") or res.get("document_name") or "Meeting Document"
                        if txt:
                            meeting_context_parts.append(f"[{fn}]\n{txt}")
                            if fn not in meeting_filenames:
                                meeting_filenames.append(fn)

                    global_context_parts: List[str] = []
                    global_filenames: List[str] = []
                    for res in global_results:
                        txt = _extract_chunk_text(res)
                        fn = res.get("filename") or res.get("document_name") or "Global Document"
                        if txt:
                            global_context_parts.append(f"[{fn}]\n{txt}")
                            if fn not in global_filenames:
                                global_filenames.append(fn)

                    meeting_context_str = "\n\n".join(meeting_context_parts)
                    global_context_str = "\n\n".join(global_context_parts)
                    previous_meeting_context_str = self._format_previous_stage2_context(previous_meeting_results)

                    # 5. Prepare window JSON for LLM (streamlined: only id, discussion_point, speakers, action_owner)
                    window_for_llm = json.dumps(
                        [
                            {
                                "id": p.get("id"),
                                "discussion_point": p.get("discussion_point"),
                                "speakers": p.get("speakers", []),
                                "action_owner": p.get("action_owner") or None,
                            }
                            for p in window
                        ],
                        ensure_ascii=False,
                    )
                    logger.info(
                        f"[Stage1\u2192Stage2] Window {win_num}/{total_windows}: sending {len(window)} point(s) to LLM "
                        f"(fields: id, discussion_point, speakers, action_owner)"
                    )

                    # 6. LLM call
                    result = provider.enhance_rom_discussion_window(
                        window_json=window_for_llm,
                        meeting_context=meeting_context_str,
                        global_context=global_context_str,
                        previous_meeting_context=previous_meeting_context_str,
                        separate_action_extraction=separate_action_extraction,
                        reference_example_points=reference_example_points,
                        use_dspy_mode=use_dspy_mode,
                    )
                    enhanced_batch = result.get("enhanced_points", [])
                    logger.info(
                        f"[Stage2\u2192Schema] Window {win_num}/{total_windows}: LLM returned {len(enhanced_batch)} point(s) "
                        f"from {len(window)} input (fields: {list(enhanced_batch[0].keys()) if enhanced_batch else 'none'})"
                    )
                    if not enhanced_batch:
                        logger.warning(
                            f"[ROM Service] Stage 2 Window {win_num}/{total_windows} got 0 enhanced points! "
                            f"Full result dict returned by provider: {result}"
                        )

                    # 7. Post-process enhanced batch: map metadata back from matched Stage 1 points
                    batch_by_id = {p.get("id"): p for p in window}
                    window_polished: List[Dict] = []

                    for ep in enhanced_batch:
                        ep["id"] = str(uuid.uuid4())

                        orig_ids = ep.get("original_point_ids")
                        if isinstance(orig_ids, str):
                            orig_ids = [orig_ids]
                        elif not isinstance(orig_ids, list):
                            orig_ids = []
                        if not orig_ids:
                            orig_ids = [p.get("id") for p in window if p.get("id")]

                        matched_points = [batch_by_id[oid] for oid in orig_ids if oid in batch_by_id]
                        if not matched_points:
                            matched_points = window
                            orig_ids = [p.get("id") for p in window if p.get("id")]

                        t_start = min(m.get("timeline_start", 0.0) for m in matched_points)
                        t_end = max(m.get("timeline_end", 0.0) for m in matched_points)

                        ep["original_point_ids"] = orig_ids
                        ep["original_point_id"] = orig_ids[0] if orig_ids else "N/A"
                        ep["timeline_start"] = t_start
                        ep["timeline_end"] = t_end

                        # ── Normalise text field: LLM now returns discussion_point; fall back for compat ──
                        ep_text = (
                            ep.get("discussion_point")
                            or ep.get("polished_text")
                            or ep.get("enhanced_text")
                            or ""
                        )
                        ep["polished_text"] = ep_text          # downstream field used by Stage 3
                        ep["discussion_point"] = ep_text        # keep alias for consistency
                        ep.pop("enhanced_text", None)           # remove old alias if present

                        # ── Reconstruct metadata from matched Stage 1 points (never from LLM) ──
                        ep["technical_terms"] = list(dict.fromkeys(
                            t for m in matched_points for t in (m.get("technical_terms") or []) if t
                        ))
                        ep["dates"] = clean_calendar_dates(list(dict.fromkeys(
                            d for m in matched_points for d in (m.get("dates") or []) if d
                        )))
                        ep["numbers"] = list(dict.fromkeys(
                            n for m in matched_points for n in (m.get("numbers") or []) if n
                        ))
                        ep["references"] = list(dict.fromkeys(
                            r for m in matched_points for r in (m.get("references") or []) if r
                        ))

                        # Speakers, validated owners, action items and turn evidence come from
                        # the matched Stage 1 points (both modes); the LLM's action items are
                        # used only when Stage 1 had none (separate mode, custom prompts).
                        _carry_stage1_attribution(ep, matched_points, allow_llm_actions=separate_action_extraction)


                        cur = ep.get("context_usage_report")
                        if not isinstance(cur, dict):
                            cur = {}

                        m_used = bool(cur.get("meeting_context_used", bool(meeting_context_str)))
                        g_used = bool(cur.get("global_context_used", bool(global_context_str)))
                        p_used = bool(previous_meeting_context_str)
                        c_added = bool(cur.get("context_added", cur.get("verified", False) or cur.get("technical_details_added", False) or cur.get("terminology_clarified", False))) or p_used

                        docs = cur.get("documents")
                        if not isinstance(docs, list):
                            docs = cur.get("meeting_context_docs", []) + cur.get("global_context_docs", [])
                            if not docs:
                                docs = meeting_filenames + global_filenames
                        docs = list(dict.fromkeys(str(d) for d in docs if d))

                        ep["context_usage_report"] = {
                            "meeting_context_used": m_used,
                            "global_context_used": g_used,
                            "previous_meeting_context_used": p_used,
                            "context_added": c_added,
                            "documents": docs,
                        }

                        ep["retrieved_context"] = {
                            "meeting_chunks": [
                                {
                                    "text": _extract_chunk_text(r),
                                    "score": float(r.get("score", 0.0)),
                                    "filename": r.get("filename") or r.get("document_name") or "Meeting Document",
                                }
                                for r in meeting_results
                            ],
                            "global_chunks": [
                                {
                                    "text": _extract_chunk_text(r),
                                    "score": float(r.get("score", 0.0)),
                                    "filename": r.get("filename") or r.get("document_name") or "Global Document",
                                }
                                for r in global_results
                            ],
                            "previous_meeting_chunks": [
                                {
                                    "text": (r.get("_text") or r.get("text") or "").strip(),
                                    "score": float(r.get("score", 0.0)),
                                    "meeting_id": r.get("meeting_id") or "",
                                    "meeting_name": r.get("meeting_name") or "Previous Meeting",
                                    "date": r.get("date") or "",
                                    "speakers": r.get("speakers") or "",
                                    "action_owner": r.get("action_owner") or "",
                                }
                                for r in previous_meeting_results
                            ],
                            "context_usage_report": ep["context_usage_report"],
                        }

                        window_polished.append(ep)

                    return win_idx, window_polished

                window_batches = []
                for win_start in range(0, total_points, discussion_window_size):
                    win_idx = len(window_batches)
                    window = discussion_points[win_start: win_start + discussion_window_size]
                    window_batches.append((win_idx, window))

                window_results_map: Dict[int, List[Dict]] = {}

                if concurrency == 1 or total_windows <= 1:
                    for item in window_batches:
                        win_idx = item[0]
                        try:
                            idx, pts = _process_single_enhance_window(item)
                            window_results_map[idx] = pts
                        except Exception as w_err:
                            logger.error(f"[ROM Service] Stage 2 Window {win_idx+1} failed ({w_err}). Continuing remaining windows...", exc_info=True)
                            # Fallback to original window points unenhanced
                            window_results_map[win_idx] = [
                                _unenhanced_stage2_point(p, separate_action_extraction) for p in item[1]
                            ]
                else:
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        future_to_win = {
                            executor.submit(_process_single_enhance_window, item): item[0]
                            for item in window_batches
                        }
                        for future in as_completed(future_to_win):
                            win_idx = future_to_win[future]
                            try:
                                idx, pts = future.result()
                                window_results_map[idx] = pts
                            except Exception as w_err:
                                logger.error(f"[ROM Service] Stage 2 Window {win_idx+1} failed ({w_err}). Continuing remaining windows...", exc_info=True)
                                window_results_map[win_idx] = [
                                    _unenhanced_stage2_point(p, separate_action_extraction)
                                    for item in window_batches if item[0] == win_idx for p in item[1]
                                ]

                # Re-assemble polished points preserving original window chronological order
                polished_points: List[Dict] = []
                for win_idx in range(len(window_batches)):
                    pts = window_results_map.get(win_idx, [])
                    polished_points.extend(pts)

                # ── Separate action-point processing for windowed path ─────────────────
                if separate_action_extraction and polished_points:
                    polished_points = self._enhance_action_points_separately(
                        polished_points=polished_points,
                        discussion_points=discussion_points,
                        discussion_window_size=discussion_window_size,
                        provider=provider,
                        embedder=embedder,
                        meeting_store=meeting_store,
                        global_store=global_store,
                        meeting_bm25=meeting_bm25,
                        global_bm25=global_bm25,
                        meeting_top_k=meeting_top_k,
                        global_top_k=global_top_k,
                        min_similarity_threshold=min_similarity_threshold,
                        _hybrid_retrieve=_hybrid_retrieve,
                    )

            # ── Stage 2 Validation Pass ──────────────────────────────────────────
            # Build a plain-text transcript for validation prompts (only if transcript was passed)
            validation_transcript_text: Optional[str] = None
            if transcript:
                try:
                    lines = []
                    for seg in transcript:
                        spk = seg.get("speaker_label") or seg.get("speaker") or "Speaker"
                        txt = (seg.get("text") or "").strip()
                        if txt:
                            lines.append(f"[{spk}]: {txt}")
                    validation_transcript_text = "\n".join(lines)
                except Exception as _te:
                    logger.warning(f"[ROM Service] Stage 2 Validation: failed to build transcript text: {_te}")

            if validation_transcript_text and polished_points:
                total_val_points = len(polished_points)
                logger.info(
                    f"[ROM Service] Stage 2 Validation Pass starting: {total_val_points} point(s) to validate"
                )

                valid_points: List[Dict] = []
                invalid_points: List[Dict] = []
                invalid_indices: List[int] = []  # original positions in polished_points

                for val_idx, p in enumerate(polished_points):
                    pt_text = p.get("polished_text") or p.get("enhanced_text") or ""
                    if not pt_text.strip():
                        # Empty text → treat as valid (keeps passthrough behaviour)
                        valid_points.append(p)
                        logger.info(
                            f"[ROM Service] Stage 2 Validation [{val_idx + 1}/{total_val_points}]: "
                            f"SKIP (empty text) → KEEP"
                        )
                        continue

                    try:
                        is_valid = provider.validate_stage2_point(pt_text, validation_transcript_text)
                    except Exception as _ve:
                        logger.warning(
                            f"[ROM Service] Stage 2 Validation [{val_idx + 1}/{total_val_points}]: "
                            f"LLM error ({_ve}) → defaulting to KEEP"
                        )
                        is_valid = True

                    verdict = "VALID" if is_valid else "INVALID"
                    logger.info(
                        f"[ROM Service] Stage 2 Validation [{val_idx + 1}/{total_val_points}]: {verdict} — "
                        f"{pt_text[:80]!r}{'...' if len(pt_text) > 80 else ''}"
                    )

                    if is_valid:
                        valid_points.append(p)
                    else:
                        invalid_points.append(p)
                        invalid_indices.append(val_idx)

                valid_count = len(valid_points)
                invalid_count = len(invalid_points)
                val_pct = round(valid_count / total_val_points * 100, 1) if total_val_points > 0 else 0.0

                logger.info(
                    f"[ROM Service] Stage 2 Validation complete: "
                    f"total={total_val_points} | validated={total_val_points} | "
                    f"valid={valid_count} | invalid={invalid_count} | "
                    f"validation_pct={val_pct}%"
                )

                if invalid_points:
                    logger.info(
                        f"[ROM Service] Stage 2 Correction Pass starting: "
                        f"{invalid_count} invalid point(s) to correct"
                    )
                    # Build compact JSON of invalid points for the LLM
                    invalid_for_llm = json.dumps(
                        [
                            {
                                "id": p.get("id"),
                                "polished_text": p.get("polished_text") or p.get("enhanced_text", ""),
                                "speakers": p.get("speakers", []),
                                "technical_terms": p.get("technical_terms", []),
                                "action_items": p.get("action_items", []),
                                "dates": p.get("dates", []),
                            }
                            for p in invalid_points
                        ],
                        ensure_ascii=False,
                    )
                    try:
                        corrected_list = provider.correct_invalid_stage2_points(
                            invalid_for_llm, validation_transcript_text
                        )
                    except Exception as _ce:
                        logger.warning(
                            f"[ROM Service] Stage 2 Correction Pass LLM error ({_ce}). "
                            "Falling back to keeping original invalid points."
                        )
                        corrected_list = []

                    correction_count = len(corrected_list)
                    logger.info(
                        f"[ROM Service] Stage 2 Correction Pass complete: "
                        f"{correction_count} point(s) corrected"
                    )

                    # Build a quick lookup: id → corrected point
                    corrected_by_id: Dict[str, Dict] = {}
                    for cp in corrected_list:
                        cid = cp.get("id")
                        if cid:
                            corrected_by_id[cid] = cp

                    # Reconstruct polished_points preserving original order:
                    # Replace each invalid point with its corrected version (or keep original if no correction)
                    final_points: List[Dict] = []
                    invalid_iter = iter(invalid_points)
                    for orig_idx, p in enumerate(polished_points):
                        if orig_idx in invalid_indices:
                            inv_p = next(invalid_iter)
                            pid = inv_p.get("id")
                            if pid and pid in corrected_by_id:
                                cp = corrected_by_id[pid]
                                # Merge: keep all original metadata, update text fields from correction
                                merged = dict(inv_p)
                                merged["polished_text"] = cp.get("polished_text") or inv_p.get("polished_text", "")
                                if cp.get("speakers"):
                                    merged["speakers"] = cp["speakers"]
                                if cp.get("technical_terms"):
                                    merged["technical_terms"] = cp["technical_terms"]
                                if cp.get("action_items"):
                                    merged["action_items"] = cp["action_items"]
                                if cp.get("dates"):
                                    merged["dates"] = cp["dates"]
                                final_points.append(merged)
                            else:
                                # No correction received for this point — keep original
                                final_points.append(inv_p)
                        else:
                            final_points.append(p)

                    polished_points = final_points
                    logger.info(
                        f"[ROM Service] Stage 2 Validation Pass fully complete: "
                        f"{len(polished_points)} final point(s) after correction "
                        f"(corrected={correction_count}, kept_as_is={invalid_count - correction_count})"
                    )
                else:
                    logger.info(
                        "[ROM Service] Stage 2 Validation Pass: no invalid points found — "
                        "skipping correction call."
                    )
            elif polished_points and not validation_transcript_text:
                logger.info(
                    "[ROM Service] Stage 2 Validation Pass: skipped (no transcript provided)."
                )

            # ── Stage 2 Final Semantic Deduplication Step (≥ 0.95 + LLM) ────
            if len(polished_points) > 1:
                try:
                    logger.info("[ROM Service] Stage 2 starting final semantic deduplication check (similarity threshold >= 0.95)")
                    texts = [p.get("polished_text", "") for p in polished_points]
                    embeddings = embedder.encode(texts)
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-9, 1.0, norms)
                    embeddings = embeddings / norms

                    sim_matrix = np.dot(embeddings, embeddings.T)
                    num_points = len(polished_points)

                    visited = [False] * num_points
                    clusters = []
                    for i in range(num_points):
                        if visited[i]:
                            continue
                        cluster = [i]
                        visited[i] = True
                        for j in range(i + 1, num_points):
                            if not visited[j] and sim_matrix[i, j] >= 0.95:
                                cluster.append(j)
                                visited[j] = True
                        clusters.append(cluster)

                    duplicate_clusters = [c for c in clusters if len(c) > 1]
                    logger.info(
                        f"[ROM Service] Stage 2 deduplication: Found {len(duplicate_clusters)} total group(s) of duplicate points "
                        f"(similarity >= 0.95, out of {len(polished_points)} points)"
                    )

                    deduped_points: List[Dict] = []

                    for cluster in clusters:
                        if len(cluster) == 1:
                            deduped_points.append(polished_points[cluster[0]])
                        else:
                            cluster_points = [polished_points[idx] for idx in cluster]
                            logger.info(
                                f"[ROM Service] Deduplicating candidate cluster of {len(cluster_points)} "
                                f"point(s) with similarity >= 0.95 via LLM"
                            )

                            cand_json = json.dumps(
                                [
                                    {
                                        "id": p.get("id"),
                                        "original_point_ids": p.get("original_point_ids", []),
                                        "polished_text": p.get("polished_text", ""),
                                        "speakers": p.get("speakers", []),
                                        "technical_terms": p.get("technical_terms", []),
                                        "dates": p.get("dates", []),
                                        "numbers": p.get("numbers", []),
                                        "action_items": p.get("action_items", []),
                                    }
                                    for p in cluster_points
                                ],
                                ensure_ascii=False,
                            )

                            eval_result = provider.deduplicate_rom_points(cand_json)
                            eval_type = eval_result.get("group_classification", "different")
                            res_points = eval_result.get("result_points", [])

                            if eval_type in ("duplicate", "complementary") and res_points:
                                all_orig_ids = list(
                                    dict.fromkeys(
                                        oid for p in cluster_points for oid in (p.get("original_point_ids") or [])
                                    )
                                )
                                all_speakers = list(
                                    dict.fromkeys(
                                        spk for p in cluster_points for spk in (p.get("speakers") or []) if spk
                                    )
                                )
                                min_start = min(p.get("timeline_start", 0.0) for p in cluster_points)
                                max_end = max(p.get("timeline_end", 0.0) for p in cluster_points)

                                for r_idx, r_pt in enumerate(res_points):
                                    r_pt["id"] = str(uuid.uuid4())
                                    r_pt["original_point_ids"] = all_orig_ids
                                    r_pt["original_point_id"] = all_orig_ids[0] if all_orig_ids else "N/A"
                                    r_pt["timeline_start"] = min_start
                                    r_pt["timeline_end"] = max_end
                                    r_pt["speakers"] = all_speakers
                                    r_pt["dates"] = clean_calendar_dates(
                                        r_pt.get("dates") or [d for p in cluster_points for d in (p.get("dates") or [])]
                                    )
                                    r_pt.setdefault("technical_terms", [])
                                    r_pt.setdefault("numbers", [])
                                    r_pt.setdefault("references", [])
                                    # Owners/speakers/evidence from all cluster members; their action
                                    # items go to the first result point only (no duplicates).
                                    _carry_stage1_attribution(
                                        r_pt, cluster_points,
                                        allow_llm_actions=separate_action_extraction, carry_actions=(r_idx == 0),
                                    )
                                    # Inherit context from the first cluster member
                                    r_pt["retrieved_context"] = cluster_points[0].get("retrieved_context", {})
                                    r_pt["context_usage_report"] = cluster_points[0].get("context_usage_report", {})
                                    # Ensure polished_text is present
                                    if "polished_text" not in r_pt and "enhanced_text" in r_pt:
                                        r_pt["polished_text"] = r_pt.pop("enhanced_text")
                                    deduped_points.append(r_pt)
                            else:
                                for p in cluster_points:
                                    deduped_points.append(p)

                    logger.info(
                        f"[ROM Service] Stage 2 final deduplication complete: "
                        f"{len(polished_points)} -> {len(deduped_points)} point(s)"
                    )
                    polished_points = deduped_points

                except Exception as e:
                    logger.warning(f"[ROM Service] Stage 2 final deduplication failed: {e}")

            logger.info(
                f"[ROM Service] Stage 2 complete: {len(polished_points)} enhanced point(s) "
                f"from {total_points} original point(s)"
            )
            if polished_points:
                sample = polished_points[0]
                logger.info(
                    f"[Stage2\u2192Stage3] Outgoing schema (sample): "
                    f"fields={list(sample.keys())} | "
                    f"polished_text={'present' if sample.get('polished_text') else 'MISSING'} | "
                    f"discussion_point={'present' if sample.get('discussion_point') else 'MISSING'} | "
                    f"action_owner={sample.get('action_owner')} | "
                    f"speakers={sample.get('speakers')}"
                )

                # Store Stage 2 points in ChromaDB for future cross-meeting context
                try:
                    self.index_stage2_points_in_chromadb(
                        recording_id=recording_id,
                        user_id=user_id,
                        points=polished_points,
                        meeting_name=meeting_name,
                        meeting_date=meeting_date,
                    )
                except Exception as e:
                    logger.warning(f"[ROM Service] Failed to auto-index Stage 2 points into ChromaDB: {e}")

            return polished_points

        finally:
            provider.unload_model()
            unload_text_embedder()
            gc.collect()

    def _enhance_action_points_separately(
        self,
        polished_points: List[Dict],
        discussion_points: List[Dict],
        discussion_window_size: int,
        provider,
        embedder,
        meeting_store,
        global_store,
        meeting_bm25,
        global_bm25,
        meeting_top_k: int,
        global_top_k: int,
        min_similarity_threshold: Optional[float],
        _hybrid_retrieve,
    ) -> List[Dict]:
        """
        Enhance action points separately using an expanded 2x window size.
        Reuses retrieved context from the corresponding discussion points
        and performs additional context retrieval for the expanded window range.
        """
        from services.ai_provider import is_prompt_configured
        if not is_prompt_configured("rom_enhance_action_points"):
            # No enhancement prompt (the default is empty): keep the transcript-validated
            # Stage 1 action items as they are instead of sending an empty prompt.
            logger.info("[ROM Service] Stage 2 action point enhancement skipped (no rom_enhance_action_points prompt configured)")
            return polished_points

        logger.info("[ROM Service] Stage 2 starting separate action point enhancement (2x window context)")

        action_window_size = discussion_window_size * 2
        total_pts = len(polished_points)
        action_window_batches = []
        for win_start in range(0, total_pts, action_window_size):
            win = polished_points[win_start : win_start + action_window_size]
            action_window_batches.append(win)

        for act_win_idx, win in enumerate(action_window_batches):
            window_action_items = []
            for p in win:
                acts = p.get("action_items") or []
                for a in acts:
                    norm = normalize_stage2_action_item(a)
                    if norm["task"]:
                        window_action_items.append({
                            "original_point_id": p.get("id"),
                            "task": norm["task"],
                            "assignee": norm["assignee"],
                            "deadline": norm["deadline"],
                            "speakers": p.get("speakers", [])
                        })

            if not window_action_items:
                continue

            # Gather context already retrieved for discussion points in this 2x window
            reused_meeting_chunks: List[Dict] = []
            reused_global_chunks: List[Dict] = []

            for p in win:
                rc = p.get("retrieved_context") or {}
                reused_meeting_chunks.extend(rc.get("meeting_chunks", []))
                reused_global_chunks.extend(rc.get("global_chunks", []))

            # Additional expanded retrieval for action points over 2x window
            act_query_text = " ".join([a["task"] for a in window_action_items if a.get("task")]).strip()

            expanded_meeting_chunks = list(reused_meeting_chunks)
            expanded_global_chunks = list(reused_global_chunks)

            if act_query_text:
                try:
                    q_vecs = embedder.encode_batch([act_query_text])
                    norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-9, 1.0, norms)
                    query_vec = (q_vecs / norms)[0]

                    if meeting_top_k > 0:
                        new_m_chunks = _hybrid_retrieve(
                            act_query_text, query_vec, meeting_store, meeting_bm25, meeting_top_k,
                            min_score=min_similarity_threshold, store_name="Action Point Meeting Context (2x Window)"
                        )
                        seen_m = {c.get("text") for c in expanded_meeting_chunks if c.get("text")}
                        for nc in new_m_chunks:
                            txt = _extract_chunk_text(nc)
                            if txt and txt not in seen_m:
                                seen_m.add(txt)
                                expanded_meeting_chunks.append({
                                    "text": txt,
                                    "score": float(nc.get("score", 0.0)),
                                    "filename": nc.get("filename") or nc.get("document_name") or "Meeting Document"
                                })

                    if global_top_k > 0:
                        new_g_chunks = _hybrid_retrieve(
                            act_query_text, query_vec, global_store, global_bm25, global_top_k,
                            min_score=min_similarity_threshold, store_name="Action Point Global Context (2x Window)"
                        )
                        seen_g = {c.get("text") for c in expanded_global_chunks if c.get("text")}
                        for nc in new_g_chunks:
                            txt = _extract_chunk_text(nc)
                            if txt and txt not in seen_g:
                                seen_g.add(txt)
                                expanded_global_chunks.append({
                                    "text": txt,
                                    "score": float(nc.get("score", 0.0)),
                                    "filename": nc.get("filename") or nc.get("document_name") or "Global Document"
                                })
                except Exception as ex:
                    logger.warning(f"[ROM Service] Expanded context retrieval for action points failed: {ex}")

            m_ctx_str = "\n\n".join([
                f"[{c.get('filename', 'Meeting Doc')}]\n{c.get('text', '')}"
                for c in expanded_meeting_chunks if c.get("text")
            ])
            g_ctx_str = "\n\n".join([
                f"[{c.get('filename', 'Global Doc')}]\n{c.get('text', '')}"
                for c in expanded_global_chunks if c.get("text")
            ])

            action_json = json.dumps(window_action_items, ensure_ascii=False)

            try:
                res = provider.enhance_rom_action_points(
                    action_points_json=action_json,
                    meeting_context=m_ctx_str,
                    global_context=g_ctx_str,
                )
                enhanced_groups = res.get("enhanced_action_points", [])

                enhanced_by_pid: Dict[str, List[Dict]] = {}
                for eg in enhanced_groups:
                    pid = eg.get("original_point_id")
                    items = eg.get("action_items") or []
                    norm_items = [normalize_stage2_action_item(it) for it in items if normalize_stage2_action_item(it)["task"]]
                    if pid and norm_items:
                        enhanced_by_pid.setdefault(pid, []).extend(norm_items)

                from services.action_owner_validator import task_similarity
                for p in win:
                    pid = p.get("id")
                    if pid in enhanced_by_pid:
                        # Enhancement may rewrite task wording, never the validated owner:
                        # copy owner + evidence from the best-matching original item.
                        originals = [normalize_stage2_action_item(a) for a in (p.get("action_items") or [])]
                        originals = [o for o in originals if o["task"]]
                        for enhanced in enhanced_by_pid[pid]:
                            if not originals:
                                break
                            best = max(originals, key=lambda o: task_similarity(o["task"], enhanced["task"]))
                            for key in ("assignee", "assigner") + ACTION_EVIDENCE_KEYS:
                                if best.get(key) is not None:
                                    enhanced[key] = best[key]
                                else:
                                    enhanced.pop(key, None)
                        p["action_items"] = enhanced_by_pid[pid]

            except Exception as e_act:
                logger.error(f"[ROM Service] Separate action point LLM enhancement failed: {e_act}", exc_info=True)

        return polished_points





    def _assign_points_to_agendas(
        self,
        provider,
        polished_points: List[Dict],
        agendas: List[Dict],
        expanded_agendas: Optional[List[Dict]],
        batch_size: int,
        log_prefix: str,
        include_point_timeline: bool,
        discussion_order_text: Optional[str] = None,
        timeline_guidance_text: Optional[str] = None,
        agenda_doc_points: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Phase 3 + 4: score every point against all agenda details with the active
        embedding model, then let the LLM assign agendas in batches with the top
        similarity candidates as additional reference.

        Similarity is a reference, not the decision: it is shown to the LLM, used as
        the fallback when the LLM returns no valid agenda, and only overrides a
        *low-confidence* LLM choice when another agenda is clearly closer.
        If embeddings are unavailable, mapping proceeds exactly as before (LLM only).
        """
        from services.text_embedding_service import get_text_embedder, unload_text_embedder

        valid_agenda_ids = {a["agenda_id"] for a in agendas}
        agenda_ref_json = json.dumps([_agenda_reference_entry(a) for a in agendas], ensure_ascii=False)

        # ── Phase 3: semantic similarity (active embedding model / trained variant) ──
        similarity: Optional[Dict[str, Any]] = None
        try:
            embedder = get_text_embedder()
            embedder.load()
            similarity = compute_point_agenda_similarity(
                polished_points, agendas, embedder,
                expanded_agendas=expanded_agendas,
                agenda_doc_points=agenda_doc_points,
            )
            logger.info(
                f"[{log_prefix}] Phase 3: similarity computed with '{embedder.loaded_model_name}' for "
                f"{len(polished_points)} point(s) × {len(agendas)} agenda(s)"
            )
        except Exception as e:
            logger.warning(f"[{log_prefix}] Phase 3: similarity unavailable, LLM-only mapping: {e}")
        finally:
            # Not needed during the LLM batches — free VRAM for the LLM.
            unload_text_embedder()

        candidates_by_pid: Dict[str, Dict] = {}
        scores_by_pid: Dict[str, Dict[str, float]] = {}
        if similarity:
            candidates_by_pid = {c["point_id"]: c for c in similarity["candidates"]}
            scores_by_pid = similarity["scores"]

        # ── Phase 4: batch LLM assignment ──
        batch_assignments: List[Dict] = []
        total_points = len(polished_points)
        total_batches = math.ceil(total_points / batch_size)

        for batch_num, batch_start in enumerate(range(0, total_points, batch_size), start=1):
            batch_points = polished_points[batch_start: batch_start + batch_size]
            logger.info(f"[{log_prefix}] Phase 4 batch {batch_num}/{total_batches}: {len(batch_points)} point(s)")

            batch_items = []
            for point in batch_points:
                pid = point["id"]
                item_dict = {
                    "point_id": pid,
                    "enhanced_point": point.get("polished_text") or point.get("text", ""),
                    "speakers": point.get("speakers", []),
                }
                if include_point_timeline:
                    item_dict["timeline"] = f"{_format_time_hhmm(point.get('timeline_start', 0))} – {_format_time_hhmm(point.get('timeline_end', 0))}"
                cand = candidates_by_pid.get(str(pid))
                if cand and cand["candidates"]:
                    item_dict["similarity_reference"] = [
                        {"agenda_id": c["agenda_id"], "score": round(c["score"], 3)} for c in cand["candidates"]
                    ]
                batch_items.append(item_dict)

            batch_json_str = json.dumps(batch_items, ensure_ascii=False)
            llm_result = provider.assign_agenda_batch(
                batch_json=batch_json_str,
                agenda_reference_json=agenda_ref_json,
                discussion_order_text=discussion_order_text,
                timeline_guidance_text=timeline_guidance_text,
                similarity_guidance_text=SIMILARITY_GUIDANCE_TEXT if similarity else None,
            )

            llm_map: Dict[str, Dict] = {a.get("point_id", ""): a for a in llm_result.get("assignments", []) if a.get("point_id")}

            for point in batch_points:
                pid = point["id"]
                llm_entry = llm_map.get(pid, {})

                assigned = llm_entry.get("assigned_agenda_id", "")
                confidence = llm_entry.get("confidence", "medium")
                reason = llm_entry.get("reason", "")
                is_probable = False
                similarity_override = False
                llm_assigned = assigned

                point_scores = scores_by_pid.get(str(pid), {})
                cand = candidates_by_pid.get(str(pid))
                top = cand["candidates"][0] if cand and cand["candidates"] else None

                if not assigned or assigned not in valid_agenda_ids:
                    is_probable = True
                    confidence = "probable"
                    if top:
                        assigned = top["agenda_id"]
                        reason = (
                            f"Assigned by semantic similarity (score {top['score']:.2f}, closest agenda detail: "
                            f"{top['best_facet'].replace('_', ' ')}) because the LLM did not return a valid agenda ID."
                        )
                    else:
                        reason = "Automatically assigned to default agenda (LLM did not return a valid agenda ID)."
                        assigned = agendas[0]["agenda_id"] if agendas else "A1"
                elif (
                    top
                    and str(confidence).lower() == "low"
                    and top["agenda_id"] != assigned
                    and top["score"] - point_scores.get(assigned, 0.0) >= _SIMILARITY_OVERRIDE_MARGIN
                ):
                    similarity_override = True
                    is_probable = True
                    reason = (
                        f"Similarity override: the LLM was uncertain ({reason or 'no reason given'}) and chose "
                        f"{assigned} (similarity {point_scores.get(assigned, 0.0):.2f}), but the point is clearly "
                        f"closer to {top['agenda_id']} (similarity {top['score']:.2f})."
                    )
                    assigned = top["agenda_id"]

                entry = {
                    "point_id": pid,
                    "assigned_agenda_id": assigned,
                    "confidence": "probable" if (is_probable and not similarity_override) else confidence,
                    "reason": reason,
                    "is_probable": is_probable,
                }
                if point_scores:
                    entry["similarity_score"] = point_scores.get(assigned, 0.0)
                    entry["similarity_top_agenda_id"] = top["agenda_id"] if top else None
                    entry["similarity_top_score"] = top["score"] if top else None
                if similarity_override:
                    entry["similarity_override"] = True
                    entry["llm_assigned_agenda_id"] = llm_assigned
                batch_assignments.append(entry)

            logger.info(f"[{log_prefix}] Phase 4 batch {batch_num}/{total_batches} complete")

        overrides = sum(1 for a in batch_assignments if a.get("similarity_override"))
        logger.info(
            f"[{log_prefix}] Phase 4 complete: {len(batch_assignments)} assignment(s)"
            + (f", {overrides} low-confidence assignment(s) corrected by similarity" if overrides else "")
        )

        return {
            "batch_assignments": batch_assignments,
            "candidate_results": similarity["candidates"] if similarity else [],
            "similarity_matrix": similarity["matrix"] if similarity else [],
            "similarity_agenda_ids": similarity["agenda_ids"] if similarity else [],
        }

    def generate_agendas_and_map(
        self,
        polished_points: List[Dict],
        agenda_text: Optional[str],
        recording_id: str,
        user_id: str,
        meeting_context_top_k: int = 5,
        global_context_top_k: int = 3,
        force_reextract: bool = False,
        transcript: Optional[List[Dict]] = None,
        previous_mom_texts: Optional[List[str]] = None,
        batch_size: int = 20,
    ) -> Dict:
        """
        Stage 3 â€“ 5-Phase Intelligent Pipeline
        ----------------------------------------
        Phase 1  Previous MoM Expansion (only if files uploaded)
        Phase 2  Per-agenda sequential RAG retrieval â†’ ExpandedAgenda representation
        Phase 3  Top-3 candidate retrieval via cosine similarity
        Phase 4  Batch LLM agenda assignment (batch_size pts/call)
                 Fallback: assign to best-scoring candidate, mark as 'probable'
        Phase 5  Agenda grouping â†’ structured output for final MoM generation
        """
        from services.ai_provider import get_provider
        from services.text_embedding_service import get_text_embedder, unload_text_embedder
        from services.vector_store import get_meeting_context_store, get_global_context_store
        from services.bm25 import BM25Index, _tokenize
        from services.rag_pipeline import (
            _load_parsed_agenda,
            _save_parsed_agenda,
            get_or_create_agenda_items,
        )

        user_id = _validate_user_id(user_id)

        if not polished_points:
            return {
                "agendas": [],
                "expanded_agendas": [],
                "candidate_results": [],
                "batch_assignments": [],
                "point_mappings": {},
                "agenda_groups": {},
                "similarity_matrix": [],
            }

        agendas: List[Dict] = []
        provider = get_provider()
        embedder = get_text_embedder()
        embedder.load()

        dim = getattr(embedder, "_dim", 1024)
        meeting_store = get_meeting_context_store(recording_id, dim)
        global_store  = get_global_context_store(user_id, dim)

        # â”€â”€ Build BM25 indexes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        def _build_bm25(store) -> Optional[BM25Index]:
            try:
                store.load_or_create()
                metas = store._meta
                if not metas:
                    return None
                texts = [m.get("_text") or m.get("text") or "" for m in metas]
                return BM25Index.from_documents(texts, metas)
            except Exception as e:
                logger.warning(f"[ROM S3] BM25 build failed: {e}")
                return None

        meeting_bm25 = _build_bm25(meeting_store)
        global_bm25  = _build_bm25(global_store)

        # â”€â”€ RRF merge helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        def _rrf_merge(result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
            rrf_scores: Dict[str, float] = {}
            best_entry: Dict[str, Dict]  = {}
            for ranked in result_lists:
                for rank, entry in enumerate(ranked, start=1):
                    key = (entry.get("_text") or entry.get("text") or "").strip()
                    if not key:
                        continue
                    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                    if key not in best_entry or entry.get("score", 0) > best_entry[key].get("score", 0):
                        best_entry[key] = entry
            return sorted(best_entry.values(),
                          key=lambda e: rrf_scores.get((e.get("_text") or e.get("text") or "").strip(), 0.0),
                          reverse=True)

        # â”€â”€ Diversity dedup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        def _diversity_dedup(results: List[Dict], threshold: float = 0.92) -> List[Dict]:
            if len(results) <= 1:
                return results
            try:
                texts = [r.get("_text") or r.get("text") or "" for r in results]
                vecs  = embedder.encode(texts)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1.0, norms)
                vecs  = vecs / norms
                kept: List[int] = []
                for i, (_, vec) in enumerate(zip(results, vecs)):
                    dup = any(np.dot(vec, vecs[j]) >= threshold for j in kept)
                    if not dup:
                        kept.append(i)
                return [results[i] for i in kept]
            except Exception:
                seen: set = set()
                deduped = []
                for r in results:
                    txt = (r.get("_text") or r.get("text") or "").strip()
                    if txt and txt not in seen:
                        seen.add(txt)
                        deduped.append(r)
                return deduped

        # â”€â”€ Hybrid retrieval for one store â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        def _hybrid_retrieve(query: str, query_vec: np.ndarray, store, bm25_idx, top_k: int) -> List[Dict]:
            fetch_k = max(top_k * 3, 15)
            try:
                sem = store.search(query_vec, k=fetch_k)
            except Exception:
                sem = []
            kw: List[Dict] = []
            if bm25_idx is not None:
                try:
                    kw = bm25_idx.search(query, k=fetch_k)
                except Exception:
                    pass
            meta: List[Dict] = []
            try:
                tokens   = _tokenize(query)
                all_meta = store._meta if hasattr(store, "_meta") else []
                meta     = BM25Index.metadata_search(tokens, all_meta, k=fetch_k)
            except Exception:
                pass
            fused   = _rrf_merge([sem, kw, meta])
            diverse = _diversity_dedup(fused)
            return diverse[:top_k]

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # Load / extract agendas (same logic as before)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        if not force_reextract:
            cached = _load_parsed_agenda(recording_id)
            if cached and isinstance(cached, list) and len(cached) > 0:
                logger.info(f"[ROM S3] Reusing {len(cached)} cached agenda item(s)")
                for i, item in enumerate(cached):
                    topic = item.get("topic") or item.get("title") or f"Agenda Topic {i+1}"
                    desc  = item.get("details") or item.get("description") or ""
                    agendas.append({
                        "agenda_id": item.get("agenda_id") or f"A{i+1}",
                        "title": topic,
                        "description": desc,
                        "speaker": item.get("speaker") or item.get("presenter"),
                        "keywords": item.get("keywords") if isinstance(item.get("keywords"), list) else [k for k in [topic, item.get("speaker")] if k],
                        "related_concepts": item.get("related_concepts") if isinstance(item.get("related_concepts"), list) else [],
                        "alternative_terminology": item.get("alternative_terminology") if isinstance(item.get("alternative_terminology"), list) else [],
                        "expected_themes": item.get("expected_themes") if isinstance(item.get("expected_themes"), list) else [],
                    })
                agendas = normalize_agendas(agendas, merge_duplicates=False)

        if not agendas:
            if agenda_text and agenda_text.strip():
                retrieved_context_str = ""
                try:
                    q_vec = embedder.encode_batch([agenda_text.strip()[:1000]])
                    norms = np.linalg.norm(q_vec, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-9, 1.0, norms)
                    q_vec = (q_vec / norms)[0]
                    parts: List[str] = []
                    if meeting_context_top_k > 0:
                        for res in meeting_store.search(q_vec, k=meeting_context_top_k):
                            txt = _extract_chunk_text(res)
                            if txt:
                                parts.append(txt)
                    if global_context_top_k > 0:
                        for res in global_store.search(q_vec, k=global_context_top_k):
                            txt = _extract_chunk_text(res)
                            if txt:
                                parts.append(txt)
                    if parts:
                        retrieved_context_str = "\n\n".join(parts)
                except Exception as e:
                    logger.warning(f"[ROM S3] Agenda-level RAG failed: {e}")

                logger.info("[ROM S3] Generating/refining agenda points via LLM from uploaded agenda text")
                agenda_result = provider.generate_rom_agendas(agenda_text, context=retrieved_context_str)
                raw_agendas   = normalize_agendas(agenda_result.get("agendas", []))
                if raw_agendas:
                    agendas = raw_agendas
                    parsed_to_save = []
                    for a in agendas:
                        if a.get("title"):
                            parsed_to_save.append({
                                "topic": a.get("title"),
                                "speaker": a.get("speaker") or a.get("presenter"),
                                "details": a.get("description") or a.get("details") or "",
                                "keywords": a.get("keywords") or [],
                                "related_concepts": a.get("related_concepts") or [],
                                "alternative_terminology": a.get("alternative_terminology") or [],
                                "expected_themes": a.get("expected_themes") or [],
                            })
                    if parsed_to_save:
                        _save_parsed_agenda(recording_id, user_id, parsed_to_save)

        if not agendas:
            logger.info("[ROM S3] No user agenda provided. Using default 'General Discussion' container.")
            agendas = [{
                "agenda_id": "A1",
                "title": "General Discussion",
                "description": "General meeting discussion points",
                "keywords": [],
                "related_concepts": [],
                "alternative_terminology": [],
                "expected_themes": [],
                "is_default_agenda": True
            }]

        logger.info(f"[ROM S3] {len(agendas)} agenda(s) loaded. Starting 5-phase pipeline.")

        try:
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 1 â€“ Previous MoM Expansion (only when files uploaded)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            previous_mom_expansion: Dict[str, Dict] = {}
            has_previous_mom = bool(previous_mom_texts and any(t.strip() for t in previous_mom_texts))

            if has_previous_mom:
                logger.info("[ROM S3] Phase 1: Expanding agendas with previous MoM documents")
                combined_mom = "\n\n---\n\n".join(t for t in previous_mom_texts if t and t.strip())
                agenda_list_for_expansion = json.dumps(
                    [{"agenda_id": a["agenda_id"], "title": a["title"], "description": a.get("description", "")} for a in agendas],
                    ensure_ascii=False
                )
                expansion_result = provider.expand_agendas_with_previous_mom(agenda_list_for_expansion, combined_mom)
                for item in expansion_result.get("expansions", []):
                    aid = item.get("agenda_id")
                    if aid:
                        previous_mom_expansion[aid] = item
                logger.info(f"[ROM S3] Phase 1 complete: {len(previous_mom_expansion)} agenda(s) expanded from previous MoMs")
            else:
                logger.info("[ROM S3] Phase 1: Skipped (no previous MoM files uploaded)")

            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 2 â€“ Per-agenda sequential RAG retrieval â†’ ExpandedAgenda
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            logger.info("[ROM S3] Phase 2: Per-agenda context expansion via RAG (sequential)")
            expanded_agendas: List[Dict] = []

            for agenda in agendas:
                aid   = agenda["agenda_id"]
                title = agenda.get("title", "")
                desc  = agenda.get("description", "")
                kws   = agenda.get("keywords", [])
                rc    = agenda.get("related_concepts", [])
                alt   = agenda.get("alternative_terminology", [])
                themes = agenda.get("expected_themes", [])

                # Build retrieval query for this agenda
                agenda_query = " ".join(filter(None, [title, desc] + kws + rc + alt + themes))

                meeting_ctx_str = ""
                global_ctx_str  = ""

                if agenda_query.strip():
                    try:
                        q_vecs = embedder.encode_batch([agenda_query[:1000]])
                        norms  = np.linalg.norm(q_vecs, axis=1, keepdims=True)
                        norms  = np.where(norms < 1e-9, 1.0, norms)
                        q_vec  = (q_vecs / norms)[0]

                        if meeting_context_top_k > 0:
                            m_results = _hybrid_retrieve(agenda_query, q_vec, meeting_store, meeting_bm25, meeting_context_top_k)
                            meeting_parts = [_extract_chunk_text(r) for r in m_results if _extract_chunk_text(r)]
                            meeting_ctx_str = "\n\n".join(meeting_parts)

                        if global_context_top_k > 0:
                            g_results = _hybrid_retrieve(agenda_query, q_vec, global_store, global_bm25, global_context_top_k)
                            global_parts = [_extract_chunk_text(r) for r in g_results if _extract_chunk_text(r)]
                            global_ctx_str = "\n\n".join(global_parts)

                    except Exception as e:
                        logger.warning(f"[ROM S3] Phase 2 RAG failed for agenda {aid}: {e}")

                # Previous MoM expansion for this agenda
                prev_expansion = previous_mom_expansion.get(aid, {})
                expanded = {
                    "agenda_id": aid,
                    "title": title,
                    "description": desc,
                    "keywords": kws,
                    "related_concepts": rc,
                    "alternative_terminology": alt,
                    "expected_themes": themes,
                    "previous_meeting_summary": prev_expansion if prev_expansion.get("found_in_previous_meeting") else {},
                    "meeting_context_snippet": meeting_ctx_str[:1500] if meeting_ctx_str else "",
                    "global_context_snippet": global_ctx_str[:1000] if global_ctx_str else "",
                }
                expanded_agendas.append(expanded)
                logger.info(f"[ROM S3] Phase 2 expanded agenda {aid}: meeting_ctx={len(meeting_ctx_str)} chars, global_ctx={len(global_ctx_str)} chars, prev_mom={'yes' if prev_expansion.get('found_in_previous_meeting') else 'no'}")

            # ═════════════════════════════════════════════════════════════════
            # PHASE 4 – Batch LLM Agenda Assignment (Direct Full Agenda Mapping)
            # ═════════════════════════════════════════════════════════════════
            logger.info(f"[ROM S3] Phase 3-4: similarity-referenced LLM agenda assignment for {len(polished_points)} point(s) against {len(agendas)} agenda(s) (batch_size={batch_size})")
            mapping_result = self._assign_points_to_agendas(
                provider=provider,
                polished_points=polished_points,
                agendas=agendas,
                expanded_agendas=expanded_agendas,
                batch_size=batch_size,
                log_prefix="ROM S3",
                include_point_timeline=True,
            )
            batch_assignments: List[Dict] = mapping_result["batch_assignments"]
            candidate_results: List[Dict] = mapping_result["candidate_results"]
            sim_matrix_list: List[List[float]] = mapping_result["similarity_matrix"]

            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 5 â€“ Agenda Grouping
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            logger.info("[ROM S3] Phase 5: Grouping discussion points by assigned agenda")
            agenda_groups: Dict[str, List[str]] = {a["agenda_id"]: [] for a in agendas}
            point_mappings: Dict[str, str]       = {}

            for assignment in batch_assignments:
                pid = assignment["point_id"]
                aid = assignment["assigned_agenda_id"]
                point_mappings[pid] = aid
                if aid in agenda_groups:
                    agenda_groups[aid].append(pid)
                else:
                    # Shouldn't happen after fallback, but guard anyway
                    fallback_id = agendas[0]["agenda_id"] if agendas else "A1"
                    point_mappings[pid] = fallback_id
                    agenda_groups.setdefault(fallback_id, []).append(pid)

            logger.info(f"[ROM S3] Phase 5 complete. Groups: { {k: len(v) for k, v in agenda_groups.items()} }")

            # Strip internal embedding key before returning
            for ea in expanded_agendas:
                ea.pop("_embedding_text", None)

            return {
                "agendas": agendas,
                "expanded_agendas": expanded_agendas,
                "candidate_results": candidate_results,
                "batch_assignments": batch_assignments,
                "point_mappings": point_mappings,
                "agenda_groups": agenda_groups,
                "similarity_matrix": sim_matrix_list,  # points × agendas (DOCX download)
                "similarity_agenda_ids": mapping_result["similarity_agenda_ids"],
            }

        finally:
            provider.unload_model()
            unload_text_embedder()
            gc.collect()


    def generate_agendas_only(
        self,
        agenda_text: Optional[str],
        recording_id: str,
        user_id: str,
        meeting_context_top_k: int = 5,
        global_context_top_k: int = 3,
        force_reextract: bool = False,
        transcript: Optional[List[Dict]] = None,
        previous_mom_texts: Optional[List[str]] = None,
        polished_points: Optional[List[Dict]] = None,
        analyze_discussion: bool = False,
    ) -> Dict:
        """
        Stage 3 â€“ Step 1: Generate agendas only (no point mapping).
        Returns agendas + expanded agenda context chunks.
        Phases 1 & 2 from the original pipeline.

        With ``analyze_discussion`` the Stage 2 points are analysed first (grouped,
        de-duplicated, classified, summarised into validated topics; see
        services/agenda_analysis.py). Without an agenda document or cached agendas the
        topics become the agendas; otherwise topics are matched to the existing agendas
        and those that match none are returned as ``suggested_agendas``.
        """
        from services.ai_provider import get_provider
        from services.text_embedding_service import get_text_embedder, unload_text_embedder
        from services.vector_store import get_meeting_context_store, get_global_context_store
        from services.bm25 import BM25Index, _tokenize
        from services.rag_pipeline import (
            _load_parsed_agenda,
            _save_parsed_agenda,
            get_or_create_agenda_items,
        )

        user_id = _validate_user_id(user_id)

        agendas: List[Dict] = []
        provider = get_provider()
        embedder = get_text_embedder()
        embedder.load()

        dim = getattr(embedder, "_dim", 1024)
        meeting_store = get_meeting_context_store(recording_id, dim)
        global_store  = get_global_context_store(user_id, dim)

        def _build_bm25(store):
            try:
                store.load_or_create()
                metas = store._meta
                if not metas:
                    return None
                texts = [m.get("_text") or m.get("text") or "" for m in metas]
                return BM25Index.from_documents(texts, metas)
            except Exception as e:
                logger.warning(f"[ROM S3-Agenda] BM25 build failed: {e}")
                return None

        meeting_bm25 = _build_bm25(meeting_store)
        global_bm25  = _build_bm25(global_store)

        def _rrf_merge(result_lists, k=60):
            rrf_scores: Dict[str, float] = {}
            best_entry: Dict[str, Dict] = {}
            for ranked in result_lists:
                for rank, entry in enumerate(ranked, start=1):
                    key = (entry.get("_text") or entry.get("text") or "").strip()
                    if not key:
                        continue
                    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                    if key not in best_entry or entry.get("score", 0) > best_entry[key].get("score", 0):
                        best_entry[key] = entry
            return sorted(best_entry.values(),
                          key=lambda e: rrf_scores.get((e.get("_text") or e.get("text") or "").strip(), 0.0),
                          reverse=True)

        def _diversity_dedup(results, threshold=0.92):
            if len(results) <= 1:
                return results
            try:
                texts = [r.get("_text") or r.get("text") or "" for r in results]
                vecs  = embedder.encode(texts)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1.0, norms)
                vecs  = vecs / norms
                kept: List[int] = []
                for i, (_, vec) in enumerate(zip(results, vecs)):
                    dup = any(np.dot(vec, vecs[j]) >= threshold for j in kept)
                    if not dup:
                        kept.append(i)
                return [results[i] for i in kept]
            except Exception:
                seen: set = set()
                deduped = []
                for r in results:
                    txt = (r.get("_text") or r.get("text") or "").strip()
                    if txt and txt not in seen:
                        seen.add(txt)
                        deduped.append(r)
                return deduped

        def _hybrid_retrieve(query, query_vec, store, bm25_idx, top_k):
            fetch_k = max(top_k * 3, 15)
            try:
                sem = store.search(query_vec, k=fetch_k)
            except Exception:
                sem = []
            kw: List[Dict] = []
            if bm25_idx is not None:
                try:
                    kw = bm25_idx.search(query, k=fetch_k)
                except Exception:
                    pass
            meta: List[Dict] = []
            try:
                tokens   = _tokenize(query)
                all_meta = store._meta if hasattr(store, "_meta") else []
                meta     = BM25Index.metadata_search(tokens, all_meta, k=fetch_k)
            except Exception:
                pass
            fused   = _rrf_merge([sem, kw, meta])
            diverse = _diversity_dedup(fused)
            return diverse[:top_k]

        try:
            # â”€â”€ Load / extract agendas (same logic as generate_agendas_and_map) â”€â”€
            if not force_reextract:
                cached = _load_parsed_agenda(recording_id)
                if cached and isinstance(cached, list) and len(cached) > 0:
                    logger.info(f"[ROM S3-Agenda] Reusing {len(cached)} cached agenda item(s)")
                    for i, item in enumerate(cached):
                        topic = item.get("topic") or item.get("title") or f"Agenda Topic {i+1}"
                        desc  = item.get("details") or item.get("description") or ""
                        agendas.append({
                            "agenda_id": item.get("agenda_id") or f"A{i+1}",
                            "title": topic,
                            "description": desc,
                            "speaker": item.get("speaker") or item.get("presenter"),
                            "keywords": item.get("keywords") if isinstance(item.get("keywords"), list) else [k for k in [topic, item.get("speaker")] if k],
                            "related_concepts": item.get("related_concepts") if isinstance(item.get("related_concepts"), list) else [],
                            "alternative_terminology": item.get("alternative_terminology") if isinstance(item.get("alternative_terminology"), list) else [],
                            "expected_themes": item.get("expected_themes") if isinstance(item.get("expected_themes"), list) else [],
                        })
                    agendas = normalize_agendas(agendas, merge_duplicates=False)

            if not agendas:
                if agenda_text and agenda_text.strip():
                    retrieved_context_str = ""
                    try:
                        q_vec = embedder.encode_batch([agenda_text.strip()[:1000]])
                        norms = np.linalg.norm(q_vec, axis=1, keepdims=True)
                        norms = np.where(norms < 1e-9, 1.0, norms)
                        q_vec = (q_vec / norms)[0]
                        parts: List[str] = []
                        if meeting_context_top_k > 0:
                            for res in meeting_store.search(q_vec, k=meeting_context_top_k):
                                txt = _extract_chunk_text(res)
                                if txt:
                                    parts.append(txt)
                        if global_context_top_k > 0:
                            for res in global_store.search(q_vec, k=global_context_top_k):
                                txt = _extract_chunk_text(res)
                                if txt:
                                    parts.append(txt)
                        if parts:
                            retrieved_context_str = "\n\n".join(parts)
                    except Exception as e:
                        logger.warning(f"[ROM S3-Agenda] RAG failed: {e}")

                    logger.info("[ROM S3-Agenda] Generating agenda points via LLM from uploaded agenda text")
                    agenda_result = provider.generate_rom_agendas(agenda_text, context=retrieved_context_str)
                    raw_agendas   = normalize_agendas(agenda_result.get("agendas", []))
                    if raw_agendas:
                        agendas = raw_agendas
                        parsed_to_save = []
                        for a in agendas:
                            if a.get("title"):
                                parsed_to_save.append({
                                    "topic": a.get("title"),
                                    "speaker": a.get("speaker") or a.get("presenter"),
                                    "details": a.get("description") or a.get("details") or "",
                                    "keywords": a.get("keywords") or [],
                                    "related_concepts": a.get("related_concepts") or [],
                                    "alternative_terminology": a.get("alternative_terminology") or [],
                                    "expected_themes": a.get("expected_themes") or [],
                                })
                        if parsed_to_save:
                            _save_parsed_agenda(recording_id, user_id, parsed_to_save)

            # Discussion analysis: what was actually discussed, grouped into validated topics.
            analysis: Optional[Dict[str, Any]] = None
            if analyze_discussion and polished_points:
                from services.agenda_analysis import analyze_discussion as _analyze_discussion, topics_to_agendas
                try:
                    analysis = _analyze_discussion(provider, embedder, polished_points, log_prefix="ROM S3-Analysis")
                except Exception as e:
                    logger.warning(f"[ROM S3-Analysis] Discussion analysis failed, continuing without it: {e}", exc_info=True)
                    analysis = None
                if analysis and not agendas and analysis.get("topics"):
                    agendas = topics_to_agendas(analysis["topics"])
                    analysis["mode"] = "derived_agendas"
                    analysis["topic_matches"] = [
                        {"topic_id": t["topic_id"], "topic_title": t["title"], "status": "derived",
                         "matched_agenda_id": a["agenda_id"], "matched_agenda_title": a["title"]}
                        for t, a in zip(analysis["topics"], agendas)
                    ]
                    logger.info(f"[ROM S3-Analysis] No agenda document: {len(agendas)} agenda(s) built from the discussion")

            if not agendas:
                logger.info("[ROM S3-Agenda] No user agenda provided. Using default 'General Discussion' container.")
                agendas = [{
                    "agenda_id": "A1",
                    "title": "General Discussion",
                    "description": "General meeting discussion points",
                    "keywords": [],
                    "related_concepts": [],
                    "alternative_terminology": [],
                    "expected_themes": [],
                    "is_default_agenda": True
                }]

            logger.info(f"[ROM S3-Agenda] {len(agendas)} agenda(s) loaded. Running Phase 1 & 2.")

            # Phase 1 – Previous MoM Expansion
            previous_mom_expansion: Dict[str, Dict] = {}
            has_previous_mom = bool(previous_mom_texts and any(t.strip() for t in previous_mom_texts))
            if has_previous_mom:
                logger.info("[ROM S3-Agenda] Phase 1: Expanding agendas with previous MoM documents")
                combined_mom = "\n\n---\n\n".join(t for t in previous_mom_texts if t and t.strip())
                agenda_list_for_expansion = json.dumps(
                    [{"agenda_id": a["agenda_id"], "title": a["title"], "description": a.get("description", "")} for a in agendas],
                    ensure_ascii=False
                )
                expansion_result = provider.expand_agendas_with_previous_mom(agenda_list_for_expansion, combined_mom)
                for item in expansion_result.get("expansions", []):
                    aid = item.get("agenda_id")
                    if aid:
                        previous_mom_expansion[aid] = item
                logger.info(f"[ROM S3-Agenda] Phase 1 complete: {len(previous_mom_expansion)} agenda(s) expanded")
            else:
                logger.info("[ROM S3-Agenda] Phase 1: Skipped (no previous MoM files uploaded)")

            # Phase 2 – Per-agenda RAG retrieval
            logger.info("[ROM S3-Agenda] Phase 2: Per-agenda context expansion via RAG")
            expanded_agendas: List[Dict] = []
            retrieved_context_chunks: Dict[str, str] = {}

            for agenda in agendas:
                aid   = agenda["agenda_id"]
                title = agenda.get("title", "")
                desc  = agenda.get("description", "")
                kws   = agenda.get("keywords", [])
                rc    = agenda.get("related_concepts", [])
                alt   = agenda.get("alternative_terminology", [])
                themes = agenda.get("expected_themes", [])

                agenda_query = " ".join(filter(None, [title, desc] + kws + rc + alt + themes))

                meeting_ctx_str = ""
                global_ctx_str  = ""

                if agenda_query.strip():
                    try:
                        q_vecs = embedder.encode_batch([agenda_query[:1000]])
                        norms  = np.linalg.norm(q_vecs, axis=1, keepdims=True)
                        norms  = np.where(norms < 1e-9, 1.0, norms)
                        q_vec  = (q_vecs / norms)[0]

                        if meeting_context_top_k > 0:
                            m_results = _hybrid_retrieve(agenda_query, q_vec, meeting_store, meeting_bm25, meeting_context_top_k)
                            meeting_parts = [_extract_chunk_text(r) for r in m_results if _extract_chunk_text(r)]
                            meeting_ctx_str = "\n\n".join(meeting_parts)

                        if global_context_top_k > 0:
                            g_results = _hybrid_retrieve(agenda_query, q_vec, global_store, global_bm25, global_context_top_k)
                            global_parts = [_extract_chunk_text(r) for r in g_results if _extract_chunk_text(r)]
                            global_ctx_str = "\n\n".join(global_parts)

                    except Exception as e:
                        logger.warning(f"[ROM S3-Agenda] Phase 2 RAG failed for agenda {aid}: {e}")

                prev_expansion = previous_mom_expansion.get(aid, {})
                expanded = {
                    "agenda_id": aid,
                    "title": title,
                    "description": desc,
                    "keywords": kws,
                    "related_concepts": rc,
                    "alternative_terminology": alt,
                    "expected_themes": themes,
                    "previous_meeting_summary": prev_expansion if prev_expansion.get("found_in_previous_meeting") else {},
                    "meeting_context_snippet": meeting_ctx_str[:1500] if meeting_ctx_str else "",
                    "global_context_snippet": global_ctx_str[:1000] if global_ctx_str else "",
                }
                expanded_agendas.append(expanded)
                retrieved_context_chunks[aid] = meeting_ctx_str[:1500] if meeting_ctx_str else ""

            # Topics vs existing agendas: similarity scores with evidence; unmatched topics are suggested.
            suggested_agendas: List[Dict[str, Any]] = []
            if analysis is not None and analysis.get("mode") != "derived_agendas":
                analysis["mode"] = "matched_existing" if not agendas[0].get("is_default_agenda") else "default_agenda"
                if analysis["mode"] == "matched_existing" and analysis.get("topics"):
                    from services.agenda_analysis import build_suggested_agendas, match_topics_to_agendas
                    points_by_id = {str(p.get("id")): p for p in polished_points or [] if p.get("id")}
                    try:
                        analysis["topic_matches"] = match_topics_to_agendas(
                            analysis["topics"], agendas, embedder, points_by_id, expanded_agendas,
                        )
                        suggested_agendas = build_suggested_agendas(analysis["topics"], analysis["topic_matches"])
                        logger.info(
                            f"[ROM S3-Analysis] {len(analysis['topics'])} topic(s) matched against {len(agendas)} agenda(s): "
                            f"{len(suggested_agendas)} unmatched topic(s) suggested as new agendas"
                        )
                    except Exception as e:
                        logger.warning(f"[ROM S3-Analysis] Topic matching failed: {e}", exc_info=True)

            logger.info(f"[ROM S3-Agenda] Complete: {len(agendas)} agendas ready for use.")
            return {
                "agendas": agendas,
                "expanded_agendas": expanded_agendas,
                "retrieved_context_chunks": retrieved_context_chunks,
                "discussion_analysis": analysis,
                "suggested_agendas": suggested_agendas,
            }

        finally:
            provider.unload_model()
            unload_text_embedder()
            gc.collect()

    def extract_agenda_document_points(
        self,
        agenda_id: str,
        agenda_title: str,
        agenda_description: str,
        document_text: str,
    ) -> Dict:
        """
        Extract 2-5 factual, agenda-specific points from a supporting document
        uploaded for an agenda item. These points are not used for retrieval/context.
        """
        from services.ai_provider import get_provider

        if not document_text or not document_text.strip():
            return {"agenda_id": agenda_id, "points": []}

        provider = get_provider()
        try:
            result = provider.generate_agenda_document_points(
                agenda_title=agenda_title,
                agenda_description=agenda_description,
                document_text=document_text,
            )
            points = result.get("points", [])
            presenter = result.get("presenter")
            logger.info(f"[ROM S3-DocPoints] Agenda {agenda_id}: {len(points)} point(s) extracted from document, presenter={presenter}")
            return {"agenda_id": agenda_id, "points": points, "presenter": presenter}
        except Exception as e:
            logger.error(f"[ROM S3-DocPoints] Failed to extract document points for agenda {agenda_id}: {e}")
            return {"agenda_id": agenda_id, "points": [], "presenter": None}
        finally:
            provider.unload_model()
            gc.collect()

    def map_points_to_agendas(
        self,
        polished_points: List[Dict],
        agendas: List[Dict],
        expanded_agendas: List[Dict],
        recording_id: str,
        user_id: str,
        meeting_context_top_k: int = 5,
        global_context_top_k: int = 3,
        batch_size: int = 20,
        include_agenda_doc_points: bool = False,
        agenda_doc_points: Optional[Dict[str, Any]] = None,
        discussion_order: Optional[List[str]] = None,
        agenda_timeline: Optional[Dict[str, Dict[str, float]]] = None,
        point_types: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict:
        """
        Stage 3 – Step 2: Map discussion points to agendas (Phases 3-5).
        Each point is scored against all details of every agenda with the active embedding
        model (trained variant if selected); the LLM assigns agendas with those scores as
        additional reference. Optionally merges agenda document points into the final ROM.
        Accepts optional discussion_order (e.g. ['A1', 'A3', 'A2']) and agenda_timeline (ranges)
        as soft contextual guidance for the LLM.
        """
        from services.ai_provider import get_provider
        from services.text_embedding_service import unload_text_embedder

        user_id = _validate_user_id(user_id)

        if not polished_points:
            return {
                "candidate_results": [],
                "batch_assignments": [],
                "point_mappings": {},
                "agenda_groups": {},
                "similarity_matrix": [],
                "final_rom_agendas": [],
            }

        if not agendas:
            return {
                "candidate_results": [],
                "batch_assignments": [],
                "point_mappings": {},
                "agenda_groups": {},
                "similarity_matrix": [],
                "final_rom_agendas": [],
            }

        provider = get_provider()

        # Format discussion order guidance text if provided
        agenda_title_map = {a.get("agenda_id", ""): a.get("title", "") for a in agendas}
        discussion_order_text: Optional[str] = None
        if discussion_order:
            order_parts = []
            for idx, aid in enumerate(discussion_order, start=1):
                title = agenda_title_map.get(aid, "")
                order_parts.append(f"{idx}. {aid} ({title})" if title else f"{idx}. {aid}")
            discussion_order_text = " -> ".join(order_parts)

        # Format timeline guidance text if provided
        timeline_guidance_text: Optional[str] = None
        if agenda_timeline and isinstance(agenda_timeline, dict):
            timeline_lines = []
            for aid, trange in agenda_timeline.items():
                title = agenda_title_map.get(aid, "")
                start_s = float(trange.get("start_sec", 0.0) if isinstance(trange, dict) else 0.0)
                end_s = float(trange.get("end_sec", 0.0) if isinstance(trange, dict) else 0.0)
                t_str = f"{_format_time_hhmm(start_s)} – {_format_time_hhmm(end_s)}"
                title_str = f" ({title})" if title else ""
                timeline_lines.append(f"- Agenda {aid}{title_str}: approximate window {t_str}")
            if timeline_lines:
                timeline_guidance_text = "\n".join(timeline_lines)

        try:
            # ── Phase 3-4: similarity scoring + batch LLM agenda assignment ──
            logger.info(
                f"[ROM S3-Map] Phase 3-4: similarity-referenced LLM agenda assignment for {len(polished_points)} point(s) against "
                f"{len(agendas)} agenda(s) (batch_size={batch_size}, has_order={bool(discussion_order_text)}, "
                f"has_timeline={bool(timeline_guidance_text)})"
            )
            mapping_result = self._assign_points_to_agendas(
                provider=provider,
                polished_points=polished_points,
                agendas=agendas,
                expanded_agendas=expanded_agendas,
                batch_size=batch_size,
                log_prefix="ROM S3-Map",
                include_point_timeline=bool(timeline_guidance_text),
                discussion_order_text=discussion_order_text,
                timeline_guidance_text=timeline_guidance_text,
                agenda_doc_points=agenda_doc_points if include_agenda_doc_points else None,
            )
            batch_assignments: List[Dict] = mapping_result["batch_assignments"]
            candidate_results: List[Dict] = mapping_result["candidate_results"]
            sim_matrix_list: List[List[float]] = mapping_result["similarity_matrix"]


            # â”€â”€ Phase 5: Agenda Grouping â”€â”€
            logger.info("[ROM S3-Map] Phase 5: Grouping discussion points by assigned agenda")
            agenda_groups: Dict[str, List[str]] = {a["agenda_id"]: [] for a in agendas}
            point_mappings: Dict[str, str]       = {}

            for assignment in batch_assignments:
                pid = assignment["point_id"]
                aid = assignment["assigned_agenda_id"]
                point_mappings[pid] = aid
                if aid in agenda_groups:
                    agenda_groups[aid].append(pid)
                else:
                    fallback_id = agendas[0]["agenda_id"] if agendas else "A1"
                    point_mappings[pid] = fallback_id
                    agenda_groups.setdefault(fallback_id, []).append(pid)

            logger.info(f"[ROM S3-Map] Phase 5 complete. Groups: { {k: len(v) for k, v in agenda_groups.items()} }")

            # â”€â”€ Build final_rom agendas â”€â”€
            assign_map: dict = {a["point_id"]: a for a in batch_assignments if a.get("point_id")}
            final_agendas = []

            for a in agendas:
                agenda_points = []
                for p in polished_points:
                    if point_mappings.get(p.get("id")) == a.get("agenda_id"):
                        p_copy = dict(p)
                        p_copy["text"] = p.get("polished_text") or p.get("text", "")
                        assignment = assign_map.get(p.get("id"), {})
                        p_copy["assignment_confidence"] = assignment.get("confidence", "medium")
                        p_copy["assignment_reason"]     = assignment.get("reason", "")
                        p_copy["is_probable"]           = assignment.get("is_probable", False)
                        p_type = (point_types or {}).get(str(p.get("id")))
                        if isinstance(p_type, dict) and p_type.get("primary_type"):
                            p_copy["point_type"] = p_type["primary_type"]
                            p_copy["point_types"] = p_type.get("types") or [p_type["primary_type"]]
                        agenda_points.append(p_copy)

                # Merge in agenda document points if requested
                agenda_id = a.get("agenda_id", "")
                if include_agenda_doc_points and agenda_doc_points and agenda_id in agenda_doc_points:
                    doc_entry = agenda_doc_points.get(agenda_id, {})
                    doc_pts   = doc_entry.get("points", []) if isinstance(doc_entry, dict) else []
                    doc_name  = doc_entry.get("doc_name", "Supporting Document") if isinstance(doc_entry, dict) else "Supporting Document"
                    for dp_text in doc_pts:
                        if dp_text and str(dp_text).strip():
                            agenda_points.append({
                                "id": str(uuid.uuid4()),
                                "text": str(dp_text).strip(),
                                "polished_text": str(dp_text).strip(),
                                "speaker": "From Document",
                                "speakers": ["From Document"],
                                "action_owner": None,
                                "decisions": [],
                                "action_items": [],
                                "technical_terms": [],
                                "dates": [],
                                "numbers": [],
                                "references": [doc_name],
                                "timeline_start": 0.0,
                                "timeline_end": 0.0,
                                "is_doc_point": True,
                                "assignment_confidence": "high",
                                "assignment_reason": f"Extracted from uploaded document: {doc_name}",
                                "is_probable": False,
                            })

                # Always preserve all agendas from original list, even if zero points
                agenda_copy = dict(a)
                agenda_copy["discussion_points"] = agenda_points
                final_agendas.append(agenda_copy)

            return {
                "candidate_results": candidate_results,
                "batch_assignments": batch_assignments,
                "point_mappings": point_mappings,
                "agenda_groups": agenda_groups,
                "similarity_matrix": sim_matrix_list,
                "similarity_agenda_ids": mapping_result["similarity_agenda_ids"],
                "final_rom_agendas": final_agendas,
            }

        finally:
            provider.unload_model()
            unload_text_embedder()
            gc.collect()

    def generate_mom_from_enhanced_rom(
        self,
        polished_points: List[Dict],
        recording_meta: Dict,
        recording_id: str = "",
        user_id: str = "",
        separate_action_extraction: bool = False,
        action_chunk_size: Optional[int] = None,
    ) -> Dict:
        """
        Generate Minutes of Meeting (MOM) using Stage 2/Stage 3 Enhanced Discussion Points.
        1. LLM is passed the points ONLY to generate Introduction and Conclusion.
        2. Key Discussion Points are constructed directly from each enhanced discussion point.
        3. Action Items: structured action_items carried on the points (validated against the
           transcript in Stage 1) are used in both modes. Only in embedded mode
           (separate_action_extraction=False) and only when no point carries any, a final
           extract_actions_from_enhanced_points() pass extracts embedded actions from polished_text.
           Owners are never widened to the speakers of a point.
        """
        from services.ai_provider import get_provider

        if user_id:
            user_id = _validate_user_id(user_id)

        filename = recording_meta.get("filename", "Meeting MoM")
        date_str = str(recording_meta.get("created_at", "") or "")
        dur_str = str(recording_meta.get("duration", "") or "")
        participants = recording_meta.get("speakers_detected", [])
        if isinstance(participants, str):
            try:
                import json as _j
                participants = _j.loads(participants)
            except Exception:
                participants = [p.strip() for p in participants.split(",") if p.strip()]

        if not polished_points:
            return {
                "title": filename,
                "date": date_str,
                "duration": dur_str,
                "participants": participants if isinstance(participants, list) else [],
                "introduction": "No enhanced discussion points available.",
                "points_discussed": [],
                "action_items": [],
                "conclusion": "",
            }

        # 1. Build points_discussed as clean strings: "Topic: Summary"
        points_discussed: list[str] = []
        points_text_lines = []

        for idx, p in enumerate(polished_points, start=1):
            pt_text = (p.get("polished_text") or p.get("text") or p.get("discussion_point") or "").strip()
            if not pt_text:
                continue

            points_discussed.append(pt_text)
            points_text_lines.append(f"Point {idx}: {pt_text}")

        # 2. Extract action items depending on mode
        action_items = []

        # ── Owner resolution helper ─────────────────────────────────────────
        # Only an explicitly extracted owner, or the source point's single validated
        # action_owner, can own an action. Speakers who merely took part in the
        # discussion are never owners.
        from services.action_owner_validator import canonicalize_owner_list
        from services.rom_transcript_format import build_roster
        roster = build_roster([{"speaker_label": p} for p in participants]) if isinstance(participants, list) else []

        def _resolve_point_owner(act_owner: Optional[str], src_point: Optional[Dict]) -> str:
            names = [n for n, _ in canonicalize_owner_list(act_owner, roster)]
            if names:
                return ", ".join(names)
            point_owners = [n for n, _ in canonicalize_owner_list((src_point or {}).get("action_owner"), roster)]
            if len(point_owners) == 1:
                return point_owners[0]
            return "Unassigned"

        def _structured_actions_from_points() -> List[Dict]:
            items = []
            for p in polished_points:
                raw_acts = p.get("action_items")
                if not raw_acts:
                    continue
                if not isinstance(raw_acts, list):
                    raw_acts = [raw_acts]

                dates = p.get("dates")
                d_fallback = None
                if dates:
                    d_str = ", ".join(dates) if isinstance(dates, list) else str(dates)
                    if d_str.strip() and d_str.strip().lower() not in ("none", "n/a", "null"):
                        d_fallback = d_str.strip()

                for act in raw_acts:
                    act_dict = normalize_action_item(act)
                    if not act_dict["task"]:
                        continue
                    displayed_task = format_action_point_display_text(act_dict) or act_dict["task"]
                    items.append({
                        "task": displayed_task,
                        "item": displayed_task,
                        "description": displayed_task,
                        "owner": _resolve_point_owner(act_dict.get("assignee"), p),
                        "owner_confidence": act_dict.get("owner_confidence"),
                        "deadline": act_dict.get("deadline") or d_fallback or "ASAP",
                        "status": "open",
                        "raw_json": act_dict,
                    })
            return items

        # Structured action items validated against the transcript in Stage 1 (and
        # carried through Stage 2) are preferred in both modes. The re-extraction
        # pass over polished prose only runs in embedded mode when none exist.
        action_items = _structured_actions_from_points()

        if not action_items and not separate_action_extraction:
            logger.info(f"[ROM Service] MoM generation: running extract_actions_from_enhanced_points (chunk_size={action_chunk_size or 'default'})")
            provider = get_provider()
            try:
                raw_extracted = provider.extract_actions_from_enhanced_points(
                    polished_points, chunk_size=action_chunk_size, participants=participants if isinstance(participants, list) else None
                )
            except Exception as _ex:
                logger.warning(f"[ROM Service] MoM action extraction pass failed ({_ex}). Continuing with no action items.")
                raw_extracted = []
            finally:
                provider.unload_model()

            point_by_id = {p.get("id"): p for p in polished_points if p.get("id")}

            for act in raw_extracted:
                task = str(act.get("task") or act.get("item") or act.get("description") or "").strip()
                if not task:
                    continue

                src_id = act.get("source_point_id") or act.get("id")
                src_point = point_by_id.get(src_id) if src_id else None
                raw_owner = act.get("owner") or act.get("assignee") or act.get("person") or act.get("action_owner")
                owner_str = _resolve_point_owner(raw_owner, src_point)

                deadline = str(act.get("deadline") or act.get("due_date") or "ASAP").strip()
                if deadline.lower() in ("none", "n/a", "null", ""):
                    deadline = "ASAP"

                expected_outcome = act.get("expected_outcome") or act.get("goal") or act.get("outcome")
                expected_outcome = str(expected_outcome).strip() if expected_outcome and str(expected_outcome).strip().lower() not in ("none", "n/a", "null", "") else None

                action_items.append({
                    "task": task,
                    "item": task,
                    "description": task,
                    "owner": owner_str,
                    "owner_confidence": act.get("owner_confidence"),
                    "deadline": deadline,
                    "expected_outcome": expected_outcome,
                    "status": "open",
                    "raw_json": act,
                })

            logger.info(f"[ROM Service] MoM action extraction pass complete: {len(action_items)} action item(s)")

        # 3. Generate Title, Introduction, and Conclusion using standard app generator
        overview = self.generate_mom_overview(points_text_lines, recording_meta)

        return {
            "title": overview["title"],
            "date": date_str,
            "duration": dur_str,
            "participants": participants if isinstance(participants, list) else [],
            "introduction": overview["introduction"],
            "points_discussed": points_discussed,
            "action_items": action_items,
            "conclusion": overview["conclusion"],
        }

    @staticmethod
    def _enforce_single_paragraph_5_to_7_sentences(text: str) -> str:
        """
        Ensure the text is strictly:
        1. Exactly one single paragraph only (newlines/carriage returns collapsed to spaces).
        2. 5 to 7 sentences (truncated to at most 7 sentences if longer).
        3. Professional meeting-minutes punctuation.
        """
        if not text or not str(text).strip():
            return ""
        collapsed = re.sub(r'[\r\n]+', ' ', str(text).strip())
        collapsed = re.sub(r'\s{2,}', ' ', collapsed).strip()

        raw_sentences = [
            s.strip() for s in re.split(r'(?<=[.!?])\s+', collapsed) if s.strip()
        ]
        if not raw_sentences:
            return collapsed

        if len(raw_sentences) > 7:
            raw_sentences = raw_sentences[:7]

        res = " ".join(raw_sentences).strip()
        if res and not re.search(r'[.!?]$', res):
            res += "."
        return res

    def generate_mom_overview(
        self,
        points_text_lines: List[str],
        recording_meta: Dict,
    ) -> Dict[str, str]:
        """
        Generate professional Title, Introduction, and Conclusion using standard application LLM prompt.
        Reusable helper used by generate_mom_from_enhanced_rom and generate_advanced_mom.
        """
        from services.ai_provider import get_provider

        filename = recording_meta.get("filename", "Meeting MoM")
        date_str = str(recording_meta.get("created_at", "") or "")
        participants = recording_meta.get("speakers_detected", [])
        if isinstance(participants, str):
            try:
                import json as _j
                participants = _j.loads(participants)
            except Exception:
                participants = [p.strip() for p in participants.split(",") if p.strip()]

        points_combined_str = "\n".join(points_text_lines)
        first_topic = points_text_lines[0][:80] if points_text_lines else filename
        part_str = ", ".join(str(p) for p in participants) if participants else "key stakeholders and team members"
        
        intro = (
            f"This executive meeting addressed {len(points_text_lines)} key discussion topic(s) pertaining to '{filename}'. "
            f"The primary purpose was to review operational, strategic, and technical deliverables while addressing immediate priorities. "
            f"Key participants including {part_str} evaluated core proposals, shared project progress, and clarified execution constraints. "
            f"Discussions established decisive consensus on critical dependencies and the direction for ongoing initiatives. "
            f"This document provides the official record of discussion points, agreed milestones, and actionable directives."
        )
        conclusion = (
            "The meeting concluded with comprehensive alignment achieved across all primary discussion topics. "
            "Key decisions and project timelines were finalized to provide clear direction for participating stakeholders. "
            "Specific action items and commitments were designated to responsible owners with defined delivery expectations. "
            "Unresolved questions and operational risks were cataloged for proactive tracking and follow-up in the next scheduled review. "
            "Participating teams agreed to maintain active communication to ensure milestone execution on schedule."
        )
        title = filename

        prompt = (
            f"You are an executive assistant generating a formal Minutes of Meeting (MoM) document.\n\n"
            f"Recording name: '{filename}'\n"
            f"Date: {date_str}\n"
            f"Participants: {part_str}\n\n"
            f"Discussion points from the meeting:\n"
            f"{points_combined_str}\n\n"
            f"Tasks:\n"
            f"1. Generate a professional, concise meeting TITLE that reflects the actual topics discussed (do NOT just use the filename; infer from content). Example: 'Q3 Engineering Review Meeting', 'Budget & Procurement Planning Session'.\n"
            f"2. Write a professional executive INTRODUCTION section.\n"
            f"   STRICT CONSTRAINTS:\n"
            f"   - Exactly ONE single paragraph only (do NOT generate multiple paragraphs or newline breaks).\n"
            f"   - Exactly 5 to 7 sentences.\n"
            f"   - Content must be based specifically on the actual meeting discussion and context.\n"
            f"   - Use a concise, professional meeting-minutes style.\n"
            f"   - Do NOT generate multiple paragraphs, excessive explanations, or generic filler.\n"
            f"3. Write a professional executive CONCLUSION section.\n"
            f"   STRICT CONSTRAINTS:\n"
            f"   - Exactly ONE single paragraph only (do NOT generate multiple paragraphs or newline breaks).\n"
            f"   - Exactly 5 to 7 sentences.\n"
            f"   - Summarize key outcomes, decisions, progress, unresolved items, and next direction based specifically on the actual meeting.\n"
            f"   - Use a concise, professional meeting-minutes style.\n"
            f"   - Do NOT generate multiple paragraphs or unnecessarily long conclusions.\n\n"
            f"Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):\n"
            f"{{\n"
            f'  "title": "...",\n'
            f'  "introduction": "...",\n'
            f'  "conclusion": "..."\n'
            f"}}\n"
        )

        provider = None
        try:
            provider = get_provider()
            if hasattr(provider, "query"):
                raw_resp = provider.query(prompt, max_tokens=2048, temperature=0.2)
            else:
                raw_resp = provider._infer(prompt, max_new_tokens=2048)
            cleaned = str(raw_resp or "").strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            import json as _json
            data_parsed = _json.loads(cleaned)
            if isinstance(data_parsed, dict):
                if data_parsed.get("title") and str(data_parsed["title"]).strip():
                    title = str(data_parsed["title"]).strip()
                if data_parsed.get("introduction") and str(data_parsed["introduction"]).strip():
                    intro = str(data_parsed["introduction"]).strip()
                if data_parsed.get("conclusion") and str(data_parsed["conclusion"]).strip():
                    conclusion = str(data_parsed["conclusion"]).strip()
        except Exception as e:
            logger.warning(f"[RomService] Title/Intro/Conclusion generation fallback: {e}")
        finally:
            if provider and hasattr(provider, "unload_model"):
                try:
                    provider.unload_model()
                except Exception:
                    pass
            gc.collect()

        intro = self._enforce_single_paragraph_5_to_7_sentences(intro)
        conclusion = self._enforce_single_paragraph_5_to_7_sentences(conclusion)

        return {
            "title": title,
            "introduction": intro,
            "conclusion": conclusion,
        }

    def generate_advanced_mom(
        self,
        final_rom: Dict,
        existing_mom: Dict,
        custom_prompt: str,
        regenerate_title: bool = False,
        regenerate_intro: bool = False,
        regenerate_conclusion: bool = False,
        recording_meta: Dict = None,
        recording_id: str = "",
        user_id: str = "",
    ) -> Dict:
        """
        Single Responsibility: Refine, enhance, organize, and enrich extracted action points using custom user prompt instructions.
        Adds clarity, ownership, deadlines, dependencies, and context while strictly preserving factual accuracy.
        If regenerate_title, regenerate_intro, or regenerate_conclusion are True, calls the existing
        generate_mom_overview generator after action item refinement is complete.
        """
        import json as _json
        from services.ai_provider import get_provider

        recording_meta = recording_meta or {}
        filename = recording_meta.get("filename", "Meeting MoM")

        agendas = final_rom.get("agendas", []) if isinstance(final_rom, dict) else []

        # Extract lean action items context containing ONLY action_point, action_owner, speaker, date
        action_points = []
        for ag in agendas:
            if isinstance(ag, dict):
                pts = ag.get("discussion_points", [])
                if isinstance(pts, list):
                    for p in pts:
                        if isinstance(p, dict):
                            pt_text = (p.get("polished_text") or p.get("text") or "").strip()
                            owner = p.get("action_owner") or "Unassigned"
                            speakers = p.get("speakers") or ([p.get("speaker")] if p.get("speaker") else [])
                            speakers_str = ", ".join([str(s) for s in speakers if s]) if speakers else (str(p.get("speaker")) if p.get("speaker") else "Unknown")
                            dates = p.get("dates") or []
                            dates_str = ", ".join([str(d) for d in dates if d]) if isinstance(dates, list) else str(dates or "ASAP")

                            if pt_text or owner != "Unassigned" or p.get("action_items"):
                                action_points.append({
                                    "action_point": pt_text,
                                    "action_owner": owner,
                                    "speaker": speakers_str,
                                    "date": dates_str if dates_str else "ASAP"
                                })

        if isinstance(existing_mom, dict):
            ex_actions = existing_mom.get("action_items", [])
            if isinstance(ex_actions, list):
                for act in ex_actions:
                    if isinstance(act, dict):
                        task = str(act.get("task") or act.get("action_point") or "").strip()
                        if task:
                            action_points.append({
                                "action_point": task,
                                "action_owner": str(act.get("owner") or act.get("action_owner") or "Unassigned"),
                                "speaker": str(act.get("speaker") or "Unknown"),
                                "date": str(act.get("deadline") or act.get("date") or "ASAP")
                            })

        action_context_json_str = _json.dumps(action_points, indent=2)

        schema_str = (
            "{\n"
            '  "action_items": [\n'
            '    {\n'
            '      "task": "Refined and enriched action item / task description (with clarity, dependencies, and context)",\n'
            '      "owner": "Speaker name or Unassigned",\n'
            '      "deadline": "Deadline date if mentioned or derived, otherwise ASAP or null"\n'
            '    }\n'
            '  ]\n'
            "}"
        )

        prompt = (
            f"You are an expert AI meeting editor specializing in refining, enhancing, organizing, and enriching Action Points for Minutes of Meeting (MoM).\n\n"
            f"Meeting Title / Context: '{filename}'\n\n"
            f"USER CUSTOM INSTRUCTIONS:\n"
            f"\"\"\"\n{custom_prompt.strip()}\n\"\"\"\n\n"
            f"STRICT RULES:\n"
            f"1. You MUST NOT modify or hallucinate core meeting facts, speaker names, action owners, or deadlines.\n"
            f"2. Focus exclusively on refining, enhancing, organizing, and enriching the extracted action points (adding clarity, ownership, deadlines if available, dependencies, and context) based on the user's instructions.\n"
            f"3. Do NOT include any agenda details or extraneous discussion point hierarchies in the output.\n\n"
            f"CURRENT ACTION POINTS & METADATA:\n"
            f"```json\n{action_context_json_str}\n```\n\n"
            f"REQUIRED TASKS:\n"
            f"Extract and refine action_items in format: [{{\"task\": \"...\", \"owner\": \"...\", \"deadline\": \"...\"}}]\n\n"
            f"Respond ONLY with a single valid JSON object in this exact schema (no markdown, no code block backticks):\n"
            f"{schema_str}\n"
        )

        provider = None
        result_data = {}
        try:
            provider = get_provider()
            if hasattr(provider, "query"):
                raw_resp = provider.query(prompt, max_tokens=4026, temperature=0.2)
            else:
                raw_resp = provider._infer(prompt, max_new_tokens=4026)

            cleaned = str(raw_resp or "").strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

            result_data = _json.loads(cleaned)
        except Exception as e:
            logger.warning(f"[RomService] Advanced MoM LLM enhancement error: {e}")
            result_data = {}
        finally:
            if provider and hasattr(provider, "unload_model"):
                try:
                    provider.unload_model()
                except Exception:
                    pass
            gc.collect()

        # Agendas remain unchanged (Advanced MoM focuses strictly on Action Point refinement)
        updated_agendas = agendas

        # Extract action items
        raw_action_items = result_data.get("action_items") if isinstance(result_data, dict) else None
        action_items = []
        if isinstance(raw_action_items, list):
            for item in raw_action_items:
                act_dict = normalize_action_item(item)
                if act_dict["task"]:
                    displayed_task = format_action_point_display_text(act_dict)
                    assignee = act_dict.get("assignee")
                    owner = assignee if assignee else "Unassigned"
                    deadline = act_dict.get("deadline") or "ASAP"
                    action_items.append({
                        "task": displayed_task or act_dict["task"],
                        "owner": str(owner).strip(),
                        "deadline": deadline,
                        "status": item.get("status", "open") if isinstance(item, dict) else "open",
                        "raw_json": act_dict,
                    })
        
        if not action_items and isinstance(existing_mom, dict):
            existing_actions = existing_mom.get("action_items", [])
            if isinstance(existing_actions, list):
                for item in existing_actions:
                    act_dict = normalize_action_item(item)
                    if act_dict["task"]:
                        displayed_task = format_action_point_display_text(act_dict)
                        assignee = act_dict.get("assignee")
                        owner = assignee if assignee else (item.get("owner") if isinstance(item, dict) and item.get("owner") else "Unassigned")
                        deadline = act_dict.get("deadline") or (item.get("deadline") if isinstance(item, dict) and item.get("deadline") else "ASAP")
                        action_items.append({
                            "task": displayed_task or act_dict["task"],
                            "owner": str(owner).strip(),
                            "deadline": deadline,
                            "status": item.get("status", "open") if isinstance(item, dict) else "open",
                            "raw_json": act_dict if "raw_json" not in item else item["raw_json"],
                        })

        # Trigger existing overview generator ONLY if any checkbox was selected
        title = existing_mom.get("title") if isinstance(existing_mom, dict) else filename
        intro = existing_mom.get("introduction") if isinstance(existing_mom, dict) else ""
        conclusion = existing_mom.get("conclusion") if isinstance(existing_mom, dict) else ""

        if regenerate_title or regenerate_intro or regenerate_conclusion:
            # Extract discussion point lines from enhanced agendas for the generator
            point_lines = []
            for ag in updated_agendas:
                pts = ag.get("discussion_points") if isinstance(ag, dict) else []
                if isinstance(pts, list):
                    for p in pts:
                        if isinstance(p, dict):
                            t = (p.get("polished_text") or p.get("text") or p.get("discussion_point") or "").strip()
                            if t:
                                point_lines.append(t)
            if not point_lines:
                point_lines = [filename]

            overview = self.generate_mom_overview(point_lines, recording_meta)
            if regenerate_title and overview.get("title"):
                title = overview["title"]
            if regenerate_intro and overview.get("introduction"):
                intro = overview["introduction"]
            if regenerate_conclusion and overview.get("conclusion"):
                conclusion = overview["conclusion"]

        return {
            "title": title,
            "introduction": intro,
            "conclusion": conclusion,
            "agendas": updated_agendas,
            "action_items": action_items,
        }


    def extract_writing_style_rules(self, reference_text: str) -> str:
        """
        Analyze a reference MoM/ROM document and return a structured, numbered list of
        Writing Rules describing its style - sentence structure, tone, formatting, action
        item style, etc. Never includes facts, names, or content from the reference.
        """
        from services.ai_provider import get_provider

        text = (reference_text or "").strip()[:10000]
        if not text:
            return ""

        prompt = (
            "You are an expert writing-style analyst. Read the following document and extract "
            "precise, actionable Writing Rules that describe HOW it is written.\n\n"
            "REFERENCE DOCUMENT:\n"
            f"\"\"\"\n{text}\n\"\"\"\n\n"
            "Generate a numbered WRITING RULES list covering ALL of these aspects:\n"
            "1. Sentence Structure - average length, active vs passive voice\n"
            "2. Tone & Formality - formal/semi-formal, person (first/third), objective/opinionated\n"
            "3. Formatting Style - use of bullets, numbering, headers, indentation, spacing\n"
            "4. Discussion Point Style - how discussion items are framed and opened\n"
            "5. Action Item Style - verb form, owner format, deadline expression\n"
            "6. Terminology Preferences - domain-specific phrasing or word choices\n"
            "7. Numbering & Bullet Conventions - punctuation, capitalization in lists\n"
            "8. Recurring Patterns - any distinctive stylistic habits or patterns\n\n"
            "CONSTRAINTS:\n"
            "- Extract STYLE RULES ONLY. Do NOT include any facts, names, decisions, or content from the document.\n"
            "- Each rule must be specific and actionable.\n"
            "- Number every rule. No preamble, no commentary after the list.\n\n"
            "Output ONLY the numbered writing rules list."
        )

        provider = get_provider()
        try:
            if hasattr(provider, "query"):
                rules = provider.query(prompt, max_tokens=2048, temperature=0.15)
            else:
                rules = provider._infer(prompt, max_new_tokens=2048)
            return str(rules or "").strip()
        except Exception as e:
            logger.warning(f"[RomService] extract_writing_style_rules error: {e}")
            return ""
        finally:
            provider.unload_model()
            gc.collect()

    # -- Private rewrite helpers -----------------------------------------------

    def _build_rules_section(self, writing_rules: str) -> str:
        """Return the writing-rules prompt block, or empty string if no rules provided."""
        if not writing_rules or not writing_rules.strip():
            return ""
        return (
            "\n\nREFERENCE WRITING RULES "
            "(apply these style rules to your rewrites - "
            "NEVER copy any content, facts, names, or decisions from the reference):\n"
            f"\"\"\"\n{writing_rules.strip()}\n\"\"\"\n"
        )

    def _parse_json_string_array(self, raw: str, expected_count: int) -> Optional[List[str]]:
        """
        Robustly parse a JSON string array from LLM output.
        Returns the list if it has exactly expected_count elements, else None.
        """
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        raw = raw[start:end + 1]
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        if not isinstance(parsed, list):
            return None
        if len(parsed) != expected_count:
            logger.warning(
                f"[RomService] JSON array length mismatch: expected {expected_count}, got {len(parsed)}"
            )
            return None
        return [str(t).strip() if t else "" for t in parsed]

    def _rewrite_window_batch(
        self,
        texts: List[str],
        instruction: str,
        rules_section: str,
        provider,
    ) -> List[str]:
        """
        Rewrite a batch (window) of discussion-point texts in a single LLM call.
        Returns the original texts unchanged on any failure so callers can continue.
        """
        if not texts:
            return texts
        count = len(texts)
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

        prompt = (
            "You are a professional document editor. Rewrite the following meeting discussion "
            f"points according to the instruction below.\n\n"
            f"INSTRUCTION:\n\"\"\"\n{instruction.strip()}\n\"\"\""
            f"{rules_section}\n\n"
            f"DISCUSSION POINTS TO REWRITE ({count} point{'s' if count != 1 else ''}):\n"
            f"{numbered}\n\n"
            "STRICT REWRITE RULES:\n"
            "1. Improve ONLY writing style, grammar, sentence structure, and phrasing.\n"
            "2. NEVER change facts, technical content, speakers, decisions, action items, names, dates, or numbers.\n"
            "3. NEVER add, remove, merge, split, summarize, or reinterpret any point.\n"
            f"4. Output EXACTLY {count} rewritten string{'s' if count != 1 else ''} - one per input point.\n"
            "5. Preserve the exact chronological order.\n\n"
            f"Respond ONLY with a valid JSON array of exactly {count} rewritten strings "
            "(no markdown, no code fences, no extra text):\n"
            '["rewritten point 1", "rewritten point 2", ...]'
        )

        try:
            if hasattr(provider, "query"):
                raw = provider.query(prompt, max_tokens=4096, temperature=0.15, task_key="rom_rewrite", think=False)
            else:
                raw = provider._infer(prompt, max_new_tokens=4096, task_key="rom_rewrite", think=False)

            parsed = self._parse_json_string_array(str(raw or ""), count)
            if parsed is not None:
                return [t if t else texts[i] for i, t in enumerate(parsed)]
            logger.warning("[RomService] _rewrite_window_batch: parse failed, keeping originals")
            return texts
        except Exception as e:
            logger.warning(f"[RomService] _rewrite_window_batch error: {e} - keeping originals")
            return texts

    def _rewrite_windowed_rom(
        self,
        agendas: List[Dict],
        instruction: str,
        rules_section: str,
        window_size: int,
        provider,
    ) -> None:
        """
        Iterate all agendas, process discussion points in windows of `window_size`.
        Modifies agendas in-place. Per-window failure keeps originals for that window.
        """
        for agenda in agendas:
            if not isinstance(agenda, dict):
                continue
            pts = agenda.get("discussion_points", [])
            if not isinstance(pts, list) or not pts:
                continue

            for win_start in range(0, len(pts), window_size):
                window = pts[win_start: win_start + window_size]
                texts = [(pt.get("polished_text") or pt.get("text") or "").strip() for pt in window]
                if not any(texts):
                    continue

                rewritten_texts = self._rewrite_window_batch(texts, instruction, rules_section, provider)

                for j, (pt, new_text) in enumerate(zip(window, rewritten_texts)):
                    if new_text and new_text != texts[j]:
                        pt["text"] = new_text
                        if pt.get("polished_text"):
                            pt["polished_text"] = new_text

    def _rewrite_complete_rom(
        self,
        agendas: List[Dict],
        instruction: str,
        rules_section: str,
        provider,
    ) -> None:
        """
        Rewrite the entire ROM in a single LLM call for global style consistency.
        Modifies agendas in-place. Falls back to originals on any failure.
        """
        flat: List[tuple] = []
        for a_idx, agenda in enumerate(agendas):
            if not isinstance(agenda, dict):
                continue
            pts = agenda.get("discussion_points", [])
            if not isinstance(pts, list):
                continue
            for p_idx, pt in enumerate(pts):
                if not isinstance(pt, dict):
                    continue
                text = (pt.get("polished_text") or pt.get("text") or "").strip()
                if text:
                    flat.append((a_idx, p_idx, text))

        if not flat:
            return

        total = len(flat)
        lines = []
        for i, (a_idx, p_idx, text) in enumerate(flat):
            agenda_title = agendas[a_idx].get("title", f"Agenda {a_idx + 1}")
            lines.append(f"{i + 1}. [{agenda_title}] {text}")
        numbered = "\n".join(lines)

        prompt = (
            "You are a professional document editor. Rewrite ALL of the following meeting "
            "discussion points in a single pass for global style consistency.\n\n"
            f"INSTRUCTION:\n\"\"\"\n{instruction.strip()}\n\"\"\""
            f"{rules_section}\n\n"
            f"ALL DISCUSSION POINTS ({total} total):\n"
            f"{numbered}\n\n"
            "STRICT REWRITE RULES:\n"
            "1. Improve ONLY writing style, grammar, sentence structure, and phrasing - globally consistent.\n"
            "2. NEVER change facts, speakers, decisions, action items, names, dates, or numbers.\n"
            "3. NEVER add, remove, merge, split, summarize, or reinterpret any points.\n"
            f"4. Output EXACTLY {total} rewritten strings - one per input point, in order.\n"
            "5. Preserve exact chronological order.\n\n"
            f"Respond ONLY with a valid JSON array of exactly {total} rewritten strings:\n"
            '["rewritten 1", "rewritten 2", ...]'
        )

        try:
            if hasattr(provider, "query"):
                raw = provider.query(prompt, max_tokens=8192, temperature=0.15, task_key="rom_rewrite", think=False)
            else:
                raw = provider._infer(prompt, max_new_tokens=8192, task_key="rom_rewrite", think=False)

            parsed = self._parse_json_string_array(str(raw or ""), total)
            if parsed is None:
                logger.warning("[RomService] _rewrite_complete_rom: parse failed, keeping all originals")
                return

            for i, (a_idx, p_idx, orig_text) in enumerate(flat):
                new_text = parsed[i] if parsed[i] else orig_text
                if new_text:
                    agendas[a_idx]["discussion_points"][p_idx]["text"] = new_text
                    if agendas[a_idx]["discussion_points"][p_idx].get("polished_text"):
                        agendas[a_idx]["discussion_points"][p_idx]["polished_text"] = new_text
        except Exception as e:
            logger.warning(f"[RomService] _rewrite_complete_rom error: {e} - keeping all originals")


    @staticmethod
    def _parse_condensed_points(raw_resp: Any) -> List[str]:
        """
        Robustly extract a list of condensed points from LLM output.
        Handles:
        - Thinking tags (<think>...</think>)
        - Markdown code fences (```json ... ```)
        - JSON array of strings: ["point 1", "point 2"]
        - JSON array of objects: [{"text": "point 1"}, ...]
        - Trailing commas in JSON
        - Bullet points or numbered lists (1. ..., - ...)
        """
        import re
        import json as _json

        if not raw_resp:
            return []

        text = str(raw_resp).strip()

        # 1. Strip <think>...</think> tags completely
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "<think>" in text:
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()
            else:
                text = text.split("<think>")[0].strip()

        # 2. Check for fenced code blocks
        fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        candidates = fence_matches if fence_matches else [text]

        # Try parsing JSON from candidates or whole text
        for cand in candidates:
            cand_str = cand.strip()
            # Find outermost [ ... ]
            start = cand_str.find("[")
            end = cand_str.rfind("]")
            if start != -1 and end != -1 and end > start:
                json_substr = cand_str[start:end + 1]
                try:
                    data = _json.loads(json_substr)
                    if isinstance(data, list):
                        pts = []
                        for item in data:
                            if isinstance(item, dict):
                                t = (
                                    item.get("text")
                                    or item.get("polished_text")
                                    or item.get("point")
                                    or item.get("discussion")
                                    or str(item)
                                )
                                if t and str(t).strip():
                                    pts.append(str(t).strip())
                            elif item and str(item).strip():
                                pts.append(str(item).strip())
                        if pts:
                            return pts
                except Exception:
                    # Try repairing trailing commas and invalid backslash escapes (e.g. \approx, Windows paths)
                    try:
                        fixed = re.sub(r",\s*([\]\}])", r"\1", json_substr)
                        fixed = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', fixed)
                        data = _json.loads(fixed)
                        if isinstance(data, list):
                            pts = []
                            for item in data:
                                if isinstance(item, dict):
                                    t = (
                                        item.get("text")
                                        or item.get("polished_text")
                                        or item.get("point")
                                        or item.get("discussion")
                                        or str(item)
                                    )
                                    if t and str(t).strip():
                                        pts.append(str(t).strip())
                                elif item and str(item).strip():
                                    pts.append(str(item).strip())
                            if pts:
                                return pts
                    except Exception:
                        # Try parsing multiple separate array blocks: [ "pt 1" ] \n [ "pt 2" ]
                        try:
                            multi_arrays = re.findall(r'\[[\s\S]*?\]', cand_str)
                            if len(multi_arrays) > 1:
                                multi_pts = []
                                for arr_str in multi_arrays:
                                    fixed_arr = re.sub(r",\s*([\]\}])", r"\1", arr_str)
                                    fixed_arr = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', fixed_arr)
                                    arr_data = _json.loads(fixed_arr)
                                    if isinstance(arr_data, list):
                                        for item in arr_data:
                                            if isinstance(item, dict):
                                                t = (
                                                    item.get("text")
                                                    or item.get("polished_text")
                                                    or item.get("point")
                                                    or item.get("discussion")
                                                    or str(item)
                                                )
                                            else:
                                                t = str(item)
                                            if t and str(t).strip():
                                                multi_pts.append(str(t).strip())
                                if multi_pts:
                                    return multi_pts
                        except Exception:
                            pass

        # 3. Fallback: Parse line-by-line (numbered lists or bullet points)
        bullet_pts = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("```") or line.startswith("#"):
                continue
            m = re.match(r"^(?:\d+[\.\)]|[-*•])\s*(.+)$", line)
            if m:
                item = m.group(1).strip()
                if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                    item = item[1:-1].strip()
                if item:
                    bullet_pts.append(item)

        return bullet_pts

    # -- Very Important Point helper ------------------------------------------

    @staticmethod
    def _is_point_very_important(pt: dict, agenda_id: str, important_points: list) -> bool:
        """
        Check if a discussion point is marked as 'Very Important'.
        Matches against direct flags on the point (is_very_important, is_important)
        or against any entry in important_points by point_id, agenda_id, or text matching.
        """
        if not isinstance(pt, dict):
            return False
        if pt.get("is_very_important") is True or pt.get("is_important") is True:
            return True
        if not important_points:
            return False

        pt_id = str(pt.get("id") or "").strip()
        pt_text = (pt.get("polished_text") or pt.get("text") or "").strip()
        pt_text_lower = pt_text.lower()

        for imp in important_points:
            if isinstance(imp, str):
                imp_clean = imp.strip().lower()
                if imp_clean and (imp_clean == pt_id.lower() or imp_clean in pt_text_lower or pt_text_lower in imp_clean):
                    return True
            elif isinstance(imp, dict):
                imp_agenda = str(imp.get("agenda_id") or "").strip()
                imp_pt_id = str(imp.get("point_id") or imp.get("id") or "").strip()
                imp_text = str(imp.get("text") or "").strip()
                imp_text_lower = imp_text.lower()

                # If point_id matches directly, it's a match
                if pt_id and imp_pt_id and pt_id == imp_pt_id:
                    return True

                # If agenda_id is specified and does not match, skip text matching for other agendas
                if imp_agenda and agenda_id and imp_agenda != agenda_id:
                    continue

                # Text substring matching
                if imp_text_lower and pt_text_lower:
                    if imp_text_lower in pt_text_lower or pt_text_lower in imp_text_lower:
                        return True

        return False

    # -- Public generate-version entry-point -----------------------------------

    def generate_rom_version(
        self,
        final_rom: Dict,
        version: str,
        writing_rules: str = "",
        important_points: list = None,
        reasoning_mode: Optional[bool] = None,
    ) -> Dict:
        """
        Generate a condensed Short or Medium version of the Final ROM.

        Each agenda's discussion points are passed together (agenda-wise) to the LLM,
        which returns a reduced set of meaningful, properly structured points.
        Every point marked as 'Very Important' is guaranteed to be included in the output
        without any summarization, preserving its complete original content fully intact.

        version: 'short' or 'medium'
        writing_rules: Optional writing rules extracted from a reference document.
        important_points: Optional list of user-marked important points.

        Returns a deep-copy with the new points; original is NOT modified.
        """
        import copy
        import re
        from services.prompt_service import get_prompt_sync

        if not final_rom or not isinstance(final_rom, dict):
            return final_rom

        version = (version or "long").strip().lower()
        if version == "long":
            # Long: Keep existing Stage 2 points as default.
            # If reference writing rules exist, apply them.
            if writing_rules and writing_rules.strip():
                return self.rewrite_final_rom(
                    final_rom=final_rom,
                    rewrite_instruction="Rewrite the ROM in a formal, professional writing style.",
                    mode="reference",
                    writing_rules=writing_rules,
                )
            return copy.deepcopy(final_rom)

        if version not in ("short", "medium"):
            return copy.deepcopy(final_rom)

        rewritten = copy.deepcopy(final_rom)
        agendas = rewritten.get("agendas", [])
        if not agendas:
            return rewritten

        prompt_key = "rom_version_short" if version == "short" else "rom_version_medium"
        prompt_template = get_prompt_sync(prompt_key)

        rules_section = self._build_rules_section(writing_rules)

        from services.ai_provider import get_provider
        provider = get_provider()

        if reasoning_mode is not None:
            use_reasoning = bool(reasoning_mode)
        else:
            try:
                active_cfg = provider._get_active_settings() if hasattr(provider, "_get_active_settings") else {}
                use_reasoning = bool(active_cfg.get("reasoning_mode_short_rom", False))
            except Exception:
                use_reasoning = False
        try:
            for agenda in agendas:
                if not isinstance(agenda, dict):
                    continue
                pts = agenda.get("discussion_points", [])
                if not isinstance(pts, list) or not pts:
                    continue

                agenda_title = agenda.get("title", "General Discussion")
                agenda_id = agenda.get("agenda_id", "")

                # 1. Identify all points in this agenda marked as Very Important
                marked_pts = []
                for pt in pts:
                    if self._is_point_very_important(pt, agenda_id, important_points):
                        marked_pts.append(pt)

                # 2. Build structured input per point: Point ID, Speaker, Discussion, Action Owner
                lines = []
                for i, pt in enumerate(pts):
                    pt_id = pt.get("id", f"P{i+1}")
                    speaker = (
                        pt.get("speaker") or
                        (pt.get("speakers") or [None])[0] or
                        "Unknown"
                    )
                    discussion = (pt.get("polished_text") or pt.get("text") or "").strip()
                    action_owner = pt.get("action_owner") or "N/A"
                    is_pt_marked = pt in marked_pts
                    imp_tag = " | *** VERY IMPORTANT - PRESERVE INTACT WITHOUT SUMMARIZATION ***" if is_pt_marked else ""
                    lines.append(
                        f"[Point {i+1}] ID={pt_id} | Speaker={speaker} | "
                        f"Action Owner={action_owner}{imp_tag}\n{discussion}"
                    )
                points_json = "\n\n".join(lines)

                # 3. Build mandatory section from marked Very Important points
                if marked_pts:
                    mandatory_section = (
                        "\n\nMANDATORY VERY IMPORTANT POINTS (MUST BE INCLUDED VERBATIM WITHOUT ANY SUMMARIZATION):\n"
                    )
                    for idx, m_pt in enumerate(marked_pts, 1):
                        m_text = (m_pt.get("polished_text") or m_pt.get("text") or "").strip()
                        m_spk = m_pt.get("speaker") or (m_pt.get("speakers") or [None])[0] or "Unknown"
                        m_act = m_pt.get("action_owner") or "N/A"
                        mandatory_section += f"- Point {idx} (Speaker: {m_spk}, Action Owner: {m_act}):\n  \"{m_text}\"\n"
                    mandatory_section += (
                        "STRICT RULES FOR VERY IMPORTANT POINTS:\n"
                        "1. EVERY point marked as 'Very Important' above MUST be included in the output JSON array.\n"
                        "2. You MUST preserve the COMPLETE ORIGINAL CONTENT of each marked point fully intact.\n"
                        "3. Do NOT summarize, shorten, merge, paraphrase, or omit any Very Important point.\n"
                        "4. Output each marked point VERBATIM alongside the condensed summary of the other points.\n"
                    )
                else:
                    mandatory_section = ""

                prompt = prompt_template.format(
                    agenda_title=agenda_title,
                    points_json=points_json,
                    rules_section=rules_section,
                    mandatory_section=mandatory_section,
                )

                try:
                    if version == "short" and use_reasoning:
                        # ── Reasoning Mode ON: 2 LLM calls ──
                        # 1. Reasoning call: Ask LLM to analyze/reason about input only. Max 2000 tokens.
                        reasoning_prompt = (
                            "You are an expert meeting-record editor.\n\n"
                            "Analyze and reason about the following agenda discussion to prepare for generating a concise Short ROM.\n\n"
                            f"AGENDA:\n{agenda_title}\n\n"
                            f"DISCUSSION POINTS:\n{points_json}\n\n"
                            f"{rules_section}{mandatory_section}\n\n"
                            "WHOLE-AGENDA SYNTHESIS & POINT-REDUCTION INSTRUCTIONS:\n"
                            "1. Consider all points together holistically under this agenda as ONE unified conversation, NOT as isolated individual points.\n"
                            "2. Actively reduce the total number of points: do NOT produce a 1-to-1 mapping of input points.\n"
                            "3. Merge and consolidate related discussion threads, debates, decisions, and updates into a substantially smaller number of comprehensive, high-density points (aim for roughly 3 to 6 synthesized points for an agenda with 8-15 input points).\n"
                            "4. Identify the unifying themes and group related points together (e.g., group infrastructure upgrades together, observability/stability reviews together, data integrity/security policies together).\n"
                            "5. Ensure critical details—explicit action owners, deadlines, key metrics, and agreed decisions—are identified so they can be smoothly woven into the merged points.\n"
                            "6. Formulate the concrete synthesis plan for each resulting consolidated point.\n\n"
                            "CRITICAL CONSTRAINT: Provide your analysis, topic groupings, and synthesis plan ONLY. Do NOT generate the final JSON array or finalized response. "
                            "Focus strictly on planning how to consolidate and reduce the discussion points across the entire agenda."
                        )
                        logger.info(f"[RomService] Short ROM agenda '{agenda_title}' executing reasoning call (max 2000 tokens)...")
                        if hasattr(provider, "query"):
                            reasoning_raw = provider.query(reasoning_prompt, max_tokens=2000, temperature=0.2)
                        else:
                            reasoning_raw = provider._infer(reasoning_prompt, max_new_tokens=2000, task_key="rom_version_short_reasoning")

                        from services.ai_provider import strip_thinking
                        reasoning_text = strip_thinking(reasoning_raw).strip()
                        logger.info(f"[RomService] Short ROM agenda '{agenda_title}' reasoning call complete ({len(reasoning_text)} chars).")

                        # 2. Generation call: Pass reasoning in REASONING BLOCK, disable thinking/reasoning, generate only final valid JSON output.
                        generation_prompt = (
                            f"{prompt}\n\n"
                            f"==================================================\n"
                            f"REASONING BLOCK:\n"
                            f"{reasoning_text}\n"
                            f"==================================================\n\n"
                            f"CRITICAL INSTRUCTIONS FOR POINT REDUCTION & WHOLE-AGENDA SYNTHESIS:\n"
                            f"- Consider all points together under this agenda as one conversation and use the REASONING BLOCK above to synthesize them.\n"
                            f"- Do NOT output a 1-to-1 mapping of input points. Actively consolidate and reduce the total number of points into a smaller, unified set of comprehensive points (aim for roughly 3 to 6 high-impact points for this agenda).\n"
                            f"- In each consolidated point, naturally weave together the relevant decisions, technical details, action owners, and deadlines identified in the reasoning block.\n"
                            f"- Disable thinking/reasoning: do NOT repeat the reasoning text or include any explanatory comments.\n"
                            f"- Return ONLY a valid JSON array of the consolidated point strings: [\"point 1\", \"point 2\", ...]\n"
                        )
                        logger.info(f"[RomService] Short ROM agenda '{agenda_title}' executing generation call (think=False)...")
                        if hasattr(provider, "query"):
                            raw = provider.query(generation_prompt, max_tokens=4096, temperature=0.2)
                        else:
                            raw = provider._infer(generation_prompt, task_key=prompt_key, think=False)
                    else:
                        # ── Reasoning Mode OFF (or medium version): 1 normal LLM call (without thinking mode) ──
                        if hasattr(provider, "query"):
                            raw = provider.query(prompt, max_tokens=4096, temperature=0.2)
                        else:
                            raw = provider._infer(prompt, task_key=prompt_key, think=False)

                    # Robustly parse points from LLM output (handles <think>, markdown fences, arrays, bullets)
                    parsed_texts = self._parse_condensed_points(raw)
                    if not parsed_texts:
                        logger.warning(
                            f"[RomService] generate_rom_version({version}): agenda='{agenda_title}' "
                            f"LLM returned unparseable response, keeping originals. Raw: {str(raw)[:150]}"
                        )
                        continue

                    # 4. Guarantee every marked Very Important point is preserved intact without summarization
                    claimed_parsed_indices = set()
                    pt_matches = {}  # m_idx -> parsed_text_idx

                    for m_idx, m_pt in enumerate(marked_pts):
                        orig_content = (m_pt.get("polished_text") or m_pt.get("text") or "").strip()
                        if not orig_content:
                            continue
                        orig_norm = re.sub(r'[\s\W_]+', ' ', orig_content.lower()).strip()
                        orig_words = set(orig_norm.split())

                        best_idx = None
                        best_score = 0.0

                        for p_idx, p_text in enumerate(parsed_texts):
                            if p_idx in claimed_parsed_indices:
                                continue
                            cand_norm = re.sub(r'[\s\W_]+', ' ', str(p_text).lower()).strip()
                            if not cand_norm:
                                continue

                            # Exact or substring containment
                            if cand_norm == orig_norm or cand_norm in orig_norm or orig_norm in cand_norm:
                                score = 1.0
                            else:
                                cand_words = set(cand_norm.split())
                                overlap = len(orig_words & cand_words)
                                orig_ratio = overlap / max(1, len(orig_words))
                                cand_ratio = overlap / max(1, len(cand_words))
                                score = max(orig_ratio, cand_ratio)

                            if score > best_score and score >= 0.35:
                                best_score = score
                                best_idx = p_idx

                        if best_idx is not None:
                            pt_matches[m_idx] = best_idx
                            claimed_parsed_indices.add(best_idx)

                    # Build initial list of point representations: (text, src_pt_or_m_pt, is_marked)
                    parsed_to_marked = {p_idx: marked_pts[m_idx] for m_idx, p_idx in pt_matches.items()}

                    processed_items = []
                    for p_idx, p_text in enumerate(parsed_texts):
                        clean_text = str(p_text).strip()
                        if not clean_text:
                            continue
                        if p_idx in parsed_to_marked:
                            # Replace with complete original content of the marked point
                            m_pt = parsed_to_marked[p_idx]
                            orig_content = (m_pt.get("polished_text") or m_pt.get("text") or "").strip()
                            processed_items.append((orig_content, m_pt, True))
                        else:
                            # Condensed point from non-marked source
                            src = pts[min(p_idx, len(pts) - 1)]
                            processed_items.append((clean_text, src, False))

                    # For any marked point that was omitted by LLM (not matched to any parsed text),
                    # insert it with its complete original content at its relative meeting position!
                    for m_idx, m_pt in enumerate(marked_pts):
                        if m_idx not in pt_matches:
                            orig_content = (m_pt.get("polished_text") or m_pt.get("text") or "").strip()
                            if not orig_content:
                                continue
                            orig_pos = pts.index(m_pt) if m_pt in pts else len(pts)
                            insert_pos = int((orig_pos / max(1, len(pts))) * (len(processed_items) + 1))
                            insert_pos = max(0, min(len(processed_items), insert_pos))
                            processed_items.insert(insert_pos, (orig_content, m_pt, True))

                    # Construct final new_pts list with full metadata preservation
                    new_pts = []
                    for j, (p_text, src, is_marked) in enumerate(processed_items):
                        if is_marked:
                            new_pt = {
                                "id": f"{agenda.get('agenda_id', 'A')}-{version.upper()}-{j+1}",
                                "text": p_text,
                                "polished_text": p_text,
                                "speaker": src.get("speaker") or (src.get("speakers") or [None])[0] or "Unknown",
                                "speakers": src.get("speakers") or ([src.get("speaker")] if src.get("speaker") else []),
                                "action_owner": src.get("action_owner"),
                                "action_items": copy.deepcopy(src.get("action_items") or []),
                                "timeline_start": src.get("timeline_start", 0.0),
                                "timeline_end": src.get("timeline_end", 0.0),
                                "references": copy.deepcopy(src.get("references") or []),
                                "technical_terms": copy.deepcopy(src.get("technical_terms") or []),
                                "dates": copy.deepcopy(src.get("dates") or []),
                                "numbers": copy.deepcopy(src.get("numbers") or []),
                                "is_very_important": True,
                                "original_point_id": src.get("id"),
                            }
                        else:
                            new_pt = {
                                "id": f"{agenda.get('agenda_id', 'A')}-{version.upper()}-{j+1}",
                                "text": p_text,
                                "polished_text": p_text,
                                "speaker": src.get("speaker") or (src.get("speakers") or [None])[0] or "Unknown",
                                "speakers": src.get("speakers") or [],
                                "action_owner": src.get("action_owner"),
                                "action_items": src.get("action_items") or [],
                                "timeline_start": src.get("timeline_start", 0.0),
                                "timeline_end": src.get("timeline_end", 0.0),
                                "references": src.get("references") or [],
                                "technical_terms": [],
                                "dates": [],
                                "numbers": [],
                                "is_very_important": False,
                                "original_point_id": src.get("id"),
                            }
                        new_pts.append(new_pt)

                    if new_pts:
                        agenda["discussion_points"] = new_pts
                        logger.info(
                            f"[RomService] generate_rom_version({version}): agenda='{agenda_title}' "
                            f"{len(pts)} pts -> {len(new_pts)} pts successfully generated (marked intact: {len(marked_pts)})"
                        )
                    else:
                        logger.warning(
                            f"[RomService] generate_rom_version({version}): agenda='{agenda_title}' "
                            f"Empty points list, keeping originals"
                        )

                except Exception as e:
                    logger.warning(
                        f"[RomService] generate_rom_version({version}): agenda='{agenda_title}' "
                        f"error={e} - keeping originals"
                    )

        finally:
            provider.unload_model()
            gc.collect()

        return rewritten

    def generate_rom_version_with_variant(
        self,
        final_rom: Dict,
        version: str,
        writing_rules: str = "",
        important_points: list = None,
        variant_id: str = "",
        adapter_path: str = "",
        base_model: str = "",
    ) -> Dict:
        """
        Generate a condensed Short ROM using a trained LoRA variant.
        Evaluates each agenda discussion using infer_short_rom_with_variant,
        bypassing Ollama and running through the fine-tuned adapter.
        """
        import copy
        import gc
        import logging
        import re

        logger = logging.getLogger(__name__)

        if not final_rom or not isinstance(final_rom, dict):
            return {}

        if version != "short" or not variant_id or not adapter_path:
            # Fallback to base model for medium version or if variant is missing
            return self.generate_rom_version(
                final_rom=final_rom,
                version=version,
                writing_rules=writing_rules,
                important_points=important_points,
            )

        rewritten = copy.deepcopy(final_rom)
        agendas = rewritten.get("agendas", [])
        if not agendas:
            return rewritten

        from services.prompt_service import get_prompt_sync
        prompt_template = get_prompt_sync("rom_version_short")
        rules_section = self._build_rules_section(writing_rules)

        from services.rom_training.rom_training_inference import infer_short_rom_with_variant

        try:
            for agenda in agendas:
                if not isinstance(agenda, dict):
                    continue
                pts = agenda.get("discussion_points", [])
                if not isinstance(pts, list) or not pts:
                    continue

                agenda_title = agenda.get("title", "General Discussion")
                agenda_id = agenda.get("agenda_id", "")

                # 1. Identify all points in this agenda marked as Very Important
                marked_pts = []
                for pt in pts:
                    if self._is_point_very_important(pt, agenda_id, important_points):
                        marked_pts.append(pt)

                # 2. Build structured input per point
                lines = []
                for i, pt in enumerate(pts):
                    pt_id = pt.get("id", f"P{i+1}")
                    speaker = (
                        pt.get("speaker") or
                        (pt.get("speakers") or [None])[0] or
                        "Unknown"
                    )
                    discussion = (
                        pt.get("polished_text")
                        or pt.get("text")
                        or pt.get("discussion_point")
                        or ""
                    ).strip()
                    lines.append(f"[Point {i+1}] {discussion}")
                points_json = "\n\n".join(lines)

                # 3. Build mandatory section from marked Very Important points
                if marked_pts:
                    mandatory_section = (
                        "\n\nMANDATORY POINTS (MUST BE PRESERVED IN OUTPUT INTACT):\n" +
                        "\n".join(
                            f"- {m.get('polished_text') or m.get('text', '')}"
                            for m in marked_pts
                            if (m.get("polished_text") or m.get("text"))
                        )
                    )
                else:
                    mandatory_section = ""

                prompt = prompt_template.format(
                    agenda_title=agenda_title,
                    points_json=points_json,
                    rules_section=rules_section,
                    mandatory_section=mandatory_section,
                )

                try:
                    logger.info(f"[RomService] Generating Short ROM for agenda '{agenda_title}' using trained variant '{variant_id}'...")
                    raw = infer_short_rom_with_variant(
                        prompt=prompt,
                        variant_id=variant_id,
                        adapter_path=adapter_path,
                        base_model=base_model,
                        max_tokens=4096,
                        use_adapter=True,
                    )

                    # Parse points from model output
                    parsed_texts = self._parse_condensed_points(raw)
                    if not parsed_texts:
                        logger.warning(
                            f"[RomService] generate_rom_version_with_variant: agenda='{agenda_title}' "
                            f"Variant returned unparseable response, keeping originals. Raw: {str(raw)[:150]}"
                        )
                        continue

                    # Guarantee every marked Very Important point is preserved intact
                    claimed_parsed_indices = set()
                    pt_matches = {}

                    for m_idx, m_pt in enumerate(marked_pts):
                        orig_content = (m_pt.get("polished_text") or m_pt.get("text") or "").strip()
                        if not orig_content:
                            continue
                        orig_norm = re.sub(r'[\s\W_]+', ' ', orig_content.lower()).strip()
                        orig_words = set(orig_norm.split())

                        best_idx = None
                        best_score = 0.0

                        for p_idx, p_text in enumerate(parsed_texts):
                            if p_idx in claimed_parsed_indices:
                                continue
                            cand_norm = re.sub(r'[\s\W_]+', ' ', str(p_text).lower()).strip()
                            if not cand_norm:
                                continue

                            if cand_norm == orig_norm or cand_norm in orig_norm or orig_norm in cand_norm:
                                score = 1.0
                            else:
                                cand_words = set(cand_norm.split())
                                overlap = len(orig_words & cand_words)
                                orig_ratio = overlap / max(1, len(orig_words))
                                cand_ratio = overlap / max(1, len(cand_words))
                                score = max(orig_ratio, cand_ratio)

                            if score > best_score and score >= 0.35:
                                best_score = score
                                best_idx = p_idx

                        if best_idx is not None:
                            pt_matches[m_idx] = best_idx
                            claimed_parsed_indices.add(best_idx)

                    parsed_to_marked = {p_idx: marked_pts[m_idx] for m_idx, p_idx in pt_matches.items()}

                    processed_items = []
                    for p_idx, p_text in enumerate(parsed_texts):
                        clean_text = str(p_text).strip()
                        if not clean_text:
                            continue
                        if p_idx in parsed_to_marked:
                            m_pt = parsed_to_marked[p_idx]
                            orig_content = (m_pt.get("polished_text") or m_pt.get("text") or "").strip()
                            processed_items.append((orig_content, m_pt, True))
                        else:
                            src = pts[min(p_idx, len(pts) - 1)]
                            processed_items.append((clean_text, src, False))

                    for m_idx, m_pt in enumerate(marked_pts):
                        if m_idx not in pt_matches:
                            orig_content = (m_pt.get("polished_text") or m_pt.get("text") or "").strip()
                            if not orig_content:
                                continue
                            orig_pos = pts.index(m_pt) if m_pt in pts else len(pts)
                            insert_pos = int((orig_pos / max(1, len(pts))) * (len(processed_items) + 1))
                            insert_pos = max(0, min(len(processed_items), insert_pos))
                            processed_items.insert(insert_pos, (orig_content, m_pt, True))

                    new_pts = []
                    for j, (p_text, src, is_marked) in enumerate(processed_items):
                        if is_marked:
                            new_pt = {
                                "id": f"{agenda.get('agenda_id', 'A')}-{version.upper()}-{j+1}",
                                "text": p_text,
                                "polished_text": p_text,
                                "speaker": src.get("speaker") or (src.get("speakers") or [None])[0] or "Unknown",
                                "speakers": src.get("speakers") or ([src.get("speaker")] if src.get("speaker") else []),
                                "action_owner": src.get("action_owner"),
                                "action_items": copy.deepcopy(src.get("action_items") or []),
                                "timeline_start": src.get("timeline_start", 0.0),
                                "timeline_end": src.get("timeline_end", 0.0),
                                "references": copy.deepcopy(src.get("references") or []),
                                "technical_terms": copy.deepcopy(src.get("technical_terms") or []),
                                "dates": copy.deepcopy(src.get("dates") or []),
                                "numbers": copy.deepcopy(src.get("numbers") or []),
                                "is_very_important": True,
                                "original_point_id": src.get("id"),
                            }
                        else:
                            new_pt = {
                                "id": f"{agenda.get('agenda_id', 'A')}-{version.upper()}-{j+1}",
                                "text": p_text,
                                "polished_text": p_text,
                                "speaker": src.get("speaker") or (src.get("speakers") or [None])[0] or "Unknown",
                                "speakers": src.get("speakers") or [],
                                "action_owner": src.get("action_owner"),
                                "action_items": src.get("action_items") or [],
                                "timeline_start": src.get("timeline_start", 0.0),
                                "timeline_end": src.get("timeline_end", 0.0),
                                "references": src.get("references") or [],
                                "technical_terms": [],
                                "dates": [],
                                "numbers": [],
                                "is_very_important": False,
                                "original_point_id": src.get("id"),
                            }
                        new_pts.append(new_pt)

                    if new_pts:
                        agenda["discussion_points"] = new_pts
                        logger.info(
                            f"[RomService] generate_rom_version_with_variant: agenda='{agenda_title}' "
                            f"{len(pts)} pts -> {len(new_pts)} pts successfully generated"
                        )
                except Exception as exc:
                    logger.warning(
                        f"[RomService] generate_rom_version_with_variant: agenda='{agenda_title}' "
                        f"error={exc} - keeping originals",
                        exc_info=True,
                    )
        finally:
            try:
                from services.rom_training.rom_training_inference import unload_variant, clean_vram
                unload_variant(variant_id)
                clean_vram()
            except Exception:
                gc.collect()

        return rewritten

    # -- AI Edit Points (Edit with AI feature) ---------------------------------

    def ai_edit_points(
        self,
        points: list,
        prompt: str,
        agenda_context: str = "",
        chat_history: list = None,
    ) -> Dict:
        """
        Edit selected discussion points using AI based on user instructions.

        Returns a dict with 'updated_points' and 'explanation'.
        """
        import json
        from services.prompt_service import get_prompt_sync

        prompt_template = get_prompt_sync("rom_ai_edit_points")

        # Build selected points text
        selected_lines = []
        for i, pt in enumerate(points):
            pt_id = pt.get("id", f"P{i+1}")
            speaker = (
                pt.get("speaker") or
                (pt.get("speakers") or [None])[0] or
                "Unknown"
            )
            text = (pt.get("text") or pt.get("polished_text") or "").strip()
            selected_lines.append(
                f"[Point {i+1}] ID={pt_id} | Speaker={speaker}\n{text}"
            )
        selected_points_str = "\n\n".join(selected_lines)

        # Build chat history section
        chat_history_section = ""
        if chat_history and len(chat_history) > 0:
            chat_lines = []
            for msg in chat_history[-10:]:  # Last 10 messages for context
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                chat_lines.append(f"{role}: {content}")
            chat_history_section = "PREVIOUS CONVERSATION:\n" + "\n".join(chat_lines)

        full_prompt = prompt_template.format(
            agenda_context=agenda_context or "General Discussion",
            selected_points=selected_points_str,
            user_prompt=prompt,
            chat_history_section=chat_history_section,
        )

        from services.ai_provider import get_provider, strip_thinking
        provider = None
        try:
            provider = get_provider()
            if hasattr(provider, "query"):
                raw = provider.query(full_prompt, max_tokens=4096, temperature=0.3, task_key="rom_ai_edit_points", think=False)
            else:
                raw = provider._infer(full_prompt, max_new_tokens=4096, task_key="rom_ai_edit_points", think=False)

            # Parse JSON response
            raw_str = strip_thinking(str(raw)).strip()

            # Remove markdown fences if present
            if raw_str.startswith("```"):
                lines = raw_str.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw_str = "\n".join(lines).strip()

            # Try to find JSON object
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_str)
            if json_match:
                raw_str = json_match.group(0)

            result = json.loads(raw_str)

            updated_points = result.get("updated_points", [])
            explanation = result.get("explanation", "Points have been updated based on your instructions.")

            return {
                "updated_points": updated_points,
                "explanation": explanation,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"[RomService] ai_edit_points: Failed to parse JSON response: {e}. Raw: {str(raw)[:300]}")
            # Fallback: return original points with error explanation
            return {
                "updated_points": [{"id": pt.get("id", ""), "text": pt.get("text") or pt.get("polished_text", "")} for pt in points],
                "explanation": f"AI response could not be parsed. Original points retained. Error: {str(e)}",
            }
        except Exception as e:
            logger.error(f"[RomService] ai_edit_points: Error: {e}")
            raise
        finally:
            if provider and hasattr(provider, "unload_model"):
                try:
                    provider.unload_model()
                except Exception:
                    pass
            gc.collect()

    # -- Conclude Points (Combine multiple points into 1 conclusion) ------------

    def conclude_points(
        self,
        points: list,
        agenda_context: str = "",
    ) -> Dict:
        """
        Combine multiple discussion points into ONE concise conclusion of 2-3 sentences.
        Preserves critical information, decisions, key facts, and meaning from all selected points.
        Thinking mode is strictly disabled.
        Returns a dict with 'concluded_point', 'replaced_point_id', 'removed_point_ids', and 'explanation'.
        """
        import json
        import re
        import gc
        from services.prompt_service import get_prompt_sync
        from services.ai_provider import get_provider, strip_thinking

        if not points or len(points) < 2:
            raise ValueError("At least 2 points are required to conclude.")

        prompt_template = get_prompt_sync("rom_conclude_points")

        # Build selected points text
        selected_lines = []
        for i, pt in enumerate(points):
            pt_id = pt.get("id", f"P{i+1}")
            speaker = (
                pt.get("speaker") or
                (pt.get("speakers") or [None])[0] or
                "Unknown"
            )
            text = (pt.get("text") or pt.get("polished_text") or "").strip()
            action_owner = pt.get("action_owner")
            owner_str = f" | Action Owner={action_owner}" if action_owner else ""
            selected_lines.append(
                f"[Point {i+1}] ID={pt_id} | Speaker={speaker}{owner_str}\n{text}"
            )
        selected_points_str = "\n\n".join(selected_lines)

        full_prompt = prompt_template.format(
            agenda_context=agenda_context or "General Discussion",
            selected_points=selected_points_str,
        )

        provider = None
        raw_str = ""
        conclusion_text = ""
        explanation = "Combined selected points into one concise conclusion of 2–3 sentences."
        try:
            provider = get_provider()
            if hasattr(provider, "query"):
                raw = provider.query(full_prompt, max_tokens=2048, temperature=0.3, task_key="rom_conclude_points", think=False)
            else:
                raw = provider._infer(full_prompt, max_new_tokens=2048, task_key="rom_conclude_points", think=False)

            raw_str = strip_thinking(str(raw)).strip()

            # Remove markdown fences if present
            if raw_str.startswith("```"):
                lines = raw_str.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw_str = "\n".join(lines).strip()

            # Extract JSON object
            json_match = re.search(r'\{[\s\S]*\}', raw_str)
            if json_match:
                raw_str = json_match.group(0)

            result = json.loads(raw_str)
            conclusion_text = (result.get("conclusion") or result.get("text") or "").strip()
            if result.get("explanation"):
                explanation = result["explanation"]

        except Exception as e:
            logger.warning(f"[RomService] conclude_points JSON parse failed or LLM error: {e}. Raw: {raw_str[:300]}")
            # Try field-level regex extraction first
            field_match = re.search(r'"(?:conclusion|text|summary)"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_str)
            if field_match:
                try:
                    conclusion_text = json.loads(f'"{field_match.group(1)}"')
                except Exception:
                    conclusion_text = field_match.group(1)
            elif raw_str and not raw_str.startswith("{"):
                conclusion_text = strip_thinking(raw_str).strip()
                conclusion_text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', conclusion_text, flags=re.DOTALL).strip()

            # If still empty, attempt direct plain-text summarization fallback (never concatenate verbatim)
            if not conclusion_text and provider:
                try:
                    logger.info("[RomService] conclude_points attempting direct plain-text executive summary fallback")
                    fb_prompt = (
                        "You are an expert meeting records editor. Synthesize and summarize the following selected discussion points "
                        "into ONE clear executive summary of strictly 1 to 3 sentences.\n"
                        "Do NOT concatenate or list points sequentially. Return ONLY the 1 to 3 sentence summary text:\n\n"
                        f"{selected_points_str}"
                    )
                    raw_fb = provider.query(fb_prompt, max_tokens=512, temperature=0.2, task_key="rom_conclude_points", think=False) if hasattr(provider, "query") else provider._infer(fb_prompt, max_new_tokens=512, task_key="rom_conclude_points", think=False)
                    fb_clean = strip_thinking(str(raw_fb)).strip()
                    fb_clean = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', fb_clean, flags=re.DOTALL).strip()
                    if fb_clean and not fb_clean.startswith("{"):
                        conclusion_text = fb_clean
                except Exception as fb_e:
                    logger.warning(f"[RomService] Direct fallback summarization failed: {fb_e}")

            if not conclusion_text:
                first_sentences = [
                    re.split(r'(?<=[.!?])\s+', (pt.get("text") or pt.get("polished_text") or "").strip())[0].strip()
                    for pt in points if (pt.get("text") or pt.get("polished_text"))
                ]
                conclusion_text = " ".join(first_sentences[:3]) if first_sentences else (points[0].get("text") or "")
            explanation = "Synthesized conclusion generated from selected points."
        finally:
            if provider and hasattr(provider, "unload_model"):
                try:
                    provider.unload_model()
                except Exception:
                    pass
            gc.collect()

        if not conclusion_text:
            first_sentences = [
                re.split(r'(?<=[.!?])\s+', (pt.get("text") or pt.get("polished_text") or "").strip())[0].strip()
                for pt in points if (pt.get("text") or pt.get("polished_text"))
            ]
            conclusion_text = " ".join(first_sentences[:3]) if first_sentences else "Selected discussion points concluded into summary."

        # Merge metadata from all points
        all_speakers = []
        for p in points:
            spks = p.get("speakers")
            if isinstance(spks, list):
                all_speakers.extend(spks)
            elif p.get("speaker"):
                all_speakers.append(p.get("speaker"))
        unique_speakers = list(dict.fromkeys(s for s in all_speakers if s and s != "—"))

        all_actions = []
        for p in points:
            acts = p.get("action_items") or p.get("action_points") or []
            if isinstance(acts, list):
                all_actions.extend(acts)

        all_terms = []
        for p in points:
            t = p.get("technical_terms") or []
            if isinstance(t, list):
                all_terms.extend(t)
        unique_terms = list(dict.fromkeys(t for t in all_terms if t))

        all_dates = []
        for p in points:
            d = p.get("dates") or []
            if isinstance(d, list):
                all_dates.extend(d)
        unique_dates = list(dict.fromkeys(d for d in all_dates if d))

        all_numbers = []
        for p in points:
            n = p.get("numbers") or []
            if isinstance(n, list):
                all_numbers.extend(n)
        unique_numbers = list(dict.fromkeys(n for n in all_numbers if n))

        all_refs = []
        for p in points:
            r = p.get("references") or []
            if isinstance(r, list):
                all_refs.extend(r)
        unique_refs = list(dict.fromkeys(r for r in all_refs if r))

        first_pt = points[0]
        replaced_id = first_pt.get("id")
        removed_ids = [p.get("id") for p in points[1:] if p.get("id")]

        valid_starts = [p.get("timeline_start") for p in points if isinstance(p.get("timeline_start"), (int, float))]
        valid_ends = [p.get("timeline_end") for p in points if isinstance(p.get("timeline_end"), (int, float))]

        concluded_point = {
            "id": replaced_id,
            "text": conclusion_text,
            "polished_text": conclusion_text,
            "speaker": unique_speakers[0] if unique_speakers else (first_pt.get("speaker") or "—"),
            "speakers": unique_speakers if unique_speakers else (first_pt.get("speakers") or []),
            "action_owner": next((p.get("action_owner") for p in points if p.get("action_owner")), None),
            "action_items": all_actions,
            "technical_terms": unique_terms,
            "dates": unique_dates,
            "numbers": unique_numbers,
            "references": unique_refs,
            "timeline_start": min(valid_starts) if valid_starts else 0.0,
            "timeline_end": max(valid_ends) if valid_ends else 0.0,
            "is_very_important": any(p.get("is_very_important") for p in points),
            "original_point_ids": [p.get("id") for p in points if p.get("id")],
            "is_concluded": True,
        }

        return {
            "concluded_point": concluded_point,
            "replaced_point_id": replaced_id,
            "removed_point_ids": removed_ids,
            "explanation": explanation,
        }

    # -- Public rewrite entry-point --------------------------------------------

    def rewrite_final_rom(
        self,
        final_rom: Dict,
        rewrite_instruction: str,
        mode: str = "window",
        window_size: int = 3,
        writing_rules: str = "",
    ) -> Dict:
        """
        Rewrite writing style of all discussion-point texts in the Final ROM.

        Modes
        -----
        'window'    -- process discussion points in configurable windows (default).
        'complete'  -- single LLM call for the whole ROM (best global consistency).
        'reference' -- window-based rewrite with writing rules from a reference doc.

        Returns a deep-copy; the caller must NOT save automatically.
        """
        import copy

        if not final_rom or not isinstance(final_rom, dict):
            return final_rom

        rewritten = copy.deepcopy(final_rom)
        agendas = rewritten.get("agendas", [])
        if not agendas:
            return rewritten

        rules_section = self._build_rules_section(writing_rules)
        effective_window = max(1, min(10, int(window_size)))

        from services.ai_provider import get_provider
        provider = get_provider()
        try:
            if mode == "complete":
                self._rewrite_complete_rom(agendas, rewrite_instruction, rules_section, provider)
            else:
                # 'window' and 'reference' both use the windowed approach
                self._rewrite_windowed_rom(agendas, rewrite_instruction, rules_section, effective_window, provider)
        finally:
            provider.unload_model()
            gc.collect()

        return rewritten

def apply_speaker_mappings_to_final_rom(final_rom: Dict) -> Dict:

    """
    Replaces every occurrence of mapped Speaker_IDs across all discussion points,
    action items, presenters, participants, and text fields throughout the Final ROM structure.
    Unmapped Speaker_IDs remain unchanged.
    """
    if not final_rom or not isinstance(final_rom, dict):
        return final_rom

    speaker_mappings = final_rom.get("speaker_mappings", {})
    if not speaker_mappings or not isinstance(speaker_mappings, dict):
        return final_rom

    from services.speaker_sync import build_clean_speaker_mappings, replace_speaker_in_text, replace_deep_speaker_names
    clean_mappings = build_clean_speaker_mappings(speaker_mappings)
    if not clean_mappings:
        return final_rom

    updated = replace_deep_speaker_names(final_rom, clean_mappings)

    # Specific field list updates
    parts = updated.get("participants")
    if isinstance(parts, list):
        updated["participants"] = [clean_mappings.get(p, replace_speaker_in_text(p, clean_mappings)) for p in parts]

    updated["speaker_mappings"] = clean_mappings
    return updated


rom_service = RomService()