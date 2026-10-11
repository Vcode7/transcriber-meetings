"""
Final ROM action points, generated per agenda.

When an agenda's "Include Action Points" option is on, all of its discussion points
(complete Stage 2 detail, never the short/medium rewrite) are analysed together by the
LLM, and the result is checked deterministically:

- vague, unsupported, low-priority and duplicate tasks are dropped;
- dates, numbers and acronyms in a task must come from the cited discussion points;
- owners and deadlines the meeting did not state become "Not specified".

Results are stored on the agenda (``action_points`` with ``action_points_mode ==
"agenda"``). Older Final ROMs keep their point-level action points and are read
through ``collect_agenda_action_points`` unchanged.
"""
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from services.action_owner_validator import (
    DUPLICATE_TASK_THRESHOLD,
    VAGUE_OWNERS,
    split_owner_names,
    task_similarity,
    verify_owner_against_candidates,
)
from services.fact_grounding import (
    content_tokens,
    fact_tokens,
    source_facts,
    split_sentences,
    token_overlap,
    unsupported_facts,
)

logger = logging.getLogger(__name__)

NOT_SPECIFIED = "Not specified"
ACTION_POINTS_MODE = "agenda"
PRIORITIES = ("high", "medium", "low")
_PRIORITY_RANK = {p: r for r, p in enumerate(reversed(PRIORITIES))}

# Characters of discussion-point JSON per LLM call; larger agendas are split chronologically.
INPUT_CHAR_BUDGET = 12000
# Share of a task's content words that must appear in the discussion points it cites.
MIN_SOURCE_OVERLAP = 0.25
# Tasks citing no valid point are re-linked to the best-matching point above this overlap.
RELINK_MIN_OVERLAP = 0.3
# A task needs at least this many specific words/facts besides vague verbs ("follow up", ...).
MIN_SPECIFIC_TOKENS = 2
# Paraphrase duplicates by embedding cosine (Qwen3-Embedding, measured on real tasks: true
# paraphrases 0.86-0.89, different tasks of one meeting up to 0.84). A shared named owner is
# strong extra evidence, so it allows the lower threshold.
EMBED_DUPLICATE_THRESHOLD = 0.88
EMBED_DUPLICATE_SAME_OWNER_THRESHOLD = 0.85
EVIDENCE_MAX_CHARS = 200
# Paraphrased duplicates are caught with the active embedding model; tests switch this off.
EMBEDDING_DEDUPE = True

_UNSPECIFIED_VALUES = {
    "", "null", "none", "n/a", "na", "unknown", "undefined", "unassigned", "not specified",
    "not mentioned", "not stated", "not provided", "tbd", "-", "—",
}
_VAGUE_PHRASE_RE = re.compile(
    r"\b(?:follow(?:\s|-)?up(?:\s+on|\s+with)?|look(?:ing)?\s+into|discuss(?:ed)?\s+further|further\s+discussion|"
    r"keep\s+(?:monitoring|tracking|track\s+of|in\s+mind|an\s+eye\s+on)|continue\s+to\s+monitor|monitor(?:ing)?|"
    r"as\s+needed|as\s+required|if\s+needed|if\s+required|when\s+possible|take\s+care\s+of|work\s+on|think\s+about|"
    r"consider(?:ing)?|check\s+on|stay\s+in\s+touch|touch\s+base|circle\s+back|revisit|accordingly|etc)\b",
    re.IGNORECASE,
)
_GENERIC_WORDS = {
    "next", "meeting", "later", "progress", "status", "update", "updates", "team", "everyone", "matter",
    "issue", "issues", "point", "points", "item", "items", "discussion", "discussions", "topic", "topics",
    "needed", "required", "possible", "soon", "things", "thing", "stuff", "same", "above", "relevant",
}
_ASAP_RE = re.compile(r"\b(?:asap|as soon as possible|immediately|urgent(?:ly)?|right away)\b", re.IGNORECASE)
_TIME_UNITS = {
    "day", "days", "week", "weeks", "weekend", "month", "months", "quarter", "quarters", "year", "years",
    "today", "tomorrow", "tonight", "morning", "evening", "afternoon", "eod", "eow", "eom",
}
_BASES = {"decision", "commitment", "assignment", "request"}


def is_unspecified(value: Any) -> bool:
    return value is None or str(value).strip().lower().rstrip(".") in _UNSPECIFIED_VALUES


def agenda_includes_action_points(agenda: Dict[str, Any], final_rom: Optional[Dict[str, Any]] = None) -> bool:
    """Per-agenda option; agendas saved before it existed follow the old global flag."""
    if not isinstance(agenda, dict) or agenda.get("skipped"):
        return False
    flag = agenda.get("include_action_points")
    if flag is not None:
        return bool(flag)
    return bool((final_rom or {}).get("include_action_points"))


def has_agenda_action_points(agenda: Dict[str, Any]) -> bool:
    return agenda.get("action_points_mode") == ACTION_POINTS_MODE and isinstance(agenda.get("action_points"), list)


def collect_agenda_action_points(agenda: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Action points shown for an agenda (Final ROM view and DOCX)."""
    if agenda.get("action_points_mode") == ACTION_POINTS_MODE:
        return [ap for ap in agenda.get("action_points") or [] if isinstance(ap, dict)]

    # Legacy Final ROMs: agenda-level list plus point-level action points / action items.
    from services.rom_service import format_action_point_display_text, normalize_action_item

    collected: List[Dict[str, Any]] = []
    seen = set()

    def _add(ap: Dict[str, Any]) -> None:
        task = ap.get("task") or ap.get("item") or ap.get("description") or ""
        key = str(task).strip().lower()
        if key and key not in seen:
            seen.add(key)
            collected.append(ap)

    if isinstance(agenda.get("action_points"), list):
        for ap in agenda["action_points"]:
            if isinstance(ap, dict):
                _add(ap)
    for pt in agenda.get("discussion_points") or []:
        action_pts = list(pt.get("action_points") or [])
        if not action_pts and pt.get("action_items"):
            for ai in pt["action_items"]:
                norm = normalize_action_item(ai)
                if norm.get("task"):
                    action_pts.append({
                        "task": format_action_point_display_text(norm) or norm["task"],
                        "assignee": norm.get("assignee") or norm.get("owner"),
                        "deadline": norm.get("deadline"),
                        "owner_confidence": norm.get("owner_confidence"),
                    })
        for ap in action_pts:
            if isinstance(ap, dict):
                _add(ap)
    return collected


# ── Input ────────────────────────────────────────────────────────────────────

def _point_text(point: Dict[str, Any]) -> str:
    return str(point.get("text") or point.get("polished_text") or point.get("discussion_point") or "").strip()


def _clean_list(values: Any) -> List[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    return [str(v).strip() for v in values if v is not None and str(v).strip()]


def _clean_owner(raw: Any) -> Optional[str]:
    if isinstance(raw, (list, tuple)):
        raw = ", ".join(str(r).strip() for r in raw if r)
    if is_unspecified(raw) or str(raw).strip().lower() in VAGUE_OWNERS:
        return None
    return str(raw).strip()


def _candidate_actions(point: Dict[str, Any], source: Dict[str, Any], stage1_by_id: Dict[str, Dict]) -> List[Dict]:
    from services.rom_service import normalize_action_item

    raw = point.get("action_items") or source.get("action_items") or []
    if not raw:
        orig_ids = []
        for holder in (point, source):
            if holder.get("original_point_id"):
                orig_ids.append(holder["original_point_id"])
            orig_ids.extend(holder.get("original_point_ids") or [])
        for oid in orig_ids:
            s1 = stage1_by_id.get(oid)
            if s1 and s1.get("action_items"):
                raw = s1["action_items"]
                break
    candidates = []
    for act in raw if isinstance(raw, list) else []:
        norm = normalize_action_item(act)
        if norm.get("task"):
            candidates.append({
                "task": norm["task"],
                "owner": norm.get("assignee"),
                "deadline": norm.get("deadline"),
            })
    return candidates


def _format_seconds(value: Any) -> Optional[str]:
    try:
        sec = float(value)
    except (TypeError, ValueError):
        return None
    if sec <= 0:
        return None
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def build_agenda_action_input(
    agenda: Dict[str, Any],
    polished_by_id: Dict[str, Dict],
    stage1_by_id: Dict[str, Dict],
) -> List[Dict[str, Any]]:
    """Every meeting discussion point of the agenda with its complete Stage 2 detail, in
    meeting order. A condensed (short/medium version) point is replaced by the complete
    Stage 2 point it was written from."""
    entries = []
    seen = set()
    for point in agenda.get("discussion_points") or []:
        if not isinstance(point, dict) or point.get("is_doc_point"):
            continue  # document extracts are not things the meeting agreed on
        pid = str(point.get("id") or "").strip()
        if not pid:
            continue
        original = point.get("original_point_id")
        if pid not in polished_by_id and original in polished_by_id:
            pid, point = original, polished_by_id[original]
        if pid in seen:
            continue
        seen.add(pid)
        source = polished_by_id.get(pid) or {}
        text = _point_text(point) or _point_text(source)
        if not text:
            continue

        def pick(key: str) -> Any:
            value = point.get(key)
            return value if value else source.get(key)

        owner = _clean_owner(pick("action_owner"))
        speakers = _clean_list(pick("speakers")) or _clean_list(point.get("speaker"))
        entry: Dict[str, Any] = {"id": pid, "text": text}
        if pick("point_type"):
            entry["point_type"] = pick("point_type")
        if speakers:
            entry["speakers"] = speakers
        if owner:
            entry["action_owner"] = owner
            entry["owner_candidates"] = split_owner_names(owner)
        for key in ("dates", "numbers", "technical_terms", "references"):
            values = _clean_list(pick(key))
            if values:
                entry[key] = values
        start, end = _format_seconds(pick("timeline_start")), _format_seconds(pick("timeline_end"))
        if start:
            entry["time"] = f"{start}-{end}" if end else start
        candidates = _candidate_actions(point, source, stage1_by_id)
        if candidates:
            entry["candidate_actions"] = candidates
        excerpt = str(pick("owner_transcript_excerpt") or "").strip()
        if excerpt:
            entry["transcript_excerpt"] = excerpt
        entry["_start"] = float(pick("timeline_start") or 0.0)
        entries.append(entry)
    entries.sort(key=lambda e: e["_start"])  # stable: equal timestamps keep their order
    for entry in entries:
        entry.pop("_start")
    return entries


def _llm_view(entry: Dict[str, Any]) -> Dict[str, Any]:
    # The transcript excerpt is only used to verify owners; it would crowd out the points.
    return {k: v for k, v in entry.items() if k != "transcript_excerpt"}


def _agenda_header(agenda: Dict[str, Any]) -> Dict[str, Any]:
    header = {
        "agenda_id": agenda.get("agenda_id"),
        "title": agenda.get("title") or "",
        "description": agenda.get("description") or "",
    }
    presenter = agenda.get("presenter") or agenda.get("speaker")
    if presenter:
        header["presenter"] = presenter
    for key in ("keywords", "expected_themes"):
        values = _clean_list(agenda.get(key))
        if values:
            header[key] = values[:15]
    return header


def split_for_budget(entries: List[Dict[str, Any]], budget: Optional[int] = None) -> List[List[Dict[str, Any]]]:
    """Chronological chunks whose JSON fits the budget (at least one point per chunk)."""
    budget = budget or INPUT_CHAR_BUDGET
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    size = 0
    for entry in entries:
        entry_size = len(json.dumps(_llm_view(entry), ensure_ascii=False))
        if current and size + entry_size > budget:
            chunks.append(current)
            current, size = [], 0
        current.append(entry)
        size += entry_size
    if current:
        chunks.append(current)
    return chunks


def request_agenda_action_points(
    provider,
    agenda: Dict[str, Any],
    entries: List[Dict[str, Any]],
    reasoning_mode: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """LLM proposals for one agenda. Returns (items, succeeded)."""
    header_json = json.dumps(_agenda_header(agenda), ensure_ascii=False, indent=2)
    items: List[Dict[str, Any]] = []
    succeeded = False
    chunks = split_for_budget(entries)
    for idx, chunk in enumerate(chunks, 1):
        points_json = json.dumps([_llm_view(e) for e in chunk], ensure_ascii=False, indent=2)
        try:
            result = provider.generate_agenda_action_points(header_json, points_json, reasoning_mode=reasoning_mode)
        except Exception as exc:
            logger.warning(f"[AgendaActions] {agenda.get('agenda_id')}: chunk {idx}/{len(chunks)} failed: {exc}")
            continue
        if not isinstance(result, dict):
            logger.warning(f"[AgendaActions] {agenda.get('agenda_id')}: chunk {idx}/{len(chunks)} returned no parsable JSON")
            continue
        succeeded = True
        proposed = result.get("action_points")
        if isinstance(proposed, list):
            items.extend(item for item in proposed if isinstance(item, dict))
    return items, succeeded


def fallback_items_from_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stage 1/2 action items of the agenda's points, used when the LLM call fails."""
    items = []
    for entry in entries:
        for cand in entry.get("candidate_actions") or []:
            items.append({
                "task": cand.get("task"),
                "owner": cand.get("owner"),
                "deadline": cand.get("deadline"),
                "source_point_ids": [entry["id"]],
                "priority": "medium",
                "basis": "commitment",
            })
    return items


# ── Validation ───────────────────────────────────────────────────────────────

def _entry_source_parts(entry: Dict[str, Any]) -> List[Any]:
    parts: List[Any] = [entry.get("text", "")]
    for key in ("dates", "numbers", "technical_terms", "references", "owner_candidates"):
        parts.append(entry.get(key) or [])
    for cand in entry.get("candidate_actions") or []:
        parts.extend([cand.get("task"), cand.get("owner"), cand.get("deadline")])
    return parts


def _entry_source_text(entry: Dict[str, Any]) -> str:
    out = []
    for part in _entry_source_parts(entry):
        if isinstance(part, (list, tuple)):
            out.extend(str(p) for p in part if p)
        elif part:
            out.append(str(part))
    return "\n".join(out)


def is_vague_task(task: str) -> bool:
    stripped = _VAGUE_PHRASE_RE.sub(" ", task)
    specific = content_tokens(stripped) - _GENERIC_WORDS
    return len(specific) + len(fact_tokens(task)) < MIN_SPECIFIC_TOKENS


def _deadline_grounded(deadline: str, source_text: str, facts: set) -> bool:
    if _ASAP_RE.search(deadline):
        return bool(_ASAP_RE.search(source_text))
    deadline_facts = fact_tokens(deadline)
    if deadline_facts:
        return not (deadline_facts - facts)
    tokens = content_tokens(deadline)
    if not tokens:
        return False
    source_tokens = content_tokens(source_text)
    if (tokens & _TIME_UNITS) - source_tokens:
        return False
    return len(tokens & source_tokens) / len(tokens) >= 0.6


def _evidence_quote(raw_evidence: Any, task: str, cited: List[Dict[str, Any]], source_text: str) -> str:
    evidence = str(raw_evidence or "").strip().strip('"')
    if evidence and token_overlap(evidence, source_text) >= 0.8:
        return evidence[:EVIDENCE_MAX_CHARS]
    sentences = [s for e in cited for s in split_sentences(e.get("text", ""))]
    if not sentences:
        return ""
    best = max(sentences, key=lambda s: token_overlap(task, s))
    return best[:EVIDENCE_MAX_CHARS]


def _ids_from(raw: Dict[str, Any]) -> List[str]:
    ids = raw.get("source_point_ids") or raw.get("source_point_id") or raw.get("point_ids") or raw.get("id") or []
    if isinstance(ids, (str, int)):
        ids = [ids]
    if not isinstance(ids, (list, tuple)):
        return []
    return [str(i).strip() for i in ids if i is not None and str(i).strip()]


def validate_action_points(
    items: List[Dict[str, Any]],
    entries: List[Dict[str, Any]],
    roster: Optional[List[Dict[str, Any]]] = None,
    embedder=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministic checks on proposed action points.

    Returns {"kept": [...], "rejected": [{task, reason}], "adjusted": [{task, field, value, reason}]}.
    Kept items use None for an unstated owner/deadline (see ``finalize_action_point``).
    """
    by_id = {e["id"]: e for e in entries}
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    adjusted: List[Dict[str, Any]] = []

    for raw in items:
        task = str(raw.get("task") or raw.get("action_point") or raw.get("description") or "").strip()
        task = re.sub(r"\s+", " ", task).strip(" -•*")
        if not task:
            continue

        def reject(reason: str) -> None:
            rejected.append({"task": task, "reason": reason})

        if is_vague_task(task):
            reject("vague: no specific deliverable")
            continue

        ids = [i for i in dict.fromkeys(_ids_from(raw)) if i in by_id]
        if not ids or token_overlap(task, "\n".join(_entry_source_text(by_id[i]) for i in ids)) < MIN_SOURCE_OVERLAP:
            # Missing or mis-cited sources: link the task to the point that supports it best.
            best = max(entries, key=lambda e: token_overlap(task, _entry_source_text(e)), default=None)
            if best is None or token_overlap(task, _entry_source_text(best)) < RELINK_MIN_OVERLAP:
                reject("not supported by this agenda's discussion points")
                continue
            ids = list(dict.fromkeys(ids + [best["id"]]))
        cited = [by_id[i] for i in ids]
        source_text = "\n".join(_entry_source_text(e) for e in cited)

        facts = source_facts(_entry_source_parts(e) for e in cited)
        if unsupported_facts(task, facts):
            invented = unsupported_facts(task, source_facts(_entry_source_parts(e) for e in entries))
            if invented:
                reject(f"details not in the discussion: {', '.join(invented)}")
                continue
            # The detail comes from another point of the same agenda: cite that point too.
            task_facts = fact_tokens(task)
            for entry in entries:
                entry_facts = source_facts(_entry_source_parts(entry))
                if entry["id"] not in ids and task_facts & (entry_facts - facts):
                    ids.append(entry["id"])
                    cited.append(entry)
                    facts |= entry_facts
                if not unsupported_facts(task, facts):
                    break
            source_text = "\n".join(_entry_source_text(e) for e in cited)

        deadline = raw.get("deadline") or raw.get("due") or raw.get("due_date")
        if is_unspecified(deadline):
            deadline = None
        else:
            deadline = str(deadline).strip()
            if not _deadline_grounded(deadline, source_text, facts):
                adjusted.append({"task": task, "field": "deadline", "value": deadline,
                                 "reason": "deadline not stated in the discussion"})
                deadline = None

        owner = _clean_owner(raw.get("owner") or raw.get("assignee") or raw.get("action_owner"))
        owner_confidence = None
        if owner:
            candidates = [c for e in cited for c in e.get("owner_candidates") or []]
            candidates += [c["owner"] for e in cited for c in e.get("candidate_actions") or [] if c.get("owner")]
            excerpt = "\n".join([source_text] + [e.get("transcript_excerpt", "") for e in cited])
            verified, confidence = verify_owner_against_candidates(owner, candidates, excerpt, roster)
            if not verified:
                adjusted.append({"task": task, "field": "owner", "value": owner,
                                 "reason": "owner not named in the discussion"})
                owner = None
            else:
                owner = verified
                owner_confidence = confidence or "medium"

        priority = str(raw.get("priority") or "medium").strip().lower()
        if priority not in PRIORITIES:
            priority = "medium"
        if priority == "low" and not owner and not deadline:
            reject("low priority with no owner or deadline")
            continue

        basis = str(raw.get("basis") or "").strip().lower()
        kept.append({
            "task": task,
            "assignee": owner,
            "deadline": deadline,
            "source_point_ids": ids,
            "priority": priority,
            "basis": basis if basis in _BASES else None,
            "evidence_quote": _evidence_quote(raw.get("evidence"), task, cited, source_text),
            "owner_confidence": owner_confidence,
        })

    kept, duplicates = dedupe_action_points(kept, embedder)
    rejected.extend(duplicates)
    return {"kept": kept, "rejected": rejected, "adjusted": adjusted}


def _same_named_owner(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    owner_a, owner_b = a.get("assignee"), b.get("assignee")
    return not is_unspecified(owner_a) and not is_unspecified(owner_b) and         str(owner_a).strip().lower() == str(owner_b).strip().lower()


def _owners_conflict(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    owner_a, owner_b = a.get("assignee"), b.get("assignee")
    if is_unspecified(owner_a) or is_unspecified(owner_b):
        return False
    return str(owner_a).strip().lower() != str(owner_b).strip().lower()


def _merge_into(keep: Dict[str, Any], dup: Dict[str, Any]) -> None:
    keep["source_point_ids"] = list(dict.fromkeys(keep["source_point_ids"] + dup["source_point_ids"]))
    for key in ("assignee", "deadline", "owner_confidence", "basis"):
        if not keep.get(key) and dup.get(key):
            keep[key] = dup[key]
    if len(dup["task"]) > len(keep["task"]):
        keep["task"] = dup["task"]
    if _PRIORITY_RANK.get(dup["priority"], 0) > _PRIORITY_RANK.get(keep["priority"], 0):
        keep["priority"] = dup["priority"]


def _task_vectors(items: List[Dict[str, Any]], embedder) -> Optional[np.ndarray]:
    if embedder is None or len(items) < 2:
        return None
    try:
        vecs = np.asarray(embedder.encode_batch([it["task"] for it in items]), dtype=np.float32)
    except Exception as exc:
        logger.warning(f"[AgendaActions] embedding dedupe skipped: {exc}")
        return None
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms < 1e-9, 1.0, norms)


def dedupe_action_points(items: List[Dict[str, Any]], embedder=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge duplicates (same task, compatible owners). Returns (kept, rejected)."""
    vecs = _task_vectors(items, embedder)
    kept_idx: List[int] = []
    removed: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        match = None
        for k in kept_idx:
            if _owners_conflict(items[k], item):
                continue
            same_words = task_similarity(items[k]["task"], item["task"]) >= DUPLICATE_TASK_THRESHOLD
            threshold = EMBED_DUPLICATE_SAME_OWNER_THRESHOLD if _same_named_owner(items[k], item) else EMBED_DUPLICATE_THRESHOLD
            same_meaning = vecs is not None and float(vecs[k] @ vecs[i]) >= threshold
            if same_words or same_meaning:
                match = k
                break
        if match is None:
            kept_idx.append(i)
            continue
        removed.append({"task": item["task"], "reason": f"duplicate of: {items[match]['task']}"})
        _merge_into(items[match], item)
    return [items[k] for k in kept_idx], removed


def finalize_action_point(item: Dict[str, Any]) -> Dict[str, Any]:
    """Stored/displayed form: unstated owner and deadline are explicitly "Not specified"."""
    entry = {
        "task": item["task"],
        "assignee": item.get("assignee") or NOT_SPECIFIED,
        "deadline": item.get("deadline") or NOT_SPECIFIED,
        "source_point_ids": list(item.get("source_point_ids") or []),
        "source_point_id": (item.get("source_point_ids") or [None])[0],
        "priority": item.get("priority") or "medium",
    }
    for key in ("basis", "evidence_quote", "owner_confidence"):
        if item.get(key):
            entry[key] = item[key]
    return entry


# ── Orchestration ────────────────────────────────────────────────────────────

def _build_roster(agendas: List[Dict[str, Any]]):
    from services.rom_transcript_format import build_roster

    names = []
    for agenda in agendas:
        for point in agenda.get("discussion_points") or []:
            for name in _clean_list(point.get("speakers")) or _clean_list(point.get("speaker")):
                if name not in names:
                    names.append(name)
    try:
        return build_roster([{"speaker_label": name} for name in names])
    except Exception:
        return None


def populate_agenda_action_points(
    agendas: List[Dict[str, Any]],
    polished_points: List[Dict[str, Any]],
    stage1_points: List[Dict[str, Any]],
    *,
    provider=None,
    reasoning_mode: Optional[bool] = None,
    target_agenda_ids: Optional[Iterable[str]] = None,
    regenerate: bool = True,
    recording_id: str = "",
    source_agendas: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Generate validated action points for the targeted, non-skipped agendas (in place).

    ``target_agenda_ids`` limits generation to those agendas (default: all non-skipped).
    With ``regenerate=False`` agendas that already have agenda-level action points are
    left as they are. ``source_agendas`` (the long Final ROM version) supplies the complete
    discussion points when ``agendas`` is a condensed version. Returns {agenda_id: audit}.
    """
    wanted = {str(a) for a in target_agenda_ids} if target_agenda_ids is not None else None
    targets = [
        a for a in agendas
        if isinstance(a, dict) and not a.get("skipped")
        and (wanted is None or str(a.get("agenda_id")) in wanted)
        and (regenerate or not has_agenda_action_points(a))
    ]
    if not targets:
        return {}

    polished_by_id = {p.get("id"): p for p in polished_points or [] if p.get("id")}
    stage1_by_id = {p.get("id"): p for p in stage1_points or [] if p.get("id")}
    source_by_id = {str(a.get("agenda_id")): a for a in source_agendas or [] if isinstance(a, dict)}
    inputs = {
        id(a): build_agenda_action_input(source_by_id.get(str(a.get("agenda_id")), a), polished_by_id, stage1_by_id)
        for a in targets
    }

    proposals: Dict[int, Tuple[List[Dict[str, Any]], str]] = {}
    if any(inputs.values()):
        owns_provider = provider is None
        if owns_provider:
            from services.ai_provider import get_provider
            provider = get_provider()
        try:
            for agenda in targets:
                entries = inputs[id(agenda)]
                if not entries:
                    continue
                items, ok = request_agenda_action_points(provider, agenda, entries, reasoning_mode)
                if ok:
                    proposals[id(agenda)] = (items, "llm")
                else:
                    logger.warning(f"[{recording_id}] [AgendaActions] {agenda.get('agenda_id')}: LLM failed, "
                                   f"using the points' extracted action items")
                    proposals[id(agenda)] = (fallback_items_from_entries(entries), "fallback_stage_items")
        finally:
            unload = getattr(provider, "unload_model", None)
            if owns_provider and callable(unload):
                unload()

    embedder, owns_embedder = None, False
    if EMBEDDING_DEDUPE and any(len(items) > 1 for items, _ in proposals.values()):
        try:
            from services.text_embedding_service import get_text_embedder
            embedder, owns_embedder = get_text_embedder(), True
        except Exception as exc:
            logger.warning(f"[AgendaActions] embedder unavailable, token-based dedupe only: {exc}")

    roster = _build_roster(agendas)
    audits: Dict[str, Dict[str, Any]] = {}
    try:
        # Tasks already listed under another included agenda are not repeated.
        target_ids = {id(a) for a in targets}
        claimed: List[Dict[str, Any]] = [
            ap for a in agendas
            if id(a) not in target_ids and has_agenda_action_points(a) and agenda_includes_action_points(a)
            for ap in a["action_points"]
        ]
        for agenda in targets:
            entries = inputs[id(agenda)]
            items, source = proposals.get(id(agenda), ([], "llm"))
            result = validate_action_points(items, entries, roster, embedder) if items else {
                "kept": [], "rejected": [], "adjusted": []}
            kept = []
            for item in result["kept"]:
                prior = next((c for c in claimed if not _owners_conflict(c, item) and
                              task_similarity(c["task"], item["task"]) >= DUPLICATE_TASK_THRESHOLD), None)
                if prior is not None:
                    result["rejected"].append({"task": item["task"], "reason": f"already listed under another agenda: {prior['task']}"})
                    continue
                kept.append(item)
            final = [finalize_action_point(item) for item in kept]
            claimed.extend(final)

            agenda["action_points"] = final
            agenda["action_points_mode"] = ACTION_POINTS_MODE
            agenda["include_action_points"] = True
            for point in agenda.get("discussion_points") or []:
                if isinstance(point, dict):
                    point.pop("action_points", None)
            audit = {
                "source": source if entries else "no_discussion_points",
                "generated": len(items),
                "kept": len(final),
                "rejected": result["rejected"],
                "adjusted": result["adjusted"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            agenda["action_point_audit"] = audit
            audits[str(agenda.get("agenda_id"))] = audit
            logger.info(
                f"[{recording_id}] [AgendaActions] {agenda.get('agenda_id')}: {len(items)} proposed, "
                f"{len(final)} kept, {len(result['rejected'])} rejected ({audit['source']})"
            )
    finally:
        if owns_embedder:
            from services.text_embedding_service import unload_text_embedder
            unload_text_embedder()
    return audits
