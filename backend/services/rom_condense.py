"""
Short/medium Final ROM versions: condensing one agenda's complete (long) discussion points.

The LLM sees every point with a short ID (P1, P2, ...), its Stage 2 type, speakers, owner and
flags (``format_points_for_condense``) and returns condensed points that cite the points they
were written from. ``assemble_condensed_points`` turns that output into Final ROM points
deterministically:

- speakers, owners, action items and timeline come from the cited points (never by position);
- a sentence stating a number or date that none of its points contains is dropped;
- Very Important points appear verbatim, once;
- repeated condensed points are merged;
- decision and action points the output lost are condensed again (or kept verbatim);
- the points keep meeting order.

Outputs without source IDs (custom prompts, trained variants) are linked to the points whose
wording and facts they cover. Each condensed point records why assembly changed it
(``condense_notes``), and the condensed agenda keeps the points it was written from
(``condense_sources``), so the explanation and Expand Point of a point see exactly its originals.
"""
import json
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from services.fact_grounding import (
    content_tokens,
    drop_unsupported_sentences,
    fact_tokens,
    mentions,
    source_facts,
    split_sentences,
    token_overlap,
)
from services.point_classification import (
    TRUSTED_SOURCES,
    dominant_type,
    heuristic_discussion_type,
    normalize_action_flag,
    normalize_discussion_type,
    point_text,
)

logger = logging.getLogger(__name__)

VERY_IMPORTANT_TAG = "*** VERY IMPORTANT - PRESERVE INTACT WITHOUT SUMMARIZATION ***"
# Outputs without source IDs: the best-covering point needs this share of the output's terms,
# each further point must add this much more.
INFER_MIN_FIRST = 0.2
INFER_MIN_GAIN = 0.25
# Two condensed points sharing this share of their terms (both ways) say the same thing.
DUPLICATE_OVERLAP = 0.8
# Numbers and month/weekday names must come from the cited points; an unknown acronym is often a
# legitimate abbreviation ("DB"), so it is only reported.
_HARD_FACT_PREFIXES = ("#", "@")
# Flags that rest on more than text cues.
_TRUSTED_FLAG_SOURCES = TRUSTED_SOURCES + ("stage1",)
# Expand Point: a new sentence needs this share of its terms in the point's sources; at most
# this many sentences are added per expansion.
EXPAND_MIN_SOURCE_OVERLAP = 0.5
EXPAND_MAX_SENTENCES = 3

_TEXT_KEYS = ("text", "polished_text", "point", "discussion", "summary", "content")
_SOURCE_KEYS = ("source_ids", "source_point_ids", "sources", "point_ids", "ids")


# ── Input ────────────────────────────────────────────────────────────────────

def _speakers(point: Dict[str, Any]) -> List[str]:
    speakers = point.get("speakers") or []
    if isinstance(speakers, str):
        speakers = [s.strip() for s in speakers.split(",") if s.strip()]
    if not speakers and point.get("speaker"):
        speakers = [point["speaker"]]
    return [str(s) for s in speakers if s]


def point_type(point: Dict[str, Any]) -> str:
    return normalize_discussion_type(point.get("type")) or heuristic_discussion_type(point_text(point))


def _flagged(point: Dict[str, Any]) -> bool:
    return bool(normalize_action_flag(point.get("include_action_point")))


def format_points_for_condense(points: List[Dict[str, Any]], marked: Iterable[Dict[str, Any]] = (),
                               labels: Optional[List[str]] = None) -> str:
    """One block per point: a header line (ID, type, speakers, owner, flags) and the full text.
    ``labels`` replace the IDs P1, P2, ... (points that keep their agenda-wide IDs)."""
    marked_keys = {id(p) for p in marked}
    blocks = []
    for i, point in enumerate(points, 1):
        header = [f"[{labels[i - 1] if labels else f'P{i}'}] Type={point_type(point)}"]
        speakers = _speakers(point)
        if speakers:
            header.append("Speakers=" + ", ".join(speakers))
        if point.get("action_owner"):
            header.append(f"Action Owner={point['action_owner']}")
        if _flagged(point):
            header.append("ACTION")
        if id(point) in marked_keys:
            header.append(VERY_IMPORTANT_TAG)
        blocks.append(" | ".join(header) + "\n" + point_text(point))
    return "\n\n".join(blocks)


def condense_source_snapshot(points: List[Dict[str, Any]], marked: Iterable[Dict[str, Any]] = ()) -> List[Dict[str, Any]]:
    """The agenda's complete points as the condense call saw them, under the same IDs (P1, P2, ...),
    with their point IDs, text and attribution. Stored with the condensed agenda, it keeps every
    condensed point traceable to its originals even after the long version changes."""
    marked_keys = {id(p) for p in marked}
    return [
        {
            "label": f"P{i}",
            "id": point.get("id"),
            "text": point_text(point),
            "speakers": _speakers(point),
            "action_owner": point.get("action_owner"),
            "type": point_type(point),
            "include_action_point": _flagged(point),
            "is_very_important": id(point) in marked_keys,
            "is_doc_point": bool(point.get("is_doc_point")),
            "timeline_start": float(point.get("timeline_start") or 0.0),
            "timeline_end": float(point.get("timeline_end") or 0.0),
            **{key: list(point.get(key) or []) for key in ("references", "technical_terms", "dates", "numbers")},
        }
        for i, point in enumerate(points, 1)
    ]


# ── Output parsing ───────────────────────────────────────────────────────────

def _repair_json(text: str) -> str:
    text = re.sub(r",\s*([\]\}])", r"\1", text)
    return re.sub(r'\\(?![/"\\bfnrtu])', r"\\\\", text)


def _json_list(candidate: str) -> Optional[list]:
    start, end = candidate.find("["), candidate.rfind("]")
    if start == -1 or end <= start:
        return None
    sub = candidate[start:end + 1]
    for attempt in (sub, _repair_json(sub)):
        try:
            data = json.loads(attempt)
        except Exception:
            continue
        if isinstance(data, list):
            return data
    # Several separate arrays: ["point 1"] ["point 2"]
    merged: list = []
    arrays = re.findall(r"\[[\s\S]*?\]", candidate)
    if len(arrays) > 1:
        for arr in arrays:
            try:
                data = json.loads(_repair_json(arr))
            except Exception:
                continue
            if isinstance(data, list):
                merged.extend(data)
    return merged or None


def _to_item(entry: Any) -> Optional[Dict[str, Any]]:
    if isinstance(entry, dict):
        text = next((entry[k] for k in _TEXT_KEYS if entry.get(k)), None)
        if not text or not str(text).strip():
            return None
        ids = next((entry[k] for k in _SOURCE_KEYS if entry.get(k)), [])
        if isinstance(ids, (str, int)):
            ids = [ids]
        ids = [str(i).strip() for i in ids if i is not None and str(i).strip()] if isinstance(ids, list) else []
        return {"text": str(text).strip(), "type": entry.get("type"), "source_ids": ids}
    if entry is not None and str(entry).strip():
        return {"text": str(entry).strip(), "type": None, "source_ids": []}
    return None


def parse_condensed_items(raw: Any) -> List[Dict[str, Any]]:
    """[{text, type, source_ids}] from an LLM response: a JSON array of objects or strings
    (possibly fenced, after <think> blocks, with trailing commas), or a bulleted/numbered list."""
    if not raw:
        return []
    text = re.sub(r"<think>.*?</think>", "", str(raw).strip(), flags=re.DOTALL).strip()
    if "<think>" in text:
        text = text.split("</think>")[-1].strip() if "</think>" in text else text.split("<think>")[0].strip()

    for candidate in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text) or [text]:
        data = _json_list(candidate.strip())
        if data:
            items = [item for item in (_to_item(e) for e in data) if item]
            if items:
                return items

    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```") or line.startswith("#"):
            continue
        m = re.match(r"^(?:\d+[\.\)]|[-*•])\s*(.+)$", line)
        if m:
            item = m.group(1).strip()
            if len(item) > 1 and item[0] == item[-1] and item[0] in "\"'":
                item = item[1:-1].strip()
            if item:
                items.append({"text": item, "type": None, "source_ids": []})
    return items


# ── Assembly ─────────────────────────────────────────────────────────────────

def _terms(text: str) -> set:
    return content_tokens(text) | fact_tokens(text)


def _point_facts(point: Dict[str, Any]) -> set:
    return source_facts([
        point_text(point), point.get("dates") or [], point.get("numbers") or [],
        point.get("technical_terms") or [], point.get("references") or [],
    ])


class _Scope:
    """The points an output may cite, with their short IDs and precomputed terms."""

    def __init__(self, points: List[Dict[str, Any]]):
        self.points = points
        self.aliases = {f"P{i}": p for i, p in enumerate(points, 1)}
        self.by_id = {str(p.get("id")): p for p in points if p.get("id")}
        self.terms = [(_terms(point_text(p)), p) for p in points]

    def resolve(self, raw_ids: List[str]) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        for raw in raw_ids:
            key = str(raw).strip().strip("[]()").strip()
            point = self.aliases.get(key.upper()) or (self.aliases.get(f"P{key}") if key.isdigit() else None) \
                or self.by_id.get(key)
            if point is not None and all(point is not f for f in found):
                found.append(point)
        return found

    def infer(self, text: str) -> List[Dict[str, Any]]:
        """Greedy cover of the output's terms by the points' terms."""
        terms = _terms(text)
        chosen: List[Dict[str, Any]] = []
        covered: set = set()
        while terms:
            gain, point = max(((len((t & terms) - covered), p) for t, p in self.terms
                               if all(p is not c for c in chosen)), key=lambda gp: gp[0], default=(0, None))
            if point is None or gain / len(terms) < (INFER_MIN_GAIN if chosen else INFER_MIN_FIRST):
                break
            chosen.append(point)
            covered |= _terms(point_text(point)) & terms
        return chosen


def _drafts(items: List[Dict[str, Any]], scope: _Scope, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    drafts = []
    for item in items:
        notes: List[str] = []
        sources = scope.resolve(item.get("source_ids") or [])
        if not sources:
            sources = scope.infer(item["text"])
            if sources:
                report["inferred_sources"] += 1
                notes.append("The model did not cite its source points; they were matched by wording.")
        drafts.append({"text": item["text"], "type": item.get("type"), "sources": sources, "very_important": False,
                       "notes": notes})
    return drafts


def _readable_fact(token: str) -> str:
    return token[1:].capitalize() if token[0] == "@" else token[1:]


def _ground(draft: Dict[str, Any], points: List[Dict[str, Any]], facts_by_point: Dict[int, set],
            report: Dict[str, Any], no_cite: set, labels: Dict[int, str]) -> None:
    """Cite the agenda point a stray fact comes from; drop sentences with numbers/dates no point states.
    Points in ``no_cite`` (Very Important, shown verbatim) support facts without being cited."""
    facts = set().union(*(facts_by_point[id(s)] for s in draft["sources"])) if draft["sources"] else set()
    extra = fact_tokens(draft["text"]) - facts
    if extra:
        for point in points:
            if any(point is s for s in draft["sources"]) or not extra & facts_by_point[id(point)]:
                continue
            if id(point) not in no_cite:
                draft["sources"].append(point)
                stated = ", ".join(sorted(_readable_fact(t) for t in extra & facts_by_point[id(point)]))
                draft["notes"].append(f"{labels[id(point)]} added as a source: the point states {stated}, which comes from it.")
            facts |= facts_by_point[id(point)]
            extra -= facts_by_point[id(point)]
            if not extra:
                break
    soft = {t for t in extra if t[0] not in _HARD_FACT_PREFIXES}
    if extra - soft:
        draft["text"], removed = drop_unsupported_sentences(draft["text"], facts | soft)
        report["dropped_sentences"].extend(removed)
        draft["notes"].extend(
            f"Removed in fact check: “{s}” states a number or date that none of its source points contains."
            for s in removed)
    if soft:
        report["unverified_terms"].extend(sorted(t[1:] for t in soft))


def _add_note(draft: Dict[str, Any], note: str) -> None:
    if note not in draft["notes"]:
        draft["notes"].append(note)


def _same_point(a: str, b: str) -> bool:
    return token_overlap(a, b) >= DUPLICATE_OVERLAP and token_overlap(b, a) >= DUPLICATE_OVERLAP


def _merge_duplicates(drafts: List[Dict[str, Any]], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for draft in drafts:
        match = None if draft["very_important"] else next(
            (k for k in kept if not k["very_important"] and _same_point(k["text"], draft["text"])), None)
        if match is None:
            kept.append(draft)
            continue
        report["merged_duplicates"] += 1
        match["sources"] += [s for s in draft["sources"] if all(s is not m for m in match["sources"])]
        if len(draft["text"]) > len(match["text"]):
            match["text"] = draft["text"]
        for note in draft["notes"]:
            _add_note(match, note)
        _add_note(match, "Merged with a near-duplicate condensed point that said the same thing.")
    return kept


def _must_keep(point: Dict[str, Any]) -> bool:
    """A decision or an action that the record may not lose (per a model, a person or Stage 1)."""
    decision = point_type(point) == "decision" and point.get("type_source") in TRUSTED_SOURCES
    action = _flagged(point) and point.get("action_flag_source") in _TRUSTED_FLAG_SOURCES
    return decision or action


def _ordered(drafts: List[Dict[str, Any]], position: Dict[int, int]) -> List[Dict[str, Any]]:
    """Meeting order: earliest source start, then earliest source position; drafts without
    sources stay after the draft they followed in the output."""
    keyed = []
    last = (float("-inf"), -1)
    for seq, draft in enumerate(drafts):
        if draft["sources"]:
            last = (min(float(s.get("timeline_start") or 0.0) for s in draft["sources"]),
                    min(position[id(s)] for s in draft["sources"]))
        keyed.append((last, seq, draft))
    return [d for _, _, d in sorted(keyed, key=lambda k: (k[0], k[1]))]


def _mentioned(values_by_source: Iterable[Any], text: str) -> List[str]:
    out: List[str] = []
    for values in values_by_source:
        for value in values or []:
            value = str(value).strip()
            if value and value not in out and mentions(text, value):
                out.append(value)
    return out


def _build_point(draft: Dict[str, Any], point_id: str, claimed_actions: set) -> Dict[str, Any]:
    from services.action_owner_validator import actions_are_duplicates
    from services.rom_service import _carry_stage1_attribution, normalize_stage2_action_item

    sources = draft["sources"]
    text = draft["text"]
    point: Dict[str, Any] = {"id": point_id, "text": text, "polished_text": text}
    # Owners (validated union of the sources), owner evidence and transcript excerpt.
    _carry_stage1_attribution(point, sources, carry_actions=False)
    speakers = list(dict.fromkeys(s for src in sources for s in _speakers(src)))
    point["speakers"] = speakers
    point["speaker"] = speakers[0] if speakers else "Unknown"

    # Each source's action items go to the first condensed point that cites it.
    actions: List[Dict[str, Any]] = []
    for src in sources:
        if id(src) in claimed_actions:
            continue
        claimed_actions.add(id(src))
        for raw in src.get("action_items") or []:
            norm = normalize_stage2_action_item(raw)
            if norm["task"] and not any(actions_are_duplicates(a, norm) for a in actions):
                actions.append(norm)
    point["action_items"] = actions

    point["timeline_start"] = min((float(s.get("timeline_start") or 0.0) for s in sources), default=0.0)
    point["timeline_end"] = max((float(s.get("timeline_end") or 0.0) for s in sources), default=0.0)
    for key in ("references", "technical_terms", "dates", "numbers"):
        values = [s.get(key) for s in sources]
        if draft["very_important"]:
            point[key] = list(dict.fromkeys(str(v) for vals in values for v in vals or []))
        else:
            point[key] = _mentioned(values, text)

    point["type"] = normalize_discussion_type(draft.get("type")) or dominant_type(point_type(s) for s in sources) \
        or heuristic_discussion_type(text)
    point["include_action_point"] = any(_flagged(s) for s in sources)
    source_ids = [s.get("id") for s in sources if s.get("id")]
    point["source_point_ids"] = source_ids
    point["original_point_ids"] = source_ids
    point["original_point_id"] = source_ids[0] if source_ids else None
    point["is_very_important"] = draft["very_important"]
    point["condense_notes"] = list(draft.get("notes") or [])
    return point


def assemble_condensed_points(
    agenda_id: str,
    version: str,
    points: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    marked: List[Dict[str, Any]],
    repair: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Final ROM points for one agenda from the parsed LLM output. ``repair(points)`` condenses
    the given points again (same output format, IDs P1.. relative to them). Returns (points, report)."""
    report: Dict[str, Any] = {
        "version": version, "outputs": len(items), "inferred_sources": 0, "dropped_sentences": [],
        "unverified_terms": [], "merged_duplicates": 0, "repaired": [], "kept_verbatim": [],
    }
    scope = _Scope(points)
    position = {id(p): i for i, p in enumerate(points)}
    labels = {id(p): f"P{i}" for i, p in enumerate(points, 1)}
    facts_by_point = {id(p): _point_facts(p) for p in points}
    drafts = _drafts(items, scope, report)

    # Very Important points: verbatim, exactly once; other outputs stop citing them.
    for vip in marked:
        solo = [d for d in drafts if len(d["sources"]) == 1 and d["sources"][0] is vip]
        for d in drafts:
            if len(d["sources"]) > 1 and any(s is vip for s in d["sources"]):
                d["sources"] = [s for s in d["sources"] if s is not vip]
                _add_note(d, f"{labels[id(vip)]} is marked Very Important and appears verbatim as its own point.")
        verbatim = {"text": point_text(vip), "type": None, "sources": [vip], "very_important": True,
                    "notes": ["Marked Very Important: kept verbatim, not condensed."]}
        if solo:
            drafts = [verbatim if d is solo[0] else d for d in drafts if not any(d is s for s in solo[1:])]
        else:
            drafts.append(verbatim)

    no_cite = {id(p) for p in marked}
    orphans: List[Dict[str, Any]] = []
    grounded = []
    for draft in drafts:
        if not draft["very_important"]:
            _ground(draft, points, facts_by_point, report, no_cite, labels)
        if draft["text"].strip():
            grounded.append(draft)
        else:
            orphans.extend(draft["sources"])  # everything the output said about them was unsupported
    drafts = _merge_duplicates(grounded, report)

    cited = {id(s) for d in drafts for s in d["sources"]}
    missing = [p for p in points if id(p) not in cited and (_must_keep(p) or any(p is o for o in orphans))]
    if missing:
        repaired: List[Dict[str, Any]] = []
        if repair is not None:
            try:
                repair_scope = _Scope(missing)
                for draft in _drafts(repair(missing) or [], repair_scope, report):
                    if not draft["sources"]:
                        continue
                    _ground(draft, points, facts_by_point, report, no_cite, labels)
                    if draft["text"].strip():
                        lost = [labels[id(s)] for s in draft["sources"] if any(s is m for m in missing)]
                        if lost:
                            _add_note(draft, f"Condensed in a second pass: the first output left out {', '.join(lost)}.")
                        repaired.append(draft)
            except Exception as exc:
                logger.warning(f"[RomCondense] {agenda_id} ({version}): repair failed: {exc}")
        report["repaired"] = [str(s.get("id")) for d in repaired for s in d["sources"]]
        covered = {id(s) for d in repaired for s in d["sources"]}
        for point in missing:
            if id(point) not in covered:
                if any(point is o for o in orphans):
                    reason = "everything the condensed output said about it failed the fact check"
                else:
                    reason = f"the condensed output left out this {'decision' if point_type(point) == 'decision' else 'action'}"
                repaired.append({"text": point_text(point), "type": None, "sources": [point], "very_important": False,
                                 "notes": [f"Kept verbatim: {reason}."]})
                report["kept_verbatim"].append(str(point.get("id")))
        drafts = _merge_duplicates(drafts + repaired, report)

    claimed: set = set()
    new_points = [
        _build_point(draft, f"{agenda_id or 'A'}-{version.upper()}-{n}", claimed)
        for n, draft in enumerate(_ordered(drafts, position), 1)
    ]
    report["points"] = len(new_points)
    return new_points, report


# ── Explanation and Expand Point ─────────────────────────────────────────────

def format_condense_notes(notes: Iterable[Any]) -> str:
    """The ``{notes_section}`` of the explanation prompt: a point's condense_notes, or "" without notes."""
    lines = [f"- {n}" for n in notes if n]
    if not lines:
        return ""
    return "\nPIPELINE NOTES (changes made after the model wrote the point):\n" + "\n".join(lines) + "\n"


def ground_expansion(candidates: Iterable[Any], current_text: str,
                     sources: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """(sentences to append to a condensed point, rejected sentences). A sentence is kept when
    every number and date it states comes from the point's sources, most of its terms do too,
    and it does not repeat the point or a sentence already kept; at most EXPAND_MAX_SENTENCES."""
    facts = set().union(*(_point_facts(s) for s in sources)) if sources else set()
    source_text = " ".join(point_text(s) for s in sources)
    kept: List[str] = []
    dropped: List[str] = []
    for candidate in candidates:
        for sentence in split_sentences(candidate):
            invented = any(t[0] in _HARD_FACT_PREFIXES for t in fact_tokens(sentence) - facts)
            repeated = any(token_overlap(sentence, other) >= DUPLICATE_OVERLAP for other in [current_text, *kept])
            off_source = token_overlap(sentence, source_text) < EXPAND_MIN_SOURCE_OVERLAP
            if invented or repeated or off_source or len(kept) >= EXPAND_MAX_SENTENCES:
                dropped.append(sentence)
            else:
                kept.append(sentence)
    return kept, dropped


def refresh_mentions(point: Dict[str, Any], sources: List[Dict[str, Any]]) -> None:
    """The point's references, terms, dates and numbers: the sources' values its text mentions."""
    text = point_text(point)
    for key in ("references", "technical_terms", "dates", "numbers"):
        point[key] = _mentioned((s.get(key) for s in sources), text)
