"""
agenda_doc_points.py — PPT Testing logic applied to Stage 3 (Agenda Matching).

When an agenda document (PPT/PDF/DOCX) is uploaded for Stage 3:
  1. analyze_agenda_documents   — the PPT Testing slide analysis (ppt_insights.analyze_presentation)
                                   turns the document into prioritised, de-duplicated "doc points".
  2. assign_insights_to_agendas — each doc point goes to the agenda it belongs to (slide title or
                                   the Stage 3 point↔agenda similarity); points matching no agenda
                                   are dropped as noise.
  3. doc points are extra context when mapping transcript points to agendas (similarity facet and
     the LLM agenda reference, see rom_service.map_points_to_agendas).
  4. refine_agenda_points       — per agenda, the PPT matching step (ppt_matching.match_points):
                                   verified matches are rewritten with the document's terminology,
                                   important unmatched doc points are kept, the rest dropped, and
                                   points that say the same thing merged (_merge_duplicates).

Nothing here runs unless the meeting has an agenda document.
"""
from __future__ import annotations

import copy
import logging
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from services.fact_grounding import fact_tokens, token_overlap
from services.ppt_insights import (
    InferFn,
    _fuzz,
    analyze_presentation,
    insight_point_text,
    parse_dates,
    slides_from_dicts,
)
from services.ppt_matching import EmbedFn, _keywords, compose_point_text, insight_match_text, match_points

logger = logging.getLogger(__name__)

# Agenda documents analysed with the PPT pipeline (other agenda files are text only).
DOC_EXTENSIONS = {"ppt", "pptx", "pdf", "doc", "docx"}
# Agenda decks often mention dates well before/after the meeting.
AGENDA_DOC_FOCUS = "wide"
# Best agenda similarity a doc point needs to be kept (lower = unrelated to every agenda).
MIN_INSIGHT_AGENDA_SCORE = 0.35
# Slide title ≈ agenda title → that agenda, regardless of similarity.
SLIDE_TITLE_MATCH = 80
# Doc points shown to the LLM per agenda in the agenda reference.
MAX_REFERENCE_DOC_POINTS = 5
# A refined point must keep at least this share of the discussion point's content words;
# otherwise the refinement is rejected and the discussion point is kept as said.
MIN_CONTENT_COVERAGE = 0.5
# The long ROM allows more detail, never less: a refinement much shorter than the point is a summary.
MIN_LENGTH_RATIO = 0.8
# Two points are duplicates when each shares this much of its wording with the other (and both
# state the same facts). One-directional (subset) matching would drop a short point into a long one.
DUPLICATE_POINT_OVERLAP = 0.85


def content_coverage(original: str, refined: str) -> float:
    """Share of the original's content words (ppt_matching keywords) still present in the rewrite."""
    words = {w for w, _ in _keywords(original)}
    if not words:
        return 1.0
    kept = {w for w, _ in _keywords(refined)}
    return len(words & kept) / len(words)


def dropped_dates(original: str, refined: str) -> List[str]:
    """Day-precise dates stated in the discussion point that the rewrite no longer contains."""
    kept = {(s.start, s.end) for s in parse_dates(refined) if s.precision == "day"}
    return [s.raw for s in parse_dates(original) if s.precision == "day" and (s.start, s.end) not in kept]


def is_agenda_document(filename: str) -> bool:
    return (filename.rsplit(".", 1)[-1].lower() if "." in (filename or "") else "") in DOC_EXTENSIONS


def provider_infer_fn() -> InferFn:
    """Synchronous `infer(prompt, max_tokens)` bound to the configured LLM (thinking disabled)."""
    from services.ai_provider import get_provider
    provider = get_provider()
    return lambda prompt, max_tokens: provider._infer(prompt, max_tokens, None, False)


# ── 1. Doc points from the agenda document(s) ────────────────────────────────

def analyze_agenda_documents(documents: List[Dict[str, Any]], meeting: date, infer: InferFn) -> List[Dict[str, Any]]:
    """
    Doc points (PPT insights) of every agenda document. The analysis is cached on each
    document entry (``doc["analysis"]``) per meeting date, so callers should save the
    documents back afterwards.
    """
    insights: List[Dict[str, Any]] = []
    for doc in documents or []:
        if not isinstance(doc, dict) or not doc.get("slides"):
            continue
        filename = doc.get("filename") or "Agenda document"
        cached = doc.get("analysis")
        if (isinstance(cached, dict) and cached.get("meeting_date") == meeting.isoformat()
                and cached.get("focus") == AGENDA_DOC_FOCUS):
            doc_insights = cached.get("insights") or []
        else:
            analysis = analyze_presentation(slides_from_dicts(doc["slides"]), meeting=meeting, infer=infer,
                                            focus=AGENDA_DOC_FOCUS, filename=filename)
            doc_insights = analysis.get("insights") or []
            doc["analysis"] = {
                "meeting_date": meeting.isoformat(),
                "focus": AGENDA_DOC_FOCUS,
                "insights": doc_insights,
                "slides_total": analysis.get("slides_total", 0),
                "slides_analyzed": analysis.get("slides_analyzed", 0),
            }
            logger.info(f"[AgendaDocPoints] '{filename}': {len(doc_insights)} doc point(s) from "
                        f"{analysis.get('slides_analyzed', 0)}/{analysis.get('slides_total', 0)} slide(s)")
        for ins in doc_insights:
            item = copy.deepcopy(ins)
            item["doc_name"] = filename
            insights.append(item)
    for n, ins in enumerate(insights, 1):
        ins["ref"] = f"P{n}"
    return insights


# ── 2. Doc points → agendas ──────────────────────────────────────────────────

def _title_score(slide_title: str, agenda_title: str) -> float:
    a, b = (slide_title or "").strip(), (agenda_title or "").strip()
    if len(a) < 4 or len(b) < 4:
        return 0.0
    fuzz = _fuzz()
    # token_set_ratio is 100 for any subset, so a one-word slide title ("Status") would match
    # every agenda containing that word: require two meaningful words for subset matching.
    if min(len([w for w in a.split() if len(w) >= 3]), len([w for w in b.split() if len(w) >= 3])) >= 2:
        return float(fuzz.token_set_ratio(a.lower(), b.lower()))
    return float(fuzz.ratio(a.lower(), b.lower()))


def assign_insights_to_agendas(
    insights: List[Dict[str, Any]],
    agendas: List[Dict[str, Any]],
    embedder=None,
    expanded_agendas: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """{agenda_id: [doc point]}. A slide titled like an agenda goes to that agenda; otherwise the
    agenda with the highest Stage 3 similarity, if it clears MIN_INSIGHT_AGENDA_SCORE."""
    if not insights or not agendas:
        return {}
    from services.rom_service import compute_point_agenda_similarity

    candidates: Dict[str, Dict[str, Any]] = {}
    if embedder is not None:
        try:
            sim = compute_point_agenda_similarity(
                [{"id": ins["id"], "text": insight_match_text(ins)} for ins in insights],
                agendas, embedder, expanded_agendas=expanded_agendas,
            )
            candidates = {c["point_id"]: c for c in sim["candidates"]}
        except Exception as exc:
            logger.warning(f"[AgendaDocPoints] Similarity unavailable, using title/lexical matching only: {exc}")

    fuzz = _fuzz()
    assigned: Dict[str, List[Dict[str, Any]]] = {}
    dropped = 0
    for ins in insights:
        agenda_id, how = None, ""
        best_title = max(((_title_score(ins.get("slide_title", ""), a.get("title", "")), a["agenda_id"]) for a in agendas),
                         default=(0.0, None))
        if best_title[0] >= SLIDE_TITLE_MATCH:
            agenda_id, how = best_title[1], "slide_title"
        elif candidates:
            ranked = (candidates.get(ins["id"]) or {}).get("candidates") or []
            if ranked and ranked[0]["score"] >= MIN_INSIGHT_AGENDA_SCORE:
                agenda_id, how = ranked[0]["agenda_id"], f"similarity {ranked[0]['score']:.2f}"
        else:
            text = insight_match_text(ins).lower()
            lexical = max(((fuzz.token_set_ratio(text, f"{a.get('title', '')} {a.get('description', '')}".lower()),
                            a["agenda_id"]) for a in agendas), default=(0.0, None))
            if lexical[0] >= 60:
                agenda_id, how = lexical[1], "lexical"
        if not agenda_id:
            dropped += 1
            continue
        ins["agenda_id"] = agenda_id
        ins["agenda_match"] = how
        assigned.setdefault(agenda_id, []).append(ins)
    logger.info(f"[AgendaDocPoints] {len(insights) - dropped} doc point(s) assigned to {len(assigned)} agenda(s); "
                f"{dropped} matched no agenda (dropped)")
    return assigned


def to_agenda_doc_points(
    assigned: Dict[str, List[Dict[str, Any]]],
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge assigned doc points into the Stage 3 ``agenda_doc_points`` shape
    ({agenda_id: {doc_name, points: [str], presenter}}) next to per-agenda uploaded documents."""
    merged: Dict[str, Any] = copy.deepcopy(existing or {})
    for agenda_id, items in assigned.items():
        entry = merged.get(agenda_id) if isinstance(merged.get(agenda_id), dict) else {}
        points = list(entry.get("points") or [])
        points.extend(insight_point_text(i) for i in items)
        merged[agenda_id] = {
            **entry,
            "doc_name": entry.get("doc_name") or items[0].get("doc_name", "Agenda document"),
            "points": points,
        }
    return merged


def reference_doc_points(assigned: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    """Short doc-point headlines per agenda for the LLM agenda reference (highest priority first)."""
    return {
        agenda_id: [i.get("headline") or i.get("summary", "")
                    for i in sorted(items, key=lambda x: x.get("priority", 0), reverse=True)][:MAX_REFERENCE_DOC_POINTS]
        for agenda_id, items in assigned.items()
    }


# ── 4. Refine the matched points of one agenda ───────────────────────────────

def _doc_point(ins: Dict[str, Any]) -> Dict[str, Any]:
    """An important unmatched doc point, in the same shape as Stage 3 document points."""
    import uuid
    text = insight_point_text(ins)
    doc_name = ins.get("doc_name", "Agenda document")
    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "polished_text": text,
        "speaker": "From Document",
        "speakers": ["From Document"],
        "action_owner": None,
        "decisions": [],
        "action_items": [],
        "technical_terms": [],
        "dates": ins.get("dates") or [],
        "numbers": [],
        "references": [doc_name],
        "timeline_start": 0.0,
        "timeline_end": 0.0,
        "is_doc_point": True,
        "from_agenda_document": True,
        "doc_slides": ins.get("slides") or [],
        "assignment_confidence": "high",
        "assignment_reason": f"Important point from the agenda document: {doc_name}",
        "is_probable": False,
    }


def _point_text(point: Dict[str, Any]) -> str:
    return str(point.get("text") or point.get("polished_text") or "").strip()


def _same_point(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Both points say the same thing: most words shared in BOTH directions (a short point whose
    words are a subset of a longer one is not a duplicate of it) and the same facts (numbers,
    dates, acronyms)."""
    ta, tb = _point_text(a), _point_text(b)
    if not ta or not tb:
        return False
    if token_overlap(ta, tb) < DUPLICATE_POINT_OVERLAP or token_overlap(tb, ta) < DUPLICATE_POINT_OVERLAP:
        return False
    return fact_tokens(ta) == fact_tokens(tb)


def _point_rank(point: Dict[str, Any]) -> tuple:
    """Which of two duplicates survives: document-refined > transcript > document-only, then the longer."""
    kind = 2 if point.get("doc_refined") else (0 if point.get("is_doc_point") else 1)
    return kind, len(_point_text(point))


def _fold_into(keep: Dict[str, Any], dup: Dict[str, Any]) -> None:
    """Keep what the duplicate adds: its speakers, owners, action items and source ids."""
    from services.action_owner_validator import actions_are_duplicates, split_owner_names
    from services.rom_service import normalize_stage2_action_item

    speakers = [s for s in keep.get("speakers") or [] if s]
    for spk in dup.get("speakers") or []:
        if spk and spk not in speakers and not (spk == "From Document" and speakers):
            speakers.append(spk)
    if speakers:
        keep["speakers"] = speakers
    owners = list(dict.fromkeys(split_owner_names(keep.get("action_owner")) + split_owner_names(dup.get("action_owner"))))
    if owners:
        keep["action_owner"] = ", ".join(owners)
    actions = list(keep.get("action_items") or [])
    for act in dup.get("action_items") or []:
        norm = normalize_stage2_action_item(act)
        if norm["task"] and not any(actions_are_duplicates(normalize_stage2_action_item(a), norm) for a in actions):
            actions.append(act)
    keep["action_items"] = actions
    source_ids = list(dict.fromkeys((keep.get("original_point_ids") or []) + (dup.get("original_point_ids") or [])))
    if source_ids:
        keep["original_point_ids"] = source_ids
    keep["merged_point_ids"] = list(keep.get("merged_point_ids") or []) + [str(dup.get("id"))]


def _merge_duplicates(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Points that say the same thing (_same_point) are merged into the stronger one
    (document-refined > transcript > document-only); the merged point keeps the other's speakers,
    owners, action items and source ids, and a transcript point is never dropped in favour of a
    document-only point. Surviving points keep their original order."""
    groups: List[List[int]] = []
    for idx, point in enumerate(points):
        group = next((g for g in groups if _same_point(points[g[0]], point)), None) if _point_text(point) else None
        if group is None:
            groups.append([idx])
        else:
            group.append(idx)
    survivors = []
    for group in groups:
        best = max(group, key=lambda i: (_point_rank(points[i]), -i))
        for i in group:
            if i != best:
                _fold_into(points[best], points[i])
        survivors.append(best)
    return [points[i] for i in sorted(survivors)]


def refine_agenda_points(
    agenda_points: List[Dict[str, Any]],
    insights: List[Dict[str, Any]],
    *,
    infer: InferFn,
    embed: Optional[EmbedFn] = None,
    meeting: Optional[date] = None,
    detail: str = "long",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    The PPT matching refinement for one agenda. Returns (points, stats).
    Transcript points are never dropped by the refinement caps: unverified points keep their text.
    """
    if not insights:
        return agenda_points, {}
    transcript = [p for p in agenda_points if not p.get("is_doc_point")]
    long_points = [{"id": str(p.get("id")), "text": (p.get("text") or p.get("polished_text") or "").strip()}
                   for p in transcript]
    long_points = [lp for lp in long_points if lp["text"]]

    result = match_points(copy.deepcopy(insights), long_points, infer=infer, embed=embed, meeting=meeting, detail=detail)

    by_id = {str(p.get("id")): p for p in agenda_points}
    refined = rejected = 0
    for group in result.get("comparison_groups", []):
        point = by_id.get(str(group["stage2_point"]["id"]))
        synth = group.get("final") or {}
        # The ROM keeps the refined discussion point itself (no meeting-prep headline).
        new_text = compose_point_text("", synth.get("summary", ""), synth.get("attention", "")) if synth else ""
        if point is None or not new_text:
            continue
        original = point.get("text") or point.get("polished_text") or ""
        coverage = content_coverage(original, new_text)
        lost_dates = dropped_dates(original, new_text)
        too_short = len(new_text) < MIN_LENGTH_RATIO * len(original)
        if coverage < MIN_CONTENT_COVERAGE or lost_dates or too_short:
            # The rewrite lost what was said, drifted to the slide's subject or changed a stated date:
            # keep the discussion point as it was said.
            logger.info(f"[AgendaDocPoints] Refinement of point {point.get('id')} rejected "
                        f"(keeps {coverage:.0%} of its content words, {len(new_text)}/{len(original)} chars; "
                        f"dates lost: {lost_dates or 'none'})")
            point["doc_match"] = {"confidence": group.get("confidence"), "relationship": group.get("relationship"),
                                  "doc_points": group.get("ppt_points", []), "refinement_rejected": True}
            rejected += 1
            continue
        point["original_text"] = original
        point["text"] = new_text
        point["polished_text"] = new_text
        point["doc_refined"] = True
        point["doc_slides"] = group.get("slides", [])
        point["doc_match"] = {"confidence": group.get("confidence"), "relationship": group.get("relationship"),
                              "doc_points": group.get("ppt_points", [])}
        refined += 1

    by_insight_id = {i["id"]: i for i in insights}
    important = [by_insight_id[u["id"]] for u in result.get("unmatched_priority_insights", []) if u["id"] in by_insight_id]
    points = list(agenda_points) + [_doc_point(i) for i in important]
    before = len(points)
    points = _merge_duplicates(points)
    stats = {
        "doc_points": len(insights),
        "refined_points": refined,
        "refinements_rejected": rejected,
        "important_doc_points_added": len(important),
        "doc_points_dropped": len(insights) - sum(len(g.get("insight_ids", [])) for g in result.get("comparison_groups", [])) - len(important),
        "duplicates_merged": before - len(points),
    }
    return points, stats


def summarize_assignment(assigned: Dict[str, List[Dict[str, Any]]], total: int) -> Dict[str, Any]:
    """What is stored in stage3.agenda_document_insights for display."""
    return {
        "total_doc_points": total,
        "assigned": {aid: [{"ref": i.get("ref"), "text": insight_point_text(i), "slides": i.get("slides", []),
                            "priority": i.get("priority"), "match": i.get("agenda_match")} for i in items]
                     for aid, items in assigned.items()},
    }
