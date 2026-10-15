"""
Discussion analysis for Stage 3 agenda generation.

Before agendas are finalised the whole discussion (Stage 2 points) is analysed:

1. every point is embedded with the active embedding model (trained variant if selected);
2. related points are grouped chronologically and near-duplicates are flagged;
3. points are classified as decision / action item / question / risk / discussion (from their
   Stage 2 type when the enhancement model or the user set one);
4. the LLM writes one titled topic per group, citing the points behind every statement;
5. topics are validated for relevance, duplication, coverage and unsupported facts;
6. topics are matched to existing agendas by semantic similarity, with scores and evidence.

With no agenda document the topics become the agendas; otherwise topics that match no
existing agenda are offered to the user as suggested agendas.
"""
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from services.fact_grounding import (
    content_tokens,
    drop_unsupported_sentences,
    fact_tokens,
    source_facts,
    split_sentences,
    token_overlap,
    unsupported_facts,
)
from services.point_classification import ACTION_RE, DECISION_RE, QUESTION_RE, RISK_RE, analysis_type_from_stage2

logger = logging.getLogger(__name__)

POINT_TYPES = ("decision", "action_item", "question", "risk", "discussion")
_TYPE_LISTS = {"decision": "decisions", "action_item": "action_items", "question": "open_questions", "risk": "risks"}
_KEYWORD_FIELDS = ("keywords", "related_concepts", "alternative_terminology", "expected_themes")

MAX_TOPICS = 12
# Embeddings of one meeting share a large common component (all points look ~0.8 alike with
# mean pooling), so grouping, duplicate-topic checks and agenda matching use vectors centred on
# the meeting's mean point vector; the resulting cosines separate subjects far better.
# Grouping: average linkage on centred cosine + a time-proximity term (meetings move through
# topics in order); groups merge while their similarity is above the meeting's average.
TIME_WEIGHT = 0.5
TIME_SCALE = 0.2  # fraction of the meeting span over which time proximity decays
CLUSTER_STD_FACTOR = 0.0
# Near-duplicate points: almost identical embeddings AND wording, and no conflicting facts.
NEAR_DUPLICATE_THRESHOLD = 0.92
NEAR_DUPLICATE_MIN_OVERLAP = 0.6
TOPIC_DUPLICATE_THRESHOLD = 0.7   # centred cosine between topic summaries
TOPIC_POINT_OVERLAP = 0.5
# Topic ↔ existing agenda, with the same similarity as Stage 3 mapping. Absolute scores are not
# comparable across meetings, so a topic is unmatched when its points are clearly less similar to
# every agenda than the meeting's typical point: mean(best score of its points) − median(best score
# of all points) < −TOPIC_UNMATCHED_GAP. (Measured: on-agenda topics ≈ −0.01…+0.08, an
# off-agenda topic −0.21.)
TOPIC_UNMATCHED_GAP = 0.10
# When most of a topic's points prefer one agenda, that agenda wins over the summary's best match.
TOPIC_MATCH_MIN_VOTE = 0.6
# Average Stage 2 points per topic: short meetings are not split into one-point topics.
MIN_POINTS_PER_TOPIC = 3
CLASSIFY_BATCH_SIZE = 20
CLASSIFY_CHAR_BUDGET = 10000
SYNTHESIS_CHAR_BUDGET = 12000
MIN_POINT_CHARS_FOR_SYNTHESIS = 240
EVIDENCE_MAX_CHARS = 200

_TYPE_ALIASES = {
    "decision": "decision", "decisions": "decision", "agreed": "decision",
    "action": "action_item", "action_item": "action_item", "action_items": "action_item", "task": "action_item",
    "commitment": "action_item",
    "question": "question", "open_question": "question", "questions": "question", "clarification": "question",
    "risk": "risk", "risks": "risk", "issue": "risk", "concern": "risk", "blocker": "risk", "dependency": "risk",
    "discussion": "discussion", "general_discussion": "discussion", "update": "discussion", "information": "discussion",
    "info": "discussion", "status_update": "discussion",
}
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*|[^\sA-Za-z0-9]+|\s+")


# ── Small helpers ────────────────────────────────────────────────────────────

def _point_text(point: Dict[str, Any]) -> str:
    return str(point.get("polished_text") or point.get("text") or point.get("discussion_point") or "").strip()


def _clean_list(values: Any, limit: int = 15) -> List[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    out: List[str] = []
    for v in values:
        s = str(v).strip() if v is not None else ""
        if s and s.lower() not in {o.lower() for o in out}:
            out.append(s)
    return out[:limit]


def _start(point: Dict[str, Any]) -> float:
    try:
        return float(point.get("timeline_start") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / np.where(norms < 1e-9, 1.0, norms)


def _embed(embedder, texts: List[str]) -> np.ndarray:
    return _normalize(embedder.encode_batch(texts))


def _point_facts(point: Dict[str, Any]) -> set:
    return source_facts([
        _point_text(point), point.get("dates") or [], point.get("numbers") or [],
        point.get("technical_terms") or [], point.get("references") or [],
    ])


def _format_time(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


# ── 1-2. Grouping ────────────────────────────────────────────────────────────

def chronological_order(points: List[Dict[str, Any]]) -> List[int]:
    starts = [_start(p) for p in points]
    if not any(starts):
        return list(range(len(points)))
    return sorted(range(len(points)), key=lambda i: (starts[i], i))


def center_vectors(vecs: np.ndarray, mean: Optional[np.ndarray] = None) -> np.ndarray:
    """Remove the meeting-wide common component (``mean``, default: mean of ``vecs``) and renormalise."""
    vecs = np.asarray(vecs, dtype=np.float32)
    if mean is None:
        mean = vecs.mean(axis=0)
    return _normalize(vecs - mean)


def find_near_duplicates(
    sims: np.ndarray,
    order: List[int],
    points: List[Dict[str, Any]],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> Dict[int, int]:
    """{duplicate index: earliest original index}: near-identical embedding and wording, and
    neither point states a fact (number, date, code) the other lacks."""
    texts = [_point_text(p) for p in points]
    facts = [_point_facts(p) for p in points]
    duplicates: Dict[int, int] = {}
    for pos, j in enumerate(order):
        for i in order[:pos]:
            if i in duplicates or sims[i, j] < threshold:
                continue
            a, b = content_tokens(texts[i]), content_tokens(texts[j])
            overlap = len(a & b) / max(1, min(len(a), len(b)))
            if overlap >= NEAR_DUPLICATE_MIN_OVERLAP and (facts[i] <= facts[j] or facts[j] <= facts[i]):
                duplicates[j] = i
                break
    return duplicates


def grouping_similarity(centered: np.ndarray, starts: List[float]) -> np.ndarray:
    """Centred cosine plus time proximity (when timestamps exist)."""
    sims = centered @ centered.T
    times = np.asarray(starts, dtype=np.float32)
    span = float(times.max() - times.min()) if len(times) else 0.0
    if span > 0:
        sims = sims + TIME_WEIGHT * np.exp(-np.abs(times[:, None] - times[None, :]) / (TIME_SCALE * span))
    return sims


def cluster_points(sims: np.ndarray, order: List[int], threshold: Optional[float] = None,
                   max_topics: int = MAX_TOPICS) -> List[List[int]]:
    """Average-linkage grouping on ``sims``; groups merge while their average similarity is at
    least ``threshold`` (default: the meeting's mean pairwise similarity + k·std), and until at
    most ``max_topics`` remain. Returns groups of indices in time order."""
    n = sims.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    if threshold is None:
        off_diagonal = sims[np.triu_indices(n, 1)]
        threshold = float(np.mean(off_diagonal) + CLUSTER_STD_FACTOR * np.std(off_diagonal))
    position = {idx: pos for pos, idx in enumerate(order)}
    clusters: List[List[int]] = [[i] for i in range(n)]
    link = sims.astype(np.float64).copy()
    np.fill_diagonal(link, -np.inf)
    active = list(range(n))
    while len(active) > 1:
        sub = link[np.ix_(active, active)]
        flat = int(np.argmax(sub))
        a, b = active[flat // len(active)], active[flat % len(active)]
        if sub.flat[flat] < threshold and len(active) <= max_topics:
            break
        size_a, size_b = len(clusters[a]), len(clusters[b])
        # Lance-Williams update for average linkage.
        merged = (size_a * link[a] + size_b * link[b]) / (size_a + size_b)
        link[a, :] = merged
        link[:, a] = merged
        link[a, a] = -np.inf
        clusters[a] = clusters[a] + clusters[b]
        active.remove(b)
    groups = [sorted(clusters[i], key=position.get) for i in active]
    groups.sort(key=lambda g: position[g[0]])
    return groups


# ── 3. Classification ────────────────────────────────────────────────────────

def _normalize_type(value: Any) -> Optional[str]:
    key = re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())
    return _TYPE_ALIASES.get(key)


def heuristic_classification(point: Dict[str, Any]) -> Dict[str, Any]:
    text = _point_text(point)
    types = []
    if DECISION_RE.search(text):
        types.append("decision")
    owner = str(point.get("action_owner") or "").strip().lower()
    if point.get("action_items") or (owner and owner not in ("none", "null", "n/a", "unassigned")) or ACTION_RE.search(text):
        types.append("action_item")
    if RISK_RE.search(text):
        types.append("risk")
    if QUESTION_RE.search(text):
        types.append("question")
    primary = types[0] if types else "discussion"
    return {"primary_type": primary, "types": types or ["discussion"], "evidence": "", "source": "heuristic"}


def _classification_batches(points: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    size = 0
    for p in points:
        entry_size = len(_point_text(p)) + 40
        if current and (len(current) >= CLASSIFY_BATCH_SIZE or size + entry_size > CLASSIFY_CHAR_BUDGET):
            batches.append(current)
            current, size = [], 0
        current.append(p)
        size += entry_size
    if current:
        batches.append(current)
    return batches


def classify_points(provider, points: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{point_id: {primary_type, types, evidence, source}}: the Stage 2 type when a model or a
    person set it, otherwise LLM classification with a heuristic fallback."""
    results: Dict[str, Dict[str, Any]] = {}
    by_id = {str(p["id"]): p for p in points}
    pending = []
    for p in points:
        stage2 = analysis_type_from_stage2(p)
        if stage2:
            results[str(p["id"])] = stage2
        else:
            pending.append(p)
    pending_ids = {str(p["id"]) for p in pending}
    for batch in _classification_batches(pending):
        payload = [{"id": str(p["id"]), "text": _point_text(p)} for p in batch]
        try:
            data = provider.classify_discussion_points(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:
            logger.warning(f"[AgendaAnalysis] classification batch failed: {exc}")
            data = None
        for item in (data or {}).get("classifications") or []:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("point_id") or item.get("id") or "").strip()
            primary = _normalize_type(item.get("primary_type") or item.get("type"))
            if pid not in pending_ids or primary is None:
                continue
            types = [t for t in (_normalize_type(v) for v in item.get("types") or []) if t]
            types = list(dict.fromkeys([primary] + types))
            evidence = str(item.get("evidence") or "").strip()
            if evidence and token_overlap(evidence, _point_text(by_id[pid])) < 0.8:
                evidence = ""
            results[pid] = {"primary_type": primary, "types": types, "evidence": evidence[:EVIDENCE_MAX_CHARS],
                            "source": "llm"}
    for pid, point in by_id.items():
        if pid not in results:
            results[pid] = heuristic_classification(point)
    return results


# ── 4. Topic synthesis ───────────────────────────────────────────────────────

def _cluster_payload(cluster_id: str, points: List[Dict[str, Any]], types: Dict[str, Dict[str, Any]],
                     duplicates: Dict[str, str]) -> Dict[str, Any]:
    per_point = max(MIN_POINT_CHARS_FOR_SYNTHESIS, SYNTHESIS_CHAR_BUDGET // max(1, len(points)))
    entries = []
    for p in points:
        pid = str(p["id"])
        entry = {"id": pid, "text": _point_text(p)[:per_point], "type": types.get(pid, {}).get("primary_type", "discussion")}
        if pid in duplicates:
            # Kept (it may add a detail) but marked, so the topic states it once.
            entry["near_duplicate_of"] = duplicates[pid]
        speakers = _clean_list(p.get("speakers")) or _clean_list(p.get("speaker"))
        if speakers:
            entry["speakers"] = speakers
        if _start(p):
            entry["time"] = _format_time(_start(p))
        entries.append(entry)
    return {"cluster_id": cluster_id, "points": entries}


def _synthesis_batches(payloads: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    size = 0
    for payload in payloads:
        payload_size = len(json.dumps(payload, ensure_ascii=False))
        if current and size + payload_size > SYNTHESIS_CHAR_BUDGET:
            batches.append(current)
            current, size = [], 0
        current.append(payload)
        size += payload_size
    if current:
        batches.append(current)
    return batches


def synthesize_topics(provider, payloads: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{cluster_id: raw topic from the LLM}. Clusters the LLM did not cover are absent."""
    topics: Dict[str, Dict[str, Any]] = {}
    wanted = {p["cluster_id"] for p in payloads}
    for batch in _synthesis_batches(payloads):
        try:
            data = provider.synthesize_agenda_topics(json.dumps(batch, ensure_ascii=False, indent=2))
        except Exception as exc:
            logger.warning(f"[AgendaAnalysis] topic synthesis batch failed: {exc}")
            data = None
        for item in (data or {}).get("topics") or []:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("cluster_id") or "").strip()
            if cid in wanted and cid not in topics:
                topics[cid] = item
    return topics


def _top_terms(points: List[Dict[str, Any]], limit: int = 4) -> List[str]:
    counts: Counter = Counter()
    for p in points:
        for term in _clean_list(p.get("technical_terms"), limit=50):
            counts[term] += 2
    if not counts:
        for p in points:
            counts.update(t for t in content_tokens(_point_text(p)) if not t.isdigit() and len(t) > 3)
    return [t for t, _ in counts.most_common(limit)]


def _fallback_title(points: List[Dict[str, Any]]) -> str:
    terms = _top_terms(points, 3)
    if not terms:
        return "Discussion Topic"
    return "Discussion on " + ", ".join(t if any(c.isupper() for c in t) else t.capitalize() for t in terms)


def _fallback_description(points: List[Dict[str, Any]]) -> str:
    sentences = []
    for p in points[:3]:
        first = split_sentences(_point_text(p))
        if first:
            sentences.append(first[0])
    return " ".join(sentences)[:600]


def fallback_topic(points: List[Dict[str, Any]], types: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    topic: Dict[str, Any] = {
        "title": _fallback_title(points),
        "description": _fallback_description(points),
        "supporting_context": "",
        "point_ids": [str(p["id"]) for p in points],
        "keywords": _top_terms(points, 8),
        "generated_by": "fallback",
    }
    for type_name, list_name in _TYPE_LISTS.items():
        topic[list_name] = [
            {"text": (split_sentences(_point_text(p)) or [_point_text(p)])[0][:300], "point_ids": [str(p["id"])]}
            for p in points if types.get(str(p["id"]), {}).get("primary_type") == type_name
        ]
    return topic


# ── 5. Validation ────────────────────────────────────────────────────────────

def _normalize_entries(values: Any) -> List[Dict[str, Any]]:
    entries = []
    for v in values if isinstance(values, list) else []:
        if isinstance(v, str) and v.strip():
            entries.append({"text": v.strip(), "point_ids": []})
        elif isinstance(v, dict) and str(v.get("text") or "").strip():
            ids = v.get("point_ids") or v.get("source_point_ids") or []
            if isinstance(ids, (str, int)):
                ids = [ids]
            entries.append({"text": str(v["text"]).strip(), "point_ids": [str(i) for i in ids if str(i).strip()]})
    return entries


def _normalize_topic(raw: Dict[str, Any], cluster_ids: List[str]) -> Dict[str, Any]:
    ids = raw.get("point_ids") or []
    if isinstance(ids, (str, int)):
        ids = [ids]
    topic = {
        "title": re.sub(r"\s+", " ", str(raw.get("title") or "")).strip().strip('"'),
        "description": str(raw.get("description") or "").strip(),
        "supporting_context": str(raw.get("supporting_context") or "").strip(),
        "point_ids": [str(i).strip() for i in ids if str(i).strip()],
        "cluster_point_ids": list(cluster_ids),
        "generated_by": raw.get("generated_by") or "llm",
    }
    for list_name in _TYPE_LISTS.values():
        topic[list_name] = _normalize_entries(raw.get(list_name))
    for field in _KEYWORD_FIELDS:
        topic[field] = _clean_list(raw.get(field))
    return topic


def topic_text(topic: Dict[str, Any]) -> str:
    return ". ".join(s for s in (topic.get("title"), topic.get("description"), topic.get("supporting_context")) if s)


def _strip_unsupported_words(text: str, facts: set) -> str:
    kept = [tok for tok in _WORD_RE.findall(text) if not (fact_tokens(tok) - facts)]
    return re.sub(r"\s+", " ", "".join(kept)).strip(" -–—:,;")


def validate_topics(
    topics: List[Dict[str, Any]],
    points_by_id: Dict[str, Dict[str, Any]],
    point_vecs: Dict[str, np.ndarray],
    duplicates: Dict[str, str],
    order_position: Dict[str, int],
    embedder=None,
    center: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Relevance, duplication, completeness and factual checks. Returns (topics, report).
    ``point_vecs`` and ``center`` are the meeting-centred point vectors and the mean they were
    centred on (topic summaries are compared in the same space)."""
    report: Dict[str, Any] = {
        "topics_in": len(topics), "topics_out": 0, "merged": [], "dropped": [], "uncovered_point_ids": [],
        "unsupported_removed": [], "coverage_ratio": 0.0, "near_duplicates": len(duplicates),
    }

    # Relevance: only real point ids; a topic citing none keeps its group's points.
    kept: List[Dict[str, Any]] = []
    for topic in topics:
        ids = [i for i in dict.fromkeys(topic["point_ids"]) if i in points_by_id]
        if not ids:
            ids = [i for i in topic["cluster_point_ids"] if i in points_by_id]
        if not ids:
            report["dropped"].append({"title": topic["title"], "reason": "no supporting discussion points"})
            continue
        topic["point_ids"] = ids
        kept.append(topic)
    topics = kept

    # Duplicates: same title, near-identical meaning, or mostly the same points.
    if len(topics) > 1:
        vecs = None
        if embedder is not None:
            try:
                vecs = _embed(embedder, [topic_text(t) for t in topics])
                if center is not None:
                    vecs = center_vectors(vecs, center)
            except Exception as exc:
                logger.warning(f"[AgendaAnalysis] topic duplicate check without embeddings: {exc}")
        merged_into: Dict[int, int] = {}
        for j in range(len(topics)):
            for i in range(j):
                if i in merged_into:
                    continue
                a, b = topics[i], topics[j]
                same_title = a["title"].strip().lower() == b["title"].strip().lower() and a["title"]
                shared = len(set(a["point_ids"]) & set(b["point_ids"])) / max(1, min(len(a["point_ids"]), len(b["point_ids"])))
                same_meaning = vecs is not None and float(vecs[i] @ vecs[j]) >= TOPIC_DUPLICATE_THRESHOLD
                if same_title or same_meaning or shared >= TOPIC_POINT_OVERLAP:
                    reason = "same title" if same_title else ("same subject" if same_meaning else "shared discussion points")
                    report["merged"].append({"kept": a["title"], "merged": b["title"], "reason": reason})
                    a["point_ids"] = list(dict.fromkeys(a["point_ids"] + b["point_ids"]))
                    a["cluster_point_ids"] = list(dict.fromkeys(a["cluster_point_ids"] + b["cluster_point_ids"]))
                    if len(b["description"]) > len(a["description"]):
                        a["description"] = b["description"]
                    if b["supporting_context"] and b["supporting_context"] not in a["supporting_context"]:
                        a["supporting_context"] = " ".join(s for s in (a["supporting_context"], b["supporting_context"]) if s)
                    for list_name in _TYPE_LISTS.values():
                        seen = {e["text"].lower() for e in a[list_name]}
                        a[list_name] += [e for e in b[list_name] if e["text"].lower() not in seen]
                    for field in _KEYWORD_FIELDS:
                        a[field] = _clean_list(a[field] + b[field])
                    merged_into[j] = i
                    break
        topics = [t for k, t in enumerate(topics) if k not in merged_into]

    # Each point belongs to exactly one topic: its own group's topic, else the first claimant.
    owner: Dict[str, int] = {}
    for k, topic in enumerate(topics):
        for pid in topic["point_ids"]:
            if pid not in owner or (pid in topic["cluster_point_ids"] and pid not in topics[owner[pid]]["cluster_point_ids"]):
                owner[pid] = k
    for k, topic in enumerate(topics):
        topic["point_ids"] = [pid for pid in topic["point_ids"] if owner.get(pid) == k]
        topic["added_by_validation"] = []

    # Near-duplicates stay with their original point.
    for dup, original in duplicates.items():
        if original in owner and owner.get(dup) != owner[original]:
            if dup in owner:
                topics[owner[dup]]["point_ids"].remove(dup)
            owner[dup] = owner[original]
            topics[owner[original]]["point_ids"].append(dup)

    # Completeness: every point is covered; uncovered points join their nearest topic.
    total = len(points_by_id)
    uncovered = [pid for pid in points_by_id if pid not in owner]
    report["coverage_ratio"] = round((total - len(uncovered)) / total, 4) if total else 1.0
    report["uncovered_point_ids"] = uncovered
    topics = [t for t in topics if t["point_ids"]]
    if uncovered and topics:
        # A point the LLM left out returns to the topic written for its own group; only points
        # whose group lost its topic go to the nearest remaining topic.
        home = {pid: k for k, t in enumerate(topics) for pid in t["cluster_point_ids"]}
        centroids = _normalize(np.stack([np.mean([point_vecs[p] for p in t["point_ids"]], axis=0) for t in topics]))
        for pid in uncovered:
            k = home.get(pid)
            target = topics[k] if k is not None else topics[int(np.argmax(centroids @ point_vecs[pid]))]
            target["point_ids"].append(pid)
            target["added_by_validation"].append(pid)

    # Factual accuracy: numbers, dates and acronyms must come from the topic's own points.
    for topic in topics:
        facts = set()
        fact_by_point = {}
        for pid in topic["point_ids"]:
            fact_by_point[pid] = _point_facts(points_by_id[pid])
            facts |= fact_by_point[pid]
        members = [points_by_id[p] for p in topic["point_ids"]]

        for list_name in _TYPE_LISTS.values():
            clean = []
            for entry in topic[list_name]:
                entry["point_ids"] = [p for p in entry["point_ids"] if p in fact_by_point]
                entry_facts = set().union(*(fact_by_point[p] for p in entry["point_ids"])) if entry["point_ids"] else facts
                missing = unsupported_facts(entry["text"], entry_facts)
                if missing and entry["point_ids"]:
                    missing = unsupported_facts(entry["text"], facts)
                if missing:
                    report["unsupported_removed"].append({"topic": topic["title"], "field": list_name,
                                                          "text": entry["text"], "unsupported": missing})
                    continue
                clean.append(entry)
            topic[list_name] = clean
        for field in _KEYWORD_FIELDS:
            topic[field] = [kw for kw in topic[field] if not unsupported_facts(kw, facts)]

        missing = unsupported_facts(topic["title"], facts)
        if missing or not topic["title"]:
            stripped = _strip_unsupported_words(topic["title"], facts) if topic["title"] else ""
            if missing:
                report["unsupported_removed"].append({"topic": topic["title"], "field": "title",
                                                      "text": topic["title"], "unsupported": missing})
            topic["title"] = stripped if len(stripped.split()) >= 2 else _fallback_title(members)
        for field in ("description", "supporting_context"):
            text, removed = drop_unsupported_sentences(topic[field], facts)
            if removed:
                report["unsupported_removed"].append({"topic": topic["title"], "field": field,
                                                      "text": " ".join(removed), "unsupported": unsupported_facts(" ".join(removed), facts)})
            topic[field] = text
        if not topic["description"]:
            topic["description"] = _fallback_description(members)

    # Chronology: topics and their points in meeting order.
    for topic in topics:
        topic["point_ids"].sort(key=lambda p: order_position.get(p, 0))
    topics.sort(key=lambda t: order_position.get(t["point_ids"][0], 0))
    report["topics_out"] = len(topics)
    return topics, report


# ── 6. Matching against existing agendas ─────────────────────────────────────

def match_topics_to_agendas(
    topics: List[Dict[str, Any]],
    agendas: List[Dict[str, Any]],
    embedder,
    points_by_id: Dict[str, Dict[str, Any]],
    expanded_agendas: Optional[List[Dict[str, Any]]] = None,
    agenda_doc_points: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Score every topic (and its points) against every existing agenda.

    Each match reports the topic's score for the agenda, the margin to the runner-up, the share
    of its points that prefer the agenda, how its points compare with the meeting's typical point
    (``relative_score``) and the two points that support the match best.
    """
    from services.rom_service import compute_point_agenda_similarity

    if not topics or not agendas:
        return []
    top_k = min(3, len(agendas))
    member_ids = [pid for t in topics for pid in t["point_ids"]]
    topic_sim = compute_point_agenda_similarity(
        [{"id": t["topic_id"], "text": topic_text(t)} for t in topics],
        agendas, embedder, expanded_agendas, agenda_doc_points, top_k=top_k,
    )
    point_sim = compute_point_agenda_similarity(
        [{"id": pid, "polished_text": _point_text(points_by_id[pid])} for pid in member_ids],
        agendas, embedder, expanded_agendas, agenda_doc_points, top_k=1,
    )
    point_best = {c["point_id"]: c["candidates"][0] for c in point_sim["candidates"] if c["candidates"]}
    point_scores = point_sim["scores"]
    typical = float(np.median([b["score"] for b in point_best.values()])) if point_best else 0.0
    titles = {a.get("agenda_id"): a.get("title", "") for a in agendas}

    matches = []
    for topic, cand in zip(topics, topic_sim["candidates"]):
        ranked = cand["candidates"]
        top = ranked[0] if ranked else None
        votes = Counter(point_best[p]["agenda_id"] for p in topic["point_ids"] if p in point_best)
        vote_aid, vote_count = votes.most_common(1)[0] if votes else (None, 0)
        vote_share = vote_count / max(1, len(topic["point_ids"]))
        best_scores = [point_best[p]["score"] for p in topic["point_ids"] if p in point_best]
        relative = (float(np.mean(best_scores)) - typical) if best_scores else 0.0

        target = top["agenda_id"] if top else None
        if vote_aid and vote_share >= TOPIC_MATCH_MIN_VOTE:
            target = vote_aid
        matched = target if target and relative >= -TOPIC_UNMATCHED_GAP else None
        target_entry = next((c for c in ranked if c["agenda_id"] == target), None)
        topic_score = topic_sim["scores"].get(topic["topic_id"], {}).get(target, 0.0) if target else 0.0

        evidence = []
        if target:
            best_points = sorted(
                (p for p in topic["point_ids"] if p in point_scores),
                key=lambda p: point_scores[p].get(target, 0.0), reverse=True,
            )[:2]
            evidence = [{
                "point_id": p,
                "text": _point_text(points_by_id[p])[:EVIDENCE_MAX_CHARS],
                "score": round(point_scores[p].get(target, 0.0), 4),
            } for p in best_points]

        matches.append({
            "topic_id": topic["topic_id"],
            "topic_title": topic["title"],
            "status": "matched" if matched else "unmatched",
            "matched_agenda_id": matched,
            "matched_agenda_title": titles.get(matched, "") if matched else "",
            "best_agenda_id": target,
            "best_agenda_title": titles.get(target, "") if target else "",
            "score": round(float(topic_score), 4),
            "margin": cand.get("margin", 0.0),
            "best_facet": (target_entry or {}).get("best_facet"),
            "best_facet_score": (target_entry or {}).get("best_facet_score"),
            "vote_agenda_id": vote_aid,
            "vote_share": round(vote_share, 4),
            "relative_score": round(relative, 4),
            "evidence": evidence,
            "candidates": ranked,
        })
    return matches


# ── Orchestration ────────────────────────────────────────────────────────────

def analyze_discussion(provider, embedder, polished_points: List[Dict[str, Any]], log_prefix: str = "ROM S3-Analysis") -> Optional[Dict[str, Any]]:
    """Steps 1-5 over the Stage 2 points. Returns the analysis, or None when there is nothing to analyse."""
    points = [
        p if isinstance(p["id"], str) else dict(p, id=str(p["id"]))
        for p in polished_points or [] if isinstance(p, dict) and p.get("id") and _point_text(p)
    ]
    if not points:
        return None
    ids = [p["id"] for p in points]
    points_by_id = dict(zip(ids, points))

    raw_vecs = _embed(embedder, [_point_text(p) for p in points])
    mean_vec = raw_vecs.mean(axis=0)
    vecs = center_vectors(raw_vecs, mean_vec)
    order = chronological_order(points)
    order_position = {ids[idx]: pos for pos, idx in enumerate(order)}
    duplicate_idx = find_near_duplicates(raw_vecs @ raw_vecs.T, order, points)
    duplicates = {ids[d]: ids[o] for d, o in duplicate_idx.items()}
    group_sims = grouping_similarity(vecs, [_start(p) for p in points])
    off_diagonal = group_sims[np.triu_indices(len(points), 1)] if len(points) > 1 else np.zeros(1)
    threshold = float(np.mean(off_diagonal) + CLUSTER_STD_FACTOR * np.std(off_diagonal))
    max_topics = max(1, min(MAX_TOPICS, len(points) // MIN_POINTS_PER_TOPIC))
    clusters = cluster_points(group_sims, order, threshold, max_topics=max_topics)
    logger.info(f"[{log_prefix}] {len(points)} point(s) -> {len(clusters)} group(s) "
                f"(threshold={threshold:.3f}, near-duplicates={len(duplicates)})")

    point_types = classify_points(provider, points)

    payloads = []
    cluster_ids = {}
    for n, cluster in enumerate(clusters, 1):
        cid = f"C{n}"
        cluster_ids[cid] = [ids[i] for i in cluster]
        payloads.append(_cluster_payload(cid, [points[i] for i in cluster], point_types, duplicates))
    raw_topics = synthesize_topics(provider, payloads)

    topics = []
    for cid, member_ids in cluster_ids.items():
        raw = raw_topics.get(cid) or fallback_topic([points_by_id[p] for p in member_ids], point_types)
        topics.append(_normalize_topic(raw, member_ids))

    topics, report = validate_topics(
        topics, points_by_id, {ids[i]: vecs[i] for i in range(len(ids))}, duplicates, order_position, embedder,
        center=mean_vec,
    )
    for n, topic in enumerate(topics, 1):
        topic["topic_id"] = f"T{n}"
        starts = [_start(points_by_id[p]) for p in topic["point_ids"]]
        ends = [float(points_by_id[p].get("timeline_end") or 0.0) for p in topic["point_ids"]]
        topic["time_start"] = min(starts) if starts else 0.0
        topic["time_end"] = max(ends) if ends else 0.0
        topic["type_counts"] = dict(Counter(point_types[p]["primary_type"] for p in topic["point_ids"] if p in point_types))
        speakers: List[str] = []
        for p in topic["point_ids"]:
            for s in _clean_list(points_by_id[p].get("speakers")) or _clean_list(points_by_id[p].get("speaker")):
                if s not in speakers:
                    speakers.append(s)
        topic["speakers"] = speakers
        topic.pop("cluster_point_ids", None)

    try:
        from config import settings
        model_name = getattr(settings, "EMBEDDING_MODEL", None)
    except Exception:
        model_name = None
    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model_name,
        "group_threshold": round(threshold, 4),
        "point_types": point_types,
        "type_counts": dict(Counter(v["primary_type"] for v in point_types.values())),
        "near_duplicates": duplicates,
        "topics": topics,
        "topic_matches": [],
        "validation": report,
        "mode": None,
    }


def discussion_summary(topic: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "topic_id": topic.get("topic_id"),
        "supporting_context": topic.get("supporting_context", ""),
        "decisions": topic.get("decisions", []),
        "action_items": topic.get("action_items", []),
        "open_questions": topic.get("open_questions", []),
        "risks": topic.get("risks", []),
        "point_ids": topic.get("point_ids", []),
        "type_counts": topic.get("type_counts", {}),
    }


def topics_to_agendas(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Agendas built from the discussion (used when no agenda document exists)."""
    agendas = []
    for n, topic in enumerate(topics, 1):
        agenda = {
            "agenda_id": f"A{n}",
            "title": topic["title"],
            "description": topic["description"],
            "speaker": None,
            "presenter": None,
            "source": "discussion",
            "discussion_summary": discussion_summary(topic),
        }
        for field in _KEYWORD_FIELDS:
            agenda[field] = list(topic.get(field) or [])
        agendas.append(agenda)
    return agendas


def build_suggested_agendas(topics: List[Dict[str, Any]], matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Topics that match no existing agenda, offered for the user to add."""
    by_topic = {m["topic_id"]: m for m in matches}
    suggestions = []
    for topic in topics:
        match = by_topic.get(topic["topic_id"])
        if not match or match["status"] != "unmatched":
            continue
        suggestion = {
            "suggestion_id": f"S{len(suggestions) + 1}",
            "topic_id": topic["topic_id"],
            "title": topic["title"],
            "description": topic["description"],
            "discussion_summary": discussion_summary(topic),
            "point_ids": topic["point_ids"],
            "best_agenda_id": match["best_agenda_id"],
            "best_agenda_title": match["best_agenda_title"],
            "best_score": match["score"],
            "evidence": match["evidence"],
        }
        for field in _KEYWORD_FIELDS:
            suggestion[field] = list(topic.get(field) or [])
        suggestions.append(suggestion)
    return suggestions
