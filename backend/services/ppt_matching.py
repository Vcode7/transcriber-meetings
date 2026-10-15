"""
ppt_matching.py — Match slide insights to Stage 2 long discussion points.

  1. Candidate retrieval — semantic similarity (text embeddings, when available) blended with
     typo-tolerant lexical/entity scoring and a project-name bonus. Top-K per long point.
  2. Strict LLM verification — same project/workstream AND same subject; shared generic words
     are not enough. Transcription errors in names and acronyms are tolerated.
  3. Global assignment — verified pairs are assigned best-score-first; each insight is used
     once, each long point takes at most MAX_INSIGHTS_PER_GROUP insights.
  4. Synthesis — one structured meeting-prep point per long point: slides decide status,
     dates and figures; the discussion adds decisions, reasons and next steps.

The LLM (`infer(prompt, max_tokens) -> str`) and embedder (`embed(texts) -> ndarray`) are
injected so the module is testable without models.
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from services.ppt_insights import (
    InferFn,
    STATUSES,
    _clean_str,
    _fuzz,
    _norm_choice,
    insight_point_text,
    parse_json_object,
    ungrounded_dates,
)

logger = logging.getLogger(__name__)

EmbedFn = Callable[[List[str]], np.ndarray]

TOP_K = 5
MAX_INSIGHTS_PER_GROUP = 3
MAX_FINAL_POINTS = 12
MIN_CANDIDATE_SCORE_SEMANTIC = 0.30
MIN_CANDIDATE_SCORE_LEXICAL = 0.20
FALLBACK_ACCEPT_SCORE = 0.62   # accept without LLM verification only when the LLM call failed

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could", "that", "this", "these",
    "those", "it", "its", "they", "them", "their", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "which", "who", "whom", "what", "where", "when", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "than", "too", "very", "during", "about",
    "meeting", "team", "discussed", "point", "update", "status", "review", "slide", "project", "also",
    "need", "needs", "attention", "current", "next", "work", "plan", "planned", "timeline",
}


# ── Lexical / entity similarity ───────────────────────────────────────────────

def normalize_acronyms(text: str) -> str:
    t = re.sub(r"\b([A-Za-z])\s+([A-Za-z])(?:\s+([A-Za-z]))*\b", lambda m: m.group(0).replace(" ", ""), text)
    t = re.sub(r"\b([A-Z]{1,4})\s+([A-Z]{1,4})\b", lambda m: m.group(1) + m.group(2), t)
    t = re.sub(r"\b([A-Z]{2,5})\s+(\d{1,4})\b", r"\1-\2", t)
    return t


def extract_entities(text: str) -> List[str]:
    norm = normalize_acronyms(text)
    ents = re.findall(
        r"\b(?:\d{1,2}[-/][A-Za-z0-9]{2,4}[-/]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s*\d{2,4}|Q[1-4]|H[1-2])\b",
        norm, re.IGNORECASE,
    )
    ents += re.findall(r"\b[A-Za-z]{2,5}[-_]\d{1,4}\b", norm)
    ents += [a for a in re.findall(r"\b[A-Z]{3,6}\b", norm) if a.lower() not in STOP_WORDS]
    return list(dict.fromkeys(ents))


def _keywords(text: str) -> List[Tuple[str, float]]:
    out = []
    for tok in re.findall(r"\b[a-zA-Z0-9_\-./]+\b", normalize_acronyms(text)):
        clean = tok.strip(".-_/").lower()
        if len(clean) > 2 and clean not in STOP_WORDS:
            out.append((clean, 2.5 if (tok.isupper() or any(c.isdigit() for c in tok)) else 1.0))
    return out


def lexical_score(a: str, b: str) -> Dict[str, Any]:
    """Typo-tolerant weighted keyword overlap + token-set similarity + entity anchoring (0–1)."""
    fuzz = _fuzz()
    ka, kb = _keywords(a), _keywords(b)
    if not ka or not kb:
        return {"score": 0.0, "terms": [], "entities": []}
    words_b = list({w for w, _ in kb})
    total = weight_sum = 0.0
    terms = []
    for word, weight in ka:
        best, best_word = 0.0, ""
        for other in words_b:
            if word == other:
                best, best_word = 1.0, word
                break
            if len(word) >= 4 and len(other) >= 4 and (word in other or other in word):
                sim = 0.88
            else:
                sim = fuzz.ratio(word, other) / 100.0
                if sim < 0.8:
                    continue
            if sim > best:
                best, best_word = sim, other
        total += best * weight
        weight_sum += weight
        if best >= 0.8:
            terms.append(word if best_word == word else f"{word}~{best_word}")
    token_sim = total / weight_sum if weight_sum else 0.0
    sentence_sim = fuzz.token_set_ratio(a, b) / 100.0
    ents_a, ents_b = extract_entities(a), extract_entities(b)
    matched_ents = [e for e in ents_a if any(e.lower() == f.lower() or fuzz.ratio(e.lower(), f.lower()) >= 85 for f in ents_b)]
    score = min(1.0, 0.5 * token_sim + 0.3 * sentence_sim + min(0.35, 0.15 * len(matched_ents)))
    return {"score": round(score, 4), "terms": list(dict.fromkeys(terms)), "entities": matched_ents}


def project_mentioned(project: str, text: str) -> bool:
    """True if the insight's project name (or its acronym) appears in the discussion, typo-tolerant."""
    if not project or project.lower() == "general":
        return False
    fuzz = _fuzz()
    norm_text = normalize_acronyms(text).lower()
    tokens = re.findall(r"[a-z0-9]+", norm_text)
    name_tokens = [t for t in re.findall(r"[a-z0-9]+", normalize_acronyms(project).lower()) if len(t) >= 3 and t not in STOP_WORDS]
    if not name_tokens:
        return False
    hits = sum(1 for nt in name_tokens if any(nt == t or (len(nt) >= 4 and fuzz.ratio(nt, t) >= 85) for t in tokens))
    if hits / len(name_tokens) >= 0.6:
        return True
    acronym = "".join(w[0] for w in re.findall(r"[A-Za-z]+", project) if w[0].isupper())
    return len(acronym) >= 2 and acronym.lower() in tokens


# ── Candidate retrieval ───────────────────────────────────────────────────────

def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return a @ b.T


def insight_match_text(ins: Dict[str, Any]) -> str:
    parts = [ins.get("project") or "", ins.get("headline") or "", ins.get("summary") or "", ins.get("attention") or ""]
    return ". ".join(p for p in parts if p)


def score_candidates(
    insights: Sequence[Dict[str, Any]],
    long_points: Sequence[Dict[str, Any]],
    embed: Optional[EmbedFn] = None,
) -> Tuple[List[List[Dict[str, Any]]], bool]:
    """
    For every long point, the top-K insight candidates with blended scores.
    Returns (candidates per long point, whether semantic similarity was used).
    """
    ins_texts = [insight_match_text(i) for i in insights]
    lp_texts = [lp["text"] for lp in long_points]
    sem = None
    if embed is not None and ins_texts and lp_texts:
        try:
            vectors = np.asarray(embed(lp_texts + ins_texts), dtype=np.float32)
            sem = _cosine_matrix(vectors[: len(lp_texts)], vectors[len(lp_texts):])
        except Exception as exc:
            logger.warning(f"[PPTMatching] Embedding similarity unavailable, using lexical only: {exc}")
            sem = None

    floor = MIN_CANDIDATE_SCORE_SEMANTIC if sem is not None else MIN_CANDIDATE_SCORE_LEXICAL
    per_point: List[List[Dict[str, Any]]] = []
    for li, lp_text in enumerate(lp_texts):
        scored = []
        for ii, ins in enumerate(insights):
            lex = lexical_score(ins_texts[ii], lp_text)
            proj = project_mentioned(ins.get("project", ""), lp_text)
            bonus = 0.15 if proj else 0.0
            if sem is not None:
                sem_n = float(np.clip((sem[li, ii] - 0.25) / 0.5, 0.0, 1.0))
                score = 0.55 * sem_n + 0.30 * lex["score"] + bonus
            else:
                sem_n = None
                score = 0.8 * lex["score"] + bonus
            if score >= floor or lex["entities"] or proj:
                scored.append({
                    "index": ii, "score": round(min(score, 1.0), 4),
                    "semantic": None if sem_n is None else round(sem_n, 4),
                    "lexical": lex["score"], "terms": lex["terms"][:8], "entities": lex["entities"][:5],
                    "project_match": proj,
                })
        scored.sort(key=lambda c: c["score"], reverse=True)
        per_point.append(scored[:TOP_K])
    return per_point, sem is not None


# ── LLM verification and synthesis ────────────────────────────────────────────

def build_verify_prompt(long_text: str, candidates: List[Dict[str, Any]], insights: Sequence[Dict[str, Any]]) -> str:
    lines = [f"{insights[c['index']]['ref']} {insight_point_text(insights[c['index']])}" for c in candidates]
    return f"""You are matching a meeting discussion point to findings from the presentation reviewed in that meeting.

DISCUSSION POINT (from the meeting transcript; names may contain speech-to-text errors):
"{long_text}"

CANDIDATE PRESENTATION FINDINGS:
{chr(10).join(lines)}

A finding MATCHES only if it concerns the same project or workstream AND the same subject (the same milestone, status, delay, risk, decision, dependency or action) as the discussion point.
- Tolerate transcription errors, spacing and acronym variants in names (e.g. "TC EC" = "TCEC", "iron brid" = "Iron Bird").
- Sharing generic words (timeline, review, update, team, plan, status) is NOT a match.
- The discussion point may match several findings or none.

Return JSON only:
{{"matches": [{{"id": "<finding id such as P3>", "confidence": "high|medium", "reason": "<few words>"}}]}}"""


def parse_verified(raw: str, allowed_refs: Dict[str, int]) -> List[Tuple[int, str, str]]:
    parsed = parse_json_object(raw)
    out = []
    for m in parsed.get("matches") or []:
        if not isinstance(m, dict):
            continue
        ref = str(m.get("id", "")).strip().upper()
        # Accept "3", "P3" and ids the model decorated, e.g. "P3 [Slide 5 · Project Atlas]".
        lead = re.match(r"P?(\d+)\b", ref)
        if lead:
            ref = f"P{lead.group(1)}"
        if ref not in allowed_refs:
            continue
        conf = str(m.get("confidence", "medium")).lower()
        conf = conf if conf in ("high", "medium") else "medium"
        out.append((allowed_refs[ref], conf, _clean_str(m.get("reason"), 160)))
    return out


# Synthesis detail levels: "standard" (PPT Testing: a meeting-prep point, slides first) and "long"
# (long-version ROM in Stage 3: the discussion point stays the base and keeps all of its content;
# the slides add their terminology, status, dates and figures — slightly more detail per point).
DETAIL_SUMMARY = {"standard": ("2-3 sentences", 700),
                  "long": ("as long as the discussion point or slightly longer, never shorter", 1200)}


def _summary_budget(long_text: str, detail: str) -> Tuple[int, int]:
    """(max summary chars, max tokens). The long ROM keeps the whole discussion point, so its
    budget grows with the point instead of compressing it."""
    chars = DETAIL_SUMMARY.get(detail, DETAIL_SUMMARY["standard"])[1]
    if detail != "long":
        return chars, 600
    chars = max(chars, int(len(long_text) * 1.6))
    return chars, max(800, chars // 3 + 150)


def build_synthesis_prompt(long_text: str, matched: List[Dict[str, Any]], meeting: Optional[date],
                           avoid_dates: Optional[List[str]] = None, detail: str = "standard") -> str:
    sentences = DETAIL_SUMMARY.get(detail, DETAIL_SUMMARY["standard"])[0]
    extra = ""
    if avoid_dates:
        extra += f"\n- Do NOT mention {', '.join(avoid_dates)}: that date does not appear in the sources."
    lines = [f"- {insight_point_text(i)}" + (f" [dates: {', '.join(i['dates'])}]" if i.get("dates") else "") for i in matched]
    when = f" (meeting date {meeting.strftime('%d %b %Y')})" if meeting else ""
    if detail == "long":
        return f"""Refine one discussion point of the minutes of a meeting{when} with the presentation reviewed in that meeting.

MEETING DISCUSSION POINT (the base — keep all of its content):
"{long_text}"

MATCHING PRESENTATION FINDINGS:
{chr(10).join(lines)}

Rules:
- Keep every fact, decision, vote, name, owner and next step of the discussion point. Do not drop any of them and do not change its subject.
- Where the discussion refers to the same thing as the presentation, use the presentation's names and terms (the transcript may misspell them).
- Add the status, dates and figures from the findings that concern this same subject; ignore findings about other subjects.
- If the discussion contradicts the slides, say so in one short clause.
- Keep every date of the discussion point exactly as it is. Attach each slide date only to the item it belongs to.
- Write it as meeting minutes: do not mention slides, the presentation or the deck. No filler, no invented facts.{extra}

Return JSON only:
{{"headline": "<at most 12 words>", "status": "completed|ongoing|delayed|at_risk|planned|unknown",
 "summary": "<{sentences}: the refined discussion point>", "attention": "<what needs attention next, or empty>"}}"""
    return f"""Write one meeting-prep point that combines what the presentation reported with what was said in the meeting{when}.

PRESENTATION FINDINGS (authoritative for status, dates and figures):
{chr(10).join(lines)}

MEETING DISCUSSION:
"{long_text}"

Rules:
- The slides decide status, dates and figures; the discussion adds decisions, reasons, owners and next steps that were actually said.
- If the discussion contradicts the slides, say so in one short clause.
- Explain what changed, why it matters and the current status. Keep exact names, dates and numbers. No filler, no invented facts.
- Attach each date only to the item it belongs to in the sources; never move a deadline from one task to another.{extra}

Return JSON only:
{{"headline": "<at most 12 words>", "status": "completed|ongoing|delayed|at_risk|planned|unknown",
 "summary": "<{sentences}>", "attention": "<what needs attention next, or empty>"}}"""


def compose_point_text(headline: str, summary: str, attention: str) -> str:
    text = headline.rstrip(".") + ". " + summary if headline and summary else (headline or summary)
    if attention:
        text = text.rstrip() + f" Needs attention: {attention.rstrip('.')}."
    return re.sub(r"\s+", " ", text).strip()


def synthesize_group(long_text: str, matched: List[Dict[str, Any]], infer: InferFn, meeting: Optional[date],
                     detail: str = "standard") -> Dict[str, str]:
    sources = [long_text] + [f"{m.get('summary', '')} {m.get('headline', '')} {' '.join(m.get('dates', []))}" for m in matched]
    parsed: Dict[str, Any] = {}
    avoid: List[str] = []
    for attempt in range(2):  # one retry if the model introduces a date that is not in the sources
        try:
            parsed = parse_json_object(infer(build_synthesis_prompt(long_text, matched, meeting, avoid, detail),
                                             _summary_budget(long_text, detail)[1]))
        except Exception as exc:
            logger.warning(f"[PPTMatching] Synthesis failed: {exc}")
            parsed = {}
            break
        bad = ungrounded_dates(f"{parsed.get('summary', '')} {parsed.get('attention', '')} {parsed.get('headline', '')}", sources)
        if not bad:
            break
        logger.info(f"[PPTMatching] Synthesis introduced ungrounded date(s) {bad}; retrying")
        avoid = bad
        if attempt == 1:
            parsed = {}  # still ungrounded → fall back to the slide findings
    headline = _clean_str(parsed.get("headline"), 140)
    summary = _clean_str(parsed.get("summary"), _summary_budget(long_text, detail)[0])
    attention = _clean_str(parsed.get("attention"), 300)
    status = _norm_choice(parsed.get("status"), STATUSES, matched[0].get("status", "unknown"))
    if len(summary) < 20:  # fall back to the slide findings themselves
        headline = matched[0]["headline"]
        summary = " ".join(i["summary"] for i in matched)
        attention = next((i["attention"] for i in matched if i.get("attention")), "")
        status = matched[0].get("status", "unknown")
    return {"headline": headline, "summary": summary, "attention": attention, "status": status,
            "text": compose_point_text(headline, summary, attention)}


# ── Orchestration ─────────────────────────────────────────────────────────────

def match_points(
    insights: List[Dict[str, Any]],
    long_points: List[Dict[str, Any]],
    *,
    infer: InferFn,
    embed: Optional[EmbedFn] = None,
    meeting: Optional[date] = None,
    progress: Optional[Callable[[str, int, int, str], None]] = None,
    detail: str = "standard",
) -> Dict[str, Any]:
    report = progress or (lambda *_: None)
    for n, ins in enumerate(insights, 1):
        ins.setdefault("ref", f"P{n}")
        ins.setdefault("priority", 0.6)

    report("retrieval", 1, 1, "Scoring candidates")
    candidates, used_semantic = score_candidates(insights, long_points, embed)

    pairs: List[Dict[str, Any]] = []   # verified (long point, insight) pairs
    for li, (lp, cands) in enumerate(zip(long_points, candidates)):
        if not cands:
            continue
        report("verification", li + 1, len(long_points), f"Verifying long point {li + 1} of {len(long_points)}")
        allowed = {insights[c["index"]]["ref"].upper(): c["index"] for c in cands}
        by_index = {c["index"]: c for c in cands}
        try:
            verified = parse_verified(infer(build_verify_prompt(lp["text"], cands, insights), 400), allowed)
            llm_ok = True
        except Exception as exc:
            logger.warning(f"[PPTMatching] Verification failed for long point {li + 1}: {exc}")
            verified, llm_ok = [], False
        if not llm_ok:
            verified = [(c["index"], "medium", "high similarity (LLM unavailable)") for c in cands
                        if c["score"] >= FALLBACK_ACCEPT_SCORE]
        for idx, conf, reason in verified:
            cand = by_index[idx]
            pairs.append({
                "long_index": li, "insight_index": idx, "confidence": conf, "reason": reason,
                "score": round(cand["score"] * (1.0 if conf == "high" else 0.85), 4), "candidate": cand,
            })

    # Global assignment: strongest pairs first, each insight used once
    pairs.sort(key=lambda p: (p["score"], insights[p["insight_index"]]["priority"]), reverse=True)
    used_insights: set = set()
    groups_by_long: Dict[int, List[Dict[str, Any]]] = {}
    for pair in pairs:
        if pair["insight_index"] in used_insights:
            continue
        bucket = groups_by_long.setdefault(pair["long_index"], [])
        if len(bucket) >= MAX_INSIGHTS_PER_GROUP:
            continue
        bucket.append(pair)
        used_insights.add(pair["insight_index"])

    ranked = sorted(
        groups_by_long.items(),
        key=lambda kv: (0.5 * max(p["score"] for p in kv[1])
                        + 0.5 * max(insights[p["insight_index"]]["priority"] for p in kv[1])),
        reverse=True,
    )[:MAX_FINAL_POINTS]

    comparison_groups: List[Dict[str, Any]] = []
    for step, (li, group_pairs) in enumerate(ranked, 1):
        report("synthesis", step, len(ranked), f"Writing final point {step} of {len(ranked)}")
        matched = [insights[p["insight_index"]] for p in group_pairs]
        synth = synthesize_group(long_points[li]["text"], matched, infer, meeting, detail)
        slides = sorted({s for m in matched for s in m.get("slides", [])})
        projects = list(dict.fromkeys(m.get("project") for m in matched if m.get("project")))
        best = max(group_pairs, key=lambda p: p["score"])
        terms = list(dict.fromkeys(t for p in group_pairs for t in p["candidate"]["terms"] + p["candidate"]["entities"]))
        comparison_groups.append({
            "rank": step,
            "stage2_point": long_points[li],
            "insight_ids": [m["id"] for m in matched],
            "insight_refs": [m["ref"] for m in matched],
            "ppt_points": [insight_point_text(m) for m in matched],
            "slides": slides,
            "projects": projects,
            "match_type": "verified_match" if best["confidence"] in ("high", "medium") and "LLM unavailable" not in best["reason"] else "similarity_match",
            "confidence": best["confidence"],
            "match_score": best["score"],
            "semantic_score": best["candidate"]["semantic"],
            "lexical_score": best["candidate"]["lexical"],
            "matched_keywords": terms[:10],
            "relationship": "; ".join(dict.fromkeys(p["reason"] for p in group_pairs if p["reason"])) or "Same project and topic",
            "final_point": synth["text"],
            "final": synth,
        })

    matched_ids = {i for g in comparison_groups for i in g["insight_ids"]}
    unmatched_priority = [
        {"id": i["id"], "ref": i["ref"], "text": insight_point_text(i), "slides": i.get("slides", []),
         "project": i.get("project", ""), "status": i.get("status"), "priority": i["priority"]}
        for i in sorted(insights, key=lambda x: x["priority"], reverse=True)
        if i["id"] not in matched_ids and i["priority"] >= 0.75
    ][:6]

    return {
        "comparison_groups": comparison_groups,
        "unmatched_priority_insights": unmatched_priority,
        "stats": {
            "long_points": len(long_points),
            "insights": len(insights),
            "candidate_pairs": sum(len(c) for c in candidates),
            "verified_pairs": len(pairs),
            "groups": len(comparison_groups),
            "semantic_similarity": used_semantic,
        },
    }


def final_point_record(group: Dict[str, Any], enabled: bool = True, point_id: Optional[str] = None) -> Dict[str, Any]:
    synth = group.get("final") or {}
    slides = group.get("slides", [])
    return {
        "id": point_id or str(uuid.uuid4()),
        "text": group.get("final_point", ""),
        "headline": synth.get("headline", ""),
        "status": synth.get("status", "unknown"),
        "attention": synth.get("attention", ""),
        "slides": slides,
        "projects": group.get("projects", []),
        "stage2_source": group.get("stage2_point", {}).get("text", ""),
        "ppt_source": "\n".join(group.get("ppt_points", [])),
        "relationship": group.get("relationship", ""),
        "match_type": group.get("match_type", "verified_match"),
        "confidence": group.get("confidence", ""),
        "match_score": group.get("match_score", 0.0),
        "rank": group.get("rank", 1),
        "enabled": enabled,
    }
