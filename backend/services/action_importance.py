"""
Importance of action points (40-100%), shared by both action-point pipelines.

Every action point generated for the Final ROM (services/agenda_action_points.py) and for the
MoM built from Stage 2 points (RomService.generate_mom_from_enhanced_rom) is scored after
validation and dedupe by one LLM pass with one prompt (task key "rom_action_importance"):

- 40-50: general information or low-priority updates;
- 60-80: moderately important tasks, updates or decisions;
- 80-100: critical project updates, important dates, deadlines or high-priority decisions.

LLM scores are checked deterministically: 80% or more needs evidence in the item itself (a
stated deadline or date, a decision, high priority, or an urgency/consequence cue in the task
or its evidence quote); otherwise the score is capped at MODERATE_CAP. When the LLM fails or
leaves an item out, a heuristic score is used (``importance_source == "heuristic"``).

The score is stored on the item. The user's minimum-importance threshold
(``user_settings.action_point_min_importance``, 0 = off) only filters what is shown and
exported; items without a score (built before scoring existed) always pass.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from services.agenda_action_points import _ASAP_RE, is_unspecified
from services.fact_grounding import fact_tokens

logger = logging.getLogger(__name__)

IMPORTANCE_MIN = 40
IMPORTANCE_MAX = 100
CRITICAL_MIN = 80
MODERATE_CAP = 75
# Action points per LLM call, and characters of source discussion sent per action point.
SCORE_BATCH_SIZE = 12
SOURCE_MAX_CHARS = 600
REASON_MAX_CHARS = 200
IMPORTANCE_KEYS = ("importance", "importance_source", "importance_reason", "importance_adjusted")

_NUMERIC_DATE_RE = re.compile(r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.]\d{1,2}(?:[/.]\d{2,4})?)\b")
_RELATIVE_DATE_RE = re.compile(
    r"\b(?:by|before|until|no\s+later\s+than)\s+(?:the\s+)?"
    r"(?:end\s+of|next|this|tomorrow|today|tonight|eod|eow|eom|q[1-4])\b",
    re.IGNORECASE,
)
_CRITICAL_CUE_RE = re.compile(
    r"\b(?:deadline|due\s+date|critical|blockers?|blocking|blocked|compliance|non-?compliance|regulatory|"
    r"risks?|penalt(?:y|ies)|escalat\w*|approvals?|sign-?off|go-?live|cost\s+overruns?|over\s+budget)\b",
    re.IGNORECASE,
)
_PRIORITY_BASE = {"high": 80, "medium": 65, "low": 45}
_NO_PRIORITY_BASE = 60


def normalize_importance(value: Any) -> Optional[int]:
    """Percentage clamped to 40-100 from 85, "85", "85%" or 0.85; None when unreadable."""
    if value is None or isinstance(value, bool):
        return None
    raw = str(value).strip()
    match = re.search(r"\d+(?:\.\d+)?", raw)
    if not match:
        return None
    score = float(match.group())
    if 0 < score <= 1 and "." in match.group():
        score *= 100  # a fraction such as 0.85
    return int(round(min(max(score, IMPORTANCE_MIN), IMPORTANCE_MAX)))


def _task_text(item: Dict[str, Any]) -> str:
    return str(item.get("task") or item.get("item") or item.get("description") or "").strip()


def _owner(item: Dict[str, Any]) -> Optional[str]:
    owner = item.get("assignee") or item.get("owner")
    return None if is_unspecified(owner) else str(owner).strip()


def _evidence_text(item: Dict[str, Any]) -> str:
    evidence = item.get("evidence_quote") or item.get("evidence") or ""
    if isinstance(evidence, dict):
        evidence = evidence.get("quote") or ""
    return str(evidence).strip()


def _is_decision(item: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
    return (str(item.get("basis") or "").lower() == "decision"
            or str((context or {}).get("point_type") or "").lower() == "decision")


def _has_urgency_cue(item: Dict[str, Any]) -> bool:
    text = f"{_task_text(item)}\n{_evidence_text(item)}"
    return bool(_ASAP_RE.search(text) or _CRITICAL_CUE_RE.search(text))


def _has_critical_evidence(item: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
    """What the 80-100 band requires: a deadline or date, a decision, high priority or a stated urgency."""
    if not is_unspecified(item.get("deadline")):
        return True
    task = _task_text(item)
    if _NUMERIC_DATE_RE.search(task) or _RELATIVE_DATE_RE.search(task) or any(t.startswith("@") for t in fact_tokens(task)):
        return True
    if _is_decision(item, context) or str(item.get("priority") or "").lower() == "high":
        return True
    return _has_urgency_cue(item)


def calibrate(score: int, item: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Tuple[int, Optional[str]]:
    """(score, adjustment): a critical score without critical evidence is capped at MODERATE_CAP."""
    if score >= CRITICAL_MIN and not _has_critical_evidence(item, context):
        return MODERATE_CAP, f"capped from {score}%: no deadline, date, decision, high priority or stated urgency"
    return score, None


def heuristic_importance(item: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> int:
    """Score used when the LLM gives none: priority, then deadline, decision, urgency and vagueness."""
    priority = str(item.get("priority") or "").lower()
    score = _PRIORITY_BASE.get(priority, _NO_PRIORITY_BASE)
    has_deadline = not is_unspecified(item.get("deadline"))
    if has_deadline:
        score += 5
    if _is_decision(item, context):
        score += 5
    if _has_urgency_cue(item):
        score += 5
    if not has_deadline and not _owner(item) and priority != "high":
        score -= 10
    score = min(max(int(round(score / 5.0)) * 5, IMPORTANCE_MIN), IMPORTANCE_MAX)
    return calibrate(score, item, context)[0]


def _item_view(ap_id: str, item: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    # The generator's own priority is left out so the score is judged from the task and discussion.
    deadline = item.get("deadline")
    view: Dict[str, Any] = {
        "id": ap_id,
        "task": _task_text(item),
        "owner": _owner(item) or "Not specified",
        "deadline": "Not specified" if is_unspecified(deadline) else str(deadline).strip(),
    }
    if item.get("basis"):
        view["basis"] = item["basis"]
    evidence = _evidence_text(item)
    if evidence:
        view["evidence"] = evidence
    if context.get("agenda"):
        view["agenda"] = context["agenda"]
    if context.get("point_type"):
        view["point_type"] = context["point_type"]
    source = str(context.get("source_discussion") or "").strip()
    if source:
        view["source_discussion"] = source[:SOURCE_MAX_CHARS]
    return view


def scoring_views(items: List[Dict[str, Any]], contexts: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """What the LLM receives for one batch, ids AP1..APn (also used by the Model Arena)."""
    contexts = list(contexts or [])
    return [
        _item_view(f"AP{n}", item, (contexts[n - 1] if n - 1 < len(contexts) and isinstance(contexts[n - 1], dict) else {}))
        for n, item in enumerate(items, 1)
    ]


def _rows_by_id(result: Any) -> Dict[str, Dict[str, Any]]:
    rows = result
    if isinstance(result, dict):
        rows = next((result[k] for k in ("scores", "action_points", "items") if isinstance(result.get(k), list)), [])
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and str(row.get("id") or "").strip():
            out[str(row["id"]).strip().upper()] = row
    return out


def score_action_points(
    items: List[Dict[str, Any]],
    contexts: Optional[List[Dict[str, Any]]] = None,
    *,
    provider=None,
    label: str = "",
) -> Dict[str, Any]:
    """Set ``importance`` (and its source, reason, adjustment) on every item, in place.

    ``contexts[i]`` describes ``items[i]``: {"agenda", "source_discussion", "point_type"} (all
    optional). Items are sent in batches of SCORE_BATCH_SIZE. Without a ``provider`` one is
    loaded and unloaded afterwards. Returns {"source", "scored", "llm_scored", "capped"}.
    """
    if not items:
        return importance_summary([])
    contexts = [c if isinstance(c, dict) else {} for c in (contexts or [])]
    contexts += [{}] * (len(items) - len(contexts))

    rows: List[Optional[Dict[str, Any]]] = [None] * len(items)
    owns_provider = provider is None
    try:
        if owns_provider:
            from services.ai_provider import get_provider
            provider = get_provider()
        request = getattr(provider, "score_action_point_importance", None)
        if callable(request):
            for start in range(0, len(items), SCORE_BATCH_SIZE):
                indices = list(range(start, min(start + SCORE_BATCH_SIZE, len(items))))
                views = scoring_views([items[i] for i in indices], [contexts[i] for i in indices])
                try:
                    result = request(json.dumps(views, ensure_ascii=False, indent=2))
                except Exception as exc:
                    logger.warning(f"[ActionImportance] {label}: batch {start // SCORE_BATCH_SIZE + 1} failed: {exc}")
                    continue
                by_id = _rows_by_id(result)
                for n, i in enumerate(indices, 1):
                    rows[i] = by_id.get(f"AP{n}")
    except Exception as exc:
        logger.warning(f"[ActionImportance] {label}: LLM scoring unavailable, using heuristic scores: {exc}")
    finally:
        unload = getattr(provider, "unload_model", None)
        if owns_provider and callable(unload):
            unload()

    for item, context, row in zip(items, contexts, rows):
        for key in IMPORTANCE_KEYS:
            item.pop(key, None)
        value = normalize_importance(row.get("importance", row.get("score"))) if row else None
        if value is None:
            item["importance"] = heuristic_importance(item, context)
            item["importance_source"] = "heuristic"
            continue
        value, adjustment = calibrate(value, item, context)
        item["importance"], item["importance_source"] = value, "llm"
        reason = str(row.get("reason") or "").strip()
        if reason:
            item["importance_reason"] = reason[:REASON_MAX_CHARS]
        if adjustment:
            item["importance_adjusted"] = adjustment

    summary = importance_summary(items)
    logger.info(f"[ActionImportance] {label}: {summary['scored']} action point(s) scored "
                f"({summary['llm_scored']} by the LLM, {summary['capped']} capped, source={summary['source']})")
    return summary


def importance_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """{"source": "llm" | "heuristic" | "llm_with_fallback" | "none", "scored", "llm_scored", "capped"}."""
    scored = [it for it in items if isinstance(it, dict) and normalize_importance(it.get("importance")) is not None]
    llm_scored = sum(1 for it in scored if it.get("importance_source") == "llm")
    if not scored:
        source = "none"
    elif llm_scored == len(scored):
        source = "llm"
    else:
        source = "heuristic" if llm_scored == 0 else "llm_with_fallback"
    return {"source": source, "scored": len(scored), "llm_scored": llm_scored,
            "capped": sum(1 for it in scored if it.get("importance_adjusted"))}


# ── Display / export ─────────────────────────────────────────────────────────

def clamp_min_importance(value: Any) -> int:
    """Stored threshold: 0 (off) to 100."""
    try:
        return max(0, min(int(value), IMPORTANCE_MAX))
    except (TypeError, ValueError):
        return 0


def passes_importance(item: Any, min_importance: Any) -> bool:
    """Shown under the threshold: unscored items and a threshold of 0 always pass."""
    threshold = clamp_min_importance(min_importance)
    if threshold <= 0 or not isinstance(item, dict):
        return True
    score = normalize_importance(item.get("importance"))
    return score is None or score >= threshold


def format_importance(item: Any) -> str:
    score = normalize_importance(item.get("importance")) if isinstance(item, dict) else None
    return f"{score}%" if score is not None else ""


def carry_importance(src: Any, dst: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a stored score (and its source/reason/adjustment) from ``src`` onto ``dst``."""
    if not isinstance(src, dict):
        return dst
    score = normalize_importance(src.get("importance"))
    if score is None:
        return dst
    dst["importance"] = score
    for key in IMPORTANCE_KEYS[1:]:
        if src.get(key):
            dst[key] = str(src[key])
    return dst


async def load_min_importance(db, user_id: str) -> int:
    """The user's minimum-importance threshold (0 = show all)."""
    from sqlalchemy import text

    try:
        result = await db.execute(
            text("SELECT action_point_min_importance FROM user_settings WHERE user_id = :uid"), {"uid": user_id},
        )
        row = result.fetchone()
    except Exception as exc:
        logger.warning(f"[ActionImportance] threshold unavailable, showing all action points: {exc}")
        return 0
    return clamp_min_importance(row[0]) if row and row[0] is not None else 0
