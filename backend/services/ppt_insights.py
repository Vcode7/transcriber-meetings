"""
ppt_insights.py — Meeting-date-aware, slide-wise insight generation for PPT Testing.

Pipeline
  1. split_into_slides      — slide / page / section segmentation with numbers and titles
  2. apply_conclusion_cutoff — stop at the conclusion / thank-you slide; drop references,
                               appendix and bibliography slides that follow it
  3. dates & timeline        — parse dates on every slide, place them relative to the meeting
                               date, find the current project phase and upcoming milestones
  4. per-slide LLM analysis  — 0–3 interpreted insights per relevant slide (not bullet copies)
  5. post-processing         — drop insights dated far from the meeting, de-duplicate,
                               prioritise, group by project, write project summaries

Everything except step 4/5's LLM calls is deterministic and unit-tested. The LLM is
injected as a plain `infer(prompt, max_tokens) -> str` callable.
"""
from __future__ import annotations

import calendar
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

InferFn = Callable[[str, int], str]
ProgressFn = Callable[[str, int, int, str], None]

# ── Relevance windows ─────────────────────────────────────────────────────────

FOCUS_WINDOWS: Dict[str, Tuple[int, int]] = {
    # name: (days before the meeting, days after the meeting)
    "narrow": (60, 120),
    "standard": (120, 180),
    "wide": (270, 365),
}
DEFAULT_FOCUS = "standard"

MAX_INSIGHTS_PER_SLIDE = 3
MAX_TOTAL_INSIGHTS = 25
SLIDE_TEXT_LIMIT = 3500
MIN_SLIDE_CHARS = 25


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Slide:
    number: int
    title: str
    text: str
    kind: str = "slide"           # slide | page | section

    def to_dict(self) -> Dict[str, Any]:
        return {"number": self.number, "title": self.title, "kind": self.kind, "chars": len(self.text)}


@dataclass
class DateSpan:
    start: date
    end: date
    raw: str
    precision: str                # day | month | quarter | half | year


@dataclass
class TimelineItem:
    label: str
    start: date
    end: date
    slide: int
    raw: List[str]
    status_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, "start": self.start.isoformat(), "end": self.end.isoformat(),
            "slide": self.slide, "dates": self.raw, "status_hint": self.status_hint,
        }


# ── 1. Slide segmentation ─────────────────────────────────────────────────────

_SLIDE_MARKER = re.compile(r"^\[Slide\s+(\d+)\]\s*$", re.MULTILINE)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip().strip("#*-•|").strip()
        if s and not s.startswith("[Speaker Notes]") and not s.startswith("[Table"):
            return s[:120]
    return ""


def split_into_slides(extracted_text: str, filename: str = "", pages: Optional[List[str]] = None) -> List[Slide]:
    """
    Split extracted document text into numbered units.
      • PPT/PPTX text carries "[Slide N]" markers from doc_extractor.
      • PDFs pass `pages` (one string per page) → one unit per page.
      • Anything else is chunked into ~1500-char sections at paragraph boundaries.
    """
    if pages:
        return [
            Slide(number=i, title=_first_line(t), text=t.strip(), kind="page")
            for i, t in enumerate(pages, 1)
        ]

    text = extracted_text or ""
    markers = list(_SLIDE_MARKER.finditer(text))
    if markers:
        slides = []
        for idx, m in enumerate(markers):
            end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
            body = text[m.end():end].strip()
            slides.append(Slide(number=int(m.group(1)), title=_first_line(body), text=body))
        return slides

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections: List[Slide] = []
    buf: List[str] = []
    size = 0
    for para in paragraphs:
        if buf and size + len(para) > 1500:
            body = "\n\n".join(buf)
            sections.append(Slide(number=len(sections) + 1, title=_first_line(body), text=body, kind="section"))
            buf, size = [], 0
        buf.append(para)
        size += len(para)
    if buf:
        body = "\n\n".join(buf)
        sections.append(Slide(number=len(sections) + 1, title=_first_line(body), text=body, kind="section"))
    return sections


def extract_document_slides(content: bytes, filename: str) -> Tuple[str, List[Slide]]:
    """
    Extract a PPT/PPTX/PDF/DOC/DOCX upload into (full text, slides). Blocking (OCR).
    PDFs go page by page through the production PDF path (text layer + scanned-page OCR);
    everything else through doc_extractor, then split_into_slides.
    Shared by PPT Testing and the Stage 3 agenda document.
    """
    import os
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        import tempfile
        from services.doc_extractor import ocr_pdf_pages_detailed
        from services.ocr_engine import unload_ocr_engine
        from services.ocr_settings import get_ocr_settings
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(content)
            text_all, details, total = ocr_pdf_pages_detailed(tmp_path, get_ocr_settings(), max_pages=10_000)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            unload_ocr_engine()
        pages = [details[i]["text"] if i in details else "" for i in range(1, total + 1)]
        return text_all, split_into_slides(text_all, filename, pages=pages)
    from services.doc_extractor import extract_text_from_bytes
    text_all = extract_text_from_bytes(content, filename)
    return text_all, split_into_slides(text_all, filename)


def slides_to_dicts(slides: List[Slide]) -> List[Dict[str, Any]]:
    return [{"number": s.number, "title": s.title, "text": s.text, "kind": s.kind} for s in slides]


def slides_from_dicts(stored: Iterable[Dict[str, Any]]) -> List[Slide]:
    return [Slide(number=s["number"], title=s.get("title", ""), text=s.get("text", ""), kind=s.get("kind", "slide"))
            for s in stored if isinstance(s, dict)]


# ── 2. Conclusion cutoff ──────────────────────────────────────────────────────

_CONCLUSION_TITLE = re.compile(
    r"^(conclusions?|concluding remarks|closing remarks|final remarks|"
    r"summary\s*(?:&|and)\s*conclusions?|conclusions?\s*(?:&|and)\s*(?:next steps|way forward|recommendations))\b",
    re.IGNORECASE,
)
_CLOSING_TITLE = re.compile(
    r"^(thank\s*you|thanks|many thanks|q\s*&\s*a|questions\s*\??|any questions\s*\??|"
    r"the end|end of (?:the )?presentation|questions\s*(?:&|and)\s*answers)\b",
    re.IGNORECASE,
)
_CLOSING_ANYWHERE = re.compile(r"\b(thank\s*you|thanks|any questions\??|q\s*&\s*a)\b", re.IGNORECASE)
_BACK_MATTER_TITLE = re.compile(
    r"^(references?|bibliography|works cited|citations|sources|appendix|appendices|"
    r"annex(?:ure)?s?|back[- ]?up(?: slides)?|supplementary(?: material)?)\b",
    re.IGNORECASE,
)


def classify_closing(slide: Slide) -> Optional[str]:
    """
    'conclusion' → a real conclusion slide (analysed, then stop)
    'closing'    → thank-you / Q&A / end slide (not analysed, stop)
    'back_matter'→ references / appendix / bibliography divider (not analysed, stop)
    None         → ordinary slide (section-ending or summary slides do NOT stop processing)
    """
    title = (slide.title or "").strip()
    if _CONCLUSION_TITLE.match(title):
        return "conclusion"
    if _CLOSING_TITLE.match(title):
        return "closing"
    if _BACK_MATTER_TITLE.match(title):
        return "back_matter"
    if len(slide.text) <= 160 and _CLOSING_ANYWHERE.search(slide.text):
        return "closing"
    return None


def apply_conclusion_cutoff(slides: List[Slide]) -> Tuple[List[Slide], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Returns (slides to analyse, excluded slide records, cutoff info).
    The first slide is never treated as an ending (title slides often say "Welcome / Thank you").
    """
    for idx, slide in enumerate(slides):
        if idx == 0:
            continue
        kind = classify_closing(slide)
        if not kind:
            continue
        keep_upto = idx + 1 if kind == "conclusion" else idx
        kept = slides[:keep_upto]
        excluded = []
        for s in slides[keep_upto:]:
            reason = "closing_slide" if s is slide else "after_conclusion"
            if s is not slide and classify_closing(s) == "back_matter":
                reason = "back_matter_after_conclusion"
            excluded.append({**s.to_dict(), "reason": reason})
        cutoff = {
            "slide": slide.number,
            "title": slide.title,
            "kind": kind,
            "message": (
                f"Stopped at slide {slide.number} ({slide.title or kind}); "
                f"{len(excluded)} slide(s) after it were excluded."
            ),
        }
        return kept, excluded, cutoff
    return list(slides), [], None


_AGENDA_TITLE = re.compile(
    r"^(agenda|today'?s agenda|contents|table of contents|outline|index|topics|overview of (?:the )?agenda)\b",
    re.IGNORECASE,
)


def is_structural(slide: Slide, position: int) -> Optional[str]:
    """Title, agenda and section-divider slides carry no insights of their own."""
    if position == 0 and slide.kind == "slide" and len(slide.text) < 250:
        return "title"
    if _AGENDA_TITLE.match(slide.title or ""):
        return "agenda"
    lines = [l for l in slide.text.splitlines() if l.strip()]
    if (len(slide.text) < 120 and len(lines) <= 3 and not parse_dates(slide.text)
            and not status_hints(slide.text) and not re.search(r"\d+(?:\.\d+)?\s*%|\$\s*\d", slide.text)):
        return "divider"
    return None


# Insights that only describe the deck itself ("this slide is the agenda", "transition to section 2")
_META_INSIGHT = re.compile(
    r"\b(this slide|the slide (?:serves|shows|is)|agenda (?:includes|covers)|transition (?:point|slide)|"
    r"section (?:marker|divider)|end of section|title slide|presentation (?:title|date))\b",
    re.IGNORECASE,
)


# ── 3. Dates, relevance and timeline ──────────────────────────────────────────

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"


def _year(y: str) -> Optional[int]:
    v = int(y)
    if v < 100:
        v += 2000
    return v if 1990 <= v <= 2100 else None


def _month_span(y: int, m: int) -> Tuple[date, date]:
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


_DATE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "iso"),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?[\s\-/.]+" + _MON + r"\.?[\s\-/.,']+(\d{4}|\d{2})\b", re.I), "dmy_text"),
    (re.compile(r"\b" + _MON + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I), "mdy_text"),
    (re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4}|\d{2})\b"), "numeric"),
    (re.compile(r"\bQ([1-4])[\s\-']*(?:FY|CY)?\s*'?(\d{4}|\d{2})\b", re.I), "quarter"),
    (re.compile(r"\b(?:FY|CY)\s*'?(\d{2,4})\s*[\s\-]*Q([1-4])\b", re.I), "quarter_rev"),
    (re.compile(r"\bH([12])[\s\-']*(?:FY|CY)?\s*'?(\d{4}|\d{2})\b", re.I), "half"),
    (re.compile(r"\b" + _MON + r"\.?(?:\s*[\-'’]\s*|\s+)(\d{4}|\d{2})\b", re.I), "month"),
    (re.compile(r"\bFY\s*'?(\d{4}|\d{2})(?:\s*[-/]\s*(\d{2,4}))?\b", re.I), "fiscal"),
    (re.compile(r"\b((?:19|20)\d{2})\b"), "year"),
]


def parse_dates(text: str) -> List[DateSpan]:
    """Find dates/periods in free text. Earlier (more specific) patterns win overlapping spans."""
    if not text:
        return []
    taken = [False] * len(text)
    spans: List[DateSpan] = []
    for pattern, kind in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            if any(taken[m.start():m.end()]):
                continue
            span = _to_span(kind, m)
            if span is None:
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            spans.append(span)
    return spans


def _to_span(kind: str, m: re.Match) -> Optional[DateSpan]:
    raw = m.group(0).strip()
    g = m.groups()
    if kind == "iso":
        d = _safe_date(int(g[0]), int(g[1]), int(g[2]))
        return DateSpan(d, d, raw, "day") if d else None
    if kind == "dmy_text":
        y = _year(g[2])
        d = _safe_date(y, _MONTHS[g[1].lower()[:3]], int(g[0])) if y else None
        return DateSpan(d, d, raw, "day") if d else None
    if kind == "mdy_text":
        y = _year(g[2])
        d = _safe_date(y, _MONTHS[g[0].lower()[:3]], int(g[1])) if y else None
        return DateSpan(d, d, raw, "day") if d else None
    if kind == "numeric":
        y = _year(g[2])
        a, b = int(g[0]), int(g[1])
        day, month = (a, b) if b <= 12 else (b, a)  # dd/mm by default, swap when unambiguous
        d = _safe_date(y, month, day) if y else None
        return DateSpan(d, d, raw, "day") if d else None
    if kind in ("quarter", "quarter_rev"):
        q, y = (int(g[0]), _year(g[1])) if kind == "quarter" else (int(g[1]), _year(g[0]))
        if not y:
            return None
        start = date(y, 3 * (q - 1) + 1, 1)
        end = _month_span(y, 3 * q)[1]
        return DateSpan(start, end, raw, "quarter")
    if kind == "half":
        h, y = int(g[0]), _year(g[1])
        if not y:
            return None
        return DateSpan(date(y, 1 if h == 1 else 7, 1), date(y, 6, 30) if h == 1 else date(y, 12, 31), raw, "half")
    if kind == "month":
        y = _year(g[1])
        if not y:
            return None
        start, end = _month_span(y, _MONTHS[g[0].lower()[:3]])
        return DateSpan(start, end, raw, "month")
    if kind == "fiscal":
        y = _year(g[0])
        if not y:
            return None
        if g[1]:  # FY2026-27 → Apr 2026 – Mar 2027
            return DateSpan(date(y, 4, 1), date(y + 1, 3, 31), raw, "year")
        return DateSpan(date(y - 1, 4, 1), date(y, 3, 31), raw, "year")  # FY26 ends Mar 2026
    if kind == "year":
        y = _year(g[0])
        return DateSpan(date(y, 1, 1), date(y, 12, 31), raw, "year") if y else None
    return None


def temporal_class(span: DateSpan, meeting: date, window: Tuple[int, int]) -> str:
    """current | recent | upcoming | past | future relative to the meeting date."""
    past_days, future_days = window
    if span.start <= meeting <= span.end:
        return "current"
    if span.end < meeting:
        return "recent" if (meeting - span.end).days <= past_days else "past"
    return "upcoming" if (span.start - meeting).days <= future_days else "future"


def describe_offset(span: DateSpan, meeting: date) -> str:
    if span.start <= meeting <= span.end:
        return "covers the meeting date"
    if span.end < meeting:
        days = (meeting - span.end).days
        return f"{_human_days(days)} before the meeting"
    days = (span.start - meeting).days
    return f"{_human_days(days)} after the meeting"


def _human_days(days: int) -> str:
    if days < 14:
        return f"{days} day{'s' if days != 1 else ''}"
    if days < 60:
        return f"{round(days / 7)} weeks"
    if days < 730:
        return f"{round(days / 30.4)} months"
    return f"{days / 365.25:.1f} years"


_RELEVANT = {"current", "recent", "upcoming"}

_LIVE_STATUS = re.compile(
    r"\b(on[- ]?going|in[- ]progress|under[- ]?way|current(?:ly)?|delay(?:ed)?|blocked|at risk|"
    r"next steps?|action items?|upcoming|pending|open issues?|risks?|issues?)\b",
    re.IGNORECASE,
)

STATUS_PATTERNS: Dict[str, re.Pattern] = {
    "completed": re.compile(
        r"(\b(completed?|done|closed|delivered|achieved|finished|signed[- ]off|go[- ]live achieved|commissioned)\b|"
        r"\b100\s*%|✓|✔|☑|✅)",
        re.IGNORECASE,
    ),
    "delayed": re.compile(
        r"\b(delay(?:ed|s)?|slipp(?:ed|age)|behind schedule|overdue|on hold|blocked|postponed|deferred|"
        r"rescheduled|at risk|missed)\b",
        re.IGNORECASE,
    ),
    "ongoing": re.compile(
        r"\b(in[- ]progress|on[- ]?going|under[- ]?way|wip|work in progress|in execution|"
        r"\d{1,2}\s*%\s*(?:complete|completed|done))\b",
        re.IGNORECASE,
    ),
    "planned": re.compile(
        r"\b(planned|upcoming|scheduled|to be|will be|target(?:ed)?|proposed|yet to start|not started|"
        r"next phase|kick[- ]?off)\b",
        re.IGNORECASE,
    ),
}


def status_hints(text: str) -> Dict[str, List[str]]:
    hints: Dict[str, List[str]] = {}
    for status, pattern in STATUS_PATTERNS.items():
        found = sorted({m.group(0).strip() for m in pattern.finditer(text or "")})
        if found:
            hints[status] = found[:6]
    return hints


_PARTIAL_PERCENT = re.compile(r"\b\d{1,2}(?:\.\d+)?\s*%\s*(?:complete|completed|done)\b", re.IGNORECASE)


def line_status(text: str) -> str:
    if STATUS_PATTERNS["delayed"].search(text or ""):
        return "delayed"
    if _PARTIAL_PERCENT.search(text or ""):  # "70% complete" is still ongoing
        return "ongoing"
    for status in ("completed", "ongoing", "planned"):
        if STATUS_PATTERNS[status].search(text or ""):
            return status
    return ""


def slide_relevance(slide: Slide, meeting: date, window: Tuple[int, int]) -> Tuple[str, List[DateSpan]]:
    """
    'dated'     → at least one date near the meeting
    'undated'   → no dates found (analysed; status is judged from content)
    'out_of_window' → every date is far from the meeting and no live status words
    """
    spans = parse_dates(slide.text)
    if not spans:
        return "undated", spans
    classes = {temporal_class(s, meeting, window) for s in spans}
    if classes & _RELEVANT:
        return "dated", spans
    # Old dates but explicitly still-live work ("contract 2019; migration still ongoing") stays relevant.
    # Far-future dates never do: that work hasn't started yet as of the meeting.
    if classes == {"past"} and _LIVE_STATUS.search(slide.text):
        return "dated", spans
    return "out_of_window", spans


_PHASE_WORDS = re.compile(
    r"\b(phase|stage|sprint|milestone|release|wave|pilot|roll[- ]?out|go[- ]live|implementation|"
    r"deployment|design|build|testing|uat|launch|construction|commissioning|handover|delivery)\b",
    re.IGNORECASE,
)


def _clean_label(line: str, spans: List[DateSpan]) -> str:
    label = line
    for s in spans:
        label = label.replace(s.raw, " ⟨D⟩ ")
    # Drop connectives that only introduced a date ("delayed from <D> to <D>", "planned for <D>")
    label = re.sub(r"\b(?:from|to|till|until|by|on|in|between|and|for|since)\s*(?=⟨D⟩)", " ", label, flags=re.I)
    label = label.replace("⟨D⟩", " ")
    label = re.sub(r"[|]+", " · ", label)
    label = re.sub(r"\b(to|till|until|from|by|on|between|and|for)\b\s*(?=[·\-–—:,;\s]*$)", " ", label, flags=re.I)
    label = re.sub(r"[\s·:\-–—,()]+$", "", re.sub(r"^[\s·:\-–—,()•*]+", "", label))
    label = re.sub(r"\s{2,}", " ", label).strip()
    label = re.sub(r"(\s*·\s*)+", " · ", label).strip(" ·")
    return label[:100]


def extract_timeline(slides: Iterable[Slide]) -> List[TimelineItem]:
    """One item per line (or table row) that carries at least one date."""
    items: List[TimelineItem] = []
    slides = list(slides)
    for slide in slides:
        if slide.number == 1 and len(slides) > 2:
            continue  # title slide dates are presentation dates, not timeline entries
        for line in slide.text.splitlines():
            if not line.strip() or set(line.strip()) <= set("|-: "):
                continue
            spans = parse_dates(line)
            if not spans:
                continue
            label = _clean_label(line, spans) or slide.title
            items.append(TimelineItem(
                label=label,
                start=min(s.start for s in spans),
                end=max(s.end for s in spans),
                slide=slide.number,
                raw=[s.raw for s in spans],
                status_hint=line_status(line),
            ))
    return items


def timeline_context(items: List[TimelineItem], meeting: date, window: Tuple[int, int]) -> Dict[str, Any]:
    """Current phase, upcoming milestones and recently finished items around the meeting date."""
    past_days, future_days = window
    current = [i for i in items if i.start <= meeting <= i.end and i.start != i.end and (i.end - i.start).days < 1100]
    # Prefer phase-like rows and the most specific (shortest) span
    current.sort(key=lambda i: (not _PHASE_WORDS.search(i.label), (i.end - i.start).days))
    upcoming = sorted(
        (i for i in items if meeting < i.start <= meeting + timedelta(days=future_days)),
        key=lambda i: i.start,
    )
    recent = sorted(
        (i for i in items if meeting - timedelta(days=past_days) <= i.end < meeting),
        key=lambda i: i.end, reverse=True,
    )
    # Explicitly planned / ongoing / delayed work whose end date has already passed
    overdue = [i for i in recent if i.status_hint in ("planned", "ongoing", "delayed")]

    def dedupe(seq: List[TimelineItem], limit: int) -> List[Dict[str, Any]]:
        out, seen = [], set()
        for i in seq:
            key = (i.label.lower(), i.start, i.end)
            if key in seen:
                continue
            seen.add(key)
            out.append(i.to_dict())
            if len(out) >= limit:
                break
        return out

    return {
        "current_phase": dedupe(current, 3),
        "upcoming": dedupe(upcoming, 5),
        "recent": dedupe(recent, 5),
        "possibly_overdue": dedupe(overdue, 5),
    }


def phase_summary(ctx: Dict[str, Any]) -> str:
    cur = ctx.get("current_phase") or []
    if not cur:
        return "not stated in the deck"
    top = cur[0]
    return f"{top['label']} ({top['start']} to {top['end']}, slide {top['slide']})"


# ── 4. LLM helpers ────────────────────────────────────────────────────────────

def _close_brackets(text: str) -> str:
    """Append the closing brackets a truncated JSON value is missing (strings are respected)."""
    stack: List[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    return text.rstrip().rstrip(",") + "".join(reversed(stack))


def parse_json_object(raw: str) -> Dict[str, Any]:
    """Tolerant JSON extraction from LLM output."""
    if not raw:
        return {}
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])
    if start != -1:
        # Small local models sometimes stop before the final closing brackets.
        candidates.append(_close_brackets(cleaned[start:]))
    for cand in candidates:
        for attempt in (cand, re.sub(r",\s*([\]}])", r"\1", cand)):
            try:
                value = json.loads(attempt)
                if isinstance(value, dict):
                    return value
            except Exception:
                continue
    return {}


INSIGHT_TYPES = ("delay", "risk", "decision", "dependency", "action", "milestone", "progress", "status")
STATUSES = ("completed", "ongoing", "delayed", "at_risk", "planned", "unknown")

_TYPE_WEIGHT = {
    "delay": 1.0, "risk": 1.0, "decision": 0.9, "action": 0.9, "dependency": 0.85,
    "milestone": 0.8, "progress": 0.7, "status": 0.6,
}
_TEMPORAL_WEIGHT = {"current": 1.0, "upcoming_soon": 0.95, "upcoming": 0.8, "recent": 0.75, "undated": 0.6}


def build_slide_prompt(
    slide: Slide,
    *,
    meeting: date,
    window: Tuple[int, int],
    deck_title: str,
    phase: str,
    prev_title: str,
    next_title: str,
    spans: List[DateSpan],
    already_reported: Optional[List[str]] = None,
) -> str:
    past_days, future_days = window
    win_start = (meeting - timedelta(days=past_days)).strftime("%b %Y")
    win_end = (meeting + timedelta(days=future_days)).strftime("%b %Y")
    date_lines = []
    seen = set()
    for s in spans:
        if s.raw in seen:
            continue
        seen.add(s.raw)
        cls = temporal_class(s, meeting, window)
        tag = {"current": "CURRENT", "recent": "recent", "upcoming": "upcoming",
               "past": "OUTSIDE WINDOW – old", "future": "OUTSIDE WINDOW – far future"}[cls]
        date_lines.append(f'- "{s.raw}" → {describe_offset(s, meeting)} ({tag})')
        if len(date_lines) >= 14:
            break
    hints = status_hints(slide.text)
    hint_text = "; ".join(f"{k}: {', '.join(v)}" for k, v in hints.items()) or "none"
    unit = {"slide": "SLIDE", "page": "PAGE", "section": "SECTION"}[slide.kind]
    body = slide.text[:SLIDE_TEXT_LIMIT]
    reported = "\n".join(f"- {h}" for h in (already_reported or [])[-15:]) or "- nothing yet"
    closing_note = ""
    if classify_closing(slide) == "conclusion":
        closing_note = ("\nThis is the presentation's CONCLUSION. Do not restate items already reported; "
                        "return at most one overall assessment or a genuinely new commitment, or nothing.\n")

    return f"""You are preparing a briefing for a project review meeting held on {meeting.strftime('%d %b %Y')}.
Analyse ONE presentation {unit.lower()} and report only what matters for this meeting.

DECK: {deck_title or 'untitled'}
CURRENT PROJECT PHASE (from the deck timeline): {phase}
NEIGHBOURING {unit}S: previous "{prev_title or '-'}" | next "{next_title or '-'}"

{unit} {slide.number} — "{slide.title or 'untitled'}"
<<<
{body}
>>>

DATES ON THIS {unit} (relative to the meeting date):
{chr(10).join(date_lines) or '- none'}
STATUS WORDS FOUND: {hint_text}

ALREADY REPORTED FROM EARLIER {unit}S (do not repeat these; report only new facts or a changed status):
{reported}
{closing_note}
RULES
1. Interpret, do not copy. Each insight explains what changed or stands, why it matters, and what needs attention. Merge related bullets into one insight.
2. Focus on work active around the meeting ({win_start} – {win_end}): ongoing work, deadlines and milestones in that window, delays, risks, decisions, dependencies and action items. Skip items marked OUTSIDE WINDOW unless the slide shows they still affect current work.
3. status must come from the slide: completed | ongoing | delayed | at_risk | planned | unknown. Use "delayed" only if the slide says so, or a planned date before the meeting date is not shown as done (then write "planned for <date>, no completion shown").
4. Keep names, numbers and dates exactly as written. Never invent owners, dates, causes or percentages.
5. Return 0 to {MAX_INSIGHTS_PER_SLIDE} insights. Return an empty list for title, agenda, divider, contact or purely background content. Never describe the slide itself (e.g. "this slide shows").
6. "project" is the specific project or workstream the {unit.lower()} is about (e.g. "Project Beacon"); for programme-wide content use the programme name.

Return JSON only:
{{"project": "<main project or workstream on this {unit.lower()}, or empty>",
 "insights": [{{"type": "delay|risk|decision|dependency|action|milestone|progress|status",
   "status": "completed|ongoing|delayed|at_risk|planned|unknown",
   "headline": "<at most 12 words>",
   "summary": "<1-2 sentences: what changed or stands and why it matters>",
   "attention": "<what needs attention or the next step, or empty>",
   "dates": ["<dates exactly as written on the slide>"],
   "owner": "<owner if stated, else empty>"}}]}}"""


def _norm_choice(value: Any, allowed: Tuple[str, ...], default: str) -> str:
    v = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if v == "at_risk" or v == "atrisk":
        v = "at_risk"
    return v if v in allowed else default


def _clean_str(value: Any, limit: int = 600) -> str:
    if isinstance(value, list):
        value = "; ".join(str(v) for v in value if v)
    s = re.sub(r"\s+", " ", str(value or "")).strip().strip('"').strip()
    return s[:limit]


def normalise_insights(parsed: Dict[str, Any], slide: Slide) -> Tuple[str, List[Dict[str, Any]]]:
    project = _clean_str(parsed.get("project"), 80)
    raw_items = parsed.get("insights") or []
    if not isinstance(raw_items, list):
        raw_items = []
    out = []
    for item in raw_items[:MAX_INSIGHTS_PER_SLIDE]:
        if not isinstance(item, dict):
            continue
        headline = _clean_str(item.get("headline"), 140)
        summary = _clean_str(item.get("summary"), 600)
        if len(headline) + len(summary) < 15:
            continue
        dates = item.get("dates") or []
        if isinstance(dates, str):
            dates = [dates]
        out.append({
            "type": _norm_choice(item.get("type"), INSIGHT_TYPES, "status"),
            "status": _norm_choice(item.get("status"), STATUSES, "unknown"),
            "headline": headline or summary[:80],
            "summary": summary,
            "attention": _clean_str(item.get("attention"), 300),
            "dates": [_clean_str(d, 60) for d in dates if _clean_str(d, 60)][:6],
            "owner": _clean_str(item.get("owner"), 80),
        })
    return project, out


def insight_temporal(insight: Dict[str, Any], slide_spans: List[DateSpan], meeting: date,
                     window: Tuple[int, int]) -> Tuple[str, List[DateSpan]]:
    """Temporal class of an insight from its own dates, falling back to its slide's dates."""
    spans: List[DateSpan] = []
    for d in insight.get("dates", []):
        spans.extend(parse_dates(d))
    if not spans:
        text = f"{insight.get('headline', '')} {insight.get('summary', '')}"
        spans = parse_dates(text)
    if not spans:
        # Undated insight: inherit the nearest relevant timing of its slide, never exclude it
        nearest = [s for s in slide_spans if temporal_class(s, meeting, window) in _RELEVANT]
        if not nearest:
            return "undated", []
        cls, nearest_spans = insight_temporal({"dates": [s.raw for s in nearest]}, [], meeting, window)
        return (cls if cls != "out_of_window" else "undated"), nearest_spans
    classes = [temporal_class(s, meeting, window) for s in spans]
    # A range written as two dates ("Apr 2026 – Dec 2026") spanning the meeting is current work
    spans_meeting = min(s.start for s in spans) <= meeting <= max(s.end for s in spans)
    if "current" in classes or (spans_meeting and len(spans) > 1 and set(classes) & {"recent", "upcoming"}):
        return "current", spans
    if "upcoming" in classes:
        soonest = min((s.start - meeting).days for s, c in zip(spans, classes) if c == "upcoming")
        return ("upcoming_soon" if soonest <= 30 else "upcoming"), spans
    if "recent" in classes:
        return "recent", spans
    return "out_of_window", spans


def priority_score(insight: Dict[str, Any], temporal: str, spans: List[DateSpan], meeting: date) -> float:
    score = 0.5 * _TEMPORAL_WEIGHT.get(temporal, 0.5) + 0.4 * _TYPE_WEIGHT.get(insight["type"], 0.6)
    if insight["status"] in ("delayed", "at_risk"):
        score += 0.1
    elif insight["status"] == "ongoing":
        score += 0.05
    if insight.get("attention"):
        score += 0.03
    if any(s.start.year <= meeting.year <= s.end.year for s in spans):
        score += 0.05
    return round(min(score, 1.2), 4)


# ── 5. De-duplication and grouping ────────────────────────────────────────────

class _DifflibFuzz:
    """Subset of rapidfuzz.fuzz (0–100 scores) used when rapidfuzz is not installed."""

    @staticmethod
    def ratio(a: str, b: str) -> float:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0

    @classmethod
    def token_set_ratio(cls, a: str, b: str) -> float:
        ta = set(re.findall(r"\w+", (a or "").lower()))
        tb = set(re.findall(r"\w+", (b or "").lower()))
        if not ta or not tb:
            return 0.0
        inter = " ".join(sorted(ta & tb))
        diff_a = " ".join(sorted(ta - tb))
        diff_b = " ".join(sorted(tb - ta))
        s1, s2 = f"{inter} {diff_a}".strip(), f"{inter} {diff_b}".strip()
        scores = [cls.ratio(s1, s2)]
        if inter:
            scores += [cls.ratio(inter, s1), cls.ratio(inter, s2)]
        return max(scores)


def _fuzz():
    try:
        from rapidfuzz import fuzz
        return fuzz
    except ImportError:
        return _DifflibFuzz


def _project_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def same_project(a: str, b: str) -> bool:
    ka, kb = _project_key(a), _project_key(b)
    if not ka or not kb:
        return False
    if ka == kb or (len(min(ka, kb, key=len)) >= 4 and (ka in kb or kb in ka)):
        return True
    return _fuzz().ratio(ka, kb) >= 85


def _repeats_earlier(item: Dict[str, Any], earlier: List[Dict[str, Any]], threshold: int = 85) -> bool:
    fuzz = _fuzz()
    text = f"{item.get('headline', '')} {item.get('summary', '')}"
    return any(fuzz.token_set_ratio(text, f"{e['headline']} {e['summary']}") >= threshold for e in earlier)


def spans_overlap(a: DateSpan, b: DateSpan) -> bool:
    return a.start <= b.end and b.start <= a.end


def grounded_dates(dates: List[str], source_text: str) -> List[str]:
    """Keep only dates that also appear (as the same or an overlapping period) in the source text."""
    source = parse_dates(source_text)
    out = []
    for d in dates:
        spans = parse_dates(d)
        if not spans:
            if d.lower() in source_text.lower():
                out.append(d)
            continue
        if all(any(spans_overlap(s, src) for src in source) for s in spans):
            out.append(d)
    return out


def ungrounded_dates(text: str, sources: List[str]) -> List[str]:
    """Day-precise dates in `text` that appear in none of the sources (likely invented)."""
    source_spans = [s for src in sources for s in parse_dates(src)]
    bad = []
    for span in parse_dates(text):
        if span.precision != "day":
            continue
        if not any(src.precision == "day" and src.start == span.start for src in source_spans):
            bad.append(span.raw)
    return bad


def deduplicate(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge near-identical insights (same project, ≥ 85% token overlap); keep the stronger one."""
    fuzz = _fuzz()
    kept: List[Dict[str, Any]] = []
    for ins in sorted(insights, key=lambda i: i["priority"], reverse=True):
        text = f"{ins['headline']} {ins['summary']}"
        dup = None
        for k in kept:
            projects_compatible = not ins["project"] or not k["project"] or same_project(ins["project"], k["project"])
            if projects_compatible and fuzz.token_set_ratio(text, f"{k['headline']} {k['summary']}") >= 85:
                dup = k
                break
        if dup:
            dup["slides"] = sorted(set(dup["slides"]) | set(ins["slides"]))
            dup["dates"] = list(dict.fromkeys(dup["dates"] + ins["dates"]))[:6]
            if not dup["attention"] and ins["attention"]:
                dup["attention"] = ins["attention"]
        else:
            kept.append(ins)
    return kept


def group_by_project(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for ins in insights:
        name = ins["project"] or "General"
        target = next((g for g in groups if g["name"] == name or same_project(g["name"], name)), None)
        if target is None:
            target = {"name": name, "insights": []}
            groups.append(target)
        target["insights"].append(ins)
    for g in groups:
        g["insights"].sort(key=lambda i: (min(i["slides"]), -i["priority"]))
        g["slides"] = sorted({s for i in g["insights"] for s in i["slides"]})
        g["priority"] = max(i["priority"] for i in g["insights"])
        statuses = {i["status"] for i in g["insights"]} - {"unknown"}
        if "delayed" in statuses:
            g["overall_status"] = "delayed"
        elif "at_risk" in statuses:
            g["overall_status"] = "at_risk"
        elif statuses == {"completed"}:
            g["overall_status"] = "completed"
        elif "ongoing" in statuses:
            g["overall_status"] = "ongoing"
        elif statuses == {"planned"}:
            g["overall_status"] = "planned"
        else:
            g["overall_status"] = "mixed" if statuses else "unknown"
    groups.sort(key=lambda g: (g["name"] == "General", -g["priority"]))
    return groups


def build_project_prompt(group: Dict[str, Any], meeting: date) -> str:
    lines = []
    for i in group["insights"]:
        dates = f" [dates: {', '.join(i['dates'])}]" if i["dates"] else ""
        att = f" Needs attention: {i['attention']}" if i["attention"] else ""
        lines.append(f"- (slide {', '.join(map(str, i['slides']))}; {i['status']}; {i['type']}) "
                     f"{i['headline']}: {i['summary']}{att}{dates}")
    return f"""You are writing a pre-meeting brief for a review meeting on {meeting.strftime('%d %b %Y')}.
Summarise the status of "{group['name']}" using ONLY the findings below.

FINDINGS:
{chr(10).join(lines)}

Write 2-3 sentences covering: where the work stands on the meeting date, what changed, why it matters.
Then list the items that need attention (at most 3). Do not add facts, dates or owners that are not in the findings.

Return JSON only:
{{"summary": "<2-3 sentences>", "needs_attention": ["<item>", "..."]}}"""


def build_overview_prompt(projects: List[Dict[str, Any]], meeting: date, phase: str, deck_title: str) -> str:
    lines = [f"- {p['name']} ({p['overall_status']}): {p.get('summary') or p['insights'][0]['summary']}" for p in projects[:10]]
    return f"""Write a 2-3 sentence executive overview of the presentation "{deck_title or 'untitled'}" for a meeting on {meeting.strftime('%d %b %Y')}.
Current phase per the deck timeline: {phase}.
Project briefs:
{chr(10).join(lines)}

Say where things stand on the meeting date and the most important item needing attention. Use only the facts above.
Return only the overview text."""


# ── Orchestration ─────────────────────────────────────────────────────────────

def analyze_presentation(
    slides: List[Slide],
    *,
    meeting: date,
    infer: InferFn,
    focus: str = DEFAULT_FOCUS,
    filename: str = "",
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Run the full slide-wise analysis. Returns a JSON-serialisable dict."""
    window = FOCUS_WINDOWS.get(focus, FOCUS_WINDOWS[DEFAULT_FOCUS])
    started = time.perf_counter()
    report = progress or (lambda *_: None)

    kept, excluded, cutoff = apply_conclusion_cutoff(slides)
    timeline_items = extract_timeline(kept)
    ctx = timeline_context(timeline_items, meeting, window)
    phase = phase_summary(ctx)
    deck_title = slides[0].title if slides else filename

    slide_records: List[Dict[str, Any]] = []
    candidates: List[Tuple[int, Slide, List[DateSpan]]] = []
    for idx, slide in enumerate(kept):
        relevance, spans = slide_relevance(slide, meeting, window)
        record = {**slide.to_dict(), "relevance": relevance,
                  "dates": [s.raw for s in spans][:8], "status": "pending", "insight_count": 0}
        structural = is_structural(slide, idx)
        if len(slide.text) < MIN_SLIDE_CHARS:
            record["status"] = "skipped_empty"
        elif structural and not (idx == len(kept) - 1 and cutoff and cutoff["kind"] == "conclusion"):
            record["status"] = "skipped_structural"
            record["structural"] = structural
        elif relevance == "out_of_window":
            record["status"] = "skipped_out_of_window"
        else:
            candidates.append((idx, slide, spans))
        slide_records.append(record)
    record_by_number = {r["number"]: r for r in slide_records}

    insights: List[Dict[str, Any]] = []
    dropped_far = 0
    for step, (idx, slide, spans) in enumerate(candidates, 1):
        report("slides", step, len(candidates), f"Analysing {slide.kind} {slide.number} of {len(slides)}")
        prompt = build_slide_prompt(
            slide, meeting=meeting, window=window, deck_title=deck_title, phase=phase,
            prev_title=kept[idx - 1].title if idx > 0 else "",
            next_title=kept[idx + 1].title if idx + 1 < len(kept) else "",
            spans=spans,
            already_reported=[f"(slide {i['slides'][0]}) {i['headline']}" for i in insights],
        )
        record = record_by_number[slide.number]
        try:
            raw = infer(prompt, 900)
        except Exception as exc:
            logger.warning(f"[PPTInsights] LLM failed on slide {slide.number}: {exc}")
            record["status"] = "error"
            continue
        project, items = normalise_insights(parse_json_object(raw), slide)
        is_conclusion = classify_closing(slide) == "conclusion"
        if is_conclusion:
            # A conclusion restates the deck: keep at most one item that adds something new
            items = [it for it in items if not _repeats_earlier(it, insights, threshold=65)][:1]
        kept_here = 0
        for item in items:
            if _META_INSIGHT.search(f"{item['headline']} {item['summary']}"):
                continue
            item["dates"] = grounded_dates(item["dates"], slide.text)
            temporal, ins_spans = insight_temporal(item, spans, meeting, window)
            if temporal == "out_of_window":
                dropped_far += 1
                continue
            item.update({
                "id": f"ins-{uuid.uuid4().hex[:10]}",
                "project": project,
                "slides": [slide.number],
                "slide_title": slide.title,
                "temporal": temporal,
                "priority": priority_score(item, temporal, ins_spans, meeting),
            })
            insights.append(item)
            kept_here += 1
        record["status"] = "analyzed"
        record["insight_count"] = kept_here
        record["project"] = project

    insights = deduplicate(insights)
    if len(insights) > MAX_TOTAL_INSIGHTS:
        insights = sorted(insights, key=lambda i: i["priority"], reverse=True)[:MAX_TOTAL_INSIGHTS]
    insights.sort(key=lambda i: (min(i["slides"]), -i["priority"]))
    for n, ins in enumerate(insights, 1):
        ins["ref"] = f"P{n}"

    projects = group_by_project(insights)
    for p_idx, group in enumerate(projects, 1):
        report("projects", p_idx, len(projects), f"Summarising {group['name']}")
        group["summary"], group["needs_attention"] = "", []
        group["insight_ids"] = [i["id"] for i in group["insights"]]
        if len(group["insights"]) == 1:
            only = group["insights"][0]
            group["summary"] = only["summary"]
            group["needs_attention"] = [only["attention"]] if only["attention"] else []
            continue
        try:
            parsed = parse_json_object(infer(build_project_prompt(group, meeting), 500))
            group["summary"] = _clean_str(parsed.get("summary"), 800)
            needs = parsed.get("needs_attention") or []
            group["needs_attention"] = [_clean_str(n, 200) for n in (needs if isinstance(needs, list) else [needs]) if _clean_str(n, 200)][:3]
        except Exception as exc:
            logger.warning(f"[PPTInsights] Project summary failed for {group['name']}: {exc}")
        if not group["summary"]:
            group["summary"] = " ".join(i["summary"] for i in group["insights"][:2])

    overview = ""
    if projects:
        report("overview", 1, 1, "Writing overview")
        try:
            overview = re.sub(r"<think>.*?</think>", "", infer(build_overview_prompt(projects, meeting, phase, deck_title), 300) or "",
                              flags=re.DOTALL).strip().strip('"')
        except Exception as exc:
            logger.warning(f"[PPTInsights] Overview failed: {exc}")
        if len(overview) < 20:
            overview = projects[0]["summary"]

    projects_out = [{k: v for k, v in g.items() if k != "insights"} for g in projects]
    return {
        "version": 2,
        "meeting_date": meeting.isoformat(),
        "focus": focus,
        "window": {"from": (meeting - timedelta(days=window[0])).isoformat(),
                   "to": (meeting + timedelta(days=window[1])).isoformat()},
        "filename": filename,
        "deck_title": deck_title,
        "overview": overview,
        "timeline": {**ctx, "phase_text": phase},
        "cutoff": cutoff,
        "excluded_slides": excluded,
        "slides": slide_records,
        "slides_total": len(slides),
        "slides_analyzed": sum(1 for r in slide_records if r["status"] == "analyzed"),
        "dropped_out_of_window": dropped_far,
        "projects": projects_out,
        "insights": insights,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }


def insight_point_text(ins: Dict[str, Any]) -> str:
    """Flat, traceable text form of an insight (used by matching and legacy views)."""
    slides = ", ".join(map(str, ins.get("slides", [])))
    head = f"[Slide {slides}" + (f" · {ins['project']}" if ins.get("project") else "") + "]"
    status = ins.get("status", "unknown").replace("_", " ")
    text = f"{head} {ins['headline']} — {ins['summary']}"
    if status != "unknown":
        text += f" (Status: {status})"
    if ins.get("attention"):
        text += f" Attention: {ins['attention']}"
    return text
