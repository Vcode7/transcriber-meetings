from transformers.utils import dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects
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

        return {
            "assigner": _clean(assigner),
            "assignee": _clean(assignee),
            "task": raw_task,
            "deadline": _clean(deadline)
        }
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
    return {
        "task": norm["task"],
        "assignee": norm["assignee"],
        "deadline": norm["deadline"],
    }


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

    text = task

    if deadline:
        text_lower = text.lower()
        d_lower = str(deadline).lower()
        if not (text_lower.endswith(f"before {d_lower}") or text_lower.endswith(f"by {d_lower}")):
            text = f"{text} before {deadline}"

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
    ) -> Dict:
        """Stage 1: Process transcript in sliding windows to extract discussion points.

        Args:
            separate_action_extraction: When True, each window runs two parallel LLM calls —
                one for discussion points (action_items omitted) and one for action items
                only.  The action items are then attached to the first discussion point and
                action_owner is derived programmatically.  When False (default), the existing
                single-call pipeline is used unchanged.
        """
        from services.ai_provider import get_provider
        from services.video_processing_service import get_overlapping_ocr_blocks
        from config import settings
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if user_id:
            user_id = _validate_user_id(user_id)
            
        if not transcript:
            return {"discussion_points": [], "windows_processed": 0}
            
        provider = get_provider()
        
        windows = []
        current_window = []
        current_start_time = transcript[0].get('start', 0.0)
        window_seconds = window_minutes * 60.0
        
        for segment in transcript:
            seg_start = segment.get('start', 0.0)
            if current_window and (seg_start - current_start_time) >= window_seconds:
                windows.append(current_window)
                current_window = [segment]
                current_start_time = seg_start
            else:
                current_window.append(segment)
                
        if current_window: 
            windows.append(current_window)
            
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
            f" | video_ocr={'enabled' if has_video_ocr else 'disabled'}"
            f" | parallel_concurrency={concurrency}"
        )

        def _process_single_window(i_and_win):
            i, window = i_and_win
            _win_start_time = time.monotonic()
            w_start = round(window[0].get('start', 0.0), 2) if window else 0.0
            w_end = round(window[-1].get('end', 0.0), 2) if window else 0.0
            
            t_range = f"{_format_time_hhmm(w_start)}-{_format_time_hhmm(w_end)}"
            logger.info(f"[ROM Service] Processing Window {i+1}/{total_windows} ({t_range}) [Parallel limit={concurrency}]")

            window_text = ""
            for seg in window:
                # Always prefer the resolved speaker label (e.g. "Speaker 1", "John")
                # so the LLM never receives raw Pyannote IDs like SPEAKER_00.
                speaker = seg.get('speaker_label') or seg.get('speaker') or 'Unknown'
                start = seg.get('start', 0.0)
                end = seg.get('end', 0.0)
                text = seg.get('text', '').strip()
                window_text += f"[{start:.1f}-{end:.1f}] {speaker}: {text}\n"

            video_context = ""
            if has_video_ocr:
                relevant_blocks = get_overlapping_ocr_blocks(video_transcript, w_start, w_end)
                if relevant_blocks:
                    lines = []
                    for b in relevant_blocks:
                        ts = f"{_format_time_hhmm(b['start'])} -> {_format_time_hhmm(b['end'])}"
                        lines.append(f"[{ts}]\n{b['text']}")
                    video_context = "\n\n".join(lines)

            # Format previous window transcript text as context when no previous points exist
            prev_window_text = ""
            if i > 0 and i - 1 < len(windows):
                prev_win = windows[i - 1]
                p_lines = []
                for p_seg in prev_win:
                    # Always prefer the resolved speaker label so the LLM never receives raw Pyannote IDs.
                    p_spk = p_seg.get('speaker_label') or p_seg.get('speaker') or 'Unknown'
                    p_start = p_seg.get('start', 0.0)
                    p_end = p_seg.get('end', 0.0)
                    p_txt = p_seg.get('text', '').strip()
                    p_lines.append(f"[{p_start:.1f}-{p_end:.1f}] {p_spk}: {p_txt}")
                prev_window_text = "PREVIOUS TRANSCRIPT WINDOW:\n" + "\n".join(p_lines)

            if separate_action_extraction:
                # ── Split-call path: run both LLM calls in parallel ──────────────────────
                # Call 1: Discussion points (no action_items) via no-actions prompt variant.
                # Call 2: Action items only via dedicated action extraction prompt.
                #         Action extraction uses a 2× window (current + next) for broader context.
                from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac

                # Build 2× action-point window text (current window + next window)
                double_window_text = window_text
                if i + 1 < len(windows):
                    next_win = windows[i + 1]
                    for n_seg in next_win:
                        n_spk = n_seg.get('speaker_label') or n_seg.get('speaker') or 'Unknown'
                        n_start = n_seg.get('start', 0.0)
                        n_end = n_seg.get('end', 0.0)
                        n_txt = n_seg.get('text', '').strip()
                        double_window_text += f"[{n_start:.1f}-{n_end:.1f}] {n_spk}: {n_txt}\n"
                    logger.info(
                        f"[ROM Service] Window {i+1} action extraction using 2× window "
                        f"(current + window {i+2})"
                    )

                # Build 2× video context for action extraction
                double_video_context = video_context
                if has_video_ocr and i + 1 < len(windows):
                    next_win = windows[i + 1]
                    nw_start = round(next_win[0].get('start', 0.0), 2) if next_win else 0.0
                    nw_end = round(next_win[-1].get('end', 0.0), 2) if next_win else 0.0
                    next_blocks = get_overlapping_ocr_blocks(video_transcript, nw_start, nw_end)
                    if next_blocks:
                        next_lines = []
                        for b in next_blocks:
                            ts = f"{_format_time_hhmm(b['start'])} -> {_format_time_hhmm(b['end'])}"
                            next_lines.append(f"[{ts}]\n{b['text']}")
                        if double_video_context:
                            double_video_context += "\n\n" + "\n\n".join(next_lines)
                        else:
                            double_video_context = "\n\n".join(next_lines)

                def _call_discussion():
                    return provider.extract_rom_discussion_points(
                        window_text, prev_window_text or None, video_context=video_context,
                        skip_action_items=True, separate_action_extraction=True
                    )

                def _call_actions():
                    return provider.extract_rom_action_points(
                        double_window_text, prev_window_text or None, video_context=double_video_context
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
                    try:
                        action_result = fut_acts.result()
                        raw_acts = action_result.get("action_items") or []
                        if not isinstance(raw_acts, list):
                            raw_acts = [raw_acts]
                        normalized_actions = [
                            normalize_action_item(a) for a in raw_acts if normalize_action_item(a)["task"]
                        ]
                    except Exception as act_err:
                        logger.warning(
                            f"[ROM Service] Separate action extraction Call 2 failed for window {i+1}: {act_err}. "
                            "Continuing with empty action_items."
                        )
                        normalized_actions = []

                # Map action items to discussion points by keyword/topic overlap
                for p in points:
                    p["action_items"] = []
                    p.setdefault("action_owner", None)

                if points and normalized_actions:
                    for act in normalized_actions:
                        act_task = (act.get("task") or "").lower()
                        act_words = set(re.findall(r'\w{3,}', act_task))
                        best_pt = points[0]
                        best_score = 0
                        for pt in points:
                            pt_text = (pt.get("discussion_point") or "").lower()
                            pt_words = set(re.findall(r'\w{3,}', pt_text))
                            common = len(act_words.intersection(pt_words))
                            if common > best_score:
                                best_score = common
                                best_pt = pt
                        best_pt["action_items"].append(act)

                # Derive and normalize action_owner for every point
                for p in points:
                    raw_owner = p.get("action_owner") or p.get("action_owners") or p.get("owner") or p.get("assignee")
                    p["action_owner"] = normalize_action_owner(
                        raw_owner,
                        p.get("discussion_point", ""),
                        p.get("action_items", [])
                    )

                logger.info(
                    f"[ROM Service] Window {i+1} separate action extraction (2× window): "
                    f"{len(normalized_actions)} action item(s) mapped across {len(points)} discussion point(s)."
                )

            else:
                # ── Default embedded-action path (separate_action_extraction=False) ─────────
                # Uses ROM_DISCUSSION_EMBEDDED_PROMPT: actions are woven into discussion_point text.
                # action_owner is extracted directly by LLM and normalized.
                result = provider.extract_rom_discussion_points(
                    window_text, prev_window_text or None, video_context=video_context,
                    separate_action_extraction=False
                )
                points = result.get("discussion_points", [])
                parse_error = bool(result.get("parse_error"))
                error_reason = result.get("error_reason", "")

                for p in points:
                    raw_owner = p.get("action_owner") or p.get("action_owners") or p.get("owner") or p.get("assignee")
                    p["action_owner"] = normalize_action_owner(
                        raw_owner,
                        p.get("discussion_point", ""),
                        p.get("action_items", [])
                    )
                    if not isinstance(p.get("action_items"), list):
                        p["action_items"] = []

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
    ) -> Dict:
        """
        Stage 1 Re-run Window: Re-extracts discussion points for a single window
        using the window's transcript, previous output, and user correction feedback.
        """
        from services.ai_provider import get_provider
        from services.video_processing_service import get_overlapping_ocr_blocks

        if not transcript:
            return {"discussion_points": [], "window_index": window_index, "transcript_text": ""}

        provider = get_provider()

        windows = []
        current_window = []
        current_start_time = transcript[0].get('start', 0.0)
        window_seconds = window_minutes * 60.0

        for segment in transcript:
            seg_start = segment.get('start', 0.0)
            if current_window and (seg_start - current_start_time) >= window_seconds:
                windows.append(current_window)
                current_window = [segment]
                current_start_time = seg_start
            else:
                current_window.append(segment)

        if current_window:
            windows.append(current_window)

        # Target index is 1-based (window_index), map to 0-based
        target_idx = max(0, min(len(windows) - 1, window_index - 1))
        target_window = windows[target_idx] if windows else []

        w_start = round(target_window[0].get('start', 0.0), 2) if target_window else 0.0
        w_end = round(target_window[-1].get('end', 0.0), 2) if target_window else 0.0

        window_text = ""
        for seg in target_window:
            speaker = seg.get('speaker_label') or seg.get('speaker') or 'Unknown'
            start = seg.get('start', 0.0)
            end = seg.get('end', 0.0)
            text = seg.get('text', '').strip()
            window_text += f"[{start:.1f}-{end:.1f}] {speaker}: {text}\n"

        video_context = ""
        if video_transcript:
            relevant_blocks = get_overlapping_ocr_blocks(video_transcript, w_start, w_end)
            if relevant_blocks:
                lines = []
                for b in relevant_blocks:
                    ts = f"{_format_time_hhmm(b['start'])} -> {_format_time_hhmm(b['end'])}"
                    lines.append(f"[{ts}]\n{b['text']}")
                video_context = "\n\n".join(lines)

        prev_window_text = ""
        if target_idx > 0 and target_idx - 1 < len(windows):
            prev_win = windows[target_idx - 1]
            p_lines = []
            for p_seg in prev_win:
                p_spk = p_seg.get('speaker_label') or p_seg.get('speaker') or 'Unknown'
                p_start = p_seg.get('start', 0.0)
                p_end = p_seg.get('end', 0.0)
                p_txt = p_seg.get('text', '').strip()
                p_lines.append(f"[{p_start:.1f}-{p_end:.1f}] {p_spk}: {p_txt}")
            prev_window_text = "PREVIOUS TRANSCRIPT WINDOW:\n" + "\n".join(p_lines)

        existing_json = json.dumps(existing_points, ensure_ascii=False) if existing_points else ""

        res = provider.extract_rom_discussion_points_with_feedback(
            window_text=window_text,
            existing_points_json=existing_json,
            user_feedback=user_feedback,
            previous_points_json=prev_window_text or None,
            video_context=video_context,
        )

        raw_points = res.get("discussion_points", [])
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

        if not points or not user_id or not recording_id:
            if return_details:
                return {
                    "total_found": len(points) if points else 0,
                    "already_indexed": 0,
                    "newly_indexed": 0,
                    "skipped_failed": len(points) if points else 0,
                }
            return 0

        user_id = _validate_user_id(user_id)
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
                "source": "stage2_point",
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
                    speakers = ep.get("speakers") or list(
                        dict.fromkeys(
                            spk for m in matched_points for spk in (m.get("speakers") or []) if spk
                        )
                    )

                    ep["original_point_ids"] = orig_ids
                    ep["original_point_id"] = orig_ids[0] if orig_ids else "N/A"
                    ep["timeline_start"] = t_start
                    ep["timeline_end"] = t_end
                    ep["speakers"] = speakers

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

                    # Carry over action items / action_owner depending on mode
                    if separate_action_extraction:
                        inherited_actions = []
                        for m in matched_points:
                            for a in (m.get("action_items") or []):
                                norm = normalize_stage2_action_item(a)
                                if norm["task"]:
                                    inherited_actions.append(norm)
                        ep.setdefault("action_items", inherited_actions)
                        ep.setdefault("action_owner", None)
                    else:
                        # Default (embedded) mode: no action_items list; carry action_owner forward.
                        ep["action_items"] = []
                        # Prefer action_owner from LLM output, then from the best matched Stage 1 point.
                        if not ep.get("action_owner"):
                            for m in matched_points:
                                if m.get("action_owner"):
                                    ep["action_owner"] = m["action_owner"]
                                    break
                        if not ep.get("action_owner"):
                            ep["action_owner"] = None

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
                        speakers = ep.get("speakers") or list(
                            dict.fromkeys(
                                spk for m in matched_points for spk in (m.get("speakers") or []) if spk
                            )
                        )

                        ep["original_point_ids"] = orig_ids
                        ep["original_point_id"] = orig_ids[0] if orig_ids else "N/A"
                        ep["timeline_start"] = t_start
                        ep["timeline_end"] = t_end
                        ep["speakers"] = speakers

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

                        if separate_action_extraction:
                            # Separate mode: normalize action_items[] from LLM output, or inherit from matched points
                            raw_acts = ep.get("action_items", [])
                            if not isinstance(raw_acts, list):
                                raw_acts = [raw_acts] if raw_acts else []
                            acts = [
                                normalize_stage2_action_item(a) for a in raw_acts if normalize_stage2_action_item(a)["task"]
                            ]
                            if not acts:
                                for m in matched_points:
                                    for a in (m.get("action_items") or []):
                                        norm = normalize_stage2_action_item(a)
                                        if norm["task"]:
                                            acts.append(norm)
                            ep["action_items"] = acts
                            ep.setdefault("action_owner", None)
                        else:
                            # Default (embedded) mode: no action_items list; carry action_owner forward.
                            ep["action_items"] = []
                            # Prefer action_owner from LLM output, then from matched Stage 1 points.
                            if not ep.get("action_owner"):
                                for m in matched_points:
                                    if m.get("action_owner"):
                                        ep["action_owner"] = m["action_owner"]
                                        break
                            if not ep.get("action_owner"):
                                ep["action_owner"] = None


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
                                {
                                    "id": str(uuid.uuid4()),
                                    "polished_text": p.get("discussion_point", ""),
                                    "timeline_start": p.get("timeline_start", 0.0),
                                    "timeline_end": p.get("timeline_end", 0.0),
                                    "speakers": p.get("speakers", []),
                                    "technical_terms": p.get("technical_terms", []),
                                    "action_items": p.get("action_items", []) if separate_action_extraction else [],
                                    "action_owner": p.get("action_owner") if not separate_action_extraction else None,
                                    "dates": [],
                                }
                                for p in item[1]
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
                                    {
                                        "id": str(uuid.uuid4()),
                                        "polished_text": p.get("discussion_point", ""),
                                        "timeline_start": p.get("timeline_start", 0.0),
                                        "timeline_end": p.get("timeline_end", 0.0),
                                        "speakers": p.get("speakers", []),
                                        "technical_terms": p.get("technical_terms", []),
                                        "action_items": p.get("action_items", []) if separate_action_extraction else [],
                                        "action_owner": p.get("action_owner") if not separate_action_extraction else None,
                                        "dates": [],
                                    }
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

                                for r_pt in res_points:
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
                                    r_pt.setdefault("action_items", [])
                                    # Carry action_owner from the first cluster point that has one
                                    if not r_pt.get("action_owner"):
                                        for _cp in cluster_points:
                                            if _cp.get("action_owner"):
                                                r_pt["action_owner"] = _cp["action_owner"]
                                                break
                                    r_pt.setdefault("action_owner", None)
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

                for p in win:
                    pid = p.get("id")
                    if pid in enhanced_by_pid:
                        p["action_items"] = enhanced_by_pid[pid]

            except Exception as e_act:
                logger.error(f"[ROM Service] Separate action point LLM enhancement failed: {e_act}", exc_info=True)

        return polished_points





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
                raw_agendas   = agenda_result.get("agendas", [])
                if raw_agendas:
                    for a in raw_agendas:
                        spk = a.get("speaker") or a.get("presenter")
                        a["speaker"] = spk
                        a["presenter"] = spk
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
            logger.info(f"[ROM S3] Phase 4: Direct LLM agenda assignment for {len(polished_points)} point(s) against {len(agendas)} agenda(s) (batch_size={batch_size})")
            valid_agenda_ids = {a["agenda_id"] for a in agendas}
            agenda_ref_json  = json.dumps(
                [{"agenda_id": a["agenda_id"], "title": a.get("title", ""), "description": a.get("description", "")} for a in agendas],
                ensure_ascii=False
            )
            candidate_results: List[Dict] = []
            sim_matrix_list: List[List[float]] = []
            batch_assignments: List[Dict] = []
            total_points      = len(polished_points)
            total_batches     = math.ceil(total_points / batch_size)

            for batch_num, batch_start in enumerate(range(0, total_points, batch_size), start=1):
                batch_points = polished_points[batch_start: batch_start + batch_size]
                logger.info(f"[ROM S3] Phase 4 batch {batch_num}/{total_batches}: {len(batch_points)} point(s)")

                batch_items = []
                for point in batch_points:
                    pid = point["id"]
                    batch_items.append({
                        "point_id": pid,
                        "enhanced_point": point.get("polished_text") or point.get("text", ""),
                        "timeline": f"{_format_time_hhmm(point.get('timeline_start', 0))} – {_format_time_hhmm(point.get('timeline_end', 0))}",
                        "speakers": point.get("speakers", []),
                    })

                batch_json_str = json.dumps(batch_items, ensure_ascii=False)
                llm_result     = provider.assign_agenda_batch(batch_json_str, agenda_ref_json)

                # Build a map from the LLM response
                llm_map: Dict[str, Dict] = {a.get("point_id", ""): a for a in llm_result.get("assignments", []) if a.get("point_id")}

                for point in batch_points:
                    pid        = point["id"]
                    llm_entry  = llm_map.get(pid, {})

                    assigned    = llm_entry.get("assigned_agenda_id", "")
                    confidence  = llm_entry.get("confidence", "medium")
                    reason      = llm_entry.get("reason", "")
                    is_probable = False

                    # Fallback: if LLM returned unknown/empty ID, assign to first agenda
                    if not assigned or assigned not in valid_agenda_ids:
                        is_probable = True
                        confidence  = "probable"
                        reason      = "Automatically assigned to default agenda (LLM did not return a valid agenda ID)."
                        assigned    = agendas[0]["agenda_id"] if agendas else "A1"

                    batch_assignments.append({
                        "point_id": pid,
                        "assigned_agenda_id": assigned,
                        "confidence": "probable" if is_probable else confidence,
                        "reason": reason,
                        "is_probable": is_probable,
                    })

                logger.info(f"[ROM S3] Phase 4 batch {batch_num}/{total_batches} complete")

            logger.info(f"[ROM S3] Phase 4 complete: {len(batch_assignments)} assignment(s)")

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
                "similarity_matrix": sim_matrix_list,  # kept for backward compat / DOCX download
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
    ) -> Dict:
        """
        Stage 3 â€“ Step 1: Generate agendas only (no point mapping).
        Returns agendas + expanded agenda context chunks.
        Phases 1 & 2 from the original pipeline.
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
                    raw_agendas   = agenda_result.get("agendas", [])
                    if raw_agendas:
                        for a in raw_agendas:
                            spk = a.get("speaker") or a.get("presenter")
                            a["speaker"] = spk
                            a["presenter"] = spk
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

            logger.info(f"[ROM S3-Agenda] Complete: {len(agendas)} agendas ready for use.")
            return {
                "agendas": agendas,
                "expanded_agendas": expanded_agendas,
                "retrieved_context_chunks": retrieved_context_chunks,
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
    ) -> Dict:
        """
        Stage 3 – Step 2: Map discussion points to agendas directly using LLM (Phases 4-5).
        Optionally merges agenda document points into the final ROM.
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
            # ── Phase 4: Batch LLM Agenda Assignment (Direct Full Agenda Mapping) ──
            logger.info(
                f"[ROM S3-Map] Phase 4: Direct LLM agenda assignment for {len(polished_points)} point(s) against "
                f"{len(agendas)} agenda(s) (batch_size={batch_size}, has_order={bool(discussion_order_text)}, "
                f"has_timeline={bool(timeline_guidance_text)})"
            )
            valid_agenda_ids = {a["agenda_id"] for a in agendas}
            agenda_ref_json  = json.dumps(
                [{"agenda_id": a["agenda_id"], "title": a.get("title", ""), "description": a.get("description", "")} for a in agendas],
                ensure_ascii=False
            )
            candidate_results: List[Dict] = []
            sim_matrix_list: List[List[float]] = []
            batch_assignments: List[Dict] = []
            total_points      = len(polished_points)
            total_batches     = math.ceil(total_points / batch_size)

            for batch_num, batch_start in enumerate(range(0, total_points, batch_size), start=1):
                batch_points = polished_points[batch_start: batch_start + batch_size]
                logger.info(f"[ROM S3-Map] Phase 4 batch {batch_num}/{total_batches}: {len(batch_points)} point(s)")

                batch_items = []
                for point in batch_points:
                    pid = point["id"]
                    item_dict = {
                        "point_id": pid,
                        "enhanced_point": point.get("polished_text") or point.get("text", ""),
                        "speakers": point.get("speakers", []),
                    }
                    if timeline_guidance_text:
                        item_dict["timeline"] = f"{_format_time_hhmm(point.get('timeline_start', 0))} – {_format_time_hhmm(point.get('timeline_end', 0))}"
                    batch_items.append(item_dict)

                batch_json_str = json.dumps(batch_items, ensure_ascii=False)
                llm_result     = provider.assign_agenda_batch(
                    batch_json=batch_json_str,
                    agenda_reference_json=agenda_ref_json,
                    discussion_order_text=discussion_order_text,
                    timeline_guidance_text=timeline_guidance_text,
                )

                llm_map: Dict[str, Dict] = {a.get("point_id", ""): a for a in llm_result.get("assignments", []) if a.get("point_id")}

                for point in batch_points:
                    pid        = point["id"]
                    llm_entry  = llm_map.get(pid, {})

                    assigned    = llm_entry.get("assigned_agenda_id", "")
                    confidence  = llm_entry.get("confidence", "medium")
                    reason      = llm_entry.get("reason", "")
                    is_probable = False

                    if not assigned or assigned not in valid_agenda_ids:
                        is_probable = True
                        confidence  = "probable"
                        reason      = "Automatically assigned to default agenda (LLM did not return a valid agenda ID)."
                        assigned    = agendas[0]["agenda_id"] if agendas else "A1"

                    batch_assignments.append({
                        "point_id": pid,
                        "assigned_agenda_id": assigned,
                        "confidence": "probable" if is_probable else confidence,
                        "reason": reason,
                        "is_probable": is_probable,
                    })

                logger.info(f"[ROM S3-Map] Phase 4 batch {batch_num}/{total_batches} complete")

            logger.info(f"[ROM S3-Map] Phase 4 complete: {len(batch_assignments)} assignment(s)")


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
        3. Action Items are extracted depending on mode:
           - separate_action_extraction=True: read directly from each point's action_items list.
           - separate_action_extraction=False (default): call extract_actions_from_enhanced_points()
             to extract embedded actions from polished_text via a final LLM pass.
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

        # ── Robust owner resolution helper ──────────────────────────────────
        def _resolve_point_owner(
            act_owner: Optional[str],
            src_point: Optional[Dict],
            task_text: str = "",
        ) -> str:
            invalid_values = {"none", "n/a", "null", "unassigned", "unknown", "undefined", ""}

            def _clean(val) -> Optional[str]:
                if val is None:
                    return None
                s = str(val).strip()
                if s.lower() in invalid_values:
                    return None
                return s

            # 1. Try explicitly extracted owner
            cleaned = _clean(act_owner)
            if cleaned:
                return cleaned

            if src_point:
                # 2. Try source point's action_owner
                p_owner = _clean(src_point.get("action_owner"))
                if p_owner:
                    return p_owner

                # 3. Try source point's speakers list
                spks = src_point.get("speakers")
                if isinstance(spks, str):
                    spks = [s.strip() for s in spks.split(",") if s.strip()]
                if isinstance(spks, list):
                    valid_spks = [s for s in spks if _clean(s) and str(s).strip().lower() != "for information"]
                    if valid_spks:
                        return ", ".join(valid_spks)

                # 4. Try source point's speaker
                p_spk = _clean(src_point.get("speaker"))
                if p_spk and p_spk.lower() != "for information":
                    return p_spk

            # 5. Check if any meeting participant's name is mentioned in the task text
            if participants and isinstance(participants, list):
                task_lower = (task_text or "").lower()
                matched = [
                    str(p).strip() for p in participants
                    if _clean(p) and str(p).strip().lower() != "for information" and str(p).strip().lower() in task_lower
                ]
                if matched:
                    return ", ".join(matched)
                
                valid_parts = [str(p).strip() for p in participants if _clean(p) and str(p).strip().lower() != "for information"]
                if len(valid_parts) == 1:
                    return valid_parts[0]

            return "Unassigned"

        if separate_action_extraction:
            # ── Separate-extraction mode: read action_items[] directly from each polished point ──
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

                    displayed_task = format_action_point_display_text(act_dict)
                    raw_assignee = act_dict.get("assignee") or act_dict.get("owner")
                    owner = _resolve_point_owner(raw_assignee, p, task_text=displayed_task or act_dict["task"])
                    deadline = act_dict.get("deadline") or d_fallback or "ASAP"

                    action_items.append({
                        "task": displayed_task or act_dict["task"],
                        "item": displayed_task or act_dict["task"],
                        "description": displayed_task or act_dict["task"],
                        "owner": owner,
                        "deadline": deadline,
                        "status": "open",
                        "raw_json": act_dict,
                    })
        else:
            # ── Default (embedded) mode: run LLM extraction pass over polished_text in chunks ──
            logger.info(f"[ROM Service] MoM generation: running extract_actions_from_enhanced_points (chunk_size={action_chunk_size or 'default'})")
            provider = get_provider()
            try:
                raw_extracted = provider.extract_actions_from_enhanced_points(polished_points, chunk_size=action_chunk_size)
            except Exception as _ex:
                logger.warning(f"[ROM Service] MoM action extraction pass failed ({_ex}). Continuing with no action items.")
                raw_extracted = []
            finally:
                provider.unload_model()

            # Build a lookup: id -> point (for fallback action_owner and speakers)
            point_by_id = {p.get("id"): p for p in polished_points if p.get("id")}

            for act in raw_extracted:
                task = str(act.get("task") or act.get("item") or act.get("description") or "").strip()
                if not task:
                    continue

                src_id = act.get("source_point_id") or act.get("id")
                src_point = point_by_id.get(src_id) if src_id else None
                raw_owner = act.get("owner") or act.get("assignee") or act.get("person") or act.get("action_owner")
                owner_str = _resolve_point_owner(raw_owner, src_point, task_text=task)

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
                    "deadline": deadline,
                    "expected_outcome": expected_outcome,
                    "status": "open",
                    "raw_json": act,
                })

            logger.info(f"[ROM Service] MoM action extraction pass complete: {len(action_items)} action item(s)")

        # ── Fallback: if no action items were extracted via LLM pass, check for pre-extracted action_items on points ──
        if not action_items and not separate_action_extraction:
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

                    displayed_task = format_action_point_display_text(act_dict)
                    raw_assignee = act_dict.get("assignee") or act_dict.get("owner")
                    owner = _resolve_point_owner(raw_assignee, p, task_text=displayed_task or act_dict["task"])
                    deadline = act_dict.get("deadline") or d_fallback or "ASAP"

                    action_items.append({
                        "task": displayed_task or act_dict["task"],
                        "item": displayed_task or act_dict["task"],
                        "description": displayed_task or act_dict["task"],
                        "owner": owner,
                        "deadline": deadline,
                        "status": "open",
                        "raw_json": act_dict,
                    })

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

        provider = get_provider()
        try:
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
            provider.unload_model()
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

        provider = get_provider()
        result_data = {}
        try:
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
            provider.unload_model()
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
                raw = provider.query(prompt, max_tokens=4096, temperature=0.15)
            else:
                raw = provider._infer(prompt, max_new_tokens=4096)

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
                raw = provider.query(prompt, max_tokens=8192, temperature=0.15)
            else:
                raw = provider._infer(prompt, max_new_tokens=8192)

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
                    # Try repairing trailing commas: [ "a", "b", ]
                    try:
                        fixed = re.sub(r",\s*([\]\}])", r"\1", json_substr)
                        data = _json.loads(fixed)
                        if isinstance(data, list):
                            pts = [str(x).strip() for x in data if str(x).strip()]
                            if pts:
                                return pts
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
                    if hasattr(provider, "query"):
                        raw = provider.query(prompt, max_tokens=4096, temperature=0.2)
                    else:
                        raw = provider._infer(prompt, max_new_tokens=4096)

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

        from services.ai_provider import get_provider
        provider = get_provider()
        try:
            if hasattr(provider, "query"):
                raw = provider.query(full_prompt, max_tokens=4096, temperature=0.3)
            else:
                raw = provider._infer(full_prompt, max_new_tokens=4096)

            # Parse JSON response
            raw_str = str(raw).strip()

            # Remove <think>...</think> blocks if present
            import re
            raw_str = re.sub(r'<think>.*?</think>', '', raw_str, flags=re.DOTALL).strip()

            # Remove markdown fences if present
            if raw_str.startswith("```"):
                lines = raw_str.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw_str = "\n".join(lines).strip()

            # Try to find JSON object
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
            provider.unload_model()
            gc.collect()

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