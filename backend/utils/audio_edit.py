"""
Audio edit timeline — pure helpers shared by the trim/cut feature.

An edit is described by *keep ranges*: sorted, non-overlapping ``(start, end)``
spans of the source timeline (seconds) that survive the edit.  The edited
audio is those spans concatenated in order.  Every consumer derives its
behaviour from the same keep ranges so that the cut audio, the transcript and
the video OCR timeline always agree:

  - ``utils.audio_utils.process_audio_edit`` renders them with ffmpeg,
  - ``remap_segments`` moves source-timeline transcript segments onto the
    edited timeline (chunked live recordings are transcribed before the user
    trims them),
  - ``remap_time_blocks`` does the same for video OCR blocks, which are always
    extracted from the untrimmed video.

No I/O happens here, so everything is unit-testable without ffmpeg.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

KeepRanges = List[Tuple[float, float]]

MIN_RANGE_SEC = 0.05     # kept slivers shorter than this are dropped
MIN_CUT_SEC = 0.01       # cuts shorter than this are ignored
EDGE_EPS_SEC = 0.01      # tolerance for "touches the start/end of the file"

# The browser trimmer works on the duration the browser reported for the file.
# When that differs from the decoded duration by more than this, the browser's
# timeline was an estimate (typically VBR MP3 without a Xing/VBRI header) and
# the edit points are rescaled proportionally.  Small differences are ordinary
# container / decoder padding and are only clamped.
SCALE_MIN_ABS_SEC = 1.5
SCALE_MIN_REL = 0.03


def build_keep_ranges(
    total_sec: float,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    cut_start_sec: Optional[float] = None,
    cut_end_sec: Optional[float] = None,
) -> KeepRanges:
    """
    Turn trim boundaries plus an optional middle cut into keep ranges.

    ``start_sec`` / ``end_sec`` default to the start / end of the file, so a
    caller that only moved one boundary never has to know the true duration.
    Raises ValueError when the edit would leave nothing to process.
    """
    total = max(0.0, float(total_sec))
    s = 0.0 if start_sec is None else min(max(0.0, float(start_sec)), total)
    e = total if end_sec is None else min(max(0.0, float(end_sec)), total)
    if e - s < MIN_RANGE_SEC:
        raise ValueError(
            f"Trim range {s:.2f}s–{e:.2f}s is empty (audio is {total:.2f}s long)."
        )

    ranges: KeepRanges = [(s, e)]
    if cut_start_sec is not None and cut_end_sec is not None:
        cs = max(s, float(cut_start_sec))
        ce = min(e, float(cut_end_sec))
        if ce - cs > MIN_CUT_SEC:
            ranges = [(a, b) for a, b in ((s, cs), (ce, e)) if b - a >= MIN_RANGE_SEC]
    if not ranges:
        raise ValueError("The removed section covers the entire selected audio.")
    return ranges


def is_identity(ranges: Sequence[Tuple[float, float]], total_sec: float) -> bool:
    """True when the keep ranges retain the whole file (no edit needed)."""
    return (
        len(ranges) == 1
        and ranges[0][0] <= EDGE_EPS_SEC
        and ranges[0][1] >= total_sec - EDGE_EPS_SEC
    )


def kept_duration(ranges: Sequence[Tuple[float, float]]) -> float:
    return sum(b - a for a, b in ranges)


def _range_offsets(ranges: Sequence[Tuple[float, float]]) -> List[float]:
    """Edited-timeline position at which each keep range begins."""
    offsets, acc = [], 0.0
    for a, b in ranges:
        offsets.append(acc)
        acc += b - a
    return offsets


def range_index(t: float, ranges: Sequence[Tuple[float, float]]) -> Optional[int]:
    """Index of the keep range containing source time ``t`` (None if removed)."""
    for i, (a, b) in enumerate(ranges):
        if a <= t <= b:
            return i
    return None


def map_time(t: float, ranges: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Source time → edited time, or None when ``t`` falls in a removed region."""
    idx = range_index(t, ranges)
    if idx is None:
        return None
    return _range_offsets(ranges)[idx] + (t - ranges[idx][0])


def map_time_clamped(t: float, ranges: Sequence[Tuple[float, float]]) -> float:
    """
    Source time → edited time; a time inside a removed region snaps to where the
    next kept audio begins (or to the end of the edited audio).  Monotonic, so
    ``[map_time_clamped(a), map_time_clamped(b)]`` is exactly the kept part of
    ``[a, b]`` on the edited timeline.
    """
    offsets = _range_offsets(ranges)
    for (a, b), off in zip(ranges, offsets):
        if t < a:
            return off
        if t <= b:
            return off + (t - a)
    return kept_duration(ranges)


def rescale_for_source_duration(
    values: Dict[str, Optional[float]],
    browser_duration_sec: Optional[float],
    reference_duration_sec: float,
) -> Tuple[Dict[str, Optional[float]], float]:
    """
    Rescale browser-timeline edit points onto the decoded timeline when the
    browser's duration was clearly an estimate.  Returns (values, scale).
    """
    if not browser_duration_sec or browser_duration_sec <= 0 or reference_duration_sec <= 0:
        return dict(values), 1.0
    diff = abs(reference_duration_sec - browser_duration_sec)
    if diff <= max(SCALE_MIN_ABS_SEC, SCALE_MIN_REL * reference_duration_sec):
        return dict(values), 1.0
    scale = reference_duration_sec / browser_duration_sec
    return {k: (None if v is None else v * scale) for k, v in values.items()}, scale


# ── Transcript remapping ─────────────────────────────────────────────────────

def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _shift_word(w: Any, a: float, b: float, off: float) -> Any:
    """Move a word dict from keep range [a, b] onto the edited timeline."""
    if not isinstance(w, dict):
        return w
    nw = dict(w)
    for key in ("start", "end"):
        v = _num(nw.get(key))
        if v is not None:
            nw[key] = round(off + min(max(v, a), b) - a, 3)
    return nw


def _segment_from_words(seg: Dict[str, Any], words: List[Any], a: float, b: float, off: float) -> Dict[str, Any]:
    new_words = [_shift_word(w, a, b, off) for w in words]
    timed = [w for w in new_words if isinstance(w, dict) and _num(w.get("start")) is not None]
    new_seg = dict(seg)
    new_seg["words"] = new_words
    new_seg["start"] = timed[0]["start"] if timed else round(off, 3)
    new_seg["end"] = max(_num(w.get("end")) or _num(w.get("start")) for w in timed) if timed else round(off + (b - a), 3)
    tokens = [str(w.get("word", "")).strip() for w in new_words if isinstance(w, dict)]
    new_seg["text"] = " ".join(t for t in tokens if t)
    return new_seg


def remap_segments(segments: List[Dict[str, Any]], ranges: Sequence[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """
    Move transcript segments from the source timeline onto the edited timeline.

    - A segment entirely inside one keep range is shifted as-is (text untouched).
    - A segment that crosses a removed region is split at word level; words in
      the removed region are dropped and each surviving run becomes its own
      segment.  Words without timestamps follow their nearest timed neighbour.
    - A segment without word timestamps is kept (clamped) when its midpoint
      survives, otherwise dropped.
    """
    ranges = list(ranges)
    offsets = _range_offsets(ranges)
    out: List[Dict[str, Any]] = []

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        s, e = _num(seg.get("start")), _num(seg.get("end"))
        if s is None or e is None:
            continue

        # Fast path: wholly inside a single keep range.
        whole = next(
            (i for i, (a, b) in enumerate(ranges) if s >= a - EDGE_EPS_SEC and e <= b + EDGE_EPS_SEC),
            None,
        )
        if whole is not None:
            a, b = ranges[whole]
            off = offsets[whole]
            new_seg = dict(seg)
            new_seg["start"] = round(off + min(max(s, a), b) - a, 3)
            new_seg["end"] = round(off + min(max(e, a), b) - a, 3)
            if isinstance(seg.get("words"), list):
                new_seg["words"] = [_shift_word(w, a, b, off) for w in seg["words"]]
            out.append(new_seg)
            continue

        words = seg.get("words") if isinstance(seg.get("words"), list) else []
        word_idx: List[Optional[int]] = []
        for w in words:
            ws = _num(w.get("start")) if isinstance(w, dict) else None
            we = _num(w.get("end")) if isinstance(w, dict) else None
            if ws is None:
                word_idx.append(-1)  # untimed: resolved below
            else:
                mid = (ws + (we if we is not None else ws)) / 2.0
                word_idx.append(range_index(mid, ranges))

        if words and any(i != -1 for i in word_idx):
            # Untimed words inherit the previous timed word's range (or the next one's).
            last: Optional[int] = next(i for i in word_idx if i != -1)
            for k, i in enumerate(word_idx):
                if i == -1:
                    word_idx[k] = last
                else:
                    last = i
            run: List[Any] = []
            run_idx: Optional[int] = None
            for w, i in zip(words, word_idx):
                if i != run_idx and run and run_idx is not None:
                    a, b = ranges[run_idx]
                    out.append(_segment_from_words(seg, run, a, b, offsets[run_idx]))
                    run = []
                run_idx = i
                if i is not None:
                    run.append(w)
            if run and run_idx is not None:
                a, b = ranges[run_idx]
                out.append(_segment_from_words(seg, run, a, b, offsets[run_idx]))
            continue

        # No usable word timestamps: keep by midpoint, clamped to its range.
        idx = range_index((s + e) / 2.0, ranges)
        if idx is None:
            continue
        a, b = ranges[idx]
        off = offsets[idx]
        new_seg = dict(seg)
        new_seg["start"] = round(off + max(s, a) - a, 3)
        new_seg["end"] = round(off + min(e, b) - a, 3)
        out.append(new_seg)

    return out


def remap_time_blocks(
    blocks: List[Dict[str, Any]],
    ranges: Sequence[Tuple[float, float]],
    min_len_sec: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Move ``{"start", "end", ...}`` blocks (e.g. video OCR) onto the edited
    timeline, dropping blocks that lie entirely in removed regions.  A block
    spanning a cut collapses to one contiguous block on the edited timeline.
    """
    out: List[Dict[str, Any]] = []
    for blk in blocks:
        s, e = _num(blk.get("start")), _num(blk.get("end"))
        if s is None:
            continue
        if e is None or e < s:
            e = s
        ms, me = map_time_clamped(s, ranges), map_time_clamped(e, ranges)
        if me - ms < min_len_sec and map_time(s, ranges) is None:
            continue
        nb = dict(blk)
        nb["start"], nb["end"] = round(ms, 2), round(me, 2)
        out.append(nb)
    return out
