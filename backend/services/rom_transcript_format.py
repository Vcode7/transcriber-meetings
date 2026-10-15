"""
Speaker-aware transcript formatting for ROM Stage 1.

Merges consecutive same-speaker transcript segments into turns with stable,
recording-global IDs (T1, T2, ...), flags turns whose speaker attribution is
uncertain (overlap, low speaker confidence, refined label), splits turns into
time windows, and renders windows for LLM prompts so extracted action items
can cite the exact turns they came from.
"""
import re
from typing import Any, Dict, List, Optional

# Turns whose speaker_confidence falls below this are flagged as uncertain.
UNCERTAIN_CONFIDENCE_THRESHOLD = 0.6
# Consecutive same-speaker segments are merged only when the pause between them
# is short and the merged turn stays reasonably short (keeps evidence precise).
MAX_MERGE_GAP_SECONDS = 2.0
MAX_TURN_SECONDS = 45.0

NON_PERSON_LABELS = {"", "unknown", "null", "none", "n/a", "for information", "from document", "—", "-"}
_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "shri", "smt"}

_REQUEST_TAIL_RE = re.compile(
    r"(\?\s*$)|\b(can you|could you|would you|will you|please|you need to|you have to|"
    r"make sure you|i want you to|take care of)\b",
    re.IGNORECASE,
)


def segment_speaker(seg: Dict[str, Any]) -> str:
    """Resolved speaker label of a transcript segment (never a raw Pyannote ID when a label exists)."""
    return str(seg.get("speaker_label") or seg.get("speaker") or "Unknown").strip() or "Unknown"


def format_timestamp(seconds: float) -> str:
    total = int(round(seconds or 0.0))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hrs:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _segment_is_uncertain(seg: Dict[str, Any]) -> bool:
    conf = seg.get("speaker_confidence")
    low_conf = isinstance(conf, (int, float)) and conf < UNCERTAIN_CONFIDENCE_THRESHOLD
    return bool(seg.get("speaker_refined")) or low_conf


def build_turns(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive same-speaker segments into turns with global IDs T1..Tn.

    A segment is merged into the previous turn only when the speaker, the
    uncertainty status and the overlap status match, the pause is short and the
    merged turn stays under MAX_TURN_SECONDS — so uncertainty stays localised.
    """
    turns: List[Dict[str, Any]] = []
    for seg in segments or []:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        speaker = segment_speaker(seg)
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        uncertain = _segment_is_uncertain(seg)
        is_overlap = bool(seg.get("is_overlap"))
        overlap_speakers = sorted({
            str(spk) for region in (seg.get("overlap_regions") or [])
            for spk in (region.get("speakers") or [])
            if spk and str(spk) != speaker
        })

        prev = turns[-1] if turns else None
        if (
            prev is not None
            and prev["speaker"] == speaker
            and prev["uncertain"] == uncertain
            and prev["is_overlap"] == is_overlap
            and start - prev["end"] <= MAX_MERGE_GAP_SECONDS
            and end - prev["start"] <= MAX_TURN_SECONDS
        ):
            prev["end"] = max(prev["end"], end)
            prev["text"] = f"{prev['text']} {text}"
            prev["overlap_speakers"] = sorted(set(prev["overlap_speakers"]) | set(overlap_speakers))
            continue

        turns.append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "text": text,
            "uncertain": uncertain,
            "is_overlap": is_overlap,
            "overlap_speakers": overlap_speakers,
        })

    for idx, turn in enumerate(turns, start=1):
        turn["turn_id"] = f"T{idx}"
    return turns


def _invites_response(turn: Dict[str, Any]) -> bool:
    """True when a turn ends with a question or a request that the next turn likely answers."""
    tail = turn.get("text", "")[-160:]
    return bool(_REQUEST_TAIL_RE.search(tail))


def split_turns_into_windows(turns: List[Dict[str, Any]], window_minutes: float) -> List[List[Dict[str, Any]]]:
    """Group turns into time windows of ``window_minutes``.

    A window is never cut directly after a question/request turn: the very next
    turn (usually the reply or acceptance) is kept in the same window.
    """
    window_seconds = float(window_minutes) * 60.0
    windows: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    window_start = 0.0
    extended = False

    for turn in turns:
        if current and (turn["start"] - window_start) >= window_seconds:
            if not extended and _invites_response(current[-1]):
                current.append(turn)
                extended = True
                continue
            windows.append(current)
            current = [turn]
            window_start = turn["start"]
            extended = False
        else:
            if not current:
                window_start = turn["start"]
            current.append(turn)

    if current:
        windows.append(current)
    return windows


def split_turns_in_half(turns: List[Dict[str, Any]], min_turns: int = 2) -> tuple:
    """Two halves of a span of turns, balanced by amount of speech, each with at least
    ``min_turns`` turns. The cut never falls directly after a question/request turn (its reply
    stays with it). Returns (turns, []) when the span is too short to split."""
    if len(turns) < 2 * min_turns:
        return turns, []
    total = sum(len(t.get("text", "")) for t in turns)
    cut, spoken = len(turns) // 2, 0
    for idx, turn in enumerate(turns):
        spoken += len(turn.get("text", ""))
        if spoken >= total / 2:
            cut = idx + 1
            break
    cut = max(min_turns, min(len(turns) - min_turns, cut))
    if _invites_response(turns[cut - 1]) and cut < len(turns) - min_turns:
        cut += 1
    return turns[:cut], turns[cut:]


def render_turn(turn: Dict[str, Any]) -> str:
    line = (
        f"[{turn['turn_id']} | {format_timestamp(turn['start'])}-{format_timestamp(turn['end'])}] "
        f"{turn['speaker']}: {turn['text']}"
    )
    notes = []
    if turn.get("is_overlap") or turn.get("overlap_speakers"):
        others = ", ".join(turn.get("overlap_speakers") or [])
        notes.append(f"overlap with {others}" if others else "overlapping speech")
    if turn.get("uncertain"):
        notes.append("speaker uncertain")
    if notes:
        line += f" ({'; '.join(notes)})"
    return line


def render_window(turns: List[Dict[str, Any]]) -> str:
    return "\n".join(render_turn(t) for t in turns)


def build_roster(
    transcript: List[Dict[str, Any]],
    speaker_mappings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Participants of the meeting, keyed by the speaker label used in the transcript.

    Each entry: {"label": str, "aliases": [str, ...]}. Aliases include the label,
    old labels that were renamed to it (speaker_mappings keys), mapped names and
    the first name of multi-word human names.
    """
    labels: List[str] = []
    for seg in transcript or []:
        label = segment_speaker(seg)
        if label.lower() in NON_PERSON_LABELS or label in labels:
            continue
        labels.append(label)

    mappings = {}
    for old, new in (speaker_mappings or {}).items():
        old_s, new_s = str(old or "").strip(), str(new or "").strip()
        if old_s and new_s and old_s != new_s:
            mappings[old_s] = new_s

    # The canonical label is always the one in the transcript turn headers; renamed
    # names in either direction are aliases.
    roster = []
    for label in labels:
        aliases = [label]
        aliases.extend(new for old, new in mappings.items() if old == label)
        aliases.extend(old for old, new in mappings.items() if new == label)
        for name in list(aliases):
            words = [w for w in name.split() if w.lower().rstrip(".") not in _HONORIFICS]
            if len(name.split()) > 1 and words and not re.search(r"\d", name) and len(words[0]) > 2:
                aliases.append(words[0])
        roster.append({"label": label, "aliases": list(dict.fromkeys(aliases))})
    return roster


def format_roster_block(roster: List[Dict[str, Any]]) -> str:
    if not roster:
        return ""
    lines = [
        "MEETING PARTICIPANTS (speaker labels used in the transcript turn headers):",
    ]
    for entry in roster:
        others = [a for a in entry["aliases"] if a != entry["label"]]
        suffix = f" (also referred to as: {', '.join(others)})" if others else ""
        lines.append(f"- {entry['label']}{suffix}")
    lines.append(
        "When naming an owner, assigner or speaker, use exactly one of the labels above. "
        "Only use a name that is not in this list when the transcript explicitly names a person "
        "outside the meeting as responsible."
    )
    return "\n".join(lines)
