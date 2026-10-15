"""Stage 1 windows: point timestamps come from the turns they cite, and a window whose
extraction fails, is cut off or leaves a large gap is re-extracted as two halves."""
import re
from unittest.mock import MagicMock, patch

from services.rom_service import RomService
from services.rom_transcript_format import build_turns, split_turns_in_half

_LONG = ("the vendor confirmed the hostel renovation schedule and the procurement team reviewed "
         "every line of the budget including the electrical and plumbing work for both wings")
TRANSCRIPT = [
    {"speaker_label": "Alice" if n % 2 == 0 else "Bob", "start": n * 10.0, "end": n * 10.0 + 8.0,
     "text": f"Segment {n} says {_LONG}."}
    for n in range(6)
]
_TURN_RE = re.compile(r"\[(T\d+) \|")


def _turn_ids(window_text):
    return _TURN_RE.findall(window_text)


def _run(respond, transcript=TRANSCRIPT):
    """Stage 1 with a scripted LLM; ``respond(turn_ids_in_window)`` returns the provider result."""
    provider = MagicMock()
    calls = []

    def fake(window_text, _prev, **kwargs):
        ids = _turn_ids(window_text)
        calls.append(ids)
        result = respond(ids)
        if isinstance(result, Exception):
            raise result
        return result

    provider.extract_rom_discussion_points.side_effect = fake
    with patch("services.ai_provider.get_provider", return_value=provider):
        result = RomService().extract_discussion_points(transcript, window_minutes=2.0, parallel_window_processing=1)
    return result, calls


def _point(ids, text="Point"):
    return {"discussion_point": f"{text} about {', '.join(ids)}", "turn_ids": list(ids)}


def test_point_timeline_comes_from_its_turns():
    result, _ = _run(lambda ids: {"discussion_points": [_point(["T3"]), _point(ids)]})
    first, whole = result["discussion_points"]
    assert (first["timeline_start"], first["timeline_end"]) == (20.0, 28.0)     # T3 only
    assert (whole["timeline_start"], whole["timeline_end"]) == (0.0, 58.0)
    assert (first["window_start"], first["window_end"]) == (0.0, 58.0)
    assert result["window_report"][0]["status"] == "ok" and result["coverage"] == 1.0


def test_parse_error_reextracts_the_window_as_two_halves():
    def respond(ids):
        if len(ids) == 6:
            return {"discussion_points": [], "parse_error": True, "error_reason": "Output was not a JSON object"}
        return {"discussion_points": [_point(ids)]}

    result, calls = _run(respond)
    assert [len(c) for c in calls] == [6, 3, 3]
    assert [p["turn_ids"] for p in result["discussion_points"]] == [["T1", "T2", "T3"], ["T4", "T5", "T6"]]
    report = result["window_report"][0]
    assert report["status"] == "split" and report["reason"] == "parse_error"
    assert result["split_windows"] == 1 and result["failed_windows"] == []


def test_truncated_output_is_reextracted_in_parts():
    def respond(ids):
        if len(ids) == 6:  # cut-off JSON: only the first point survived recovery
            return {"discussion_points": [_point(["T1"])], "recovered": True}
        return {"discussion_points": [_point(ids)]}

    result, calls = _run(respond)
    assert [len(c) for c in calls] == [6, 3, 3]
    assert len(result["discussion_points"]) == 2
    assert result["window_report"][0]["reason"] == "truncated output"


def test_coverage_gap_splits_only_when_points_cite_turns():
    # Points cite only T1 of six long turns: a large measurable gap -> split.
    result, calls = _run(lambda ids: {"discussion_points": [_point(ids if len(ids) < 6 else ["T1"])]})
    assert [len(c) for c in calls] == [6, 3, 3]
    assert result["window_report"][0]["reason"] == "coverage gap"
    assert result["coverage"] == 1.0

    # Points cite no turns at all: coverage cannot be judged -> no split.
    _, calls = _run(lambda ids: {"discussion_points": [{"discussion_point": "Summary without turn ids"}]})
    assert [len(c) for c in calls] == [6]


def test_window_that_keeps_failing_is_reported_and_split_depth_is_bounded():
    result, calls = _run(lambda ids: RuntimeError("model crashed"))
    assert len(calls) <= 1 + 2 + 4
    assert result["discussion_points"] == []
    assert [w["status"] for w in result["failed_windows"]] == ["failed"]
    assert result["failed_windows"][0]["window_index"] == 0


def test_default_prompts_ask_for_self_contained_single_subject_points():
    from services import ai_provider as ap
    for prompt in (ap.ROM_DISCUSSION_EXTRACTION_PROMPT, ap.ROM_DISCUSSION_NO_ACTION_ITEMS_PROMPT,
                   ap.ROM_DISCUSSION_EMBEDDED_PROMPT):
        assert "SELF-CONTAINED POINTS" in prompt and "ONE SUBJECT PER DISCUSSION POINT" in prompt
        assert "AVOID DUPLICATION" not in prompt and "minimum number of discussion points" not in prompt
    for prompt in (ap.ROM_ENHANCE_WINDOW_PROMPT, ap.ROM_ENHANCE_ALL_TOGETHER_PROMPT):
        assert "each output point covers ONE subject" in prompt


def test_split_keeps_a_question_with_its_answer():
    turns = build_turns([
        {"speaker_label": "Alice", "start": 0.0, "end": 5.0,
         "text": "We reviewed the budget lines in detail today, including the electrical, plumbing and painting work for both hostel wings."},
        {"speaker_label": "Bob", "start": 6.0, "end": 9.0, "text": "Ravi, can you send the revised quotation?"},
        {"speaker_label": "Ravi", "start": 10.0, "end": 12.0, "text": "Yes, I will send it by Friday."},
        {"speaker_label": "Alice", "start": 13.0, "end": 18.0, "text": "Next we move to the canteen contract renewal."},
        {"speaker_label": "Bob", "start": 19.0, "end": 24.0, "text": "The canteen vendor contract expires in March."},
    ])
    first, second = split_turns_in_half(turns)
    assert [t["turn_id"] for t in first] == ["T1", "T2", "T3"]
    assert [t["turn_id"] for t in second] == ["T4", "T5"]
    assert split_turns_in_half(turns[:3]) == (turns[:3], [])
