from services.rom_transcript_format import (
    build_roster,
    build_turns,
    format_roster_block,
    render_window,
    split_turns_into_windows,
)


def test_build_turns_merges_consecutive_same_speaker_segments_with_global_ids():
    segments = [
        {"speaker_label": "Speaker 1", "start": 0.0, "end": 3.0, "text": "Good morning."},
        {"speaker_label": "Speaker 1", "start": 3.4, "end": 6.0, "text": "Let's start."},
        {"speaker_label": "Speaker 2", "start": 6.2, "end": 8.0, "text": "Sure."},
        {"speaker": "SPEAKER_00", "speaker_label": "Speaker 1", "start": 8.5, "end": 10.0, "text": "First item."},
    ]
    turns = build_turns(segments)
    assert [t["turn_id"] for t in turns] == ["T1", "T2", "T3"]
    assert turns[0]["text"] == "Good morning. Let's start."
    assert (turns[0]["start"], turns[0]["end"]) == (0.0, 6.0)
    assert turns[2]["speaker"] == "Speaker 1"


def test_build_turns_does_not_merge_across_long_gaps_or_uncertainty():
    segments = [
        {"speaker_label": "Speaker 1", "start": 0.0, "end": 3.0, "text": "One."},
        {"speaker_label": "Speaker 1", "start": 130.0, "end": 133.0, "text": "Two."},
        {"speaker_label": "Speaker 1", "start": 133.2, "end": 134.0, "text": "Three.", "speaker_refined": True},
        {"speaker_label": "Speaker 1", "start": 134.1, "end": 135.0, "text": "Four.", "speaker_confidence": 0.3},
    ]
    turns = build_turns(segments)
    assert len(turns) == 3
    assert turns[1]["uncertain"] is False
    assert turns[2]["uncertain"] is True
    assert turns[2]["text"] == "Three. Four."


def test_render_window_marks_overlap_and_uncertain_turns():
    segments = [
        {"speaker_label": "Speaker 1", "start": 61.0, "end": 64.0, "text": "Can you take this?",
         "is_overlap": True, "overlap_regions": [{"start": 62, "end": 63, "speakers": ["Speaker 1", "Speaker 2"]}]},
        {"speaker_label": "Speaker 2", "start": 64.5, "end": 65.5, "text": "Yes.", "speaker_confidence": 0.5},
    ]
    text = render_window(build_turns(segments))
    lines = text.splitlines()
    assert lines[0] == "[T1 | 01:01-01:04] Speaker 1: Can you take this? (overlap with Speaker 2)"
    assert lines[1] == "[T2 | 01:04-01:06] Speaker 2: Yes. (speaker uncertain)"


def test_split_windows_keeps_reply_with_question_at_boundary():
    segments = [
        {"speaker_label": "Speaker 1", "start": 0.0, "end": 10.0, "text": "Status update."},
        {"speaker_label": "Speaker 1", "start": 100.0, "end": 118.0, "text": "Ravi, can you send the report?"},
        {"speaker_label": "Speaker 2", "start": 121.0, "end": 123.0, "text": "Yes, I'll send it today."},
        {"speaker_label": "Speaker 1", "start": 125.0, "end": 130.0, "text": "Next topic."},
    ]
    windows = split_turns_into_windows(build_turns(segments), 2.0)
    assert [[t["turn_id"] for t in w] for w in windows] == [["T1", "T2", "T3"], ["T4"]]


def test_split_windows_matches_time_boundaries_without_questions():
    segments = [
        {"speaker_label": "Speaker 1", "start": 0.0, "end": 10.0, "text": "Window 1 speech text."},
        {"speaker_label": "Speaker 2", "start": 130.0, "end": 140.0, "text": "Window 2 speech text."},
        {"speaker_label": "Speaker 1", "start": 260.0, "end": 270.0, "text": "Window 3 speech text."},
    ]
    windows = split_turns_into_windows(build_turns(segments), 2.0)
    assert len(windows) == 3


def test_build_roster_uses_transcript_labels_with_mapping_aliases():
    transcript = [
        {"speaker_label": "Speaker 1", "text": "a"},
        {"speaker_label": "Dr. Meera Rao", "text": "b"},
        {"speaker_label": "Unknown", "text": "c"},
        {"speaker_label": "Speaker 1", "text": "d"},
    ]
    roster = build_roster(transcript, {"Speaker 1": "Vikas Gupta"})
    assert [r["label"] for r in roster] == ["Speaker 1", "Dr. Meera Rao"]
    assert roster[0]["aliases"] == ["Speaker 1", "Vikas Gupta", "Vikas"]
    # Honorifics are not used as first-name aliases
    assert "Dr." not in roster[1]["aliases"] and "Meera" in roster[1]["aliases"]
    block = format_roster_block(roster)
    assert "- Speaker 1 (also referred to as: Vikas Gupta, Vikas)" in block
