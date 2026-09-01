"""
Unit tests for Transcript Segment Non-Overlapping Merge & Recovery Sanitization.
"""
import pytest
from services.transcript_normalizer import sanitize_and_merge_segments


def test_no_overlap_clean_merge():
    """Verify that distinct non-overlapping segments are preserved and sorted."""
    existing = [
        {"start": 0.0, "end": 2.5, "text": "Good morning everyone."},
        {"start": 5.0, "end": 8.0, "text": "Let us begin the review."},
    ]
    new_corr = [
        {"start": 3.0, "end": 4.5, "text": "Thank you chair."},
    ]

    result = sanitize_and_merge_segments(existing, new_corr)
    assert len(result) == 3
    assert result[0]["text"] == "Good morning everyone."
    assert result[0]["start"] == 0.0 and result[0]["end"] == 2.5
    assert result[1]["text"] == "Thank you chair."
    assert result[1]["start"] == 3.0 and result[1]["end"] == 4.5
    assert result[2]["text"] == "Let us begin the review."
    assert result[2]["start"] == 5.0 and result[2]["end"] == 8.0

    # Verify synthetic words were created for new_corr
    assert len(result[1]["words"]) == 3
    assert [w["word"] for w in result[1]["words"]] == ["Thank", "you", "chair."]


def test_partial_overlap_clamped_strictly():
    """Verify that partial overlaps between adjacent segments are clamped cleanly."""
    existing = [
        {"start": 0.0, "end": 5.0, "text": "First primary segment here."},
        {"start": 7.0, "end": 12.0, "text": "Third segment following."},
    ]
    # New segment starts at 4.0 (overlaps with First at 5.0) and ends at 7.5 (overlaps with Third at 7.0)
    new_corr = [
        {"start": 4.0, "end": 7.0, "text": "Second recovered speech segment."},
    ]

    result = sanitize_and_merge_segments(existing, new_corr)
    assert len(result) == 3

    # Check that all adjacent boundaries strictly satisfy seg[i].end <= seg[i+1].start
    for i in range(len(result) - 1):
        assert result[i]["end"] <= result[i + 1]["start"], f"Overlap between seg {i} and {i+1}"

    assert result[0]["end"] == 4.0
    assert result[1]["start"] == 4.0
    assert result[1]["end"] == 7.0
    assert result[2]["start"] == 7.0


def test_duplicate_deduplication():
    """Verify that duplicate segments (identical/similar text and overlapping time) are pruned."""
    existing = [
        {"start": 10.0, "end": 15.0, "text": "We need to discuss the budget.", "words": [{"word": "budget", "start": 13.0, "end": 14.0}]},
    ]
    duplicate = [
        {"start": 10.2, "end": 14.9, "text": "we need to discuss the budget"},
    ]

    result = sanitize_and_merge_segments(existing, duplicate)
    assert len(result) == 1
    assert result[0]["start"] == 10.0
    assert result[0]["end"] == 15.0
    # Retained word info
    assert len(result[0]["words"]) >= 1


def test_engulfed_distinct_segment():
    """Verify that a distinct segment inside a long silence interval of another segment partitions cleanly."""
    existing = [
        {"start": 0.0, "end": 10.0, "text": "Opening long block"},
    ]
    new_corr = [
        {"start": 4.0, "end": 6.0, "text": "Quiet interjection"},
    ]

    result = sanitize_and_merge_segments(existing, new_corr)
    assert len(result) == 2
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 4.0
    assert result[1]["start"] == 4.0
    assert result[1]["end"] == 6.0
    assert result[0]["end"] <= result[1]["start"]


def test_invalid_and_empty_segments():
    """Verify that segments with zero/negative duration or empty text are safely discarded."""
    bad_segments = [
        {"start": 5.0, "end": 5.0, "text": "Zero duration"},
        {"start": 8.0, "end": 6.0, "text": "Negative duration"},
        {"start": 1.0, "end": 2.0, "text": "   "},
        {"start": 2.0, "end": 3.0, "text": "Valid speech"},
    ]

    result = sanitize_and_merge_segments(bad_segments)
    assert len(result) == 1
    assert result[0]["text"] == "Valid speech"
    assert result[0]["start"] == 2.0
    assert result[0]["end"] == 3.0


def test_synthetic_words_distribution():
    """Verify that synthetic words have evenly distributed, strictly monotonic non-overlapping timestamps."""
    seg = [{"start": 10.0, "end": 12.0, "text": "One two three four"}]
    result = sanitize_and_merge_segments(seg)

    words = result[0]["words"]
    assert len(words) == 4
    for i in range(len(words) - 1):
        assert words[i]["end"] <= words[i + 1]["start"]
        assert words[i]["start"] >= 10.0
        assert words[i]["end"] <= 12.0
    assert words[-1]["end"] <= 12.0


def test_complex_multi_gap_and_heavy_overlap_recovery():
    """Verify multiple recovered segments in gaps and heavily overlapping boundaries."""
    primary = [
        {"start": 0.0, "end": 3.0, "text": "First primary utterance."},
        {"start": 6.0, "end": 9.0, "text": "Second primary utterance."},
        {"start": 12.0, "end": 15.0, "text": "Third primary utterance."},
    ]
    recovered = [
        {"start": 2.5, "end": 5.5, "text": "Recovered speech between first and second."},
        {"start": 9.5, "end": 11.5, "text": "Clean gap recovery between second and third."},
        {"start": 14.5, "end": 18.0, "text": "Trailing recovered speech extending past end."},
    ]

    result = sanitize_and_merge_segments(primary, recovered)
    assert len(result) >= 4

    # Strict invariant: no overlap anywhere
    for i in range(len(result) - 1):
        assert result[i]["end"] <= result[i + 1]["start"]
        assert result[i]["end"] > result[i]["start"]
        for w in result[i].get("words", []):
            assert w["start"] >= result[i]["start"]
            assert w["end"] <= result[i]["end"]
            assert w["end"] >= w["start"]

    assert result[0]["start"] == 0.0
    assert result[-1]["end"] == 18.0


def test_duplicate_recovery_with_slightly_different_casing():
    """Verify deduplication when recovery Whisper returns identical words with different casing/punctuation."""
    primary = [
        {"start": 1.0, "end": 4.0, "text": "Hello world, welcome to our meeting."},
    ]
    recovered = [
        {"start": 1.1, "end": 3.9, "text": "hello world welcome to our meeting"},
    ]

    result = sanitize_and_merge_segments(primary, recovered)
    assert len(result) == 1
    assert result[0]["start"] == 1.0
    assert result[0]["end"] == 4.0

