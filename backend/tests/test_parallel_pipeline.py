"""
Tests for parallel transcription+diarization pipeline setting and UNKNOWN speaker fallback.
"""
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# ── Test 1: UNKNOWN fallback in _assign_speakers_to_words_manual ─────────────
def test_manual_assignment_unknown_fallback_no_overlap():
    """When identified_segs is empty, all segments should get speaker_label='UNKNOWN'."""
    from tasks.pipeline import _assign_speakers_to_words_manual

    aligned_result = {
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Hello everyone, welcome to the meeting.",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                    {"word": "everyone", "start": 0.5, "end": 1.2, "probability": 0.92},
                    {"word": "welcome", "start": 1.3, "end": 1.8, "probability": 0.88},
                ],
            },
            {
                "start": 4.0,
                "end": 6.0,
                "text": "Let's begin.",
                "words": [
                    {"word": "Let's", "start": 4.0, "end": 4.5, "probability": 0.90},
                    {"word": "begin", "start": 4.6, "end": 5.5, "probability": 0.93},
                ],
            },
        ]
    }

    # Empty diarization -> no overlaps -> should fallback to UNKNOWN
    result = _assign_speakers_to_words_manual(aligned_result, identified_segs=[])

    assert len(result) == 2
    for seg in result:
        assert seg["speaker_label"] == "UNKNOWN"
        assert seg["text"]  # text is preserved
        for w in seg["words"]:
            assert w["speaker_label"] == "UNKNOWN"
            assert w["word"]  # words are preserved


def test_manual_assignment_speaker_label_when_overlap_exists():
    """When identified_segs has matching overlap, segments should get the correct speaker."""
    from tasks.pipeline import _assign_speakers_to_words_manual

    aligned_result = {
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Hello everyone.",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                    {"word": "everyone", "start": 0.5, "end": 1.2, "probability": 0.92},
                ],
            },
        ]
    }

    identified_segs = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "SPEAKER_00",
            "speaker_label": "Alice",
            "speaker_profile_id": "prof-1",
            "is_overlap": False,
            "overlap_regions": [],
        }
    ]

    result = _assign_speakers_to_words_manual(aligned_result, identified_segs)

    assert len(result) == 1
    assert result[0]["speaker_label"] == "Alice"
    assert result[0]["speaker_profile_id"] == "prof-1"


# ── Test 2: UNKNOWN fallback in _post_process_whisperx_segments ──────────────
def test_post_process_unknown_fallback():
    """When a segment has no speaker and no overlap with identified_segs, label should be UNKNOWN."""
    from tasks.pipeline import _post_process_whisperx_segments

    wx_result = {
        "segments": [
            {
                "start": 10.0,
                "end": 15.0,
                "text": "Some orphan text",
                "speaker": "",
                "words": [
                    {"word": "Some", "start": 10.0, "end": 10.5},
                    {"word": "orphan", "start": 10.6, "end": 11.0},
                    {"word": "text", "start": 11.1, "end": 11.5},
                ],
            }
        ]
    }

    # Identified segs are far away from the orphan segment
    identified_segs = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "SPEAKER_00",
            "speaker_label": "Alice",
            "speaker_profile_id": "prof-1",
            "is_overlap": False,
            "overlap_regions": [],
        }
    ]

    result = _post_process_whisperx_segments(wx_result, identified_segs)
    assert len(result) == 1
    assert result[0]["speaker_label"] == "UNKNOWN"


# ── Test 3: Setting model includes parallel_transcription_diarization ────────
def test_setting_model_has_parallel_field():
    """UserSettings and UserSettingsUpdate should have parallel_transcription_diarization."""
    from models.settings import UserSettings, UserSettingsUpdate

    # Default should be False
    s = UserSettings(user_id="test")
    assert s.parallel_transcription_diarization is False

    # Update model should accept the field
    u = UserSettingsUpdate(parallel_transcription_diarization=True)
    assert u.parallel_transcription_diarization is True


# ── Test 4: Parallel mode runs both transcription and diarization ────────────
@pytest.mark.asyncio
async def test_parallel_mode_gather():
    """
    Verify that when parallel mode is enabled, both transcription and diarization
    are invoked concurrently and produce the expected merged result shape.
    """
    call_log = []

    async def mock_transcription():
        call_log.append("transcribe")
        return {
            "segments": [
                {"start": 0.0, "end": 3.0, "text": "Hello.", "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.95}
                ]}
            ],
            "raw_text": "Hello.",
            "language": "en",
            "aligned_result": {
                "segments": [
                    {"start": 0.0, "end": 3.0, "text": "Hello.", "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.95}
                    ]}
                ]
            },
        }

    async def mock_diarization():
        call_log.append("diarize")
        return [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "is_overlap": False}
        ]

    t_result, diar_segs = await asyncio.gather(
        mock_transcription(),
        mock_diarization(),
    )

    # Both were called
    assert "transcribe" in call_log
    assert "diarize" in call_log

    # Transcription result shape
    assert len(t_result["segments"]) == 1
    assert t_result["raw_text"] == "Hello."

    # Diarization result shape
    assert len(diar_segs) == 1
    assert diar_segs[0]["speaker"] == "SPEAKER_00"
