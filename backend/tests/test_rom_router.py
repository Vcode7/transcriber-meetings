import pytest
from unittest.mock import patch, MagicMock

def test_rom_service_extract_discussion_points():
    from services.rom_service import rom_service

    transcript = [
        {"speaker": "Speaker 1", "start": 0.0, "end": 10.0, "text": "Discussing LCA Mk2 flight controls."},
        {"speaker": "Speaker 2", "start": 10.0, "end": 20.0, "text": "Testing scheduled for next quarter."}
    ]

    mock_provider = MagicMock()
    mock_provider.extract_rom_discussion_points.return_value = {
        "discussion_points": [
            {
                "discussion_point": "LCA Mk2 flight control testing timeline",
                "timeline_start": 0.0,
                "timeline_end": 20.0,
                "speakers": ["Speaker 1", "Speaker 2"],
                "technical_terms": ["LCA Mk2"],
                "decisions": [],
                "action_items": []
            }
        ]
    }

    with patch("services.ai_provider.get_provider", return_value=mock_provider):
        res = rom_service.extract_discussion_points(transcript, window_minutes=2.0)
        assert len(res["discussion_points"]) == 1
        assert res["discussion_points"][0]["discussion_point"] == "LCA Mk2 flight control testing timeline"
        assert res["windows_processed"] == 1


def test_rom_service_extract_discussion_points_with_video_transcript():
    from services.rom_service import rom_service

    transcript = [
        {"speaker": "Speaker 1", "start": 0.0, "end": 10.0, "text": "Reviewing system architecture."}
    ]
    video_transcript = [
        {"start": 0.0, "end": 10.0, "text": "Slide Title: System Architecture Diagram"}
    ]

    mock_provider = MagicMock()
    mock_provider.extract_rom_discussion_points.return_value = {
        "discussion_points": [
            {
                "discussion_point": "System Architecture Review",
                "timeline_start": 0.0,
                "timeline_end": 10.0,
                "speakers": ["Speaker 1"]
            }
        ]
    }

    with patch("services.ai_provider.get_provider", return_value=mock_provider):
        res = rom_service.extract_discussion_points(
            transcript,
            window_minutes=2.0,
            video_transcript=video_transcript,
            source_type="video"
        )
        assert len(res["discussion_points"]) == 1
        pt = res["discussion_points"][0]
        assert "video_transcript_context" in pt
        assert "System Architecture Diagram" in pt["video_transcript_context"]

        # Verify provider was called with video_context
        _, kwargs = mock_provider.extract_rom_discussion_points.call_args
        assert "video_context" in kwargs
        assert "System Architecture Diagram" in kwargs["video_context"]

