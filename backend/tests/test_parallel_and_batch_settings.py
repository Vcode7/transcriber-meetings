import pytest
from unittest.mock import patch, MagicMock
from models.settings import UserSettings, UserSettingsUpdate

def test_settings_validation_bounds():
    # Test defaults
    s = UserSettings(user_id="test-user")
    assert s.rom_parallel_window_processing == 2
    assert s.whisper_batch_size == 8

    # Test update bounds
    u = UserSettingsUpdate(rom_parallel_window_processing=4, whisper_batch_size=16)
    assert u.rom_parallel_window_processing == 4
    assert u.whisper_batch_size == 16

    # Test invalid values trigger validation errors
    with pytest.raises(ValueError):
        UserSettingsUpdate(rom_parallel_window_processing=10)

    with pytest.raises(ValueError):
        UserSettingsUpdate(whisper_batch_size=64)


def test_stage1_parallel_window_processing_concurrency_and_order():
    from services.rom_service import rom_service

    transcript = [
        {"speaker": "Speaker 1", "start": 0.0, "end": 10.0, "text": "Window 1 speech text."},
        {"speaker": "Speaker 2", "start": 130.0, "end": 140.0, "text": "Window 2 speech text."},
        {"speaker": "Speaker 1", "start": 260.0, "end": 270.0, "text": "Window 3 speech text."}
    ]

    mock_provider = MagicMock()
    def fake_extract(window_text, prev, video_context=None, skip_action_items=False, **kwargs):
        if "Window 1" in window_text:
            return {"discussion_points": [{"discussion_point": "Point 1"}]}
        elif "Window 2" in window_text:
            return {"discussion_points": [{"discussion_point": "Point 2"}]}
        else:
            return {"discussion_points": [{"discussion_point": "Point 3"}]}

    mock_provider.extract_rom_discussion_points.side_effect = fake_extract
    mock_provider.extract_rom_action_points.return_value = {"action_items": []}
    mock_provider.extract_rom_action_items_window.return_value = {"action_extractions": []}

    with patch("services.ai_provider.get_provider", return_value=mock_provider):
        res = rom_service.extract_discussion_points(
            transcript, window_minutes=2.0, parallel_window_processing=3
        )
        points = res["discussion_points"]
        assert len(points) == 3
        # Chronological order must be strictly preserved
        assert points[0]["discussion_point"] == "Point 1"
        assert points[1]["discussion_point"] == "Point 2"
        assert points[2]["discussion_point"] == "Point 3"


def test_stage1_parallel_window_processing_task_failure_resilience():
    from services.rom_service import rom_service

    transcript = [
        {"speaker": "Speaker 1", "start": 0.0, "end": 10.0, "text": "Window 1 speech text."},
        {"speaker": "Speaker 2", "start": 130.0, "end": 140.0, "text": "Window 2 speech text causing error."},
        {"speaker": "Speaker 1", "start": 260.0, "end": 270.0, "text": "Window 3 speech text."}
    ]

    mock_provider = MagicMock()
    def fake_extract(window_text, prev, video_context=None, skip_action_items=False, **kwargs):
        if "causing error" in window_text:
            raise RuntimeError("LLM API Timeout Error")
        elif "Window 1" in window_text:
            return {"discussion_points": [{"discussion_point": "Point 1"}]}
        else:
            return {"discussion_points": [{"discussion_point": "Point 3"}]}

    mock_provider.extract_rom_discussion_points.side_effect = fake_extract
    mock_provider.extract_rom_action_points.return_value = {"action_items": []}
    mock_provider.extract_rom_action_items_window.return_value = {"action_extractions": []}

    with patch("services.ai_provider.get_provider", return_value=mock_provider):
        res = rom_service.extract_discussion_points(
            transcript, window_minutes=2.0, parallel_window_processing=3
        )
        points = res["discussion_points"]
        assert len(points) == 2
        assert points[0]["discussion_point"] == "Point 1"
        assert points[1]["discussion_point"] == "Point 3"


def test_merge_action_items_into_points_basic():
    """_merge_action_items_into_points assigns actions to the best-matching discussion point."""
    from services.rom_service import _merge_action_items_into_points

    discussion_points = [
        {"discussion_point": "Alice asked Bob to prepare the budget report by Friday."},
        {"discussion_point": "Carol discussed the roadmap timeline for Q3 delivery."},
    ]
    action_extractions = [
        {
            "source_point_ref": "Bob to prepare the budget report",
            "action_items": [
                {"assigner": "Alice", "assignee": "Bob", "task": "Prepare budget report", "deadline": "Friday", "evidence": "Alice asked Bob", "conditions": None}
            ],
            "action_owner": "Bob",
        },
        {
            "source_point_ref": "roadmap timeline for Q3",
            "action_items": [
                {"assigner": None, "assignee": "Carol", "task": "Deliver Q3 roadmap", "deadline": "Q3", "evidence": "Carol discussed", "conditions": None}
            ],
            "action_owner": "Carol",
        },
    ]

    result = _merge_action_items_into_points(discussion_points, action_extractions)

    assert len(result[0]["action_items"]) == 1
    assert result[0]["action_items"][0]["task"] == "Prepare budget report"
    assert result[0]["action_owner"] == "Bob"

    assert len(result[1]["action_items"]) == 1
    assert result[1]["action_items"][0]["task"] == "Deliver Q3 roadmap"
    assert result[1]["action_owner"] == "Carol"


def test_merge_action_items_no_actions():
    """Points with no action extractions get empty action_items and null action_owner."""
    from services.rom_service import _merge_action_items_into_points

    pts = [{"discussion_point": "Some discussion with no actions."}]
    result = _merge_action_items_into_points(pts, [])
    assert result[0]["action_items"] == []
    assert result[0]["action_owner"] is None


def test_merge_action_items_fallback_to_last_point():
    """When source_point_ref does not match any point, action falls back to the last point."""
    from services.rom_service import _merge_action_items_into_points

    pts = [{"discussion_point": "Completely unrelated discussion about something else."}]
    extractions = [
        {
            "source_point_ref": "xyz totally unmatched phrase",
            "action_items": [
                {"assigner": "A", "assignee": "B", "task": "Do something", "deadline": None, "evidence": "...", "conditions": None}
            ],
            "action_owner": "B",
        }
    ]
    result = _merge_action_items_into_points(pts, extractions)
    assert len(result[0]["action_items"]) == 1
    assert result[0]["action_owner"] == "B"
