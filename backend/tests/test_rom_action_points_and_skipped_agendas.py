import pytest
import asyncio
from unittest.mock import MagicMock, patch
from routers.rom_router import _populate_action_points_for_agendas

@pytest.mark.asyncio
async def test_populate_action_points_reuse_existing():
    """Verify that existing action_items on Stage 2 polished points are reused and formatted."""
    agendas = [
        {
            "agenda_id": "A1",
            "title": "Budget Review",
            "discussion_points": [
                {
                    "id": "pt-1",
                    "text": "Discussed Q3 budget allocations.",
                    "speaker": "Alice",
                }
            ]
        }
    ]
    polished_points = [
        {
            "id": "pt-1",
            "polished_text": "Discussed Q3 budget allocations.",
            "action_items": [
                {
                    "task": "Finalize Q3 budget spreadsheet",
                    "assignee": "Bob",
                    "deadline": "Friday"
                }
            ]
        }
    ]
    stage1_points = []

    res = await _populate_action_points_for_agendas(agendas, polished_points, stage1_points)
    dp = res[0]["discussion_points"][0]
    assert "action_points" in dp
    assert len(dp["action_points"]) == 1
    assert "Finalize Q3 budget spreadsheet" in dp["action_points"][0]["task"]
    assert dp["action_points"][0]["assignee"] == "Bob"
    assert dp["action_points"][0]["deadline"] == "Friday"
    assert dp["action_points"][0]["source_point_id"] == "pt-1"

@pytest.mark.asyncio
async def test_populate_action_points_fallback_extraction():
    """Verify that when no action items exist on points, fallback extraction is invoked."""
    agendas = [
        {
            "agenda_id": "A1",
            "title": "Strategy Discussion",
            "discussion_points": [
                {
                    "id": "pt-2",
                    "text": "Team discussed marketing outreach.",
                    "speaker": "Charlie",
                }
            ]
        }
    ]
    polished_points = [
        {
            "id": "pt-2",
            "polished_text": "Team discussed marketing outreach.",
            "action_items": []  # Empty!
        }
    ]
    stage1_points = []

    mock_extracted = [
        {
            "task": "Launch email campaign",
            "owner": "Diana",
            "deadline": "Next week",
            "source_point_id": "pt-2",
        }
    ]

    mock_provider = MagicMock()
    mock_provider.extract_actions_from_enhanced_points.return_value = mock_extracted

    with patch("services.ai_provider.get_provider", return_value=mock_provider):
        res = await _populate_action_points_for_agendas(agendas, polished_points, stage1_points)

    dp = res[0]["discussion_points"][0]
    assert len(dp["action_points"]) == 1
    assert "Launch email campaign" in dp["action_points"][0]["task"]
    assert dp["action_points"][0]["assignee"] == "Diana"
    assert dp["action_points"][0]["deadline"] == "Next week"

@pytest.mark.asyncio
async def test_populate_action_points_skipped_agenda_ignored():
    """Verify that skipped agendas are not modified during action points population."""
    agendas = [
        {
            "agenda_id": "A1",
            "title": "Deferred Topic",
            "skipped": True,
            "skip_note": "Postponed to next quarter",
            "discussion_points": []
        }
    ]
    polished_points = [{"id": "pt-1", "polished_text": "Sample"}]
    res = await _populate_action_points_for_agendas(agendas, polished_points, [])
    assert res[0]["skipped"] is True
    assert res[0]["skip_note"] == "Postponed to next quarter"
    assert res[0]["discussion_points"] == []
