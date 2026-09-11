import pytest
from unittest.mock import patch, MagicMock
from services.prompt_service import PROMPT_META, VALID_KEYS, get_prompt_sync
from services.rom_service import RomService


def test_rom_version_prompts_registered():
    """Verify Short and Medium ROM rewrite prompts are registered."""
    keys = {m["key"] for m in PROMPT_META}
    assert "rom_version_short" in keys
    assert "rom_version_medium" in keys

    assert "rom_version_short" in VALID_KEYS
    assert "rom_version_medium" in VALID_KEYS

    p_short = get_prompt_sync("rom_version_short")
    assert p_short and "{agenda_title}" in p_short and "{points_json}" in p_short

    p_med = get_prompt_sync("rom_version_medium")
    assert p_med and "{agenda_title}" in p_med and "{points_json}" in p_med


def test_generate_rom_version_long_preserves_points():
    """Verify Long version keeps existing Stage 2 points as default with no unnecessary rewriting."""
    rom_service = RomService()
    final_rom = {
        "meeting_date": "2026-09-01",
        "meeting_time": "10:00 AM",
        "members_present": ["Alice", "Bob"],
        "agendas": [
            {
                "agenda_id": "A1",
                "title": "Roadmap",
                "discussion_points": [
                    {
                        "id": "p1",
                        "text": "Alice presented the Q3 roadmap milestones.",
                        "speaker": "Alice",
                        "action_owner": "Alice",
                    }
                ],
            }
        ],
    }

    result = rom_service.generate_rom_version(final_rom, version="long")
    # Date, Time, Members Present must be retained
    assert result["meeting_date"] == "2026-09-01"
    assert result["meeting_time"] == "10:00 AM"
    assert result["members_present"] == ["Alice", "Bob"]
    # Points unchanged
    assert result["agendas"][0]["discussion_points"][0]["text"] == "Alice presented the Q3 roadmap milestones."


@patch("services.ai_provider.get_provider")
def test_generate_rom_version_short_processes_agenda_wise(mock_get_provider):
    """Verify Short version processes agenda-wise with Point ID, Speaker, Discussion, Action Owner."""
    mock_provider = MagicMock()
    mock_provider.query.return_value = '["Condensed action point: Alice to complete roadmap by Friday."]'
    mock_get_provider.return_value = mock_provider

    rom_service = RomService()
    final_rom = {
        "date": "2026-09-01",
        "time": "14:00",
        "members_present": ["Alice", "Bob", "Charlie"],
        "agendas": [
            {
                "agenda_id": "A1",
                "title": "Sprint Planning",
                "discussion_points": [
                    {
                        "id": "p1",
                        "text": "Alice discussed feature A timeline with the team.",
                        "speaker": "Alice",
                        "action_owner": None,
                    },
                    {
                        "id": "p2",
                        "text": "Bob assigned Alice to finish the roadmap draft by Friday.",
                        "speaker": "Bob",
                        "action_owner": "Alice",
                    },
                ],
            }
        ],
    }

    result = rom_service.generate_rom_version(final_rom, version="short", writing_rules="Use active voice.")

    # Provider must be called with agenda title and structured fields
    assert mock_provider.query.called
    call_prompt = mock_provider.query.call_args[0][0]
    assert "Sprint Planning" in call_prompt
    assert "Speaker=Alice" in call_prompt
    assert "Action Owner=Alice" in call_prompt
    assert "Use active voice." in call_prompt

    # Header metadata preserved
    assert result["date"] == "2026-09-01"
    assert result["time"] == "14:00"
    assert result["members_present"] == ["Alice", "Bob", "Charlie"]

    # Points replaced with condensed version
    pts = result["agendas"][0]["discussion_points"]
    assert len(pts) == 1
    assert pts[0]["text"] == "Condensed action point: Alice to complete roadmap by Friday."


@patch("services.ai_provider.get_provider")
def test_map_points_to_agendas_unloads_without_error(mock_get_provider):
    """Verify map_points_to_agendas runs and invokes finally block (unload_model, unload_text_embedder) cleanly."""
    mock_provider = MagicMock()
    mock_provider.assign_agenda_batch.return_value = {
        "assignments": [
            {
                "point_id": "p1",
                "assigned_agenda_id": "A1",
                "confidence": "high",
                "reason": "Roadmap planning",
            }
        ]
    }
    mock_get_provider.return_value = mock_provider

    rom_service = RomService()
    polished_points = [
        {
            "id": "p1",
            "polished_text": "Alice discussed roadmap.",
            "speakers": ["Alice"],
            "action_owner": "Alice",
        }
    ]
    agendas = [
        {
            "agenda_id": "A1",
            "title": "Roadmap",
            "description": "Planning roadmap",
        }
    ]

    result = rom_service.map_points_to_agendas(
        polished_points=polished_points,
        agendas=agendas,
        expanded_agendas=[],
        recording_id="rec-123",
        user_id="test_user",
    )

    assert result["batch_assignments"][0]["assigned_agenda_id"] == "A1"
    assert len(result["final_rom_agendas"]) == 1
    assert result["final_rom_agendas"][0]["discussion_points"][0]["id"] == "p1"
    # Verify unload_model was called in finally
    assert mock_provider.unload_model.called


@patch("services.ai_provider.get_provider")
def test_map_points_to_agendas_preserves_zero_point_agendas(mock_get_provider):
    """Verify map_points_to_agendas includes all agendas from the agenda list, including zero-point agendas."""
    mock_provider = MagicMock()
    mock_provider.assign_agenda_batch.return_value = {
        "assignments": [
            {
                "point_id": "p1",
                "assigned_agenda_id": "A1",
                "confidence": "high",
                "reason": "Roadmap planning",
            }
        ]
    }
    mock_get_provider.return_value = mock_provider

    rom_service = RomService()
    polished_points = [
        {
            "id": "p1",
            "polished_text": "Alice discussed roadmap.",
            "speakers": ["Alice"],
            "action_owner": "Alice",
        }
    ]
    agendas = [
        {"agenda_id": "A1", "title": "Roadmap", "description": "Planning roadmap"},
        {"agenda_id": "A2", "title": "Budget Review", "description": "Budget details"},
        {"agenda_id": "A3", "title": "Any Other Business", "description": "Closing"},
    ]

    result = rom_service.map_points_to_agendas(
        polished_points=polished_points,
        agendas=agendas,
        expanded_agendas=[],
        recording_id="rec-123",
        user_id="test_user",
    )

    final_agendas = result["final_rom_agendas"]
    assert len(final_agendas) == 3
    assert [a["agenda_id"] for a in final_agendas] == ["A1", "A2", "A3"]
    assert len(final_agendas[0]["discussion_points"]) == 1
    assert len(final_agendas[1]["discussion_points"]) == 0
    assert len(final_agendas[2]["discussion_points"]) == 0


