import pytest
import copy
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException
from routers.rom_router import create_stage3_agenda, delete_stage3_agenda, CreateAgendaItemRequest


@pytest.mark.asyncio
@patch("routers.rom_router._save_rom_data", new_callable=AsyncMock)
@patch("routers.rom_router._get_rom_data", new_callable=AsyncMock)
@patch("services.rag_pipeline._save_parsed_agenda")
async def test_create_stage3_agenda_flow(mock_save_parsed, mock_get_rom, mock_save_rom):
    initial_data = {
        "stage3": {
            "agendas": [
                {"agenda_id": "A1", "title": "Existing Agenda", "description": "Existing"}
            ]
        },
        "final_rom": {
            "agendas": [
                {"agenda_id": "A1", "title": "Existing Agenda", "discussion_points": [{"id": "p1", "text": "pt"}]}
            ]
        },
        "final_rom_versions": {
            "long": {
                "agendas": [
                    {"agenda_id": "A1", "title": "Existing Agenda", "discussion_points": [{"id": "p1", "text": "pt"}]}
                ]
            }
        }
    }
    mock_get_rom.return_value = copy.deepcopy(initial_data)

    req = CreateAgendaItemRequest(
        title="New Topic",
        description="Discuss future plans",
        presenter="Bob"
    )

    res = await create_stage3_agenda(
        recording_id="rec-test",
        req=req,
        current_user={"sub": "user-123"},
        db=MagicMock()
    )

    assert res["status"] == "success"
    assert res["agenda"]["agenda_id"] == "A2"
    assert res["agenda"]["title"] == "New Topic"

    # Verify stage3 agendas
    saved_data = mock_save_rom.call_args[0][2]
    stage3_ids = [a["agenda_id"] for a in saved_data["stage3"]["agendas"]]
    assert stage3_ids == ["A1", "A2"]

    # Verify final_rom agendas has A2 with 0 points
    final_ids = [a["agenda_id"] for a in saved_data["final_rom"]["agendas"]]
    assert final_ids == ["A1", "A2"]
    a2_final = next(a for a in saved_data["final_rom"]["agendas"] if a["agenda_id"] == "A2")
    assert a2_final["discussion_points"] == []

    # Verify final_rom_versions has A2
    assert "A2" in [a["agenda_id"] for a in saved_data["final_rom_versions"]["long"]["agendas"]]


@pytest.mark.asyncio
@patch("routers.rom_router._save_rom_data", new_callable=AsyncMock)
@patch("routers.rom_router._get_rom_data", new_callable=AsyncMock)
@patch("services.rag_pipeline._save_parsed_agenda")
async def test_delete_stage3_agenda_flow(mock_save_parsed, mock_get_rom, mock_save_rom):
    initial_data = {
        "stage3": {
            "agendas": [
                {"agenda_id": "A1", "title": "General Discussion"},
                {"agenda_id": "A2", "title": "To Delete"},
            ],
            "point_mappings": {"p2": "A2"},
            "agenda_groups": {"A2": ["p2"], "A1": []}
        },
        "final_rom": {
            "agendas": [
                {"agenda_id": "A1", "title": "General Discussion", "discussion_points": []},
                {"agenda_id": "A2", "title": "To Delete", "discussion_points": [{"id": "p2", "text": "point to move"}]}
            ]
        },
        "final_rom_versions": {
            "long": {
                "agendas": [
                    {"agenda_id": "A1", "title": "General Discussion", "discussion_points": []},
                    {"agenda_id": "A2", "title": "To Delete", "discussion_points": [{"id": "p2", "text": "point to move"}]}
                ]
            }
        }
    }
    mock_get_rom.return_value = copy.deepcopy(initial_data)

    res = await delete_stage3_agenda(
        recording_id="rec-test",
        agenda_id="A2",
        current_user={"sub": "user-123"},
        db=MagicMock()
    )

    assert res["status"] == "success"
    saved_data = mock_save_rom.call_args[0][2]

    # A2 removed from stage3
    assert "A2" not in [a["agenda_id"] for a in saved_data["stage3"]["agendas"]]
    # Point mapped to A1
    assert saved_data["stage3"]["point_mappings"]["p2"] == "A1"

    # A2 removed from final_rom and point moved to A1
    final_agendas = saved_data["final_rom"]["agendas"]
    assert len(final_agendas) == 1
    assert final_agendas[0]["agenda_id"] == "A1"
    assert len(final_agendas[0]["discussion_points"]) == 1
    assert final_agendas[0]["discussion_points"][0]["id"] == "p2"

    # A2 removed from final_rom_versions and point moved to A1
    v_agendas = saved_data["final_rom_versions"]["long"]["agendas"]
    assert len(v_agendas) == 1
    assert v_agendas[0]["agenda_id"] == "A1"
    assert len(v_agendas[0]["discussion_points"]) == 1
