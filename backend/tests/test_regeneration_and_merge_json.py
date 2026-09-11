import pytest
from unittest.mock import MagicMock, patch
from services.ai_provider import QwenProvider
from routers.rom_router import (
    GenerateFinalRomRequest,
    generate_final_rom_from_agendas,
    generate_stage1,
    generate_stage2,
    merge_stage2_points,
    MergePointsRequest,
    rom_service,
)

def test_extract_clean_text_value_plain_text():
    provider = QwenProvider.__new__(QwenProvider)
    assert provider._extract_clean_text_value("Direct discussion point text.") == "Direct discussion point text."

def test_extract_clean_text_value_stringified_json():
    provider = QwenProvider.__new__(QwenProvider)
    raw_json = '{"polished_text": "The committee reviewed the Q3 budget.", "speakers": ["Alice"]}'
    extracted = provider._extract_clean_text_value(raw_json)
    assert extracted == "The committee reviewed the Q3 budget."

def test_extract_clean_text_value_nested_json():
    provider = QwenProvider.__new__(QwenProvider)
    nested = '{"polished_text": "{\\"polished_text\\": \\"Nested discussion content.\\"}"}'
    extracted = provider._extract_clean_text_value(nested)
    assert extracted == "Nested discussion content."

def test_extract_clean_text_value_with_think_tags_and_markdown():
    provider = QwenProvider.__new__(QwenProvider)
    raw_with_think = """
    <think>
    Thinking about merging the two points...
    Let's produce the final JSON.
    </think>
    ```json
    {
      "polished_text": "Deployment is scheduled for Friday midnight.",
      "action_items": []
    }
    ```
    """
    extracted = provider._extract_clean_text_value(raw_with_think)
    assert extracted == "Deployment is scheduled for Friday midnight."

def test_extract_clean_text_value_dict_input():
    provider = QwenProvider.__new__(QwenProvider)
    data_dict = {"polished_text": "Point from dictionary."}
    assert provider._extract_clean_text_value(data_dict) == "Point from dictionary."

def test_merge_discussion_points_parses_json_and_extracts_clean_text():
    provider = QwenProvider.__new__(QwenProvider)
    raw_response = """
    <think>
    Merging points 1 and 2 in order.
    </think>
    ```json
    {
      "polished_text": "First, the team reviewed the project scope. Second, budget approval was granted.",
      "speakers": ["Bob", "Carol"],
      "action_items": []
    }
    ```
    """
    with patch.object(provider, "_infer", return_value=raw_response):
        result = provider.merge_discussion_points([
            {"polished_text": "The team reviewed the project scope.", "speakers": ["Bob"]},
            {"polished_text": "Budget approval was granted.", "speakers": ["Carol"]}
        ])
        assert isinstance(result, dict)
        assert result.get("polished_text") == "First, the team reviewed the project scope. Second, budget approval was granted."
        assert not result.get("polished_text").startswith("{")

def test_merge_discussion_points_malformed_json_extracts_text_fallback():
    provider = QwenProvider.__new__(QwenProvider)
    # Malformed JSON with trailing comma or bad syntax, but containing "polished_text"
    raw_response = '{"polished_text": "Extracted despite bad syntax", "broken": ,,,}'
    with patch.object(provider, "_infer", return_value=raw_response):
        result = provider.merge_discussion_points([
            {"polished_text": "Fallback point 1", "speakers": []},
            {"polished_text": "Fallback point 2", "speakers": []}
        ])
        assert isinstance(result, dict)
        assert result.get("polished_text") == "Extracted despite bad syntax"

@pytest.mark.asyncio
async def test_downstream_data_preserved_on_stage1_regenerate():
    from unittest.mock import AsyncMock
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.execute.return_value.fetchone = MagicMock(return_value=None)
    mock_db.commit = AsyncMock()
    existing_rom = {
        "stage1": {"discussion_points": [{"id": "p1", "text": "Old s1"}]},
        "stage2": {"status": "done", "polished_points": [{"id": "p1", "polished_text": "Old s2"}]},
        "stage3": {"status": "agendas_ready", "agendas": [{"agenda_id": "A1", "title": "Old s3"}]},
        "final_rom": {"agendas": [{"agenda_id": "A1", "discussion_points": [{"id": "p1"}]}]},
    }

    req_mock = MagicMock()
    req_mock.transcript_window_minutes = 5
    req_mock.batch_window_processing = False
    req_mock.max_windows_parallel = 2
    req_mock.use_video_context = False
    req_mock.reference_example_points = None
    req_mock.previous_meeting_mode = "none"
    req_mock.previous_meeting_id = None
    req_mock.previous_meeting_top_k = 3
    req_mock.use_dspy_mode = False

    with patch("routers.rom_router._get_recording_or_404") as mock_get_rec, \
         patch("routers.rom_router._get_rom_data", return_value=existing_rom), \
         patch("routers.rom_router._save_rom_data") as mock_save, \
         patch.object(rom_service, "extract_discussion_points", return_value={"discussion_points": [{"id": "new_p1"}], "windows_processed": 1}):

        mock_get_rec.return_value = {"transcript": '[{"start": 0, "end": 10, "text": "Hello"}]', "meeting_name": "Test", "meeting_date": "2026-09-01"}
        await generate_stage1("rec1", req_mock, current_user={"user_id": "u1"}, db=mock_db)

        saved_data = mock_save.call_args[0][2]
        # Stage 2, Stage 3, and final_rom MUST NOT be deleted!
        assert "stage2" in saved_data
        assert "stage3" in saved_data
        assert "final_rom" in saved_data
        assert saved_data["stage2"]["polished_points"][0]["polished_text"] == "Old s2"
        # Outdated warnings must be present
        assert "outdated_warnings" in saved_data
        assert "stage2" in saved_data["outdated_warnings"]
        assert "stage3" in saved_data["outdated_warnings"]
        assert "final_rom" in saved_data["outdated_warnings"]

@pytest.mark.asyncio
async def test_downstream_data_preserved_on_stage2_regenerate():
    from unittest.mock import AsyncMock
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.execute.return_value.fetchone = MagicMock(return_value=None)
    mock_db.commit = AsyncMock()
    existing_rom = {
        "stage1": {"discussion_points": [{"id": "p1"}]},
        "stage2": {"status": "done", "polished_points": [{"id": "p1", "polished_text": "Old s2"}]},
        "stage3": {"status": "agendas_ready", "agendas": [{"agenda_id": "A1"}]},
        "final_rom": {"agendas": [{"agenda_id": "A1"}]},
        "outdated_warnings": {"stage2": "Old warning"}
    }

    req_mock = MagicMock()
    req_mock.discussion_window_size = 5
    req_mock.parallel_window_processing = False
    req_mock.min_similarity_threshold = 0.5
    req_mock.process_all_together = False
    req_mock.separate_action_extraction = False
    req_mock.reference_example_points = None
    req_mock.previous_meeting_mode = "none"
    req_mock.previous_meeting_id = None
    req_mock.previous_meeting_top_k = 3
    req_mock.use_dspy_mode = False

    with patch("routers.rom_router._get_recording_or_404") as mock_get_rec, \
         patch("routers.rom_router._get_rom_data", return_value=existing_rom), \
         patch("routers.rom_router._save_rom_data") as mock_save, \
         patch("routers.rom_router._clear_stage2_edit_history"), \
         patch.object(rom_service, "enhance_discussion_points", return_value=[{"id": "fresh_p1"}]):

        mock_get_rec.return_value = {"transcript": '[{"start": 0, "end": 10, "text": "Hello"}]', "meeting_name": "Test", "meeting_date": "2026-09-01"}
        await generate_stage2("rec1", req_mock, current_user={"user_id": "u1"}, db=mock_db)

        saved_data = mock_save.call_args[0][2]
        # Stage 3 and final_rom MUST NOT be deleted!
        assert "stage3" in saved_data
        assert "final_rom" in saved_data
        # Stage 2 warning cleared
        assert "stage2" not in saved_data["outdated_warnings"]
        # Stage 3 and Final ROM marked outdated
        assert "stage3" in saved_data["outdated_warnings"]
        assert "final_rom" in saved_data["outdated_warnings"]

@pytest.mark.asyncio
async def test_timeline_and_order_optional_in_final_rom_generation():
    mock_db = MagicMock()
    rom_data = {
        "stage2": {"polished_points": [{"id": "p1", "polished_text": "Point 1", "timeline_start": 0, "timeline_end": 60}]},
        "stage3": {
            "agendas": [{"agenda_id": "A1", "title": "Agenda 1"}],
            "expanded_agendas": []
        },
        "outdated_warnings": {"final_rom": "Was outdated"}
    }

    # Test with BOTH DISABLED
    req_disabled = GenerateFinalRomRequest(
        discussion_order=["A1"],
        agenda_timeline={"A1": {"start_sec": 0, "end_sec": 60}},
        enable_discussion_order=False,
        enable_agenda_timeline=False,
    )

    with patch("routers.rom_router._get_recording_or_404", return_value={"id": "rec1"}), \
         patch("routers.rom_router._get_rom_data", return_value=rom_data), \
         patch("routers.rom_router._save_rom_data") as mock_save, \
         patch.object(rom_service, "map_points_to_agendas", return_value={
             "final_rom_agendas": [{"agenda_id": "A1", "discussion_points": []}],
             "candidate_results": [],
             "batch_assignments": [],
             "point_mappings": {},
             "agenda_groups": {},
             "similarity_matrix": []
         }) as mock_map:

        await generate_final_rom_from_agendas("rec1", req_disabled, current_user={"user_id": "u1"}, db=mock_db)

        # map_points_to_agendas must receive None for discussion_order and agenda_timeline when disabled
        args, kwargs = mock_map.call_args
        assert kwargs.get("discussion_order") is None
        assert kwargs.get("agenda_timeline") is None

        # Final ROM outdated warning must be cleared
        saved = mock_save.call_args[0][2]
        assert "final_rom" not in saved.get("outdated_warnings", {})
