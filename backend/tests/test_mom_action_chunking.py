import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from unittest.mock import MagicMock, patch
from models.settings import UserSettings, UserSettingsUpdate
from services.ai_provider import QwenProvider

def test_user_settings_rom_action_generation_chunk_size():
    """Verify UserSettings and UserSettingsUpdate support rom_action_generation_chunk_size with default 10."""
    us = UserSettings(user_id="u123")
    assert us.rom_action_generation_chunk_size == 10

    # Test update model
    update = UserSettingsUpdate(rom_action_generation_chunk_size=15)
    assert update.rom_action_generation_chunk_size == 15

    # Out of bounds should raise ValidationError
    with pytest.raises(Exception):
        UserSettings(user_id="u123", rom_action_generation_chunk_size=0)

    with pytest.raises(Exception):
        UserSettings(user_id="u123", rom_action_generation_chunk_size=101)


def test_extract_actions_from_enhanced_points_chunking():
    """Verify extract_actions_from_enhanced_points chunks points by 10 (or custom size) and combines results."""
    provider = QwenProvider.__new__(QwenProvider)

    # 25 dummy polished points
    points = [
        {
            "id": f"pt-{i}",
            "discussion_point": f"Discussion point {i} text describing task {i}.",
            "speakers": [f"Speaker {i}"],
            "action_owner": f"Owner {i}" if i % 2 == 0 else None,
        }
        for i in range(1, 26)
    ]

    captured_chunks = []

    def fake_infer(prompt, max_new_tokens=4096, task_key=None):
        # Extract the chunk JSON from the prompt
        # We know prompt contains the points JSON
        # Return dummy action items corresponding to that chunk
        import re
        marker = "DISCUSSION POINTS:"
        if marker in prompt:
            after = prompt.split(marker)[1]
            # find first [ ... ] before "Return ONLY valid JSON"
            m = re.search(r'\[\s*\{.*?\}\s*\]', after, re.DOTALL)
            if m:
                chunk_data = json.loads(m.group())
                captured_chunks.append(chunk_data)
                items = [
                    {
                        "task": f"Complete action for {p['id']}",
                        "owner": p.get("action_owner") or "Unassigned",
                        "deadline": "2026-10-01",
                        "source_point_id": p["id"]
                    }
                    for p in chunk_data
                ]
                return json.dumps({"action_items": items})
        return json.dumps({"action_items": []})

    provider._infer = fake_infer
    provider._get_active_settings = MagicMock(return_value={"max_tokens_mom_extract_actions": 4096, "rom_action_generation_chunk_size": 10})

    # Call with default chunking (10)
    result = provider.extract_actions_from_enhanced_points(points, chunk_size=10)

    # 25 points / 10 = 3 chunks: sizes 10, 10, 5
    assert len(captured_chunks) == 3
    assert len(captured_chunks[0]) == 10
    assert len(captured_chunks[1]) == 10
    assert len(captured_chunks[2]) == 5

    # Check that all 25 actions were collected
    assert len(result) == 25
    assert result[0]["source_point_id"] == "pt-1"
    assert result[24]["source_point_id"] == "pt-25"


def test_extract_actions_returns_full_list_not_single_action():
    """Verify that extract_actions_from_enhanced_points extracts full list of actions across multiple points."""
    provider = QwenProvider.__new__(QwenProvider)
    provider._get_active_settings = MagicMock(return_value={"max_tokens_mom_extract_actions": 4096, "rom_action_generation_chunk_size": 10})

    points = [
        {"id": "pt-1", "enhanced_text": "Alice will deploy the backend by Friday.", "action_owner": "Alice"},
        {"id": "pt-2", "enhanced_text": "Bob will update the database schema and Charlie will review API docs.", "action_owner": "Bob"},
        {"id": "pt-3", "enhanced_text": "General team discussion with no actionable items.", "action_owner": None},
    ]

    # Mock infer returning multiple action items for points 1 and 2 (including 2 actions for point 2)
    mock_llm_response = """
```json
{
  "action_items": [
    {
      "source_point_id": "pt-1",
      "task": "Deploy the backend",
      "owner": "Alice",
      "deadline": "Friday"
    },
    {
      "source_point_id": "pt-2",
      "task": "Update the database schema",
      "owner": "Bob",
      "deadline": null
    },
    {
      "source_point_id": "pt-2",
      "task": "Review API docs",
      "owner": "Charlie",
      "deadline": null
    }
  ]
}
```
"""
    provider._infer = MagicMock(return_value=mock_llm_response)

    results = provider.extract_actions_from_enhanced_points(points)
    assert isinstance(results, list)
    assert len(results) == 3
    assert results[0]["task"] == "Deploy the backend"
    assert results[0]["owner"] == "Alice"
    assert results[1]["task"] == "Update the database schema"
    assert results[2]["task"] == "Review API docs"
    assert results[2]["owner"] == "Charlie"


def test_extract_actions_handles_top_level_array():
    """Verify parser handles top-level JSON array output from LLM."""
    provider = QwenProvider.__new__(QwenProvider)
    provider._get_active_settings = MagicMock(return_value={"max_tokens_mom_extract_actions": 4096, "rom_action_generation_chunk_size": 10})

    points = [
        {"id": "pt-1", "enhanced_text": "Alice to finish task 1."},
        {"id": "pt-2", "enhanced_text": "Bob to finish task 2."},
    ]

    mock_llm_response = """
[
  {
    "source_point_id": "pt-1",
    "task": "Finish task 1",
    "owner": "Alice",
    "deadline": null
  },
  {
    "source_point_id": "pt-2",
    "task": "Finish task 2",
    "owner": "Bob",
    "deadline": null
  }
]
"""
    provider._infer = MagicMock(return_value=mock_llm_response)

    results = provider.extract_actions_from_enhanced_points(points)
    assert len(results) == 2
    assert results[0]["task"] == "Finish task 1"
    assert results[1]["task"] == "Finish task 2"


def test_extract_actions_handles_alternate_keys():
    """Verify parser handles alternate keys like 'actions' or 'action_points'."""
    provider = QwenProvider.__new__(QwenProvider)
    provider._get_active_settings = MagicMock(return_value={"max_tokens_mom_extract_actions": 4096, "rom_action_generation_chunk_size": 10})

    points = [{"id": "pt-1", "enhanced_text": "Alice to finish task 1."}]

    mock_llm_response = '{"action_points": [{"source_point_id": "pt-1", "task": "Finish task 1", "owner": "Alice"}]}'
    provider._infer = MagicMock(return_value=mock_llm_response)

    results = provider.extract_actions_from_enhanced_points(points)
    assert len(results) == 1
    assert results[0]["task"] == "Finish task 1"


def test_extract_actions_owner_fallback_to_point_speakers_and_action_owner():
    """Verify owner fallback resolves to point action_owner or speakers when LLM returns null/Unassigned."""
    provider = QwenProvider.__new__(QwenProvider)
    provider._get_active_settings = MagicMock(return_value={"max_tokens_mom_extract_actions": 4096, "rom_action_generation_chunk_size": 10})

    points = [
        {"id": "pt-1", "enhanced_text": "Vikas will setup the database replica.", "action_owner": "Vikas", "speakers": ["Vikas"]},
        {"id": "pt-2", "enhanced_text": "Alice will conduct security audit.", "action_owner": None, "speakers": ["Alice"]},
        {"id": "pt-3", "enhanced_text": "Bob and Charlie will review API schemas.", "action_owner": "Unassigned", "speakers": ["Bob", "Charlie"]},
    ]

    mock_llm_response = """
{
  "action_items": [
    {"source_point_id": "pt-1", "task": "Setup database replica with automated daily snapshots.", "owner": null},
    {"source_point_id": "pt-2", "task": "Conduct comprehensive security audit across all microservices.", "owner": "Unassigned"},
    {"source_point_id": "pt-3", "task": "Review API schemas and publish OpenAPI specifications.", "owner": "none"}
  ]
}
"""
    provider._infer = MagicMock(return_value=mock_llm_response)

    results = provider.extract_actions_from_enhanced_points(points)
    assert len(results) == 3
    assert results[0]["owner"] == "Vikas"
    assert results[1]["owner"] == "Alice"
    assert results[2]["owner"] == "Bob, Charlie"


def test_generate_mom_from_enhanced_rom_resolves_owners_from_speakers():
    """Verify rom_service.generate_mom_from_enhanced_rom resolves owners from speakers and participant context."""
    from services.rom_service import RomService

    rom_srv = RomService()
    polished_points = [
        {
            "id": "pt-10",
            "polished_text": "Vikas discussed database optimization and committed to migrating indices.",
            "speakers": ["Vikas"],
            "action_owner": None
        }
    ]

    mock_extracted = [
        {
            "source_point_id": "pt-10",
            "task": "Migrate database indices to B-tree partitioned tables and benchmark query performance.",
            "owner": "Unassigned",
            "deadline": "2026-10-15"
        }
    ]

    with patch("services.ai_provider.get_provider") as mock_get_p:
        mock_p = MagicMock()
        mock_p.extract_actions_from_enhanced_points.return_value = mock_extracted
        mock_p.unload_model = MagicMock()
        mock_get_p.return_value = mock_p

        mom = rom_srv.generate_mom_from_enhanced_rom(
            polished_points=polished_points,
            recording_meta={"filename": "Tech Sync", "speakers_detected": ["Vikas"]},
            recording_id="rec-123",
            user_id="user-123",
            separate_action_extraction=False
        )

        assert len(mom["action_items"]) == 1
        assert mom["action_items"][0]["owner"] == "Vikas"
        assert "Migrate database indices" in mom["action_items"][0]["task"]


