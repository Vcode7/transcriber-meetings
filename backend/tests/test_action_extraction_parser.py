import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock
from services.ai_provider import QwenProvider

@pytest.fixture
def provider():
    p = QwenProvider.__new__(QwenProvider)
    return p

def test_parse_markdown_code_fences_triple_backticks(provider):
    raw = """
```json
{
  "action_items": [
    {
      "task": "Deploy backend update",
      "owner": "Vikas",
      "deadline": "Friday",
      "expected_outcome": "Updated API in production"
    }
  ]
}
```
"""
    items = provider.parse_action_extraction_response(raw)
    assert len(items) == 1
    assert items[0]["task"] == "Deploy backend update"
    assert items[0]["owner"] == "Vikas"
    assert items[0]["deadline"] == "Friday"
    assert items[0]["expected_outcome"] == "Updated API in production"


def test_parse_markdown_code_fences_single_backtick(provider):
    raw = "`json\n{\n  \"action_items\": [\n    {\n      \"task\": \"Run unit tests\",\n      \"owner\": null,\n      \"deadline\": null\n    }\n  ]\n}\n`"
    items = provider.parse_action_extraction_response(raw)
    assert len(items) == 1
    assert items[0]["task"] == "Run unit tests"
    assert items[0]["owner"] is None
    assert items[0]["deadline"] is None
    assert items[0]["expected_outcome"] is None


def test_parse_unclosed_code_fence(provider):
    raw = "```json\n{\n  \"action_items\": [\n    {\n      \"task\": \"Check server metrics\",\n      \"owner\": \"Ops\",\n      \"deadline\": null\n    }\n  ]\n}"
    items = provider.parse_action_extraction_response(raw)
    assert len(items) == 1
    assert items[0]["task"] == "Check server metrics"
    assert items[0]["owner"] == "Ops"


def test_parse_with_surrounding_prose(provider):
    raw = """
Based on the discussion, here are the extracted action points:
```json
{
  "action_items": [
    {
      "task": "Refactor authentication flow",
      "owner": null,
      "deadline": "Next sprint",
      "expected_outcome": null
    }
  ]
}
```
Let me know if you need any further analysis. {extra: text}
"""
    items = provider.parse_action_extraction_response(raw)
    assert len(items) == 1
    assert items[0]["task"] == "Refactor authentication flow"
    assert items[0]["owner"] is None
    assert items[0]["deadline"] == "Next sprint"
    assert items[0]["expected_outcome"] is None


def test_accept_null_for_optional_fields(provider):
    raw = """
{
  "action_items": [
    {
      "task": "Migrate database schema",
      "owner": null,
      "deadline": null,
      "expected_outcome": null
    },
    {
      "task": null,
      "owner": "Alice",
      "deadline": "2026-10-01",
      "expected_outcome": null
    }
  ]
}
"""
    items = provider.parse_action_extraction_response(raw)
    assert len(items) == 2
    assert items[0]["task"] == "Migrate database schema"
    assert items[0]["owner"] is None
    assert items[0]["deadline"] is None
    assert items[0]["expected_outcome"] is None

    assert items[1]["task"] is None
    assert items[1]["owner"] == "Alice"
    assert items[1]["deadline"] == "2026-10-01"


def test_empty_action_items_not_discarded(provider):
    raw = """
```json
{
  "action_items": []
}
```
"""
    items = provider.parse_action_extraction_response(raw)
    assert items == []


def test_clear_logging_on_invalid_json(provider, caplog):
    raw = "I could not find any action items in the provided discussion."
    with caplog.at_level("WARNING"):
        items = provider.parse_action_extraction_response(raw, chunk_info="chunk 1/1")
    assert items == []
    # Check that error is clearly logged
    assert any("JSON parsing/validation failed" in record.message for record in caplog.records)
    assert any("Parsing error:" in record.message for record in caplog.records)


def test_extract_actions_from_enhanced_points_integration(provider):
    points = [
        {
            "id": "pt-1",
            "polished_text": "Team agreed to fix CSS layout issues.",
            "action_owner": None,
            "speakers": ["Vikas"]
        }
    ]
    mock_llm_response = """
```json
{
  "action_items": [
    {
      "source_point_id": "pt-1",
      "task": "Fix CSS layout issues on mobile viewports",
      "owner": null,
      "deadline": null,
      "expected_outcome": null
    }
  ]
}
```
"""
    provider._infer = MagicMock(return_value=mock_llm_response)
    results = provider.extract_actions_from_enhanced_points(points)
    assert len(results) == 1
    assert results[0]["task"] == "Fix CSS layout issues on mobile viewports"
    # Null owner in response falls back to speaker if present
    assert results[0]["owner"] == "Vikas"
    assert results[0]["expected_outcome"] is None
