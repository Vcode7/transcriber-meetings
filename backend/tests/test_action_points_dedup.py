import json
import pytest
from services.ai_provider import QwenProvider, MOM_DEDUPLICATE_ACTION_POINTS_PROMPT

def test_deduplicate_action_points_empty_and_single():
    provider = QwenProvider()
    assert provider.deduplicate_action_points([]) == []
    single_item = [{"assigner": "Alice", "assignee": "Bob", "task": "Prepare report", "deadline": "Friday"}]
    assert provider.deduplicate_action_points(single_item) == single_item

def test_deduplicate_action_points_mock_llm(monkeypatch):
    provider = QwenProvider()
    items = [
        {"assigner": "Alice", "assignee": "Bob", "task": "Prepare project budget report", "deadline": "Friday"},
        {"assigner": "Alice", "assignee": "Bob", "task": "Prepare the project budget report by EOD Friday", "deadline": "Friday"},
        {"assigner": "Charlie", "assignee": "David", "task": "Schedule client kickoff meeting", "deadline": "Next Monday"},
    ]

    mock_llm_response = json.dumps({
        "action_items": [
            {"assigner": "Alice", "assignee": "Bob", "task": "Prepare the project budget report by EOD Friday", "deadline": "Friday"},
            {"assigner": "Charlie", "assignee": "David", "task": "Schedule client kickoff meeting", "deadline": "Next Monday"},
        ]
    })

    monkeypatch.setattr(provider, "_infer", lambda prompt, **kwargs: mock_llm_response)

    res = provider.deduplicate_action_points(items)
    assert len(res) == 2
    assert res[0]["task"] == "Prepare the project budget report by EOD Friday"
    assert res[1]["assignee"] == "David"
