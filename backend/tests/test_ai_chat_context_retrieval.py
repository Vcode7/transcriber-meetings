"""
Test suite for AI Chat Context Retrieval Logic.
Verifies:
1. Semantic search vs Direct meeting selection triage routing.
2. Direct meeting selection when user explicitly mentions a meeting.
3. Meeting list viewing / listing intent.
4. get_direct_meeting_context extraction (ChromaDB + SQLite fallback).
5. Graceful fallback to semantic search.
"""
import os
import sys
import uuid
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ai_chat_service import (
    TriageResult,
    find_matching_meeting,
    filter_candidate_meetings,
    format_candidate_meetings_for_prompt,
    select_meeting_from_list,
    is_meeting_listing_query,
    format_meeting_list_response,
    get_direct_meeting_context,
    triage_query,
    AI_CHAT_TRIAGE_PROMPT,
)


def test_triage_result_backward_compatibility():
    """Verify that TriageResult unpacks seamlessly into (ctx_req, detail) as a 2-tuple."""
    tr = TriageResult(
        context_required=True,
        detail="Qwen3 and Ollama optimization",
        retrieval_mode="direct_meeting",
        target_meeting_id="rec_abc123",
        target_meeting_name="AI Architecture Sync",
    )
    # Tuple unpacking
    ctx_req, detail = tr
    assert ctx_req is True
    assert detail == "Qwen3 and Ollama optimization"

    # Indexing
    assert tr[0] is True
    assert tr[1] == "Qwen3 and Ollama optimization"
    assert len(tr) == 2

    # Rich attributes
    assert tr.retrieval_mode == "direct_meeting"
    assert tr.target_meeting_id == "rec_abc123"
    assert tr.target_meeting_name == "AI Architecture Sync"


def test_meeting_listing_intent_detection():
    """Verify detection of user requests to see/list available meetings."""
    assert is_meeting_listing_query("What meetings do I have?") is True
    assert is_meeting_listing_query("List my meetings") is True
    assert is_meeting_listing_query("Show available meetings") is True
    assert is_meeting_listing_query("Which meetings are recorded?") is True
    assert is_meeting_listing_query("Display meeting list") is True
    assert is_meeting_listing_query("How many meetings exist?") is True

    # Non-listing queries
    assert is_meeting_listing_query("What was discussed about budget?") is False
    assert is_meeting_listing_query("Summarize the Sprint 42 meeting") is False
    assert is_meeting_listing_query("Hello there!") is False


def test_meeting_list_response_formatting():
    """Verify formatting of the meeting list response for AI Chat."""
    mock_meetings = [
        {"id": "rec_001", "name": "Sprint 42 Retrospective", "date": "2026-03-28", "has_stage2": True},
        {"id": "rec_002", "name": "Client Onboarding", "date": "2026-03-25", "has_stage2": False},
    ]
    resp = format_meeting_list_response(mock_meetings)
    assert "Sprint 42 Retrospective" in resp
    assert "Client Onboarding" in resp
    assert "Stage 2 points indexed" in resp
    assert "2026-03-28" in resp


def test_find_matching_meeting_explicit_mention():
    """Verify that find_matching_meeting accurately detects explicit mentions."""
    mock_meetings = [
        {"id": "rec_101", "name": "Sprint 42 Retrospective", "date": "2026-03-28"},
        {"id": "rec_102", "name": "Product Design Review", "date": "2026-03-25"},
        {"id": "rec_103", "name": "Budget & Finance Sync", "date": "2026-03-20"},
    ]

    # Full title in query
    match1 = find_matching_meeting("Can you summarize the Sprint 42 Retrospective?", mock_meetings)
    assert match1 is not None
    assert match1["id"] == "rec_101"

    # Distinct significant keyword match
    match2 = find_matching_meeting("What was decided in the Product Design meeting?", mock_meetings)
    assert match2 is not None
    assert match2["id"] == "rec_102"

    # Exact ID match
    match3 = find_matching_meeting("Tell me about rec_103", mock_meetings)
    assert match3 is not None
    assert match3["id"] == "rec_103"

    # No match for generic query
    match_none = find_matching_meeting("What are our general company goals?", mock_meetings)
    assert match_none is None


def test_triage_prefers_direct_meeting_when_explicitly_mentioned():
    """Verify triage_query prioritizes direct_meeting mode when user explicitly mentions a meeting."""
    mock_meetings = [
        {"id": "rec_sprint42", "name": "Sprint 42 Retrospective", "date": "2026-03-28", "has_stage2": True},
        {"id": "rec_q3kickoff", "name": "Q3 Strategy Kickoff", "date": "2026-03-25", "has_stage2": True},
    ]
    res = triage_query(
        question="What decisions were made in the Sprint 42 Retrospective?",
        chat_history=[],
        available_meetings=mock_meetings,
    )
    assert res.context_required is True
    assert res.retrieval_mode == "direct_meeting"
    assert res.target_meeting_id == "rec_sprint42"
    assert res.target_meeting_name == "Sprint 42 Retrospective"


def test_triage_lists_meetings_directly():
    """Verify triage_query immediately returns the meeting list when user asks to list meetings."""
    mock_meetings = [
        {"id": "rec_sprint42", "name": "Sprint 42 Retrospective", "date": "2026-03-28", "has_stage2": True},
    ]
    res = triage_query(
        question="What meetings are available?",
        chat_history=[],
        available_meetings=mock_meetings,
    )
    assert res.context_required is False
    assert res.retrieval_mode == "none"
    assert "Sprint 42 Retrospective" in res.detail


def test_direct_meeting_context_extraction_with_db_mock(monkeypatch):
    """Verify get_direct_meeting_context extracts points chronologically."""
    mock_docs = ["Point 1: Discussed requirements", "Point 2: Assigned tasks to Bob"]
    mock_metas = [
        {"chunk_index": 0, "speakers": "Alice", "meeting_name": "Test Meeting"},
        {"chunk_index": 1, "speakers": "Bob", "meeting_name": "Test Meeting"},
    ]

    class MockCollection:
        def get(self, where=None, include=None):
            return {"documents": mock_docs, "metadatas": mock_metas}

    class MockStore:
        def exists(self):
            return True
        def load_or_create(self):
            pass
        _collection = MockCollection()

    monkeypatch.setattr("services.vector_store.get_stage2_points_store", lambda uid, dim=0: MockStore())

    results = get_direct_meeting_context(user_id="test_user", meeting_id="rec_mock_001")
    assert len(results) == 2
    assert results[0]["_text"] == "Point 1: Discussed requirements"
    assert results[0]["chunk_index"] == 0
    assert results[0]["retrieval_mode"] == "direct_meeting"
    assert results[1]["_text"] == "Point 2: Assigned tasks to Bob"
    assert results[1]["chunk_index"] == 1


def test_first_call_triage_prompt_has_no_meeting_list():
    """Verify that the first triage prompt does NOT contain available meetings."""
    assert "{available_meetings_text}" not in AI_CHAT_TRIAGE_PROMPT
    assert "available_meetings" not in AI_CHAT_TRIAGE_PROMPT
    # Ensures prompt is strictly lightweight for Call 1
    formatted = AI_CHAT_TRIAGE_PROMPT.format(
        conversation_history="",
        question="What was discussed about performance?",
    )
    assert "What was discussed about performance?" in formatted


def test_filter_candidate_meetings_scale_to_1000():
    """Verify that filter_candidate_meetings handles 1,000 meetings without blowing up tokens."""
    # Generate 1,000 mock meetings
    large_meeting_list = [
        {"id": f"rec_{i}", "name": f"Weekly Team Sync #{i}", "date": f"2025-01-{(i % 28) + 1:02d}"}
        for i in range(1, 1001)
    ]
    # Insert target meeting somewhere in the middle
    target = {"id": "rec_target_999", "name": "Sprint 42 Architecture Deep Dive", "date": "2026-03-28"}
    large_meeting_list.insert(542, target)

    # Filter candidates for reference "Sprint 42"
    candidates = filter_candidate_meetings(
        reference="Sprint 42",
        question="What were the decisions in the Sprint 42 meeting?",
        available_meetings=large_meeting_list,
        max_candidates=30,
    )

    # Must be capped at max_candidates (30), never 1,000
    assert len(candidates) <= 30
    # Must have matched the relevant meeting as top candidate!
    assert candidates[0]["id"] == "rec_target_999"
    assert "Sprint 42" in candidates[0]["name"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])

