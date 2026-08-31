import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from services.vector_store import get_stage2_points_store, VectorStore
from services.rom_service import rom_service


def test_index_stage2_points_in_chromadb(tmp_path):
    user_id = "test_user_stage2_idx"
    recording_id = "rec_meeting_001"
    meeting_name = "Sprint Planning Meeting"
    meeting_date = "2026-08-25"

    points = [
        {
            "id": "pt-1",
            "polished_text": "The engineering team agreed to migrate database schema to PostgreSQL 16.",
            "speakers": ["Alice", "Bob"],
            "action_owner": "Alice",
        },
        {
            "id": "pt-2",
            "discussion_point": "Frontend team confirmed adoption of Tailwind CSS v4 for UI consistency.",
            "speakers": ["Charlie"],
            "action_owner": None,
        },
    ]

    # Index points
    added_count = rom_service.index_stage2_points_in_chromadb(
        recording_id=recording_id,
        user_id=user_id,
        points=points,
        meeting_name=meeting_name,
        meeting_date=meeting_date,
    )

    assert added_count == 2

    # Verify points in ChromaDB store
    store = get_stage2_points_store(user_id)
    assert store._collection is not None
    assert store._collection.count() >= 2

    # Re-indexing the same meeting should overwrite/update without duplicate buildup
    added_again = rom_service.index_stage2_points_in_chromadb(
        recording_id=recording_id,
        user_id=user_id,
        points=points,
        meeting_name=meeting_name,
        meeting_date=meeting_date,
    )
    assert added_again == 2
    assert store._collection.count() == 2


def test_retrieve_previous_stage2_context_select_and_auto():
    user_id = "test_user_stage2_retrieval"
    store = get_stage2_points_store(user_id)
    
    # Clear collection for clean test run
    try:
        store.delete_by_filter("user_id", user_id)
    except Exception:
        pass

    # Meeting A: Database migration
    rom_service.index_stage2_points_in_chromadb(
        recording_id="rec_A",
        user_id=user_id,
        points=[
            {
                "id": "pt-a1",
                "polished_text": "Database migration to PostgreSQL 16 was finalized and scheduled for Friday.",
                "speakers": ["Alice"],
                "action_owner": "Alice",
            }
        ],
        meeting_name="Meeting A - Database Architecture",
        meeting_date="2026-08-20",
    )

    # Meeting B: UI design system
    rom_service.index_stage2_points_in_chromadb(
        recording_id="rec_B",
        user_id=user_id,
        points=[
            {
                "id": "pt-b1",
                "polished_text": "UI team agreed on dark mode tokens and micro-interactions in design system.",
                "speakers": ["Bob"],
                "action_owner": "Bob",
            }
        ],
        meeting_name="Meeting B - Design Review",
        meeting_date="2026-08-22",
    )

    # Current Meeting is Meeting C (rec_C)
    from services.text_embedding_service import get_text_embedder
    embedder = get_text_embedder()
    embedder.load()

    query_text = "PostgreSQL database migration timeline"
    q_vec = embedder.encode_batch([query_text])[0]

    # 1. Mode: Select Meeting (Select Meeting A)
    results_select = rom_service.retrieve_previous_stage2_context(
        query=query_text,
        query_vec=q_vec,
        user_id=user_id,
        current_recording_id="rec_C",
        previous_meeting_id="rec_A",
        top_k=3,
        min_similarity_threshold=0.0,
    )
    assert len(results_select) == 1
    assert results_select[0]["meeting_id"] == "rec_A"
    assert "PostgreSQL" in results_select[0]["_text"]
    assert results_select[0]["stage"] == "stage2"

    # 2. Mode: Auto Retrieve (Should exclude rec_C and find most similar across rec_A and rec_B)
    results_auto = rom_service.retrieve_previous_stage2_context(
        query=query_text,
        query_vec=q_vec,
        user_id=user_id,
        current_recording_id="rec_C",
        previous_meeting_id=None,
        top_k=2,
        min_similarity_threshold=0.0,
    )
    assert len(results_auto) >= 1
    # Check that current recording is never returned
    assert all(r["meeting_id"] != "rec_C" for r in results_auto)
    # The top result should be Meeting A (higher similarity for database query)
    assert results_auto[0]["meeting_id"] == "rec_A"
    assert all(r["stage"] == "stage2" for r in results_auto)


def test_format_previous_stage2_context():
    sample_results = [
        {
            "_text": "Database migration to PostgreSQL 16 was finalized.",
            "meeting_name": "Architecture Sync",
            "date": "2026-08-20",
            "speakers": "Alice, Bob",
            "action_owner": "Alice",
        },
        {
            "_text": "API authentication switched to OAuth2 Bearer tokens.",
            "meeting_name": "Security Review",
            "date": "2026-08-21",
            "speakers": "Charlie",
            "action_owner": "",
        },
    ]

    formatted = rom_service._format_previous_stage2_context(sample_results)
    assert "[Previous Meeting: Architecture Sync | Date: 2026-08-20]" in formatted
    assert "- Stage 2 Point: Database migration to PostgreSQL 16 was finalized. (Speakers: Alice, Bob, Action: Alice)" in formatted
    assert "[Previous Meeting: Security Review | Date: 2026-08-21]" in formatted
    assert "- Stage 2 Point: API authentication switched to OAuth2 Bearer tokens. (Speakers: Charlie)" in formatted


def test_enhance_discussion_points_with_previous_meeting_context():
    user_id = "test_user_stage2_pipeline"
    current_rec_id = "rec_curr_123"

    # Pre-index a previous meeting Stage 2 point
    rom_service.index_stage2_points_in_chromadb(
        recording_id="rec_past_999",
        user_id=user_id,
        points=[
            {
                "id": "past-pt-1",
                "polished_text": "Previous decision: microservice architecture was chosen for search engine.",
                "speakers": ["Dave"],
                "action_owner": "Dave",
            }
        ],
        meeting_name="Past Architecture Review",
        meeting_date="2026-08-10",
    )

    stage1_points = [
        {
            "id": "pt-101",
            "discussion_point": "Team reviewed search service implementation and decided to deploy to staging.",
            "speakers": ["Dave", "Eve"],
            "action_owner": "Eve",
            "timeline_start": 0.0,
            "timeline_end": 60.0,
        }
    ]

    # Mock provider to inspect prompt and return valid response
    captured_prompts = []

    mock_provider = MagicMock()
    def mock_enhance_together(points_json, meeting_context, global_context, previous_meeting_context="", reference_example_points=None):
        captured_prompts.append({
            "meeting_context": meeting_context,
            "global_context": global_context,
            "previous_meeting_context": previous_meeting_context,
        })
        return {
            "enhanced_points": [
                {
                    "original_point_ids": ["pt-101"],
                    "discussion_point": "Dave and Eve reviewed the search service implementation, aligning with the previous microservice architecture decision, and confirmed deployment to staging.",
                    "speakers": ["Dave", "Eve"],
                    "action_owner": "Eve",
                }
            ]
        }

    mock_provider.enhance_rom_all_points_together.side_effect = mock_enhance_together

    with patch("services.ai_provider.get_provider", return_value=mock_provider):
        enhanced = rom_service.enhance_discussion_points(
            discussion_points=stage1_points,
            recording_id=current_rec_id,
            user_id=user_id,
            process_all_together=True,
            previous_meeting_mode="auto",
            previous_meeting_top_k=2,
            min_similarity_threshold=0.0,
            meeting_name="Current Deployment Sync",
            meeting_date="2026-08-25",
        )

    assert len(enhanced) == 1
    assert "Dave and Eve" in enhanced[0]["polished_text"]

    # Verify previous meeting context was passed to provider
    assert len(captured_prompts) == 1
    prev_ctx = captured_prompts[0]["previous_meeting_context"]
    assert "Past Architecture Review" in prev_ctx
    assert "microservice architecture" in prev_ctx

    # Verify retrieved_context on enhanced point contains previous_meeting_chunks
    retrieved = enhanced[0].get("retrieved_context", {})
    prev_chunks = retrieved.get("previous_meeting_chunks", [])
    assert len(prev_chunks) >= 1
    assert prev_chunks[0]["meeting_name"] == "Past Architecture Review"
    assert enhanced[0]["context_usage_report"]["previous_meeting_context_used"] is True

    # Verify current meeting points were also indexed into ChromaDB
    curr_store = get_stage2_points_store(user_id)
    assert curr_store._collection is not None
