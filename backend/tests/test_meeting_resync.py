import os
import sys
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import connect_db, get_db_context
from services.vector_store import get_stage2_points_store
from services.rom_service import rom_service
from services.ai_chat_service import search_stage2_points, build_answer_prompt
from routers.ai_chat_router import (
    _safe_parse_rom_data,
    _extract_stage2_points,
    get_syncable_meetings,
    resync_meeting,
)
from sqlalchemy import text


@pytest.mark.asyncio
async def test_safe_parse_and_extract_stage2():
    # 1. Normal dict
    norm = {"stage2": {"polished_points": [{"polished_text": "Point 1"}]}}
    assert len(_extract_stage2_points(_safe_parse_rom_data(norm))) == 1

    # 2. JSON string
    json_str = json.dumps(norm)
    assert len(_extract_stage2_points(_safe_parse_rom_data(json_str))) == 1

    # 3. Double-encoded JSON string
    double_json = json.dumps(json_str)
    assert len(_extract_stage2_points(_safe_parse_rom_data(double_json))) == 1

    # 4. Corrupt JSON / string null
    assert _safe_parse_rom_data('"null"') == {}
    assert _safe_parse_rom_data('invalid {json') == {}
    assert _safe_parse_rom_data(None) == {}
    assert _extract_stage2_points({}) == []


@pytest.mark.asyncio
async def test_meeting_resync_flow():
    await connect_db()

    user_id = "test_user_resync_e2e"
    recording_id = "rec_resync_test_001"
    meeting_name = "Quarterly Strategy Review"
    meeting_date = "2026-09-16T10:00:00Z"

    points = [
        {
            "id": "p1",
            "polished_text": "The executive committee approved expansion of the telemetry pipeline into APAC region.",
            "speakers": ["Director Jane", "VP Alex"],
            "action_owner": "Director Jane",
        },
        {
            "id": "p2",
            "polished_text": "Security team mandated zero-trust service mesh authentication across all internal microservices.",
            "speakers": ["CISO Mark"],
            "action_owner": "CISO Mark",
        },
    ]

    rom_data_dict = {
        "stage1": {"status": "done"},
        "stage2": {
            "status": "done",
            "polished_points": points,
        },
    }

    # 1. Seed database with user and recording
    async with get_db_context() as session:
        # Create user if needed
        await session.execute(
            text("INSERT OR IGNORE INTO users (id, name, email, hashed_password) VALUES (:id, :n, :e, :p)"),
            {"id": user_id, "n": "Test User", "e": f"{user_id}@test.com", "p": "hash"},
        )
        # Clean recording if exists
        await session.execute(
            text("DELETE FROM recordings WHERE id = :id"),
            {"id": recording_id},
        )
        await session.execute(
            text("DELETE FROM rom_metadata WHERE recording_id = :id"),
            {"id": recording_id},
        )
        # Insert recording
        await session.execute(
            text("""
                INSERT INTO recordings (id, user_id, filename, title, file_path, duration, status, created_at, rom_data)
                VALUES (:id, :uid, :fn, :title, :fp, 120.0, 'completed', :dt, :rd)
            """),
            {
                "id": recording_id,
                "uid": user_id,
                "fn": "strategy_review.m4a",
                "title": meeting_name,
                "fp": "/dummy/path.m4a",
                "dt": meeting_date,
                "rd": json.dumps(rom_data_dict),
            },
        )
        await session.commit()

    # Ensure ChromaDB starts fresh for this test meeting
    store = get_stage2_points_store(user_id)
    store.load_or_create()
    store.delete_by_filter("meeting_id", recording_id)

    mock_user = {"id": user_id, "name": "Test User", "email": f"{user_id}@test.com"}

    # 2. Test get_syncable_meetings detects the meeting needing sync
    syncable = await get_syncable_meetings(user=mock_user)
    target = next((m for m in syncable if m["id"] == recording_id), None)
    assert target is not None
    assert target["stage2_points_count"] == 2
    assert target["indexed_points_count"] == 0
    assert target["needs_sync"] is True
    assert target["sync_status"] == "not_indexed"

    # 3. Test resync_meeting indexes the 2 points
    res = await resync_meeting(recording_id=recording_id, force=False, user=mock_user)
    assert res["status"] == "success"
    assert res["recording_id"] == recording_id
    assert res["total_found"] == 2
    assert res["already_indexed"] == 0
    assert res["newly_indexed"] == 2
    assert res["skipped_failed"] == 0

    # 4. Verify ChromaDB contains the 2 indexed points
    chroma_res = store._collection.get(where={"meeting_id": recording_id}, include=["metadatas"])
    assert len(chroma_res["ids"]) == 2

    # 5. Test idempotency: re-running resync_meeting should NOT duplicate vectors
    res2 = await resync_meeting(recording_id=recording_id, force=False, user=mock_user)
    assert res2["status"] == "success"
    assert res2["total_found"] == 2
    assert res2["already_indexed"] == 2
    assert res2["newly_indexed"] == 0

    chroma_res2 = store._collection.get(where={"meeting_id": recording_id}, include=["metadatas"])
    assert len(chroma_res2["ids"]) == 2, "Duplicate vectors were created!"

    # 6. Verify syncable meetings status is updated to 'synced'
    syncable_after = await get_syncable_meetings(user=mock_user)
    target_after = next((m for m in syncable_after if m["id"] == recording_id), None)
    assert target_after is not None
    assert target_after["indexed_points_count"] == 2
    assert target_after["needs_sync"] is False
    assert target_after["sync_status"] == "synced"

    # 7. Verify AI Chat retrieval finds the newly synced context
    search_results = search_stage2_points(
        query="telemetry pipeline APAC expansion",
        user_id=user_id,
        meeting_id=recording_id,
        max_results=5,
    )
    assert len(search_results) > 0
    top_doc = search_results[0].get("_text", "")
    assert "telemetry pipeline into APAC region" in top_doc
    assert search_results[0].get("meeting_name") == meeting_name

    # 8. Verify prompt building includes the meeting discussion point
    prompt = build_answer_prompt(
        question="What was decided about the telemetry pipeline?",
        context_results=search_results,
        chat_history=[],
    )
    assert "telemetry pipeline into APAC region" in prompt
    assert f"--- {meeting_name} ---" in prompt

    # Clean up test data
    store.delete_by_filter("meeting_id", recording_id)
    async with get_db_context() as session:
        await session.execute(text("DELETE FROM recordings WHERE id = :id"), {"id": recording_id})
        await session.commit()
