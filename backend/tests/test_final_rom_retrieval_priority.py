import os
import sys
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import connect_db, get_db_context
from services.vector_store import get_stage2_points_store, get_final_rom_points_store
from services.rom_service import rom_service
from services.ai_chat_service import (
    is_detailed_query,
    get_direct_meeting_context,
    search_meeting_points,
    get_final_rom_meeting_ids,
    get_stage2_meeting_ids,
)
from routers.ai_chat_router import (
    _safe_parse_rom_data,
    _extract_stage2_points,
    _extract_final_rom_points,
    get_syncable_meetings,
    resync_meeting,
)
from sqlalchemy import text


@pytest.fixture(autouse=True)
def mock_embedder():
    """Mock text embedder to return deterministic 1024-dim vectors without loading LLM."""
    mock_emb = MagicMock()
    mock_emb.embedding_dim.return_value = 1024
    mock_emb._dim = 1024
    mock_emb.load = MagicMock()

    def _encode(text_str):
        # Deterministic vector based on text length
        v = np.zeros(1024, dtype=np.float32)
        v[0] = 1.0
        v[1] = len(text_str) / 1000.0
        norm = np.linalg.norm(v)
        return (v / norm).tolist()

    def _encode_batch(texts):
        return [_encode(t) for t in texts]

    mock_emb.encode.side_effect = _encode
    mock_emb.encode_batch.side_effect = _encode_batch

    with patch("services.text_embedding_service.get_text_embedder", return_value=mock_emb), \
         patch("services.ai_chat_service._get_embedder", return_value=mock_emb):
        yield mock_emb


def test_is_detailed_query_detection():
    assert is_detailed_query("give me the detailed points from the meeting") is True
    assert is_detailed_query("deep dive into discussion points") is True
    assert is_detailed_query("show granular stage 2 points") is True
    assert is_detailed_query("what were the stage 2 raw points?") is True
    assert is_detailed_query("summarize the meeting") is False
    assert is_detailed_query("what were the action items?") is False
    assert is_detailed_query("who attended the call?") is False


def test_extract_final_rom_points():
    rom_dict = {
        "final_rom": {
            "agendas": [
                {
                    "agenda_id": "ag_1",
                    "title": "Cloud Architecture",
                    "discussion_points": [
                        {
                            "id": "fp_1",
                            "text": "Migrate data ingestion to AWS Kinesis.",
                            "speakers": ["Alex"],
                            "action_owner": "Alex",
                        },
                        {
                            "id": "fp_2",
                            "text": "Establish secondary disaster recovery in Frankfurt.",
                            "speakers": ["Jane"],
                            "action_owner": "Jane",
                        },
                    ],
                }
            ]
        }
    }

    pts = _extract_final_rom_points(rom_dict)
    assert len(pts) == 2
    assert pts[0]["agenda_title"] == "Cloud Architecture"
    assert pts[0]["text"] == "Migrate data ingestion to AWS Kinesis."
    assert pts[1]["action_owner"] == "Jane"

    # Empty cases
    assert _extract_final_rom_points({}) == []
    assert _extract_final_rom_points(None) == []


@pytest.mark.asyncio
async def test_final_rom_priority_retrieval():
    await connect_db()

    user_id = "test_user_priority_e2e"
    recording_id = "rec_priority_test_001"
    meeting_name = "Sprint Planning Priority Test"
    meeting_date = "2026-10-08"

    # Stage 2 points
    s2_points = [
        {
            "id": "s2_p1",
            "polished_text": "Detailed raw debate regarding Cassandra cluster sizing and memory allocation.",
            "speakers": ["Engineer Bob"],
            "action_owner": "Bob",
        }
    ]

    # Final ROM points
    final_rom_obj = {
        "agendas": [
            {
                "agenda_id": "ag_db",
                "title": "Database Sizing",
                "discussion_points": [
                    {
                        "id": "final_p1",
                        "text": "Final consensus to provision a 6-node Cassandra cluster with 64GB RAM.",
                        "speakers": ["Architect Sam"],
                        "action_owner": "Sam",
                    }
                ],
            }
        ]
    }

    # Clean prior collections
    final_store = get_final_rom_points_store(user_id, 1024)
    final_store.load_or_create()
    final_store.delete_by_filter("meeting_id", recording_id)

    s2_store = get_stage2_points_store(user_id, 1024)
    s2_store.load_or_create()
    s2_store.delete_by_filter("meeting_id", recording_id)

    # 1. Index both Stage 2 and Final ROM
    rom_service.index_stage2_points_in_chromadb(
        recording_id=recording_id,
        user_id=user_id,
        points=s2_points,
        meeting_name=meeting_name,
        meeting_date=meeting_date,
        only_missing=False,
    )

    rom_service.index_final_rom_points_in_chromadb(
        recording_id=recording_id,
        user_id=user_id,
        final_rom=final_rom_obj,
        meeting_name=meeting_name,
        meeting_date=meeting_date,
        only_missing=False,
    )

    # 2. Priority 1 Test: Standard direct query without is_detailed MUST return Final ROM
    results_std = get_direct_meeting_context(user_id, recording_id, is_detailed=False)
    assert len(results_std) > 0
    assert results_std[0]["source"] == "final_roim"
    assert "Cassandra cluster with 64GB RAM" in results_std[0]["clean_text"]

    # 3. Priority Override Test: Detailed direct query MUST return Stage 2
    results_detailed = get_direct_meeting_context(user_id, recording_id, is_detailed=True)
    assert len(results_detailed) > 0
    assert results_detailed[0]["source"] == "stage2"
    assert "Detailed raw debate" in results_detailed[0]["clean_text"]

    # 4. Search Priority Test: Semantic search across meetings
    search_std = search_meeting_points("Cassandra cluster", user_id=user_id, is_detailed=False)
    assert len(search_std) > 0
    # Because Final ROM exists for this meeting, Stage 2 points should NOT be included
    sources = [r.get("source") for r in search_std if r.get("meeting_id") == recording_id]
    assert all(s == "final_roim" for s in sources)

    # Clean up
    final_store.delete_by_filter("meeting_id", recording_id)
    s2_store.delete_by_filter("meeting_id", recording_id)


@pytest.mark.asyncio
async def test_resync_meeting_dual_collections():
    await connect_db()

    user_id = "test_user_resync_dual"
    recording_id = "rec_resync_dual_002"
    meeting_name = "Dual Sync Meeting"

    s2_points = [
        {"id": "pt_s2_1", "polished_text": "Stage 2 discussion point for dual sync."},
    ]
    final_rom = {
        "agendas": [
            {
                "agenda_id": "ag_1",
                "title": "General",
                "discussion_points": [
                    {"id": "pt_final_1", "text": "Final ROM discussion point for dual sync."},
                ],
            }
        ]
    }

    rom_data = {
        "stage2": {"polished_points": s2_points},
        "final_rom": final_rom,
    }

    # Clean prior collections
    final_store = get_final_rom_points_store(user_id, 1024)
    final_store.load_or_create()
    final_store.delete_by_filter("meeting_id", recording_id)

    s2_store = get_stage2_points_store(user_id, 1024)
    s2_store.load_or_create()
    s2_store.delete_by_filter("meeting_id", recording_id)

    # Insert test user and recording into DB
    async with get_db_context() as session:
        await session.execute(
            text("INSERT OR IGNORE INTO users (id, name, email, hashed_password) VALUES (:id, :n, :e, :p)"),
            {"id": user_id, "n": "Test Dual", "e": f"{user_id}@test.com", "p": "hash"},
        )
        await session.execute(
            text("DELETE FROM recordings WHERE id = :id"),
            {"id": recording_id},
        )
        await session.execute(
            text("""
                INSERT INTO recordings (id, user_id, filename, title, file_path, duration, status, created_at, rom_data)
                VALUES (:id, :uid, :fn, :title, 'test.mp4', 60.0, 'completed', '2026-10-08T00:00:00Z', :rd)
            """),
            {
                "id": recording_id,
                "uid": user_id,
                "fn": "dual_test.mp4",
                "title": meeting_name,
                "rd": json.dumps(rom_data),
            },
        )
        await session.commit()

    # 1. Call syncable meetings endpoint
    user_ctx = {"id": user_id}
    syncables = await get_syncable_meetings(user=user_ctx)
    target = next((m for m in syncables if m["id"] == recording_id), None)
    assert target is not None
    assert target["stage2_points_count"] == 1
    assert target["final_rom_points_count"] == 1
    assert target["stage2_sync_status"] == "not_indexed"
    assert target["final_rom_sync_status"] == "not_indexed"
    assert target["needs_sync"] is True

    # 2. Call resync_meeting
    res = await resync_meeting(recording_id=recording_id, force=False, user=user_ctx)
    assert res["status"] == "success"
    assert res["total_found"] == 2
    assert res["newly_indexed"] == 2
    assert res["final_rom_result"]["newly_indexed"] == 1
    assert res["stage2_result"]["newly_indexed"] == 1

    # 3. Call resync_meeting second time with force=False -> duplicate prevention
    res_second = await resync_meeting(recording_id=recording_id, force=False, user=user_ctx)
    assert res_second["status"] == "success"
    assert res_second["newly_indexed"] == 0
    assert res_second["already_indexed"] == 2

    # 4. Check syncable meetings after sync -> should be synced
    syncables_after = await get_syncable_meetings(user=user_ctx)
    target_after = next((m for m in syncables_after if m["id"] == recording_id), None)
    assert target_after is not None
    assert target_after["stage2_sync_status"] == "synced"
    assert target_after["final_rom_sync_status"] == "synced"
    assert target_after["needs_sync"] is False

    # Cleanup DB and vector stores
    async with get_db_context() as session:
        await session.execute(text("DELETE FROM recordings WHERE id = :id"), {"id": recording_id})
        await session.commit()
    final_store.delete_by_filter("meeting_id", recording_id)
    s2_store.delete_by_filter("meeting_id", recording_id)
