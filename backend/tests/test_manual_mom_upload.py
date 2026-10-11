"""
test_manual_mom_upload.py — Comprehensive unit and integration tests for Manual MoM Upload & AI Chat Retrieval.
"""
import os
import sys
import uuid
import pytest
import asyncio
from unittest.mock import MagicMock, patch

# Ensure backend directory is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from database import connect_db, get_db_context, text
from services.manual_mom_service import (
    extract_metadata_and_sections,
    chunk_manual_mom,
    process_manual_mom_file,
    list_manual_moms,
    get_manual_mom_detail,
    delete_manual_mom,
    reindex_manual_mom,
)
from services.ai_chat_service import (
    format_context,
    extract_meeting_sources,
    get_direct_meeting_context,
    search_meeting_points,
)


SAMPLE_MOM_TEXT = """\
Meeting Title: Q4 Product Strategy & Architecture Sync
Date: 2026-10-08
Attendees: John Doe, Jane Smith, Alice Wong, Bob Vance

Executive Summary:
The team convened to review Q4 product roadmap goals, finalize the microservices migration timeline, and allocate cloud budget limits.

## Agenda 1: Cloud Architecture Migration
Jane Smith presented the transition plan from monolithic services to event-driven microservices on Kubernetes.
The team agreed that Phase 1 should begin with the user authentication and notification pipelines.
Performance benchmarks indicate a 35% latency drop once the service mesh is configured.

## Agenda 2: Security & Compliance
Bob Vance reviewed SOC 2 audit readiness requirements.
All API endpoints must enforce TLS 1.3 and require JWT claims validation.
Penetration testing is scheduled for late November.

## Key Decisions
• Transition to Kubernetes microservices starts on November 1st.
• All microservice communications must use gRPC with mTLS.
• Approved additional budget of $15,000 for cloud infrastructure testing.

## Action Items
• Complete architecture diagram for microservices (@Jane Smith) (Due: 2026-10-15)
• Implement JWT validation middleware (@Alice Wong) (Due: 2026-10-20)
• Schedule vendor security penetration test (@Bob Vance) (Due: 2026-10-25)
"""


import numpy as np


@pytest.fixture(autouse=True)
def mock_embedder():
    """Mock text embedder to return deterministic 1024-dim vectors instantly without loading ML weights."""
    mock_emb = MagicMock()
    mock_emb.embedding_dim.return_value = 1024
    mock_emb._dim = 1024
    mock_emb.load = MagicMock()

    def _encode(text_str):
        if isinstance(text_str, list):
            return [_encode(t) for t in text_str]
        v = np.zeros(1024, dtype=np.float32)
        v[0] = 1.0
        v[1] = len(text_str) / 1000.0
        norm = np.linalg.norm(v)
        return (v / norm).tolist()

    mock_emb.encode.side_effect = _encode

    with patch("services.text_embedding_service.get_text_embedder", return_value=mock_emb), \
         patch("services.ai_chat_service._get_embedder", return_value=mock_emb):
        yield mock_emb


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_metadata_extraction():
    """Verify that title, date, attendees, agendas, action items, and decisions are accurately extracted."""
    meta = extract_metadata_and_sections(SAMPLE_MOM_TEXT, "q4_strategy_sync.txt")

    assert meta["title"] == "Q4 Product Strategy & Architecture Sync"
    assert meta["meeting_date"] == "2026-10-08"
    assert "John Doe" in meta["speakers"]
    assert "Jane Smith" in meta["speakers"]
    assert "Alice Wong" in meta["speakers"]
    assert len(meta["agendas"]) >= 2

    # Check action items
    assert len(meta["action_items"]) == 3
    owners = [a["owner"] for a in meta["action_items"]]
    assert "Jane Smith" in owners
    assert "Alice Wong" in owners

    # Check decisions
    assert len(meta["decisions"]) >= 3
    assert any("Kubernetes" in d for d in meta["decisions"])

    # Check summary
    assert "roadmap" in meta["summary"].lower()


def test_chunking_strategy():
    """Verify semantic chunking splits the MoM into structured, tagged segments."""
    meta = extract_metadata_and_sections(SAMPLE_MOM_TEXT, "test.txt")
    chunks = chunk_manual_mom(SAMPLE_MOM_TEXT, meta)

    assert len(chunks) >= 4
    agenda_titles = [c.get("agenda_title") for c in chunks]
    assert "Executive Summary" in agenda_titles
    assert "Key Decisions" in agenda_titles
    assert "Action Items" in agenda_titles

    # Check chunk structure
    for c in chunks:
        assert "text" in c
        assert "chunk_index" in c
        assert len(c["text"]) > 10


@pytest.mark.asyncio
async def test_manual_mom_lifecycle():
    """Test full upload, list, detail, retrieval, and delete lifecycle."""
    await connect_db()
    test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"

    # 1. Process upload
    file_bytes = SAMPLE_MOM_TEXT.encode("utf-8")
    result = await process_manual_mom_file(
        user_id=test_user_id,
        file_bytes=file_bytes,
        filename="Q4_Strategy_Sync.txt",
    )

    mom_id = result["id"]
    assert mom_id.startswith("mom_")
    assert result["title"] == "Q4 Product Strategy & Architecture Sync"
    assert result["chunks_count"] >= 4

    # 2. List manual MoMs
    moms = await list_manual_moms(test_user_id)
    assert len(moms) == 1
    assert moms[0]["id"] == mom_id
    assert moms[0]["action_items_count"] == 3

    # 3. Get detail
    detail = await get_manual_mom_detail(test_user_id, mom_id)
    assert detail is not None
    assert detail["title"] == "Q4 Product Strategy & Architecture Sync"
    assert len(detail["chunks"]) >= 4
    assert len(detail["action_items"]) == 3

    # 4. Direct retrieval
    direct_pts = get_direct_meeting_context(
        user_id=test_user_id,
        meeting_id=mom_id,
    )
    assert len(direct_pts) >= 4
    assert direct_pts[0]["source"] == "manual_mom"
    assert direct_pts[0]["is_manual_mom"] is True

    # 5. Format Context
    formatted = format_context(direct_pts)
    assert "[Manual MoM]" in formatted
    assert "Q4 Product Strategy & Architecture Sync" in formatted

    # 6. Extract meeting sources
    sources = extract_meeting_sources(direct_pts)
    assert len(sources) == 1
    assert sources[0]["id"] == mom_id
    assert sources[0]["is_manual_mom"] is True

    # 7. Re-index
    reindex_res = await reindex_manual_mom(test_user_id, mom_id)
    assert reindex_res["status"] == "success"
    assert reindex_res["chunks_indexed"] >= 4

    # 8. Delete
    deleted = await delete_manual_mom(test_user_id, mom_id)
    assert deleted is True

    # Verify deleted
    detail_after = await get_manual_mom_detail(test_user_id, mom_id)
    assert detail_after is None
    moms_after = await list_manual_moms(test_user_id)
    assert len(moms_after) == 0


@pytest.mark.asyncio
async def test_manual_mom_router_and_available_meetings():
    """Verify that uploaded MoMs appear in available meetings with is_manual_mom: True."""
    from routers.ai_chat_router import _load_user_available_meetings

    await connect_db()
    test_user_id = f"test_user_meetings_{uuid.uuid4().hex[:8]}"

    # Process 2 different files (multi-file upload test)
    res1 = await process_manual_mom_file(
        user_id=test_user_id,
        file_bytes=b"Meeting Title: Sprint 42 Planning\nDate: 2026-10-01\nAttendees: Alice, Bob\n## Discussion\nDiscussed sprint backlog.",
        filename="sprint_42_planning.txt",
    )
    res2 = await process_manual_mom_file(
        user_id=test_user_id,
        file_bytes=b"Meeting Title: Design System Review\nDate: 2026-10-05\nAttendees: Charlie\n## Notes\nReviewed color tokens.",
        filename="design_review.md",
    )

    # Load meetings for user
    available = await _load_user_available_meetings(test_user_id)
    assert len(available) >= 2

    m1 = next((m for m in available if m["id"] == res1["id"]), None)
    m2 = next((m for m in available if m["id"] == res2["id"]), None)

    assert m1 is not None
    assert m1["name"] == "Sprint 42 Planning"
    assert m1["is_manual_mom"] is True
    assert m1["has_final_rom"] is True

    assert m2 is not None
    assert m2["name"] == "Design System Review"
    assert m2["is_manual_mom"] is True
    assert m2["has_final_rom"] is True

    # Cleanup
    await delete_manual_mom(test_user_id, res1["id"])
    await delete_manual_mom(test_user_id, res2["id"])

