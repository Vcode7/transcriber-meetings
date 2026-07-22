import pytest
from unittest.mock import patch, MagicMock
from services.rag_pipeline import retrieve_transcript_hybrid

def test_retrieve_transcript_hybrid_excludes_timeline_duplicates():
    """
    Verify that retrieve_transcript_hybrid filters out semantic chunks
    that are already present in the agenda timeline window.
    """
    # Parameters for the test:
    recording_id = "test_recording"
    agenda_index = 0
    total_agendas = 3
    recording_duration = 300.0  # Slot size = 100.0s. Win for agenda 0: [0, 110s] with stride 10
    k_transcript = 5
    high_confidence_threshold = 0.70
    timeline_stride = 10.0
    relative_cutoff = 0.5  # Large enough so chunks with score 0.75 are not filtered out

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.embedding_dim.return_value = 384
    
    # Mock transcript store metadata
    # The store metadata defines what chunks exist in the timeline window
    mock_meta = [
        {"chunk_index": 0, "start": 10.0, "end": 25.0, "_text": "Chunk 0 - inside timeline window", "speakers": ["Speaker A"]},
        {"chunk_index": 1, "start": 95.0, "end": 105.0, "_text": "Chunk 1 - inside timeline window", "speakers": ["Speaker B"]},
        {"chunk_index": 2, "start": 200.0, "end": 215.0, "_text": "Chunk 2 - outside timeline window", "speakers": ["Speaker C"]},
    ]

    # Mock search results from FAISS
    # FAISS search returns chunks 0, 1, and 2
    mock_search_results = [
        {"chunk_index": 0, "start": 10.0, "end": 25.0, "_text": "Chunk 0 - inside timeline window", "speakers": ["Speaker A"], "score": 0.85},
        {"chunk_index": 1, "start": 95.0, "end": 105.0, "_text": "Chunk 1 - inside timeline window", "speakers": ["Speaker B"], "score": 0.90},
        {"chunk_index": 2, "start": 200.0, "end": 215.0, "_text": "Chunk 2 - outside timeline window", "speakers": ["Speaker C"], "score": 0.75},
    ]

    mock_store = MagicMock()
    mock_store.exists.return_value = True
    mock_store._meta = mock_meta
    mock_store.search.return_value = mock_search_results

    with patch("services.rag_pipeline._get_embedder", return_value=mock_embedder), \
         patch("services.vector_store.get_transcript_store", return_value=mock_store):
        
        # Run retrieval
        results = retrieve_transcript_hybrid(
            recording_id=recording_id,
            query_embedding=[0.1] * 384,
            agenda_index=agenda_index,
            total_agendas=total_agendas,
            recording_duration=recording_duration,
            k_transcript=k_transcript,
            high_confidence_threshold=high_confidence_threshold,
            timeline_stride=timeline_stride,
            relative_cutoff=relative_cutoff,
        )

        # Assertions
        # 1. Chunks 0 and 1 are in the timeline window [0s, 110s], so they must be excluded.
        # 2. Only Chunk 2 (outside timeline window) should be included.
        assert len(results) == 1
        assert results[0]["chunk_index"] == 2
        assert results[0]["text"] == "Chunk 2 - outside timeline window"
        assert results[0]["score"] == 0.75


def test_retrieve_transcript_hybrid_with_timeline_disabled():
    """
    Verify that when retrieve_by_timeline is False, retrieve_transcript_hybrid
    does not include timeline window chunks and returns all matching semantic chunks.
    """
    # Parameters for the test:
    recording_id = "test_recording"
    agenda_index = 0
    total_agendas = 3
    recording_duration = 300.0
    k_transcript = 5
    high_confidence_threshold = 0.70
    timeline_stride = 10.0
    relative_cutoff = 0.5

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.embedding_dim.return_value = 384
    
    mock_meta = [
        {"chunk_index": 0, "start": 10.0, "end": 25.0, "_text": "Chunk 0", "speakers": ["Speaker A"]},
        {"chunk_index": 1, "start": 95.0, "end": 105.0, "_text": "Chunk 1", "speakers": ["Speaker B"]},
        {"chunk_index": 2, "start": 200.0, "end": 215.0, "_text": "Chunk 2", "speakers": ["Speaker C"]},
    ]

    mock_search_results = [
        {"chunk_index": 0, "start": 10.0, "end": 25.0, "_text": "Chunk 0", "speakers": ["Speaker A"], "score": 0.85},
        {"chunk_index": 1, "start": 95.0, "end": 105.0, "_text": "Chunk 1", "speakers": ["Speaker B"], "score": 0.90},
        {"chunk_index": 2, "start": 200.0, "end": 215.0, "_text": "Chunk 2", "speakers": ["Speaker C"], "score": 0.75},
    ]

    mock_store = MagicMock()
    mock_store.exists.return_value = True
    mock_store._meta = mock_meta
    mock_store.search.return_value = mock_search_results

    with patch("services.rag_pipeline._get_embedder", return_value=mock_embedder), \
         patch("services.vector_store.get_transcript_store", return_value=mock_store):
        
        # Run retrieval with retrieve_by_timeline = False
        results = retrieve_transcript_hybrid(
            recording_id=recording_id,
            query_embedding=[0.1] * 384,
            agenda_index=agenda_index,
            total_agendas=total_agendas,
            recording_duration=recording_duration,
            k_transcript=k_transcript,
            high_confidence_threshold=high_confidence_threshold,
            timeline_stride=timeline_stride,
            relative_cutoff=relative_cutoff,
            retrieve_by_timeline=False,
        )

        # Assertions
        # Since retrieve_by_timeline is False, we completely skip timeline chunks scan.
        # So we should get all three chunks from semantic search (no deduplication because timeline_chunks_by_index is empty).
        assert len(results) == 3
        # Sorted chronologically by start time: Chunk 0 (10s), Chunk 1 (95s), Chunk 2 (200s)
        assert results[0]["chunk_index"] == 0
        assert results[1]["chunk_index"] == 1
        assert results[2]["chunk_index"] == 2


def test_retrieve_transcript_hybrid_enforces_k_transcript_limit():
    """
    Verify that retrieve_transcript_hybrid strictly caps the returned semantic chunks
    to k_transcript even when more high-confidence chunks match.
    """
    from services.rag_pipeline import retrieve_transcript_hybrid

    mock_embedder = MagicMock()
    mock_embedder.embedding_dim.return_value = 384

    # 5 matching search results all with score >= 0.70
    mock_search_results = [
        {"chunk_index": 0, "start": 10.0, "end": 20.0, "_text": "Chunk 0", "speakers": ["A"], "score": 0.95},
        {"chunk_index": 1, "start": 30.0, "end": 40.0, "_text": "Chunk 1", "speakers": ["B"], "score": 0.90},
        {"chunk_index": 2, "start": 50.0, "end": 60.0, "_text": "Chunk 2", "speakers": ["C"], "score": 0.85},
        {"chunk_index": 3, "start": 70.0, "end": 80.0, "_text": "Chunk 3", "speakers": ["D"], "score": 0.80},
        {"chunk_index": 4, "start": 90.0, "end": 100.0, "_text": "Chunk 4", "speakers": ["E"], "score": 0.75},
    ]

    mock_store = MagicMock()
    mock_store.exists.return_value = True
    mock_store._meta = []
    mock_store.search.return_value = mock_search_results

    with patch("services.rag_pipeline._get_embedder", return_value=mock_embedder), \
         patch("services.vector_store.get_transcript_store", return_value=mock_store):
        
        # Set k_transcript = 2
        results = retrieve_transcript_hybrid(
            recording_id="rec_test",
            query_embedding=[0.1] * 384,
            agenda_index=0,
            total_agendas=1,
            recording_duration=120.0,
            k_transcript=2,
            high_confidence_threshold=0.70,
            timeline_stride=10.0,
            relative_cutoff=0.5,
            retrieve_by_timeline=False,
        )

        # Must strictly be capped at k_transcript = 2 (highest scores 0.95 and 0.90)
        assert len(results) == 2
        chunk_indices = [r["chunk_index"] for r in results]
        assert set(chunk_indices) == {0, 1}


def test_retrieve_evidence_chunkwise_enforces_max_overlap_chunks():
    """
    Verify that retrieve_evidence_chunkwise caps chunk assignment to at most
    max_overlap_chunks agendas per transcript chunk.
    """
    from services.rag_pipeline import retrieve_evidence_chunkwise

    mock_embedder = MagicMock()
    mock_embedder.embedding_dim.return_value = 4
    mock_embedder.encode.side_effect = lambda text: [1.0, 0.0, 0.0, 0.0]

    mock_meta = [
        {"chunk_index": 0, "start": 0.0, "end": 10.0, "_text": "Chunk 0 matches all agendas", "speakers": ["A"]},
    ]

    mock_store = MagicMock()
    mock_store.exists.return_value = True
    mock_store._meta = mock_meta
    mock_store._index.reconstruct.return_value = [1.0, 0.0, 0.0, 0.0]

    agenda_items = [
        {"topic": "Agenda 1"},
        {"topic": "Agenda 2"},
        {"topic": "Agenda 3"},
        {"topic": "Agenda 4"},
    ]

    with patch("services.rag_pipeline._get_embedder", return_value=mock_embedder), \
         patch("services.vector_store.get_transcript_store", return_value=mock_store), \
         patch("services.vector_store.get_meeting_context_store") as mock_m, \
         patch("services.vector_store.get_global_context_store") as mock_g:
        mock_m.return_value.exists.return_value = False
        mock_g.return_value.exists.return_value = False

        # Set max_overlap_chunks = 2
        agendas = retrieve_evidence_chunkwise(
            recording_id="rec_test",
            user_id="user_test",
            agenda_items=agenda_items,
            k_global=0,
            k_meeting=0,
            k_transcript=10,
            relative_cutoff=0.5,
            max_overlap_chunks=2,
        )

        # Count total agendas that received Chunk 0
        agendas_with_chunk_0 = [a for a in agendas if len(a["transcript_chunks"]) > 0]
        assert len(agendas_with_chunk_0) == 2


def test_two_stage_collection_retrieval_and_fallback():
    """
    Verify two-stage retrieval:
    1. Stage 1 parses JSON triage with context_required=true and detail string.
    2. Stage 2 searches vector store using detail plan with max_context cap.
    """
    from services.collection_ai_service import retrieve_collection_context_two_stage

    mock_embedder = MagicMock()
    mock_embedder.embedding_dim.return_value = 4
    mock_embedder.encode.return_value = [0.1, 0.2, 0.3, 0.4]

    mock_provider = MagicMock()
    mock_provider._infer.return_value = '{"context_required": true, "detail": "avionics testing timeline delivery"}'

    mock_store = MagicMock()
    mock_store.exists.return_value = True
    mock_store.search.return_value = [
        {"_text": "Chunk 1: Avionics software test completed", "score": 0.9, "start": 0.0, "end": 10.0},
        {"_text": "Chunk 2: Delivery date confirmed", "score": 0.8, "start": 10.0, "end": 20.0},
        {"_text": "Chunk 3: QA status update", "score": 0.7, "start": 20.0, "end": 30.0},
    ]

    with patch("services.collection_ai_service._get_embedder", return_value=mock_embedder), \
         patch("services.ai_provider.get_provider", return_value=mock_provider), \
         patch("services.vector_store.get_transcript_store", return_value=mock_store), \
         patch("services.vector_store.get_global_context_store") as mock_g:
        mock_g.return_value.exists.return_value = False

        # Max context = 2 chunks
        ctx, detail, context_req = retrieve_collection_context_two_stage(
            meeting_ids=["m1"],
            meeting_meta={"m1": {"filename": "Meeting 1", "created_at": "2026-01-01"}},
            query="When is avionics testing complete?",
            user_id="u1",
            max_context=2,
        )

        assert context_req is True
        assert detail == "avionics testing timeline delivery"
        assert len(ctx.chunks) == 2
        assert ctx.chunks[0].text == "Chunk 1: Avionics software test completed"
        assert ctx.chunks[1].text == "Chunk 2: Delivery date confirmed"


def test_two_stage_no_context_required_direct_answer():
    """
    Verify Stage 1 triage bypasses RAG search when context_required=false:
    Direct answer detail is returned and 0 vector store chunks are retrieved.
    """
    from services.collection_ai_service import retrieve_collection_context_two_stage

    mock_provider = MagicMock()
    mock_provider._infer.return_value = '{"context_required": false, "detail": "Hello! I am your AI meeting assistant."}'

    with patch("services.ai_provider.get_provider", return_value=mock_provider), \
         patch("services.vector_store.get_transcript_store") as mock_store:

        ctx, detail, context_req = retrieve_collection_context_two_stage(
            meeting_ids=["m1"],
            meeting_meta={"m1": {"filename": "Meeting 1", "created_at": "2026-01-01"}},
            query="Hi, who are you?",
            user_id="u1",
        )

        assert context_req is False
        assert detail == "Hello! I am your AI meeting assistant."
        assert len(ctx.chunks) == 0
        # Verify vector store was NOT called
        mock_store.assert_not_called()


