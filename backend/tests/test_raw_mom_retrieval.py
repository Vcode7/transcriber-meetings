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
