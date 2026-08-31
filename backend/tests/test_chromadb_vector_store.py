"""
tests/test_chromadb_vector_store.py

Comprehensive test suite for ChromaDB-backed vector storage, rich metadata extraction,
and metadata-aware retrieval.
"""
from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import pytest

# Ensure backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.vector_store import (
    VectorStore,
    get_global_context_store,
    get_meeting_context_store,
    get_transcript_store,
    _sanitize_collection_name,
)
from services.text_chunker import chunk_document


# ══════════════════════════════════════════════════════════════════
# Collection Name Sanitizer Tests
# ══════════════════════════════════════════════════════════════════

class TestCollectionNameSanitizer:
    def test_valid_name_unchanged(self):
        assert _sanitize_collection_name("global_context_user1") == "global_context_user1"

    def test_special_chars_replaced(self):
        name = _sanitize_collection_name("user@domain.com/recording:123")
        assert "@" not in name
        assert ":" not in name
        assert "/" not in name
        assert len(name) >= 3

    def test_short_name_padded(self):
        assert len(_sanitize_collection_name("a")) >= 3

    def test_long_name_truncated(self):
        long_name = "a" * 100
        assert len(_sanitize_collection_name(long_name)) <= 63


# ══════════════════════════════════════════════════════════════════
# Rich Chunk Metadata Extraction Tests
# ══════════════════════════════════════════════════════════════════

class TestRichChunkMetadata:
    def test_rich_metadata_fields_extracted(self):
        sample_doc = """
        # Phoenix Project Architecture
        
        ## 1.1 Hardware Specifications
        The Phoenix Project kickoff was held on January 15, 2024.
        Dr. Smith from Acme Corp presented the system design.
        The device operates at 3.5 GHz with 16 GB RAM and 99.9% uptime requirement.
        
        | Parameter | Value |
        |-----------|-------|
        | Version   | v2.1  |
        | Standard  | ISO-26262 |
        """
        chunks = chunk_document(
            sample_doc,
            filename="phoenix_spec.md",
            doc_scope="global_context",
        )
        assert len(chunks) > 0
        c0 = chunks[0]

        # Basic metadata
        assert c0["document_name"] == "phoenix_spec.md"
        assert c0["scope"] == "global_context"
        assert c0["context_type"] == "global"
        assert "total_chunks" in c0
        assert c0["total_chunks"] == len(chunks)

        # Extracted entities and dates across all chunks
        all_dates = [d for c in chunks for d in c.get("dates", [])]
        all_projects = [p for c in chunks for p in c.get("project_names", [])]
        all_entities = [e for c in chunks for e in c.get("entities", [])]
        all_numbers = [n for c in chunks for n in c.get("numbers", [])]
        all_tech = [t for c in chunks for t in c.get("technical_entities", [])]

        assert any("January 15, 2024" in d for d in all_dates)
        assert any("Phoenix Project" in p for p in all_projects)
        assert any("Acme Corp" in e or "Dr. Smith" in e for e in all_entities)
        assert any("3.5 GHz" in n or "99.9%" in n for n in all_numbers)
        assert any("ISO-26262" in t or "v2.1" in t for t in all_tech)
        assert len(c0.get("important_terms", [])) > 0


# ══════════════════════════════════════════════════════════════════
# ChromaDB VectorStore CRUD & Search Tests
# ══════════════════════════════════════════════════════════════════

class TestChromaVectorStoreCRUD:
    @pytest.fixture(autouse=True)
    def setup_store(self, tmp_path):
        os.environ["CHROMADB_DIR"] = str(tmp_path / "chromadb")
        self.store = VectorStore(collection_name="test_collection_crud", dim=128)
        self.store.load_or_create()
        yield
        self.store.delete_store()

    def test_add_and_count(self):
        texts = ["First chunk of text about signals", "Second chunk about software"]
        metas = [
            {"doc_id": "d1", "chunk_index": 0, "source": "test", "keywords": ["signal"]},
            {"doc_id": "d1", "chunk_index": 1, "source": "test", "keywords": ["software"]},
        ]
        vecs = np.random.randn(2, 128).astype(np.float32)
        added = self.store.add(texts, metas, embeddings=vecs)
        assert added == 2
        assert self.store.total_vectors() == 2
        assert self.store.exists()

    def test_search_returns_results(self):
        texts = ["Deep learning neural network training", "Database SQL query optimization"]
        metas = [
            {"doc_id": "d1", "chunk_index": 0, "source": "test", "project_names": ["AI Engine"]},
            {"doc_id": "d2", "chunk_index": 0, "source": "test", "project_names": ["DB Platform"]},
        ]
        # Create distinct normalized vectors
        v1 = np.zeros(128, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(128, dtype=np.float32)
        v2[1] = 1.0

        vecs = np.vstack([v1, v2])
        self.store.add(texts, metas, embeddings=vecs)

        # Query close to v1
        results = self.store.search(v1, k=2)
        assert len(results) == 2
        assert results[0]["_text"] == texts[0]
        assert results[0]["score"] > results[1]["score"]
        assert "project_names" in results[0]

    def test_delete_by_filter(self):
        texts = ["Doc 1 chunk", "Doc 2 chunk"]
        metas = [
            {"doc_id": "doc_alpha", "chunk_index": 0, "source": "test"},
            {"doc_id": "doc_beta", "chunk_index": 0, "source": "test"},
        ]
        vecs = np.random.randn(2, 128).astype(np.float32)
        self.store.add(texts, metas, embeddings=vecs)

        assert self.store.total_vectors() == 2
        removed = self.store.delete_by_filter("doc_id", "doc_alpha")
        assert removed == 1
        assert self.store.total_vectors() == 1

    def test_clear_collection(self):
        texts = ["Chunk 1", "Chunk 2"]
        metas = [{"doc_id": "d1", "chunk_index": i, "source": "test"} for i in range(2)]
        vecs = np.random.randn(2, 128).astype(np.float32)
        self.store.add(texts, metas, embeddings=vecs)

        assert self.store.total_vectors() == 2
        self.store.clear()
        assert self.store.total_vectors() == 0


# ══════════════════════════════════════════════════════════════════
# Store Helper Factory Isolation Tests
# ══════════════════════════════════════════════════════════════════

class TestStoreHelpers:
    def test_scope_isolation(self, tmp_path):
        os.environ["CHROMADB_DIR"] = str(tmp_path / "chromadb")

        g_store = get_global_context_store("user_123", dim=64)
        m_store = get_meeting_context_store("rec_456", dim=64)
        t_store = get_transcript_store("rec_456", dim=64)

        # Add data to global store
        vec = np.random.randn(1, 64).astype(np.float32)
        g_store.add(["Global doc"], [{"doc_id": "g1", "chunk_index": 0, "source": "global"}], embeddings=vec)

        assert g_store.total_vectors() == 1
        assert m_store.total_vectors() == 0
        assert t_store.total_vectors() == 0

        # Clean up
        g_store.delete_store()
        m_store.delete_store()
        t_store.delete_store()
