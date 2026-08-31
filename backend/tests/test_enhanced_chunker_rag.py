"""
test_enhanced_chunker_rag.py — Comprehensive unit tests for enhanced document chunking & hybrid ChromaDB retrieval.
"""

import pytest
import numpy as np
from services.text_chunker import chunk_document
from services.vector_store import VectorStore


def test_acronym_bidirectional_mapping():
    text = """
    # Section 1. Overview
    We are implementing Configuration Management (CM) for the avionics system.
    The System Requirements Specification (SRS) dictates compliance with DO-178C.
    CM protocols ensure full traceability.
    """
    chunks = chunk_document(text, filename="avionics_spec.pdf")
    assert len(chunks) > 0
    c = chunks[0]

    # Verify acronym mappings
    mappings = c.get("acronym_mappings", {})
    assert "CM" in mappings
    assert mappings["CM"] == "Configuration Management"
    assert mappings["Configuration Management"] == "CM"

    # Verify expanded acronyms list
    acrs = c.get("acronyms", [])
    assert "CM" in acrs
    assert "Configuration Management" in acrs
    assert "SRS" in acrs


def test_generic_identifier_extraction():
    text = """
    Requirement REQ-1049 states that flight module PN-99482-A must pass thermal tests.
    Bug ticket JIRA-8841 was closed under DOC-2024-991.
    Hardware version v2.4.1 conforms to ISO 9001.
    """
    chunks = chunk_document(text, filename="requirements.pdf")
    assert len(chunks) > 0
    all_identifiers = [ident for c in chunks for ident in c.get("identifiers", [])]

    assert any("REQ-1049" in i for i in all_identifiers)
    assert any("PN-99482-A" in i for i in all_identifiers)
    assert any("JIRA-8841" in i for i in all_identifiers)
    assert any("DOC-2024-991" in i for i in all_identifiers)
    assert any("v2.4.1" in i for i in all_identifiers)


def test_search_terms_consolidation_and_normalization():
    text = """
    The Neural Network Architecture processes real-time radar data for Project Apollo.
    System Requirements Specification (SRS) defines REQ-201.
    """
    chunks = chunk_document(text, filename="architecture.pdf")
    assert len(chunks) > 0
    c = chunks[0]

    s_terms = c.get("search_terms", [])
    assert len(s_terms) > 0

    # Case-normalized terms
    assert "srs" in s_terms
    assert "project apollo" in s_terms
    assert "req-201" in s_terms or "req 201" in s_terms or "req201" in s_terms


test_contextual_embedding_string_data = """
# Chapter 1. Introduction
## Section 1.2 Avionics Interface
This chunk defines the primary bus communication parameters.
"""

def test_contextual_embedding_string():
    chunks = chunk_document(test_contextual_embedding_string_data, filename="system_doc.pdf")
    assert len(chunks) > 0
    all_emb_ctx = "\n".join(c.get("embedding_context", "") for c in chunks)
    assert "[Document: system_doc.pdf" in all_emb_ctx
    assert "Chapter 1. Introduction" in all_emb_ctx
    assert "primary bus communication parameters" in all_emb_ctx


def test_neighbor_chunk_indices_and_ids():
    text = "\n\n".join([f"## Section {i}\nParagraph text for section {i} containing important information about system setup and maintenance." for i in range(1, 6)])
    chunks = chunk_document(text, filename="multi_section.pdf", min_chunk_words=10, max_chunk_words=30)
    assert len(chunks) >= 2

    # Check neighbor indices and IDs on second chunk
    c1 = chunks[1]
    assert c1["chunk_index"] == 1
    assert c1["previous_chunk_index"] == 0
    assert c1["next_chunk_index"] == 2
    assert c1["previous_chunk_id"] is not None
    assert c1["next_chunk_id"] is not None


def test_vector_store_search_hybrid(tmp_path):
    store = VectorStore(collection_name="test_hybrid_store", dim=128)
    store.load_or_create()

    # Clear any previous runs
    store.delete_by_filter("doc_id", "doc1")

    texts = [
        "Configuration Management (CM) procedures for avionics system PN-99482-A.",
        "General meeting discussion on office supplies and cafeteria menu.",
    ]
    metadatas = [
        {
            "doc_id": "doc1",
            "chunk_index": 0,
            "filename": "spec.pdf",
            "identifiers": ["PN-99482-A", "CM"],
            "search_terms": ["configuration management", "cm", "pn-99482-a", "avionics"],
            "acronyms": ["CM", "Configuration Management"],
            "project_names": ["Avionics Project"],
            "previous_chunk_index": None,
            "next_chunk_index": 1,
        },
        {
            "doc_id": "doc1",
            "chunk_index": 1,
            "filename": "spec.pdf",
            "search_terms": ["cafeteria", "supplies"],
            "previous_chunk_index": 0,
            "next_chunk_index": None,
        },
    ]

    np.random.seed(42)
    embeddings = np.random.randn(2, 128).astype(np.float32)

    store.add(texts, metadatas, embeddings=embeddings)

    # Perform hybrid search for query containing requirement ID and acronym
    query_vec = embeddings[0]
    results = store.search_hybrid(query_vec, query_text="Searching for PN-99482-A CM avionics", k=2, expand_neighbors=True)

    assert len(results) > 0
    top = results[0]
    assert top["doc_id"] == "doc1"
    assert top["chunk_index"] == 0
    assert top["boost"] > 0.0
    assert "expanded_text" in top
