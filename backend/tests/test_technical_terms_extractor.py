"""
test_technical_terms_extractor.py — Unit tests for multi-signal technical terminology extraction.
"""

import pytest
from services.text_chunker import chunk_document, _TechnicalTerminologyExtractor


def test_compound_technical_terms_detection():
    text = """
    # Section 2. AI Pipeline Architecture
    We deployed a diffusion model with image embeddings into our vector database.
    The fine-tuning pipeline uses PyTorch CUDA acceleration and ONNX Runtime.
    Cosine similarity and reciprocal rank fusion optimize the semantic retrieval layer.
    """
    chunks = chunk_document(text, filename="ai_architecture.pdf")
    assert len(chunks) > 0
    c = chunks[0]

    tech_terms = c.get("technical_terms", [])

    # Compound terms
    assert any("diffusion model" in t.lower() for t in tech_terms)
    assert any("image embeddings" in t.lower() for t in tech_terms)
    assert any("vector database" in t.lower() for t in tech_terms)
    assert any("fine-tuning pipeline" in t.lower() or "fine tuning pipeline" in t.lower() for t in tech_terms)
    assert any("reciprocal rank fusion" in t.lower() for t in tech_terms)
    assert any("semantic retrieval" in t.lower() for t in tech_terms)


def test_technology_and_standard_names():
    text = """
    Compliance with DO-178C and MIL-STD-810G is mandatory for flight software.
    Built with FastAPI, Uvicorn, ChromaDB, and PyTorch ONNX Runtime.
    """
    chunks = chunk_document(text, filename="compliance.pdf")
    assert len(chunks) > 0
    c = chunks[0]

    tech_terms = c.get("technical_terms", [])
    assert any("do-178c" in t.lower() for t in tech_terms)
    assert any("mil-std-810g" in t.lower() for t in tech_terms)
    assert any("chromadb" in t.lower() for t in tech_terms)
    assert any("fastapi" in t.lower() for t in tech_terms)


def test_acronym_and_expanded_terminology_pairing():
    text = """
    We implemented Retrieval-Augmented Generation (RAG) for global context search.
    The System Requirements Specification (SRS) governs the avionics interface.
    """
    chunks = chunk_document(text, filename="rag_spec.pdf")
    assert len(chunks) > 0
    c = chunks[0]

    tech_terms = c.get("technical_terms", [])
    mappings = c.get("acronym_mappings", {})

    assert "RAG" in mappings
    assert mappings["RAG"] == "Retrieval-Augmented Generation"
    assert any("retrieval-augmented generation" in t.lower() for t in tech_terms)
    assert any("rag" == t.lower() for t in tech_terms)


test_noise_filtering_text = """
# Document Details
Showing preview (first 1,200 chars).
Click below to extract full text on demand. Saved at /docs/spec.pdf.
The software provides a vector database and a fine-tuning pipeline.
"""

def test_noise_filtering():
    chunks = chunk_document(test_noise_filtering_text, filename="spec.pdf")
    assert len(chunks) > 0
    c = chunks[0]

    tech_terms = c.get("technical_terms", [])
    keywords = c.get("keywords", [])

    # Confirm technical terms exist
    assert any("vector database" in t.lower() for t in tech_terms)

    # Confirm UI noise is NOT classified as technical terms
    assert not any("showing preview" in t.lower() for t in tech_terms)
    assert not any("first 1,200 chars" in t.lower() for t in tech_terms)
    assert not any("click below" in t.lower() for t in tech_terms)
    assert not any("document details" in t.lower() for t in tech_terms)


def test_normalized_term_variants():
    extractor = _TechnicalTerminologyExtractor()
    variants = extractor.generate_normalized_variants("DO-178C")

    assert "do-178c" in variants
    assert "do178c" in variants
    assert "do 178c" in variants
