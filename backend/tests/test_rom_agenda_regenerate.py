"""
test_rom_agenda_regenerate.py — Unit tests for ROM agenda creation, regeneration (cache bypass), and PDF/document agenda parsing.
"""

import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from services.doc_extractor import _extract_pdf, extract_text_from_file


def test_extract_pdf_with_agenda_content():
    """Verify PDF extraction extracts text and structures correctly."""
    try:
        import fitz
    except ImportError:
        try:
            import pymupdf as fitz
        except ImportError:
            fitz = None

    if fitz is None:
        pytest.skip("PyMuPDF / fitz not installed in test environment")

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        fitz.Point(50, 50),
        "Meeting Agenda:\n1. Opening & Welcome (Speaker: Alice)\n2. Project Architecture Review (Speaker: Bob)\n3. Budget & Resource Allocation (Speaker: Charlie)"
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()
    doc.save(tmp_path)
    doc.close()

    try:
        extracted = _extract_pdf(tmp_path)
        assert "Meeting Agenda" in extracted
        assert "Opening & Welcome" in extracted
        assert "Project Architecture Review" in extracted
        assert "Budget & Resource Allocation" in extracted
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_generate_agendas_only_reuses_cache_when_force_false():
    """Verify generate_agendas_only reuses cached agenda when force_reextract=False."""
    from services.rom_service import rom_service

    mock_cached = [
        {"agenda_id": "A1", "topic": "Old Cached Topic", "details": "Old details", "speaker": "Speaker 1"}
    ]

    mock_embedder = MagicMock()
    mock_embedder._dim = 1024
    mock_embedder.encode_batch.return_value = np.zeros((1, 1024), dtype=np.float32)

    with patch("services.rag_pipeline._load_parsed_agenda", return_value=mock_cached), \
         patch("services.text_embedding_service.get_text_embedder", return_value=mock_embedder):
        res = rom_service.generate_agendas_only(
            agenda_text="New Agenda Text",
            recording_id="rec-123",
            user_id="user-1",
            force_reextract=False,
        )

        assert len(res["agendas"]) == 1
        assert res["agendas"][0]["title"] == "Old Cached Topic"


def test_generate_agendas_only_bypasses_cache_when_force_true():
    """Verify generate_agendas_only generates fresh agendas via LLM when force_reextract=True."""
    from services.rom_service import rom_service

    mock_cached = [
        {"agenda_id": "A1", "topic": "Old Cached Topic", "details": "Old details", "speaker": "Speaker 1"}
    ]

    mock_provider = MagicMock()
    mock_provider.generate_rom_agendas.return_value = {
        "agendas": [
            {
                "agenda_id": "A1",
                "title": "Fresh Generated Topic 1",
                "description": "Fresh description",
                "presenter": "New Speaker",
                "keywords": ["fresh", "topic"],
                "related_concepts": [],
                "alternative_terminology": [],
                "expected_themes": [],
            }
        ]
    }

    mock_embedder = MagicMock()
    mock_embedder._dim = 1024
    mock_embedder.encode_batch.return_value = np.zeros((1, 1024), dtype=np.float32)

    with patch("services.rag_pipeline._load_parsed_agenda", return_value=mock_cached), \
         patch("services.rag_pipeline._save_parsed_agenda") as mock_save, \
         patch("services.text_embedding_service.get_text_embedder", return_value=mock_embedder), \
         patch("services.ai_provider.get_provider", return_value=mock_provider):

        res = rom_service.generate_agendas_only(
            agenda_text="Fresh Agenda Text Uploaded by User",
            recording_id="rec-123",
            user_id="user-1",
            force_reextract=True,
        )

        assert len(res["agendas"]) == 1
        assert res["agendas"][0]["title"] == "Fresh Generated Topic 1"
        assert res["agendas"][0]["speaker"] == "New Speaker"
        assert mock_save.called
