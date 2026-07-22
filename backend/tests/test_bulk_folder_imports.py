"""
test_bulk_folder_imports.py — Unit tests for Voice Profiles and Global Context bulk folder import logic and endpoints.
"""

import json
import sys
import uuid
import pytest
from unittest.mock import patch, MagicMock

# Mock optional heavy runtime modules if missing in dev test env
for mod in ["librosa", "reportlab", "reportlab.lib", "reportlab.lib.pagesizes", "reportlab.lib.styles", "reportlab.platypus", "reportlab.lib.colors", "docx", "pptx", "fitz"]:
    if mod not in sys.modules:
        m = MagicMock()
        m.__spec__ = MagicMock()
        sys.modules[mod] = m

from fastapi.testclient import TestClient
from main import app
from routers.auth import get_current_user


@pytest.fixture
def client_and_auth():
    """Setup TestClient with mocked get_current_user dependency and initialized lifespan."""
    mock_user = {"id": "test_user_123", "email": "test@voicesum.ai", "name": "Test User"}
    app.dependency_overrides[get_current_user] = lambda: mock_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_bulk_folder_import_voices(client_and_auth):
    """Test bulk folder import for voice profiles with subfolder structure."""
    fake_audio = b"RIFF....WAVEfmt ....data...."

    files = [
        ("files", ("sample1.wav", fake_audio, "audio/wav")),
        ("files", ("sample2.wav", fake_audio, "audio/wav")),
        ("files", ("bob1.wav", fake_audio, "audio/wav")),
        ("files", ("loose_file.wav", fake_audio, "audio/wav")),
    ]
    relative_paths = json.dumps([
        "RootFolder/Alice/sample1.wav",
        "RootFolder/Alice/sample2.wav",
        "RootFolder/Bob/bob1.wav",
        "RootFolder/loose_file.wav",
    ])

    with patch("services.embedding.extract_embedding_from_file", return_value=MagicMock(tolist=lambda: [0.1] * 192)):
        response = client_and_auth.post(
            "/voice/bulk-folder-import",
            files=files,
            data={"relative_paths": relative_paths},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total_speakers"] == 2
    assert data["successful_speakers"] == 2
    assert len(data["skipped_files"]) == 1
    assert data["skipped_files"][0]["relative_path"] == "RootFolder/loose_file.wav"

    speaker_names = {r["speaker"] for r in data["speaker_results"]}
    assert "Alice" in speaker_names
    assert "Bob" in speaker_names


def test_global_context_folder_import(client_and_auth):
    """Test Global Context document folder import preserving relative_path."""
    uid = uuid.uuid4().hex
    files = [
        ("files", ("Architecture.pdf", f"PDF content test {uid}".encode(), "application/pdf")),
        ("files", ("Notes.txt", f"Plain text notes {uid}".encode(), "text/plain")),
    ]
    relative_paths = json.dumps([
        "Docs/Engineering/Architecture.pdf",
        "Docs/General/Notes.txt",
    ])

    with patch("routers.global_context_router._embed_doc", return_value=5):
        response = client_and_auth.post(
            "/global-context/upload",
            files=files,
            data={"relative_paths": relative_paths},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["uploaded"]) == 2
    assert data["uploaded"][0]["relative_path"] == "Docs/Engineering/Architecture.pdf"
    assert data["uploaded"][1]["relative_path"] == "Docs/General/Notes.txt"
