import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from services.ai_provider import QwenProvider

def test_ollama_fallback_unavailable():
    """Verify that we fall back to local Qwen when Ollama server is completely unavailable."""
    provider = QwenProvider()
    
    # Mock settings, DB connection failing (so it uses settings port 11434), and urlopen failing
    with patch("config.settings.DATABASE_URL", "sqlite:///dummy.db"), \
         patch("sqlite3.connect") as mock_conn, \
         patch("urllib.request.urlopen") as mock_urlopen, \
         patch.object(QwenProvider, "_get_pipeline") as mock_get_pipeline:
        
        # SQLite fails
        mock_conn.side_effect = Exception("No DB file")
        # Ollama fails
        mock_urlopen.side_effect = Exception("Connection refused")
        
        # Local Qwen mock
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "Local Qwen Response"}]
        mock_get_pipeline.return_value = mock_pipe
        provider._tokenizer = MagicMock()
        provider._tokenizer.apply_chat_template.return_value = "dummy template"
        
        res = provider._infer("Hello", max_new_tokens=10)
        assert res == "Local Qwen Response"
        mock_get_pipeline.assert_called_once()

def test_ollama_success_running_model():
    """Verify that when a model is running, we use it directly and bypass local Qwen."""
    provider = QwenProvider()
    
    # Mock HTTP response helper
    class MockResponse:
        def __init__(self, data, status=200):
            self.data = data
            self.status = status
        def read(self):
            import json
            return json.dumps(self.data).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    with patch("config.settings.DATABASE_URL", "sqlite:///dummy.db"), \
         patch("sqlite3.connect") as mock_conn, \
         patch("urllib.request.urlopen") as mock_urlopen, \
         patch.object(QwenProvider, "_get_pipeline") as mock_get_pipeline:
        
        mock_conn.side_effect = Exception("No DB file")
        
        # Define mock behavior for urlopen
        # 1. GET /api/ps -> Gemma is running
        # 2. POST /api/chat -> Ollama response
        def side_effect(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else req
            if "/api/ps" in url:
                return MockResponse({
                    "models": [
                        {
                            "name": "gemma2:9b",
                            "model": "gemma2:9b",
                            "details": {"family": "gemma2", "families": ["gemma2"]}
                        }
                    ]
                })
            elif "/api/chat" in url:
                return MockResponse({
                    "message": {"role": "assistant", "content": "Ollama Response"}
                })
            return MockResponse({}, status=404)
            
        mock_urlopen.side_effect = side_effect
        
        res = provider._infer("Hello", max_new_tokens=10)
        assert res == "Ollama Response"
        mock_get_pipeline.assert_not_called()
        
        # Verify caching
        assert QwenProvider._cached_ollama_model == "gemma2:9b"
        assert QwenProvider._cached_ollama_server_url == "http://localhost:11434"

def test_ollama_success_installed_model():
    """Verify that if no model is running, but a matching model is installed, we use it."""
    provider = QwenProvider()
    QwenProvider._cached_ollama_model = None
    
    class MockResponse:
        def __init__(self, data, status=200):
            self.data = data
            self.status = status
        def read(self):
            import json
            return json.dumps(self.data).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    with patch("config.settings.DATABASE_URL", "sqlite:///dummy.db"), \
         patch("sqlite3.connect") as mock_conn, \
         patch("urllib.request.urlopen") as mock_urlopen, \
         patch.object(QwenProvider, "_get_pipeline") as mock_get_pipeline:
        
        mock_conn.side_effect = Exception("No DB file")
        
        def side_effect(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else req
            if "/api/ps" in url:
                return MockResponse({"models": []}) # none running
            elif "/api/tags" in url:
                return MockResponse({
                    "models": [
                        {
                            "name": "llama3:latest",
                            "model": "llama3:latest",
                            "details": {"family": "llama", "families": ["llama"]}
                        }
                    ]
                })
            elif "/api/chat" in url:
                return MockResponse({
                    "message": {"role": "assistant", "content": "Ollama Installed Response"}
                })
            return MockResponse({}, status=404)
            
        mock_urlopen.side_effect = side_effect
        
        res = provider._infer("Hello", max_new_tokens=10)
        assert res == "Ollama Installed Response"
        mock_get_pipeline.assert_not_called()
        assert QwenProvider._cached_ollama_model == "llama3:latest"

def test_transcript_embedding_speaker_labels_stripped():
    """Verify that we strip speaker labels before passing text to the embedder."""
    from services.rag_pipeline import embed_transcript
    
    mock_embedder = MagicMock()
    mock_embedder.embedding_dim.return_value = 384
    mock_embedder.encode_batch.return_value = [[0.1] * 384]
    
    mock_store = MagicMock()
    mock_store.add.return_value = 1
    
    transcript = [
        {"speaker_label": "John", "text": "Hello world", "start": 0.0, "end": 2.0},
        {"speaker_label": "Alice", "text": "Hi John", "start": 2.0, "end": 4.0}
    ]
    
    mock_session = MagicMock()
    mock_session.__aenter__.return_value = MagicMock()
    
    with patch("database.AsyncSessionLocal", return_value=mock_session), \
         patch("services.dictionary_service.list_shortcuts", return_value=[]), \
         patch("services.rag_pipeline._get_embedder", return_value=mock_embedder), \
         patch("services.vector_store.get_transcript_store", return_value=mock_store), \
         patch("services.rag_pipeline._get_rag_settings") as mock_settings, \
         patch("services.dictionary_service.expand_terms_in_chunks") as mock_expand:
         
         # Mock RAG settings: chunk size 400, overlap 50, etc.
         mock_settings.return_value = (400, 50, 2, 3, 10, 0.01)
         # Just return the original chunk texts from expand_terms
         mock_expand.side_effect = lambda chunks, shortcuts: [c["text"] for c in chunks]
         
         embed_transcript("rec_1", transcript, "user_1")
         
         # Check encode_batch arguments: speaker labels must be removed!
         # The input text to encode_batch is list of cleaned chunks.
         # The original chunk text has "John: Hello world\nAlice: Hi John"
         # The expected cleaned text is "Hello world\nHi John"
         mock_embedder.encode_batch.assert_called_once()
         called_arg = mock_embedder.encode_batch.call_args[0][0]
         assert len(called_arg) == 1
         assert called_arg[0] == "Hello world\nHi John"
         
         # Verify we still store the original texts (containing speaker labels)
         mock_store.add.assert_called_once()
         stored_texts = mock_store.add.call_args[0][0]
         stored_metadatas = mock_store.add.call_args[0][1]
         assert len(stored_texts) == 1
         assert stored_texts[0] == "John: Hello world\nAlice: Hi John"
         assert stored_metadatas[0]["speakers"] == ["John", "Alice"]

def test_normalize_ollama_url():
    """Verify that Ollama URLs are properly trimmed, formatted with protocol, and trailing slashes removed."""
    from routers.settings_router import normalize_ollama_url

    assert normalize_ollama_url("  192.168.1.100:11434/  ") == "http://192.168.1.100:11434"
    assert normalize_ollama_url("https://ollama.company.com/") == "https://ollama.company.com"
    assert normalize_ollama_url("http://localhost:11434") == "http://localhost:11434"
    assert normalize_ollama_url("") == "http://localhost:11434"

def test_custom_ollama_server_url():
    """Verify that QwenProvider connects to a custom remote Ollama Server URL when specified."""
    provider = QwenProvider()
    QwenProvider._cached_ollama_model = None

    class MockResponse:
        def __init__(self, data, status=200):
            self.data = data
            self.status = status
        def read(self):
            import json
            return json.dumps(self.data).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    called_urls = []

    with patch("config.settings.DATABASE_URL", "sqlite:///dummy.db"), \
         patch("config.settings.OLLAMA_SERVER_URL", "http://192.168.1.50:11434"), \
         patch("sqlite3.connect") as mock_conn, \
         patch("urllib.request.urlopen") as mock_urlopen, \
         patch.object(QwenProvider, "_get_pipeline") as mock_get_pipeline:

        mock_conn.side_effect = Exception("No DB file")

        def side_effect(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else req
            called_urls.append(url)
            if "/api/ps" in url:
                return MockResponse({
                    "models": [
                        {
                            "name": "qwen2.5:7b",
                            "model": "qwen2.5:7b",
                            "details": {"family": "qwen2", "families": ["qwen2"]}
                        }
                    ]
                })
            elif "/api/chat" in url:
                return MockResponse({
                    "message": {"role": "assistant", "content": "Remote Ollama Response"}
                })
            return MockResponse({}, status=404)

        mock_urlopen.side_effect = side_effect

        res = provider._infer("Hello", max_new_tokens=10)
        assert res == "Remote Ollama Response"
        assert any("http://192.168.1.50:11434/api/ps" in u for u in called_urls)
        assert any("http://192.168.1.50:11434/api/chat" in u for u in called_urls)
        assert QwenProvider._cached_ollama_server_url == "http://192.168.1.50:11434"


def test_generate_mom_bypasses_local_pipeline_load():
    """Verify that generate_mom does not load the local pipeline when Ollama is enabled."""
    provider = QwenProvider()
    provider._cached_ollama_model = None

    class MockResponse:
        def __init__(self, data, status=200):
            self.data = data
            self.status = status
        def read(self):
            import json
            return json.dumps(self.data).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    with patch("config.settings.DATABASE_URL", "sqlite:///dummy.db"), \
         patch("sqlite3.connect") as mock_conn, \
         patch("urllib.request.urlopen") as mock_urlopen, \
         patch.object(QwenProvider, "_get_pipeline") as mock_get_pipeline:

        # Mock DB settings to use Ollama
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("http://localhost:11434", 11434, "gemma", 1)
        mock_conn.return_value.cursor.return_value = mock_cursor

        def side_effect(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else req
            if "/api/ps" in url:
                return MockResponse({
                    "models": [{"name": "gemma2:9b", "model": "gemma2:9b", "details": {"family": "gemma2"}}]
                })
            elif "/api/chat" in url:
                return MockResponse({
                    "message": {
                        "role": "assistant",
                        "content": '{"title": "MOM Title", "points_discussed": ["topic 1"], "action_items": [], "introduction": "intro", "conclusion": "outro"}'
                    }
                })
            return MockResponse({}, status=404)

        mock_urlopen.side_effect = side_effect

        transcript = [{"speaker_label": "User", "text": "Hello, let's start the meeting.", "start": 0.0, "end": 2.0}]
        meta = {"filename": "test.wav", "created_at": "2026-07-20", "duration": 10.0, "speakers_detected": ["User"]}

        with patch.object(QwenProvider, "_hierarchical_summarize", return_value="Hello context"):
            res = provider.generate_mom(transcript, meta)

        assert res["title"] == "MOM Title"
        assert res["points_discussed"] == ["topic 1"]
        mock_get_pipeline.assert_not_called()


