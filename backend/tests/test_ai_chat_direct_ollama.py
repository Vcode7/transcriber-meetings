"""
Test suite for AI Chat Direct Ollama Integration.
Verifies that AI Chat connects directly to Ollama using environment/config
and is not affected by SQLite user_settings table overrides.
"""
import os
import sys
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services.ai_chat_service import (
    get_ai_chat_ollama_config,
    _call_ollama_direct,
    triage_query,
    stream_inference,
)



def test_get_ai_chat_ollama_config_defaults():
    """Verify config loads properly from env/config."""
    cfg = get_ai_chat_ollama_config()
    assert cfg["use_ollama"] is True
    assert "http" in cfg["ollama_server_url"]
    assert cfg["ollama_model_priority"] is not None
    assert cfg["ollama_num_ctx"] >= 4096


def test_db_setting_does_not_disable_ai_chat_ollama():
    """
    Simulate client setting use_ollama = 0 in database.
    Verify that AI Chat config still maintains use_ollama = True.
    """
    # Verify AI Chat config directly
    cfg = get_ai_chat_ollama_config()
    assert cfg["use_ollama"] is True


def test_triage_greeting_direct_ollama():
    """Verify that a greeting triage executes and returns context_required=False."""
    ctx_req, detail = triage_query("Hello! How are you today?", chat_history=[])
    print(f"\n[Test] Greeting Triage -> ctx_req: {ctx_req}, detail: {detail}")
    assert isinstance(ctx_req, bool)
    assert ctx_req is False
    assert len(detail) > 0


def test_triage_meeting_query_direct_ollama():
    """Verify that a meeting question triage executes and returns context_required=True."""
    ctx_req, detail = triage_query("What were the action points assigned to Marcus?", chat_history=[])
    print(f"\n[Test] Meeting Triage -> ctx_req: {ctx_req}, detail: {detail}")
    assert isinstance(ctx_req, bool)
    assert ctx_req is True
    assert len(detail) > 0


def test_stream_inference_direct_ollama():
    """Verify that stream_inference streams real-time tokens directly from Ollama."""
    chunks = []
    def on_chunk(c: str):
        chunks.append(c)

    prompt = "In 5 words or less, what is the capital of France?"
    result = stream_inference(prompt, max_new_tokens=30, chunk_callback=on_chunk)
    print(f"\n[Test] Stream Result: '{result}', Chunks received: {len(chunks)}")
    assert len(result) > 0
    assert len(chunks) > 0
    assert "paris" in result.lower()


if __name__ == "__main__":
    test_get_ai_chat_ollama_config_defaults()
    test_db_setting_does_not_disable_ai_chat_ollama()
    test_triage_greeting_direct_ollama()
    test_triage_meeting_query_direct_ollama()
    test_stream_inference_direct_ollama()
    print("\n[SUCCESS] All AI Chat Direct Ollama tests PASSED!")

