"""
test_embedding_model_setting.py — Unit tests for embedding model setting selection and listing.
"""

import pytest
from unittest.mock import patch, MagicMock
from models.settings import UserSettings, UserSettingsUpdate
from config import settings


def test_user_settings_model_default():
    """Verify that UserSettings has default embedding_model field."""
    us = UserSettings(user_id="test_user")
    assert us.embedding_model == "Qwen3-Embedding-0.6B"


def test_user_settings_update_optional_embedding_model():
    """Verify UserSettingsUpdate accepts embedding_model."""
    update = UserSettingsUpdate(embedding_model="Qwen3-Embedding-4B-Instruct-INT8")
    assert update.embedding_model == "Qwen3-Embedding-4B-Instruct-INT8"


@pytest.mark.asyncio
async def test_get_embedding_models_endpoint():
    """Test get_embedding_models endpoint logic."""
    from routers.settings_router import get_embedding_models

    mock_user = {"id": "user123"}
    res = await get_embedding_models(current_user=mock_user)

    assert "active_model" in res
    assert "models" in res
    assert isinstance(res["models"], list)

    model_ids = [m["id"] for m in res["models"]]
    assert "Qwen3-Embedding-0.6B" in model_ids
    assert "Qwen3-Embedding-4B-Instruct-INT8" in model_ids
