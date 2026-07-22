"""
test_text_embedding_service.py — Unit tests for PyTorch HuggingFace TextEmbeddingService.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from services.text_embedding_service import TextEmbeddingService, get_text_embedder, unload_text_embedder


def test_embedding_service_cpu_fallback():
    """Verify that TextEmbeddingService loads with AutoModel and AutoTokenizer on CPU when CUDA is disabled."""
    service = TextEmbeddingService()

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "[PAD]"
    mock_tokenizer.eos_token = "[EOS]"

    mock_outputs = MagicMock()
    # Mock last_hidden_state: (batch=1, seq=3, dim=4)
    import torch
    mock_outputs.last_hidden_state = torch.ones((1, 3, 4), dtype=torch.float32)

    mock_model = MagicMock()
    mock_model.config.hidden_size = 4
    mock_model.to.return_value = mock_model
    def mock_tokenizer_call(texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.ones((batch_size, 3), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 3), dtype=torch.long),
        }

    def mock_model_forward(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        out = MagicMock()
        out.last_hidden_state = torch.ones((batch_size, 3, 4), dtype=torch.float32)
        return out

    mock_tokenizer.side_effect = mock_tokenizer_call
    mock_model.side_effect = mock_model_forward

    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.iterdir", return_value=[MagicMock()]), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.AutoModel.from_pretrained", return_value=mock_model):

        service.load()

        assert service._loaded is True
        assert service._device == "cpu"
        assert service._use_8bit is False
        assert service.embedding_dim() == 4

        # Test single string encode
        vec = service.encode("Hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (4,)
        # Verify L2-normalization
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0)

        # Test batch encode
        batch_vecs = service.encode_batch(["Hello world", "Test string"])
        assert isinstance(batch_vecs, np.ndarray)
        assert batch_vecs.shape == (2, 4)

        # Test unload and cleanup
        service.unload()
        assert service._loaded is False
        assert service._model is None
        assert service._tokenizer is None


def test_embedding_service_bitsandbytes_cuda():
    """Verify that TextEmbeddingService configures BitsAndBytes 8-bit quantization when CUDA is available."""
    service = TextEmbeddingService()

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "[PAD]"

    import torch
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = torch.ones((1, 2, 8), dtype=torch.float32)

    mock_model = MagicMock()
    mock_model.config.hidden_size = 8
    mock_model.device = torch.device("cuda:0")
    mock_model.return_value = mock_outputs

    def mock_tokenizer_call(texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[10, 20]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    mock_tokenizer.side_effect = mock_tokenizer_call

    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.iterdir", return_value=[MagicMock()]), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=512 * 1024 * 1024), \
         patch("torch.cuda.memory_reserved", return_value=1024 * 1024 * 1024), \
         patch("torch.cuda.empty_cache") as mock_empty_cache, \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.BitsAndBytesConfig") as mock_bnb_config, \
         patch("transformers.AutoModel.from_pretrained", return_value=mock_model) as mock_automodel:

        service.load()

        assert service._loaded is True
        assert service._device == "cuda"
        assert service._use_8bit is True
        assert service.embedding_dim() == 8

        # Check quantization_config was passed (torch_dtype omitted to prevent meta-tensor error)
        mock_bnb_config.assert_called_once_with(load_in_8bit=True)
        _, kwargs = mock_automodel.call_args
        assert kwargs.get("device_map") == "auto"
        assert kwargs.get("torch_dtype") is None

        # Test unload clears CUDA cache
        service.unload()
        assert service._loaded is False
        mock_empty_cache.assert_called_once()
