"""
text_embedding_service.py — Offline text embedding using Qwen embedding models.

Architecture
------------
- Isolated behind a clean interface so the model can be swapped later
  without changing the rest of the RAG pipeline.
- Model is loaded ONCE per process and cached as a module-level singleton.
- Always loads from the local runtime directory; NEVER downloads from the internet.
- Direct Hugging Face model loading via AutoModel and AutoTokenizer.
- Automatic 8-bit BitsAndBytes quantization when CUDA is available.
- Fallback to CPU inference when CUDA is not available.

Model path (development):
    <project_root>/Application/runtime/embeddings/Qwen3-Embedding-0.6B/

Model path (production / PyInstaller):
    <exe_parent>/../runtime/embeddings/Qwen3-Embedding-0.6B/
"""
from __future__ import annotations

import gc
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Module-level singleton ────────────────────────────────────────────────────
_embedder: Optional["TextEmbeddingService"] = None


class TextEmbeddingService:
    """
    Offline text embedding service backing vector retrieval for RAG.

    Public API
    ----------
    load()                      — load model from local runtime dir (call once)
    encode(text)                — embed a single string → np.ndarray (dim,)
    encode_batch(texts)         — embed a list of strings → np.ndarray (n, dim)
    embedding_dim()             — return the vector dimension (for FAISS index creation)
    unload()                    — free model memory
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._dim: Optional[int] = None
        self._loaded = False
        self._device: str = "cpu"
        self._use_8bit: bool = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the embedding model from the local runtime directory.

        Raises
        ------
        FileNotFoundError
            If the model directory does not exist at the expected path.
            The application MUST NOT silently attempt an internet download.
        RuntimeError
            If the model files are present but cannot be loaded.
        """
        if self._loaded:
            return

        from config import settings

        model_dir = Path(settings.QWEN_EMBEDDING_MODEL_DIR)
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(
                f"[TextEmbedding] Embedding model directory not found: {model_dir}\n"
                f"Expected model: {settings.EMBEDDING_MODEL}\n"
                "Please place the model folder at the path above before starting the application.\n"
                "The application will NOT download models from the internet."
            )

        model_files = list(model_dir.iterdir())
        if not model_files:
            raise FileNotFoundError(
                f"[TextEmbedding] Model directory is empty: {model_dir}\n"
                f"Please populate it with the {settings.EMBEDDING_MODEL} model files."
            )

        logger.info(f"[TextEmbedding] Loading {settings.EMBEDDING_MODEL} from {model_dir}...")

        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch
        from transformers import AutoTokenizer, AutoModel

        has_cuda = torch.cuda.is_available()
        self._device = "cuda" if has_cuda else "cpu"
        self._use_8bit = False

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            if has_cuda:
                loaded_with_bnb = False
                try:
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("[TextEmbedding] Configuring BitsAndBytes 8-bit quantization for CUDA GPU...")
                    self._model = AutoModel.from_pretrained(
                        str(model_dir),
                        local_files_only=True,
                        trust_remote_code=True,
                        quantization_config=quant_config,
                        device_map="auto",
                        # NOTE: do NOT pass torch_dtype here.
                        # Combining load_in_8bit=True with torch_dtype triggers a
                        # "Cannot copy out of meta tensor" error on larger models
                        # (e.g. 4B+) because BitsAndBytes uses meta-tensor sharding
                        # and cannot apply a dtype copy afterwards.
                        # BitsAndBytes manages its own internal precision (int8 + fp16
                        # outliers) without needing an explicit dtype.
                    )
                    self._use_8bit = True
                    loaded_with_bnb = True
                except Exception as bnb_err:
                    logger.warning(
                        f"[TextEmbedding] BitsAndBytes 8-bit load failed on CUDA ({bnb_err}). "
                        "Falling back to standard float16 load..."
                    )

                if not loaded_with_bnb:
                    self._model = AutoModel.from_pretrained(
                        str(model_dir),
                        local_files_only=True,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map="auto",
                    )
                    self._use_8bit = False
            else:
                logger.info("[TextEmbedding] CUDA not available. Loading model in float32 for CPU inference...")
                self._model = AutoModel.from_pretrained(
                    str(model_dir),
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                )
                self._model = self._model.to("cpu")
                self._use_8bit = False

            self._model.eval()

            if hasattr(self._model, "config") and hasattr(self._model.config, "hidden_size"):
                self._dim = self._model.config.hidden_size
            else:
                self._dim = self._get_dim()

            self._loaded = True

            gpu_mem_info = "N/A (CPU Mode)"
            if has_cuda:
                try:
                    allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                    gpu_mem_info = f"Allocated={allocated_mb:.1f} MB, Reserved={reserved_mb:.1f} MB"
                except Exception:
                    gpu_mem_info = "CUDA Memory Info Unavailable"

            logger.info(
                f"[TextEmbedding] Model loaded successfully ✓\n"
                f"  - Model Path: {model_dir}\n"
                f"  - Device: {self._device.upper()}\n"
                f"  - BitsAndBytes 8-bit Quantization: {'ENABLED' if self._use_8bit else 'DISABLED'}\n"
                f"  - Embedding Dimension: {self._dim}\n"
                f"  - GPU Memory Usage: {gpu_mem_info}"
            )

        except Exception as e:
            self._tokenizer = None
            self._model = None
            self._loaded = False
            self._dim = None
            self._use_8bit = False
            raise RuntimeError(
                f"[TextEmbedding] Failed to load embedding model from {model_dir}: {e}"
            ) from e

    def _get_dim(self) -> int:
        """Run a tiny batch to determine the embedding dimension."""
        sample = self._encode_raw(["test"])
        return sample.shape[1]

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_raw(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        Internal: tokenize + forward pass + mean pooling + L2-normalize.
        Returns np.ndarray of shape (n, dim) in float32.
        """
        import torch

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if self._device == "cuda":
            target_device = getattr(self._model, "device", torch.device("cuda"))
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean pooling over non-padding tokens
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state  # (batch, seq, dim)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_emb = (token_embeddings * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_emb / count  # (batch, dim)

        # L2-normalize
        norms = pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
        normalized = pooled / norms

        return normalized.cpu().float().numpy()

    def encode(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Returns
        -------
        np.ndarray of shape (dim,) — L2-normalized float32.
        """
        self._ensure_loaded()
        if not text or not text.strip():
            return np.zeros(self._dim, dtype=np.float32)
        result = self._encode_raw([text.strip()])
        return result[0]

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed a list of strings in mini-batches.

        Returns
        -------
        np.ndarray of shape (n, dim) — L2-normalized float32.

        Notes
        -----
        Empty or whitespace-only strings are replaced with a zero vector rather
        than being filtered out. This preserves the 1-to-1 correspondence between
        ``texts`` and the returned rows, so callers can safely zip texts with
        embeddings without index mismatches.
        """
        self._ensure_loaded()
        n = len(texts)
        if n == 0:
            return np.zeros((0, self._dim), dtype=np.float32)

        # Pre-allocate the full result array. Zero rows are returned for empty inputs.
        result = np.zeros((n, self._dim), dtype=np.float32)

        # Collect (original_index, cleaned_text) for non-empty strings only.
        non_empty_items = [
            (i, texts[i].strip()) for i in range(n) if texts[i].strip()
        ]

        if not non_empty_items:
            return result  # all inputs were empty

        # Process non-empty texts in mini-batches.
        for batch_start in range(0, len(non_empty_items), batch_size):
            batch_items = non_empty_items[batch_start: batch_start + batch_size]
            orig_indices = [item[0] for item in batch_items]
            batch_texts = [item[1] for item in batch_items]

            embs = self._encode_raw(batch_texts)  # shape (batch, dim)
            for out_pos, orig_idx in enumerate(orig_indices):
                result[orig_idx] = embs[out_pos]

        return result

    def embedding_dim(self) -> int:
        """Return the vector dimension of this embedding model."""
        self._ensure_loaded()
        return self._dim

    # ── Memory management ─────────────────────────────────────────────────────

    def unload(self) -> None:
        """Free model from memory (GPU + CPU)."""
        if not self._loaded:
            return
        logger.info("[TextEmbedding] Unloading embedding model...")
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._dim = None
        self._use_8bit = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[TextEmbedding] Embedding model unloaded and CUDA memory released.")

    # ── Compatibility Aliases ──────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """Alias for encode for backward compatibility."""
        return self.encode(text)

    def embed_chunks(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Alias for encode_batch for backward compatibility."""
        return self.encode_batch(texts, batch_size=batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        """Alias for encode for backward compatibility."""
        return self.encode(text)


# ── Singleton accessor ────────────────────────────────────────────────────────

def get_text_embedder() -> TextEmbeddingService:
    """Return the shared TextEmbeddingService instance (lazy-loaded)."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbeddingService()
    return _embedder


def unload_text_embedder() -> None:
    """Unload the embedding model to free memory."""
    global _embedder
    if _embedder is not None:
        _embedder.unload()


# ── Compatibility Module-Level Wrappers ──────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    """Module-level wrapper for embed_text."""
    return get_text_embedder().embed_text(text)


def embed_chunks(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Module-level wrapper for embed_chunks."""
    return get_text_embedder().embed_chunks(texts, batch_size=batch_size)


def embed_query(text: str) -> np.ndarray:
    """Module-level wrapper for embed_query."""
    return get_text_embedder().embed_query(text)
