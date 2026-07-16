"""
text_embedding_service.py — Offline text embedding using Qwen3-Embedding-0.6B.

Architecture
------------
- Isolated behind a clean interface so the model can be swapped later
  (e.g. Qwen3-Embedding-8B) without changing the rest of the RAG pipeline.
- Model is loaded ONCE per process and cached as a module-level singleton.
- Always loads from the local runtime directory; NEVER downloads from the internet.
- If the model directory is missing, raises a FileNotFoundError with a clear message.

Model path (development):
    <project_root>/Application/runtime/embeddings/Qwen3-Embedding-0.6B/

Model path (production / PyInstaller):
    <exe_parent>/../runtime/embeddings/Qwen3-Embedding-0.6B/

The model directory name is controlled by settings.QWEN_EMBEDDING_MODEL_NAME
so upgrading to a larger model only requires changing one .env variable and
re-running the embedding pipeline (all vectors must be regenerated on model change).
"""
from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Module-level singleton ────────────────────────────────────────────────────
_embedder: Optional["TextEmbeddingService"] = None


class TextEmbeddingService:
    """
    Offline text embedding service backed by Qwen3-Embedding-0.6B.

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

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the Qwen3-Embedding model from the local runtime directory.

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
                f"Expected model: {settings.QWEN_EMBEDDING_MODEL_NAME}\n"
                "Please place the model folder at the path above before starting the application.\n"
                "The application will NOT download models from the internet."
            )

        # Verify the directory contains some model files
        model_files = list(model_dir.iterdir())
        if not model_files:
            raise FileNotFoundError(
                f"[TextEmbedding] Model directory is empty: {model_dir}\n"
                f"Please populate it with the {settings.QWEN_EMBEDDING_MODEL_NAME} model files."
            )

        logger.info(
            f"[TextEmbedding] Loading {settings.QWEN_EMBEDDING_MODEL_NAME} "
            f"from {model_dir} ..."
        )

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[TextEmbedding] Using device: {self._device}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True,
            )
            self._model = AutoModel.from_pretrained(
                str(model_dir),
                local_files_only=True,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )
            self._model.eval()
            if self._device == "cuda":
                self._model = self._model.cuda()

            # Determine embedding dimension via a dry-run
            self._dim = self._get_dim()
            self._loaded = True

            logger.info(
                f"[TextEmbedding] {settings.QWEN_EMBEDDING_MODEL_NAME} loaded "
                f"(device={self._device}, dim={self._dim}) ✓"
            )

        except Exception as e:
            self._tokenizer = None
            self._model = None
            self._loaded = False
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

        # Qwen3-Embedding uses a special instruction prefix for retrieval tasks.
        # For document embeddings (asymmetric retrieval), no prefix is needed.
        # For query embeddings, the same approach is used here (symmetric search).
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

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
        """
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [t.strip() for t in texts[i: i + batch_size] if t.strip()]
            if not batch:
                continue
            embs = self._encode_raw(batch)
            all_embeddings.append(embs)

        if not all_embeddings:
            return np.zeros((0, self._dim), dtype=np.float32)

        return np.vstack(all_embeddings).astype(np.float32)

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
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[TextEmbedding] Embedding model unloaded.")


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
