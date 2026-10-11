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
import threading
from contextlib import contextmanager
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ── Module-level singleton & lock ─────────────────────────────────────────────
_embedder: Optional["TextEmbeddingService"] = None
_embedder_lock = threading.Lock()

# While set (e.g. during a ChromaDB reindex), the active embedding model may not change.
_model_switch_lock_reason: Optional[str] = None
_model_switch_guard = threading.Lock()


class EmbeddingModelLockedError(RuntimeError):
    """Raised when the active embedding model cannot change because an operation pins it."""


class TextEmbeddingService:
    """
    Offline text embedding service backing vector retrieval for RAG.

    By default the service follows the *active* embedding model
    (``settings.EMBEDDING_MODEL`` — a base model or a trained variant): if the
    active model changes, the next load/encode transparently reloads it, so
    callers can never embed with a stale model. Pass ``model_name`` /
    ``model_dir`` to pin an instance to one specific model instead.

    Public API
    ----------
    load()                      — load model from local runtime dir (call once)
    encode(text)                — embed a single string → np.ndarray (dim,)
    encode_batch(texts)         — embed a list of strings → np.ndarray (n, dim)
    embedding_dim()             — return the vector dimension (for FAISS index creation)
    loaded_model_name           — name of the model currently in memory (or None)
    unload()                    — free model memory
    """

    def __init__(self, model_name: Optional[str] = None, model_dir: Optional[str] = None):
        self._tokenizer = None
        self._model = None
        self._dim: Optional[int] = None
        self._loaded = False
        self._loading = False
        self._device: str = "cpu"
        self._use_8bit: bool = False
        self._lock = threading.Lock()
        self._model_name: Optional[str] = None
        self._pinned_name = model_name
        self._pinned_dir = model_dir

    @property
    def loaded_model_name(self) -> Optional[str]:
        return self._model_name if self._loaded else None

    def _target_model(self) -> Tuple[str, Path]:
        """Return (model_name, model_dir) this instance should have loaded."""
        from config import settings
        if self._pinned_dir:
            return self._pinned_name or Path(self._pinned_dir).name, Path(self._pinned_dir)
        if self._pinned_name:
            return self._pinned_name, Path(settings.embedding_model_dir(self._pinned_name))
        return settings.EMBEDDING_MODEL, Path(settings.QWEN_EMBEDDING_MODEL_DIR)

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the embedding model from the local runtime directory.

        Thread-safe: guarded by self._lock so concurrent callers will not
        load the model multiple times or crash GPU memory allocation.
        If a different model is in memory than the one this instance targets
        (e.g. the user selected a trained variant), it is unloaded and replaced.

        Raises
        ------
        FileNotFoundError
            If the model directory does not exist at the expected path.
            The application MUST NOT silently attempt an internet download.
        RuntimeError
            If the model files are present but cannot be loaded.
        """
        target_name, _ = self._target_model()
        if self._loaded and self._model_name == target_name:
            return

        with self._lock:
            target_name, model_dir = self._target_model()
            if self._loaded and self._model_name == target_name:
                return
            if self._loaded:
                logger.info(
                    f"[TextEmbedding] Active embedding model changed "
                    f"({self._model_name} -> {target_name}); reloading."
                )
                self._release_model()
            self._load_locked(target_name, model_dir)

    def _load_locked(self, model_name: str, model_dir: Path) -> None:
        """Load ``model_dir`` into memory. Caller must hold self._lock."""
        self._loading = True
        if not model_dir.exists() or not model_dir.is_dir():
            self._loading = False
            raise FileNotFoundError(
                f"[TextEmbedding] Embedding model directory not found: {model_dir}\n"
                f"Expected model: {model_name}\n"
                "Please place the model folder at the path above before starting the application.\n"
                "The application will NOT download models from the internet."
            )

        model_files = list(model_dir.iterdir())
        if not model_files:
            self._loading = False
            raise FileNotFoundError(
                f"[TextEmbedding] Model directory is empty: {model_dir}\n"
                f"Please populate it with the {model_name} model files."
            )

        # Ensure OCR engine is unloaded first so both models never co-exist in memory
        try:
            from services.ocr_engine import unload_ocr_engine
            unload_ocr_engine()
        except Exception:
            pass

        logger.info(f"[TextEmbedding] Loading {model_name} from {model_dir}...")

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

            self._model_name = model_name
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
                f"  - Model: {model_name}\n"
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
            self._model_name = None
            raise RuntimeError(
                f"[TextEmbedding] Failed to load embedding model from {model_dir}: {e}"
            ) from e
        finally:
            self._loading = False

    def _get_dim(self) -> int:
        """Run a tiny batch to determine the embedding dimension."""
        sample = self._encode_raw(["test"])
        return sample.shape[1]

    def _ensure_loaded(self) -> None:
        # load() is a cheap no-op when the targeted model is already in memory,
        # and reloads when the active embedding model has changed.
        self.load()

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_raw(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        Internal: tokenize + forward pass + mean pooling + L2-normalize.
        Returns np.ndarray of shape (n, dim) in float32.
        """
        import torch

        # Hold local references so a concurrent unload() cannot pull the model
        # out from under an in-flight forward pass.
        tokenizer, model = self._tokenizer, self._model
        if tokenizer is None or model is None:
            self.load()
            tokenizer, model = self._tokenizer, self._model

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if self._device == "cuda":
            target_device = getattr(model, "device", torch.device("cuda"))
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

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

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embed a single string (or list of strings).

        Returns
        -------
        np.ndarray of shape (dim,) for a single string, or (n, dim) if passed a list.
        """
        self._ensure_loaded()
        if isinstance(text, list):
            if not text:
                return np.zeros((0, self._dim), dtype=np.float32)
            if len(text) == 1:
                t = text[0]
                if not isinstance(t, str) or not t.strip():
                    return np.zeros((1, self._dim), dtype=np.float32)
                return self._encode_raw([t.strip()])
            return self.encode_batch(text)

        if not text or not isinstance(text, str) or not text.strip():
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
        with self._lock:
            if not self._loaded:
                return
            self._release_model()

    def _release_model(self) -> None:
        """Drop the in-memory model. Caller must hold self._lock."""
        logger.info(f"[TextEmbedding] Unloading embedding model ({self._model_name})...")
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._loading = False
        self._dim = None
        self._use_8bit = False
        self._model_name = None
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
    """Return the shared TextEmbeddingService instance (lazy-loaded, thread-safe)."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = TextEmbeddingService()
    return _embedder


def unload_text_embedder() -> None:
    """Unload the embedding model to free memory (thread-safe)."""
    global _embedder
    with _embedder_lock:
        if _embedder is not None:
            _embedder.unload()


# ── Active embedding model management ────────────────────────────────────────
#
# The embedding model is process-wide: one embedder singleton and shared ChromaDB
# collections stamped with the model that produced their vectors. Every change of
# the active model (Settings, variant selection, startup restore) goes through
# set_active_embedding_model() so the runtime config, the loaded embedder and the
# vector stores can never disagree.

def get_active_embedding_model() -> str:
    """Name of the embedding model (base model or trained variant) used for all embeddings."""
    from config import settings
    return settings.EMBEDDING_MODEL


def validate_embedding_model_dir(model_dir: Union[str, Path]) -> Optional[str]:
    """Return an error message if ``model_dir`` is not a loadable model directory, else None."""
    path = Path(model_dir)
    if not path.is_dir():
        return f"Embedding model directory not found: {path}"
    if not (path / "config.json").is_file():
        return f"Embedding model directory has no config.json: {path}"
    if not (any(path.glob("*.safetensors")) or any(path.glob("*.bin"))):
        return f"Embedding model directory has no weight files (*.safetensors / *.bin): {path}"
    return None


def get_embedding_model_switch_lock() -> Optional[str]:
    """Reason the active model is currently pinned (e.g. a running reindex), else None."""
    return _model_switch_lock_reason


@contextmanager
def lock_active_embedding_model(reason: str):
    """Pin the active embedding model for the duration of the block."""
    global _model_switch_lock_reason
    with _model_switch_guard:
        if _model_switch_lock_reason is not None:
            raise EmbeddingModelLockedError(_model_switch_lock_reason)
        _model_switch_lock_reason = reason
    try:
        yield
    finally:
        with _model_switch_guard:
            _model_switch_lock_reason = None


def set_active_embedding_model(model_name: str, *, source: str = "") -> bool:
    """
    Make ``model_name`` the active embedding model for the whole application.

    Validates the model directory, updates the runtime config (both setting names)
    and unloads the cached embedder so the next encode loads the new model.
    Returns True if the active model changed.

    Raises
    ------
    ValueError                 — empty model name
    FileNotFoundError          — model directory missing or incomplete
    EmbeddingModelLockedError  — the model is pinned (e.g. ChromaDB reindex running)
    """
    from config import settings

    name = (model_name or "").strip()
    if not name:
        raise ValueError("Embedding model name cannot be empty.")

    with _model_switch_guard:
        if name == settings.EMBEDDING_MODEL:
            # Keep the backward-compatible alias in sync even when nothing else changes.
            settings.QWEN_EMBEDDING_MODEL_NAME = name
            return False
        if _model_switch_lock_reason is not None:
            raise EmbeddingModelLockedError(
                f"Cannot change the embedding model while {_model_switch_lock_reason}."
            )
        error = validate_embedding_model_dir(settings.embedding_model_dir(name))
        if error:
            raise FileNotFoundError(error)

        previous = settings.EMBEDDING_MODEL
        settings.EMBEDDING_MODEL = name
        settings.QWEN_EMBEDDING_MODEL_NAME = name

    unload_text_embedder()
    logger.info(
        f"[TextEmbedding] Active embedding model: {previous} -> {name}"
        + (f" (source: {source})" if source else "")
    )
    return True


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
