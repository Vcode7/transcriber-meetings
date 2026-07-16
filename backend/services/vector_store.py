"""
vector_store.py — FAISS-backed vector store for RAG retrieval.

Design
------
- One VectorStore instance per scope (global_context, meeting_<id>, transcript_<id>)
- Uses FAISS IndexFlatIP (inner product on L2-normalized vectors = cosine similarity)
- Metadata is stored in a JSON sidecar file alongside the .faiss binary
- The embedding model is injected at call time (not stored), keeping the store
  model-agnostic and allowing future model swaps without data loss

File layout
-----------
<VECTOR_STORE_DIR>/
  global_context_<user_id>/
    index.faiss
    index_meta.json
  meeting_<recording_id>/
    index.faiss
    index_meta.json
  transcript_<recording_id>/
    index.faiss
    index_meta.json

Thread safety
-------------
FAISS is not thread-safe for concurrent writes. All write operations
(add, delete) must be called from a single thread (the FastAPI thread executor).
Read operations (search) are safe to call concurrently after loading.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS IndexFlatIP vector store with JSON metadata sidecar.

    Parameters
    ----------
    store_dir  : Directory where index.faiss and index_meta.json are stored.
    dim        : Embedding dimension (must match the embedding model output).
    """

    INDEX_FILENAME = "index.faiss"
    META_FILENAME = "index_meta.json"

    def __init__(self, store_dir: str, dim: int):
        self._dir = Path(store_dir)
        self._dim = dim
        self._index = None   # faiss.IndexFlatIP
        self._meta: List[Dict[str, Any]] = []  # parallel list to FAISS vectors
        self._loaded = False

    # ── Paths ─────────────────────────────────────────────────────────────────

    @property
    def _index_path(self) -> Path:
        return self._dir / self.INDEX_FILENAME

    @property
    def _meta_path(self) -> Path:
        return self._dir / self.META_FILENAME

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def exists(self) -> bool:
        """True if an index exists on disk."""
        return self._index_path.exists() and self._meta_path.exists()

    def load_or_create(self) -> None:
        """Load an existing index from disk, or create a new empty one."""
        if self._loaded:
            return
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for the RAG vector store. "
                "Install it with: pip install faiss-cpu"
            )

        self._dir.mkdir(parents=True, exist_ok=True)

        if self.exists():
            logger.info(f"[VectorStore] Loading existing index from {self._dir}")
            self._index = faiss.read_index(self._index_path.as_posix())
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
            logger.info(
                f"[VectorStore] Loaded {self._index.ntotal} vectors from {self._dir}"
            )
        else:
            logger.info(f"[VectorStore] Creating new index at {self._dir} (dim={self._dim})")
            self._index = faiss.IndexFlatIP(self._dim)
            self._meta = []

        self._loaded = True

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        if not self._loaded or self._index is None:
            return
        import faiss
        self._dir.mkdir(parents=True, exist_ok=True)

        # Atomic save using temp files to prevent locks/sharing violations on Windows
        temp_index_path = self._index_path.with_suffix(".faiss.tmp")
        temp_meta_path = self._meta_path.with_suffix(".json.tmp")

        try:
            faiss.write_index(self._index, temp_index_path.as_posix())
            with open(temp_meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False, default=str)

            # Atomic swap
            if temp_index_path.exists():
                os.replace(temp_index_path, self._index_path)
            if temp_meta_path.exists():
                os.replace(temp_meta_path, self._meta_path)

            logger.debug(
                f"[VectorStore] Saved {self._index.ntotal} vectors to {self._dir}"
            )
        except Exception as e:
            logger.error(f"[VectorStore] Failed to save index atomically: {e}")
            # Fallback to direct write if atomic replacement fails
            try:
                faiss.write_index(self._index, self._index_path.as_posix())
                with open(self._meta_path, "w", encoding="utf-8") as f:
                    json.dump(self._meta, f, ensure_ascii=False, default=str)
            except Exception as e2:
                logger.error(f"[VectorStore] Direct fallback save also failed: {e2}")
                raise e2

    def clear(self) -> None:
        """Delete all vectors and metadata from this store (in-memory and on-disk)."""
        if not self._loaded:
            self.load_or_create()
        import faiss
        self._index = faiss.IndexFlatIP(self._dim)
        self._meta = []
        self.save()
        logger.info(f"[VectorStore] Cleared all vectors in {self._dir}")

    def delete_store(self) -> None:
        """Delete the entire store directory from disk."""
        import shutil
        if self._dir.exists():
            shutil.rmtree(self._dir, ignore_errors=True)
            logger.info(f"[VectorStore] Deleted store directory {self._dir}")
        self._index = None
        self._meta = []
        self._loaded = False

    # ── Write operations ──────────────────────────────────────────────────────

    def add(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
    ) -> int:
        """
        Add text chunks with metadata to the store.

        Parameters
        ----------
        texts      : Raw text strings (used for text storage in metadata).
        metadatas  : Parallel list of metadata dicts (source, doc_id, etc.).
        embeddings : Pre-computed embeddings (n, dim). If None, caller must
                     provide them — this store does NOT call the embedding model.

        Returns
        -------
        Number of vectors added.
        """
        if not self._loaded:
            self.load_or_create()

        if not texts or embeddings is None:
            return 0

        if len(texts) != len(metadatas) or len(texts) != len(embeddings):
            raise ValueError(
                f"[VectorStore] Mismatch: texts={len(texts)}, "
                f"meta={len(metadatas)}, embeddings={len(embeddings)}"
            )

        vecs = np.asarray(embeddings, dtype=np.float32)

        # L2-normalize (belt-and-suspenders — embedding service already normalizes)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        vecs = vecs / norms

        self._index.add(vecs)

        for text, meta in zip(texts, metadatas):
            entry = dict(meta)
            entry["_text"] = text
            self._meta.append(entry)

        self.save()
        logger.info(
            f"[VectorStore] Added {len(texts)} vectors to {self._dir} "
            f"(total={self._index.ntotal})"
        )
        return len(texts)

    def delete_by_filter(self, filter_key: str, filter_value: str) -> int:
        """
        Remove all vectors whose metadata[filter_key] == filter_value.

        FAISS IndexFlatIP does not support in-place deletion; we rebuild the
        index without the matching vectors.

        Returns
        -------
        Number of vectors removed.
        """
        if not self._loaded:
            self.load_or_create()

        import faiss

        keep_indices = [
            i for i, m in enumerate(self._meta)
            if str(m.get(filter_key, "")) != str(filter_value)
        ]
        removed = len(self._meta) - len(keep_indices)

        if removed == 0:
            return 0

        if not keep_indices:
            # All vectors removed
            self._index = faiss.IndexFlatIP(self._dim)
            self._meta = []
        else:
            # Reconstruct index from kept vectors
            old_vectors = self._index.reconstruct_n(0, self._index.ntotal)
            new_vectors = old_vectors[keep_indices]
            self._index = faiss.IndexFlatIP(self._dim)
            self._index.add(new_vectors)
            self._meta = [self._meta[i] for i in keep_indices]

        self.save()
        logger.info(
            f"[VectorStore] Removed {removed} vectors "
            f"(filter: {filter_key}={filter_value}) from {self._dir}"
        )
        return removed

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k most similar chunks to the query embedding.

        Parameters
        ----------
        query_embedding : 1-D float32 array of shape (dim,).
        k               : Maximum number of results to return.
        score_threshold : Minimum cosine similarity (0.0 = return all).

        Returns
        -------
        List of dicts sorted by descending score:
        [{"score": float, "text": str, <metadata fields>}, ...]
        """
        if not self._loaded:
            self.load_or_create()

        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)
        if k == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        # L2-normalize query
        norm = np.linalg.norm(q)
        if norm > 1e-9:
            q = q / norm
        q = q.reshape(1, -1)

        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            if float(score) < score_threshold:
                continue
            entry = dict(self._meta[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results

    def total_vectors(self) -> int:
        """Return the number of vectors currently in the index."""
        if not self._loaded:
            return 0
        return self._index.ntotal if self._index else 0


# ── Factory helpers ───────────────────────────────────────────────────────────

def _get_store_base_dir() -> Path:
    """Resolve the base directory for all vector stores."""
    from config import settings
    return Path(settings.VECTOR_STORE_DIR)


def get_global_context_store(user_id: str, dim: int) -> VectorStore:
    """Return the VectorStore for global context documents for a specific user."""
    store_dir = _get_store_base_dir() / f"global_context_{user_id}"
    store = VectorStore(str(store_dir), dim=dim)
    store.load_or_create()
    return store


def get_meeting_context_store(recording_id: str, dim: int) -> VectorStore:
    """Return the VectorStore for meeting context attachments."""
    store_dir = _get_store_base_dir() / f"meeting_{recording_id}"
    store = VectorStore(str(store_dir), dim=dim)
    store.load_or_create()
    return store


def get_transcript_store(recording_id: str, dim: int) -> VectorStore:
    """Return the VectorStore for transcript chunks."""
    store_dir = _get_store_base_dir() / f"transcript_{recording_id}"
    store = VectorStore(str(store_dir), dim=dim)
    store.load_or_create()
    return store
