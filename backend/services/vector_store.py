"""
vector_store.py — ChromaDB-backed vector store for RAG retrieval.

Design
------
- One VectorStore instance per scope (global_context, meeting_<id>, transcript_<id>)
- Uses ChromaDB PersistentClient with cosine similarity
- Metadata is stored natively in ChromaDB (no JSON sidecar needed)
- The embedding model is injected at call time (not stored), keeping the store
  model-agnostic and allowing future model swaps without data loss
- Embedding model name is tracked in collection metadata to detect mismatches.
  A collection whose vectors were produced by a different model than the active
  one is never queried or extended as-is and never deleted: it is re-embedded
  with the active model first (see services/embedding_reindex.py).

Storage layout
--------------
<CHROMADB_DIR>/
  (ChromaDB manages internal storage structure)

Collections:
  global_context_<user_id>
  meeting_<recording_id>
  transcript_<recording_id>

Thread safety
-------------
ChromaDB PersistentClient handles its own thread safety.
Operations on one collection are additionally serialized by a per-collection
lock so a re-embed (lazy migration or Settings reindex) can swap the collection
without racing concurrent reads/writes.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Module-level ChromaDB client singleton ────────────────────────────────────
_chroma_client = None
_chroma_client_path: Optional[str] = None

# ── Per-collection locks ──────────────────────────────────────────────────────
_collection_locks: Dict[str, threading.RLock] = {}
_collection_locks_guard = threading.Lock()


def collection_lock(collection_name: str) -> threading.RLock:
    """Re-entrant lock serializing operations on one ChromaDB collection."""
    with _collection_locks_guard:
        lock = _collection_locks.get(collection_name)
        if lock is None:
            lock = threading.RLock()
            _collection_locks[collection_name] = lock
        return lock


def _active_model_name() -> str:
    from config import settings
    return settings.EMBEDDING_MODEL


def _get_chroma_client():
    """Return a shared ChromaDB PersistentClient instance (lazy-loaded)."""
    global _chroma_client, _chroma_client_path
    from config import settings

    target_path = settings.CHROMADB_DIR
    if _chroma_client is not None and _chroma_client_path == target_path:
        return _chroma_client

    try:
        import chromadb
    except ImportError:
        raise ImportError(
            "chromadb is required for the RAG vector store. "
            "Install it with: pip install chromadb"
        )

    os.makedirs(target_path, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=target_path)
    _chroma_client_path = target_path
    logger.info(f"[VectorStore] ChromaDB PersistentClient initialized at {target_path}")
    return _chroma_client


def get_chroma_client():
    """Public accessor for the shared ChromaDB client."""
    return _get_chroma_client()


class VectorStore:
    """
    ChromaDB-backed vector store with native metadata support.

    Parameters
    ----------
    collection_name : Unique name for this ChromaDB collection.
    dim             : Embedding dimension (used for validation, not required by ChromaDB).
    """

    def __init__(self, collection_name: str, dim: int = 0):
        self._collection_name = collection_name
        self._dim = dim
        self._collection = None
        self._loaded = False
        # Model / dimension the stored vectors were produced with (from collection metadata).
        self._stored_model: Optional[str] = None
        self._stored_dim: int = 0

    # ── Model consistency ─────────────────────────────────────────────────────

    def is_stale(self) -> bool:
        """True if the stored vectors were produced by a different model than the active one."""
        if self._stored_model and self._stored_model != _active_model_name():
            return True
        return bool(self._stored_dim and self._dim > 0 and self._stored_dim != self._dim)

    def _ensure_live(self) -> None:
        """Make sure the collection handle is loaded (no model check — metadata-only ops)."""
        if not self._loaded or self._collection is None:
            self._loaded = False
            self.load_or_create()

    def _ensure_current(self) -> None:
        """
        Make sure the collection is loaded AND its vectors match the active embedding
        model. A stale collection is re-embedded in place before any vector operation,
        so a query/insert never mixes vectors from two different models.
        """
        with collection_lock(self._collection_name):
            self._ensure_live()
            if not self.is_stale():
                return
            from services.embedding_reindex import reembed_collection

            logger.warning(
                f"[VectorStore] Collection '{self._collection_name}' was embedded with "
                f"'{self._stored_model}' (dim={self._stored_dim}) but the active model is "
                f"'{_active_model_name()}'. Re-embedding it before use."
            )
            reembed_collection(self._collection_name)
            self._loaded = False
            self._collection = None
            self.load_or_create()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def exists(self) -> bool:
        """True if the collection exists and has at least one document."""
        try:
            client = _get_chroma_client()
            existing = [c.name for c in client.list_collections()]
            if self._collection_name not in existing:
                return False
            coll = client.get_collection(name=self._collection_name)
            return coll.count() > 0
        except Exception:
            return False

    def load_or_create(self) -> None:
        """Load an existing collection or create a new one."""
        if self._loaded:
            return

        client = _get_chroma_client()
        current_model_name = _active_model_name()
        new_metadata = {
            "hnsw:space": "cosine",
            "embedding_model": current_model_name,
            "embedding_dim": self._dim,
        }

        try:
            # Check if collection already exists
            existing_names = [c.name for c in client.list_collections()]
            if self._collection_name in existing_names:
                self._collection = client.get_collection(
                    name=self._collection_name,
                )

                # Record which model produced the stored vectors. A mismatch is NOT
                # resolved by deleting data: vector operations re-embed the collection
                # with the active model first (see _ensure_current).
                coll_meta = self._collection.metadata or {}
                self._stored_model = coll_meta.get("embedding_model") or None
                try:
                    self._stored_dim = int(coll_meta.get("embedding_dim") or 0)
                except (TypeError, ValueError):
                    self._stored_dim = 0

                if self.is_stale():
                    logger.warning(
                        f"[VectorStore] Embedding model mismatch in '{self._collection_name}': "
                        f"stored={self._stored_model} (dim={self._stored_dim}), "
                        f"active={current_model_name} (dim={self._dim}). "
                        "Collection will be re-embedded before its next vector operation."
                    )
                else:
                    logger.info(
                        f"[VectorStore] Loaded existing collection '{self._collection_name}' "
                        f"({self._collection.count()} vectors)"
                    )
            else:
                self._collection = client.get_or_create_collection(
                    name=self._collection_name,
                    metadata=new_metadata,
                )
                self._stored_model = current_model_name
                self._stored_dim = self._dim
                logger.info(
                    f"[VectorStore] Created new collection '{self._collection_name}' "
                    f"(dim={self._dim}, model={current_model_name})"
                )

        except Exception as e:
            # Never delete an existing collection here: a transient read error must not
            # wipe indexed data. get_or_create returns the existing collection if present.
            logger.error(
                f"[VectorStore] Error loading collection '{self._collection_name}': {e}. "
                "Retrying with get_or_create."
            )
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata=new_metadata,
            )
            coll_meta = self._collection.metadata or {}
            self._stored_model = coll_meta.get("embedding_model") or None
            try:
                self._stored_dim = int(coll_meta.get("embedding_dim") or 0)
            except (TypeError, ValueError):
                self._stored_dim = 0

        self._loaded = True

    def save(self) -> None:
        """No-op — ChromaDB PersistentClient persists automatically."""
        pass

    def clear(self) -> None:
        """Delete all vectors and metadata from this collection."""
        with collection_lock(self._collection_name):
            self._ensure_live()

            try:
                client = _get_chroma_client()
                current_model_name = _active_model_name()

                # Delete and recreate the collection to clear all data (also resolves
                # any model mismatch, since the new collection is stamped with the active model)
                meta = {"hnsw:space": "cosine", "embedding_model": current_model_name}
                if self._dim > 0:
                    meta["embedding_dim"] = self._dim
                client.delete_collection(name=self._collection_name)
                self._collection = client.get_or_create_collection(
                    name=self._collection_name,
                    metadata=meta,
                )
                self._stored_model = current_model_name
                self._stored_dim = self._dim
                logger.info(f"[VectorStore] Cleared all vectors in collection '{self._collection_name}'")
            except Exception as e:
                logger.error(f"[VectorStore] Failed to clear collection '{self._collection_name}': {e}")

    def delete_store(self) -> None:
        """Delete the entire collection from ChromaDB."""
        with collection_lock(self._collection_name):
            try:
                client = _get_chroma_client()
                client.delete_collection(name=self._collection_name)
                logger.info(f"[VectorStore] Deleted collection '{self._collection_name}'")
            except Exception as e:
                logger.warning(f"[VectorStore] Could not delete collection '{self._collection_name}': {e}")
            self._collection = None
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
        texts      : Raw text strings.
        metadatas  : Parallel list of metadata dicts (source, doc_id, etc.).
        embeddings : Pre-computed embeddings (n, dim). Required.

        Returns
        -------
        Number of vectors added.
        """
        if not texts or embeddings is None:
            return 0

        with collection_lock(self._collection_name):
            # New vectors come from the active model — never append them to a
            # collection holding another model's vectors.
            self._ensure_current()
            return self._add_locked(texts, metadatas, embeddings)

    def _add_locked(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> int:

        if len(texts) != len(metadatas) or len(texts) != len(embeddings):
            raise ValueError(
                f"[VectorStore] Mismatch: texts={len(texts)}, "
                f"meta={len(metadatas)}, embeddings={len(embeddings)}"
            )

        # Generate deterministic IDs for upsert (prevents duplicates)
        ids = []
        for i, meta in enumerate(metadatas):
            doc_id = meta.get("doc_id") or meta.get("recording_id") or "doc"
            chunk_idx = meta.get("chunk_index", i)
            source = meta.get("source", "unknown")
            ids.append(f"{source}_{doc_id}_chunk_{chunk_idx}")

        # Sanitize metadata: ChromaDB requires values to be str, int, float, or bool
        sanitized_metas = []
        for meta in metadatas:
            sanitized = {}
            for key, val in meta.items():
                if val is None:
                    sanitized[key] = ""
                elif isinstance(val, (str, int, float, bool)):
                    sanitized[key] = val
                elif isinstance(val, list):
                    # Convert lists to comma-separated strings
                    sanitized[key] = ",".join(str(v) for v in val if v is not None)
                else:
                    sanitized[key] = str(val)
            sanitized_metas.append(sanitized)

        # Convert embeddings to list of lists (ChromaDB format)
        vecs = np.asarray(embeddings, dtype=np.float32)
        # L2-normalize (belt-and-suspenders — embedding service already normalizes)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        vecs = vecs / norms
        embeddings_list = vecs.tolist()

        # Upsert in batches (ChromaDB has a batch size limit)
        batch_size = 5000
        total_added = 0
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            self._collection.upsert(
                ids=ids[batch_start:batch_end],
                documents=texts[batch_start:batch_end],
                metadatas=sanitized_metas[batch_start:batch_end],
                embeddings=embeddings_list[batch_start:batch_end],
            )
            total_added += batch_end - batch_start

        logger.info(
            f"[VectorStore] Added {total_added} vectors to '{self._collection_name}' "
            f"(total={self._collection.count()})"
        )
        return total_added

    def delete_by_filter(self, filter_key: str, filter_value: str) -> int:
        """
        Remove all vectors whose metadata[filter_key] == filter_value.

        Returns
        -------
        Number of vectors removed.
        """
        with collection_lock(self._collection_name):
            self._ensure_live()
            return self._delete_by_filter_locked(filter_key, filter_value)

    def _delete_by_filter_locked(self, filter_key: str, filter_value: str) -> int:
        try:
            count_before = self._collection.count()

            # Get matching IDs
            results = self._collection.get(
                where={filter_key: str(filter_value)},
                include=[],
            )
            matching_ids = results.get("ids", [])

            if not matching_ids:
                return 0

            self._collection.delete(ids=matching_ids)
            removed = count_before - self._collection.count()

            logger.info(
                f"[VectorStore] Removed {removed} vectors "
                f"(filter: {filter_key}={filter_value}) from '{self._collection_name}'"
            )
            return removed

        except Exception as e:
            logger.error(f"[VectorStore] delete_by_filter failed: {e}")
            return 0

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        score_threshold: float = 0.0,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k most similar chunks to the query embedding.

        Parameters
        ----------
        query_embedding : 1-D float32 array of shape (dim,).
        k               : Maximum number of results to return.
        score_threshold : Minimum cosine similarity (0.0 = return all).
        where           : Optional metadata filter dict for ChromaDB query.

        Returns
        -------
        List of dicts sorted by descending score:
        [{"score": float, "_text": str, <metadata fields>}, ...]
        """
        with collection_lock(self._collection_name):
            try:
                # The query vector comes from the active model, so the stored vectors must too.
                self._ensure_current()
            except Exception as e:
                logger.error(
                    f"[VectorStore] '{self._collection_name}' could not be re-embedded with the "
                    f"active model ({e}); skipping search instead of comparing vectors from different models."
                )
                return []
            return self._search_locked(query_embedding, k, score_threshold, where)

    def _search_locked(
        self,
        query_embedding: np.ndarray,
        k: int,
        score_threshold: float,
        where: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self._collection is None or self._collection.count() == 0:
            return []

        k = min(k, self._collection.count())
        if k == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        # L2-normalize query
        norm = np.linalg.norm(q)
        if norm > 1e-9:
            q = q / norm
        query_list = q.tolist()

        try:
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_list],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where
            results = self._collection.query(**query_kwargs)
        except Exception as e:
            logger.error(f"[VectorStore] ChromaDB query failed: {e}")
            return []

        output = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            # ChromaDB cosine distance = 1 - cosine_similarity
            # Convert to similarity score for backward compatibility
            distance = distances[i] if i < len(distances) else 1.0
            score = 1.0 - distance

            if score < score_threshold:
                continue

            entry = dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {}

            # Restore list fields from comma-separated strings
            for list_field in ("keywords", "technical_entities", "acronyms",
                               "technical_terms", "entities", "numbers",
                               "important_terms", "speakers", "search_terms",
                               "identifiers", "project_names", "dates"):
                val = entry.get(list_field, "")
                if isinstance(val, str) and val:
                    entry[list_field] = [v.strip() for v in val.split(",") if v.strip()]
                elif not isinstance(val, list):
                    entry[list_field] = []

            entry["score"] = float(score)
            entry["_text"] = documents[i] if i < len(documents) else ""
            output.append(entry)

        return output

    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str = "",
        k: int = 10,
        score_threshold: float = 0.0,
        expand_neighbors: bool = False,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining vector similarity + lexical metadata boosting.

        Parameters
        ----------
        query_embedding  : Vector embedding of the query.
        query_text       : Raw text of the query for exact lexical/identifier matching.
        k                : Top-k items to return.
        score_threshold  : Minimum score cutoff.
        expand_neighbors : If True, appends adjacent chunk context to top results.
        where            : Optional metadata filter dict for ChromaDB query.
        """
        # 1. Base vector search (fetch candidate pool)
        candidates = self.search(query_embedding, k=min(k * 3, 50), score_threshold=0.0, where=where)
        if not candidates:
            return []

        # 2. Extract query terms for exact metadata matching
        import re
        q_clean = query_text.lower().strip()
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-.]+\b", q_clean))
        q_identifiers = set(re.findall(r"\b(?:[A-Z]{2,6}-\d{3,}[\w.-]*|REQ-?\d+[\w.-]*|DOC-?\d+[\w.-]*|PN-?\d+[\w.-]*|v\d+\.\d+[\w.-]*)\b", query_text, re.I))

        # 3. Rescore candidates with lexical + metadata boosting
        boosted_results = []
        for entry in candidates:
            base_score = entry.get("score", 0.0)
            boost = 0.0

            s_terms = set(t.lower() for t in entry.get("search_terms", []))
            t_terms = set(t.lower() for t in entry.get("technical_terms", []))
            idents = set(i.lower() for i in entry.get("identifiers", []))
            techs = set(t.lower() for t in entry.get("technical_entities", []))
            acrs = set(a.lower() for a in entry.get("acronyms", []))
            projs = set(p.lower() for p in entry.get("project_names", []))
            heading = (entry.get("heading") or "").lower()

            # Exact requirement / part number / identifier match: +0.25
            if q_identifiers and any(ident.lower() in [qi.lower() for qi in q_identifiers] for ident in idents):
                boost += 0.25

            # Technical terms / Project / Acronym / Tech Entity match: +0.15
            for token in q_tokens:
                if len(token) >= 2 and (token in acrs or token in techs or token in projs or token in t_terms):
                    boost += 0.15
                    break

            # Heading / Section overlap match: +0.10
            if heading and any(tok in heading for tok in q_tokens if len(tok) > 3):
                boost += 0.10

            # General search_terms overlap match: +0.08
            matched_terms = q_tokens.intersection(s_terms)
            if matched_terms:
                boost += min(0.12, len(matched_terms) * 0.04)

            final_score = base_score + boost
            entry["hybrid_score"] = float(final_score)
            entry["vector_score"] = float(base_score)
            entry["boost"] = float(boost)
            boosted_results.append(entry)

        # Sort by boosted hybrid score
        boosted_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_results = [r for r in boosted_results if r["hybrid_score"] >= score_threshold][:k]

        if expand_neighbors and top_results:
            top_results = self.expand_neighboring_chunks(top_results)

        return top_results

    def expand_neighboring_chunks(
        self,
        results: List[Dict[str, Any]],
        max_neighbors: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Expand top RAG results by appending text from neighboring chunks.
        """
        all_metas = self._meta
        meta_by_doc_index = {}
        for m in all_metas:
            fn = m.get("filename") or m.get("document_name")
            c_idx = m.get("chunk_index")
            if fn and c_idx is not None:
                meta_by_doc_index[(fn, c_idx)] = m

        expanded = []
        for r in results:
            entry = dict(r)
            fn = entry.get("filename") or entry.get("document_name")
            c_idx = entry.get("chunk_index")

            if fn and c_idx is not None:
                neighbor_texts = [entry.get("_text", "")]

                # Previous chunk
                if entry.get("previous_chunk_index") is not None:
                    prev_meta = meta_by_doc_index.get((fn, entry["previous_chunk_index"]))
                    if prev_meta:
                        prev_txt = prev_meta.get("_text") or prev_meta.get("text") or ""
                        if prev_txt:
                            neighbor_texts.insert(0, f"[Prev Chunk]\n{prev_txt}")

                # Next chunk
                if entry.get("next_chunk_index") is not None:
                    next_meta = meta_by_doc_index.get((fn, entry["next_chunk_index"]))
                    if next_meta:
                        next_txt = next_meta.get("_text") or next_meta.get("text") or ""
                        if next_txt:
                            neighbor_texts.append(f"[Next Chunk]\n{next_txt}")

                entry["expanded_text"] = "\n\n".join(neighbor_texts)

            expanded.append(entry)

        return expanded

    def total_vectors(self) -> int:
        """Return the number of vectors currently in the collection."""
        if not self._loaded:
            return 0
        try:
            return self._collection.count() if self._collection else 0
        except Exception:
            return 0

    # ── Metadata access (for BM25 and timeline scan compatibility) ────────────

    @property
    def _meta(self) -> List[Dict[str, Any]]:
        """
        Return all metadata entries from the collection.

        This property provides backward compatibility with code that previously
        accessed the FAISS store's _meta list directly (e.g., BM25 index
        construction, total vector counts, structure stats).
        """
        with collection_lock(self._collection_name):
            self._ensure_live()
            return self._meta_locked()

    def _meta_locked(self) -> List[Dict[str, Any]]:
        if self._collection is None or self._collection.count() == 0:
            return []

        try:
            results = self._collection.get(include=["documents", "metadatas"])
            metadatas = results.get("metadatas", [])
            documents = results.get("documents", [])

            output = []
            for i, meta in enumerate(metadatas):
                entry = dict(meta) if meta else {}
                # Restore list fields from comma-separated strings
                for list_field in ("keywords", "technical_entities", "acronyms",
                                   "technical_terms", "entities", "numbers",
                                   "important_terms", "speakers", "search_terms",
                                   "identifiers", "project_names", "dates"):
                    val = entry.get(list_field, "")
                    if isinstance(val, str) and val:
                        entry[list_field] = [v.strip() for v in val.split(",") if v.strip()]
                    elif not isinstance(val, list):
                        entry[list_field] = []

                entry["_text"] = documents[i] if i < len(documents) else ""
                output.append(entry)

            return output
        except Exception as e:
            logger.error(f"[VectorStore] Failed to retrieve _meta property: {e}")
            return []

    @property
    def _index(self):
        """
        Backward-compatibility shim for code that checks _index.ntotal or
        calls _index.reconstruct(). Returns a proxy object.
        """
        return _IndexProxy(self)


class _IndexProxy:
    """
    Minimal proxy that mimics the FAISS index interface used by existing code:
      - .ntotal  → number of vectors in the collection
      - .reconstruct(pos) → returns the embedding vector at position pos
    """
    def __init__(self, store: VectorStore):
        self._store = store

    @property
    def ntotal(self) -> int:
        return self._store.total_vectors()

    def reconstruct(self, pos: int) -> np.ndarray:
        """
        Reconstruct the embedding vector at the given position.

        ChromaDB doesn't support positional access, so we retrieve all
        embeddings and index by position. This is only used by
        retrieve_evidence_chunkwise() which iterates all chunks anyway.
        """
        try:
            # Callers compare these vectors with active-model query embeddings.
            self._store._ensure_current()
            with collection_lock(self._store._collection_name):
                results = self._store._collection.get(
                    include=["embeddings"],
                )
            embeddings = results.get("embeddings", [])
            if pos < len(embeddings):
                return np.asarray(embeddings[pos], dtype=np.float32)
        except Exception as e:
            logger.warning(f"[VectorStore] reconstruct({pos}) failed: {e}")

        # Return zero vector as fallback
        dim = self._store._dim or 1024
        return np.zeros(dim, dtype=np.float32)


# ── Factory helpers ───────────────────────────────────────────────────────────

def _sanitize_collection_name(name: str) -> str:
    """
    Sanitize a collection name for ChromaDB compatibility.
    ChromaDB collection names must:
    - Be 3-63 characters long
    - Start and end with an alphanumeric character
    - Contain only alphanumeric characters, underscores, or hyphens
    - Not contain two consecutive periods
    """
    import re
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Ensure it starts with alphanumeric
    if sanitized and not sanitized[0].isalnum():
        sanitized = 'c' + sanitized
    # Ensure it ends with alphanumeric
    if sanitized and not sanitized[-1].isalnum():
        sanitized = sanitized + '0'
    # Ensure minimum length
    while len(sanitized) < 3:
        sanitized += '0'
    # Truncate to max length
    if len(sanitized) > 63:
        sanitized = sanitized[:63]
        if not sanitized[-1].isalnum():
            sanitized = sanitized[:-1] + '0'
    return sanitized


_STORE_CACHE: Dict[str, VectorStore] = {}


def invalidate_store_cache(collection_name: str) -> None:
    """Force the cached VectorStore for a collection to re-open it (e.g. after a re-embed swap)."""
    store = _STORE_CACHE.get(collection_name)
    if store is not None:
        store._collection = None
        store._loaded = False


def _get_or_create_store(name: str, dim: int = 0) -> VectorStore:
    sanitized = _sanitize_collection_name(name)
    if sanitized in _STORE_CACHE:
        store = _STORE_CACHE[sanitized]
        if dim > 0 and store._dim != dim:
            store._dim = dim
        if not store._loaded:
            store.load_or_create()
        return store
    store = VectorStore(collection_name=sanitized, dim=dim)
    store.load_or_create()
    _STORE_CACHE[sanitized] = store
    return store


def get_global_context_store(user_id: str, dim: int = 0) -> VectorStore:
    """Return the VectorStore for global context documents for a specific user."""
    return _get_or_create_store(f"global_context_{user_id}", dim=dim)


def get_meeting_context_store(recording_id: str, dim: int = 0) -> VectorStore:
    """Return the VectorStore for meeting context attachments."""
    return _get_or_create_store(f"meeting_{recording_id}", dim=dim)


def get_transcript_store(recording_id: str, dim: int = 0) -> VectorStore:
    """Return the VectorStore for transcript chunks."""
    return _get_or_create_store(f"transcript_{recording_id}", dim=dim)


def get_stage2_points_store(user_id: str, dim: int = 0) -> VectorStore:
    """Return the VectorStore for Stage 2 polished discussion points for a specific user."""
    return _get_or_create_store(f"stage2_points_{user_id}", dim=dim)


def get_final_rom_points_store(user_id: str, dim: int = 0) -> VectorStore:
    """Return the VectorStore for Final ROM points for a specific user."""
    return _get_or_create_store(f"final_rom_{user_id}", dim=dim)

