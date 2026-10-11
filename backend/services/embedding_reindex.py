"""
embedding_reindex.py — Re-embed existing ChromaDB data with the active embedding model.

Every ChromaDB collection is stamped with the embedding model that produced its
vectors (collection metadata ``embedding_model``). When the active model changes —
e.g. a trained embedding variant is selected in Settings — existing vectors no
longer live in the same vector space as new query embeddings. This module
re-embeds collections so the stored vectors match the active model again:

* ``reembed_collection()``  — re-embed one collection (used lazily by VectorStore
                               and by the reindex job).
* ``start_reindex_job()``   — background job that re-embeds every stale collection
                               (Settings → "Reindex ChromaDB with Trained Variant").
* ``get_index_status()``    — which model each collection was embedded with.

Safety
------
* Documents, ids and metadata are never modified — only vectors are recomputed.
* Embedding inputs are rebuilt exactly like the original indexing did
  (contextual headers, dictionary expansion, transcript header stripping).
* A collection is rebuilt into a temporary collection first and swapped in only
  after every vector was written and the count verified. A failure or cancellation
  leaves the original collection untouched.
* If the process dies between "delete original" and "rename temp", the complete
  temp collection is restored by ``recover_interrupted_reindex()`` (run at startup
  and before each reindex job).
* While the reindex job runs, the active embedding model is pinned (Settings and
  variant selection refuse to switch models).
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TEMP_PREFIX = "rxtmp_"
_READ_PAGE_SIZE = 500
_ENCODE_BATCH_SIZE = 16
_WRITE_BATCH_SIZE = 500


class ReindexCancelled(Exception):
    """Raised inside a re-embed when the reindex job was cancelled."""


class ReindexAlreadyRunningError(RuntimeError):
    """Raised when a reindex job is requested while another one is running."""


# ── SQLite helpers (sync — callable from any thread) ──────────────────────────

def _sqlite_db_path() -> Optional[str]:
    from config import settings
    db_url = getattr(settings, "DATABASE_URL", "") or ""
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if db_url.startswith(prefix):
            return db_url[len(prefix):]
    return db_url or None


def _sqlite_query(sql: str, params: tuple) -> List[tuple]:
    db_path = _sqlite_db_path()
    if not db_path or not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _load_shortcuts(user_id: Optional[str], cache: Dict[str, List[Dict]]) -> List[Dict]:
    """Dictionary shortcuts used for term expansion at indexing time."""
    if not user_id:
        return []
    if user_id not in cache:
        try:
            rows = _sqlite_query(
                "SELECT shortcut, full_form FROM shortcut_dictionary WHERE user_id = ? ORDER BY shortcut ASC",
                (user_id,),
            )
            cache[user_id] = [{"shortcut": r[0], "full_form": r[1]} for r in rows]
        except Exception as e:
            logger.warning(f"[Reindex] Could not load dictionary shortcuts for user {user_id}: {e}")
            cache[user_id] = []
    return cache[user_id]


def _recording_owner(recording_id: Optional[str], cache: Dict[str, Optional[str]]) -> Optional[str]:
    if not recording_id:
        return None
    if recording_id not in cache:
        try:
            rows = _sqlite_query("SELECT user_id FROM recordings WHERE id = ?", (recording_id,))
            cache[recording_id] = rows[0][0] if rows else None
        except Exception:
            cache[recording_id] = None
    return cache[recording_id]


# ── Embedding-input reconstruction ────────────────────────────────────────────

def collection_kind(collection_name: str) -> str:
    """Which indexing path produced a collection (decides how its embedding inputs are built)."""
    for prefix, kind in (
        ("global_context_", "global_context"),
        ("meeting_", "meeting_context"),
        ("transcript_", "transcript"),
        ("stage2_points_", "stage2_points"),
        ("final_rom_", "final_rom"),
    ):
        if collection_name.startswith(prefix):
            return kind
    return "other"


def _chunk_index(meta: Optional[Dict[str, Any]]) -> int:
    try:
        return int((meta or {}).get("chunk_index", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _expand_per_document(
    documents: List[str],
    metadatas: List[Optional[Dict[str, Any]]],
    group_key: Callable[[Dict[str, Any]], Any],
    shortcuts_for: Callable[[Dict[str, Any]], List[Dict]],
) -> List[str]:
    """
    Apply dictionary term expansion per source document in chunk order —
    expand_terms_in_chunks() expands each shortcut once per document, so the
    grouping/ordering must match the original indexing call.
    """
    from services.dictionary_service import expand_terms_in_chunks

    expanded = list(documents)
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i, meta in enumerate(metadatas):
        groups[group_key(meta or {})].append(i)
    for idxs in groups.values():
        idxs.sort(key=lambda i: _chunk_index(metadatas[i]))
        shortcuts = shortcuts_for(metadatas[idxs[0]] or {})
        if not shortcuts:
            continue
        texts = expand_terms_in_chunks([{"text": documents[i]} for i in idxs], shortcuts)
        for i, text in zip(idxs, texts):
            expanded[i] = text
    return expanded


def build_embedding_inputs(
    collection_name: str,
    documents: List[str],
    metadatas: List[Optional[Dict[str, Any]]],
) -> List[str]:
    """Rebuild the exact texts the original indexing embedded for these stored chunks."""
    from services.rag_pipeline import (
        build_contextual_embedding_input,
        clean_transcript_text_for_embedding,
    )

    kind = collection_kind(collection_name)
    shortcut_cache: Dict[str, List[Dict]] = {}
    owner_cache: Dict[str, Optional[str]] = {}

    if kind == "global_context":
        default_uid = collection_name[len("global_context_"):]
        expanded = _expand_per_document(
            documents, metadatas,
            group_key=lambda m: m.get("doc_id") or m.get("filename") or "",
            shortcuts_for=lambda m: _load_shortcuts(m.get("user_id") or default_uid, shortcut_cache),
        )
        return [
            build_contextual_embedding_input((meta or {}).get("embedding_context") or doc, exp)
            for doc, meta, exp in zip(documents, metadatas, expanded)
        ]

    if kind == "meeting_context":
        default_rid = collection_name[len("meeting_"):]
        expanded = _expand_per_document(
            documents, metadatas,
            group_key=lambda m: m.get("filename") or m.get("document_name") or "",
            shortcuts_for=lambda m: _load_shortcuts(
                _recording_owner(m.get("recording_id") or default_rid, owner_cache), shortcut_cache
            ),
        )
        return [
            build_contextual_embedding_input((meta or {}).get("embedding_context") or doc, exp)
            for doc, meta, exp in zip(documents, metadatas, expanded)
        ]

    if kind == "transcript":
        default_rid = collection_name[len("transcript_"):]
        expanded = _expand_per_document(
            documents, metadatas,
            group_key=lambda m: "transcript",
            shortcuts_for=lambda m: _load_shortcuts(
                _recording_owner(m.get("recording_id") or default_rid, owner_cache), shortcut_cache
            ),
        )
        return [clean_transcript_text_for_embedding(exp) for exp in expanded]

    # Stage 2 points, Final ROM points, manual MoMs: the stored document is the embedded text.
    return list(documents)


# ── Re-embedding one collection ───────────────────────────────────────────────

def _read_collection(collection) -> Dict[str, List]:
    """Read every id/document/metadata of a collection, page by page, in storage order."""
    total = collection.count()
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Optional[Dict[str, Any]]] = []
    offset = 0
    while offset < total:
        page = collection.get(limit=_READ_PAGE_SIZE, offset=offset, include=["documents", "metadatas"])
        page_ids = page.get("ids") or []
        if not page_ids:
            break
        page_docs = page.get("documents") or [None] * len(page_ids)
        page_metas = page.get("metadatas") or [None] * len(page_ids)
        ids.extend(page_ids)
        documents.extend(d if isinstance(d, str) else "" for d in page_docs)
        metadatas.extend(dict(m) if m else None for m in page_metas)
        offset += len(page_ids)
    return {"ids": ids, "documents": documents, "metadatas": metadatas}


def _collection_is_current(metadata: Optional[Dict[str, Any]], model_name: str, dim: int) -> bool:
    meta = metadata or {}
    if meta.get("embedding_model") != model_name:
        return False
    try:
        stored_dim = int(meta.get("embedding_dim") or 0)
    except (TypeError, ValueError):
        stored_dim = 0
    return not (stored_dim and dim and stored_dim != dim)


def reembed_collection(
    collection_name: str,
    embedder=None,
    *,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    skip_if_current: bool = False,
) -> Dict[str, Any]:
    """
    Recompute every vector of ``collection_name`` with ``embedder`` (default: the
    shared embedder, i.e. the active model) and swap the result in atomically.

    progress_cb(n) is called with the number of vectors encoded so far.
    Returns {"collection", "status", "vectors", "model", "dim"}.
    """
    from services.text_embedding_service import get_text_embedder
    from services.vector_store import collection_lock, get_chroma_client, invalidate_store_cache

    embedder = embedder or get_text_embedder()

    with collection_lock(collection_name):
        client = get_chroma_client()
        source = client.get_collection(name=collection_name)

        embedder.load()
        model_name = embedder.loaded_model_name
        dim = int(embedder.embedding_dim())

        if skip_if_current and _collection_is_current(source.metadata, model_name, dim):
            return {"collection": collection_name, "status": "already_current",
                    "vectors": source.count(), "model": model_name, "dim": dim}

        new_metadata = {
            k: v for k, v in (source.metadata or {}).items()
            if not k.startswith("hnsw:") and k != "reindex_target"
        }
        new_metadata.update({"hnsw:space": "cosine", "embedding_model": model_name, "embedding_dim": dim})

        data = _read_collection(source)
        ids, documents, metadatas = data["ids"], data["documents"], data["metadatas"]

        if not ids:
            # Nothing to re-embed: recreate the empty collection stamped with the new model.
            client.delete_collection(name=collection_name)
            client.get_or_create_collection(name=collection_name, metadata=new_metadata)
            invalidate_store_cache(collection_name)
            return {"collection": collection_name, "status": "reindexed",
                    "vectors": 0, "model": model_name, "dim": dim}

        logger.info(
            f"[Reindex] Re-embedding '{collection_name}' ({len(ids)} vectors, "
            f"kind={collection_kind(collection_name)}) with '{model_name}' (dim={dim})"
        )

        inputs = build_embedding_inputs(collection_name, documents, metadatas)
        vectors = np.zeros((len(inputs), dim), dtype=np.float32)
        for start in range(0, len(inputs), _ENCODE_BATCH_SIZE):
            if cancel_event is not None and cancel_event.is_set():
                raise ReindexCancelled()
            batch = inputs[start:start + _ENCODE_BATCH_SIZE]
            vectors[start:start + len(batch)] = embedder.encode_batch(batch, batch_size=_ENCODE_BATCH_SIZE)
            if progress_cb is not None:
                progress_cb(start + len(batch))

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms < 1e-9, 1.0, norms)

        temp_name = f"{TEMP_PREFIX}{uuid.uuid4().hex[:24]}"
        temp = client.create_collection(
            name=temp_name,
            metadata={**new_metadata, "reindex_target": collection_name},
        )
        original_deleted = False
        try:
            for start in range(0, len(ids), _WRITE_BATCH_SIZE):
                end = start + _WRITE_BATCH_SIZE
                temp.add(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    embeddings=vectors[start:end].tolist(),
                )
            written = temp.count()
            if written != len(ids):
                raise RuntimeError(
                    f"Re-embedded copy of '{collection_name}' has {written} vectors, expected {len(ids)}."
                )
            if cancel_event is not None and cancel_event.is_set():
                raise ReindexCancelled()

            client.delete_collection(name=collection_name)
            original_deleted = True
            temp.modify(name=collection_name)
        except BaseException:
            if not original_deleted:
                try:
                    client.delete_collection(name=temp_name)
                except Exception:
                    pass
            else:
                # The complete re-embedded copy survives under temp_name; restore it now.
                logger.critical(
                    f"[Reindex] Swap of '{collection_name}' failed after the original was removed; "
                    f"restoring from '{temp_name}'."
                )
                try:
                    recover_interrupted_reindex()
                except Exception as rec_err:
                    logger.critical(f"[Reindex] Automatic restore failed: {rec_err}. "
                                    "It will be retried at the next application start.")
            raise
        finally:
            invalidate_store_cache(collection_name)

    logger.info(f"[Reindex] '{collection_name}' re-embedded: {len(ids)} vectors with '{model_name}'")
    return {"collection": collection_name, "status": "reindexed",
            "vectors": len(ids), "model": model_name, "dim": dim}


def recover_interrupted_reindex() -> List[str]:
    """
    Clean up temporary collections left behind by an interrupted re-embed.

    A temp collection whose target no longer exists is a complete copy (the swap was
    interrupted after deleting the original) and is renamed back. Any other temp
    collection is an incomplete copy and is deleted — the original is intact.
    Returns the names of restored collections.
    """
    from services.vector_store import collection_lock, get_chroma_client, invalidate_store_cache

    client = get_chroma_client()
    collections = client.list_collections()
    names = {c.name for c in collections}
    restored: List[str] = []
    for coll in collections:
        if not coll.name.startswith(TEMP_PREFIX):
            continue
        target = (coll.metadata or {}).get("reindex_target")
        lock = collection_lock(target) if target else None
        if lock is not None and not lock.acquire(blocking=False):
            continue  # a re-embed of the target is in progress right now
        try:
            if target and target not in names:
                coll.modify(name=target)
                invalidate_store_cache(target)
                names.add(target)
                restored.append(target)
                logger.warning(f"[Reindex] Restored interrupted re-embed of '{target}' from '{coll.name}'.")
            else:
                client.delete_collection(name=coll.name)
                logger.info(f"[Reindex] Removed incomplete temporary collection '{coll.name}'.")
        finally:
            if lock is not None:
                lock.release()
    return restored


# ── Index status ──────────────────────────────────────────────────────────────

def get_index_status() -> Dict[str, Any]:
    """Summarize which embedding model every ChromaDB collection was embedded with."""
    from services.text_embedding_service import get_active_embedding_model
    from services.vector_store import get_chroma_client

    active = get_active_embedding_model()
    client = get_chroma_client()
    collections = []
    models: Dict[str, int] = defaultdict(int)
    for coll in client.list_collections():
        if coll.name.startswith(TEMP_PREFIX):
            continue
        meta = coll.metadata or {}
        stored = meta.get("embedding_model") or None
        try:
            count = coll.count()
        except Exception:
            count = 0
        models[stored or "unknown"] += count
        collections.append({
            "name": coll.name,
            "kind": collection_kind(coll.name),
            "vectors": count,
            "embedding_model": stored,
            "is_current": stored == active,
        })
    stale = [c for c in collections if not c["is_current"]]
    return {
        "active_model": active,
        "total_collections": len(collections),
        "total_vectors": sum(c["vectors"] for c in collections),
        "stale_collections": len(stale),
        "stale_vectors": sum(c["vectors"] for c in stale),
        "vectors_by_model": dict(models),
        "collections": collections,
    }


# ── Background reindex job ────────────────────────────────────────────────────

_job_lock = threading.Lock()
_job: Optional[Dict[str, Any]] = None
_job_cancel = threading.Event()

_MAX_JOB_LOGS = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_log(job: Dict[str, Any], message: str, level: int = logging.INFO) -> None:
    logger.log(level, f"[Reindex] {message}")
    with _job_lock:
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        del job["logs"][:-_MAX_JOB_LOGS]


def get_reindex_status() -> Optional[Dict[str, Any]]:
    """Snapshot of the current (or last) reindex job, or None if none ran in this process."""
    with _job_lock:
        if _job is None:
            return None
        snap = dict(_job)
        snap["failures"] = list(_job["failures"])
        snap["logs"] = list(_job["logs"])
        return snap


def is_reindex_running() -> bool:
    with _job_lock:
        return _job is not None and _job["status"] in ("running", "cancelling")


def cancel_reindex() -> bool:
    with _job_lock:
        if _job is None or _job["status"] != "running":
            return False
        _job["status"] = "cancelling"
    _job_cancel.set()
    return True


def start_reindex_job(force: bool = False, requested_by: Optional[str] = None) -> Dict[str, Any]:
    """
    Start re-embedding all ChromaDB collections with the active embedding model.
    Only stale collections (embedded with another/unknown model) are processed
    unless ``force`` is True. Raises ReindexAlreadyRunningError if a job is running.
    """
    global _job
    from config import settings
    from services.text_embedding_service import get_active_embedding_model, validate_embedding_model_dir

    target = get_active_embedding_model()
    error = validate_embedding_model_dir(settings.embedding_model_dir(target))
    if error:
        raise FileNotFoundError(error)

    with _job_lock:
        if _job is not None and _job["status"] in ("running", "cancelling"):
            raise ReindexAlreadyRunningError("A ChromaDB reindex is already running.")
        _job_cancel.clear()
        _job = {
            "job_id": str(uuid.uuid4()),
            "status": "running",
            "target_model": target,
            "force": force,
            "requested_by": requested_by,
            "started_at": _now(),
            "finished_at": None,
            "total_collections": 0,
            "processed_collections": 0,
            "reindexed_collections": 0,
            "skipped_collections": 0,
            "failed_collections": 0,
            "total_vectors": 0,
            "processed_vectors": 0,
            "current_collection": None,
            "progress_percent": 0.0,
            "failures": [],
            "logs": [],
            "error": None,
        }
        job = _job

    threading.Thread(target=_run_reindex_job, args=(job, force), daemon=True, name="chroma-reindex").start()
    return get_reindex_status()


def _run_reindex_job(job: Dict[str, Any], force: bool) -> None:
    from services.text_embedding_service import (
        EmbeddingModelLockedError,
        TextEmbeddingService,
        get_active_embedding_model,
        lock_active_embedding_model,
        unload_text_embedder,
    )
    from services.vector_store import get_chroma_client

    target = job["target_model"]
    embedder: Optional[TextEmbeddingService] = None
    try:
        with lock_active_embedding_model(f"ChromaDB is being reindexed with '{target}'"):
            if get_active_embedding_model() != target:
                raise RuntimeError(
                    f"The active embedding model changed to '{get_active_embedding_model()}' "
                    f"before the reindex started; please start it again."
                )
            restored = recover_interrupted_reindex()
            if restored:
                _job_log(job, f"Restored {len(restored)} collection(s) from an interrupted earlier reindex.")

            # Dedicated instance pinned to the target model: unaffected by other features
            # unloading the shared embedder while the job runs.
            unload_text_embedder()
            embedder = TextEmbeddingService(model_name=target)
            _job_log(job, f"Loading embedding model '{target}'...")
            embedder.load()
            dim = int(embedder.embedding_dim())

            plan = []
            for coll in get_chroma_client().list_collections():
                if coll.name.startswith(TEMP_PREFIX):
                    continue
                if not force and _collection_is_current(coll.metadata, target, dim):
                    continue
                plan.append((coll.name, coll.count()))

            with _job_lock:
                job["total_collections"] = len(plan)
                job["total_vectors"] = sum(n for _, n in plan)
            _job_log(
                job,
                f"{len(plan)} collection(s) / {job['total_vectors']} vector(s) to re-embed with "
                f"'{target}' (dim={dim}){' — forced full reindex' if force else ''}.",
            )

            vectors_done_before = 0
            for name, count in plan:
                if _job_cancel.is_set():
                    break
                with _job_lock:
                    job["current_collection"] = name

                def _progress(done: int, base: int = vectors_done_before) -> None:
                    with _job_lock:
                        job["processed_vectors"] = base + done
                        total = max(1, job["total_vectors"])
                        job["progress_percent"] = round(100.0 * job["processed_vectors"] / total, 1)

                try:
                    result = reembed_collection(
                        name, embedder,
                        progress_cb=_progress,
                        cancel_event=_job_cancel,
                        skip_if_current=not force,
                    )
                    with _job_lock:
                        if result["status"] == "already_current":
                            job["skipped_collections"] += 1
                        else:
                            job["reindexed_collections"] += 1
                    if result["status"] != "already_current":
                        _job_log(job, f"✓ {name}: {result['vectors']} vector(s) re-embedded.")
                except ReindexCancelled:
                    _job_log(job, f"Cancelled while processing {name}; it was left unchanged.", logging.WARNING)
                    break
                except Exception as e:
                    logger.exception(f"[Reindex] Failed to re-embed '{name}'")
                    with _job_lock:
                        job["failed_collections"] += 1
                        job["failures"].append({"collection": name, "vectors": count, "error": str(e)})
                    _job_log(job, f"✗ {name}: {e} (original vectors kept)", logging.ERROR)
                finally:
                    vectors_done_before += count
                    with _job_lock:
                        job["processed_collections"] += 1
                        job["processed_vectors"] = vectors_done_before
                        total = max(1, job["total_vectors"])
                        job["progress_percent"] = round(100.0 * vectors_done_before / total, 1)

        with _job_lock:
            if _job_cancel.is_set():
                job["status"] = "cancelled"
            elif job["failed_collections"]:
                job["status"] = "completed_with_errors"
            else:
                job["status"] = "completed"
                job["progress_percent"] = 100.0
        _job_log(
            job,
            f"Reindex {job['status']}: {job['reindexed_collections']} re-embedded, "
            f"{job['skipped_collections']} already current, {job['failed_collections']} failed.",
        )
    except EmbeddingModelLockedError as e:
        with _job_lock:
            job["status"] = "failed"
            job["error"] = f"Another operation is pinning the embedding model: {e}"
        _job_log(job, job["error"], logging.ERROR)
    except Exception as e:
        logger.exception("[Reindex] Reindex job failed")
        with _job_lock:
            job["status"] = "failed"
            job["error"] = str(e)
        _job_log(job, f"Reindex failed: {e}", logging.ERROR)
    finally:
        if embedder is not None:
            embedder.unload()
        with _job_lock:
            job["current_collection"] = None
            job["finished_at"] = _now()
