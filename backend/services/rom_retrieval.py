"""
rom_retrieval.py — Context retrieval for the ROM pipeline (Stage 2 enhancement, its context
preview, previous-meeting context and Stage 3 agenda expansion).

Queries: one per discussion point (its text plus its technical terms), so every point is heard
and no query exceeds the embedder's 512-token input. Each query is searched three ways — vector
similarity, BM25 keywords and metadata overlap — and every ranked list of every query is fused
with Reciprocal Rank Fusion. Chunks are ranked by the fused score only: the raw scores of the
three searches are on different scales (cosine 0–1, BM25 unbounded, metadata hit counts) and
are never compared with each other.

Relevance cutoff (store-relative): with mean-pooled embeddings almost every chunk of a store
scores ~0.7–0.85 against any query, so an absolute cosine threshold filters close to randomly.
A chunk is kept only when it is clearly closer to some query than that query's average chunk:

    lift(c) = max over queries q of  cos(q, c) − q·centroid

(q·centroid is exactly q's average cosine to every chunk of the store, see
VectorStore.centroid). The user's minimum similarity, when set, is an additional raw floor.
"""
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

RRF_K = 60
# A kept chunk must beat its query's store-average cosine by this much (tunable).
MIN_LIFT = 0.05
# Below this many chunks the centroid says little about the store: only the raw floor applies.
MIN_CHUNKS_FOR_LIFT = 5
# Two kept chunks this similar say the same thing: the lower-ranked one is dropped.
DIVERSITY_THRESHOLD = 0.92


def chunk_text(entry: Any) -> str:
    if isinstance(entry, dict):
        txt = (entry.get("_text") or entry.get("text") or entry.get("content") or entry.get("chunk")
               or entry.get("page_content") or entry.get("snippet") or entry.get("raw_text") or "")
    else:
        txt = getattr(entry, "_text", None) or getattr(entry, "text", None) or ""
    return str(txt).strip()


def point_query(point: Dict[str, Any]) -> str:
    """The retrieval query of one discussion point: its text plus technical terms it does not already contain."""
    text = str(point.get("discussion_point") or point.get("polished_text") or point.get("text") or "").strip()
    terms = [str(t).strip() for t in point.get("technical_terms") or [] if t is not None and str(t).strip()]
    extra = [t for t in dict.fromkeys(terms) if t.lower() not in text.lower()]
    return " ".join([text] + extra).strip()


def normalize_rows(vecs: Any) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms < 1e-9, 1.0, norms)


def build_bm25(store, log_prefix: str = "ROM Retrieval"):
    """BM25 index over a vector store's chunks (None when the store is empty or unreadable)."""
    from services.bm25 import BM25Index
    try:
        store.load_or_create()
        metas = store._meta
        if not metas:
            return None
        return BM25Index.from_documents([m.get("_text") or m.get("text") or "" for m in metas], metas)
    except Exception as e:
        logger.warning(f"[{log_prefix}] BM25 index build failed: {e}")
        return None


def rrf_fuse(ranked_lists: Sequence[Sequence[Dict[str, Any]]], k: int = RRF_K) -> List[Dict[str, Any]]:
    """Fuse ranked result lists by Reciprocal Rank Fusion (entries keyed by chunk text).
    Returns copies of the entries ordered by fused score, each with ``rrf_score``."""
    scores: Dict[str, float] = {}
    entries: Dict[str, Dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, entry in enumerate(ranked or [], start=1):
            key = chunk_text(entry)
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            entries.setdefault(key, dict(entry))
    fused = sorted(entries.values(), key=lambda e: scores[chunk_text(e)], reverse=True)
    for e in fused:
        e["rrf_score"] = round(scores[chunk_text(e)], 6)
    return fused


def store_size(store) -> int:
    try:
        if not getattr(store, "_loaded", True):
            store.load_or_create()
        return int(store.total_vectors())
    except Exception:
        try:
            return int(getattr(getattr(store, "_index", None), "ntotal", 0) or 0)
        except Exception:
            return 0


def store_baselines(store, qvecs: np.ndarray, size: int) -> Optional[np.ndarray]:
    """Average cosine of each query to the store's chunks, or None when the lift rule is off."""
    if size < MIN_CHUNKS_FOR_LIFT or not hasattr(store, "centroid"):
        return None
    centroid = store.centroid()
    if centroid is None:
        return None
    return qvecs @ np.asarray(centroid, dtype=np.float32)


def _diverse(entries: List[Dict[str, Any]], embedder, top_k: int) -> List[Dict[str, Any]]:
    """The first ``top_k`` entries that are not near-copies of an earlier kept entry."""
    if len(entries) <= 1:
        return entries[:top_k]
    pool = entries[:max(top_k * 3, top_k + 5)]
    try:
        vecs = normalize_rows(embedder.encode_batch([chunk_text(e) for e in pool]))
    except Exception:
        seen, out = set(), []
        for e in pool:
            if chunk_text(e) not in seen:
                seen.add(chunk_text(e))
                out.append(e)
        return out[:top_k]
    kept: List[int] = []
    for i in range(len(pool)):
        if all(float(vecs[i] @ vecs[j]) < DIVERSITY_THRESHOLD for j in kept):
            kept.append(i)
        if len(kept) == top_k:
            break
    return [pool[i] for i in kept]


def gate_by_relevance(
    fused: List[Dict[str, Any]],
    best_cos: Dict[str, float],
    best_lift: Dict[str, float],
    lift_active: bool,
    min_similarity: Optional[float],
) -> tuple:
    """(kept entries in fused order, {"floor": n, "lift": n} dropped counts). Kept entries get
    ``score`` (best cosine across queries, the value shown to users) and ``lift``."""
    kept, dropped = [], {"floor": 0, "lift": 0}
    floor = min_similarity if (min_similarity is not None and min_similarity > 0.0) else None
    for e in fused:
        key = chunk_text(e)
        cos = best_cos.get(key, 0.0)
        if floor is not None and cos < floor:
            dropped["floor"] += 1
            continue
        if lift_active and best_lift.get(key, float("-inf")) < MIN_LIFT:
            dropped["lift"] += 1
            continue
        e["score"] = round(cos, 4)
        e["lift"] = round(best_lift[key], 4) if key in best_lift else None
        kept.append(e)
    return kept, dropped


def retrieve_for_points(
    query_texts: Sequence[str],
    store,
    bm25,
    embedder,
    top_k: int,
    min_similarity: Optional[float] = None,
    store_name: str = "Context Store",
    query_vecs: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Top-k chunks of ``store`` for a group of discussion points (one query per point)."""
    from services.bm25 import BM25Index, _tokenize

    queries = [q for q in (str(t or "").strip() for t in query_texts or []) if q]
    if top_k <= 0 or not queries:
        return []
    size = store_size(store)
    if size == 0:
        logger.info(f"[ROM Retrieval] {store_name}: store is empty, nothing to retrieve")
        return []

    qvecs = normalize_rows(query_vecs if query_vecs is not None else embedder.encode_batch(queries))
    baselines = store_baselines(store, qvecs, size)
    fetch_k = max(top_k * 3, 15)
    try:
        all_metas = store._meta
    except Exception:
        all_metas = []

    ranked_lists: List[List[Dict[str, Any]]] = []
    best_cos: Dict[str, float] = {}
    best_lift: Dict[str, float] = {}
    for qi, (query, qvec) in enumerate(zip(queries, qvecs)):
        try:
            semantic = store.search(qvec, k=fetch_k)
        except Exception as e:
            logger.warning(f"[ROM Retrieval] {store_name}: vector search failed: {e}")
            semantic = []
        for entry in semantic:
            key = chunk_text(entry)
            if not key:
                continue
            cos = float(entry.get("score", 0.0))
            best_cos[key] = max(best_cos.get(key, float("-inf")), cos)
            if baselines is not None:
                best_lift[key] = max(best_lift.get(key, float("-inf")), cos - float(baselines[qi]))
        ranked_lists.append(semantic)
        if bm25 is not None:
            try:
                ranked_lists.append(bm25.search(query, k=fetch_k))
            except Exception as e:
                logger.warning(f"[ROM Retrieval] {store_name}: BM25 search failed: {e}")
        try:
            ranked_lists.append(BM25Index.metadata_search(_tokenize(query), all_metas, k=fetch_k))
        except Exception as e:
            logger.warning(f"[ROM Retrieval] {store_name}: metadata search failed: {e}")

    fused = rrf_fuse(ranked_lists)
    pool = fused[:max(top_k * 6, 30)]

    # Chunks found only by keyword/metadata: their similarity comes from embedding their text.
    missing = [chunk_text(e) for e in pool if chunk_text(e) not in best_cos]
    if missing:
        try:
            sims = normalize_rows(embedder.encode_batch(missing)) @ qvecs.T
            for i, key in enumerate(missing):
                best_cos[key] = float(sims[i].max())
                if baselines is not None:
                    best_lift[key] = float((sims[i] - baselines).max())
        except Exception as e:
            logger.warning(f"[ROM Retrieval] {store_name}: could not embed keyword-only chunks: {e}")

    kept, dropped = gate_by_relevance(pool, best_cos, best_lift, baselines is not None, min_similarity)
    selected = _diverse(kept, embedder, top_k)
    logger.info(
        f"[ROM Retrieval] {store_name}: {len(selected)} chunk(s) from {len(fused)} candidate(s) "
        f"({len(queries)} point quer{'y' if len(queries) == 1 else 'ies'}, store={size} chunks; "
        f"dropped below floor={dropped['floor']}, below lift {MIN_LIFT}={dropped['lift']}"
        f"{'' if baselines is not None else ', lift rule off (small store)'}) "
        f"scores={[e['score'] for e in selected]} lifts={[e['lift'] for e in selected]}"
    )
    return selected
