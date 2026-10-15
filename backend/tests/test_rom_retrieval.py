"""Context retrieval (services/rom_retrieval.py) and previous-meeting retrieval: rank fusion
instead of mixed raw scores, one query per point, the store-relative relevance cutoff, and the
previous-meeting filters (never the current meeting, never a later one). No model is loaded."""
import numpy as np
import pytest

import services.rom_retrieval as rr
from services.rom_service import RomService


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = {k: _unit(v) for k, v in vectors.items()}

    def load(self):
        pass

    def encode_batch(self, texts):
        return np.stack([self.vectors[t] for t in texts])


class FakeStore:
    """Minimal VectorStore: cosine search over fixed chunks, Chroma-style ``where`` filters."""

    def __init__(self, chunks, metas=None):
        self.chunks = {text: _unit(vec) for text, vec in chunks.items()}
        self.metas = metas or {}
        self.searches = []

    def total_vectors(self):
        return len(self.chunks)

    @property
    def _meta(self):
        return []

    def centroid(self):
        return np.stack(list(self.chunks.values())).mean(axis=0)

    def _match(self, meta, where):
        if not where:
            return True
        if "$and" in where:
            return all(self._match(meta, w) for w in where["$and"])
        for key, cond in where.items():
            if isinstance(cond, dict) and "$ne" in cond:
                if meta.get(key) == cond["$ne"]:
                    return False
            elif meta.get(key) != cond:
                return False
        return True

    def search(self, qvec, k=10, score_threshold=0.0, where=None):
        self.searches.append(where)
        hits = []
        for text, vec in self.chunks.items():
            meta = self.metas.get(text, {})
            if self._match(meta, where):
                hits.append({**meta, "_text": text, "score": float(np.dot(_unit(qvec), vec))})
        return sorted(hits, key=lambda h: h["score"], reverse=True)[:k]


class FakeBM25:
    def __init__(self, ranked):
        self.ranked = ranked

    def search(self, query, k=10):
        return [dict(r) for r in self.ranked][:k]


def test_rank_fusion_beats_a_large_keyword_score():
    store = FakeStore({"both": [1, 0.1, 0, 0], "other": [1, 0.3, 0, 0], "kw": [0, 0, 1, 0]})
    embedder = FakeEmbedder({"query": [1, 0, 0, 0], "both": [1, 0.1, 0, 0], "other": [1, 0.3, 0, 0], "kw": [0, 0, 1, 0]})
    bm25 = FakeBM25([{"_text": "kw", "score": 25.0}, {"_text": "both", "score": 3.0}])

    out = rr.retrieve_for_points(["query"], store, bm25, embedder, top_k=3)

    texts = [e["_text"] for e in out]
    assert texts[0] == "both"                     # top in both lists beats a huge BM25 score
    assert texts.index("kw") > texts.index("both")
    assert next(e for e in out if e["_text"] == "kw")["score"] == pytest.approx(0.0, abs=1e-4)  # cosine, not BM25


def test_every_point_gets_its_own_query():
    store = FakeStore({"budget doc": [1, 0, 0, 0], "canteen doc": [0, 1, 0, 0], "sports doc": [0, 0, 1, 0]})
    embedder = FakeEmbedder({"budget point": [1, 0, 0, 0], "canteen point": [0, 1, 0, 0],
                             "budget doc": [1, 0, 0, 0], "canteen doc": [0, 1, 0, 0], "sports doc": [0, 0, 1, 0]})

    out = rr.retrieve_for_points(["budget point", "canteen point"], store, None, embedder, top_k=2)

    assert len(store.searches) == 2
    assert sorted(e["_text"] for e in out) == ["budget doc", "canteen doc"]


def _alike_store():
    """Every chunk scores ~0.87+ against the query (like mean-pooled embeddings); only one is relevant."""
    generic = {f"generic {n}": np.array([0.5, 0.5, 0.5, 0.5]) + np.eye(4)[n % 4] * 0.05 * (n < 4) for n in range(5)}
    chunks = {**generic, "relevant": [0.9, 0.3, 0.3, 0.3]}
    vectors = {**chunks, "query": [0.9, 0.3, 0.3, 0.3]}
    return chunks, vectors


def test_store_relative_cutoff_drops_chunks_no_better_than_average():
    chunks, vectors = _alike_store()
    store, embedder = FakeStore(chunks), FakeEmbedder(vectors)

    out = rr.retrieve_for_points(["query"], store, None, embedder, top_k=5, min_similarity=0.6)

    assert [e["_text"] for e in out] == ["relevant"]
    assert out[0]["lift"] >= rr.MIN_LIFT
    # All generic chunks passed the raw floor: only the store-relative rule told them apart.
    assert all(float(np.dot(_unit(vectors["query"]), _unit(v))) > 0.85 for v in chunks.values())


def test_raw_floor_still_applies():
    store = FakeStore({"close": [1, 0, 0, 0], "far": [1, 1, 0, 0]})
    embedder = FakeEmbedder({"query": [1, 0, 0, 0], "close": [1, 0, 0, 0], "far": [1, 1, 0, 0]})
    out = rr.retrieve_for_points(["query"], store, None, embedder, top_k=5, min_similarity=0.8)
    assert [e["_text"] for e in out] == ["close"]


def test_small_store_uses_only_the_raw_floor():
    chunks, vectors = _alike_store()
    small = {k: v for k, v in list(chunks.items())[:3]}
    out = rr.retrieve_for_points(["query"], FakeStore(small), None, FakeEmbedder(vectors), top_k=5)
    assert out and all(e["lift"] is None for e in out)


# ── Previous-meeting retrieval ────────────────────────────────────────────────

def _previous_meeting_env(monkeypatch):
    chunks = {
        "current meeting point": [1, 0, 0, 0],
        "earlier meeting point": [0.9, 0.1, 0, 0],
        "same-day meeting point": [0.8, 0.2, 0, 0],
        "later meeting point": [0.95, 0.05, 0, 0],
        "unrelated point": [0, 0, 1, 0],
    }
    metas = {
        "current meeting point": {"meeting_id": "cur", "date": "2026-10-12", "stage": "stage2"},
        "earlier meeting point": {"meeting_id": "m-early", "date": "2026-10-01", "stage": "stage2"},
        "same-day meeting point": {"meeting_id": "m-same", "date": "2026-10-12", "stage": "stage2"},
        "later meeting point": {"meeting_id": "m-late", "date": "2026-10-20", "stage": "stage2"},
        "unrelated point": {"meeting_id": "m-other", "date": "2026-09-01", "stage": "stage2"},
    }
    store = FakeStore(chunks, metas)
    monkeypatch.setattr("services.vector_store.get_stage2_points_store", lambda user_id, dim=0: store)
    monkeypatch.setattr("services.text_embedding_service.get_text_embedder", lambda: FakeEmbedder({}))
    return store


def test_previous_meeting_auto_excludes_current_and_later_meetings(monkeypatch):
    store = _previous_meeting_env(monkeypatch)
    out = RomService().retrieve_previous_stage2_context(
        query="q", query_vec=_unit([1, 0, 0, 0]), user_id="u1", current_recording_id="cur",
        top_k=3, min_similarity_threshold=None, current_meeting_date="2026-10-12",
    )
    ids = [r["meeting_id"] for r in out]
    assert "cur" not in ids and "m-late" not in ids
    assert ids[:2] == ["m-early", "m-same"]
    # The current meeting is excluded inside the query, so it cannot take a slot.
    assert {"meeting_id": {"$ne": "cur"}} in store.searches[0]["$and"]


def test_previous_meeting_select_mode_keeps_the_chosen_meeting(monkeypatch):
    _previous_meeting_env(monkeypatch)
    out = RomService().retrieve_previous_stage2_context(
        query="q", query_vec=_unit([1, 0, 0, 0]), user_id="u1", current_recording_id="cur",
        previous_meeting_id="m-late", top_k=3, min_similarity_threshold=None, current_meeting_date="2026-10-12",
    )
    assert [r["meeting_id"] for r in out] == ["m-late"]


def test_previous_meeting_fuses_one_query_per_point(monkeypatch):
    store = _previous_meeting_env(monkeypatch)
    out = RomService().retrieve_previous_stage2_context(
        query="q", query_vec=_unit([1, 0, 0, 0]), query_vecs=np.stack([_unit([1, 0, 0, 0]), _unit([0, 0, 1, 0])]),
        user_id="u1", current_recording_id="cur", top_k=5, min_similarity_threshold=None,
        current_meeting_date="2026-10-12",
    )
    assert len(store.searches) == 2
    assert "m-other" in [r["meeting_id"] for r in out]   # found by the second point's query
