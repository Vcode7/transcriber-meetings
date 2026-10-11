"""
tests/test_embedding_variant_consistency.py

The selected embedding model (base or trained variant) must be used consistently:
  * switching models never deletes ChromaDB data; stale collections are re-embedded
  * the reindex job re-embeds every stale collection safely and reports progress/failures
  * a failed re-embed leaves the original collection untouched
  * the embedder follows the active model; model switches are validated and can be pinned
  * Stage 3 agenda mapping scores points against all agenda details with the active model

Uses an isolated temporary ChromaDB directory and a deterministic fake embedder.
"""
from __future__ import annotations

import json
import os
import sys
import time
import zlib
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import settings  # noqa: E402
from services import embedding_reindex, text_embedding_service, vector_store  # noqa: E402


# ── Fake embedder ─────────────────────────────────────────────────────────────

DIM, WIDE_DIM = 512, 768

def _fake_vector(model: str, text: str) -> np.ndarray:
    """Bag-of-words vector; different models produce different (rolled) spaces."""
    dim = WIDE_DIM if model.endswith("-wide") else DIM
    v = np.zeros(dim, dtype=np.float32)
    for word in text.lower().replace(",", " ").replace(".", " ").split():
        v[zlib.crc32(word.encode()) % dim] += 1.0
    if model == "model-B":
        v = np.roll(v, 3)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class FakeEmbedder:
    def __init__(self, model_name=None, model_dir=None, fail=False):
        self._pinned = model_name
        self.fail = fail
        self.loaded = None

    def _target(self):
        return self._pinned or settings.EMBEDDING_MODEL

    def load(self):
        self.loaded = self._target()

    @property
    def loaded_model_name(self):
        return self.loaded

    def embedding_dim(self):
        self.load()
        return WIDE_DIM if self.loaded.endswith("-wide") else DIM

    def unload(self):
        self.loaded = None

    def encode_batch(self, texts, batch_size=32):
        self.load()
        if self.fail:
            raise RuntimeError("simulated encoder failure")
        return np.stack([_fake_vector(self.loaded, t) for t in texts]) if texts else np.zeros((0, DIM))

    def encode(self, text):
        if isinstance(text, list):
            return self.encode_batch(text)
        return self.encode_batch([text])[0]


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    """Temporary ChromaDB + fake shared embedder + model-A active."""
    monkeypatch.setattr(settings, "CHROMADB_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "model-A")
    monkeypatch.setattr(settings, "QWEN_EMBEDDING_MODEL_NAME", "model-A")
    monkeypatch.setattr(vector_store, "_STORE_CACHE", {})
    fake = FakeEmbedder()
    monkeypatch.setattr(text_embedding_service, "_embedder", fake)
    yield fake


def _add_docs(store, texts, source="global_context", **extra):
    metas = [{"doc_id": "d1", "chunk_index": i, "source": source, **extra} for i in range(len(texts))]
    vecs = text_embedding_service.get_text_embedder().encode_batch(texts)
    store.add(texts, metas, embeddings=vecs)


def _stored(collection_name):
    coll = vector_store.get_chroma_client().get_collection(collection_name)
    data = coll.get(include=["documents", "metadatas", "embeddings"])
    order = np.argsort(data["ids"])
    return {
        "metadata": coll.metadata,
        "ids": [data["ids"][i] for i in order],
        "documents": [data["documents"][i] for i in order],
        "metadatas": [data["metadatas"][i] for i in order],
        "embeddings": np.asarray([data["embeddings"][i] for i in order]),
    }


# ── Vector store: no data loss, lazy re-embed ─────────────────────────────────

def test_model_switch_reembeds_instead_of_deleting(isolated_chroma):
    texts = ["radar altimeter qualification", "budget approval for phase two"]
    store = vector_store.get_stage2_points_store("u1", DIM)
    _add_docs(store, texts, source="stage2")
    before = _stored(store._collection_name)
    assert before["metadata"]["embedding_model"] == "model-A"

    settings.EMBEDDING_MODEL = "model-B"  # active model changes (e.g. variant selected)

    # Loading the stale collection must keep the data
    assert vector_store.get_stage2_points_store("u1", DIM)._meta  # metadata still readable

    results = store.search(_fake_vector("model-B", "radar altimeter"), k=1)
    after = _stored(store._collection_name)

    assert after["metadata"]["embedding_model"] == "model-B"
    assert after["ids"] == before["ids"]
    assert after["documents"] == before["documents"]
    assert after["metadatas"] == before["metadatas"]
    expected = np.stack([_fake_vector("model-B", d) for d in after["documents"]])
    assert np.allclose(after["embeddings"], expected, atol=1e-5)
    assert results and results[0]["_text"] == "radar altimeter qualification"


def test_add_to_stale_collection_reembeds_existing_vectors_first(isolated_chroma):
    store = vector_store.get_global_context_store("u1", DIM)
    _add_docs(store, ["first document chunk"], user_id="u1", embedding_context="first document chunk")

    settings.EMBEDDING_MODEL = "model-B"
    store.add(
        ["second chunk"],
        [{"doc_id": "d2", "chunk_index": 0, "source": "global_context", "user_id": "u1"}],
        embeddings=np.stack([_fake_vector("model-B", "second chunk")]),
    )

    data = _stored(store._collection_name)
    assert data["metadata"]["embedding_model"] == "model-B"
    assert len(data["ids"]) == 2
    first = data["documents"].index("first document chunk")
    # global context embedding input = embedding_context + "\n" + (expanded) text
    expected = _fake_vector("model-B", "first document chunk\nfirst document chunk")
    assert np.allclose(data["embeddings"][first], expected, atol=1e-5)


def test_cached_store_detects_model_switch(isolated_chroma):
    store = vector_store.get_transcript_store("rec1", DIM)
    _add_docs(store, ["[00:00:01 - 00:00:05] Alice\nwe agreed the test plan"], source="transcript",
              recording_id="rec1")
    settings.EMBEDDING_MODEL = "model-B"
    cached = vector_store.get_transcript_store("rec1", DIM)
    assert cached is store
    cached.search(_fake_vector("model-B", "test plan"), k=1)
    data = _stored(store._collection_name)
    assert data["metadata"]["embedding_model"] == "model-B"
    # transcript embedding input strips the timeline header
    assert np.allclose(data["embeddings"][0], _fake_vector("model-B", "we agreed the test plan"), atol=1e-5)


def test_reembed_handles_dimension_change(isolated_chroma):
    store = vector_store.get_final_rom_points_store("u1", DIM)
    _add_docs(store, ["alpha point", "beta point"], source="final_roim")
    settings.EMBEDDING_MODEL = "model-B-wide"  # wider model
    result = embedding_reindex.reembed_collection(store._collection_name)
    assert result["status"] == "reindexed" and result["dim"] == WIDE_DIM
    data = _stored(store._collection_name)
    assert data["embeddings"].shape == (2, WIDE_DIM)
    assert data["metadata"]["embedding_dim"] == WIDE_DIM


def test_failed_reembed_leaves_original_untouched(isolated_chroma):
    store = vector_store.get_stage2_points_store("u1", DIM)
    _add_docs(store, ["keep me safe", "and me too"], source="stage2")
    before = _stored(store._collection_name)

    settings.EMBEDDING_MODEL = "model-B"
    with pytest.raises(RuntimeError):
        embedding_reindex.reembed_collection(store._collection_name, FakeEmbedder(fail=True))

    after = _stored(store._collection_name)
    assert after["metadata"]["embedding_model"] == "model-A"
    assert np.allclose(after["embeddings"], before["embeddings"])
    names = [c.name for c in vector_store.get_chroma_client().list_collections()]
    assert not any(n.startswith(embedding_reindex.TEMP_PREFIX) for n in names)

    # A search must not compare model-B queries with model-A vectors
    vector_store.invalidate_store_cache(store._collection_name)
    text_embedding_service._embedder.fail = True
    assert store.search(_fake_vector("model-B", "keep me safe"), k=1) == []


def test_recover_interrupted_reindex(isolated_chroma):
    client = vector_store.get_chroma_client()
    # complete copy whose original was already deleted -> restored
    tmp_complete = client.create_collection(
        f"{embedding_reindex.TEMP_PREFIX}aaa111",
        metadata={"hnsw:space": "cosine", "embedding_model": "model-B", "reindex_target": "final_rom_lost"},
    )
    tmp_complete.add(ids=["x"], documents=["restored doc"], embeddings=[_fake_vector("model-B", "restored doc").tolist()])
    # incomplete copy whose original still exists -> dropped
    client.create_collection("final_rom_alive", metadata={"hnsw:space": "cosine", "embedding_model": "model-A"})
    client.create_collection(
        f"{embedding_reindex.TEMP_PREFIX}bbb222",
        metadata={"hnsw:space": "cosine", "embedding_model": "model-B", "reindex_target": "final_rom_alive"},
    )

    restored = embedding_reindex.recover_interrupted_reindex()

    names = {c.name for c in client.list_collections()}
    assert restored == ["final_rom_lost"]
    assert "final_rom_lost" in names and "final_rom_alive" in names
    assert not any(n.startswith(embedding_reindex.TEMP_PREFIX) for n in names)
    assert client.get_collection("final_rom_lost").get()["documents"] == ["restored doc"]


# ── Reindex job ───────────────────────────────────────────────────────────────

def _wait_for_job(timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = embedding_reindex.get_reindex_status()
        if status and status["status"] not in ("running", "cancelling"):
            return status
        time.sleep(0.05)
    raise AssertionError("reindex job did not finish")


def test_reindex_job_reembeds_all_stale_collections(isolated_chroma, monkeypatch):
    _add_docs(vector_store.get_stage2_points_store("u1", DIM), ["s2 point one", "s2 point two"], source="stage2")
    _add_docs(vector_store.get_final_rom_points_store("u1", DIM), ["final point"], source="final_roim")

    settings.EMBEDDING_MODEL = "model-B"
    _add_docs(vector_store.get_stage2_points_store("u2", DIM), ["already current"], source="stage2")

    monkeypatch.setattr(text_embedding_service, "TextEmbeddingService", FakeEmbedder)
    monkeypatch.setattr(text_embedding_service, "validate_embedding_model_dir", lambda path: None)

    status_before = embedding_reindex.get_index_status()
    assert status_before["stale_collections"] == 2 and status_before["stale_vectors"] == 3

    embedding_reindex.start_reindex_job(force=False, requested_by="tester")
    job = _wait_for_job()

    assert job["status"] == "completed", job
    assert job["target_model"] == "model-B"
    assert job["total_collections"] == 2 and job["reindexed_collections"] == 2
    assert job["total_vectors"] == 3 and job["processed_vectors"] == 3
    assert job["progress_percent"] == 100.0 and job["failures"] == []

    status_after = embedding_reindex.get_index_status()
    assert status_after["stale_collections"] == 0
    assert status_after["vectors_by_model"] == {"model-B": 4}


def test_reindex_job_reports_failures_and_keeps_data(isolated_chroma, monkeypatch):
    _add_docs(vector_store.get_stage2_points_store("u1", DIM), ["will fail"], source="stage2")
    settings.EMBEDDING_MODEL = "model-B"

    monkeypatch.setattr(text_embedding_service, "TextEmbeddingService",
                        lambda model_name=None, model_dir=None: FakeEmbedder(model_name, fail=True))
    monkeypatch.setattr(text_embedding_service, "validate_embedding_model_dir", lambda path: None)

    embedding_reindex.start_reindex_job()
    job = _wait_for_job()

    assert job["status"] == "completed_with_errors"
    assert job["failed_collections"] == 1
    assert job["failures"][0]["collection"] == "stage2_points_u1"
    assert "simulated encoder failure" in job["failures"][0]["error"]
    assert _stored("stage2_points_u1")["metadata"]["embedding_model"] == "model-A"


def test_model_switch_blocked_while_pinned(monkeypatch, tmp_path):
    with text_embedding_service.lock_active_embedding_model("ChromaDB is being reindexed"):
        with pytest.raises(text_embedding_service.EmbeddingModelLockedError):
            text_embedding_service.set_active_embedding_model("some-other-model")
    assert text_embedding_service.get_embedding_model_switch_lock() is None


# ── Active model switching ────────────────────────────────────────────────────

def test_set_active_embedding_model_validates_and_syncs(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "base-model")
    monkeypatch.setattr(settings, "QWEN_EMBEDDING_MODEL_NAME", "base-model")
    model_dir = tmp_path / "my-variant"
    model_dir.mkdir()
    monkeypatch.setattr(type(settings), "embedding_model_dir", lambda self, name=None: str(tmp_path / (name or self.EMBEDDING_MODEL)))
    unloads = []
    monkeypatch.setattr(text_embedding_service, "unload_text_embedder", lambda: unloads.append(1))

    with pytest.raises(FileNotFoundError):  # no config / weights yet
        text_embedding_service.set_active_embedding_model("my-variant")
    assert settings.EMBEDDING_MODEL == "base-model"

    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"0")
    assert text_embedding_service.set_active_embedding_model("my-variant") is True
    assert settings.EMBEDDING_MODEL == "my-variant"
    assert settings.QWEN_EMBEDDING_MODEL_NAME == "my-variant"
    assert unloads == [1]
    assert text_embedding_service.set_active_embedding_model("my-variant") is False


def test_embedder_reloads_when_active_model_changes(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "model-A")
    svc = text_embedding_service.TextEmbeddingService()
    loads = []

    def fake_load_locked(name, path):
        loads.append(name)
        svc._model_name = name
        svc._model = object()
        svc._tokenizer = object()
        svc._dim = 4
        svc._loaded = True

    monkeypatch.setattr(svc, "_load_locked", fake_load_locked)
    monkeypatch.setattr(svc, "_release_model", lambda: setattr(svc, "_loaded", False))

    svc.load()
    svc.load()  # no-op
    assert loads == ["model-A"]

    settings.EMBEDDING_MODEL = "variant-X"
    svc.load()
    assert loads == ["model-A", "variant-X"]
    assert svc.loaded_model_name == "variant-X"


def test_pinned_embedder_ignores_active_model(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "model-A")
    svc = text_embedding_service.TextEmbeddingService(model_dir=str(tmp_path))
    name, path = svc._target_model()
    assert path == tmp_path and name == tmp_path.name


# ── Variant manager safety ────────────────────────────────────────────────────

def test_variant_name_sanitized():
    from services.embedding_training.variant_manager import sanitize_variant_name
    assert sanitize_variant_name("../../evil name") == "evil-name"
    assert sanitize_variant_name("ADA Technical v1.2") == "ADA-Technical-v1.2"
    for bad in ("", "...", "///"):
        with pytest.raises(ValueError):
            sanitize_variant_name(bad)


def test_staged_path_must_be_inside_staging_dir(tmp_path):
    from services.embedding_training.embedding_trainer import STAGING_DIR
    from services.embedding_training.variant_manager import _resolve_staged_path
    with pytest.raises(ValueError):
        _resolve_staged_path(str(tmp_path))
    with pytest.raises(ValueError):
        _resolve_staged_path(str(STAGING_DIR / ".." / ".."))
    with pytest.raises(ValueError):
        _resolve_staged_path(str(STAGING_DIR))


def _staged_model(tmp_path):
    staged = tmp_path / "staging" / "staged_job1"
    staged.mkdir(parents=True)
    (staged / "config.json").write_text(json.dumps({"hidden_size": 1024}))
    (staged / "model.safetensors").write_bytes(b"weights")
    (staged / "training_report.json").write_text(json.dumps({"verification": {"status": "ok", "embedding_shift": 0.01}}))
    return staged


def _patch_variant_io(monkeypatch, tmp_path, staged, fail_db=False):
    from contextlib import asynccontextmanager
    from services.embedding_training import variant_manager as vm

    executed = []

    class FakeDb:
        async def execute(self, *args, **kwargs):
            if fail_db:
                raise RuntimeError("database is locked")
            executed.append(args)

        async def commit(self):
            pass

    @asynccontextmanager
    async def fake_ctx():
        yield FakeDb()

    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    monkeypatch.setattr(vm, "get_db_context", fake_ctx)
    monkeypatch.setattr(vm, "_get_embeddings_base_dir", lambda: embeddings)
    monkeypatch.setattr(vm, "discover_base_models", lambda: [])
    monkeypatch.setattr(vm, "_resolve_staged_path", lambda p: staged)
    return vm, embeddings, executed


def test_save_variant_moves_staged_model_atomically(monkeypatch, tmp_path):
    import asyncio
    staged = _staged_model(tmp_path)
    vm, embeddings, executed = _patch_variant_io(monkeypatch, tmp_path, staged)

    saved = asyncio.run(vm.save_trained_variant("u1", "My Variant", "Qwen3-Embedding-0.6B", str(staged), 1, ["a.pdf"], 10, {}))

    target = embeddings / "My-Variant"
    assert saved["name"] == "My-Variant" and executed
    assert (target / "model.safetensors").is_file() and (target / "variant_meta.json").is_file()
    assert saved["metrics"]["verification"]["embedding_shift"] == 0.01
    assert not staged.exists()
    assert [p.name for p in embeddings.iterdir()] == ["My-Variant"]  # no leftover .partial dir


def test_save_variant_rolls_back_to_staging_on_failure(monkeypatch, tmp_path):
    import asyncio
    staged = _staged_model(tmp_path)
    vm, embeddings, _ = _patch_variant_io(monkeypatch, tmp_path, staged, fail_db=True)

    with pytest.raises(RuntimeError, match="database is locked"):
        asyncio.run(vm.save_trained_variant("u1", "My Variant", "Qwen3-Embedding-0.6B", str(staged), 1, [], 10, {}))

    assert list(embeddings.iterdir()) == []  # no half-saved variant
    assert (staged / "model.safetensors").is_file()  # can retry without retraining
    assert not (staged / "variant_meta.json").exists()


# ── Stage 3 similarity-referenced agenda mapping ──────────────────────────────

AGENDAS = [
    {"agenda_id": "A1", "title": "Budget Review", "description": "Quarterly finances",
     "alternative_terminology": ["cost estimate", "spend forecast"], "keywords": ["budget"]},
    {"agenda_id": "A2", "title": "Radar Testing", "description": "Flight trials of the radar",
     "keywords": ["radar", "altimeter"], "related_concepts": ["EMI qualification"]},
]
POINTS = [
    {"id": "p1", "polished_text": "The cost estimate for phase two rose by ten percent.", "speakers": ["Alice"]},
    {"id": "p2", "polished_text": "Radar altimeter EMI qualification finished last week.", "speakers": ["Bob"]},
]


def test_similarity_uses_all_agenda_details():
    from services.rom_service import compute_point_agenda_similarity
    sim = compute_point_agenda_similarity(POINTS, AGENDAS, FakeEmbedder(model_name="model-A"))

    assert sim["agenda_ids"] == ["A1", "A2"]
    assert len(sim["matrix"]) == 2 and len(sim["matrix"][0]) == 2
    p1, p2 = sim["candidates"]
    assert p1["candidates"][0]["agenda_id"] == "A1"
    # p1 never mentions "budget": it matches through the alternative terminology detail
    assert p1["candidates"][0]["best_facet"] == "alternative_terminology"
    assert p2["candidates"][0]["agenda_id"] == "A2"
    assert p1["margin"] > 0


def _run_mapping(monkeypatch, assignments):
    from services.rom_service import RomService
    provider = MagicMock()
    provider.assign_agenda_batch.return_value = {"assignments": assignments}
    monkeypatch.setattr("services.ai_provider.get_provider", lambda: provider)
    monkeypatch.setattr(text_embedding_service, "get_text_embedder", lambda: FakeEmbedder(model_name="model-A"))
    result = RomService().map_points_to_agendas(
        polished_points=POINTS, agendas=AGENDAS, expanded_agendas=[],
        recording_id="rec-1", user_id="test_user",
    )
    return result, provider


def test_mapping_passes_similarity_and_full_agenda_details_to_llm(monkeypatch):
    result, provider = _run_mapping(monkeypatch, [
        {"point_id": "p1", "assigned_agenda_id": "A1", "confidence": "high", "reason": "cost"},
        {"point_id": "p2", "assigned_agenda_id": "A2", "confidence": "high", "reason": "radar"},
    ])
    kwargs = provider.assign_agenda_batch.call_args.kwargs
    items = json.loads(kwargs["batch_json"])
    assert items[0]["similarity_reference"][0]["agenda_id"] == "A1"
    agenda_ref = json.loads(kwargs["agenda_reference_json"])
    assert agenda_ref[0]["alternative_terminology"] == ["cost estimate", "spend forecast"]
    assert kwargs["similarity_guidance_text"]

    assert result["similarity_agenda_ids"] == ["A1", "A2"]
    assert len(result["similarity_matrix"]) == 2
    assert [c["point_id"] for c in result["candidate_results"]] == ["p1", "p2"]
    a1 = result["batch_assignments"][0]
    assert a1["assigned_agenda_id"] == "A1" and a1["similarity_top_agenda_id"] == "A1"


def test_invalid_llm_answer_falls_back_to_most_similar_agenda(monkeypatch):
    result, _ = _run_mapping(monkeypatch, [
        {"point_id": "p2", "assigned_agenda_id": "A99", "confidence": "high", "reason": "?"},
    ])
    by_pid = {a["point_id"]: a for a in result["batch_assignments"]}
    # previously both would land in the first agenda (A1)
    assert by_pid["p2"]["assigned_agenda_id"] == "A2"
    assert by_pid["p2"]["is_probable"] is True
    assert "semantic similarity" in by_pid["p2"]["reason"]


def test_low_confidence_llm_choice_corrected_high_confidence_kept(monkeypatch):
    result, _ = _run_mapping(monkeypatch, [
        {"point_id": "p1", "assigned_agenda_id": "A2", "confidence": "low", "reason": "unsure"},
        {"point_id": "p2", "assigned_agenda_id": "A1", "confidence": "high", "reason": "explicit mention"},
    ])
    by_pid = {a["point_id"]: a for a in result["batch_assignments"]}
    assert by_pid["p1"]["assigned_agenda_id"] == "A1"
    assert by_pid["p1"]["similarity_override"] is True
    assert by_pid["p1"]["llm_assigned_agenda_id"] == "A2"
    # a confident LLM decision is never overridden by similarity
    assert by_pid["p2"]["assigned_agenda_id"] == "A1"
    assert "similarity_override" not in by_pid["p2"]


def test_mapping_still_works_when_embeddings_unavailable(monkeypatch):
    from services.rom_service import RomService

    def broken():
        raise FileNotFoundError("model folder missing")

    provider = MagicMock()
    provider.assign_agenda_batch.return_value = {"assignments": [
        {"point_id": "p1", "assigned_agenda_id": "A1", "confidence": "high", "reason": "ok"},
    ]}
    monkeypatch.setattr("services.ai_provider.get_provider", lambda: provider)
    monkeypatch.setattr(text_embedding_service, "get_text_embedder", broken)
    result = RomService().map_points_to_agendas(
        polished_points=POINTS, agendas=AGENDAS, expanded_agendas=[],
        recording_id="rec-1", user_id="test_user",
    )
    assert result["batch_assignments"][0]["assigned_agenda_id"] == "A1"
    assert result["similarity_matrix"] == [] and result["candidate_results"] == []
    assert provider.assign_agenda_batch.call_args.kwargs["similarity_guidance_text"] is None


def test_similarity_guidance_survives_custom_prompt_without_placeholder(monkeypatch):
    from services import ai_provider

    captured = {}
    custom = "REF {agenda_reference}\n{timeline_guidance}\nPOINTS {batch_json}"
    monkeypatch.setattr(ai_provider, "_get_prompt", lambda key: custom)
    dummy = MagicMock()
    dummy._infer.side_effect = lambda prompt, **kw: captured.setdefault("prompt", prompt) and '{"assignments": []}'

    ai_provider.QwenProvider.assign_agenda_batch(
        dummy, batch_json="[]", agenda_reference_json="[]", similarity_guidance_text="SIMILARITY GUIDE",
    )
    assert "SIMILARITY GUIDE" in captured["prompt"]
    assert "{similarity_guidance}" not in captured["prompt"]
