"""Stage 3 agenda matching: agendas built from discussion topics keep their points, points that
match no agenda go to "Other Discussion", each point is sent with its neighbours, and similarity
is centred on the meeting's common component. No model is loaded."""
import json
import zlib
from unittest.mock import MagicMock

import numpy as np

import services.text_embedding_service as tes
from services.rom_service import OTHER_AGENDA_ID, RomService, compute_point_agenda_similarity


class FakeEmbedder:
    """Fixed vectors for known texts, deterministic random vectors for the rest."""
    loaded_model_name = "fake"

    def __init__(self, vectors=None, dim=8):
        self.vectors, self.dim = vectors or {}, dim

    def load(self):
        pass

    def encode_batch(self, texts):
        out = []
        for t in texts:
            if t in self.vectors:
                v = np.asarray(self.vectors[t], dtype=np.float32)
            else:
                v = np.random.default_rng(zlib.crc32(t.encode("utf-8"))).normal(size=self.dim).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


POINTS = [
    {"id": "p1", "polished_text": "Hostel renovation budget of 42 lakh was presented.", "timeline_start": 0.0},
    {"id": "p2", "polished_text": "Ravi will float the hostel tender by Friday.", "timeline_start": 60.0},
    {"id": "p3", "polished_text": "The annual sports day logistics were discussed.", "timeline_start": 120.0},
]
AGENDAS = [
    {"agenda_id": "A1", "title": "Hostel renovation", "description": "Budget and tender for the hostel"},
    {"agenda_id": "A2", "title": "Canteen contract", "description": "Renewal of the canteen vendor"},
]


def _map(monkeypatch, agendas, answer):
    provider = MagicMock()
    provider.assign_agenda_batch.side_effect = lambda **kw: {"assignments": answer(json.loads(kw["batch_json"]))}
    monkeypatch.setattr("services.ai_provider.get_provider", lambda: provider)
    monkeypatch.setattr(tes, "get_text_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(tes, "unload_text_embedder", lambda: None)
    result = RomService().map_points_to_agendas(
        polished_points=[dict(p) for p in POINTS], agendas=agendas, expanded_agendas=[],
        recording_id="rec-1", user_id="test_user",
    )
    return result, provider


def test_points_matching_no_agenda_go_to_other_discussion(monkeypatch):
    def answer(items):
        by_text = {i["enhanced_point"]: i["point_id"] for i in items}
        return [
            {"point_id": by_text[POINTS[0]["polished_text"]], "assigned_agenda_id": "A1", "confidence": "high"},
            {"point_id": by_text[POINTS[1]["polished_text"]], "assigned_agenda_id": "A1", "confidence": "high"},
            {"point_id": by_text[POINTS[2]["polished_text"]], "assigned_agenda_id": OTHER_AGENDA_ID, "confidence": "high"},
        ]

    result, provider = _map(monkeypatch, AGENDAS, answer)

    reference = json.loads(provider.assign_agenda_batch.call_args.kwargs["agenda_reference_json"])
    assert reference[-1]["agenda_id"] == OTHER_AGENDA_ID
    assert result["point_mappings"]["p3"] == OTHER_AGENDA_ID
    other = result["final_rom_agendas"][-1]
    assert other["agenda_id"] == OTHER_AGENDA_ID and other["is_other_agenda"] is True
    assert [p["id"] for p in other["discussion_points"]] == ["p3"]
    assert [a["agenda_id"] for a in result["final_rom_agendas"]] == ["A1", "A2", OTHER_AGENDA_ID]


def test_no_other_agenda_when_nothing_lands_there_or_only_the_default_exists(monkeypatch):
    answer = lambda items: [{"point_id": i["point_id"], "assigned_agenda_id": "A1", "confidence": "high"} for i in items]
    result, _ = _map(monkeypatch, AGENDAS, answer)
    assert [a["agenda_id"] for a in result["final_rom_agendas"]] == ["A1", "A2"]

    default = [{"agenda_id": "A1", "title": "General Discussion", "is_default_agenda": True}]
    result, provider = _map(monkeypatch, default, answer)
    reference = json.loads(provider.assign_agenda_batch.call_args.kwargs["agenda_reference_json"])
    assert [r["agenda_id"] for r in reference] == ["A1"]


def test_agendas_built_from_topics_keep_their_points(monkeypatch):
    agendas = [
        {**AGENDAS[0], "source": "discussion", "discussion_summary": {"point_ids": ["p1", "p2"]}},
        {**AGENDAS[1], "source": "discussion", "discussion_summary": {"point_ids": ["p-old-gone"]}},
    ]
    sent = []

    def answer(items):
        sent.extend(i["enhanced_point"] for i in items)
        return [{"point_id": i["point_id"], "assigned_agenda_id": "A2", "confidence": "high"} for i in items]

    result, _ = _map(monkeypatch, agendas, answer)

    assert sent == [POINTS[2]["polished_text"]]               # only the point no topic claimed
    assert result["point_mappings"] == {"p1": "A1", "p2": "A1", "p3": "A2"}
    by_pid = {a["point_id"]: a for a in result["batch_assignments"]}
    assert by_pid["p1"]["assignment_source"] == "topic_membership" and by_pid["p1"]["confidence"] == "high"
    assert len(result["similarity_matrix"]) == 3             # every point is still scored


def test_each_point_is_sent_with_its_neighbours(monkeypatch):
    batches = []

    def answer(items):
        batches.append(items)
        return [{"point_id": i["point_id"], "assigned_agenda_id": "A1", "confidence": "high"} for i in items]

    _map(monkeypatch, AGENDAS, answer)

    first, middle, last = batches[0]
    assert "previous_point" not in first and first["next_point"].startswith("Ravi will float")
    assert middle["previous_point"].startswith("Hostel renovation") and middle["next_point"].startswith("The annual sports")
    assert last["previous_point"].startswith("Ravi will float") and "next_point" not in last


def test_centred_similarity_separates_agendas_that_raw_scores_blur():
    common = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.float32) * 2.0
    e1, e2 = np.eye(8, dtype=np.float32)[6], np.eye(8, dtype=np.float32)[7]
    points = [{"id": "a", "polished_text": "pa"}, {"id": "b", "polished_text": "pb"}, {"id": "c", "polished_text": "pc"}]
    agendas = [{"agenda_id": "A1", "title": "ta"}, {"agenda_id": "A2", "title": "tb"}]
    vectors = {"pa": common + e1, "pb": common + e2, "pc": common + e1, "ta": common + e1, "tb": common + e2}
    embedder = FakeEmbedder(vectors)

    raw = compute_point_agenda_similarity(points, agendas, embedder)
    centred = compute_point_agenda_similarity(points, agendas, embedder, center=True)

    raw_margin = raw["candidates"][0]["margin"]
    centred_margin = centred["candidates"][0]["margin"]
    assert raw["candidates"][0]["candidates"][0]["agenda_id"] == "A1"
    assert centred["candidates"][0]["candidates"][0]["agenda_id"] == "A1"
    assert centred_margin > 3 * raw_margin
