"""
Stage 3 discussion analysis (services/agenda_analysis.py) and its use in agenda creation.
A bag-of-words embedder stands in for the embedding model and a scripted provider for the
LLM, so grouping, validation and matching are deterministic.
"""
import copy
import json
import re
import zlib

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import services.agenda_analysis as aa
from services.rom_service import _agenda_similarity_facets, _join_detail, normalize_agendas, rom_service

DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "and", "was", "will", "with", "for", "that", "this", "there", "is", "a", "to", "of", "in", "it", "on"}


class BowEmbedder:
    _dim = DIM

    def load(self):
        pass

    def encode_batch(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(str(text).lower()):
                if len(tok) >= 3 and tok not in _STOP:
                    out[i, zlib.crc32(tok.encode()) % DIM] += 1.0
        return out

    encode = encode_batch


DB = "database cluster migration to the Mumbai region cutover window"
BUDGET = "quarterly budget forecast finance variance"
POINTS = [
    {"id": "p1", "polished_text": f"Arjun explained the {DB} plan.", "timeline_start": 10, "timeline_end": 40,
     "speakers": ["Arjun"], "technical_terms": ["Mumbai region"]},
    {"id": "p2", "polished_text": f"Neha reviewed the {BUDGET} numbers.", "timeline_start": 50, "timeline_end": 80,
     "speakers": ["Neha"]},
    {"id": "p3", "polished_text": f"It was decided that the {DB} happens on 12 March.", "timeline_start": 90,
     "timeline_end": 110, "speakers": ["Arjun"], "dates": ["12 March"]},
    {"id": "p4", "polished_text": f"Neha will revise the {BUDGET} figures.", "timeline_start": 120, "timeline_end": 140,
     "speakers": ["Neha"], "action_owner": "Neha"},
    {"id": "p5", "polished_text": f"There is a risk of delay to the {DB}.", "timeline_start": 150, "timeline_end": 165,
     "speakers": ["Ravi"]},
    {"id": "p6", "polished_text": f"Neha will revise the {BUDGET} figures.", "timeline_start": 170, "timeline_end": 180,
     "speakers": ["Neha"], "action_owner": "Neha"},
]


def _db_topic(cluster_id):
    return {
        "cluster_id": cluster_id,
        "title": "Database Cluster Migration to Mumbai Region",
        "description": "The team discussed the database cluster migration to the Mumbai region. The cutover was decided "
                       "for 12 March. A spend of 40 lakh was approved.",
        "supporting_context": "Cutover window in the Mumbai region.",
        "decisions": [{"text": "Cutover on 12 March", "point_ids": ["p3"]},
                      {"text": "Approve the 40 lakh spend", "point_ids": ["p3"]}],
        "risks": [{"text": "Migration may delay the cutover window", "point_ids": ["p5"]}],
        "keywords": ["database cluster", "Mumbai region", "FY27"],
        "point_ids": ["p1", "p3"],  # p5 left out: validation must still cover it
    }


def _budget_topic(cluster_id):
    return {
        "cluster_id": cluster_id,
        "title": "Q3 Budget Forecast Revision",
        "description": "Neha reviewed the quarterly budget forecast and will revise the figures.",
        "action_items": [{"text": "Neha to revise the quarterly budget forecast", "point_ids": ["p4"]}],
        "keywords": ["budget forecast", "finance variance"],
        "point_ids": ["p2", "p4", "p6", "bogus"],
    }


class AnalysisProvider:
    def __init__(self, fail_synthesis=False, duplicate_titles=False):
        self.fail_synthesis = fail_synthesis
        self.duplicate_titles = duplicate_titles
        self.classify_calls = []
        self.synth_calls = []

    def classify_discussion_points(self, points_json):
        batch = json.loads(points_json)
        self.classify_calls.append(batch)
        known = {
            "p1": {"primary_type": "discussion", "evidence": "explained"},
            "p2": {"primary_type": "discussion", "evidence": "not in the text at all"},
            "p3": {"primary_type": "Decision", "types": ["decision", "action"], "evidence": "It was decided"},
        }
        return {"classifications": [dict(known[p["id"]], point_id=p["id"]) for p in batch if p["id"] in known]}

    def synthesize_agenda_topics(self, clusters_json):
        clusters = json.loads(clusters_json)
        self.synth_calls.append(clusters)
        if self.fail_synthesis:
            return None
        topics = []
        for cluster in clusters:
            text = " ".join(p["text"] for p in cluster["points"]).lower()
            topic = _db_topic(cluster["cluster_id"]) if "database" in text else _budget_topic(cluster["cluster_id"])
            topics.append(topic)
        return {"topics": topics}

    def generate_rom_agendas(self, agenda_text, context="", **kwargs):
        return {"agendas": [{"title": "Database migration", "description": "Database cluster migration to the Mumbai region",
                             "keywords": ["database", "cluster", "migration"]}]}

    def unload_model(self):
        pass


# ── Grouping and classification ──────────────────────────────────────────────

def _grouping_sims(points):
    raw = aa._embed(BowEmbedder(), [aa._point_text(p) for p in points])
    return aa.grouping_similarity(aa.center_vectors(raw), [aa._start(p) for p in points])


def test_clustering_groups_related_points_in_time_order():
    order = aa.chronological_order(POINTS)
    clusters = aa.cluster_points(_grouping_sims(POINTS), order)
    groups = [[POINTS[i]["id"] for i in c] for c in clusters]
    assert groups == [["p1", "p3", "p5"], ["p2", "p4", "p6"]]


def test_centering_separates_points_that_share_meeting_vocabulary():
    # Every point repeats the meeting's common wording; only the subject differs.
    common = "committee meeting council review discussion agenda item proposal members "
    points = [
        {"id": f"d{i}", "polished_text": common * 3 + f"database cluster migration region {i}"} for i in range(3)
    ] + [
        {"id": f"b{i}", "polished_text": common * 3 + f"budget forecast finance variance {i}"} for i in range(3)
    ]
    raw = aa._embed(BowEmbedder(), [aa._point_text(p) for p in points])
    assert (raw @ raw.T).min() > 0.8  # raw cosines: everything looks alike
    clusters = aa.cluster_points(aa.grouping_similarity(aa.center_vectors(raw), [0.0] * 6), list(range(6)))
    assert sorted(sorted(points[i]["id"] for i in c) for c in clusters) == [["b0", "b1", "b2"], ["d0", "d1", "d2"]]


def test_time_proximity_is_added_when_timestamps_exist():
    vecs = aa._normalize(np.eye(3, dtype=np.float32))
    sims = aa.grouping_similarity(vecs, [0.0, 10.0, 100.0])
    assert sims[0, 1] > sims[0, 2] > 0.0
    assert np.allclose(aa.grouping_similarity(vecs, [0.0, 0.0, 0.0]), vecs @ vecs.T)


def test_near_duplicates_need_same_wording_and_no_conflicting_facts():
    raw = aa._embed(BowEmbedder(), [aa._point_text(p) for p in POINTS])
    dups = aa.find_near_duplicates(raw @ raw.T, aa.chronological_order(POINTS), POINTS)
    assert {POINTS[d]["id"]: POINTS[o]["id"] for d, o in dups.items()} == {"p6": "p4"}

    a = {"id": "x1", "polished_text": "Ordinance 2026-0108 was moved for a due pass recommendation."}
    b = {"id": "x2", "polished_text": "Ordinance 2026-0083 was moved for a due pass recommendation."}
    same = np.ones((2, 2), dtype=np.float32)
    assert aa.find_near_duplicates(same, [0, 1], [a, b]) == {}


def test_topic_count_is_capped():
    rng = np.random.default_rng(0)
    vecs = aa._normalize(rng.normal(size=(20, 16)))
    clusters = aa.cluster_points(vecs @ vecs.T, list(range(20)), threshold=0.99, max_topics=5)
    assert len(clusters) <= 5
    assert sorted(i for c in clusters for i in c) == list(range(20))


def test_default_threshold_is_the_meeting_average():
    sims = np.array([[1.0, 0.9, 0.1, 0.1], [0.9, 1.0, 0.1, 0.1], [0.1, 0.1, 1.0, 0.9], [0.1, 0.1, 0.9, 1.0]])
    assert aa.cluster_points(sims, [0, 1, 2, 3]) == [[0, 1], [2, 3]]


@pytest.mark.parametrize("text,expected", [
    ("The committee approved the revised tariff.", "decision"),
    ("Ravi will send the vendor list.", "action_item"),
    ("There is a concern about supply delays.", "risk"),
    ("Is the licence renewal still pending?", "question"),
    ("The team shared an overview of last month.", "discussion"),
])
def test_heuristic_classification(text, expected):
    assert aa.heuristic_classification({"polished_text": text})["primary_type"] == expected


def test_classification_uses_llm_and_falls_back_per_point():
    result = aa.classify_points(AnalysisProvider(), POINTS)
    assert result["p3"] == {"primary_type": "decision", "types": ["decision", "action_item"],
                            "evidence": "It was decided", "source": "llm"}
    assert result["p2"]["evidence"] == ""  # evidence not found in the text is dropped
    assert result["p4"]["source"] == "heuristic" and result["p4"]["primary_type"] == "action_item"
    assert result["p5"]["primary_type"] == "risk"


def test_classification_reuses_model_set_stage2_types():
    class CountingProvider:
        payloads = []

        def classify_discussion_points(self, points_json):
            self.payloads.append([p["id"] for p in json.loads(points_json)])
            return {"classifications": [{"point_id": "p1", "primary_type": "risk"}]}  # p1 is not asked

    points = copy.deepcopy(POINTS)
    points[0].update(type="question_answer", type_source="llm", include_action_point=False)
    points[1].update(type="update", type_source="user", include_action_point=True)
    points[2].update(type="decision", type_source="heuristic")  # text cues only: still classified by the LLM
    provider = CountingProvider()
    result = aa.classify_points(provider, points)

    assert provider.payloads == [["p3", "p4", "p5", "p6"]]
    assert result["p1"] == {"primary_type": "question", "types": ["question"], "evidence": "", "source": "stage2"}
    assert result["p2"]["primary_type"] == "action_item" and result["p2"]["source"] == "stage2"
    assert result["p3"]["source"] == "heuristic"


# ── Topics and validation ────────────────────────────────────────────────────

def test_analysis_builds_validated_topics():
    provider = AnalysisProvider()
    analysis = aa.analyze_discussion(provider, BowEmbedder(), copy.deepcopy(POINTS))

    # Near-duplicate p6 is sent marked (it may add a detail) and stays with p4's topic.
    sent = {p["id"]: p for batch in provider.synth_calls for c in batch for p in c["points"]}
    assert sent["p6"]["near_duplicate_of"] == "p4" and analysis["near_duplicates"] == {"p6": "p4"}

    db, budget = analysis["topics"]
    assert db["topic_id"] == "T1" and budget["topic_id"] == "T2"
    assert db["point_ids"] == ["p1", "p3", "p5"]
    assert db["added_by_validation"] == ["p5"]
    assert budget["point_ids"] == ["p2", "p4", "p6"]

    # Invented facts removed: "40 lakh" sentence and decision, "FY27" keyword, "Q3" in the title.
    assert "40" not in db["description"] and "12 March" in db["description"]
    assert [d["text"] for d in db["decisions"]] == ["Cutover on 12 March"]
    assert "FY27" not in db["keywords"]
    assert budget["title"] == "Budget Forecast Revision"

    report = analysis["validation"]
    assert report["topics_in"] == 2 and report["topics_out"] == 2
    assert report["uncovered_point_ids"] == ["p5"]
    assert report["coverage_ratio"] == pytest.approx(5 / 6, abs=1e-4)
    removed_fields = {r["field"] for r in report["unsupported_removed"]}
    assert {"description", "decisions", "title"} <= removed_fields

    assert db["type_counts"]["decision"] == 1 and db["type_counts"]["risk"] == 1
    assert db["time_start"] == 10 and db["speakers"] == ["Arjun", "Ravi"]
    assert analysis["type_counts"]["action_item"] == 2


def test_duplicate_topics_are_merged(monkeypatch):
    # Force three groups so two of them describe the same subject.
    monkeypatch.setattr(aa, "cluster_points", lambda vecs, order, threshold, max_topics=12: [[0, 2], [1, 3, 5], [4]])
    analysis = aa.analyze_discussion(AnalysisProvider(), BowEmbedder(), copy.deepcopy(POINTS))
    titles = [t["title"] for t in analysis["topics"]]
    assert titles.count("Database Cluster Migration to Mumbai Region") == 1
    assert analysis["validation"]["merged"][0]["reason"] == "same title"
    db = analysis["topics"][0]
    assert db["point_ids"] == ["p1", "p3", "p5"]


def test_point_left_out_by_the_llm_returns_to_its_own_group(monkeypatch):
    # p5 is grouped with the budget points; neither topic lists it, so validation must put it
    # back into the topic written for its group, not into the nearest topic.
    monkeypatch.setattr(aa, "cluster_points", lambda sims, order, threshold, max_topics=12: [[0, 2], [1, 3, 4, 5]])

    class BudgetFirstProvider(AnalysisProvider):
        def synthesize_agenda_topics(self, clusters_json):
            clusters = json.loads(clusters_json)
            return {"topics": [
                _budget_topic(c["cluster_id"]) if "budget" in " ".join(p["text"] for p in c["points"]).lower()
                else _db_topic(c["cluster_id"]) for c in clusters
            ]}

    analysis = aa.analyze_discussion(BudgetFirstProvider(), BowEmbedder(), copy.deepcopy(POINTS))
    db, budget = analysis["topics"]
    assert "p5" in budget["added_by_validation"] and "p5" not in db["point_ids"]
    assert analysis["validation"]["uncovered_point_ids"] == ["p5"]


def test_failed_synthesis_falls_back_to_grounded_topics():
    analysis = aa.analyze_discussion(AnalysisProvider(fail_synthesis=True), BowEmbedder(), copy.deepcopy(POINTS))
    assert len(analysis["topics"]) == 2
    for topic in analysis["topics"]:
        assert topic["generated_by"] == "fallback"
        assert topic["title"].startswith("Discussion on")
        assert topic["description"]
    assert analysis["topics"][0]["decisions"][0]["point_ids"] == ["p3"]


# ── Matching ─────────────────────────────────────────────────────────────────

def test_topics_are_matched_with_scores_and_evidence():
    analysis = aa.analyze_discussion(AnalysisProvider(), BowEmbedder(), copy.deepcopy(POINTS))
    agendas = normalize_agendas([{"title": "Database migration", "description": "Database cluster migration to the Mumbai region",
                                  "keywords": ["database", "cluster", "migration"]}])
    points_by_id = {p["id"]: p for p in POINTS}
    matches = aa.match_topics_to_agendas(analysis["topics"], agendas, BowEmbedder(), points_by_id)

    db, budget = matches
    assert db["status"] == "matched" and db["matched_agenda_id"] == "A1"
    assert db["score"] > budget["score"]
    assert db["relative_score"] > 0 > -aa.TOPIC_UNMATCHED_GAP > budget["relative_score"]
    assert db["evidence"] and db["evidence"][0]["point_id"] in {"p1", "p3", "p5"}
    assert db["best_facet"]
    assert budget["status"] == "unmatched" and budget["matched_agenda_id"] is None
    assert budget["vote_share"] == 1.0  # every point still prefers the only agenda, but weakly

    suggestions = aa.build_suggested_agendas(analysis["topics"], matches)
    assert [s["title"] for s in suggestions] == ["Budget Forecast Revision"]
    assert suggestions[0]["suggestion_id"] == "S1" and suggestions[0]["point_ids"] == ["p2", "p4", "p6"]
    assert suggestions[0]["discussion_summary"]["action_items"]


# ── Agenda creation ──────────────────────────────────────────────────────────

class _Store:
    _meta = []

    def load_or_create(self):
        pass

    def search(self, *args, **kwargs):
        return []


@pytest.fixture
def stage3_env(monkeypatch):
    import services.ai_provider as ai_provider
    import services.rag_pipeline as rag_pipeline
    import services.text_embedding_service as tes
    import services.vector_store as vector_store

    provider = AnalysisProvider()
    saved = []
    cache = {"value": None}
    monkeypatch.setattr(ai_provider, "get_provider", lambda: provider)
    monkeypatch.setattr(tes, "get_text_embedder", lambda: BowEmbedder())
    monkeypatch.setattr(tes, "unload_text_embedder", lambda: None)
    monkeypatch.setattr(vector_store, "get_meeting_context_store", lambda *a, **k: _Store())
    monkeypatch.setattr(vector_store, "get_global_context_store", lambda *a, **k: _Store())
    monkeypatch.setattr(rag_pipeline, "_load_parsed_agenda", lambda rid: cache["value"])
    monkeypatch.setattr(rag_pipeline, "_save_parsed_agenda", lambda rid, uid, items: saved.append(items))
    return {"provider": provider, "saved": saved, "cache": cache}


def test_without_agenda_document_topics_become_agendas(stage3_env):
    result = rom_service.generate_agendas_only(
        agenda_text=None, recording_id="rec-1", user_id="user-1",
        polished_points=copy.deepcopy(POINTS), analyze_discussion=True,
    )
    agendas = result["agendas"]
    assert [a["agenda_id"] for a in agendas] == ["A1", "A2"]
    assert agendas[0]["title"] == "Database Cluster Migration to Mumbai Region"
    assert agendas[0]["source"] == "discussion"
    assert agendas[0]["discussion_summary"]["point_ids"] == ["p1", "p3", "p5"]
    assert "database cluster" in agendas[0]["keywords"]
    assert result["discussion_analysis"]["mode"] == "derived_agendas"
    assert result["discussion_analysis"]["topic_matches"][1]["status"] == "derived"
    assert result["suggested_agendas"] == []
    assert len(result["expanded_agendas"]) == 2
    assert stage3_env["saved"] == []  # derived agendas are not written to the shared agenda cache


def test_with_existing_agendas_unmatched_topics_are_suggested(stage3_env):
    stage3_env["cache"]["value"] = [{"topic": "Database migration", "details": "Database cluster migration to the Mumbai region",
                                     "keywords": ["database", "cluster", "migration"]}]
    result = rom_service.generate_agendas_only(
        agenda_text=None, recording_id="rec-1", user_id="user-1",
        polished_points=copy.deepcopy(POINTS), analyze_discussion=True,
    )
    assert [a["title"] for a in result["agendas"]] == ["Database migration"]
    analysis = result["discussion_analysis"]
    assert analysis["mode"] == "matched_existing"
    assert [m["status"] for m in analysis["topic_matches"]] == ["matched", "unmatched"]
    assert [s["title"] for s in result["suggested_agendas"]] == ["Budget Forecast Revision"]


def test_analysis_off_keeps_previous_behaviour(stage3_env):
    result = rom_service.generate_agendas_only(
        agenda_text=None, recording_id="rec-1", user_id="user-1",
        polished_points=copy.deepcopy(POINTS), analyze_discussion=False,
    )
    assert [a["title"] for a in result["agendas"]] == ["General Discussion"]
    assert result["discussion_analysis"] is None and result["suggested_agendas"] == []
    assert stage3_env["provider"].classify_calls == []


def test_llm_agendas_without_ids_are_normalised(stage3_env):
    result = rom_service.generate_agendas_only(
        agenda_text="1. Database migration", recording_id="rec-1", user_id="user-1",
        polished_points=copy.deepcopy(POINTS), analyze_discussion=True,
    )
    assert result["agendas"][0]["agenda_id"] == "A1"
    assert result["agendas"][0]["speaker"] is None and result["agendas"][0]["presenter"] is None
    assert [s["title"] for s in result["suggested_agendas"]] == ["Budget Forecast Revision"]


@pytest.mark.asyncio
async def test_adding_a_suggested_agenda_keeps_its_terms():
    from routers.rom_router import CreateAgendaItemRequest, create_stage3_agenda

    data = {"stage3": {
        "agendas": [{"agenda_id": "A1", "title": "Database migration"}],
        "suggested_agendas": [{"suggestion_id": "S1", "topic_id": "T2", "title": "Budget Forecast Revision",
                               "description": "Neha reviewed the quarterly budget forecast.",
                               "keywords": ["budget forecast"], "related_concepts": ["finance variance"],
                               "discussion_summary": {"point_ids": ["p2", "p4"]}}],
        "discussion_analysis": {"topic_matches": [{"topic_id": "T2", "status": "unmatched", "matched_agenda_id": None}]},
    }}
    with patch("routers.rom_router._get_rom_data", AsyncMock(return_value=data)), \
         patch("routers.rom_router._save_rom_data", AsyncMock()), \
         patch("routers.rom_router._validate_user_id", return_value="test-user"), \
         patch("services.rag_pipeline._save_parsed_agenda", MagicMock()):
        res = await create_stage3_agenda("rec-1", CreateAgendaItemRequest(title="Budget Forecast Revision", suggestion_id="S1"),
                                         current_user={}, db=MagicMock())

    agenda = res["agenda"]
    assert agenda["agenda_id"] == "A2"
    assert agenda["keywords"] == ["budget forecast"] and agenda["related_concepts"] == ["finance variance"]
    assert agenda["description"] == "Neha reviewed the quarterly budget forecast."
    assert agenda["source"] == "discussion"
    assert data["stage3"]["suggested_agendas"] == []
    match = data["stage3"]["discussion_analysis"]["topic_matches"][0]
    assert match["status"] == "added" and match["matched_agenda_id"] == "A2"


# ── Helpers ──────────────────────────────────────────────────────────────────

def test_normalize_agendas_assigns_ids_and_merges_duplicate_titles():
    agendas = normalize_agendas([
        {"agenda_id": "A2", "title": "Budget", "keywords": ["capex"]},
        {"title": "Hiring", "presenter": "Meena"},
        {"agenda_id": "A2", "title": "Vendors"},
        {"title": "budget", "keywords": ["opex"], "description": "Budget details"},
        {"title": "  "},
    ])
    assert [(a["agenda_id"], a["title"]) for a in agendas] == [("A2", "Budget"), ("A1", "Hiring"), ("A3", "Vendors")]
    assert agendas[0]["keywords"] == ["capex", "opex"] and agendas[0]["description"] == "Budget details"
    assert agendas[1]["speaker"] == "Meena" and agendas[1]["related_concepts"] == []
    kept = normalize_agendas([{"title": "Budget"}, {"title": "Budget"}], merge_duplicates=False)
    assert [a["agenda_id"] for a in kept] == ["A1", "A2"]


def test_previous_meeting_facet_skips_bookkeeping_values():
    expansion = {"agenda_id": "A1", "found_in_previous_meeting": True, "previous_discussion": "Vendor shortlist reviewed",
                 "pending_work": ["Finalise SLA"]}
    assert _join_detail(expansion) == "Vendor shortlist reviewed Finalise SLA"
    facets = dict(_agenda_similarity_facets({"title": "Vendors"}, {"previous_meeting_summary": expansion}))
    assert facets["previous_meeting"] == "Vendor shortlist reviewed Finalise SLA"


def test_point_types_reach_the_final_rom(monkeypatch):
    import services.ai_provider as ai_provider
    import services.text_embedding_service as tes

    provider = MagicMock()
    monkeypatch.setattr(ai_provider, "get_provider", lambda: provider)
    monkeypatch.setattr(tes, "unload_text_embedder", lambda: None)
    monkeypatch.setattr(type(rom_service), "_assign_points_to_agendas", lambda self, **kw: {
        "batch_assignments": [{"point_id": "p3", "assigned_agenda_id": "A1", "confidence": "high"}],
        "candidate_results": [], "similarity_matrix": [], "similarity_agenda_ids": ["A1"],
    })
    result = rom_service.map_points_to_agendas(
        polished_points=[POINTS[2]], agendas=[{"agenda_id": "A1", "title": "Database migration"}], expanded_agendas=[],
        recording_id="rec-1", user_id="user-1",
        point_types={"p3": {"primary_type": "decision", "types": ["decision", "action_item"]}},
    )
    point = result["final_rom_agendas"][0]["discussion_points"][0]
    assert point["point_type"] == "decision" and point["point_types"] == ["decision", "action_item"]
