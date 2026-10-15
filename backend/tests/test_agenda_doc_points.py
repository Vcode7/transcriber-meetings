"""
Stage 3 with an agenda document (services/agenda_doc_points.py): doc points from the PPT Testing
analysis are assigned to agendas, used as mapping context, and refine each agenda's matched points.
A bag-of-words embedder and a scripted LLM keep it deterministic.
"""
import copy
import json
import re
import zlib
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest

import services.text_embedding_service as tes
from ppt_fixtures import MEETING_DATE, build_pptx
from services import agenda_doc_points as adp
from services.ppt_insights import extract_document_slides, slides_to_dicts
from services.ppt_matching import build_synthesis_prompt
from services.rom_service import RomService, _agenda_reference_entry

DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BowEmbedder:
    _dim = DIM
    loaded_model_name = "bow"

    def load(self):
        pass

    def encode_batch(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOKEN_RE.findall(str(t).lower()):
                if len(tok) >= 3:
                    out[i, zlib.crc32(tok.encode()) % DIM] += 1.0
        return out

    encode = encode_batch


AGENDAS = [
    {"agenda_id": "A1", "title": "Project Atlas build status", "description": "SAP integration and data migration",
     "keywords": [], "related_concepts": [], "alternative_terminology": [], "expected_themes": []},
    {"agenda_id": "A2", "title": "Beacon mobile app", "description": "Pilot users and store submission",
     "keywords": [], "related_concepts": [], "alternative_terminology": [], "expected_themes": []},
]


def _insight(ref, slide, slide_title, project, headline, summary, priority, dates=()):
    return {"id": f"ins-{ref.lower()}", "ref": ref, "project": project, "slides": [slide], "slide_title": slide_title,
            "type": "status", "status": "ongoing", "headline": headline, "summary": summary, "attention": "",
            "dates": list(dates), "priority": priority, "doc_name": "agenda.pptx"}


INSIGHTS = [
    _insight("P1", 4, "Project Atlas – Build Status", "Project Atlas", "Data migration dry-run 2 slipped",
             "Dry-run 2 moved because of vendor data quality issues.", 0.9),
    _insight("P2", 5, "Project Beacon – Mobile App", "Project Beacon", "Store submission planned",
             "App store submission is planned after the beta.", 0.8),
    _insight("P3", 5, "Project Beacon – Mobile App", "Project Beacon", "Crash rate reduced",
             "Crash rate fell from 4.1% to 1.2% since beta.", 0.8),
    _insight("P4", 4, "Project Atlas – Build Status", "Project Atlas", "API gateway deployed",
             "The API gateway is in production.", 0.5),
    _insight("P5", 9, "Long-term Vision 2030", "", "Five regions by 2030",
             "Expand the platform to five regions.", 0.3),
]

POINTS = [
    {"id": "p1", "polished_text": "Ravi said the Atlas data migration dry-run 2 slipped because of vendor data quality issues."},
    {"id": "p2", "polished_text": "The team discussed the Beacon store submission timeline with the pilot users."},
]

REFINED_SUMMARY = "Dry-run 2 of the data migration slipped because of vendor data quality issues; Ravi will chase the vendor."


def scripted_infer(prompt, max_tokens):
    if "DISCUSSION POINT (from the meeting transcript" in prompt:
        discussion = re.search(r'DISCUSSION POINT[^\n]*\n"(.*?)"', prompt, re.S).group(1)
        lines = re.findall(r"^(P\d+) (.*)$", prompt, re.M)
        wanted = "migration" if "migration" in discussion else "Store submission" if "store" in discussion else None
        return json.dumps({"matches": [{"id": ref, "confidence": "high", "reason": "same subject"}
                                       for ref, line in lines if wanted and wanted in line]})
    if "Refine one discussion point" in prompt:
        if "ignore-the-discussion" in prompt:            # a rewrite that drops what was said
            return json.dumps({"headline": "Slide status", "status": "ongoing",
                               "summary": "The presentation reports the programme status for the quarter.", "attention": ""})
        if "migration" in prompt:
            return json.dumps({"headline": "Data migration dry-run 2 slipped", "status": "delayed",
                               "summary": REFINED_SUMMARY, "attention": ""})
        return json.dumps({"headline": "Store submission planned", "status": "planned",
                           "summary": "The Beacon store submission is planned after the beta with pilot users.",
                           "attention": ""})
    return "{}"


# ── Pieces ────────────────────────────────────────────────────────────────────

def test_long_detail_only_changes_the_synthesis_prompt():
    matched = [INSIGHTS[0]]
    standard = build_synthesis_prompt("discussion", matched, MEETING_DATE)
    long = build_synthesis_prompt("discussion", matched, MEETING_DATE, detail="long")
    assert "<2-3 sentences>" in standard and "presentation's names and terms" not in standard
    assert "Write one meeting-prep point" in standard
    assert "presentation's names and terms" in long and "Keep every fact, decision, vote, name" in long
    assert "MEETING DISCUSSION POINT (the base" in long


def test_agenda_reference_entry_adds_document_points_only_when_given():
    assert "document_points" not in _agenda_reference_entry(AGENDAS[0])
    entry = _agenda_reference_entry(AGENDAS[0], ["Data migration dry-run 2 slipped"])
    assert entry["document_points"] == ["Data migration dry-run 2 slipped"]


def test_doc_points_go_to_their_agenda_and_noise_is_dropped():
    assigned = adp.assign_insights_to_agendas(copy.deepcopy(INSIGHTS), AGENDAS, BowEmbedder())
    assert {i["ref"] for i in assigned["A1"]} == {"P1", "P4"}
    assert {i["ref"] for i in assigned["A2"]} == {"P2", "P3"}
    assert all(i["agenda_match"] == "slide_title" for items in assigned.values() for i in items)
    # the "Long-term Vision" slide matches no agenda
    assert "P5" not in {i["ref"] for items in assigned.values() for i in items}

    doc_points = adp.to_agenda_doc_points(assigned, {"A1": {"doc_name": "budget.pdf", "points": ["Manual point"]}})
    assert doc_points["A1"]["points"][0] == "Manual point" and len(doc_points["A1"]["points"]) == 3
    assert doc_points["A2"]["doc_name"] == "agenda.pptx"


def test_refinement_rewrites_matches_keeps_important_and_drops_noise():
    points = [dict(POINTS[0], text=POINTS[0]["polished_text"])]
    a1 = [INSIGHTS[0], INSIGHTS[3]]                         # P1 matches p1; P4 (priority 0.5) matches nothing
    refined, stats = adp.refine_agenda_points(points, a1, infer=scripted_infer, meeting=MEETING_DATE)
    assert len(refined) == 1                                 # P4 is noise: not added
    p1 = refined[0]
    assert p1["doc_refined"] and p1["doc_slides"] == [4]
    assert p1["text"] == p1["polished_text"] == REFINED_SUMMARY        # the refined discussion point itself
    assert p1["original_text"] == POINTS[0]["polished_text"]
    assert stats["refined_points"] == 1 and stats["important_doc_points_added"] == 0

    points = [dict(POINTS[1], text=POINTS[1]["polished_text"])]
    refined, stats = adp.refine_agenda_points(points, [INSIGHTS[1], INSIGHTS[2]], infer=scripted_infer, meeting=MEETING_DATE)
    assert refined[0]["doc_refined"]
    doc_only = [p for p in refined if p.get("is_doc_point")]
    assert len(doc_only) == 1 and "Crash rate" in doc_only[0]["text"]     # important (0.8) but unmatched → kept
    assert doc_only[0]["speaker"] == "From Document" and doc_only[0]["doc_slides"] == [5]


def test_refinement_that_drops_the_discussion_is_rejected():
    original = "Ravi said the Atlas data migration dry-run 2 slipped; ignore-the-discussion vendor quality."
    points = [{"id": "p1", "text": original, "polished_text": original}]
    refined, stats = adp.refine_agenda_points(points, [INSIGHTS[0]], infer=scripted_infer, meeting=MEETING_DATE)
    assert refined[0]["text"] == original and not refined[0].get("doc_refined")
    assert refined[0]["doc_match"]["refinement_rejected"] and stats["refinements_rejected"] == 1
    assert adp.content_coverage(original, original) == 1.0


def test_refinement_must_keep_the_dates_that_were_said():
    # Seen with a small local model: the slide's expiry date replaced the date stated in the meeting.
    said = "Amendment one extends the approval period from June 17th to August 17th, 2026."
    mixed_up = "Amendment 1 extends the approval period from June 17th to August 16th, 2026."
    assert adp.dropped_dates(said, mixed_up) == ["August 17th, 2026"]
    assert adp.dropped_dates(said, said + " The period expires 16-Aug-2026.") == []


def test_duplicates_are_merged_in_order():
    pts = [
        {"id": "a", "text": "Budget approved for the QA team expansion."},
        {"id": "b", "text": "Hiring plan reviewed."},
        {"id": "c", "text": "The budget was approved for QA team expansion.", "doc_refined": True},
    ]
    out = adp._merge_duplicates(pts)
    assert [p["id"] for p in out] == ["b", "c"]
    assert out[1]["merged_point_ids"] == ["a"]


def test_short_point_is_not_dropped_into_a_longer_one():
    pts = [
        {"id": "a", "text": "Budget approved for the QA team expansion."},
        {"id": "b", "text": "Budget approved for the QA team expansion; Ravi will hire 12 testers by March.",
         "doc_refined": True},
    ]
    assert [p["id"] for p in adp._merge_duplicates(pts)] == ["a", "b"]


def test_points_stating_different_facts_are_not_merged():
    pts = [{"id": "a", "text": "The hostel budget of 42 lakh was approved."},
           {"id": "b", "text": "The hostel budget of 45 lakh was approved."}]
    assert len(adp._merge_duplicates(pts)) == 2


def test_merged_duplicate_keeps_the_others_owner_and_actions():
    pts = [
        {"id": "a", "text": "Ravi will send the revised quotation to the committee.", "speakers": ["Ravi"],
         "action_owner": "Ravi", "original_point_ids": ["s1"],
         "action_items": [{"task": "Send the revised quotation", "assignee": "Ravi"}]},
        {"id": "doc", "text": "Ravi will send the revised quotation to the committee.", "is_doc_point": True,
         "speakers": ["From Document"]},
        {"id": "b", "text": "Ravi will send the Committee the revised quotation.", "speakers": ["Meena"],
         "action_owner": "Meena", "original_point_ids": ["s2"], "doc_refined": True,
         "action_items": [{"task": "Confirm the committee meeting slot", "assignee": "Meena"}]},
    ]
    out = adp._merge_duplicates(pts)
    assert [p["id"] for p in out] == ["b"]                       # transcript/refined wins over doc-only
    kept = out[0]
    assert kept["speakers"] == ["Meena", "Ravi"] and kept["action_owner"] == "Meena, Ravi"
    assert [a["task"] for a in kept["action_items"]] == ["Confirm the committee meeting slot", "Send the revised quotation"]
    assert kept["original_point_ids"] == ["s2", "s1"] and set(kept["merged_point_ids"]) == {"a", "doc"}


def test_document_analysis_is_cached_per_meeting_date(tmp_path):
    from test_ppt_testing_api import scripted_infer as slide_infer

    deck = build_pptx(str(tmp_path / "agenda.pptx"))
    with open(deck, "rb") as fh:
        text_all, slides = extract_document_slides(fh.read(), "agenda.pptx")
    docs = [{"filename": "agenda.pptx", "extracted_text": text_all, "slides": slides_to_dicts(slides)}]
    calls = []

    def counting(prompt, max_tokens):
        calls.append(prompt)
        return slide_infer(prompt, max_tokens)

    first = adp.analyze_agenda_documents(docs, MEETING_DATE, counting)
    assert [i["slides"] for i in first] == [[4], [5]] and first[0]["doc_name"] == "agenda.pptx"
    n = len(calls)
    again = adp.analyze_agenda_documents(docs, MEETING_DATE, counting)
    assert len(calls) == n and [i["headline"] for i in again] == [i["headline"] for i in first]
    adp.analyze_agenda_documents(docs, date(2026, 11, 1), counting)
    assert len(calls) > n                                    # another meeting date → re-analysed


# ── map_points_to_agendas ─────────────────────────────────────────────────────

@pytest.fixture
def mapping_env(monkeypatch):
    provider = MagicMock()
    provider._infer.side_effect = lambda prompt, max_tokens, *_: scripted_infer(prompt, max_tokens)
    monkeypatch.setattr("services.ai_provider.get_provider", lambda: provider)
    monkeypatch.setattr(tes, "get_text_embedder", lambda: BowEmbedder())
    monkeypatch.setattr(tes, "unload_text_embedder", lambda: None)
    monkeypatch.setattr(tes, "embed_chunks", lambda texts: BowEmbedder().encode_batch(texts))
    seen = {}

    def fake_assign(self, **kw):
        seen.update(kw)
        return {"batch_assignments": [{"point_id": "p1", "assigned_agenda_id": "A1", "confidence": "high"},
                                      {"point_id": "p2", "assigned_agenda_id": "A2", "confidence": "high"}],
                "candidate_results": [], "similarity_matrix": [], "similarity_agenda_ids": ["A1", "A2"]}
    monkeypatch.setattr(RomService, "_assign_points_to_agendas", fake_assign)
    return seen


def _map(**extra):
    return RomService().map_points_to_agendas(
        polished_points=copy.deepcopy(POINTS), agendas=copy.deepcopy(AGENDAS), expanded_agendas=[],
        recording_id="rec-1", user_id="user-1", **extra)


def test_without_agenda_document_mapping_is_unchanged(mapping_env):
    result = _map()
    assert mapping_env["agenda_doc_points"] is None and mapping_env["reference_doc_points"] is None
    assert "agenda_document_insights" not in result
    texts = [p["text"] for a in result["final_rom_agendas"] for p in a["discussion_points"]]
    assert texts == [p["polished_text"] for p in POINTS]


def test_agenda_document_points_are_context_and_refine_the_long_rom(mapping_env):
    result = _map(agenda_doc_insights=copy.deepcopy(INSIGHTS), meeting_date=MEETING_DATE)

    # extra context for the mapping: similarity facet + LLM agenda reference
    assert set(mapping_env["agenda_doc_points"]) == {"A1", "A2"}
    assert mapping_env["reference_doc_points"]["A1"][0] == "Data migration dry-run 2 slipped"

    a1, a2 = result["final_rom_agendas"]
    assert [p["id"] for p in a1["discussion_points"]] == ["p1"]
    assert a1["discussion_points"][0]["doc_refined"] and REFINED_SUMMARY in a1["discussion_points"][0]["text"]
    assert a2["discussion_points"][0]["doc_refined"]
    doc_only = [p for p in a2["discussion_points"] if p.get("is_doc_point")]
    assert len(doc_only) == 1 and "Crash rate reduced" in doc_only[0]["text"]

    summary = result["agenda_document_insights"]
    assert summary["total_doc_points"] == 5
    assert summary["refinement"]["A1"]["refined_points"] == 1
    assert summary["refinement"]["A2"]["important_doc_points_added"] == 1


def test_provided_agendas_are_used_as_is(monkeypatch):
    """Agendas edited on the upload page skip the cache and the LLM extraction, order kept."""
    saved = {}
    embedder = MagicMock()
    embedder._dim = 1024
    embedder.encode_batch.return_value = np.zeros((1, 1024), dtype=np.float32)
    provider = MagicMock()
    monkeypatch.setattr("services.ai_provider.get_provider", lambda: provider)
    monkeypatch.setattr(tes, "get_text_embedder", lambda: embedder)
    monkeypatch.setattr("services.rag_pipeline._load_parsed_agenda", lambda rid: [{"topic": "Cached"}])
    monkeypatch.setattr("services.rag_pipeline._save_parsed_agenda", lambda rid, uid, items: saved.update(items=items))

    class _EmptyStore:
        _meta = []

        def load_or_create(self):
            pass

        def search(self, *a, **k):
            return []
    monkeypatch.setattr("services.vector_store.get_meeting_context_store", lambda *a, **k: _EmptyStore())
    monkeypatch.setattr("services.vector_store.get_global_context_store", lambda *a, **k: _EmptyStore())

    edited = [{"agenda_id": "A2", "title": "Beacon mobile app", "description": "Renamed"},
              {"agenda_id": "A1", "title": "Project Atlas build status"},
              {"title": "New agenda added by hand"}]
    res = RomService().generate_agendas_only(agenda_text="ignored", recording_id="rec-1", user_id="user-1",
                                             provided_agendas=edited)
    assert [(a["agenda_id"], a["title"]) for a in res["agendas"]] == [
        ("A2", "Beacon mobile app"), ("A1", "Project Atlas build status"), ("A3", "New agenda added by hand")]
    provider.generate_rom_agendas.assert_not_called()
    assert [i["agenda_id"] for i in saved["items"]] == ["A2", "A1", "A3"]
