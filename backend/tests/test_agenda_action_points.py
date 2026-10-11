"""
Per-agenda Final ROM action points: validation, generation, endpoints and DOCX export.
No model is loaded: the LLM is a scripted provider and embedding dedupe is off (or faked).
"""
import copy
import io
import json

import numpy as np
import pytest
from docx import Document
from unittest.mock import AsyncMock, MagicMock, patch

import services.agenda_action_points as aap
from services.agenda_action_points import (
    NOT_SPECIFIED,
    agenda_includes_action_points,
    build_agenda_action_input,
    collect_agenda_action_points,
    is_vague_task,
    populate_agenda_action_points,
    split_for_budget,
    validate_action_points,
)


@pytest.fixture(autouse=True)
def _no_embedding_dedupe(monkeypatch):
    monkeypatch.setattr(aap, "EMBEDDING_DEDUPE", False)


class ScriptedProvider:
    def __init__(self, by_agenda=None):
        self.by_agenda = by_agenda or {}
        self.calls = []
        self.unloaded = False

    def generate_agenda_action_points(self, agenda_json, points_json, reasoning_mode=None):
        agenda = json.loads(agenda_json)
        self.calls.append({"agenda": agenda, "points": json.loads(points_json)})
        return {"action_points": copy.deepcopy(self.by_agenda.get(agenda["agenda_id"], []))}

    def unload_model(self):
        self.unloaded = True


def _entry(pid, text, **extra):
    entry = {"id": pid, "text": text}
    entry.update(extra)
    return entry


# ── Validation ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", ["Follow up on this", "Look into the issue", "Monitor progress", "Discuss further in the next meeting"])
def test_vague_tasks_are_detected(task):
    assert is_vague_task(task)


@pytest.mark.parametrize("task", [
    "Follow up with the vendor on the contract renewal terms",
    "Share the SAP-MM migration plan with the CFO",
    "Pay ₹5 L advance",
])
def test_specific_tasks_are_not_vague(task):
    assert not is_vague_task(task)


def test_vague_task_is_rejected():
    entries = [_entry("p1", "We should follow up on this later.")]
    result = validate_action_points([{"task": "Follow up on this", "source_point_ids": ["p1"]}], entries)
    assert result["kept"] == []
    assert result["rejected"][0]["reason"].startswith("vague")


def test_facts_are_preserved_and_grounded_owner_and_deadline_kept():
    entries = [_entry(
        "p1",
        "Ravi agreed to share the SAP-MM migration plan with the CFO by Oct 24; the approved budget is ₹2.5 Cr.",
        action_owner="Ravi", owner_candidates=["Ravi"], dates=["Oct 24"], numbers=["2.5 Cr"], technical_terms=["SAP-MM"],
    )]
    result = validate_action_points([{
        "task": "Share the SAP-MM migration plan (budget ₹2.5 Cr) with the CFO",
        "owner": "Ravi", "deadline": "Oct 24", "source_point_ids": ["p1"], "priority": "high",
        "evidence": "Ravi agreed to share the SAP-MM migration plan with the CFO by Oct 24",
    }], entries)
    assert result["rejected"] == []
    item = result["kept"][0]
    assert item["task"] == "Share the SAP-MM migration plan (budget ₹2.5 Cr) with the CFO"
    assert item["assignee"] == "Ravi"
    assert item["owner_confidence"] == "high"
    assert item["deadline"] == "Oct 24"
    assert item["evidence_quote"].startswith("Ravi agreed")


def test_hyphenated_numbers_match_their_unhyphenated_form():
    entries = [_entry("p1", "The committee reviewed Proposed Ordinance 20260108 on the surplus property sale.")]
    result = validate_action_points([{"task": "Circulate the amended text of Proposed Ordinance 2026-0108 to the committee",
                                      "source_point_ids": ["p1"]}], entries)
    assert result["kept"] and result["rejected"] == []


def test_invented_numbers_and_dates_are_rejected():
    entries = [_entry("p1", "Ravi will send the revised quotation to the client.")]
    result = validate_action_points([{
        "task": "Send the revised quotation of ₹2.5 Cr to the client by 24 October",
        "owner": "Ravi", "source_point_ids": ["p1"],
    }], entries)
    assert result["kept"] == []
    reason = result["rejected"][0]["reason"]
    assert reason.startswith("details not in the discussion")
    assert "2.5" in reason and "24" in reason


def test_invented_acronym_is_rejected():
    entries = [_entry("p1", "The team will finalise the vendor onboarding checklist.")]
    result = validate_action_points([{"task": "Finalise the ERP vendor onboarding checklist", "source_point_ids": ["p1"]}], entries)
    assert result["kept"] == []


def test_fact_from_another_point_of_the_agenda_is_accepted_and_cited():
    entries = [
        _entry("p1", "Meena will circulate the revised rollout plan."),
        _entry("p2", "The rollout covers 14 branches in phase one."),
    ]
    result = validate_action_points([{
        "task": "Circulate the revised rollout plan for the 14 branches", "owner": "Meena", "source_point_ids": ["p1"],
    }], entries)
    item = result["kept"][0]
    assert item["source_point_ids"] == ["p1", "p2"]


def test_ungrounded_deadline_and_owner_become_not_specified():
    entries = [_entry("p1", "The revised floor plan must be shared with the facilities vendor.")]
    result = validate_action_points([{
        "task": "Share the revised floor plan with the facilities vendor",
        "owner": "Anil", "deadline": "Friday", "source_point_ids": ["p1"],
    }], entries)
    item = result["kept"][0]
    assert item["assignee"] is None and item["deadline"] is None
    fields = {a["field"] for a in result["adjusted"]}
    assert fields == {"owner", "deadline"}
    final = aap.finalize_action_point(item)
    assert final["assignee"] == NOT_SPECIFIED and final["deadline"] == NOT_SPECIFIED


def test_relative_deadline_must_be_stated():
    entries = [_entry("p1", "Priya will close the audit observations by next week.")]
    kept = validate_action_points([{"task": "Close the audit observations", "owner": "Priya", "deadline": "next week",
                                    "source_point_ids": ["p1"]}], entries)["kept"][0]
    assert kept["deadline"] == "next week"
    changed = validate_action_points([{"task": "Close the audit observations", "owner": "Priya",
                                       "deadline": "end of the month", "source_point_ids": ["p1"]}], entries)["kept"][0]
    assert changed["deadline"] is None


def test_vague_owner_is_not_specified_without_adjustment():
    entries = [_entry("p1", "The team will update the compliance tracker.")]
    result = validate_action_points([{"task": "Update the compliance tracker", "owner": "team", "source_point_ids": ["p1"]}], entries)
    assert result["kept"][0]["assignee"] is None
    assert result["adjusted"] == []


def test_low_priority_dropped_unless_owner_or_deadline_stated():
    entries = [_entry("p1", "Kiran mentioned the office pantry signage could be refreshed.")]
    rejected = validate_action_points([{"task": "Refresh the office pantry signage", "priority": "low",
                                        "source_point_ids": ["p1"]}], entries)
    assert rejected["kept"] == [] and rejected["rejected"][0]["reason"].startswith("low priority")
    kept = validate_action_points([{"task": "Refresh the office pantry signage", "priority": "low", "owner": "Kiran",
                                    "source_point_ids": ["p1"]}], entries)
    assert kept["kept"][0]["assignee"] == "Kiran"


def test_duplicates_merge_sources_and_keep_owner():
    entries = [
        _entry("p1", "Ravi will send the revised quotation to the client."),
        _entry("p2", "Again it was noted the revised quotation must go to the client."),
    ]
    result = validate_action_points([
        {"task": "Send the revised quotation to the client", "source_point_ids": ["p2"]},
        {"task": "Send revised quotation to client", "owner": "Ravi", "source_point_ids": ["p1"]},
    ], entries)
    assert len(result["kept"]) == 1
    item = result["kept"][0]
    assert set(item["source_point_ids"]) == {"p1", "p2"}
    assert item["assignee"] == "Ravi"
    assert result["rejected"][0]["reason"].startswith("duplicate of")


def test_same_task_for_different_owners_is_not_a_duplicate():
    entries = [_entry("p1", "Ravi and Meena will each review the vendor contract.")]
    result = validate_action_points([
        {"task": "Review the vendor contract", "owner": "Ravi", "source_point_ids": ["p1"]},
        {"task": "Review the vendor contract", "owner": "Meena", "source_point_ids": ["p1"]},
    ], entries)
    assert len(result["kept"]) == 2


def test_paraphrased_duplicates_are_merged_with_embeddings():
    class Embedder:
        def encode_batch(self, texts):
            # Both phrasings mean the same thing; the fake model maps them to one vector.
            return np.array([[1.0, 0.0] if "quot" in t.lower() else [0.0, 1.0] for t in texts], dtype=np.float32)

    entries = [_entry("p1", "Ravi will send the revised quotation to the client; the updated quote goes to the customer.")]
    result = validate_action_points([
        {"task": "Send the revised quotation to the client", "owner": "Ravi", "source_point_ids": ["p1"]},
        {"task": "Share the updated quote with the customer", "owner": "Ravi", "source_point_ids": ["p1"]},
    ], entries, embedder=Embedder())
    assert len(result["kept"]) == 1


# ── Provider parsing ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("response,expected", [
    ('{"action_points": [{"task": "Send the plan"}]}', [{"task": "Send the plan"}]),
    ('Here you go:\n[{"task": "Send the plan", "owner": "Not specified"}]', [{"task": "Send the plan", "owner": "Not specified"}]),
    ('{"action_items": [{"task": "Send the plan"}]}', [{"task": "Send the plan"}]),
    ('No genuine action points.', None),
])
def test_provider_parses_wrapped_and_bare_action_point_lists(monkeypatch, response, expected):
    from services.ai_provider import QwenProvider

    provider = QwenProvider()
    monkeypatch.setattr(provider, "_infer", lambda prompt, **kwargs: response)
    result = provider.generate_agenda_action_points('{"agenda_id": "A1"}', "[]", reasoning_mode=False)
    assert (result["action_points"] if result else None) == expected


@pytest.mark.parametrize("owner_a,owner_b,merged", [
    ("Speaker 5", "Speaker 5", True),   # same named owner: 0.86 is enough
    (None, None, False),                # no shared owner: needs 0.88
    ("Speaker 5", None, False),
])
def test_paraphrase_threshold_depends_on_a_shared_owner(owner_a, owner_b, merged):
    class Embedder:
        def encode_batch(self, texts):
            return np.array([[1.0, 0.0], [0.86, (1 - 0.86 ** 2) ** 0.5]], dtype=np.float32)

    items = [
        {"task": "Provide the AMI calculation for a family of four", "assignee": owner_a, "deadline": None,
         "source_point_ids": ["p1"], "priority": "medium"},
        {"task": "Share the specific income number for two parents with two children", "assignee": owner_b,
         "deadline": None, "source_point_ids": ["p2"], "priority": "medium"},
    ]
    kept, removed = aap.dedupe_action_points(items, Embedder())
    assert (len(kept) == 1) is merged


# ── Input and generation ─────────────────────────────────────────────────────

def test_build_input_uses_complete_points_and_stage_detail():
    long_text = "Ravi presented the vendor evaluation. " * 80
    agenda = {"agenda_id": "A1", "title": "Vendors", "discussion_points": [
        {"id": "p1", "text": long_text, "speakers": ["Ravi"]},
        {"id": "doc-1", "text": "Document fact.", "is_doc_point": True},
    ]}
    polished_by_id = {"p1": {
        "id": "p1", "polished_text": long_text, "action_owner": "Ravi", "dates": ["Oct 24"], "numbers": ["3"],
        "technical_terms": ["RFP"], "original_point_ids": ["s1"], "owner_transcript_excerpt": "[T1] Ravi: I'll do it",
        "timeline_start": 65.0, "timeline_end": 130.0,
    }}
    stage1_by_id = {"s1": {"id": "s1", "action_items": [{"task": "Issue the RFP", "assignee": "Ravi", "deadline": "Oct 24"}]}}

    entries = build_agenda_action_input(agenda, polished_by_id, stage1_by_id)
    assert [e["id"] for e in entries] == ["p1"]
    entry = entries[0]
    assert entry["text"] == long_text.strip()
    assert entry["dates"] == ["Oct 24"] and entry["numbers"] == ["3"] and entry["technical_terms"] == ["RFP"]
    assert entry["owner_candidates"] == ["Ravi"]
    assert entry["candidate_actions"] == [{"task": "Issue the RFP", "owner": "Ravi", "deadline": "Oct 24"}]
    assert entry["time"] == "01:05-02:10"
    assert "transcript_excerpt" not in aap._llm_view(entry)


def test_split_for_budget_is_chronological():
    entries = [_entry(f"p{i}", "x" * 300) for i in range(5)]
    chunks = split_for_budget(entries, budget=700)
    assert [len(c) for c in chunks] == [2, 2, 1]
    assert [e["id"] for c in chunks for e in c] == [f"p{i}" for i in range(5)]


def test_each_agenda_gets_its_own_call_and_large_agendas_are_split(monkeypatch):
    monkeypatch.setattr(aap, "INPUT_CHAR_BUDGET", 250)
    agendas = [
        {"agenda_id": "A1", "title": "Budget", "discussion_points": [
            {"id": f"b{i}", "text": f"Budget point {i} about the capital expenditure plan and its review."} for i in range(4)
        ]},
        {"agenda_id": "A2", "title": "Hiring", "discussion_points": [{"id": "h1", "text": "Hiring plan for two analysts."}]},
    ]
    provider = ScriptedProvider()
    populate_agenda_action_points(agendas, [], [], provider=provider)
    a1_calls = [c for c in provider.calls if c["agenda"]["agenda_id"] == "A1"]
    a2_calls = [c for c in provider.calls if c["agenda"]["agenda_id"] == "A2"]
    assert len(a1_calls) > 1
    assert {p["id"] for c in a1_calls for p in c["points"]} == {"b0", "b1", "b2", "b3"}
    assert [p["id"] for c in a2_calls for p in c["points"]] == ["h1"]
    assert not provider.unloaded  # a caller-supplied provider is left loaded


def test_same_task_under_two_agendas_is_kept_once():
    agendas = [
        {"agenda_id": "A1", "title": "Budget", "discussion_points": [{"id": "p1", "text": "Ravi will send the revised budget sheet to finance."}]},
        {"agenda_id": "A2", "title": "Finance", "discussion_points": [{"id": "p2", "text": "Ravi will send the revised budget sheet to finance today."}]},
    ]
    task = {"task": "Send the revised budget sheet to finance", "owner": "Ravi"}
    provider = ScriptedProvider({"A1": [dict(task, source_point_ids=["p1"])], "A2": [dict(task, source_point_ids=["p2"])]})
    audits = populate_agenda_action_points(agendas, [], [], provider=provider)
    assert len(agendas[0]["action_points"]) == 1
    assert agendas[1]["action_points"] == []
    assert audits["A2"]["rejected"][0]["reason"].startswith("already listed under another agenda")


def test_condensed_version_uses_complete_long_points():
    short = [{"agenda_id": "A1", "title": "Housing", "discussion_points": [
        {"id": "A1-SHORT-1", "text": "Housing loan discussed.", "original_point_id": "p1"}]}]
    long = [{"agenda_id": "A1", "title": "Housing", "discussion_points": [
        {"id": "p1", "text": "Speaker 5 committed to providing the income limit for a family of four."},
        {"id": "p2", "text": "The committee reviewed the interim loan repayment terms."}]}]
    provider = ScriptedProvider({"A1": [{"task": "Provide the income limit for a family of four",
                                          "owner": "Speaker 5", "source_point_ids": ["p1"]}]})
    populate_agenda_action_points(short, [], [], provider=provider, source_agendas=long)
    assert [p["id"] for p in provider.calls[0]["points"]] == ["p1", "p2"]
    assert short[0]["action_points"][0]["source_point_ids"] == ["p1"]
    assert short[0]["action_points"][0]["assignee"] == "Speaker 5"


def test_condensed_points_resolve_to_their_stage2_originals_in_meeting_order():
    agenda = {"agenda_id": "A3", "discussion_points": [
        {"id": "A3-SHORT-1", "text": "Loan discussed.", "original_point_id": "p2"},
        {"id": "A3-SHORT-2", "text": "Loan discussed again.", "original_point_id": "p2"},
        {"id": "p1", "text": "Earlier point."},
    ]}
    polished_by_id = {
        "p1": {"id": "p1", "polished_text": "Earlier point.", "timeline_start": 10},
        "p2": {"id": "p2", "polished_text": "Speaker 5 committed to providing the income limit for a family of four.",
               "timeline_start": 90},
    }
    entries = build_agenda_action_input(agenda, polished_by_id, {})
    assert [e["id"] for e in entries] == ["p1", "p2"]
    assert entries[1]["text"].startswith("Speaker 5 committed")


def test_source_agendas_add_mapped_points_missing_from_an_overwritten_long_version():
    from routers.rom_router import _action_point_source_agendas

    data = {
        "stage2": {"polished_points": [{"id": f"p{i}", "polished_text": f"Point {i}"} for i in range(1, 5)]},
        "stage3": {"point_mappings": {"p1": "A1", "p2": "A2", "p3": "A2", "p4": "A2"}},
        "final_rom_versions": {"long": {"agendas": [
            {"agenda_id": "A1", "discussion_points": [{"id": "p1"}, {"id": "p4"}]},  # p4 moved here by the user
            {"agenda_id": "A2", "discussion_points": [{"id": "A2-SHORT-1", "original_point_id": "p2"}]},
        ]}},
    }
    sources = {a["agenda_id"]: [p["id"] for p in a["discussion_points"]] for a in _action_point_source_agendas(data)}
    assert sources == {"A1": ["p1", "p4"], "A2": ["A2-SHORT-1", "p3"]}


def test_regenerate_false_keeps_existing_agenda_lists():
    agendas = [{"agenda_id": "A1", "title": "Budget", "include_action_points": True, "action_points_mode": "agenda",
                "action_points": [{"task": "Existing", "assignee": NOT_SPECIFIED, "deadline": NOT_SPECIFIED}],
                "discussion_points": [{"id": "p1", "text": "Ravi will send the budget."}]}]
    provider = ScriptedProvider()
    assert populate_agenda_action_points(agendas, [], [], provider=provider, regenerate=False) == {}
    assert provider.calls == []
    assert agendas[0]["action_points"][0]["task"] == "Existing"


def test_include_flag_and_legacy_fallback():
    assert agenda_includes_action_points({"include_action_points": True}) is True
    assert agenda_includes_action_points({"include_action_points": False}, {"include_action_points": True}) is False
    assert agenda_includes_action_points({}, {"include_action_points": True}) is True
    assert agenda_includes_action_points({"skipped": True, "include_action_points": True}) is False


def test_collect_legacy_agenda_merges_point_level_items():
    legacy = {"discussion_points": [
        {"id": "p1", "action_points": [{"task": "Brief the council", "assignee": "Staff"}]},
        {"id": "p2", "action_items": [{"task": "Prepare budget report", "assignee": "Bob", "deadline": "Friday"}]},
    ]}
    tasks = [ap["task"] for ap in collect_agenda_action_points(legacy)]
    assert tasks[0] == "Brief the council"
    assert "Prepare budget report" in tasks[1]
    agenda_mode = {"action_points_mode": "agenda", "action_points": [{"task": "Only this"}],
                   "discussion_points": [{"id": "p1", "action_items": ["ignored"]}]}
    assert [ap["task"] for ap in collect_agenda_action_points(agenda_mode)] == ["Only this"]


# ── Endpoints ────────────────────────────────────────────────────────────────

def _rom_data():
    agendas = [
        {"agenda_id": "A1", "title": "Budget", "discussion_points": [
            {"id": "p1", "text": "Ravi will send the revised budget sheet to finance by Friday."}]},
        {"agenda_id": "A2", "title": "Hiring", "discussion_points": [
            {"id": "p2", "text": "Meena will open two analyst requisitions."}]},
        {"agenda_id": "A3", "title": "Deferred", "skipped": True, "discussion_points": []},
    ]
    return {
        "stage1": {"discussion_points": []},
        "stage2": {"polished_points": [
            {"id": "p1", "polished_text": agendas[0]["discussion_points"][0]["text"], "action_owner": "Ravi"},
            {"id": "p2", "polished_text": agendas[1]["discussion_points"][0]["text"], "action_owner": "Meena"},
        ]},
        "final_rom": {"agendas": agendas, "include_action_points": False},
        "final_rom_versions": {"long": {"agendas": copy.deepcopy(agendas)}, "short": {"agendas": copy.deepcopy(agendas)}},
    }


PROPOSALS = {
    "A1": [{"task": "Send the revised budget sheet to finance", "owner": "Ravi", "deadline": "Friday", "source_point_ids": ["p1"]}],
    "A2": [{"task": "Open two analyst requisitions", "owner": "Meena", "source_point_ids": ["p2"]}],
}


def _endpoint_patches(data):
    return [
        patch("routers.rom_router._get_rom_data", AsyncMock(return_value=data)),
        patch("routers.rom_router._get_recording_or_404", AsyncMock(return_value={})),
        patch("routers.rom_router._save_rom_data", AsyncMock()),
        patch("routers.rom_router._sync_final_rom_embeddings", AsyncMock()),
        patch("routers.rom_router._validate_user_id", return_value="test-user"),
    ]


async def _call(coro_factory, data, provider):
    patches = _endpoint_patches(data) + [patch("services.ai_provider.get_provider", return_value=provider)]
    for p in patches:
        p.start()
    try:
        return await coro_factory()
    finally:
        for p in reversed(patches):
            p.stop()


@pytest.mark.asyncio
async def test_agenda_endpoint_generates_only_that_agenda_and_syncs_versions():
    from routers.rom_router import AgendaActionPointsRequest, set_agenda_action_points

    data = _rom_data()
    provider = ScriptedProvider(PROPOSALS)
    res = await _call(lambda: set_agenda_action_points(
        "rec-1", "A2", AgendaActionPointsRequest(include=True), current_user={}, db=MagicMock()), data, provider)

    assert [c["agenda"]["agenda_id"] for c in provider.calls] == ["A2"]
    a1, a2 = data["final_rom"]["agendas"][:2]
    assert "action_points" not in a1
    assert a2["include_action_points"] is True
    assert a2["action_points"][0]["task"] == "Open two analyst requisitions"
    assert a2["action_points"][0]["assignee"] == "Meena"
    assert a2["action_points"][0]["deadline"] == NOT_SPECIFIED
    assert data["final_rom"]["include_action_points"] is False  # A1 is not included
    for version in ("long", "short"):
        v_a2 = data["final_rom_versions"][version]["agendas"][1]
        assert v_a2["action_points"] == a2["action_points"]
        assert v_a2["include_action_points"] is True
    assert res["action_point_audit"]["kept"] == 1

    # Turning it off keeps the generated list; turning it back on does not call the LLM again.
    await _call(lambda: set_agenda_action_points(
        "rec-1", "A2", AgendaActionPointsRequest(include=False), current_user={}, db=MagicMock()), data, provider)
    assert a2["include_action_points"] is False and a2["action_points"]
    await _call(lambda: set_agenda_action_points(
        "rec-1", "A2", AgendaActionPointsRequest(include=True), current_user={}, db=MagicMock()), data, provider)
    assert len(provider.calls) == 1

    await _call(lambda: set_agenda_action_points(
        "rec-1", "A2", AgendaActionPointsRequest(include=True, regenerate=True), current_user={}, db=MagicMock()), data, provider)
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_agenda_endpoint_rejects_skipped_and_unknown_agendas():
    from fastapi import HTTPException
    from routers.rom_router import AgendaActionPointsRequest, set_agenda_action_points

    data = _rom_data()
    with pytest.raises(HTTPException) as skipped:
        await _call(lambda: set_agenda_action_points(
            "rec-1", "A3", AgendaActionPointsRequest(include=True), current_user={}, db=MagicMock()), data, ScriptedProvider())
    assert skipped.value.status_code == 400
    with pytest.raises(HTTPException) as missing:
        await _call(lambda: set_agenda_action_points(
            "rec-1", "A9", AgendaActionPointsRequest(include=True), current_user={}, db=MagicMock()), data, ScriptedProvider())
    assert missing.value.status_code == 404


@pytest.mark.asyncio
async def test_global_toggle_covers_all_agendas_even_when_points_have_items():
    from routers.rom_router import ToggleActionPointsRequest, toggle_final_rom_action_points

    data = _rom_data()
    # Legacy point-level action points used to block generation for every other agenda.
    data["final_rom"]["agendas"][0]["discussion_points"][0]["action_points"] = [{"task": "old"}]
    provider = ScriptedProvider(PROPOSALS)
    res = await _call(lambda: toggle_final_rom_action_points(
        "rec-1", ToggleActionPointsRequest(include_action_points=True), current_user={}, db=MagicMock()), data, provider)

    assert sorted(c["agenda"]["agenda_id"] for c in provider.calls) == ["A1", "A2"]
    agendas = data["final_rom"]["agendas"]
    assert agendas[0]["action_points"][0]["deadline"] == "Friday"
    assert "action_points" not in agendas[0]["discussion_points"][0]
    assert "action_points" not in agendas[2] and not agendas[2].get("include_action_points")
    assert res["include_action_points"] is True

    await _call(lambda: toggle_final_rom_action_points(
        "rec-1", ToggleActionPointsRequest(include_action_points=False), current_user={}, db=MagicMock()), data, provider)
    assert all(a.get("include_action_points") is False for a in agendas[:2])
    assert agendas[0]["action_points"]  # data kept
    assert data["final_rom"]["include_action_points"] is False


@pytest.mark.asyncio
async def test_generate_final_rom_keeps_individual_selections():
    from routers.rom_router import GenerateFinalRomRequest, generate_final_rom_from_agendas

    data = _rom_data()
    data["final_rom"]["agendas"][1]["include_action_points"] = True  # only A2 was selected
    data["stage3"] = {"agendas": [{k: v for k, v in a.items() if k != "discussion_points"} for a in data["final_rom"]["agendas"]],
                      "expanded_agendas": []}
    mapped = [copy.deepcopy(a) for a in data["final_rom"]["agendas"] if not a.get("skipped")]
    for a in mapped:
        a.pop("include_action_points", None)
    mapping = {"final_rom_agendas": mapped, "similarity_agenda_ids": ["A1", "A2"]}
    provider = ScriptedProvider(PROPOSALS)

    with patch("routers.rom_router.rom_service.map_points_to_agendas", return_value=mapping), \
         patch("services.rom_service.apply_speaker_mappings_to_final_rom", side_effect=lambda rom: rom):
        await _call(lambda: generate_final_rom_from_agendas(
            "rec-1", GenerateFinalRomRequest(include_action_points=False, skipped_agendas={"A3": {}}),
            current_user={}, db=MagicMock()), data, provider)

    assert [c["agenda"]["agenda_id"] for c in provider.calls] == ["A2"]
    final = {a["agenda_id"]: a for a in data["final_rom"]["agendas"]}
    assert final["A1"]["include_action_points"] is False and "action_points" not in final["A1"]
    assert final["A2"]["include_action_points"] is True and final["A2"]["action_points"]
    assert data["final_rom"]["include_action_points"] is False


# ── DOCX ─────────────────────────────────────────────────────────────────────

async def _docx_rows(rom_data, include_action_points=None):
    from routers.rom_router import download_final_docx

    with patch("routers.rom_router._get_rom_data", AsyncMock(return_value=rom_data)), \
         patch("routers.rom_router._validate_user_id", return_value="test-user"):
        response = await download_final_docx(recording_id="rec-1", version=None, include_action_points=include_action_points,
                                             current_user={}, db=MagicMock())
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
    table = Document(io.BytesIO(body)).tables[0]
    return [[c.text.strip() for c in row.cells] for row in table.rows]


@pytest.mark.asyncio
async def test_docx_shows_action_points_only_for_included_agendas():
    ap = {"task": "Open two analyst requisitions", "assignee": NOT_SPECIFIED, "deadline": NOT_SPECIFIED}
    rom = {"final_rom": {"agendas": [
        {"agenda_id": "A1", "title": "Budget", "include_action_points": False, "action_points_mode": "agenda",
         "action_points": [{"task": "Hidden task", "assignee": "Ravi", "deadline": "Friday"}],
         "discussion_points": [{"id": "p1", "text": "Budget discussion."}]},
        {"agenda_id": "A2", "title": "Hiring", "include_action_points": True, "action_points_mode": "agenda",
         "action_points": [ap], "discussion_points": [{"id": "p2", "text": "Hiring discussion.",
                                                         "action_items": [{"task": "Legacy item ignored"}]}]},
    ]}}
    rows = await _docx_rows(rom, include_action_points=True)
    flat = " | ".join(" ".join(r) for r in rows)
    assert "Hidden task" not in flat
    assert "Legacy item ignored" not in flat
    ap_row = next(r for r in rows if "Open two analyst requisitions" in " ".join(r))
    assert ap_row[-1] == "Not specified [Due: Not specified]"

    rows_hidden = await _docx_rows(rom, include_action_points=False)
    assert "Open two analyst requisitions" not in " ".join(" ".join(r) for r in rows_hidden)


@pytest.mark.asyncio
async def test_docx_legacy_rom_follows_global_flag():
    legacy_agendas = [{"agenda_id": "A1", "title": "Budget", "discussion_points": [
        {"id": "p1", "text": "Budget discussion.", "action_points": [{"task": "Brief the council", "assignee": "Staff", "deadline": "2026-10-01"}]},
    ]}]
    shown = await _docx_rows({"final_rom": {"include_action_points": True, "agendas": copy.deepcopy(legacy_agendas)}})
    row = next(r for r in shown if "Brief the council" in " ".join(r))
    assert row[-1] == "Staff [Due: 2026-10-01]"
    hidden = await _docx_rows({"final_rom": {"include_action_points": False, "agendas": copy.deepcopy(legacy_agendas)}})
    assert "Brief the council" not in " ".join(" ".join(r) for r in hidden)
