import json

import pytest
from unittest.mock import patch

import services.agenda_action_points as aap
from routers.rom_router import _populate_action_points_for_agendas


@pytest.fixture(autouse=True)
def _no_embedding_dedupe(monkeypatch):
    # Token-based dedupe only: unit tests must not load the embedding model.
    monkeypatch.setattr(aap, "EMBEDDING_DEDUPE", False)


class AgendaProvider:
    """generate_agenda_action_points returns the proposals configured per agenda id."""

    def __init__(self, by_agenda=None, fail=False):
        self.by_agenda = by_agenda or {}
        self.fail = fail
        self.calls = []

    def generate_agenda_action_points(self, agenda_json, points_json, reasoning_mode=None):
        agenda = json.loads(agenda_json)
        self.calls.append({"agenda": agenda, "points": json.loads(points_json), "reasoning_mode": reasoning_mode})
        if self.fail:
            raise RuntimeError("LLM unavailable")
        return {"action_points": self.by_agenda.get(agenda["agenda_id"], [])}

    def unload_model(self):
        pass


@pytest.mark.asyncio
async def test_populate_action_points_falls_back_to_existing_items_when_llm_fails():
    """If the LLM call fails, the points' Stage 1/2 action items are used (validated) and
    stored at agenda level; point-level action points are cleared."""
    agendas = [{
        "agenda_id": "A1",
        "title": "Budget Review",
        "discussion_points": [
            {"id": "pt-1", "text": "Discussed Q3 budget allocations.", "speaker": "Alice",
             "action_points": [{"task": "stale"}]},
        ],
    }]
    polished_points = [{
        "id": "pt-1",
        "polished_text": "Discussed Q3 budget allocations.",
        "action_items": [{"task": "Finalize Q3 budget spreadsheet", "assignee": "Bob", "deadline": "Friday"}],
    }]

    with patch("services.ai_provider.get_provider", return_value=AgendaProvider(fail=True)):
        res = await _populate_action_points_for_agendas(agendas, polished_points, [])

    agenda = res[0]
    assert agenda["action_points_mode"] == "agenda"
    assert agenda["include_action_points"] is True
    assert agenda["action_point_audit"]["source"] == "fallback_stage_items"
    assert "action_points" not in agenda["discussion_points"][0]
    aps = agenda["action_points"]
    assert len(aps) == 1
    assert aps[0]["task"] == "Finalize Q3 budget spreadsheet"
    assert aps[0]["assignee"] == "Bob"
    assert aps[0]["deadline"] == "Friday"
    assert aps[0]["source_point_id"] == "pt-1"


@pytest.mark.asyncio
async def test_populate_action_points_uses_agenda_llm_proposals():
    """The agenda's points are analysed together by one LLM call; grounded owner and
    deadline are kept."""
    agendas = [{
        "agenda_id": "A1",
        "title": "Strategy Discussion",
        "discussion_points": [{"id": "pt-2", "text": "Diana agreed to launch the email campaign for the marketing outreach next week."}],
    }]
    polished_points = [{"id": "pt-2", "polished_text": agendas[0]["discussion_points"][0]["text"],
                        "action_owner": "Diana", "action_items": []}]
    provider = AgendaProvider({"A1": [{
        "task": "Launch the email campaign for the marketing outreach", "owner": "Diana",
        "deadline": "next week", "source_point_ids": ["pt-2"], "priority": "high",
    }]})

    with patch("services.ai_provider.get_provider", return_value=provider):
        res = await _populate_action_points_for_agendas(agendas, polished_points, [], reasoning_mode=True)

    assert len(provider.calls) == 1
    assert provider.calls[0]["reasoning_mode"] is True
    aps = res[0]["action_points"]
    assert len(aps) == 1
    assert aps[0]["task"] == "Launch the email campaign for the marketing outreach"
    assert aps[0]["assignee"] == "Diana"
    assert aps[0]["deadline"] == "next week"
    assert aps[0]["priority"] == "high"


@pytest.mark.asyncio
async def test_populate_action_points_skipped_agenda_ignored():
    """Verify that skipped agendas are not modified during action points population."""
    agendas = [
        {
            "agenda_id": "A1",
            "title": "Deferred Topic",
            "skipped": True,
            "skip_note": "Postponed to next quarter",
            "discussion_points": []
        }
    ]
    polished_points = [{"id": "pt-1", "polished_text": "Sample"}]
    res = await _populate_action_points_for_agendas(agendas, polished_points, [])
    assert res[0]["skipped"] is True
    assert res[0]["skip_note"] == "Postponed to next quarter"
    assert res[0]["discussion_points"] == []
    assert "action_points" not in res[0]


@pytest.mark.asyncio
async def test_populate_action_points_relinks_unknown_source_and_rejects_unsupported():
    """A task citing an unknown point is linked to the agenda point that supports it; the same
    task proposed under an unrelated agenda is rejected as unsupported. An owner the
    discussion never names becomes "Not specified"."""
    agendas = [
        {"agenda_id": "A1", "title": "Budget", "discussion_points": [
            {"id": "pt-1", "text": "Reviewed the quarterly budget numbers."},
        ]},
        {"agenda_id": "A2", "title": "Infra", "discussion_points": [
            {"id": "pt-2", "text": "Discussed migrating the database cluster to the new region."},
        ]},
    ]
    polished_points = [
        {"id": "pt-1", "polished_text": "Reviewed the quarterly budget numbers.", "action_items": []},
        {"id": "pt-2", "polished_text": "Discussed migrating the database cluster to the new region.", "action_items": []},
    ]
    proposal = {"task": "Plan the database cluster migration to the new region", "owner": "Speaker 2",
                "source_point_ids": ["unknown-id"]}
    provider = AgendaProvider({"A1": [dict(proposal)], "A2": [dict(proposal)]})

    with patch("services.ai_provider.get_provider", return_value=provider):
        res = await _populate_action_points_for_agendas(agendas, polished_points, [])

    assert res[0]["action_points"] == []
    assert res[0]["action_point_audit"]["rejected"][0]["reason"].startswith("not supported")
    aps = res[1]["action_points"]
    assert len(aps) == 1
    assert aps[0]["source_point_id"] == "pt-2"
    assert aps[0]["assignee"] == aap.NOT_SPECIFIED
    assert aps[0]["deadline"] == aap.NOT_SPECIFIED
    assert res[1]["action_point_audit"]["adjusted"][0]["field"] == "owner"


@pytest.mark.asyncio
async def test_populate_action_points_only_targets_requested_agendas():
    agendas = [
        {"agenda_id": "A1", "title": "Budget", "discussion_points": [{"id": "pt-1", "text": "Bob will send the revised budget to finance."}]},
        {"agenda_id": "A2", "title": "Infra", "discussion_points": [{"id": "pt-2", "text": "Carol will migrate the database cluster."}]},
    ]
    provider = AgendaProvider({
        "A1": [{"task": "Send the revised budget to finance", "owner": "Bob", "source_point_ids": ["pt-1"]}],
        "A2": [{"task": "Migrate the database cluster", "owner": "Carol", "source_point_ids": ["pt-2"]}],
    })
    with patch("services.ai_provider.get_provider", return_value=provider):
        res = await _populate_action_points_for_agendas(agendas, [], [], target_agenda_ids=["A2"])

    assert [c["agenda"]["agenda_id"] for c in provider.calls] == ["A2"]
    assert "action_points" not in res[0]
    assert res[1]["action_points"][0]["assignee"] == "Carol"
