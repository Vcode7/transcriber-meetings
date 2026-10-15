"""
Main-MoM action items from enhanced (Stage 2) points: when Stage 2 flagged the points, only
flagged points give action items, sent in groups of 5 with their neighbours as context and
checked like the Final ROM action points. No model is loaded.
"""
import json
from unittest.mock import patch

import pytest

import services.agenda_action_points as aap
from services.ai_provider import QwenProvider
from services.rom_service import RomService


@pytest.fixture(autouse=True)
def _no_embedding_dedupe(monkeypatch):
    monkeypatch.setattr(aap, "EMBEDDING_DEDUPE", False)


def _points(n, flagged, texts=None, per_point=None):
    texts, per_point = texts or {}, per_point or {}
    points = []
    for i in range(n):
        point = {"id": f"p{i}", "polished_text": texts.get(i, f"Point {i} covered routine operations item number {i}."),
                 "speakers": ["Ravi", "Meena"], "timeline_start": float(i)}
        if flagged is not None:
            point["include_action_point"] = i in flagged
        point.update(per_point.get(i, {}))
        points.append(point)
    return points


def _extract(points, respond, chunk_size=10):
    """Run extraction with a scripted LLM; returns (actions, payloads sent per call)."""
    provider = QwenProvider()
    payloads = []

    def fake_infer(prompt, **kwargs):
        points_sent = json.loads(prompt.split("POINTS:\n", 1)[1])
        payloads.append(points_sent)
        return json.dumps({"action_items": respond(points_sent)})

    with patch.object(provider, "_infer", side_effect=fake_infer), \
            patch.object(provider, "_get_active_settings", return_value={}), \
            patch("services.ai_provider._get_prompt", return_value="EXTRACT ACTIONS\nPOINTS:\n{points_json}"):
        actions = provider.extract_actions_from_enhanced_points(points, chunk_size=chunk_size, reasoning_mode=False)
    return actions, payloads


def test_only_groups_with_flagged_points_are_sent_with_roles():
    points = _points(12, {2, 11}, per_point={3: {"action_owner": "Meena", "owner_transcript_excerpt": "[T1 | 00:01] Meena: ok"}})
    _, payloads = _extract(points, lambda pts: [])
    assert len(payloads) == 2
    assert [p["id"] for p in payloads[0]] == ["p0", "p1", "p2", "p3", "p4", "p5"]
    assert [p["id"] for p in payloads[1]] == ["p9", "p10", "p11"]
    roles = {p["id"]: p["role"] for p in payloads[0]}
    assert roles["p2"] == "target" and {r for pid, r in roles.items() if pid != "p2"} == {"context"}
    context = next(p for p in payloads[0] if p["id"] == "p3")
    assert "action_owner" not in context and "transcript_excerpt" not in context
    assert "include_action_point" not in payloads[0][2]


def test_flagged_group_proposals_are_validated_and_deduplicated():
    texts = {
        2: "Ravi agreed to send the revised quotation to the client.",
        3: "The client asked to receive it by 24 October.",
        7: "Ravi again confirmed he will send the revised quotation to the client.",
    }
    points = _points(10, {2, 7}, texts, per_point={2: {"action_owner": "Ravi"}, 7: {"action_owner": "Ravi"}})

    def respond(pts):
        target = next(p["id"] for p in pts if p["role"] == "target")
        return [
            {"source_point_id": target, "task": "Send the revised quotation to the client by 24 October",
             "owner": "Ravi", "deadline": "24 October", "expected_outcome": "Client has the revised quotation"},
            {"source_point_id": "p5", "task": "Prepare the onboarding slides for the new vendor", "owner": None},
            {"source_point_id": target, "task": "Approved the revised quotation", "owner": None},
        ]

    actions, payloads = _extract(points, respond)
    assert len(payloads) == 2
    # Group 1 (p0-p5) states the date in context point p3: kept, citing it. Group 2 (p4-p9) never
    # saw the date, so its copy is an invented detail. Context-only and completed work are removed.
    assert len(actions) == 1
    action = actions[0]
    assert action["owner"] == "Ravi" and action["deadline"] == "24 October"
    assert action["source_point_id"] == "p2" and action["source_point_ids"] == ["p2", "p3"]
    assert action["expected_outcome"] == "Client has the revised quotation"


def test_same_task_from_two_groups_is_merged():
    texts = {2: "Ravi agreed to send the revised quotation to the client.",
             7: "Ravi again confirmed he will send the revised quotation to the client."}
    points = _points(10, {2, 7}, texts, per_point={2: {"action_owner": "Ravi"}, 7: {"action_owner": "Ravi"}})
    actions, payloads = _extract(points, lambda pts: [{
        "source_point_id": next(p["id"] for p in pts if p["role"] == "target"),
        "task": "Send the revised quotation to the client", "owner": "Ravi",
    }])
    assert len(payloads) == 2 and len(actions) == 1
    assert actions[0]["source_point_ids"] == ["p2", "p7"]


def test_unstated_deadline_is_dropped():
    points = _points(3, {0}, {0: "Meena will book the training room for the onboarding session."})
    actions, _ = _extract(points, lambda pts: [
        {"source_point_id": "p0", "task": "Book the training room for the onboarding session", "owner": "Meena", "deadline": "Friday"},
    ])
    assert actions[0]["deadline"] is None


def test_no_flagged_point_means_no_call():
    actions, payloads = _extract(_points(6, set()), lambda pts: [{"source_point_id": "p0", "task": "Anything"}])
    assert payloads == [] and actions == []


def test_points_without_flags_keep_plain_chunks():
    actions, payloads = _extract(_points(12, None), lambda pts: [], chunk_size=5)
    assert [len(p) for p in payloads] == [5, 5, 2]
    assert all("role" not in p for batch in payloads for p in batch)


# ── MoM generation ───────────────────────────────────────────────────────────

class _MomProvider:
    def __init__(self, actions):
        self.actions = actions
        self.extract_calls = 0

    def query(self, prompt, **kwargs):
        return '{"title": "T", "introduction": "Intro.", "conclusion": "End."}'

    def _infer(self, prompt, **kwargs):
        return self.query(prompt)

    def extract_actions_from_enhanced_points(self, points, **kwargs):
        self.extract_calls += 1
        return [dict(a) for a in self.actions]

    def unload_model(self):
        pass


def test_mom_uses_not_specified_for_validated_missing_deadline():
    points = _points(3, {0}, {0: "Meena will book the training room."}, per_point={0: {"action_owner": "Meena"}})
    provider = _MomProvider([{"source_point_id": "p0", "task": "Book the training room", "owner": "Meena", "deadline": None}])
    with patch("services.ai_provider.get_provider", return_value=provider):
        mom = RomService().generate_mom_from_enhanced_rom(points, {"filename": "m", "speakers_detected": ["Meena"]})
    assert provider.extract_calls == 1
    assert mom["action_items"][0]["deadline"] == aap.NOT_SPECIFIED


def test_mom_skips_stage1_items_of_points_flagged_without_action():
    points = _points(2, {1}, per_point={
        0: {"action_items": [{"task": "Unflagged item", "assignee": "Ravi"}], "action_flag_source": "user"},
        1: {"action_items": [{"task": "Send the budget sheet", "assignee": "Meena"}]},
    })
    provider = _MomProvider([])
    with patch("services.ai_provider.get_provider", return_value=provider):
        mom = RomService().generate_mom_from_enhanced_rom(points, {"filename": "m", "speakers_detected": ["Ravi", "Meena"]})
    assert [a["raw_json"]["task"] for a in mom["action_items"]] == ["Send the budget sheet"]


def test_mom_structured_action_without_deadline_is_not_specified_not_point_dates():
    # The point mentions a date, but not as this action's deadline.
    points = _points(1, {0}, {0: "Meena will send the budget sheet; the audit was held on 12 March."}, per_point={
        0: {"dates": ["12 March"], "action_items": [{"task": "Send the budget sheet", "assignee": "Meena"}]},
    })
    with patch("services.ai_provider.get_provider", return_value=_MomProvider([])):
        mom = RomService().generate_mom_from_enhanced_rom(points, {"filename": "m", "speakers_detected": ["Meena"]})
    assert mom["action_items"][0]["deadline"] == aap.NOT_SPECIFIED


class _ScoringMomProvider(_MomProvider):
    def __init__(self, actions, scores, default=60):
        super().__init__(actions)
        self.scores = scores
        self.default = default
        self.score_calls = []

    def score_action_point_importance(self, action_points_json):
        views = json.loads(action_points_json)
        self.score_calls.append(views)
        return {"scores": [{"id": v["id"], "importance": self.scores.get(v["task"], self.default)} for v in views]}


def test_mom_extracted_action_items_are_scored_from_their_source_point():
    points = _points(3, {0}, {0: "Meena will book the training room for the onboarding session on 24 October."},
                     per_point={0: {"action_owner": "Meena", "type": "decision"}})
    provider = _ScoringMomProvider(
        [{"source_point_id": "p0", "task": "Book the training room", "owner": "Meena", "deadline": "24 October"}],
        {"Book the training room": 85},
    )
    with patch("services.ai_provider.get_provider", return_value=provider):
        mom = RomService().generate_mom_from_enhanced_rom(points, {"filename": "m", "speakers_detected": ["Meena"]})
    item = mom["action_items"][0]
    assert (item["importance"], item["importance_source"], item["source_point_id"]) == (85, "llm", "p0")
    view = provider.score_calls[0][0]
    assert view["source_discussion"].startswith("Meena will book the training room") and view["point_type"] == "decision"


def test_mom_structured_stage1_items_are_scored_too():
    points = _points(1, {0}, {0: "Meena will send the budget sheet."}, per_point={
        0: {"action_items": [{"task": "Send the budget sheet", "assignee": "Meena"}]},
    })
    provider = _ScoringMomProvider([], {}, default=55)
    with patch("services.ai_provider.get_provider", return_value=provider):
        mom = RomService().generate_mom_from_enhanced_rom(points, {"filename": "m", "speakers_detected": ["Meena"]})
    assert provider.extract_calls == 0
    item = mom["action_items"][0]
    assert (item["importance"], item["importance_source"], item["source_point_id"]) == (55, "llm", "p0")
    assert provider.score_calls[0][0]["source_discussion"] == "Meena will send the budget sheet."


def test_mom_scoring_falls_back_to_heuristic_without_llm_scores():
    points = _points(3, {0}, {0: "Meena will book the training room."}, per_point={0: {"action_owner": "Meena"}})
    provider = _MomProvider([{"source_point_id": "p0", "task": "Book the training room", "owner": "Meena"}])
    with patch("services.ai_provider.get_provider", return_value=provider):
        mom = RomService().generate_mom_from_enhanced_rom(points, {"filename": "m", "speakers_detected": ["Meena"]})
    assert mom["action_items"][0]["importance_source"] == "heuristic"


def test_mom_router_keeps_importance_through_normalization():
    from routers.mom_router import ActionItem, _normalize_action_items

    stored = _normalize_action_items([
        {"task": "Book the hall", "owner": "Meena", "deadline": "Friday", "importance": 85, "importance_source": "llm",
         "importance_reason": "deadline Friday"},
        {"task": "Older item", "owner": "we", "deadline": ""},
    ])
    assert stored[0] == {"task": "Book the hall", "owner": "Meena", "deadline": "Friday", "importance": 85,
                         "importance_source": "llm", "importance_reason": "deadline Friday"}
    assert stored[1] == {"task": "Older item", "owner": "Unassigned", "deadline": "ASAP"}
    # The PATCH body model keeps the score and leaves no keys on unscored items (exclude_none).
    assert ActionItem(**stored[0]).model_dump(exclude_none=True)["importance"] == 85
    assert "importance" not in ActionItem(**stored[1]).model_dump(exclude_none=True)


class _OverviewProvider(_MomProvider):
    def __init__(self, response):
        super().__init__([])
        self.response = response

    def query(self, prompt, **kwargs):
        return self.response


def _overview(response, points=None, meta=None):
    with patch("services.ai_provider.get_provider", return_value=_OverviewProvider(response)):
        return RomService().generate_mom_overview(
            points or ["Point 1: Meena presented the hostel budget of 42 lakh."],
            meta or {"filename": "Hostel review", "created_at": "2026-10-12 10:00:00", "speakers_detected": ["Meena", "Ravi"]},
        )


def test_mom_overview_unparsable_answer_states_only_known_facts():
    overview = _overview("Sure! Here is the overview you asked for.")
    assert overview["title"] == "Hostel review"
    assert overview["introduction"] == "This record covers 1 discussion point(s) from 'Hostel review' with Meena, Ravi."
    text = (overview["introduction"] + overview["conclusion"]).lower()
    assert "consensus" not in text and "finalized" not in text


def test_mom_overview_parses_fenced_json_and_drops_invented_facts():
    response = (
        "Here you go:\n```json\n"
        '{"title": "Q3 Hostel Budget Review", '
        '"introduction": "Meena presented the hostel budget of 42 lakh on October 12, 2026. A 75 lakh grant was approved.", '
        '"conclusion": "The meeting reviewed the renovation plan."}\n```'
    )
    overview = _overview(response)
    assert overview["title"] == "Hostel review"               # "Q3" is not a fact of this meeting
    assert overview["introduction"] == "Meena presented the hostel budget of 42 lakh on October 12, 2026."
    assert overview["conclusion"] == "The meeting reviewed the renovation plan."
