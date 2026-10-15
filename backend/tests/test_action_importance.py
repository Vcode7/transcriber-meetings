"""
Action-point importance (40-100%): parsing, calibration against inflated scores, the heuristic
fallback, the shared LLM scoring pass, and the display/export threshold. No model is loaded.
"""
import json

import pytest

from services.action_importance import (
    MODERATE_CAP,
    SCORE_BATCH_SIZE,
    calibrate,
    carry_importance,
    clamp_min_importance,
    format_importance,
    heuristic_importance,
    importance_summary,
    normalize_importance,
    passes_importance,
    score_action_points,
    scoring_views,
)


class ScoringProvider:
    """Scores each action point from a {task: score} table; ids it does not know are left out."""

    def __init__(self, scores, fail=False, reply=None):
        self.scores = scores
        self.fail = fail
        self.reply = reply
        self.calls = []
        self.unloaded = False

    def score_action_point_importance(self, action_points_json):
        views = json.loads(action_points_json)
        self.calls.append(views)
        if self.fail:
            raise RuntimeError("LLM down")
        if self.reply is not None:
            return self.reply
        return {"scores": [
            {"id": v["id"], "importance": self.scores[v["task"]], "reason": f"evidence for {v['task']}"}
            for v in views if v["task"] in self.scores
        ]}

    def unload_model(self):
        self.unloaded = True


# ── Parsing ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [
    (85, 85), ("85", 85), ("85%", 85), (" 90 % ", 90), (0.85, 85), ("0.7", 70), (72.4, 72),
    (120, 100), (10, 40), (0, 40), (1, 40), ("high", None), (None, None), (True, None), ("", None),
])
def test_normalize_importance(value, expected):
    assert normalize_importance(value) == expected


# ── Calibration ──────────────────────────────────────────────────────────────

def test_critical_score_without_evidence_is_capped():
    item = {"task": "Prepare the onboarding slides for the new vendor", "assignee": "Meena", "deadline": "Not specified"}
    score, adjustment = calibrate(95, item, {})
    assert score == MODERATE_CAP
    assert adjustment.startswith("capped from 95%")


@pytest.mark.parametrize("item, context", [
    ({"task": "Prepare the onboarding slides", "deadline": "24 October"}, {}),
    ({"task": "Submit the audit response by 2026-10-30", "deadline": "Not specified"}, {}),
    ({"task": "Submit the audit response by the end of the week"}, {}),
    ({"task": "Implement the approved fee structure", "basis": "decision"}, {}),
    ({"task": "Implement the revised fee structure"}, {"point_type": "decision"}),
    ({"task": "Replace the failing pump", "priority": "high"}, {}),
    ({"task": "Resolve the blocker on the integration build"}, {}),
    ({"task": "Fix the payroll export", "evidence_quote": "this is urgent, salaries are due"}, {}),
])
def test_critical_score_with_evidence_is_kept(item, context):
    assert calibrate(90, item, context) == (90, None)


def test_moderate_and_low_scores_are_not_changed():
    item = {"task": "Prepare the onboarding slides"}
    assert calibrate(75, item) == (75, None)
    assert calibrate(45, item) == (45, None)


# ── Heuristic fallback ───────────────────────────────────────────────────────

def test_heuristic_bands():
    critical = {"task": "Submit the compliance report", "priority": "high", "deadline": "30 Oct", "assignee": "Ravi"}
    moderate = {"task": "Share the revised layout drawings", "priority": "medium", "assignee": "Meena"}
    vague = {"task": "Collect feedback on the newsletter", "priority": "low"}
    mom_item = {"task": "Share the revised layout drawings", "owner": "Meena", "deadline": "Not specified"}
    assert heuristic_importance(critical) >= 80
    assert 60 <= heuristic_importance(moderate) < 80
    assert heuristic_importance(vague) <= 50
    assert 60 <= heuristic_importance(mom_item) < 80
    for item in (critical, moderate, vague, mom_item):
        assert heuristic_importance(item) % 5 == 0


def test_heuristic_never_exceeds_cap_without_evidence():
    item = {"task": "Collect the attendance sheets", "priority": "medium", "assignee": "Ravi", "basis": "commitment"}
    assert heuristic_importance(item) <= MODERATE_CAP


# ── Shared LLM scoring pass ──────────────────────────────────────────────────

def test_llm_scores_are_applied_by_id_with_reason_and_cap():
    items = [
        {"task": "Submit the tender documents", "assignee": "Ravi", "deadline": "24 October"},
        {"task": "Prepare the onboarding slides", "assignee": "Meena", "deadline": "Not specified"},
        {"task": "Circulate the meeting photos", "assignee": "Not specified", "deadline": "Not specified"},
    ]
    provider = ScoringProvider({"Submit the tender documents": 90, "Prepare the onboarding slides": 95,
                                "Circulate the meeting photos": "45%"})
    summary = score_action_points(items, [{"agenda": "Procurement", "source_discussion": "Ravi will submit..."}],
                                  provider=provider)
    assert [it["importance"] for it in items] == [90, MODERATE_CAP, 45]
    assert all(it["importance_source"] == "llm" for it in items)
    assert items[0]["importance_reason"] == "evidence for Submit the tender documents"
    assert "importance_adjusted" in items[1] and "importance_adjusted" not in items[0]
    assert summary == {"source": "llm", "scored": 3, "llm_scored": 3, "capped": 1}
    assert not provider.unloaded  # a caller-supplied provider stays loaded
    view = provider.calls[0][0]
    assert view["id"] == "AP1" and view["agenda"] == "Procurement" and view["source_discussion"] == "Ravi will submit..."
    assert "priority" not in view


def test_missing_score_falls_back_to_heuristic():
    items = [{"task": "Submit the tender documents", "deadline": "24 October"}, {"task": "Book the hall"}]
    summary = score_action_points(items, provider=ScoringProvider({"Submit the tender documents": 85}))
    assert items[0]["importance_source"] == "llm"
    assert items[1]["importance_source"] == "heuristic" and 40 <= items[1]["importance"] <= 100
    assert summary["source"] == "llm_with_fallback"


def test_llm_failure_or_missing_method_gives_heuristic_scores():
    items = [{"task": "Submit the tender documents", "deadline": "24 October"}]
    assert score_action_points(items, provider=ScoringProvider({}, fail=True))["source"] == "heuristic"
    assert items[0]["importance_source"] == "heuristic"

    class NoScoring:
        pass

    items = [{"task": "Book the hall"}]
    assert score_action_points(items, provider=NoScoring())["source"] == "heuristic"
    assert items[0]["importance_source"] == "heuristic"


def test_bare_array_and_lowercase_ids_are_read():
    items = [{"task": "Submit the tender documents", "deadline": "24 October"}]
    provider = ScoringProvider({}, reply=[{"id": "ap1", "importance": 80}])
    score_action_points(items, provider=provider)
    assert items[0]["importance"] == 80 and items[0]["importance_source"] == "llm"


def test_large_lists_are_sent_in_batches_with_ids_restarting():
    items = [{"task": f"Deliver report {i}", "deadline": "Friday"} for i in range(SCORE_BATCH_SIZE + 3)]
    provider = ScoringProvider({it["task"]: 70 for it in items})
    score_action_points(items, provider=provider)
    assert [len(c) for c in provider.calls] == [SCORE_BATCH_SIZE, 3]
    assert provider.calls[1][0]["id"] == "AP1"
    assert all(it["importance"] == 70 for it in items)


def test_rescoring_replaces_old_fields():
    items = [{"task": "Book the hall", "importance": 95, "importance_adjusted": "old", "importance_reason": "old"}]
    score_action_points(items, provider=ScoringProvider({"Book the hall": 60}))
    assert items[0]["importance"] == 60 and "importance_adjusted" not in items[0]


def test_no_items_loads_no_provider(monkeypatch):
    def fail():
        raise AssertionError("provider must not be loaded")

    monkeypatch.setattr("services.ai_provider.get_provider", fail)
    assert score_action_points([])["source"] == "none"


def test_owned_provider_is_unloaded(monkeypatch):
    provider = ScoringProvider({"Book the hall": 60})
    monkeypatch.setattr("services.ai_provider.get_provider", lambda: provider)
    score_action_points([{"task": "Book the hall"}])
    assert provider.unloaded


def test_scoring_views_truncate_source():
    views = scoring_views([{"task": "Book the hall", "owner": "Unassigned"}], [{"source_discussion": "x" * 2000}])
    assert views[0]["owner"] == "Not specified" and len(views[0]["source_discussion"]) == 600


# ── Display / export ─────────────────────────────────────────────────────────

def test_threshold_filter():
    assert passes_importance({"importance": 70}, 70)
    assert not passes_importance({"importance": 65}, 70)
    assert passes_importance({"task": "unscored"}, 90)      # scored before importance existed
    assert passes_importance({"importance": 45}, 0)          # off
    assert passes_importance({"importance": 45}, None)
    assert clamp_min_importance("80") == 80 and clamp_min_importance(150) == 100 and clamp_min_importance(-3) == 0


def test_format_and_carry():
    assert format_importance({"importance": 85}) == "85%" and format_importance({}) == ""
    dst = carry_importance({"importance": "85", "importance_source": "llm", "importance_reason": "deadline"}, {"task": "t"})
    assert dst == {"task": "t", "importance": 85, "importance_source": "llm", "importance_reason": "deadline"}
    assert carry_importance({"task": "t"}, {"task": "t"}) == {"task": "t"}


@pytest.mark.asyncio
async def test_threshold_setting_defaults_to_off_and_is_saved():
    from unittest.mock import AsyncMock, MagicMock, patch

    from pydantic import ValidationError

    from models.settings import UserSettings, UserSettingsUpdate
    from routers.settings_router import get_settings, update_settings

    assert UserSettings(user_id="u").action_point_min_importance == 0
    with pytest.raises(ValidationError):
        UserSettingsUpdate(action_point_min_importance=150)

    db = MagicMock()
    no_row = MagicMock()
    no_row.mappings.return_value.fetchone.return_value = None
    db.execute = AsyncMock(return_value=no_row)
    with patch("routers.settings_router.get_db_context") as ctx:
        ctx.return_value.__aenter__.return_value = db
        ctx.return_value.__aexit__ = AsyncMock()
        assert (await get_settings(current_user={"id": "u"}))["action_point_min_importance"] == 0

    updated = MagicMock()
    updated.rowcount = 1
    db.execute = AsyncMock(return_value=updated)
    db.commit = AsyncMock()
    with patch("routers.settings_router.get_db_context") as ctx:
        ctx.return_value.__aenter__.return_value = db
        ctx.return_value.__aexit__ = AsyncMock()
        await update_settings(body=UserSettingsUpdate(action_point_min_importance=70), current_user={"id": "u"})
    sql, params = db.execute.call_args[0]
    assert "action_point_min_importance = :action_point_min_importance" in str(sql)
    assert params["action_point_min_importance"] == 70


@pytest.mark.asyncio
async def test_load_min_importance():
    from unittest.mock import AsyncMock, MagicMock

    from services.action_importance import load_min_importance

    db = MagicMock()
    result = MagicMock()
    result.fetchone.return_value = (80,)
    db.execute = AsyncMock(return_value=result)
    assert await load_min_importance(db, "u") == 80
    result.fetchone.return_value = None
    assert await load_min_importance(db, "u") == 0
    db.execute = AsyncMock(side_effect=RuntimeError("no such column"))
    assert await load_min_importance(db, "u") == 0


def test_summary_counts():
    items = [{"importance": 80, "importance_source": "llm"}, {"importance": 60, "importance_source": "heuristic",
                                                                "importance_adjusted": "x"}, {"task": "unscored"}]
    assert importance_summary(items) == {"source": "llm_with_fallback", "scored": 2, "llm_scored": 1, "capped": 1}
