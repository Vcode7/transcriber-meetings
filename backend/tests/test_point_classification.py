"""
Stage 2 discussion-point type and action flag (services/point_classification.py), and their
use by Stage 2 enhancement and Stage 3 analysis. No model is loaded.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from services.point_classification import (
    analysis_type_from_stage2,
    classify_stage2_point,
    dominant_type,
    heuristic_discussion_type,
    normalize_action_flag,
    normalize_discussion_type,
)


@pytest.mark.parametrize("value,expected", [
    ("decision", "decision"), ("Q&A", "question_answer"), ("question-answer", "question_answer"),
    ("Question", "question_answer"), ("disagreement", "debate"), ("Risk", "issue"), ("blocker", "issue"),
    ("status update", "update"), ("general discussion", "discussion"), ("nonsense", None), (None, None),
])
def test_type_aliases(value, expected):
    assert normalize_discussion_type(value) == expected


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False), ("true", True), ("Yes", True), ("no", False), (1, True), (0, False),
    (None, None), ("maybe", None), ("", None),
])
def test_action_flag_normalisation(value, expected):
    assert normalize_action_flag(value) == expected


def test_dominant_type_follows_precedence():
    assert dominant_type(["update", "debate", "discussion"]) == "debate"
    assert dominant_type(["question_answer", "decision"]) == "decision"
    assert dominant_type(["bogus"]) is None


@pytest.mark.parametrize("text,expected", [
    ("The committee approved the revised budget.", "decision"),
    ("Ravi argued for vendor A while Meena objected on cost grounds.", "debate"),
    ("Priya asked whether the audit covers Q3; Ravi replied that it does.", "question_answer"),
    ("The vendor delay is a risk to the launch.", "issue"),
    ("The migration is on track and 80% complete.", "update"),
    ("The team talked about the office layout.", "discussion"),
])
def test_heuristic_types(text, expected):
    assert heuristic_discussion_type(text) == expected


def test_llm_type_and_flag_win():
    ep = {"polished_text": "The board approved the plan.", "type": "Q&A", "include_action_point": "false"}
    classify_stage2_point(ep, [{"discussion_point": "x"}])
    assert (ep["type"], ep["type_source"]) == ("question_answer", "llm")
    assert (ep["include_action_point"], ep["action_flag_source"]) == (False, "llm")


def test_stage1_action_items_set_the_flag():
    ep = {"polished_text": "Budget discussed.", "type": "update", "include_action_point": False,
          "action_items": [{"task": "Send the revised budget", "assignee": "Ravi"}]}
    classify_stage2_point(ep, [])
    assert (ep["include_action_point"], ep["action_flag_source"]) == (True, "stage1")


def test_missing_llm_fields_fall_back_to_text_cues():
    ep = {"polished_text": "Ravi will send the revised quotation to the client."}
    classify_stage2_point(ep, [{"discussion_point": "..."}])
    assert ep["type_source"] == "heuristic"
    assert (ep["include_action_point"], ep["action_flag_source"]) == (True, "heuristic")


def test_merge_takes_strongest_type_and_any_flag():
    sources = [
        {"type": "update", "type_source": "llm", "include_action_point": False, "action_flag_source": "llm"},
        {"type": "decision", "type_source": "user", "include_action_point": True, "action_flag_source": "llm"},
    ]
    merged = {"polished_text": "Merged text."}
    classify_stage2_point(merged, sources)
    assert (merged["type"], merged["type_source"]) == ("decision", "user")
    assert merged["include_action_point"] is True


def test_user_switched_off_flag_is_not_reenabled_by_stage1_items():
    sources = [{"type": "update", "include_action_point": False, "action_flag_source": "user",
                "action_items": [{"task": "Send the plan"}]}]
    merged = {"polished_text": "Merged text."}
    classify_stage2_point(merged, sources)
    assert merged["include_action_point"] is False


@pytest.mark.parametrize("point,expected", [
    ({"type": "decision", "type_source": "llm", "include_action_point": True}, ("decision", ["decision", "action_item"])),
    ({"type": "update", "type_source": "llm", "include_action_point": True}, ("action_item", ["action_item"])),
    ({"type": "question_answer", "type_source": "user", "include_action_point": False}, ("question", ["question"])),
    ({"type": "issue", "type_source": "llm"}, ("risk", ["risk"])),
    ({"type": "debate", "type_source": "llm"}, ("discussion", ["discussion"])),
])
def test_stage3_mapping(point, expected):
    result = analysis_type_from_stage2(point)
    assert (result["primary_type"], result["types"]) == expected
    assert result["source"] == "stage2"


def test_stage3_mapping_ignores_heuristic_types():
    assert analysis_type_from_stage2({"type": "decision", "type_source": "heuristic"}) is None
    assert analysis_type_from_stage2({"polished_text": "No type"}) is None


# ── Stage 2 enhancement sets the fields ─────────────────────────────────────

class _Embedder:
    _dim = 4

    def load(self):
        pass

    def encode_batch(self, texts):
        return np.ones((len(texts), self._dim), dtype=np.float32)

    def encode(self, texts):
        return np.eye(len(texts), max(len(texts), self._dim), dtype=np.float32)  # all points distinct


def _store():
    return SimpleNamespace(_index=SimpleNamespace(ntotal=0), _meta=[], _dir="mem", load_or_create=lambda: None)


class _Stage2Provider:
    def __init__(self, enhanced):
        self.enhanced = enhanced

    def enhance_rom_discussion_window(self, **kwargs):
        return {"enhanced_points": [dict(p) for p in self.enhanced]}

    def unload_model(self):
        pass


def _run_stage2(stage1_points, enhanced):
    from services.rom_service import RomService

    service = RomService()
    with patch("services.ai_provider.get_provider", return_value=_Stage2Provider(enhanced)), \
            patch("services.text_embedding_service.get_text_embedder", return_value=_Embedder()), \
            patch("services.text_embedding_service.unload_text_embedder"), \
            patch("services.vector_store.get_meeting_context_store", return_value=_store()), \
            patch("services.vector_store.get_global_context_store", return_value=_store()), \
            patch.object(RomService, "index_stage2_points_in_chromadb"):
        return service.enhance_discussion_points(
            discussion_points=stage1_points, recording_id="rec-cls", user_id="user-cls",
            previous_meeting_mode="off", parallel_window_processing=1,
        )


def test_stage2_points_carry_llm_type_and_flag():
    stage1 = [
        {"id": "s1", "discussion_point": "Priya asked about the audit scope.", "speakers": ["Priya"], "timeline_start": 1.0},
        {"id": "s2", "discussion_point": "Ravi will send the audit plan by Friday.", "speakers": ["Ravi"], "timeline_start": 9.0},
    ]
    enhanced = [
        {"original_point_ids": ["s1"], "discussion_point": "Priya asked about the audit scope; Ravi clarified it.",
         "speakers": ["Priya"], "type": "Q&A", "include_action_point": False},
        {"original_point_ids": ["s2"], "discussion_point": "Ravi committed to sending the audit plan by Friday.",
         "speakers": ["Ravi"], "type": "update", "include_action_point": True},
    ]
    points = _run_stage2(stage1, enhanced)
    assert [(p["type"], p["include_action_point"]) for p in points] == [("question_answer", False), ("update", True)]
    assert all(p["type_source"] == "llm" for p in points)


def test_stage2_without_llm_fields_uses_stage1_actions_and_cues():
    stage1 = [{"id": "s1", "discussion_point": "Budget review.", "speakers": ["Ravi"],
               "action_items": [{"task": "Send the revised budget to finance", "assignee": "Ravi"}]}]
    enhanced = [{"original_point_ids": ["s1"], "discussion_point": "The committee approved the revised budget.",
                 "speakers": ["Ravi"]}]
    point = _run_stage2(stage1, enhanced)[0]
    assert (point["type"], point["type_source"]) == ("decision", "heuristic")
    assert (point["include_action_point"], point["action_flag_source"]) == (True, "stage1")


# ── Manual correction endpoint ───────────────────────────────────────────────

async def _set_classification(data, **body):
    from routers.rom_router import PointClassificationRequest, set_stage2_point_classification

    with patch("routers.rom_router._get_rom_data", AsyncMock(return_value=data)), \
            patch("routers.rom_router._save_rom_data", AsyncMock()), \
            patch("routers.rom_router._validate_user_id", return_value="user-cls"):
        return await set_stage2_point_classification(
            "rec-cls", "p1", PointClassificationRequest(**body), current_user={}, db=MagicMock())


@pytest.mark.asyncio
async def test_endpoint_sets_user_type_and_flag_and_marks_final_rom_outdated():
    data = {"stage2": {"polished_points": [{"id": "p1", "polished_text": "x", "type": "update", "type_source": "llm",
                                            "include_action_point": False, "action_flag_source": "llm"}]},
            "final_rom": {"agendas": []}}
    point = (await _set_classification(data, type="Q&A", include_action_point=True))["updated_point"]
    assert (point["type"], point["type_source"]) == ("question_answer", "user")
    assert (point["include_action_point"], point["action_flag_source"]) == (True, "user")
    assert "final_rom" in data["outdated_warnings"]


@pytest.mark.asyncio
async def test_endpoint_rejects_unknown_type():
    from fastapi import HTTPException

    data = {"stage2": {"polished_points": [{"id": "p1", "polished_text": "x"}]}}
    with pytest.raises(HTTPException) as err:
        await _set_classification(data, type="gossip")
    assert err.value.status_code == 400
