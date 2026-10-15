"""
Short/Medium ROM points: the on-demand explanation, Expand Point and its Revert
(services/rom_service.py, routers/rom_router.py), and Conclude keeping the original point IDs.
No model is loaded.
"""
import copy
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from routers.rom_router import (
    CondensedPointRequest,
    ExplainPointRequest,
    _condense_source_index,
    _condensed_point_request,
    _patch_version_point,
    expand_condensed_point,
    explain_condensed_point,
    revert_expanded_point,
)
from services.rom_condense import condense_source_snapshot
from services.rom_service import RomService

CONDENSED_TEXT = "The committee approved a 2.5 Cr migration budget; the migration is 80% complete."


def _long_points():
    return [
        {"id": "s2-a", "text": "Priya asked whether the audit covers the Q3 ledgers; Ravi replied that it covers Q3 only.",
         "speakers": ["Priya", "Ravi"], "type": "question_answer"},
        {"id": "s2-b", "text": "Meena reported the data migration is 80% complete and the cut-over is planned for 3 November.",
         "speakers": ["Meena"], "type": "update"},
        {"id": "s2-c", "text": "The committee approved a budget of 2.5 Cr for the migration after Anil compared three vendor quotes.",
         "speakers": ["Anil"], "type": "decision"},
    ]


def _rom_data(active="short", snapshot=True):
    long_points = _long_points()
    hiring = {"agenda_id": "A2", "title": "Hiring",
              "discussion_points": [{"id": "s2-z", "text": "Ravi will hire two analysts by March."}]}
    agenda = {"agenda_id": "A1", "title": "Migration", "discussion_points": [
        {"id": "A1-SHORT-1", "text": CONDENSED_TEXT, "source_point_ids": ["s2-b", "s2-c"],
         "condense_notes": ["Merged with a near-duplicate condensed point that said the same thing."]},
    ]}
    if snapshot:
        agenda["condense_sources"] = condense_source_snapshot(long_points)
    short = {"agendas": [agenda]}
    return {
        "final_rom_versions": {"long": {"agendas": [{"agenda_id": "A1", "title": "Migration",
                                                     "discussion_points": long_points}, hiring]},
                               "short": short},
        "final_rom": copy.deepcopy(short),
        "final_rom_active_version": active,
        "stage2": {"polished_points": long_points + hiring["discussion_points"]
                   + [{"id": "s2-y", "polished_text": "Only in Stage 2."}]},
    }


def _request(**extra):
    return CondensedPointRequest(version="short", point_id="A1-SHORT-1", **extra)


def _provider(reply):
    provider = MagicMock()
    provider.query.return_value = reply
    return provider


# ── Router helpers ───────────────────────────────────────────────────────────

def test_source_index_prefers_the_snapshot_then_falls_back_to_long_and_stage2():
    index = _condense_source_index(_rom_data(), "short")
    assert index["s2-c"]["label"] == "P3" and index["s2-c"]["text"].startswith("The committee approved")

    old = _condense_source_index(_rom_data(snapshot=False), "short")  # generated before snapshots
    assert old["s2-c"]["label"] == "P3" and old["s2-z"]["label"] == "P1"
    assert old["s2-y"]["text"] == "Only in Stage 2." and "label" not in old["s2-y"]


def test_point_request_resolves_only_the_mapped_original_points():
    version, agenda, stored, text, ids, sources = _condensed_point_request(_rom_data(), _request())
    assert (version, agenda["agenda_id"], text, ids) == ("short", "A1", CONDENSED_TEXT, ["s2-b", "s2-c"])
    assert [s["id"] for s in sources] == ["s2-b", "s2-c"] and stored["id"] == "A1-SHORT-1"

    # Unsaved client edits win: text and a merged (concluded) point's source IDs.
    _, _, _, text, ids, sources = _condensed_point_request(
        _rom_data(), _request(point_text="Edited.", source_point_ids=["s2-a", "s2-c"]))
    assert text == "Edited." and [s["label"] for s in sources] == ["P1", "P3"]

    with pytest.raises(HTTPException) as exc:
        _condensed_point_request(_rom_data(), _request(source_point_ids=["unknown"]))
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException):
        _condensed_point_request(_rom_data(), CondensedPointRequest(version="long", point_id="A1-SHORT-1"))


def test_patch_updates_the_version_and_the_active_final_rom_only():
    data = _rom_data(active="short")
    assert _patch_version_point(data, "short", "A1-SHORT-1", {"text": "New."}, remove=("condense_notes",))
    for rom in (data["final_rom_versions"]["short"], data["final_rom"]):
        point = rom["agendas"][0]["discussion_points"][0]
        assert point["text"] == "New." and "condense_notes" not in point

    data = _rom_data(active="long")
    assert _patch_version_point(data, "short", "A1-SHORT-1", {"text": "New."})
    assert data["final_rom"]["agendas"][0]["discussion_points"][0]["text"] == CONDENSED_TEXT
    assert not _patch_version_point(data, "short", "missing", {"text": "New."})


# ── Service ──────────────────────────────────────────────────────────────────

@patch("services.ai_provider.get_provider")
def test_expand_uses_only_mapped_sources_and_appends_grounded_sentences(mock_get_provider):
    provider = _provider(json.dumps({"sentences": [
        "The cut-over is planned for 3 November.",
        "Anil compared three vendor quotes before the approval.",
        "The budget may rise to 4 Cr.",
    ]}))
    mock_get_provider.return_value = provider
    _, _, _, text, _, sources = _condensed_point_request(_rom_data(), _request())

    result = RomService().expand_condensed_point(text, sources, "Migration")

    prompt = provider.query.call_args[0][0]
    assert "cut-over is planned for 3 November" in prompt and "three vendor quotes" in prompt
    assert "Q3 ledgers" not in prompt and "two analysts" not in prompt
    assert "[P2] Type=update" in prompt and "[P3] Type=decision" in prompt
    assert provider.query.call_args[1]["task_key"] == "rom_expand_condensed_point"
    assert provider.query.call_args[1]["think"] is False and provider.unload_model.called
    assert result["added_sentences"] == ["The cut-over is planned for 3 November.",
                                         "Anil compared three vendor quotes before the approval."]
    assert result["text"] == f"{CONDENSED_TEXT} The cut-over is planned for 3 November. " \
                             "Anil compared three vendor quotes before the approval."
    assert result["dropped_sentences"] == ["The budget may rise to 4 Cr."]


@patch("services.ai_provider.get_provider")
def test_explain_parses_the_fields_and_passes_pipeline_notes(mock_get_provider):
    provider = _provider('```json\n{"retained": "The 2.5 Cr approval.", "removed": "The vendor comparison.", '
                         '"merged": "P2 and P3.", "why": "Short keeps outcomes.",}\n```')
    mock_get_provider.return_value = provider
    _, _, _, text, _, sources = _condensed_point_request(_rom_data(), _request())

    explanation = RomService().explain_condensed_point(
        text, sources, notes=["Removed in fact check: X."], version="short", agenda_title="Migration")

    assert explanation == {"retained": "The 2.5 Cr approval.", "removed": "The vendor comparison.",
                           "merged": "P2 and P3.", "why": "Short keeps outcomes."}
    prompt = provider.query.call_args[0][0]
    assert "- Removed in fact check: X." in prompt and "A Short Record of Meeting" in prompt
    assert "[P2] Type=update" in prompt and "Q3 ledgers" not in prompt


@patch("services.ai_provider.get_provider")
def test_explain_verbatim_point_needs_no_model(mock_get_provider):
    source = condense_source_snapshot(_long_points())[1]
    explanation = RomService().explain_condensed_point(
        source["text"], [source], notes=["Kept verbatim: the condensed output left out this action."])
    assert not mock_get_provider.called
    assert explanation["merged"] == "Not merged: written from P2 only."
    assert explanation["why"].startswith("Kept verbatim")


@patch("services.ai_provider.get_provider")
def test_conclude_keeps_the_original_points_of_condensed_points(mock_get_provider):
    mock_get_provider.return_value = _provider(
        '{"conclusion": "The budget was approved and the migration is 80% complete.", "explanation": "Combined."}')
    points = [{"id": "A1-SHORT-1", "text": "Budget approved.", "source_point_ids": ["s2-c"]},
              {"id": "A1-SHORT-2", "text": "Migration 80% complete.", "source_point_ids": ["s2-b", "s2-c"]}]
    concluded = RomService().conclude_points(points, agenda_context="A1: Migration")["concluded_point"]
    assert concluded["source_point_ids"] == ["s2-c", "s2-b"]
    assert concluded["condense_notes"] == ["Concluded from 2 condensed points."]


# ── Endpoints ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("routers.rom_router._sync_final_rom_embeddings", new_callable=AsyncMock)
@patch("routers.rom_router._save_rom_data", new_callable=AsyncMock)
@patch("routers.rom_router._get_rom_data", new_callable=AsyncMock)
async def test_expand_saves_the_point_and_revert_restores_it(mock_get, mock_save, mock_sync):
    data = _rom_data()
    mock_get.return_value = data
    added = ["The cut-over is planned for 3 November."]
    with patch("routers.rom_router.rom_service.expand_condensed_point",
               return_value={"text": f"{CONDENSED_TEXT} {added[0]}", "added_sentences": added, "dropped_sentences": []}):
        res = await expand_condensed_point("rec-1", _request(), current_user={"sub": "user-1"}, db=MagicMock())

    assert res["persisted"] is True and res["added_sentences"] == added
    for rom in (data["final_rom_versions"]["short"], data["final_rom"]):
        point = rom["agendas"][0]["discussion_points"][0]
        assert point["text"] == point["polished_text"] == f"{CONDENSED_TEXT} {added[0]}"
        assert point["pre_expand_text"] == CONDENSED_TEXT and point["expanded_sentences"] == added
    assert mock_save.await_count == 1 and mock_sync.await_count == 1

    res = await revert_expanded_point("rec-1", _request(), current_user={"sub": "user-1"}, db=MagicMock())
    point = data["final_rom_versions"]["short"]["agendas"][0]["discussion_points"][0]
    assert res["point"]["text"] == point["text"] == CONDENSED_TEXT
    assert "pre_expand_text" not in point and "expanded_sentences" not in point
    assert mock_save.await_count == 2


@pytest.mark.asyncio
@patch("routers.rom_router._save_rom_data", new_callable=AsyncMock)
@patch("routers.rom_router._get_rom_data", new_callable=AsyncMock)
async def test_expand_with_nothing_supported_changes_nothing(mock_get, mock_save):
    mock_get.return_value = _rom_data()
    with patch("routers.rom_router.rom_service.expand_condensed_point",
               return_value={"text": CONDENSED_TEXT, "added_sentences": [], "dropped_sentences": ["Made up."]}):
        res = await expand_condensed_point("rec-1", _request(), current_user={"sub": "user-1"}, db=MagicMock())
    assert res["point"] is None and res["persisted"] is False and not mock_save.called

    with pytest.raises(HTTPException):
        await revert_expanded_point("rec-1", _request(), current_user={"sub": "user-1"}, db=MagicMock())


@pytest.mark.asyncio
@patch("routers.rom_router._save_rom_data", new_callable=AsyncMock)
@patch("routers.rom_router._get_rom_data", new_callable=AsyncMock)
async def test_explanation_is_generated_once_then_served_from_the_point(mock_get, mock_save):
    data = _rom_data()
    mock_get.return_value = data
    fields = {"retained": "R.", "removed": "X.", "merged": "P2 and P3.", "why": "W."}
    req = ExplainPointRequest(version="short", point_id="A1-SHORT-1")
    with patch("routers.rom_router.rom_service.explain_condensed_point", return_value=fields) as explain:
        first = await explain_condensed_point("rec-1", req, current_user={"sub": "user-1"}, db=MagicMock())
        second = await explain_condensed_point("rec-1", req, current_user={"sub": "user-1"}, db=MagicMock())

    assert explain.call_count == 1 and mock_save.await_count == 1
    assert explain.call_args[1]["notes"] == ["Merged with a near-duplicate condensed point that said the same thing."]
    assert first["explanation"]["merged"] == second["explanation"]["merged"] == "P2 and P3."
    saved = data["final_rom_versions"]["short"]["agendas"][0]["discussion_points"][0]["condense_explanation"]
    assert saved["why"] == "W." and saved["generated_at"]
