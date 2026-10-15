"""
Short/medium Final ROM condensing (services/rom_condense.py): attribution from cited points,
grounding, Very Important points, coverage repair and ordering. No model is loaded.
"""
import json
from unittest.mock import MagicMock, patch

from services.rom_condense import (
    assemble_condensed_points,
    condense_source_snapshot,
    format_points_for_condense,
    ground_expansion,
    parse_condensed_items,
)
from services.rom_service import RomService


def _pt(pid, text, speakers, start, **extra):
    point = {"id": pid, "text": text, "polished_text": text, "speakers": speakers, "speaker": speakers[0],
             "timeline_start": float(start), "timeline_end": float(start + 30)}
    point.update(extra)
    return point


def _points():
    return [
        _pt("p1", "Priya asked whether the audit covers the Q3 ledgers; Ravi replied that it covers Q3 only.",
            ["Priya", "Ravi"], 10, type="question_answer", type_source="llm"),
        _pt("p2", "Meena reported the data migration is 80% complete.", ["Meena"], 60, type="update", type_source="llm"),
        _pt("p3", "Ravi agreed to share the SAP-MM migration plan with the CFO by 24 October.", ["Ravi"], 120,
            type="update", type_source="llm", include_action_point=True, action_flag_source="llm", action_owner="Ravi",
            action_items=[{"task": "Share the SAP-MM migration plan with the CFO", "assignee": "Ravi", "deadline": "24 October"}]),
        _pt("p4", "The committee approved a budget of 2.5 Cr for the migration.", ["Anil"], 180,
            type="decision", type_source="llm"),
    ]


def _item(text, *ids, type=None):
    return {"text": text, "type": type, "source_ids": list(ids)}


# ── Input / parsing ──────────────────────────────────────────────────────────

def test_input_header_shows_type_speakers_owner_and_flags():
    pts = _points()
    text = format_points_for_condense(pts, [pts[1]])
    assert "[P1] Type=question_answer | Speakers=Priya, Ravi\n" in text
    assert "[P3] Type=update | Speakers=Ravi | Action Owner=Ravi | ACTION\n" in text
    assert "[P2] Type=update | Speakers=Meena | *** VERY IMPORTANT - PRESERVE INTACT WITHOUT SUMMARIZATION ***" in text


def test_parse_objects_strings_and_fences():
    raw = '<think>plan</think>```json\n[{"text": "A.", "type": "decision", "source_ids": ["P1", "P3"]}, "B.",]\n```'
    assert parse_condensed_items(raw) == [
        {"text": "A.", "type": "decision", "source_ids": ["P1", "P3"]},
        {"text": "B.", "type": None, "source_ids": []},
    ]
    assert parse_condensed_items("1. First point\n2. Second point")[1]["text"] == "Second point"
    assert parse_condensed_items('{"points": [{"text": "C.", "source_point_ids": "P2"}]}')[0]["source_ids"] == ["P2"]


# ── Attribution ──────────────────────────────────────────────────────────────

@patch("services.ai_provider.get_provider")
def test_attribution_comes_from_cited_points_not_position(mock_get_provider):
    """Regression: condensed point j used to take speaker/owner/timeline of input point j."""
    provider = MagicMock()
    provider.query.return_value = json.dumps([
        _item("The committee approved a 2.5 Cr migration budget, and Ravi will share the SAP-MM migration plan "
              "with the CFO by 24 October.", "P3", "P4", type="decision"),
        _item("Ravi confirmed the audit covers the Q3 ledgers only.", "P1", type="question_answer"),
        _item("The data migration is 80% complete.", "P2", type="update"),
    ])
    mock_get_provider.return_value = provider
    final_rom = {"agendas": [{"agenda_id": "A1", "title": "Migration", "discussion_points": _points()}]}

    result = RomService().generate_rom_version(final_rom, version="short", reasoning_mode=False)

    assert provider.query.call_count == 1  # nothing lost, no repair call
    pts = result["agendas"][0]["discussion_points"]
    assert [p["source_point_ids"] for p in pts] == [["p1"], ["p2"], ["p3", "p4"]]  # meeting order
    merged = pts[2]
    assert merged["id"] == "A1-SHORT-3"
    assert merged["speakers"] == ["Ravi", "Anil"] and merged["speaker"] == "Ravi"
    assert merged["action_owner"] == "Ravi"
    assert (merged["timeline_start"], merged["timeline_end"]) == (120.0, 210.0)
    assert [a["task"] for a in merged["action_items"]] == ["Share the SAP-MM migration plan with the CFO"]
    assert merged["type"] == "decision" and merged["include_action_point"] is True
    assert pts[0]["speakers"] == ["Priya", "Ravi"] and pts[0]["action_items"] == []
    assert result["agendas"][0]["condense_report"]["dropped_sentences"] == []


def test_plain_string_output_is_linked_by_wording():
    pts = _points()[:2]
    items = parse_condensed_items(json.dumps([
        "Meena reported that the migration is 80% complete.",
        "Priya asked whether the audit covers the Q3 ledgers; Ravi said Q3 only.",
    ]))
    out, report = assemble_condensed_points("A1", "short", pts, items, [])
    assert [p["source_point_ids"] for p in out] == [["p1"], ["p2"]]
    assert out[1]["speakers"] == ["Meena"]
    assert report["inferred_sources"] == 2


# ── Grounding ────────────────────────────────────────────────────────────────

def test_sentence_with_invented_number_is_dropped():
    pts = [_points()[2]]
    items = [_item("Ravi will share the SAP-MM migration plan with the CFO by 24 October. The plan costs 7 Cr.", "P1")]
    out, report = assemble_condensed_points("A1", "short", pts, items, [])
    assert out[0]["text"] == "Ravi will share the SAP-MM migration plan with the CFO by 24 October."
    assert report["dropped_sentences"] == ["The plan costs 7 Cr."]


def test_fact_from_another_point_cites_that_point():
    pts = _points()[2:]
    items = [_item("Ravi will share the SAP-MM migration plan, funded with 2.5 Cr, with the CFO by 24 October.", "P1")]
    out, _ = assemble_condensed_points("A1", "short", pts, items, [])
    assert out[0]["source_point_ids"] == ["p3", "p4"]


def test_unknown_abbreviation_is_reported_not_dropped():
    pts = [_points()[1]]
    out, report = assemble_condensed_points("A1", "short", pts, [_item("The DB migration is 80% complete.", "P1")], [])
    assert out[0]["text"] == "The DB migration is 80% complete."
    assert report["unverified_terms"] == ["DB"]


def test_point_whose_only_sentence_is_unsupported_is_kept_verbatim():
    pts = [_points()[1]]
    out, report = assemble_condensed_points("A1", "short", pts, [_item("The migration is 95% complete.", "P1")], [])
    assert out[0]["text"] == pts[0]["text"]
    assert report["kept_verbatim"] == ["p2"]


# ── Coverage ─────────────────────────────────────────────────────────────────

def test_lost_decision_is_condensed_again():
    pts = [_points()[0], _points()[3]]
    calls = []

    def repair(missing):
        calls.append([p["id"] for p in missing])
        return [_item("The committee approved a 2.5 Cr migration budget.", "P1", type="decision")]

    out, report = assemble_condensed_points(
        "A1", "short", pts, [_item("Ravi confirmed the audit covers Q3 only.", "P1")], [], repair=repair)
    assert calls == [["p4"]]
    assert [p["text"] for p in out] == ["Ravi confirmed the audit covers Q3 only.",
                                        "The committee approved a 2.5 Cr migration budget."]
    assert out[1]["source_point_ids"] == ["p4"] and report["repaired"] == ["p4"]


def test_lost_action_without_repair_is_kept_verbatim():
    pts = [_points()[0], _points()[2]]
    out, report = assemble_condensed_points("A1", "short", pts, [_item("Ravi confirmed the audit covers Q3 only.", "P1")], [])
    assert out[1]["text"] == pts[1]["text"]
    assert out[1]["action_items"][0]["task"] == "Share the SAP-MM migration plan with the CFO"
    assert report["kept_verbatim"] == ["p3"]


def test_points_typed_only_by_text_cues_are_not_forced_back():
    pts = [_points()[0], _pt("p9", "The board approved the office move.", ["Anil"], 99)]  # no Stage 2 type
    calls = []
    out, _ = assemble_condensed_points("A1", "short", pts, [_item("Ravi confirmed the audit covers Q3 only.", "P1")],
                                       [], repair=lambda missing: calls.append(missing) or [])
    assert calls == [] and len(out) == 1


# ── Very Important points and duplicates ─────────────────────────────────────

def test_very_important_point_stays_verbatim_and_mixed_outputs_stop_citing_it():
    pts = _points()[:2]
    items = [_item("The audit covers Q3 only and the migration is 80% complete.", "P1", "P2")]
    out, _ = assemble_condensed_points("A1", "short", pts, items, [pts[1]])
    assert [p["source_point_ids"] for p in out] == [["p1"], ["p2"]]
    assert out[0]["text"] == items[0]["text"]  # the 80% stays supported by the VI point
    assert out[1]["text"] == pts[1]["text"] and out[1]["is_very_important"] is True


def test_repeated_outputs_are_merged():
    pts = _points()[:2]
    items = [
        _item("Ravi confirmed the audit covers the Q3 ledgers only.", "P1"),
        _item("Ravi confirmed that the audit covers the Q3 ledgers only.", "P1"),
        _item("The data migration is 80% complete.", "P2"),
    ]
    out, report = assemble_condensed_points("A1", "short", pts, items, [])
    assert len(out) == 2 and report["merged_duplicates"] == 1


# ── Traceability: source snapshot, pipeline notes, medium citations ──────────

def test_source_snapshot_keeps_the_ids_the_condense_call_used():
    pts = _points()
    snapshot = condense_source_snapshot(pts, [pts[1]])
    assert [(s["label"], s["id"]) for s in snapshot] == [("P1", "p1"), ("P2", "p2"), ("P3", "p3"), ("P4", "p4")]
    assert snapshot[2]["text"] == pts[2]["text"] and snapshot[2]["speakers"] == ["Ravi"]
    assert snapshot[2]["include_action_point"] is True and snapshot[2]["type"] == "update"
    assert [s["is_very_important"] for s in snapshot] == [False, True, False, False]
    assert format_points_for_condense(snapshot[2:], labels=["P3", "P4"]).startswith("[P3] Type=update")


def test_notes_record_linked_sources_and_fact_check_removals():
    pts = _points()[2:]
    items = [_item("Ravi will share the SAP-MM migration plan, funded with 2.5 Cr, with the CFO by 24 October. "
                   "It costs 7 Cr.", "P1")]
    out, _ = assemble_condensed_points("A1", "short", pts, items, [])
    notes = out[0]["condense_notes"]
    assert any(n.startswith("P2 added as a source") and "2.5" in n for n in notes)
    assert any(n.startswith("Removed in fact check") and "It costs 7 Cr." in n for n in notes)


def test_notes_explain_points_kept_verbatim():
    pts = [_points()[0], _points()[2], _points()[1]]  # P1 audit, P2 action, P3 migration (Very Important)
    items = [_item("Ravi confirmed the audit covers Q3 only and the migration is 80% complete.", "P1", "P3")]
    out, _ = assemble_condensed_points("A1", "short", pts, items, [pts[2]])
    by_sources = {tuple(p["source_point_ids"]): p["condense_notes"] for p in out}
    assert by_sources[("p1",)] == ["P3 is marked Very Important and appears verbatim as its own point."]
    assert by_sources[("p2",)] == ["Marked Very Important: kept verbatim, not condensed."]
    assert by_sources[("p3",)] == ["Kept verbatim: the condensed output left out this action."]


def test_uncited_output_notes_that_sources_were_matched_by_wording():
    pts = _points()[:2]
    out, _ = assemble_condensed_points("A1", "short", pts, parse_condensed_items(
        json.dumps(["Meena reported that the migration is 80% complete."])), [])
    assert out[0]["source_point_ids"] == ["p2"]
    assert out[0]["condense_notes"] == ["The model did not cite its source points; they were matched by wording."]


@patch("services.ai_provider.get_provider")
def test_medium_version_cites_every_merged_point_and_keeps_its_sources(mock_get_provider):
    provider = MagicMock()
    provider.query.return_value = json.dumps([
        _item("Ravi confirmed the audit covers the Q3 ledgers only, and the committee approved a 2.5 Cr "
              "migration budget.", "P1", "P4", type="decision"),
        _item("Meena reported the data migration is 80% complete; Ravi will share the SAP-MM migration plan "
              "with the CFO by 24 October.", "P2", "P3", type="update"),
    ])
    mock_get_provider.return_value = provider
    final_rom = {"agendas": [{"agenda_id": "A1", "title": "Migration", "discussion_points": _points()}]}

    result = RomService().generate_rom_version(final_rom, version="medium")

    assert '"source_ids"' in provider.query.call_args[0][0]
    agenda = result["agendas"][0]
    assert [p["source_point_ids"] for p in agenda["discussion_points"]] == [["p1", "p4"], ["p2", "p3"]]
    assert agenda["condense_report"]["inferred_sources"] == 0
    assert [(s["label"], s["id"]) for s in agenda["condense_sources"]] == [
        ("P1", "p1"), ("P2", "p2"), ("P3", "p3"), ("P4", "p4")]


# ── Expand Point grounding ───────────────────────────────────────────────────

def test_expansion_keeps_only_new_sentences_grounded_in_the_sources():
    sources = _points()[2:]
    current = "Ravi will share the SAP-MM migration plan with the CFO by 24 October."
    kept, dropped = ground_expansion([
        "The committee approved a budget of 2.5 Cr for the migration.",
        "The plan will cost 7 Cr.",  # number no source states
        "Ravi will share the SAP-MM migration plan with the CFO by 24 October.",  # repeats the point
        "The vendor shortlist includes three cloud providers.",  # not in the sources
    ], current, sources)
    assert kept == ["The committee approved a budget of 2.5 Cr for the migration."]
    assert len(dropped) == 3


def test_expansion_adds_at_most_three_sentences():
    source = _pt("p1", "Anil chaired the review. Meena presented the vendor list. Ravi raised the licence risk. "
                       "Priya confirmed the training plan.", ["Anil"], 0)
    kept, dropped = ground_expansion([source["text"]], "The review covered vendors.", [source])
    assert len(kept) == 3 and dropped == ["Priya confirmed the training plan."]
