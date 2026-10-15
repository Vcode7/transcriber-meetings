"""Stage 2 output linking: every Stage 1 point is accounted for, attribution comes only from the
points an output covers, and the correction pass changes text only."""
from services.rom_service import _apply_stage2_corrections, _link_enhanced_points


def _stage1():
    return [
        {"id": "uuid-budget", "discussion_point": "Meena presented the hostel renovation budget of 42 lakh.",
         "speakers": ["Meena"], "action_owner": None, "timeline_start": 0.0, "timeline_end": 120.0,
         "technical_terms": ["PWD"], "dates": ["July 8, 2026"], "numbers": ["42 lakh"]},
        {"id": "uuid-tender", "discussion_point": "Ravi will float the tender for the hostel renovation by Friday.",
         "speakers": ["Ravi"], "action_owner": "Ravi", "owner_confidence": "high", "timeline_start": 0.0,
         "timeline_end": 120.0, "action_items": [{"task": "Float the renovation tender", "assignee": "Ravi"}]},
        {"id": "uuid-canteen", "discussion_point": "The canteen vendor contract expires in March and needs review.",
         "speakers": ["Anita"], "action_owner": None, "timeline_start": 120.0, "timeline_end": 240.0},
        {"id": "uuid-library", "discussion_point": "Library extended hours were approved for the exam season.",
         "speakers": ["Anita"], "action_owner": None, "timeline_start": 120.0, "timeline_end": 240.0,
         "include_action_point": False},
    ]


def _texts(points):
    return [p["polished_text"] for p in points]


def test_empty_answer_carries_every_stage1_point_in_order():
    s1 = _stage1()
    out = _link_enhanced_points([], s1, separate_action_extraction=False)

    assert [p["original_point_ids"] for p in out] == [["uuid-budget"], ["uuid-tender"], ["uuid-canteen"], ["uuid-library"]]
    assert all(p["enhancement_fallback"] for p in out)
    # Facts are kept on the fallback (they were dropped before).
    assert out[0]["dates"] == ["July 8, 2026"] and out[0]["numbers"] == ["42 lakh"]
    assert out[1]["action_owner"] == "Ravi"


def test_partial_answer_links_short_ids_and_carries_the_missing_point():
    s1 = _stage1()
    enhanced = [
        {"original_point_ids": ["P1", "P2"], "discussion_point": "Meena presented the 42 lakh hostel renovation budget; "
                                                                  "Ravi will float the tender by Friday.", "speakers": ["Meena", "Ravi"]},
        {"original_point_ids": ["P4"], "discussion_point": "Extended library hours were approved for exams.", "speakers": ["Anita"]},
    ]
    out = _link_enhanced_points(enhanced, s1, separate_action_extraction=False)

    assert [p["original_point_ids"] for p in out] == [["uuid-budget", "uuid-tender"], ["uuid-canteen"], ["uuid-library"]]
    merged, carried, library = out
    assert not merged.get("enhancement_fallback") and carried["enhancement_fallback"] and not library.get("enhancement_fallback")
    assert merged["action_owner"] == "Ravi"
    assert merged["dates"] == ["July 8, 2026"] and merged["technical_terms"] == ["PWD"]
    assert carried["polished_text"] == s1[2]["discussion_point"]


def test_stage1_uuids_and_bare_numbers_still_resolve():
    s1 = _stage1()
    enhanced = [
        {"original_point_ids": ["uuid-canteen"], "discussion_point": "The canteen vendor contract expires in March."},
        {"original_point_ids": [2], "discussion_point": "Ravi will float the renovation tender."},
    ]
    out = _link_enhanced_points(enhanced, s1, separate_action_extraction=False)
    by_source = {p["original_point_ids"][0]: p for p in out}
    assert not by_source["uuid-canteen"].get("enhancement_fallback")
    assert not by_source["uuid-tender"].get("enhancement_fallback")
    assert by_source["uuid-budget"]["enhancement_fallback"] and by_source["uuid-library"]["enhancement_fallback"]


def test_output_without_ids_is_linked_by_wording_and_unrelated_output_is_dropped():
    s1 = _stage1()
    enhanced = [
        {"discussion_point": "Ravi will float the tender for the hostel renovation by Friday."},
        {"original_point_ids": ["P9"], "discussion_point": "Quarterly sports day logistics were finalised."},
    ]
    out = _link_enhanced_points(enhanced, s1, separate_action_extraction=False)

    assert len(out) == 4                                    # the unrelated output is not kept
    tender = next(p for p in out if p["original_point_ids"] == ["uuid-tender"])
    assert not tender.get("enhancement_fallback")
    assert "sports day" not in " ".join(_texts(out))


def test_missing_ids_never_copy_one_points_owner_onto_another():
    s1 = _stage1()
    # Neither output cites ids; the canteen output must not inherit Ravi's owner or action.
    enhanced = [{"discussion_point": "The canteen vendor contract expires in March and needs review."}]
    out = _link_enhanced_points(enhanced, s1, separate_action_extraction=True)

    canteen = next(p for p in out if p["original_point_ids"] == ["uuid-canteen"])
    assert canteen["action_owner"] is None and canteen["action_items"] == []
    assert canteen["timeline_start"] == 120.0


def test_a_stage1_points_actions_go_to_the_first_output_citing_it_only():
    s1 = _stage1()
    enhanced = [
        {"original_point_ids": ["P2"], "discussion_point": "Ravi will float the tender."},
        {"original_point_ids": ["P1", "P2"], "discussion_point": "The 42 lakh budget covers the tender Ravi will float."},
    ]
    out = _link_enhanced_points(enhanced, s1, separate_action_extraction=True)

    by_text = {p["polished_text"]: p for p in out}
    first, second = by_text[enhanced[0]["discussion_point"]], by_text[enhanced[1]["discussion_point"]]
    # Meeting order: the point that also covers P1 comes first, but the first *output* owns the action.
    assert out.index(second) < out.index(first)
    assert [a["task"] for a in first["action_items"]] == ["Float the renovation tender"]
    assert second["action_items"] == []
    # Owners still describe both points that cite the commitment.
    assert first["action_owner"] == "Ravi" and second["action_owner"] == "Ravi"


def test_correction_changes_text_only():
    points = [
        {"id": "s2-a", "polished_text": "Ravi will float the tender.", "speakers": ["Ravi"],
         "action_items": [{"task": "Float the tender", "assignee": "Ravi"}], "dates": ["Friday"], "technical_terms": ["PWD"]},
        {"id": "s2-b", "polished_text": "Anita approved library hours.", "speakers": ["Anita"], "action_items": []},
    ]
    corrected = [
        {"id": "P1", "polished_text": "Ravi will float the hostel tender.", "speakers": ["Bob"],
         "action_items": [{"task": "Invented task", "assignee": "Bob"}], "dates": ["Monday"]},
        {"id": "unknown", "polished_text": "Ignored."},
    ]
    out, count = _apply_stage2_corrections(points, corrected)

    assert count == 1
    assert out[0]["polished_text"] == out[0]["discussion_point"] == "Ravi will float the hostel tender."
    assert out[0]["stage2_corrected"] is True
    assert out[0]["speakers"] == ["Ravi"] and out[0]["dates"] == ["Friday"] and out[0]["technical_terms"] == ["PWD"]
    assert out[0]["action_items"] == [{"task": "Float the tender", "assignee": "Ravi"}]
    assert out[1] is points[1]
