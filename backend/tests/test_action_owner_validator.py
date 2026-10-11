from services.action_owner_validator import (
    canonicalize_owner,
    canonicalize_owner_list,
    dedupe_point_actions,
    evidence_outside_focus,
    validate_action,
    validate_point_owners,
    verify_owner_against_candidates,
)
from services.rom_transcript_format import build_roster, build_turns


def _meeting():
    segments = [
        {"speaker_label": "Alice Kumar", "start": 0.0, "end": 4.0, "text": "Let's review the budget."},
        {"speaker_label": "Alice Kumar", "start": 4.5, "end": 8.0, "text": "Speaker 3, can you send the revised budget by Friday?"},
        {"speaker_label": "Speaker 3", "start": 8.2, "end": 9.0, "text": "Sure, I'll send it."},
        {"speaker_label": "Speaker 10", "start": 9.5, "end": 15.0, "text": "I will update the test plan.", "speaker_confidence": 0.4},
        {"speaker_label": "Alice Kumar", "start": 16.0, "end": 19.0, "text": "Could someone check the logs?"},
        {"speaker_label": "Speaker 3", "start": 19.5, "end": 21.0, "text": "Will do, I'll check them today."},
    ]
    turns = build_turns(segments)
    roster = build_roster(segments, {"Speaker 3": "Ravi Shah"})
    return {t["turn_id"]: t for t in turns}, roster


def test_requester_named_as_owner_is_corrected_to_addressed_person():
    turns_by_id, roster = _meeting()
    v = validate_action(
        {"task": "Send revised budget", "assigner": "Alice", "assignee": "Alice Kumar",
         "assignee_basis": "direct_assignment", "evidence_turn_ids": ["T1", "T2"], "owner_confidence": "high"},
        turns_by_id, roster,
    )
    assert v["assignee"] == "Speaker 3"
    assert v["assigner"] == "Alice Kumar"
    # Corrected owners are never "high"
    assert v["owner_confidence"] == "medium"
    assert v["evidence"]["turn_ids"] == ["T1", "T2"]


def test_accepted_request_alias_is_confirmed_high():
    turns_by_id, roster = _meeting()
    v = validate_action(
        {"task": "Send revised budget", "assignee": "Ravi", "assignee_basis": "accepted_request",
         "evidence_turn_ids": "T1, T2"},
        turns_by_id, roster,
    )
    assert v["assignee"] == "Speaker 3"
    assert v["owner_resolved"] is True
    assert v["owner_confidence"] == "high"


def test_acceptor_of_open_request_is_confirmed():
    turns_by_id, roster = _meeting()
    v = validate_action(
        {"task": "Check the logs", "assignee": "Speaker 3", "assignee_basis": "accepted_request",
         "evidence_turn_ids": ["T4", "T5"]},
        turns_by_id, roster,
    )
    assert v["assignee"] == "Speaker 3"
    assert v["assigner"] == "Alice Kumar"
    assert v["owner_confidence"] == "high"


def test_self_commitment_by_other_speaker_is_corrected_and_uncertain_turn_is_low():
    turns_by_id, roster = _meeting()
    v = validate_action(
        {"task": "Update the test plan", "assignee": "Speaker 1", "assignee_basis": "self_commitment",
         "evidence_turn_ids": ["T3"]},
        turns_by_id, roster,
    )
    assert v["assignee"] == "Speaker 10"
    # T3 has speaker_confidence 0.4 -> deciding turn is uncertain
    assert v["owner_confidence"] == "low"
    # Generic labels are not recorded as a "mentioned as" hint
    assert "assignee_mentioned_as" not in v


def test_missing_or_invalid_evidence_is_low():
    turns_by_id, roster = _meeting()
    missing = validate_action({"task": "x", "assignee": "Speaker 3", "assignee_basis": "self_commitment"}, turns_by_id, roster)
    invalid = validate_action(
        {"task": "x", "assignee": "Speaker 3", "assignee_basis": "self_commitment", "evidence_turn_ids": ["T99"]},
        turns_by_id, roster,
    )
    assert missing["owner_confidence"] == "low" and missing["evidence"] is None
    assert invalid["owner_confidence"] == "low" and invalid["evidence"] is None


def test_external_person_is_kept_unresolved():
    turns_by_id, roster = _meeting()
    v = validate_action(
        {"task": "Prepare the vendor contract", "assignee": "Priya from Legal", "assignee_basis": "named_responsibility",
         "evidence_turn_ids": ["T1"]},
        turns_by_id, roster,
    )
    assert v["assignee"] == "Priya from Legal"
    assert v["owner_resolved"] is False


def test_canonicalize_owner_never_confuses_numbered_labels():
    roster = [{"label": "Speaker 1", "aliases": ["Speaker 1"]}, {"label": "Speaker 10", "aliases": ["Speaker 10"]}]
    assert canonicalize_owner("speaker 10", roster) == ("Speaker 10", True)
    assert canonicalize_owner("Speaker_1", roster) == ("Speaker 1", True)
    assert canonicalize_owner("Speaker 2", roster) == ("Speaker 2", False)


def test_canonicalize_owner_fuzzy_alias_and_vague():
    _, roster = _meeting()
    assert canonicalize_owner("Alise Kumar", roster) == ("Alice Kumar", True)
    assert canonicalize_owner("ravi shah", roster) == ("Speaker 3", True)
    assert canonicalize_owner("we", roster) == (None, False)
    assert canonicalize_owner_list("Alice, Ravi and our team", roster) == [("Alice Kumar", True), ("Speaker 3", True)]


def test_ambiguous_first_name_is_not_resolved():
    roster = build_roster([{"speaker_label": "Alice Kumar"}, {"speaker_label": "Alice Smith"}])
    assert canonicalize_owner("Alice", roster) == ("Alice", False)


def test_validate_point_owners_embedded_mode():
    turns_by_id, roster = _meeting()
    point = {
        "discussion_point": "Alice asked Speaker 3 to send the revised budget.",
        "action_owner": "Alice",
        "owner_evidence": [{"owner": "Alice", "basis": "direct_assignment", "turn_ids": ["T1", "T2"]}],
    }
    validate_point_owners(point, turns_by_id, roster)
    assert point["action_owner"] == "Speaker 3"
    assert point["owner_confidence"] == "medium"
    assert point["owner_evidence"][0]["owner"] == "Speaker 3"


def test_validate_point_owners_without_evidence_canonicalizes_and_marks_low():
    turns_by_id, roster = _meeting()
    point = {"discussion_point": "...", "action_owner": "ravi"}
    validate_point_owners(point, turns_by_id, roster)
    assert point["action_owner"] == "Speaker 3"
    assert point["owner_confidence"] == "low"


def test_evidence_outside_focus():
    assert evidence_outside_focus({"evidence": {"turn_ids": ["T7", "T8"]}}, ["T1", "T2"]) is True
    assert evidence_outside_focus({"evidence": {"turn_ids": ["T2", "T8"]}}, ["T1", "T2"]) is False
    assert evidence_outside_focus({"evidence": None}, ["T1"]) is False


def test_verify_owner_against_candidates():
    _, roster = _meeting()
    excerpt = "[T2 | 00:08-00:09] Speaker 3: Sure, I'll send it."
    assert verify_owner_against_candidates("Ravi Shah", ["Speaker 3"], "", roster) == ("Speaker 3", "high")
    assert verify_owner_against_candidates("Speaker 3", [], excerpt, roster) == ("Speaker 3", "medium")
    assert verify_owner_against_candidates("Dr. Who", ["Speaker 3"], excerpt, roster) == (None, "low")
    # Nothing to verify against: keep the owner, no confidence verdict
    assert verify_owner_against_candidates("Vikas", [], "", None) == ("Vikas", None)


def test_dedupe_point_actions_across_windows():
    points = [
        {"action_items": [{"task": "Send the revised budget to finance", "assignee": "Speaker 3", "deadline": None}]},
        {"action_items": [
            {"task": "Send revised budget to the finance team", "assignee": "Speaker 3", "deadline": "Friday"},
            {"task": "Check the server logs", "assignee": "Speaker 3"},
        ]},
    ]
    removed = dedupe_point_actions(points)
    assert removed == 1
    assert len(points[0]["action_items"]) == 1
    assert points[0]["action_items"][0]["deadline"] == "Friday"
    assert [a["task"] for a in points[1]["action_items"]] == ["Check the server logs"]
