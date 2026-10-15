"""The action flag keeps Stage 1 evidence of an action (a validated owner, as in the default
embedded Stage 1 mode where points carry no action items), and the fallback used when the
action-point LLM call fails builds items from the owner's own sentence."""
from services.agenda_action_points import fallback_items_from_entries, validate_action_points
from services.point_classification import classify_stage2_point


def test_validated_owner_sets_the_flag_when_the_llm_says_no_action():
    source = {"id": "s1", "action_owner": "Ravi", "owner_confidence": "medium", "action_items": [],
              "owner_evidence": [{"owner": "Ravi", "basis": "self_commitment", "turn_ids": ["T4"]}]}
    ep = classify_stage2_point({"polished_text": "Ravi will float the tender by Friday.", "include_action_point": False}, [source])
    assert ep["include_action_point"] is True and ep["action_flag_source"] == "stage1"


def test_low_confidence_owner_without_evidence_does_not_set_the_flag():
    source = {"id": "s1", "action_owner": "Ravi", "owner_confidence": "low", "action_items": []}
    ep = classify_stage2_point({"polished_text": "Ravi mentioned the tender.", "include_action_point": False}, [source])
    assert ep["include_action_point"] is False


def test_a_flag_the_user_switched_off_stays_off():
    source = {"id": "s1", "owner_evidence": [{"owner": "Ravi"}], "action_flag_source": "user"}
    ep = classify_stage2_point({"polished_text": "x", "include_action_point": False, "action_flag_source": "user"}, [source])
    assert ep["include_action_point"] is False


def test_fallback_uses_the_owners_action_sentence_when_there_are_no_action_items():
    entry = {
        "id": "p1",
        "text": "The hostel renovation budget was reviewed. Ravi will send the revised PWD quotation to the committee.",
        "action_owner": "Ravi", "owner_candidates": ["Ravi"],
    }
    items = fallback_items_from_entries([entry], ["p1"])
    assert [(i["task"], i["owner"]) for i in items] == [
        ("Ravi will send the revised PWD quotation to the committee.", "Ravi")]

    kept = validate_action_points(items, [entry], target_ids=["p1"], dedupe=False)["kept"]
    assert kept and kept[0]["assignee"] == "Ravi"


def test_fallback_ignores_points_that_are_not_targets_or_have_no_owner_sentence():
    entry = {"id": "p1", "text": "Ravi presented the status update.", "action_owner": "Ravi", "owner_candidates": ["Ravi"]}
    assert fallback_items_from_entries([entry], ["p1"]) == []
    assert fallback_items_from_entries([{**entry, "text": "Ravi will send the quotation."}], ["other"]) == []
