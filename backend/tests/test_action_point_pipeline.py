import pytest
from services.rom_service import RomService, normalize_action_item, format_action_point_display_text


def test_normalize_action_item():
    # Test structured dict
    item1 = {
        "assigner": "Alice",
        "assignee": "Bob",
        "task": "Prepare project proposal",
        "deadline": "Friday"
    }
    norm1 = normalize_action_item(item1)
    assert norm1 == {
        "assigner": "Alice",
        "assignee": "Bob",
        "task": "Prepare project proposal",
        "deadline": "Friday"
    }

    # Test string input
    item2 = "Review design specs"
    norm2 = normalize_action_item(item2)
    assert norm2 == {
        "assigner": None,
        "assignee": None,
        "task": "Review design specs",
        "deadline": None
    }

    # Test null strings
    item3 = {
        "assigner": "null",
        "assignee": "Unassigned",
        "task": "Fix bug",
        "deadline": "ASAP"
    }
    norm3 = normalize_action_item(item3)
    assert norm3["assigner"] is None
    assert norm3["assignee"] is None
    assert norm3["task"] == "Fix bug"

    # Test vague assignee references
    item4 = {
        "assigner": "Alice",
        "assignee": "our team",
        "task": "Update documentation",
        "deadline": "Friday"
    }
    norm4 = normalize_action_item(item4)
    assert norm4["assignee"] is None



def test_format_action_point_display_text():
    # Test assigner + assignee + task + deadline
    item1 = {
        "assigner": "Alice",
        "assignee": "Bob",
        "task": "prepare budget report",
        "deadline": "Friday"
    }
    text1 = format_action_point_display_text(item1)
    assert text1 == "Alice assigned Bob to complete prepare budget report before Friday"

    # Test assigner only
    item2 = {
        "assigner": "Charlie",
        "assignee": None,
        "task": "update documentation",
        "deadline": "July 10"
    }
    text2 = format_action_point_display_text(item2)
    assert text2 == "Charlie assigned to complete update documentation before July 10"

    # Test task only
    item3 = {
        "assigner": None,
        "assignee": None,
        "task": "Verify deployment status",
        "deadline": None
    }
    text3 = format_action_point_display_text(item3)
    assert text3 == "Verify deployment status"


def test_generate_mom_from_enhanced_rom_action_items():
    rom_service = RomService()
    polished_points = [
        {
            "id": "p1",
            "polished_text": "Discussion on budget",
            "action_items": [
                {
                    "assigner": "Alice",
                    "assignee": "Bob",
                    "task": "Prepare budget report",
                    "deadline": "Friday"
                },
                {
                    "assigner": "Carol",
                    "assignee": None,
                    "task": "Draft RFC document",
                    "deadline": "July 15"
                }
            ]
        }
    ]
    recording_meta = {"filename": "Test Meeting"}

    mom = rom_service.generate_mom_from_enhanced_rom(polished_points, recording_meta)
    actions = mom["action_items"]
    assert len(actions) == 2

    # First action: Bob (Assigned)
    assert actions[0]["owner"] == "Bob"
    assert actions[0]["task"] == "Alice assigned Bob to complete Prepare budget report before Friday"
    assert actions[0]["raw_json"]["assigner"] == "Alice"
    assert actions[0]["raw_json"]["assignee"] == "Bob"

    # Second action: General (assignee is null)
    assert actions[1]["owner"] == "Unassigned"
    assert actions[1]["task"] == "Carol assigned to complete Draft RFC document before July 15"
    assert actions[1]["raw_json"]["assigner"] == "Carol"
    assert actions[1]["raw_json"]["assignee"] is None
