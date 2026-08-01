import pytest
from services.rom_service import apply_speaker_mappings_to_final_rom

def test_apply_speaker_mappings_every_occurrence():
    final_rom = {
        "speaker_mappings": {
            "Speaker_1": "Alice Smith",
            "Speaker_2": "Bob Jones",
        },
        "agendas": [
            {
                "agenda_id": "A1",
                "title": "Discussion on Speaker_1 report",
                "description": "Speaker_1 and Speaker_2 led the session",
                "discussion_points": [
                    {
                        "id": "P1",
                        "speaker": "Speaker_1",
                        "speakers": ["Speaker_1", "Speaker_2"],
                        "action_owner": "Speaker_1",
                        "text": "Speaker_1 presented the quarterly update.",
                        "polished_text": "Speaker_1 presented the quarterly update.",
                        "decisions": ["Speaker_1 approved budget."],
                        "action_items": ["Speaker_2 to complete review by Friday."],
                    },
                    {
                        "id": "P2",
                        "speaker": "Speaker_3",
                        "speakers": ["Speaker_3"],
                        "text": "Speaker_3 asked a question about Speaker_1 proposal.",
                    }
                ]
            }
        ]
    }

    result = apply_speaker_mappings_to_final_rom(final_rom)

    a1 = result["agendas"][0]
    assert a1["title"] == "Discussion on Alice Smith report"
    assert a1["description"] == "Alice Smith and Bob Jones led the session"

    p1 = a1["discussion_points"][0]
    assert p1["speaker"] == "Alice Smith"
    assert p1["speakers"] == ["Alice Smith", "Bob Jones"]
    assert p1["action_owner"] == "Alice Smith"
    assert p1["text"] == "Alice Smith presented the quarterly update."
    assert p1["decisions"] == ["Alice Smith approved budget."]
    assert p1["action_items"] == ["Bob Jones to complete review by Friday."]

    p2 = a1["discussion_points"][1]
    # Speaker_3 is unmapped -> remains unchanged
    assert p2["speaker"] == "Speaker_3"
    assert p2["speakers"] == ["Speaker_3"]
    assert p2["text"] == "Speaker_3 asked a question about Alice Smith proposal."


def test_apply_speaker_mappings_prefix_prevention():
    final_rom = {
        "speaker_mappings": {
            "Speaker_1": "Alice Smith",
            "Speaker_10": "Charlie Brown",
        },
        "agendas": [
            {
                "agenda_id": "A1",
                "discussion_points": [
                    {
                        "id": "P1",
                        "speaker": "Speaker_10",
                        "text": "Speaker_1 and Speaker_10 both joined.",
                    }
                ]
            }
        ]
    }

    result = apply_speaker_mappings_to_final_rom(final_rom)
    p1 = result["agendas"][0]["discussion_points"][0]
    assert p1["speaker"] == "Charlie Brown"
    assert p1["text"] == "Alice Smith and Charlie Brown both joined."
