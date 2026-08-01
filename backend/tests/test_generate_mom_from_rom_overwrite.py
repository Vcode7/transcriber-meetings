import pytest
from database import to_json

def test_generate_mom_from_rom_schema_fields():
    polished_points = [
        {
            "polished_text": "The engineering team finalized the API architecture for Q3.",
            "speakers": ["Vikas", "Alice"],
            "action_owner": "Vikas",
            "action_items": ["Implement endpoints by Friday"],
            "dates": ["2026-08-07"]
        }
    ]
    meta = {
        "filename": "Test Meeting.mp3",
        "created_at": "2026-07-31",
        "duration": 300,
        "speakers_detected": ["Vikas", "Alice"]
    }

    from services.rom_service import rom_service
    mom_out = rom_service.generate_mom_from_enhanced_rom(
        polished_points=polished_points,
        recording_meta=meta,
        recording_id="rec-test",
        user_id="user-test"
    )

    assert "title" in mom_out
    assert "introduction" in mom_out
    assert "points_discussed" in mom_out
    assert len(mom_out["points_discussed"]) > 0
    assert len(mom_out["action_items"]) > 0
    assert mom_out["action_items"][0]["owner"] == "Vikas"
    assert mom_out["action_items"][0]["deadline"] == "2026-08-07"
