import pytest
from services.speaker_sync import (
    apply_speaker_mappings_to_recording_dict,
    apply_speaker_mappings_to_mom_dict,
    build_clean_speaker_mappings,
)

def test_build_clean_speaker_mappings_exact_only():
    """Verify build_clean_speaker_mappings returns exact mappings without variant/alias expansion."""
    raw = {"Speaker 1": "Vikas", "Speaker 2": "Alice"}
    clean = build_clean_speaker_mappings(raw)
    assert clean == {"Speaker 1": "Vikas", "Speaker 2": "Alice"}
    assert "Speaker 01" not in clean
    assert "Speaker 02" not in clean

def test_apply_speaker_mappings_recording_deep():
    rec = {
        "id": "rec-123",
        "transcript": [
            {
                "speaker_label": "Speaker 1",
                "text": "Hello, I am Speaker 1.",
                "words": [{"text": "Hello", "speaker_label": "Speaker 1"}]
            },
            {
                "speaker_label": "Speaker 10",
                "text": "Speaker 10 here.",
                "words": [{"text": "Speaker 10", "speaker_label": "Speaker 10"}]
            }
        ],
        "speakers_detected": ["Speaker 1", "Speaker 10"],
        "speaker_summary": {
            "Speaker 1": {
                "summary": "Speaker 1 discussed budget.",
                "action_items": ["Speaker 1 to complete task"]
            }
        },
        "summary": "Speaker 1 presented the project status.",
        "key_points": ["Speaker 1 highlighted key metrics."],
        "action_items": [{"task": "Review doc", "owner": "Speaker 1"}],
        "rom_data": {
            "stage1": {
                "discussion_points": [
                    {"speaker": "Speaker 1", "point": "Point by Speaker 1"}
                ]
            },
            "stage2": {
                "polished_points": [
                    {"speaker": "Speaker 1", "action_owner": "Speaker 1", "speakers": ["Speaker 1"]}
                ]
            },
            "stage3": {
                "agendas": [
                    {"speaker": "Speaker 1", "presenter": "Speaker 1", "title": "Budget"}
                ]
            },
            "final_rom": {
                "agendas": [
                    {
                        "title": "Budget",
                        "discussion_points": [
                            {"speaker": "Speaker 1", "action_owner": "Speaker 1", "point": "Approved by Speaker 1"}
                        ]
                    }
                ]
            }
        }
    }

    mappings = {"Speaker 1": "Vikas"}
    updated = apply_speaker_mappings_to_recording_dict(rec, mappings)

    # 1. Transcript segment & word tokens
    assert updated["transcript"][0]["speaker_label"] == "Vikas"
    assert updated["transcript"][0]["words"][0]["speaker_label"] == "Vikas"

    # Prefix prevention: Speaker 10 must NOT be renamed to Vikas0
    assert updated["transcript"][1]["speaker_label"] == "Speaker 10"

    # 2. Speakers detected
    assert "Vikas" in updated["speakers_detected"]
    assert "Speaker 10" in updated["speakers_detected"]
    assert "Speaker 1" not in updated["speakers_detected"]

    # 3. Speaker summary
    assert "Vikas" in updated["speaker_summary"]
    assert "Speaker 1" not in updated["speaker_summary"]

    # 4. Summaries and action items
    assert "Vikas" in updated["summary"]
    assert updated["action_items"][0]["owner"] == "Vikas"

    # 5. ROM Data (Stage 1, 2, 3, Final ROM)
    stage1_pt = updated["rom_data"]["stage1"]["discussion_points"][0]
    assert stage1_pt["speaker"] == "Vikas"

    stage2_pt = updated["rom_data"]["stage2"]["polished_points"][0]
    assert stage2_pt["speaker"] == "Vikas"
    assert stage2_pt["action_owner"] == "Vikas"
    assert stage2_pt["speakers"] == ["Vikas"]

    stage3_ag = updated["rom_data"]["stage3"]["agendas"][0]
    assert stage3_ag["speaker"] == "Vikas"
    assert stage3_ag["presenter"] == "Vikas"

    final_dp = updated["rom_data"]["final_rom"]["agendas"][0]["discussion_points"][0]
    assert final_dp["speaker"] == "Vikas"
    assert final_dp["action_owner"] == "Vikas"

def test_apply_speaker_mappings_mom_deep():
    mom = {
        "participants": ["Speaker 1", "Speaker 2"],
        "action_items": [
            {"task": "Prepare report", "owner": "Speaker 1"},
            "Speaker 1 will present tomorrow"
        ],
        "points_discussed": ["Speaker 1 introduced the topic."],
        "introduction": "Welcome by Speaker 1."
    }

    mappings = {"Speaker 1": "Vikas"}
    updated = apply_speaker_mappings_to_mom_dict(mom, mappings)

    assert "Vikas" in updated["participants"]
    assert "Speaker 1" not in updated["participants"]
    assert updated["action_items"][0]["owner"] == "Vikas"
    assert "Vikas" in updated["points_discussed"][0]
    assert "Vikas" in updated["introduction"]


def test_raw_diarization_id_propagation_to_trained_name():
    """
    Validation test:
    SPEAKER_03 -> Speaker 1 -> Rod Domowski
    Ensures SPEAKER_03 and Speaker 1 are both updated to Rod Domowski across all ROM stages,
    Raw MoM, transcript, speaker summaries, participants, and action items.
    """
    rec = {
        "id": "rec-validation",
        "transcript": [
            {
                "speaker": "SPEAKER_03",
                "speaker_label": "Speaker 1",
                "text": "SPEAKER_03 speech segment.",
                "words": [{"word": "speech", "speaker": "SPEAKER_03", "speaker_label": "Speaker 1"}]
            }
        ],
        "speakers_detected": ["Speaker 1", "SPEAKER_03"],
        "speaker_summary": {
            "Speaker 1": {"summary": "SPEAKER_03 discussed project timeline."}
        },
        "rom_data": {
            "stage1": {
                "discussion_points": [
                    {"speaker": "SPEAKER_03", "speakers": ["SPEAKER_03"], "text": "SPEAKER_03 outlined milestone 1."}
                ]
            },
            "stage2": {
                "polished_points": [
                    {"speaker": "Speaker 1", "action_owner": "SPEAKER_03", "text": "Report by SPEAKER_03."}
                ]
            },
            "stage3": {
                "agendas": [
                    {"speaker": "SPEAKER_03", "presenter": "Speaker 1", "title": "Milestone 1"}
                ]
            },
            "final_rom": {
                "agendas": [
                    {
                        "title": "Milestone 1",
                        "presenter": "SPEAKER_03",
                        "discussion_points": [
                            {"speaker": "SPEAKER_03", "action_owner": "Speaker 1", "text": "Delivered by SPEAKER_03."}
                        ],
                        "action_items": [
                            {"owner": "SPEAKER_03", "task": "Follow up with team"}
                        ]
                    }
                ],
                "participants": ["Speaker 1", "SPEAKER_03"]
            }
        }
    }

    new_mappings = {"Speaker 1": "Rod Domowski"}
    updated = apply_speaker_mappings_to_recording_dict(rec, new_mappings)

    # 1. Transcript
    assert updated["transcript"][0]["speaker_label"] == "Rod Domowski"
    assert updated["transcript"][0]["words"][0]["speaker_label"] == "Rod Domowski"

    # 2. Speakers detected
    assert updated["speakers_detected"] == ["Rod Domowski"]

    # 3. Speaker summary
    assert "Rod Domowski" in updated["speaker_summary"]
    assert "SPEAKER_03" not in updated["speaker_summary"]

    # 4. Stage 1
    s1 = updated["rom_data"]["stage1"]["discussion_points"][0]
    assert s1["speaker"] == "Rod Domowski"
    assert s1["speakers"] == ["Rod Domowski"]
    assert "Rod Domowski" in s1["text"]

    # 5. Stage 2
    s2 = updated["rom_data"]["stage2"]["polished_points"][0]
    assert s2["speaker"] == "Rod Domowski"
    assert s2["action_owner"] == "Rod Domowski"

    # 6. Stage 3
    s3 = updated["rom_data"]["stage3"]["agendas"][0]
    assert s3["speaker"] == "Rod Domowski"
    assert s3["presenter"] == "Rod Domowski"

    # 7. Final ROM
    final_agenda = updated["rom_data"]["final_rom"]["agendas"][0]
    assert final_agenda["presenter"] == "Rod Domowski"
    assert final_agenda["discussion_points"][0]["speaker"] == "Rod Domowski"
    assert final_agenda["discussion_points"][0]["action_owner"] == "Rod Domowski"
    assert final_agenda["action_items"][0]["owner"] == "Rod Domowski"
    assert updated["rom_data"]["final_rom"]["participants"] == ["Rod Domowski"]
def test_single_speaker_rename_no_cross_contamination():
    """
    Ensures that mapping 'Speaker 3' -> 'gam' ONLY renames Speaker 3 / SPEAKER_03,
    and NEVER renames Speaker 1, Speaker 2, or Speaker 4.
    """
    rec = {
        "id": "rec-multi-spk",
        "transcript": [
            {"speaker": "SPEAKER_00", "speaker_label": "Speaker 1", "text": "Speech 1"},
            {"speaker": "SPEAKER_01", "speaker_label": "Speaker 2", "text": "Speech 2"},
            {"speaker": "SPEAKER_02", "speaker_label": "Speaker 3", "text": "Speech 3"},
            {"speaker": "SPEAKER_03", "speaker_label": "Speaker 4", "text": "Speech 4"},
        ],
        "speakers_detected": ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"],
        "speaker_summary": {
            "Speaker 1": {"summary": "Overview 1"},
            "Speaker 2": {"summary": "Overview 2"},
            "Speaker 3": {"summary": "Overview 3"},
            "Speaker 4": {"summary": "Overview 4"},
        }
    }

    new_mappings = {"Speaker 3": "gam"}
    updated = apply_speaker_mappings_to_recording_dict(rec, new_mappings)

    # Check transcript
    assert updated["transcript"][0]["speaker_label"] == "Speaker 1"
    assert updated["transcript"][1]["speaker_label"] == "Speaker 2"
    assert updated["transcript"][2]["speaker_label"] == "gam"
    assert updated["transcript"][3]["speaker_label"] == "Speaker 4"

    # Check speakers detected
    assert "gam" in updated["speakers_detected"]
    assert "Speaker 1" in updated["speakers_detected"]
    assert "Speaker 2" in updated["speakers_detected"]
    assert "Speaker 4" in updated["speakers_detected"]
    assert "Speaker 3" not in updated["speakers_detected"]

    # Check speaker summary
    assert "gam" in updated["speaker_summary"]
    assert "Speaker 1" in updated["speaker_summary"]
    assert "Speaker 2" in updated["speaker_summary"]
    assert "Speaker 4" in updated["speaker_summary"]
    assert "Speaker 3" not in updated["speaker_summary"]


def test_stale_db_mappings_no_cross_contamination():
    """
    Regression test for the bug reported in logs:
      Selected speaker: Speaker 13
      BUT ALSO renamed: SPEAKER_12, SPEAKER_03, SPEAKER_09 (all wrong!)

    Root cause: sync_global_speaker_rename was merging stale expanded aliases
    from the DB (speaker_mappings) back into the new rename operation, causing
    all those stale aliases to be re-applied.

    This test verifies that resolve_recording_speaker_mappings ONLY resolves
    aliases that actually belong to the selected speaker (Speaker 13 / SPEAKER_00
    in the transcript), and does NOT rename Speaker 12, Speaker 9, etc.
    """
    # Simulate a real recording with 5 speakers
    rec = {
        "id": "rec-stale-maps",
        "transcript": [
            {"speaker": "SPEAKER_00", "speaker_label": "Speaker 13", "text": "I am Speaker 13"},
            {"speaker": "SPEAKER_01", "speaker_label": "Speaker 12", "text": "I am Speaker 12"},
            {"speaker": "SPEAKER_02", "speaker_label": "Speaker 9", "text": "I am Speaker 9"},
            {"speaker": "SPEAKER_03", "speaker_label": "Speaker 5", "text": "I am Speaker 5"},
            {"speaker": "SPEAKER_04", "speaker_label": "Speaker 3", "text": "I am Speaker 3"},
        ],
        "speakers_detected": ["Speaker 13", "Speaker 12", "Speaker 9", "Speaker 5", "Speaker 3"],
        # Simulate stale expanded speaker_mappings in DB from a previous (bad) training run
        "speaker_mappings": {
            "SPEAKER_12": "gam",
            "SPEAKER_03": "gam",
            "SPEAKER_09": "gam",
            "Speaker 3": "gam",
        },
        "speaker_summary": {
            "Speaker 13": {"summary": "Budget overview"},
            "Speaker 12": {"summary": "Tech updates"},
            "Speaker 9": {"summary": "Marketing report"},
            "Speaker 5": {"summary": "HR update"},
            "Speaker 3": {"summary": "Operations"},
        }
    }

    # User trains: Speaker 13 -> John
    new_mappings = {"Speaker 13": "John"}

    # resolve_recording_speaker_mappings must ONLY resolve aliases for Speaker 13
    from services.speaker_sync import resolve_recording_speaker_mappings
    resolved = resolve_recording_speaker_mappings(rec, new_mappings)

    # All resolved keys must map to "John" (the selected speaker)
    assert all(v == "John" for v in resolved.values()), \
        f"Expected all values to be 'John', got: {set(resolved.values())}"

    # Speaker 13 and its raw diarization ID must be renamed
    assert "Speaker 13" in resolved
    assert resolved["Speaker 13"] == "John"
    # SPEAKER_00 is Speaker 13's raw diarization ID (from transcript)
    assert "SPEAKER_00" in resolved or "Speaker 13" in resolved

    # No other speakers should be in the mapping
    for bad_key in ["Speaker 12", "SPEAKER_01", "Speaker 9", "SPEAKER_02",
                    "Speaker 5", "SPEAKER_03", "Speaker 3", "SPEAKER_04",
                    # Stale DB entries must NOT be in resolved
                    "SPEAKER_12", "SPEAKER_09"]:
        assert bad_key not in resolved, \
            f"Unexpected key '{bad_key}' found in resolved mappings: {resolved}"

    # Now apply to recording and verify only Speaker 13 is renamed
    from services.speaker_sync import apply_speaker_mappings_to_recording_dict
    updated = apply_speaker_mappings_to_recording_dict(rec, new_mappings)

    assert updated["transcript"][0]["speaker_label"] == "John"     # Speaker 13
    assert updated["transcript"][1]["speaker_label"] == "Speaker 12"  # untouched
    assert updated["transcript"][2]["speaker_label"] == "Speaker 9"   # untouched
    assert updated["transcript"][3]["speaker_label"] == "Speaker 5"   # untouched
    assert updated["transcript"][4]["speaker_label"] == "Speaker 3"   # untouched

    assert "John" in updated["speakers_detected"]
    assert "Speaker 12" in updated["speakers_detected"]
    assert "Speaker 9" in updated["speakers_detected"]
    assert "Speaker 5" in updated["speakers_detected"]
    assert "Speaker 3" in updated["speakers_detected"]
    assert "Speaker 13" not in updated["speakers_detected"]


def test_chain_reaction_through_shared_raw_id():
    """
    Regression test for the exact scenario from the user logs:

    Resolved aliases included:
      SPEAKER_03, SPEAKER_07, SPEAKER_09, jjjno, SPEAKER_11, SPEAKER_06,
      SPEAKER_08, SPEAKER_12, SPEAKER_05 — all mapped to 'vv'

    Root cause: Speaker 4 had multiple raw IDs (SPEAKER_03, SPEAKER_07, etc.)
    because voice re-identification assigned the same display label to multiple
    diarization chunks. When SPEAKER_03 was added to the mapping (because it
    appeared with Speaker 4 in some segments), the two-way lookup then found
    SPEAKER_03 also appeared with 'jjjno' in other segments, and transitively
    added jjjno → vv. This cascaded to all other speakers sharing those raw IDs.

    Fix: ONE-DIRECTION lookup only. We add raw_id when label is mapped,
    but NEVER go raw_id → other labels.
    """
    from services.speaker_sync import resolve_recording_speaker_mappings, apply_speaker_mappings_to_recording_dict

    rec = {
        "id": "rec-chain-reaction",
        "transcript": [
            # Speaker 4 appears with multiple raw diarization IDs (real re-ID scenario)
            {"speaker": "SPEAKER_04", "speaker_label": "Speaker 4", "text": "Main budget speech"},
            {"speaker": "SPEAKER_03", "speaker_label": "Speaker 4", "text": "Also assigned Speaker 4"},
            {"speaker": "SPEAKER_07", "speaker_label": "Speaker 4", "text": "Also assigned Speaker 4"},
            # jjjno appears with SPEAKER_03 — same raw ID shared with Speaker 4!
            # This was the chain: SPEAKER_03 in clean → jjjno gets added
            {"speaker": "SPEAKER_03", "speaker_label": "jjjno", "text": "Previously trained person"},
            # Other speakers — must NOT be touched
            {"speaker": "SPEAKER_05", "speaker_label": "Speaker 2", "text": "Different speaker"},
            {"speaker": "SPEAKER_09", "speaker_label": "Speaker 6", "text": "Different speaker"},
        ],
        "speakers_detected": ["Speaker 4", "jjjno", "Speaker 2", "Speaker 6"],
        "speaker_summary": {
            "Speaker 4": {"summary": "Budget discussion"},
            "jjjno": {"summary": "Previous trained speaker"},
            "Speaker 2": {"summary": "Tech updates"},
            "Speaker 6": {"summary": "Marketing"},
        }
    }

    # User trains: Speaker 4 -> vv
    new_mappings = {"Speaker 4": "vv"}

    # Verify resolved mapping contains ONLY Speaker 4 aliases
    resolved = resolve_recording_speaker_mappings(rec, new_mappings)

    # All values must be "vv"
    assert all(v == "vv" for v in resolved.values()), \
        f"Found unexpected target values: {set(resolved.values())}"

    # jjjno must NEVER appear in the mapping
    assert "jjjno" not in resolved, \
        f"'jjjno' was incorrectly added to the mapping: {resolved}"

    # Other speaker raw IDs must NOT appear
    for bad_key in ["SPEAKER_05", "SPEAKER_09", "Speaker 2", "Speaker 6"]:
        assert bad_key not in resolved, \
            f"'{bad_key}' was incorrectly added to the mapping: {resolved}"

    # Apply and verify only Speaker 4 segments changed
    updated = apply_speaker_mappings_to_recording_dict(rec, new_mappings)

    assert updated["transcript"][0]["speaker_label"] == "vv"    # Speaker 4
    assert updated["transcript"][1]["speaker_label"] == "vv"    # Speaker 4 (SPEAKER_03 segment)
    assert updated["transcript"][2]["speaker_label"] == "vv"    # Speaker 4 (SPEAKER_07 segment)
    assert updated["transcript"][3]["speaker_label"] == "jjjno" # MUST stay jjjno!
    assert updated["transcript"][4]["speaker_label"] == "Speaker 2"  # untouched
    assert updated["transcript"][5]["speaker_label"] == "Speaker 6"  # untouched

    assert "vv" in updated["speakers_detected"]
    assert "jjjno" in updated["speakers_detected"]   # must stay
    assert "Speaker 2" in updated["speakers_detected"]
    assert "Speaker 6" in updated["speakers_detected"]
    assert "Speaker 4" not in updated["speakers_detected"]


def test_collect_raw_ids_duration_filtering():
    """
    Verify that collect_raw_ids_for_label filters out minor ECAPA refinement overrides
    (e.g., a 1-second SPEAKER_04 segment) and returns only primary raw IDs (>= 20% duration).
    """
    from services.speaker_sync import collect_raw_ids_for_label

    transcript = [
        # Primary speaker for Speaker 2: SPEAKER_02 (100 seconds)
        {"speaker": "SPEAKER_02", "speaker_label": "Speaker 2", "start": 0.0, "end": 50.0},
        {"speaker": "SPEAKER_02", "speaker_label": "Speaker 2", "start": 50.0, "end": 100.0},
        # Minor ECAPA override: SPEAKER_04 (1.5 seconds)
        {"speaker": "SPEAKER_04", "speaker_label": "Speaker 2", "start": 100.0, "end": 101.5},
    ]

    raw_ids = collect_raw_ids_for_label(transcript, "Speaker 2")
    assert raw_ids == {"SPEAKER_02"}
    assert "SPEAKER_04" not in raw_ids


def test_raw_id_format_variation_matching():
    """
    Verify that Speaker_01, speaker_01, and SPEAKER_01 in Stage 1 ROM data are all
    matched and updated when raw_id_map contains SPEAKER_01 -> yoyoyo.
    """
    rec = {
        "id": "rec-format-test",
        "transcript": [
            {"speaker": "SPEAKER_01", "speaker_label": "Speaker 1", "start": 0.0, "end": 10.0}
        ],
        "rom_data": {
            "stage1": {
                "discussion_points": [
                    {
                        "speaker": "Speaker_01",
                        "speakers": ["speaker_01"],
                        "action_owner": "Speaker_01",
                        "text": "Report by Speaker_01."
                    }
                ]
            }
        }
    }

    new_mappings = {"Speaker 1": "yoyoyo"}
    updated = apply_speaker_mappings_to_recording_dict(rec, new_mappings)

    dp = updated["rom_data"]["stage1"]["discussion_points"][0]
    assert dp["speaker"] == "yoyoyo"
    assert dp["speakers"] == ["yoyoyo"]
    assert dp["action_owner"] == "yoyoyo"
    assert "yoyoyo" in dp["text"]


def test_generate_advanced_mom_selective_regeneration(monkeypatch):
    """
    Verify that generate_advanced_mom refines action points and only regenerates Title/Intro/Conclusion
    if their respective flags are set to True.
    """
    import json
    from services.rom_service import rom_service

    final_rom = {
        "agendas": [
            {
                "agenda_id": "a1",
                "title": "Q3 Planning",
                "discussion_points": [
                    {
                        "point_id": "p1",
                        "speaker": "Speaker 1",
                        "text": "Discussed roadmap and budget allocations.",
                        "action_items": ["Prepare budget proposal"]
                    }
                ]
            }
        ]
    }

    existing_mom = {
        "title": "Original Title",
        "introduction": "Original Intro",
        "conclusion": "Original Conclusion",
        "action_items": [{"task": "Existing Task", "owner": "Speaker 1", "deadline": "ASAP"}]
    }

    # Mock provider to return structured json response
    class DummyProvider:
        def query(self, prompt, **kwargs):
            return json.dumps({
                "title": "New Refined Title",
                "introduction": "New Refined Intro",
                "conclusion": "New Refined Conclusion",
                "action_items": [
                    {"task": "Prepare budget proposal", "owner": "Speaker 1", "deadline": "Friday"}
                ],
                "agendas": final_rom["agendas"]
            })

        def unload_model(self):
            pass

    monkeypatch.setattr("services.ai_provider.get_provider", lambda: DummyProvider())

    # Test case 1: Selective flags False (keep original title/intro/conclusion)
    res_no_regen = rom_service.generate_advanced_mom(
        final_rom=final_rom,
        existing_mom=existing_mom,
        custom_prompt="Polish discussion points for executive review.",
        regenerate_title=False,
        regenerate_intro=False,
        regenerate_conclusion=False,
    )
    assert res_no_regen["title"] == "Original Title"
    assert res_no_regen["introduction"] == "Original Intro"
    assert res_no_regen["conclusion"] == "Original Conclusion"
    assert len(res_no_regen["action_items"]) == 1

    # Test case 2: Selective flags True (update title/intro/conclusion)
    res_regen_all = rom_service.generate_advanced_mom(
        final_rom=final_rom,
        existing_mom=existing_mom,
        custom_prompt="Polish discussion points for executive review.",
        regenerate_title=True,
        regenerate_intro=True,
        regenerate_conclusion=True,
    )
    assert res_regen_all["title"] == "New Refined Title"
    assert res_regen_all["introduction"] == "New Refined Intro"
    assert res_regen_all["conclusion"] == "New Refined Conclusion"


def test_sequential_multi_speaker_training():
    """
    Verify that training multiple voice profiles sequentially (Speaker 1 -> Alice, then Speaker 2 -> Bob)
    accumulates and preserves all mappings across transcript, speakers_detected, speaker_mappings,
    and rom_data.final_rom.speaker_mappings.
    """
    rec = {
        "id": "rec-multi-speaker",
        "transcript": [
            {"speaker": "SPEAKER_00", "speaker_label": "Speaker 1", "text": "Hello from speaker 1", "start": 0.0, "end": 10.0},
            {"speaker": "SPEAKER_01", "speaker_label": "Speaker 2", "text": "Hello from speaker 2", "start": 10.0, "end": 20.0},
        ],
        "speakers_detected": ["Speaker 1", "Speaker 2"],
        "speaker_mappings": {},
        "rom_data": {
            "final_rom": {
                "participants": ["Speaker 1", "Speaker 2"],
                "agendas": [
                    {
                        "title": "Agenda 1",
                        "discussion_points": [
                            {"speaker": "SPEAKER_00", "action_owner": "SPEAKER_00", "text": "Point 1"},
                            {"speaker": "SPEAKER_01", "action_owner": "SPEAKER_01", "text": "Point 2"}
                        ]
                    }
                ],
                "speaker_mappings": {}
            }
        }
    }

    # Step 1: Train Speaker 1 -> Alice
    from services.speaker_sync import apply_speaker_mappings_to_recording_dict
    step1 = apply_speaker_mappings_to_recording_dict(rec, {"Speaker 1": "Alice"})

    assert step1["transcript"][0]["speaker_label"] == "Alice"
    assert step1["transcript"][1]["speaker_label"] == "Speaker 2"
    assert "Alice" in step1["speakers_detected"]
    assert step1["speaker_mappings"] == {"Speaker 1": "Alice"}
    assert step1["rom_data"]["final_rom"]["speaker_mappings"] == {"Speaker 1": "Alice"}
    assert step1["rom_data"]["final_rom"]["agendas"][0]["discussion_points"][0]["speaker"] == "Alice"

    # Step 2: Train Speaker 2 -> Bob (passing only {"Speaker 2": "Bob"} incrementally)
    step2 = apply_speaker_mappings_to_recording_dict(step1, {"Speaker 2": "Bob"})

    assert step2["transcript"][0]["speaker_label"] == "Alice"
    assert step2["transcript"][1]["speaker_label"] == "Bob"
    assert "Alice" in step2["speakers_detected"]
    assert "Bob" in step2["speakers_detected"]
    assert step2["speaker_mappings"] == {"Speaker 1": "Alice", "Speaker 2": "Bob"}
    assert step2["rom_data"]["final_rom"]["speaker_mappings"] == {"Speaker 1": "Alice", "Speaker 2": "Bob"}
    assert step2["rom_data"]["final_rom"]["agendas"][0]["discussion_points"][0]["speaker"] == "Alice"
    assert step2["rom_data"]["final_rom"]["agendas"][0]["discussion_points"][1]["speaker"] == "Bob"





