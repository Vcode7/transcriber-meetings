"""End-to-end checks of speaker-precise action ownership through the ROM pipeline
(Stage 1 extraction + validation, Stage 2 carry-over, MoM owner resolution) and
the diarization-layer helpers it relies on."""
from unittest.mock import MagicMock, patch

from services.rom_service import RomService, _carry_stage1_attribution

TRANSCRIPT = [
    {"speaker_label": "Alice Kumar", "start": 0.0, "end": 5.0, "text": "Let's go over the release."},
    {"speaker_label": "Alice Kumar", "start": 5.5, "end": 9.0, "text": "Ravi, can you prepare the release notes by Friday?"},
    {"speaker_label": "Speaker 2", "start": 9.5, "end": 11.0, "text": "Sure, I'll prepare them."},
    {"speaker_label": "Alice Kumar", "start": 12.0, "end": 20.0, "text": "I will book the demo room."},
]
MAPPINGS = {"Speaker 2": "Ravi Shah"}


def _run_stage1(provider, separate):
    with patch("services.ai_provider.get_provider", return_value=provider):
        return RomService().extract_discussion_points(
            TRANSCRIPT, window_minutes=2.0, parallel_window_processing=1,
            separate_action_extraction=separate, speaker_mappings=MAPPINGS,
        )["discussion_points"]


def test_stage1_separate_mode_validates_owners_maps_by_turns_and_dedupes():
    provider = MagicMock()
    provider.extract_rom_discussion_points.return_value = {"discussion_points": [
        {"discussion_point": "Release notes preparation", "speakers": ["Alice Kumar", "Ravi Shah", "Bob"], "turn_ids": ["T1", "T2"]},
        {"discussion_point": "Demo room booking", "speakers": ["Alice Kumar"], "turn_ids": ["T3"]},
    ]}
    provider.extract_rom_action_points.return_value = {"action_items": [
        # Requester wrongly named as owner -> corrected to the addressed person (Ravi = Speaker 2)
        {"task": "Prepare the release notes", "assigner": "Alice", "assignee": "Alice Kumar", "deadline": "Friday",
         "assignee_basis": "direct_assignment", "evidence_turn_ids": ["T1", "T2"]},
        # Self-commitment attributed to the wrong speaker -> corrected to the committer
        {"task": "Book the demo room", "assignee": "Speaker 2", "assignee_basis": "self_commitment", "evidence_turn_ids": ["T3"]},
        # Duplicate of the first item
        {"task": "prepare release notes", "assignee": "Speaker 2", "assignee_basis": "accepted_request", "evidence_turn_ids": ["T1", "T2"]},
    ]}

    points = _run_stage1(provider, separate=True)

    release, demo = points
    assert release["speakers"] == ["Alice Kumar", "Speaker 2"]  # "Bob" was not a participant
    assert len(release["action_items"]) == 1
    assert release["action_items"][0]["assignee"] == "Speaker 2"
    assert release["action_items"][0]["assigner"] == "Alice Kumar"
    assert release["action_items"][0]["evidence"]["turn_ids"] == ["T1", "T2"]
    assert release["action_owner"] == "Speaker 2"
    assert [a["task"] for a in demo["action_items"]] == ["Book the demo room"]
    assert demo["action_owner"] == "Alice Kumar"

    # Prompts receive the participant roster and speaker-turn formatted transcript
    _, d_kwargs = provider.extract_rom_discussion_points.call_args
    assert "MEETING PARTICIPANTS" in d_kwargs["participants"]
    assert "Ravi Shah" in d_kwargs["participants"]
    window_text = provider.extract_rom_discussion_points.call_args[0][0]
    assert window_text.startswith("[T1 | 00:00-00:09] Alice Kumar:")


def test_stage1_embedded_mode_validates_point_owner_against_evidence():
    provider = MagicMock()
    provider.extract_rom_discussion_points.return_value = {"discussion_points": [
        {"discussion_point": "Alice asked Ravi to prepare the release notes by Friday.", "speakers": ["Alice Kumar"],
         "turn_ids": ["T1", "T2"], "action_owner": "Alice Kumar",
         "owner_evidence": [{"owner": "Alice Kumar", "basis": "direct_assignment", "turn_ids": ["T1", "T2"]}]},
    ]}

    points = _run_stage1(provider, separate=False)

    assert points[0]["action_owner"] == "Speaker 2"
    assert points[0]["owner_confidence"] == "medium"
    assert points[0]["action_items"] == []
    provider.extract_rom_action_points.assert_not_called()


def test_stage1_look_ahead_actions_are_extracted_only_by_their_own_window():
    transcript = TRANSCRIPT + [
        {"speaker_label": "Speaker 2", "start": 130.0, "end": 134.0, "text": "I'll also update the changelog."},
    ]
    provider = MagicMock()

    def fake_points(window_text, prev, **kwargs):
        if "T4 |" in window_text:
            return {"discussion_points": [{"discussion_point": "Changelog", "turn_ids": ["T4"]}]}
        return {"discussion_points": [{"discussion_point": "Release", "turn_ids": ["T1", "T2", "T3"]}]}

    changelog = {"task": "Update the changelog", "assignee": "Speaker 2", "assignee_basis": "self_commitment",
                 "evidence_turn_ids": ["T4"]}
    provider.extract_rom_discussion_points.side_effect = fake_points
    provider.extract_rom_action_points.side_effect = lambda *a, **k: {"action_items": [dict(changelog)]}

    with patch("services.ai_provider.get_provider", return_value=provider):
        points = RomService().extract_discussion_points(
            transcript, window_minutes=2.0, parallel_window_processing=1, separate_action_extraction=True,
        )["discussion_points"]

    # Window 1 saw T4 only as look-ahead; the action belongs to window 2 only.
    assert points[0]["action_items"] == []
    assert [a["task"] for a in points[1]["action_items"]] == ["Update the changelog"]
    # Window 1's action call received the look-ahead marker
    first_call_text = provider.extract_rom_action_points.call_args_list[0][0][0]
    assert "LOOK-AHEAD" in first_call_text


def test_stage2_carry_attribution_unions_owners_and_drops_llm_only_names():
    s1 = [
        {"id": "a", "speakers": ["Alice Kumar", "Speaker 2"], "action_owner": "Speaker 2", "owner_confidence": "high",
         "turn_ids": ["T1", "T2"], "action_items": [{"task": "Prepare the release notes", "assignee": "Speaker 2",
                                                     "evidence": {"turn_ids": ["T1", "T2"]}}],
         "raw_transcript_text": "[T1 | 00:00-00:09] Alice Kumar: Ravi, can you prepare the notes?\n[T2 | 00:09-00:11] Speaker 2: Sure, I'll prepare them.\n[T3 | 00:12-00:20] Alice Kumar: I will book the demo room."},
        {"id": "b", "speakers": ["Alice Kumar"], "action_owner": "Alice Kumar", "owner_confidence": "medium",
         "turn_ids": ["T3"], "action_items": [{"task": "Book the demo room", "assignee": "Alice Kumar"}]},
    ]
    ep = {"speakers": ["Alice Kumar", "Bob"], "action_owner": "Bob"}

    _carry_stage1_attribution(ep, s1, allow_llm_actions=True)

    assert ep["speakers"] == ["Alice Kumar"]
    assert ep["action_owner"] == "Speaker 2, Alice Kumar"
    assert ep["owner_confidence"] == "medium"
    assert [a["task"] for a in ep["action_items"]] == ["Prepare the release notes", "Book the demo room"]
    assert ep["stage1_turn_ids"] == ["T1", "T2", "T3"]
    assert ep["owner_transcript_excerpt"].splitlines()[0].startswith("[T1 |")


def test_stage2_action_sources_limit_inherited_actions_but_not_owners():
    s1 = [
        {"id": "a", "speakers": ["Ravi"], "action_owner": "Ravi", "owner_confidence": "high",
         "action_items": [{"task": "Float the tender", "assignee": "Ravi"}]},
        {"id": "b", "speakers": ["Meena"], "action_owner": None, "action_items": []},
    ]
    ep = {"speakers": [], "action_items": [{"task": "LLM-proposed task", "assignee": "Ravi"}]}

    # "a" already gave its action items to another point: none are inherited, and the LLM's
    # own proposal is not used as a substitute.
    _carry_stage1_attribution(ep, s1, allow_llm_actions=True, action_sources=[])

    assert ep["action_items"] == []
    assert ep["action_owner"] == "Ravi"


def test_stage2_llm_invented_owner_is_marked_low():
    ep = {"speakers": [], "action_owner": "Bob"}
    _carry_stage1_attribution(ep, [{"id": "a", "speakers": ["Alice"], "action_owner": None}])
    assert ep["action_owner"] == "Bob"
    assert ep["owner_confidence"] == "low"


def test_mom_owner_never_falls_back_to_all_speakers():
    points = [{
        "id": "p1",
        "polished_text": "The team discussed vendor follow-ups.",
        "speakers": ["Alice", "Bob"],
        "action_items": [{"task": "Call all vendors", "assignee": None}],
    }]
    with patch("services.ai_provider.get_provider", return_value=MagicMock()):
        mom = RomService().generate_mom_from_enhanced_rom(points, {"filename": "m", "speakers_detected": ["Al", "Bob"]})
    assert mom["action_items"][0]["owner"] == "Unassigned"


def test_mom_owner_uses_single_validated_point_owner_and_keeps_confidence():
    points = [{
        "id": "p1", "polished_text": "Ravi will prepare the notes.", "action_owner": "Speaker 2",
        "action_items": [{"task": "Prepare the notes", "assignee": "Speaker 2", "owner_confidence": "low"}],
    }]
    with patch("services.ai_provider.get_provider", return_value=MagicMock()):
        mom = RomService().generate_mom_from_enhanced_rom(points, {"filename": "m", "speakers_detected": ["Speaker 2"]})
    assert mom["action_items"][0]["owner"] == "Speaker 2"
    assert mom["action_items"][0]["owner_confidence"] == "low"


# ── Diarization layer ─────────────────────────────────────────────────────────

def test_diarize_keeps_short_pyannote_turns_but_not_short_energy_segments():
    import services.diarization as diarization

    raw = [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00", "is_overlap": False},
        {"start": 4.1, "end": 4.9, "speaker": "SPEAKER_01", "is_overlap": False},  # "Yes, I'll do it"
        {"start": 5.0, "end": 5.1, "speaker": "SPEAKER_00", "is_overlap": False},  # blip
    ]
    with patch.object(diarization, "is_pyannote_available", return_value=True), \
         patch.object(diarization, "_diarize_pyannote", return_value=list(raw)):
        kept = diarization.diarize("x.wav")
    assert [s["speaker"] for s in kept] == ["SPEAKER_00", "SPEAKER_01"]

    with patch.object(diarization, "is_pyannote_available", return_value=False), \
         patch.object(diarization, "_diarize_energy", return_value=list(raw)):
        kept_energy = diarization.diarize("x.wav")
    assert [s["speaker"] for s in kept_energy] == ["SPEAKER_00"]


def test_generic_speaker_labels_are_per_speaker_not_per_segment():
    from tasks.pipeline import _generic_speaker_labels

    segs = [{"speaker": "SPEAKER_01"}, {"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_01"}]
    labels = [s["speaker_label"] for s in _generic_speaker_labels(segs)]
    assert labels == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_speaker_confidence_is_word_agreement():
    from tasks.pipeline import _speaker_confidence

    seg = {"speaker_label": "Speaker 1", "words": [
        {"word": "can", "speaker_label": "Speaker 1"}, {"word": "you", "speaker_label": "Speaker 1"},
        {"word": "sure", "speaker_label": "Speaker 2"}, {"word": "x"},
    ]}
    assert _speaker_confidence(seg) == round(2 / 3, 3)
    assert _speaker_confidence({"speaker_label": "Speaker 1", "words": []}) is None
