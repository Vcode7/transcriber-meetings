"""
Speaker identification robustness: overlap masking, short-turn concatenation,
one-to-one (Hungarian) assignment, duplicate-cluster merging, profile-guided
cluster splitting and S-norm rescue.

Audio is synthetic: each speaker's regions are filled with a distinct constant
value and the mocked encoder maps that value to a fixed embedding, so the tests
exercise the real identification logic without any model.
"""
from unittest.mock import patch

import numpy as np
import pytest

from config import settings
from services import identification as ident

SR = 16000
DIM = 192


def _unit(*weights):
    v = np.zeros(DIM, dtype=np.float32)
    for i, w in enumerate(weights):
        v[i] = w
    return v / np.linalg.norm(v)


ALICE = _unit(1.0)
BOB = _unit(0.0, 1.0)
CAROL = _unit(0.0, 0.0, 1.0)
# audio fill value → voice embedding
VOICES = {0.1: ALICE, 0.2: BOB, 0.3: CAROL}


def _fake_embed(audio, sr=SR, model=None, **kwargs):
    if len(audio) == 0:
        return None
    value = round(float(np.median(audio)), 2)
    return VOICES.get(value)


def _audio(regions, total_sec=60.0):
    """regions: [(start, end, fill_value)]"""
    audio = np.zeros(int(total_sec * SR), dtype=np.float32)
    for s, e, v in regions:
        audio[int(s * SR):int(e * SR)] = v
    return audio


def _profile(pid, label, emb):
    return {"id": pid, "label": label, "ecapa_embeddings": [emb.tolist()]}


def _identify(audio, segments, profiles, **kwargs):
    with patch.object(ident, "load_audio", return_value=(audio, SR)), \
         patch.object(ident.embedding_router, "vad_extract_speaker_embedding", side_effect=_fake_embed), \
         patch.object(ident.embedding_router, "extract_embedding", side_effect=_fake_embed):
        return ident.identify_speakers(
            file_path="dummy.wav",
            diarization_segments=segments,
            voice_profiles=profiles,
            similarity_threshold=0.72,
            use_model_default_threshold=False,
            embedding_model="ecapa",
            **kwargs,
        )


def _labels_by_speaker(identified):
    return {s["speaker"]: s["speaker_label"] for s in identified}


# ── _clean_speech_spans ──────────────────────────────────────────────────────

def test_clean_spans_removes_overlap_windows():
    seg = {"start": 0.0, "end": 10.0, "overlap_regions": [
        {"start": 4.0, "end": 4.5, "speakers": ["A", "B"]},
        {"start": 9.0, "end": 12.0, "speakers": ["A", "C"]},
    ]}
    assert ident._clean_speech_spans(seg) == [(0.0, 4.0), (4.5, 9.0)]


def test_clean_spans_without_overlap_is_whole_segment():
    assert ident._clean_speech_spans({"start": 1.0, "end": 3.0}) == [(1.0, 3.0)]


# ── Overlap masking ──────────────────────────────────────────────────────────

def _interrupted_meeting():
    # Alice talks 0–20 s but is interrupted twice by Bob → every Alice turn is
    # flagged is_overlap. Bob has his own clean turn 20–26 s.
    audio = _audio([(0, 20, 0.1), (5, 5.5, 0.2), (14, 14.4, 0.2), (20, 26, 0.2)])
    regions = [
        {"start": 5.0, "end": 5.5, "speakers": ["SPEAKER_00", "SPEAKER_01"]},
        {"start": 14.0, "end": 14.4, "speakers": ["SPEAKER_00", "SPEAKER_01"]},
    ]
    segments = [
        {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00", "is_overlap": True, "overlap_regions": [regions[0]]},
        {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_00", "is_overlap": True, "overlap_regions": [regions[1]]},
        {"start": 20.0, "end": 26.0, "speaker": "SPEAKER_01", "is_overlap": False, "overlap_regions": []},
    ]
    return audio, segments


def test_interrupted_speaker_is_identified_with_masking():
    audio, segments = _interrupted_meeting()
    profiles = [_profile("a", "Alice", ALICE), _profile("b", "Bob", BOB)]
    labels = _labels_by_speaker(_identify(audio, segments, profiles))
    assert labels == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}


def test_interrupted_speaker_unidentified_without_masking(monkeypatch):
    # Legacy behaviour (every turn touching an overlap dropped) — the bug being fixed.
    monkeypatch.setattr(settings, "SPEAKER_EMBED_MASK_OVERLAP", False)
    audio, segments = _interrupted_meeting()
    profiles = [_profile("a", "Alice", ALICE), _profile("b", "Bob", BOB)]
    labels = _labels_by_speaker(_identify(audio, segments, profiles))
    assert labels["SPEAKER_00"].startswith("Speaker ")
    assert labels["SPEAKER_01"] == "Bob"


# ── Short-turn concatenation ─────────────────────────────────────────────────

def test_short_turn_speaker_identified_by_concatenation():
    # Bob only ever answers briefly (1.2 s turns, 3.6 s in total).
    audio = _audio([(0, 10, 0.1), (10, 11.2, 0.2), (12, 22, 0.1), (22, 23.2, 0.2), (24, 25.2, 0.2)])
    segments = [
        {"start": 0, "end": 10, "speaker": "SPEAKER_00"},
        {"start": 10, "end": 11.2, "speaker": "SPEAKER_01"},
        {"start": 12, "end": 22, "speaker": "SPEAKER_00"},
        {"start": 22, "end": 23.2, "speaker": "SPEAKER_01"},
        {"start": 24, "end": 25.2, "speaker": "SPEAKER_01"},
    ]
    profiles = [_profile("a", "Alice", ALICE), _profile("b", "Bob", BOB)]
    labels = _labels_by_speaker(_identify(audio, segments, profiles))
    assert labels == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}


def test_too_little_short_speech_stays_generic():
    audio = _audio([(0, 10, 0.1), (10, 11.0, 0.2), (12, 13.0, 0.2)])
    segments = [
        {"start": 0, "end": 10, "speaker": "SPEAKER_00"},
        {"start": 10, "end": 11.0, "speaker": "SPEAKER_01"},
        {"start": 12, "end": 13.0, "speaker": "SPEAKER_01"},
    ]
    profiles = [_profile("a", "Alice", ALICE), _profile("b", "Bob", BOB)]
    labels = _labels_by_speaker(_identify(audio, segments, profiles))
    assert labels["SPEAKER_00"] == "Alice"
    assert labels["SPEAKER_01"].startswith("Speaker ")


# ── One-to-one assignment ────────────────────────────────────────────────────

def test_hungarian_beats_greedy_on_contested_profile():
    scores = np.array([[0.80, 0.79], [0.78, 0.30]])
    accept = scores >= 0.75
    greedy = ident._assign_one_to_one(accept, scores, hungarian=False)
    hungarian = ident._assign_one_to_one(accept, scores, hungarian=True)
    assert greedy == [(0, 0)]                          # speaker 1 left without a profile
    assert sorted(hungarian) == [(0, 1), (1, 0)]       # both speakers matched


def test_assignment_never_uses_rejected_pairs():
    scores = np.array([[0.9, 0.1], [0.2, 0.3]])
    accept = scores >= 0.72
    assert ident._assign_one_to_one(accept, scores, hungarian=True) == [(0, 0)]


# ── Duplicate clusters ───────────────────────────────────────────────────────

def test_over_split_cluster_gets_same_profile_label():
    # pyannote split Alice into two clusters; both clearly sound like Alice.
    audio = _audio([(0, 10, 0.1), (10, 20, 0.2), (20, 30, 0.1)])
    segments = [
        {"start": 0, "end": 10, "speaker": "SPEAKER_00"},
        {"start": 10, "end": 20, "speaker": "SPEAKER_01"},
        {"start": 20, "end": 30, "speaker": "SPEAKER_02"},
    ]
    profiles = [_profile("a", "Alice", ALICE), _profile("b", "Bob", BOB)]
    identified = _identify(audio, segments, profiles)
    labels = _labels_by_speaker(identified)
    assert labels == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob", "SPEAKER_02": "Alice"}
    methods = {s["speaker"]: s["scoring_method"] for s in identified}
    assert "merged_duplicate" in (methods["SPEAKER_00"], methods["SPEAKER_02"])


def test_duplicate_merge_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "SPEAKER_MERGE_DUPLICATE_CLUSTERS", False)
    audio = _audio([(0, 10, 0.1), (20, 30, 0.1)])
    segments = [
        {"start": 0, "end": 10, "speaker": "SPEAKER_00"},
        {"start": 20, "end": 30, "speaker": "SPEAKER_01"},
    ]
    labels = _labels_by_speaker(_identify(audio, segments, [_profile("a", "Alice", ALICE)]))
    assert sorted(labels.values())[0] == "Alice"
    assert any(v.startswith("Speaker ") for v in labels.values())


# ── Profile-guided cluster splitting ─────────────────────────────────────────

def _merged_meeting(bob_turns=2):
    # One pyannote cluster contains Alice (3 turns) and Bob (bob_turns turns).
    regions, segments, t = [], [], 0.0
    for i in range(3 + bob_turns):
        value = 0.1 if i < 3 else 0.2
        regions.append((t, t + 3.0, value))
        segments.append({"start": t, "end": t + 3.0, "speaker": "SPEAKER_00"})
        t += 4.0
    return _audio(regions), segments


def test_merged_participant_is_split_out():
    audio, segments = _merged_meeting(bob_turns=2)
    profiles = [_profile("a", "Alice", ALICE), _profile("b", "Bob", BOB)]
    identified = _identify(audio, segments, profiles, participants_selected=True)
    labels = [s["speaker_label"] for s in identified]
    assert labels == ["Alice", "Alice", "Alice", "Bob", "Bob"]
    bob = [s for s in identified if s["speaker_label"] == "Bob"]
    assert all(s["speaker"].startswith("SPEAKER_SPLIT_") for s in bob)
    assert all(s["speaker_profile_id"] == "b" and s["scoring_method"] == "cluster_split" for s in bob)


def test_split_rolls_back_without_enough_evidence():
    audio, segments = _merged_meeting(bob_turns=1)  # one 3 s turn < 2 turns / 4 s
    profiles = [_profile("a", "Alice", ALICE), _profile("b", "Bob", BOB)]
    identified = _identify(audio, segments, profiles, participants_selected=True)
    assert {s["speaker_label"] for s in identified} == {"Alice"}
    assert {s["speaker"] for s in identified} == {"SPEAKER_00"}


def test_no_split_unless_profiles_are_meeting_participants():
    audio, segments = _merged_meeting(bob_turns=2)
    profiles = [_profile("a", "Alice", ALICE), _profile("b", "Bob", BOB)]
    identified = _identify(audio, segments, profiles, participants_selected=False)
    assert {s["speaker_label"] for s in identified} == {"Alice"}


# ── S-norm rescue ────────────────────────────────────────────────────────────

def test_snorm_rescues_borderline_match(monkeypatch):
    monkeypatch.setattr(settings, "SPEAKER_SCORE_NORMALIZATION", True)
    monkeypatch.setattr(settings, "SPEAKER_SNORM_MIN_COHORT", 2)
    monkeypatch.setattr(settings, "SPEAKER_SNORM_THRESHOLD", 1.0)
    dave = _unit(0.67, 0.0, 0.0, 0.74)  # cos(dave, ALICE) ≈ 0.67: below 0.72, within the 0.10 band
    VOICES[0.4] = dave
    try:
        audio = _audio([(0, 10, 0.4)])
        segments = [{"start": 0, "end": 10, "speaker": "SPEAKER_00"}]
        profiles = [_profile("a", "Alice", ALICE), _profile("b", "Bob", BOB), _profile("c", "Carol", CAROL)]
        assert _labels_by_speaker(_identify(audio, segments, profiles)) == {"SPEAKER_00": "Alice"}
        monkeypatch.setattr(settings, "SPEAKER_SCORE_NORMALIZATION", False)
        assert _labels_by_speaker(_identify(audio, segments, profiles))["SPEAKER_00"].startswith("Speaker ")
    finally:
        VOICES.pop(0.4, None)


@pytest.mark.parametrize("hungarian", [True, False])
def test_identify_works_with_either_assignment(monkeypatch, hungarian):
    monkeypatch.setattr(settings, "SPEAKER_ASSIGNMENT_HUNGARIAN", hungarian)
    audio, segments = _interrupted_meeting()
    profiles = [_profile("a", "Alice", ALICE), _profile("b", "Bob", BOB)]
    assert _labels_by_speaker(_identify(audio, segments, profiles)) == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
