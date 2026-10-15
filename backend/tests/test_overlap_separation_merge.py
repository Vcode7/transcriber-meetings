"""Overlap separation: window selection, source matching, merge and graceful degradation."""
from unittest.mock import patch

import numpy as np

from services import overlap_separation as sep


def _seg(start, end, speaker, regions):
    return {"start": start, "end": end, "speaker": speaker, "is_overlap": bool(regions), "overlap_regions": regions}


def test_select_windows_pads_merges_and_filters():
    short = {"start": 2.0, "end": 2.3, "speakers": ["A", "B"]}      # < 0.5 s → dropped
    w1 = {"start": 10.0, "end": 11.0, "speakers": ["A", "B"]}
    w2 = {"start": 11.4, "end": 12.0, "speakers": ["B", "C"]}       # touches w1 after padding
    segs = [_seg(0, 20, "A", [short, w1]), _seg(9, 13, "B", [w1, w2])]
    windows = sep.select_overlap_windows(segs, duration=30.0, min_sec=0.5, pad_sec=0.3, max_windows=10)
    assert len(windows) == 1
    w = windows[0]
    assert (round(w["start"], 2), round(w["end"], 2)) == (9.7, 12.3)
    assert (w["core_start"], w["core_end"]) == (10.0, 12.0)
    assert w["speakers"] == ["A", "B", "C"]


def test_select_windows_caps_count_keeping_longest():
    regions = [{"start": float(t), "end": t + 0.6 + 0.1 * (t % 3), "speakers": ["A", "B"]} for t in range(0, 50, 5)]
    windows = sep.select_overlap_windows([_seg(0, 60, "A", regions)], duration=60.0,
                                         min_sec=0.5, pad_sec=0.0, max_windows=3)
    assert len(windows) == 3
    assert windows == sorted(windows, key=lambda w: w["start"])


def test_match_sources_is_one_to_one():
    a, b = np.eye(3, dtype=np.float32)[0], np.eye(3, dtype=np.float32)[1]
    refs = {"A": a, "B": b}
    with patch("services.embedding.cosine_similarity", side_effect=lambda x, y: float(np.dot(x, y))):
        matches = sep._match_sources([b * 0.9 + a * 0.1, a], ["A", "B"], refs)
    assert {k: v[0] for k, v in matches.items()} == {0: "B", 1: "A"}


def _turn(start, end, label, raw, words):
    return {"start": start, "end": end, "speaker_label": label, "speaker": raw, "speaker_profile_id": None,
            "words": [{"word": w, "start": s, "end": e, "speaker_label": label} for w, s, e in words]}


def test_merge_adds_missing_words_and_skips_duplicates():
    main = [_turn(0.0, 3.0, "Vikas", "SPEAKER_00", [("the", 0.0, 0.3), ("budget", 0.3, 1.0), ("is", 1.1, 1.3)])]
    identified = [{"speaker": "SPEAKER_00", "speaker_label": "Vikas"},
                  {"speaker": "SPEAKER_01", "speaker_label": "Rahul", "speaker_profile_id": "r1"}]
    recovered = [
        {"word": "budget", "start": 0.35, "end": 0.95, "speaker": "SPEAKER_00", "from_separation": True},  # dup
        {"word": "no", "start": 1.0, "end": 1.2, "speaker": "SPEAKER_01", "from_separation": True},
        {"word": "wait", "start": 1.2, "end": 1.5, "speaker": "SPEAKER_01", "from_separation": True},
    ]
    merged, stats = sep.merge_recovered_words(main, recovered, identified)
    assert stats == {"recovered": 3, "duplicates": 1, "added_words": 2, "added_segments": 1}
    added = [s for s in merged if s.get("from_separation")]
    assert len(added) == 1
    turn = added[0]
    assert turn["speaker_label"] == "Rahul" and turn["speaker_profile_id"] == "r1"
    assert turn["text"] == "no wait"
    assert turn["is_overlap"] and turn["overlap_speakers"] == ["Vikas"]
    assert merged == sorted(merged, key=lambda s: (s["start"], s["end"]))


def test_merge_splits_runs_on_long_gaps():
    identified = [{"speaker": "SPEAKER_01", "speaker_label": "Rahul"}]
    recovered = [
        {"word": "one", "start": 1.0, "end": 1.2, "speaker": "SPEAKER_01"},
        {"word": "two", "start": 5.0, "end": 5.2, "speaker": "SPEAKER_01"},
    ]
    merged, stats = sep.merge_recovered_words([], recovered, identified)
    assert stats["added_segments"] == 2 and len(merged) == 2


def test_merge_with_nothing_recovered_is_identity():
    main = [_turn(0.0, 1.0, "Vikas", "SPEAKER_00", [("hi", 0.0, 0.5)])]
    merged, stats = sep.merge_recovered_words(main, [], [])
    assert merged is main and stats["added_words"] == 0


def test_missing_model_returns_no_words():
    segs = [_seg(0, 5, "A", [{"start": 1.0, "end": 2.0, "speakers": ["A", "B"]}])]
    with patch.object(sep, "_model_dir", return_value=None):
        assert sep.is_separation_available() is False
        assert sep.recover_overlap_words("x.wav", segs) == []


def test_failure_inside_recovery_is_non_fatal():
    segs = [_seg(0, 5, "A", [{"start": 1.0, "end": 2.0, "speakers": ["A", "B"]}])]
    with patch.object(sep, "is_separation_available", return_value=True), \
         patch("utils.audio_utils.load_audio", side_effect=RuntimeError("boom")):
        assert sep.recover_overlap_words("x.wav", segs) == []


def test_enabled_flag_prefers_user_setting(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "OVERLAP_SEPARATION_ENABLED", False)
    assert sep.is_separation_enabled({}) is False
    assert sep.is_separation_enabled({"overlap_separation_enabled": 1}) is True
    monkeypatch.setattr(settings, "OVERLAP_SEPARATION_ENABLED", True)
    assert sep.is_separation_enabled({"overlap_separation_enabled": 0}) is False
    assert sep.is_separation_enabled({"overlap_separation_enabled": None}) is True
