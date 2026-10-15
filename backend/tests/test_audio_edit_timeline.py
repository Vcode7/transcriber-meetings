"""
Audio trim / middle-cut: keep-range math, transcript + OCR remapping, the
router-level request handling, and sample-exact ffmpeg rendering.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException

from utils.audio_edit import (
    build_keep_ranges,
    is_identity,
    kept_duration,
    map_time,
    map_time_clamped,
    remap_segments,
    remap_time_blocks,
    rescale_for_source_duration,
)

SR = 16000


# ── build_keep_ranges ────────────────────────────────────────────────────────

def test_keep_ranges_omitted_boundaries_mean_file_edges():
    assert build_keep_ranges(100.0) == [(0.0, 100.0)]
    assert build_keep_ranges(100.0, start_sec=10.0) == [(10.0, 100.0)]
    assert build_keep_ranges(100.0, end_sec=60.0) == [(0.0, 60.0)]


def test_keep_ranges_end_beyond_file_is_clamped_not_rejected():
    # The browser may report a slightly longer duration than the decoded audio.
    assert build_keep_ranges(100.0, start_sec=5.0, end_sec=100.4) == [(5.0, 100.0)]


def test_keep_ranges_trim_and_cut():
    assert build_keep_ranges(100.0, 10.0, 90.0, 30.0, 50.0) == [(10.0, 30.0), (50.0, 90.0)]


def test_keep_ranges_cut_partly_outside_trim_is_intersected():
    # Cut [5, 30] overlaps the trimmed-away head: only [10, 30] is removed.
    assert build_keep_ranges(100.0, 10.0, 90.0, 5.0, 30.0) == [(30.0, 90.0)]
    # Cut entirely outside the trimmed window has no effect.
    assert build_keep_ranges(100.0, 10.0, 50.0, 60.0, 70.0) == [(10.0, 50.0)]


def test_keep_ranges_cut_touching_edges():
    assert build_keep_ranges(10.0, cut_start_sec=0.0, cut_end_sec=3.0) == [(3.0, 10.0)]
    assert build_keep_ranges(10.0, cut_start_sec=7.0, cut_end_sec=10.0) == [(0.0, 7.0)]


def test_keep_ranges_reject_empty_selection():
    with pytest.raises(ValueError):
        build_keep_ranges(10.0, start_sec=5.0, end_sec=5.0)
    with pytest.raises(ValueError):
        build_keep_ranges(10.0, start_sec=20.0)  # start past the end of the audio
    with pytest.raises(ValueError):
        build_keep_ranges(10.0, 2.0, 8.0, 1.0, 9.0)  # cut swallows the whole window


def test_is_identity():
    assert is_identity([(0.0, 10.0)], 10.0)
    assert is_identity([(0.005, 9.995)], 10.0)
    assert not is_identity([(0.5, 10.0)], 10.0)
    assert not is_identity([(0.0, 4.0), (5.0, 10.0)], 10.0)


# ── time mapping ─────────────────────────────────────────────────────────────

KEEP = [(10.0, 30.0), (50.0, 90.0)]  # edited audio: 0–20 from [10,30], 20–60 from [50,90]


def test_map_time():
    assert map_time(10.0, KEEP) == 0.0
    assert map_time(25.0, KEEP) == 15.0
    assert map_time(50.0, KEEP) == 20.0
    assert map_time(90.0, KEEP) == 60.0
    assert map_time(5.0, KEEP) is None
    assert map_time(40.0, KEEP) is None
    assert map_time(95.0, KEEP) is None


def test_map_time_clamped_snaps_removed_time_forward():
    assert map_time_clamped(5.0, KEEP) == 0.0
    assert map_time_clamped(40.0, KEEP) == 20.0
    assert map_time_clamped(95.0, KEEP) == 60.0
    assert map_time_clamped(60.0, KEEP) == 30.0
    assert kept_duration(KEEP) == 60.0


def test_rescale_only_when_browser_duration_is_clearly_off():
    pts = {"trim_start_sec": 60.0, "trim_end_sec": 540.0, "cut_start_sec": None, "cut_end_sec": None}
    same, scale = rescale_for_source_duration(pts, 600.4, 600.0)
    assert scale == 1.0 and same == pts
    scaled, scale = rescale_for_source_duration(pts, 600.0, 660.0)  # VBR estimate 10% short
    assert scale == pytest.approx(1.1)
    assert scaled["trim_start_sec"] == pytest.approx(66.0)
    assert scaled["trim_end_sec"] == pytest.approx(594.0)
    assert scaled["cut_start_sec"] is None
    unchanged, scale = rescale_for_source_duration(pts, None, 600.0)
    assert scale == 1.0 and unchanged == pts


# ── transcript remapping ─────────────────────────────────────────────────────

def _w(word, start, end):
    return {"word": word, "start": start, "end": end, "score": 0.9}


def test_remap_segment_inside_keep_range_is_shifted_with_text_untouched():
    seg = {"start": 55.0, "end": 58.0, "text": " Hello there.", "speaker": "A",
           "words": [_w("Hello", 55.0, 55.5), _w("there.", 56.0, 58.0)]}
    [out] = remap_segments([seg], KEEP)
    assert (out["start"], out["end"]) == (25.0, 28.0)
    assert out["text"] == " Hello there."
    assert out["speaker"] == "A"
    assert [(w["start"], w["end"]) for w in out["words"]] == [(25.0, 25.5), (26.0, 28.0)]
    assert seg["start"] == 55.0  # input not mutated


def test_remap_drops_segments_in_removed_regions():
    segs = [
        {"start": 2.0, "end": 4.0, "text": "before trim", "words": [_w("before", 2.0, 3.0), _w("trim", 3.0, 4.0)]},
        {"start": 35.0, "end": 40.0, "text": "in the cut", "words": []},
        {"start": 92.0, "end": 95.0, "text": "after trim"},
    ]
    assert remap_segments(segs, KEEP) == []


def test_remap_splits_segment_crossing_a_cut_at_word_level():
    seg = {"start": 28.0, "end": 52.0, "text": "one two three four",
           "words": [_w("one", 28.0, 28.5), _w("two", 29.0, 29.5),
                     _w("three", 40.0, 41.0),            # removed
                     _w("four", 51.0, 52.0)]}
    out = remap_segments([seg], KEEP)
    assert [s["text"] for s in out] == ["one two", "four"]
    assert (out[0]["start"], out[0]["end"]) == (18.0, 19.5)
    assert (out[1]["start"], out[1]["end"]) == (21.0, 22.0)


def test_remap_untimed_words_follow_their_neighbour():
    seg = {"start": 8.0, "end": 12.0, "text": "cut 42 kept",
           "words": [_w("cut", 8.0, 9.0), {"word": "42"}, _w("kept", 11.0, 12.0)]}
    [out] = remap_segments([seg], KEEP)
    assert out["text"] == "kept"
    seg2 = {"start": 11.0, "end": 13.0, "text": "kept 42",
            "words": [_w("kept", 11.0, 12.0), {"word": "42"}]}
    [out2] = remap_segments([seg2], KEEP)
    assert out2["text"] == "kept 42"


def test_remap_segment_without_words_kept_by_midpoint_and_clamped():
    [out] = remap_segments([{"start": 28.0, "end": 31.0, "text": "edge"}], KEEP)
    assert (out["start"], out["end"]) == (18.0, 20.0)


def test_remap_time_blocks_for_ocr():
    blocks = [
        {"start": 0.0, "end": 8.0, "text": "title slide"},     # trimmed away
        {"start": 12.0, "end": 20.0, "text": "agenda"},         # kept
        {"start": 25.0, "end": 60.0, "text": "spans the cut"},  # collapses across the cut
        {"start": 35.0, "end": 45.0, "text": "only in cut"},    # removed
        {"start": 85.0, "end": 100.0, "text": "closing"},       # clipped at trim end
    ]
    out = remap_time_blocks(blocks, KEEP)
    assert [(b["text"], b["start"], b["end"]) for b in out] == [
        ("agenda", 2.0, 10.0),
        ("spans the cut", 15.0, 30.0),
        ("closing", 55.0, 60.0),
    ]


# ── ffmpeg rendering is sample-exact ─────────────────────────────────────────

@pytest.fixture
def noise_wav():
    """10 s of seeded int16 noise: any shift or dropped sample changes the data."""
    rng = np.random.default_rng(1234)
    data = rng.integers(-20000, 20000, size=SR * 10, dtype=np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        path = tf.name
    sf.write(path, data, SR, subtype="PCM_16")
    yield path, data
    for p in (path, os.path.splitext(path)[0] + "_edited.wav"):
        if os.path.exists(p):
            os.remove(p)


def _read_int16(path):
    out, sr = sf.read(path, dtype="int16")
    assert sr == SR
    return out


def test_process_audio_edit_keeps_exact_samples(noise_wav):
    from utils.audio_utils import process_audio_edit
    path, data = noise_wav
    edited = process_audio_edit(path, start_sec=1.25, end_sec=9.0, cut_start_sec=3.5, cut_end_sec=5.75)
    expected = np.concatenate([data[int(1.25 * SR):int(3.5 * SR)], data[int(5.75 * SR):int(9.0 * SR)]])
    np.testing.assert_array_equal(_read_int16(edited), expected)


def test_process_audio_edit_start_only_keeps_true_end(noise_wav):
    from utils.audio_utils import process_audio_edit
    path, data = noise_wav
    edited = process_audio_edit(path, start_sec=2.0)
    np.testing.assert_array_equal(_read_int16(edited), data[2 * SR:])


def test_process_audio_edit_whole_file_is_a_no_op(noise_wav):
    from utils.audio_utils import process_audio_edit
    path, _ = noise_wav
    assert process_audio_edit(path, start_sec=0.0, end_sec=10.0) == path
    assert process_audio_edit(path) == path


# ── router helpers ───────────────────────────────────────────────────────────

def test_build_edit_request():
    from routers.audio import build_edit_request
    assert build_edit_request(None, None, None, None, 600.0) is None
    assert build_edit_request(None, None, 5.0, 5.0, 600.0) is None  # zero-length cut = no edit
    assert build_edit_request(1.23456, None, None, None, 600.0) == {
        "trim_start_sec": 1.235, "trim_end_sec": None,
        "cut_start_sec": None, "cut_end_sec": None, "source_duration_sec": 600.0,
    }
    # Stable through a JSON round-trip (used for duplicate-upload matching).
    req = build_edit_request(1.0, 9.0, 3.0, 4.0, 10.0)
    assert json.loads(json.dumps(req)) == req


def test_apply_edit_request_cuts_and_records(noise_wav):
    from routers.audio import apply_edit_request, build_edit_request, keep_ranges_of
    path, data = noise_wav
    req = build_edit_request(1.0, None, 4.0, 6.0, 10.0)
    new_path, edit = apply_edit_request(path, req)
    assert new_path != path and not os.path.exists(path)  # source replaced
    assert edit["keep_ranges"] == [[1.0, 4.0], [6.0, 10.0]]
    assert edit["output_duration"] == pytest.approx(7.0, abs=0.01)
    assert edit["audio_path"] == new_path
    assert edit["request"] == req
    assert keep_ranges_of(json.dumps(edit)) == [(1.0, 4.0), (6.0, 10.0)]
    np.testing.assert_array_equal(
        _read_int16(new_path), np.concatenate([data[SR:4 * SR], data[6 * SR:]])
    )


def test_apply_edit_request_whole_file_selection_keeps_source(noise_wav):
    from routers.audio import apply_edit_request, build_edit_request
    path, _ = noise_wav
    new_path, edit = apply_edit_request(path, build_edit_request(0.0, 10.4, None, None, 10.4))
    assert new_path == path and os.path.exists(path)
    assert edit["keep_ranges"] is None


def test_apply_edit_request_invalid_selection_is_an_error_not_a_silent_fallback(noise_wav):
    from routers.audio import apply_edit_request, build_edit_request
    path, _ = noise_wav
    with pytest.raises(HTTPException) as exc:
        apply_edit_request(path, build_edit_request(2.0, 8.0, 1.0, 9.0, 10.0))
    assert exc.value.status_code == 422
    assert os.path.exists(path)  # left for the caller to clean up


def test_apply_edit_request_without_request_is_untouched(noise_wav):
    from routers.audio import apply_edit_request
    path, _ = noise_wav
    assert apply_edit_request(path, None) == (path, None)


def test_keep_ranges_of_handles_missing_and_bad_values():
    from routers.audio import keep_ranges_of
    assert keep_ranges_of(None) is None
    assert keep_ranges_of("") is None
    assert keep_ranges_of("not json") is None
    assert keep_ranges_of(json.dumps({"keep_ranges": None})) is None
