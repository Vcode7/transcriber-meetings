"""Word→turn resegmentation: per-turn speaker identity and overlap interjections."""
from config import settings
from tasks.pipeline import _resegment_by_word_speakers


def _w(word, start, end, label, raw, pid):
    return {"word": word, "start": start, "end": end, "probability": 0.9,
            "speaker_label": label, "speaker": raw, "speaker_profile_id": pid}


def _segment(words, regions=None):
    return {
        "start": words[0]["start"], "end": words[-1]["end"], "text": "",
        "words": words, "avg_logprob": -0.2,
        "speaker_label": "Vikas", "speaker_profile_id": "vikas_id", "speaker": "SPEAKER_00",
        "is_overlap": bool(regions), "overlap_regions": regions or [],
    }


def test_each_turn_gets_its_own_profile_and_raw_id():
    words = [
        _w("so", 0.0, 0.3, "Vikas", "SPEAKER_00", "vikas_id"),
        _w("we", 0.3, 0.6, "Vikas", "SPEAKER_00", "vikas_id"),
        _w("I", 1.0, 1.2, "Rahul", "SPEAKER_01", "rahul_id"),
        _w("agree", 1.2, 1.6, "Rahul", "SPEAKER_01", "rahul_id"),
    ]
    out = _resegment_by_word_speakers([_segment(words)])
    assert [(s["speaker_label"], s["speaker_profile_id"], s["speaker"]) for s in out] == [
        ("Vikas", "vikas_id", "SPEAKER_00"),
        ("Rahul", "rahul_id", "SPEAKER_01"),
    ]
    # the per-word profile key is internal and stripped
    assert all("speaker_profile_id" not in w for s in out for w in s["words"])


def test_interjection_inside_overlap_window_is_kept():
    region = {"start": 0.9, "end": 1.3, "speakers": ["Rahul", "Vikas"]}
    words = [
        _w("the", 0.0, 0.3, "Vikas", "SPEAKER_00", "vikas_id"),
        _w("plan", 0.3, 0.8, "Vikas", "SPEAKER_00", "vikas_id"),
        _w("yes", 1.0, 1.2, "Rahul", "SPEAKER_01", "rahul_id"),  # 1 word, 0.2 s
        _w("is", 1.4, 1.6, "Vikas", "SPEAKER_00", "vikas_id"),
        _w("set", 1.6, 1.9, "Vikas", "SPEAKER_00", "vikas_id"),
    ]
    out = _resegment_by_word_speakers([_segment(words, [region])])
    assert [s["speaker_label"] for s in out] == ["Vikas", "Rahul", "Vikas"]
    interjection = out[1]
    assert interjection["is_overlap"] is True
    assert interjection["overlap_speakers"] == ["Vikas"]
    # turns that do not touch the window are not marked as overlapping
    assert out[2]["is_overlap"] is False and out[2]["overlap_regions"] == []


def test_isolated_word_outside_overlap_is_still_folded():
    words = [
        _w("the", 0.0, 0.3, "Vikas", "SPEAKER_00", "vikas_id"),
        _w("plan", 0.3, 0.8, "Vikas", "SPEAKER_00", "vikas_id"),
        _w("yes", 1.0, 1.2, "Rahul", "SPEAKER_01", "rahul_id"),
        _w("is", 1.4, 1.6, "Vikas", "SPEAKER_00", "vikas_id"),
    ]
    out = _resegment_by_word_speakers([_segment(words)])
    assert len(out) == 1 and out[0]["speaker_label"] == "Vikas"


def test_interjection_folding_restored_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "OVERLAP_KEEP_INTERJECTIONS", False)
    region = {"start": 0.9, "end": 1.3, "speakers": ["Rahul", "Vikas"]}
    words = [
        _w("the", 0.0, 0.3, "Vikas", "SPEAKER_00", "vikas_id"),
        _w("plan", 0.3, 0.8, "Vikas", "SPEAKER_00", "vikas_id"),
        _w("yes", 1.0, 1.2, "Rahul", "SPEAKER_01", "rahul_id"),
    ]
    out = _resegment_by_word_speakers([_segment(words, [region])])
    assert len(out) == 1


def test_words_without_raw_ids_fall_back_to_segment_identity():
    words = [{"word": "hi", "start": 0.0, "end": 0.4, "speaker_label": "Vikas"},
             {"word": "there", "start": 0.4, "end": 0.8, "speaker_label": "Vikas"}]
    out = _resegment_by_word_speakers([_segment(words)])
    assert out[0]["speaker_profile_id"] == "vikas_id" and out[0]["speaker"] == "SPEAKER_00"
