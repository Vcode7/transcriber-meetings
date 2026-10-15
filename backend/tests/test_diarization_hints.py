"""Diarization speaker-count hints and community-1 hyper-parameter overrides."""
from unittest.mock import MagicMock, patch

from config import settings
from services import diarization


def test_expected_speakers_becomes_num_speakers():
    assert diarization.resolve_speaker_hints(participant_count=2, expected_speakers=4) == {"num_speakers": 4}


def test_participants_become_min_speakers():
    assert diarization.resolve_speaker_hints(participant_count=3) == {"min_speakers": 3}


def test_single_participant_or_disabled_gives_no_hint(monkeypatch):
    assert diarization.resolve_speaker_hints(participant_count=1) == {}
    monkeypatch.setattr(settings, "DIARIZATION_USE_PARTICIPANT_COUNT", False)
    assert diarization.resolve_speaker_hints(participant_count=3) == {}


def test_diarize_forwards_hints_to_pyannote():
    annotation = MagicMock()
    annotation.itertracks.return_value = []
    pipeline = MagicMock(return_value=annotation)
    with patch.object(diarization, "is_pyannote_available", return_value=True), \
         patch.object(diarization, "get_diarization_pipeline", return_value=pipeline), \
         patch.object(diarization, "_load_for_pyannote", return_value={"waveform": None, "sample_rate": 16000}):
        diarization.diarize("x.wav", min_speakers=3)
    _, kwargs = pipeline.call_args
    assert kwargs == {"min_speakers": 3}


def test_diarize_without_hints_passes_no_kwargs():
    annotation = MagicMock()
    annotation.itertracks.return_value = []
    pipeline = MagicMock(return_value=annotation)
    with patch.object(diarization, "is_pyannote_available", return_value=True), \
         patch.object(diarization, "get_diarization_pipeline", return_value=pipeline), \
         patch.object(diarization, "_load_for_pyannote", return_value={"waveform": None, "sample_rate": 16000}):
        diarization.diarize("x.wav")
    _, kwargs = pipeline.call_args
    assert kwargs == {}


def test_hyperparameter_overrides_only_touch_set_values(monkeypatch):
    monkeypatch.setattr(settings, "DIARIZATION_CLUSTERING_THRESHOLD", 0.55)
    monkeypatch.setattr(settings, "DIARIZATION_SEG_MIN_DURATION_OFF", 0.2)
    monkeypatch.setattr(settings, "DIARIZATION_VBX_FA", None)
    monkeypatch.setattr(settings, "DIARIZATION_VBX_FB", None)
    pipeline = MagicMock()
    pipeline.parameters.return_value = {
        "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8},
        "segmentation": {"min_duration_off": 0.0},
    }
    applied = diarization._apply_hyperparameter_overrides(pipeline)
    assert applied == {"clustering": {"threshold": 0.55}, "segmentation": {"min_duration_off": 0.2}}
    pipeline.instantiate.assert_called_once_with({
        "clustering": {"threshold": 0.55, "Fa": 0.07, "Fb": 0.8},
        "segmentation": {"min_duration_off": 0.2},
    })


def test_no_overrides_leaves_pipeline_untouched(monkeypatch):
    for attr in diarization._HYPERPARAMETER_SETTINGS:
        monkeypatch.setattr(settings, attr, None)
    pipeline = MagicMock()
    assert diarization._apply_hyperparameter_overrides(pipeline) == {}
    pipeline.instantiate.assert_not_called()


def test_rejected_override_is_non_fatal(monkeypatch):
    monkeypatch.setattr(settings, "DIARIZATION_CLUSTERING_THRESHOLD", 5.0)
    pipeline = MagicMock()
    pipeline.parameters.return_value = {"clustering": {"threshold": 0.6}}
    pipeline.instantiate.side_effect = ValueError("out of range")
    assert diarization._apply_hyperparameter_overrides(pipeline) == {}


def test_collect_overlap_windows_dedupes_and_sorts():
    w1 = {"start": 5.0, "end": 6.0, "speakers": ["A", "B"]}
    w2 = {"start": 1.0, "end": 2.0, "speakers": ["A", "C"]}
    segs = [{"overlap_regions": [w1]}, {"overlap_regions": [w1, w2]}, {}]
    assert diarization.collect_overlap_windows(segs) == [w2, w1]
