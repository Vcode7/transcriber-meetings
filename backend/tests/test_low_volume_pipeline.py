"""
Unit tests for Low-Volume Speech Transcription Pipeline Enhancements.

Tests:
1. Audio normalization & dynamic range compression
2. Speech padding & region boundary expansion
3. Speech segment merging with configurable silence gap
4. Low-volume rejected region energy analysis
5. Settings model configuration and defaults
"""
import pytest
import numpy as np

from services.audio_preprocessing import (
    _apply_dynamic_range_compression,
    _normalize_loudness,
    apply_speech_padding,
    merge_speech_segments,
    detect_rejected_low_volume_regions,
)
from models.settings import UserSettings, UserSettingsUpdate


def test_dynamic_range_compression():
    """Verify that dynamic range compression boosts lower energy signals."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    # Signal with a quiet section (amplitude 0.05) and a loud peak (amplitude 0.8)
    signal = np.sin(2 * np.pi * 440 * t)
    signal[:sr // 2] *= 0.05
    signal[sr // 2:] *= 0.8

    compressed = _apply_dynamic_range_compression(signal, ratio=2.5, threshold_db=-24.0)

    quiet_rms_orig = np.sqrt(np.mean(signal[:sr // 2] ** 2))
    quiet_rms_comp = np.sqrt(np.mean(compressed[:sr // 2] ** 2))

    # Quiet region RMS energy should increase relative to peak after compression
    assert quiet_rms_comp > quiet_rms_orig
    assert np.max(np.abs(compressed)) <= 1.0


def test_speech_padding():
    """Verify pre- and post-segment padding expansion and merging of overlapping regions."""
    regions = [(1.0, 2.0), (4.0, 5.0)]
    pad_ms = 400.0  # 0.4 seconds
    total_dur = 10.0

    padded = apply_speech_padding(regions, pad_ms, total_dur)
    assert len(padded) == 2
    assert padded[0] == (0.6, 2.4)
    assert padded[1] == (3.6, 5.4)

    # Overlapping case
    regions_close = [(1.0, 2.0), (2.5, 3.5)]
    padded_close = apply_speech_padding(regions_close, 400.0, total_dur)
    assert len(padded_close) == 1
    assert padded_close[0] == (0.6, 3.9)


def test_speech_segment_merging():
    """Verify merging of nearby speech segments within max_merge_silence_ms."""
    regions = [(1.0, 2.0), (2.3, 3.5), (6.0, 7.0)]
    # Gap between region 1 & 2 is 0.3s (300ms)
    merged_500 = merge_speech_segments(regions, max_merge_silence_ms=500.0)
    assert len(merged_500) == 2
    assert merged_500[0] == (1.0, 3.5)
    assert merged_500[1] == (6.0, 7.0)

    merged_200 = merge_speech_segments(regions, max_merge_silence_ms=200.0)
    assert len(merged_200) == 3


def test_rejected_low_volume_region_detection():
    """Verify detection of rejected audio regions with energy above recovery threshold."""
    sr = 16000
    duration = 5.0
    audio = np.zeros(int(sr * duration), dtype=np.float32)

    # Add quiet speech in interval [2.0s, 3.0s] (amplitude 0.02 ~ -34 dBFS)
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio[int(2.0 * sr):int(3.0 * sr)] = 0.02 * np.sin(2 * np.pi * 300 * t)

    # VAD detected only [0.5s, 1.5s] as primary speech
    speech_regions = [(0.5, 1.5)]

    # Analyze rejected regions with recovery_energy_threshold = -45.0 dB
    recovered = detect_rejected_low_volume_regions(
        audio, sr, speech_regions,
        energy_threshold_db=-45.0,
        min_duration_sec=0.3
    )

    # The rejected region [1.5s, 5.0s] contains quiet speech [2.0s, 3.0s] with energy > -45dB
    assert len(recovered) >= 1
    assert any(r[0] <= 2.0 and r[1] >= 3.0 for r in recovered)


def test_user_settings_low_volume_defaults():
    """Verify Pydantic settings defaults for low-volume and missing segment features."""
    s = UserSettings(user_id="test_user")
    assert s.enable_vad is True
    assert s.enable_transcription_vad is True
    assert s.enable_alignment_vad is True
    assert s.enable_audio_normalization is True
    assert s.norm_target_dbfs == -3.0
    assert s.norm_compression_ratio == 2.0
    assert s.enable_adaptive_vad is True
    assert s.vad_speech_threshold == 0.15
    assert s.enable_speech_padding is True
    assert s.speech_pad_ms == 400
    assert s.enable_speech_segment_merging is True
    assert s.max_merge_silence_ms == 500
    assert s.enable_low_volume_recovery is True
    assert s.recovery_energy_threshold == -45.0
    assert s.missing_segment_min_duration_sec == 2.0


def test_missing_segment_min_duration_threshold_filtering():
    """Verify that gaps shorter than the configured threshold are ignored, while gaps >= threshold are recovered."""
    sr = 16000
    duration = 10.0
    audio = np.zeros(int(sr * duration), dtype=np.float32)

    # Add signal energy across distinct quiet regions:
    # Region A: [1.0s, 1.8s] (duration 0.8s) -> below 2.0s threshold
    # Region B: [3.0s, 4.5s] (duration 1.5s) -> below 2.0s threshold
    # Region C: [6.0s, 8.5s] (duration 2.5s) -> EQUAL/GREATER than 2.0s threshold
    t_a = np.linspace(0, 0.8, int(0.8 * sr), dtype=np.float32)
    audio[int(1.0 * sr):int(1.8 * sr)] = 0.05 * np.sin(2 * np.pi * 300 * t_a)

    t_b = np.linspace(0, 1.5, int(1.5 * sr), dtype=np.float32)
    audio[int(3.0 * sr):int(4.5 * sr)] = 0.05 * np.sin(2 * np.pi * 300 * t_b)

    t_c = np.linspace(0, 2.5, int(2.5 * sr), dtype=np.float32)
    audio[int(6.0 * sr):int(8.5 * sr)] = 0.05 * np.sin(2 * np.pi * 300 * t_c)

    # Primary speech spans leaving gaps [0-1.0], [1.8-3.0], [4.5-6.0], [8.5-10.0]
    speech_regions = [(0.0, 1.0), (1.8, 3.0), (4.5, 6.0), (8.5, 10.0)]

    # Test 1: Default 2.0s threshold -> only gap [6.0s, 8.5s] (2.5s dur) qualifies
    recovered_2s = detect_rejected_low_volume_regions(
        audio, sr, speech_regions,
        energy_threshold_db=-45.0,
        min_duration_sec=2.0
    )
    assert len(recovered_2s) == 1
    assert recovered_2s[0][0] == 6.0 and recovered_2s[0][1] == 8.5

    # Test 2: Lower threshold 1.0s -> both [3.0s, 4.5s] (1.5s dur) and [6.0s, 8.5s] (2.5s dur) qualify
    # Gap [1.0s, 1.8s] (0.8s dur) is still ignored (< 1.0s)
    recovered_1s = detect_rejected_low_volume_regions(
        audio, sr, speech_regions,
        energy_threshold_db=-45.0,
        min_duration_sec=1.0
    )
    assert len(recovered_1s) == 2
    assert (3.0, 4.5) in recovered_1s
    assert (6.0, 8.5) in recovered_1s

    # Test 3: Higher threshold 3.0s -> no gaps qualify
    recovered_3s = detect_rejected_low_volume_regions(
        audio, sr, speech_regions,
        energy_threshold_db=-45.0,
        min_duration_sec=3.0
    )
    assert len(recovered_3s) == 0


def test_user_settings_update_missing_segment_duration():
    """Verify validation on UserSettingsUpdate for missing_segment_min_duration_sec."""
    update = UserSettingsUpdate(missing_segment_min_duration_sec=3.5)
    assert update.missing_segment_min_duration_sec == 3.5

    # Negative / zero should raise ValidationError
    with pytest.raises(Exception):
        UserSettingsUpdate(missing_segment_min_duration_sec=-1.0)

    with pytest.raises(Exception):
        UserSettingsUpdate(missing_segment_min_duration_sec=0.0)


def test_stage1_action_owner_normalization():
    """Verify Stage 1 action owner extraction, normalization, and fallback heuristics."""
    from services.rom_service import normalize_action_owner

    # Direct valid string
    assert normalize_action_owner("Bob Smith") == "Bob Smith"

    # List of owners with duplicate deduplication
    assert normalize_action_owner(["Alice", "Bob", "Alice"]) == "Alice, Bob"

    # Null, None, undefined, vague pronouns
    assert normalize_action_owner("null") is None
    assert normalize_action_owner("none") is None
    assert normalize_action_owner("we") is None
    assert normalize_action_owner("my team") is None
    assert normalize_action_owner("someone") is None
    assert normalize_action_owner(None) is None

    # Fallback from action_items array
    act_items = [
        {"task": "Prepare report", "assignee": "Charlie"},
        {"task": "Review PR", "assignee": "David"}
    ]
    assert normalize_action_owner(None, action_items=act_items) == "Charlie, David"

    # Fallback from discussion_point text with embedded pattern
    text_with_owner = "Alice proposed the new feature. (Owner: Bob Johnson)"
    assert normalize_action_owner(None, point_text=text_with_owner) == "Bob Johnson"

    bracket_text = "Review security requirements [Assignee: Sarah Connor]"
    assert normalize_action_owner(None, point_text=bracket_text) == "Sarah Connor"

