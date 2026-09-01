import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import pytest
from utils.audio_utils import process_audio_edit, get_duration, trim_audio


@pytest.fixture
def sample_wav_path():
    """Generates a synthetic 10-second 16kHz mono WAV file."""
    sr = 16000
    duration_sec = 10.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    sf.write(wav_path, audio, sr, subtype="PCM_16")
    yield wav_path
    if os.path.exists(wav_path):
        os.remove(wav_path)


def test_trim_start_end_only(sample_wav_path):
    """Test trimming start and end without middle cut."""
    edited = process_audio_edit(sample_wav_path, start_sec=2.0, end_sec=7.0)
    assert os.path.exists(edited)
    dur = get_duration(edited)
    assert abs(dur - 5.0) < 0.1
    if os.path.exists(edited) and edited != sample_wav_path:
        os.remove(edited)


def test_cut_middle_section(sample_wav_path):
    """Test removing 2:00 to 4:00 (cut_start=2.0, cut_end=4.0) from a 10s audio -> 8s audio."""
    edited = process_audio_edit(sample_wav_path, cut_start_sec=2.0, cut_end_sec=4.0)
    assert os.path.exists(edited)
    dur = get_duration(edited)
    # Expected duration: [0..2] (2s) + [4..10] (6s) = 8s
    assert abs(dur - 8.0) < 0.1
    if os.path.exists(edited) and edited != sample_wav_path:
        os.remove(edited)


def test_trim_and_cut_combined(sample_wav_path):
    """Test trimming [1.0..9.0] and cutting [3.0..5.0] -> [1..3] (2s) + [5..9] (4s) = 6s total."""
    edited = process_audio_edit(
        sample_wav_path,
        start_sec=1.0,
        end_sec=9.0,
        cut_start_sec=3.0,
        cut_end_sec=5.0,
    )
    assert os.path.exists(edited)
    dur = get_duration(edited)
    assert abs(dur - 6.0) < 0.1
    if os.path.exists(edited) and edited != sample_wav_path:
        os.remove(edited)


def test_cut_from_start_edge(sample_wav_path):
    """Test removing from 0.0 to 3.0 -> keeps [3.0..10.0] = 7s."""
    edited = process_audio_edit(sample_wav_path, cut_start_sec=0.0, cut_end_sec=3.0)
    assert os.path.exists(edited)
    dur = get_duration(edited)
    assert abs(dur - 7.0) < 0.1
    if os.path.exists(edited) and edited != sample_wav_path:
        os.remove(edited)


def test_cut_to_end_edge(sample_wav_path):
    """Test removing from 7.0 to 10.0 -> keeps [0.0..7.0] = 7s."""
    edited = process_audio_edit(sample_wav_path, cut_start_sec=7.0, cut_end_sec=10.0)
    assert os.path.exists(edited)
    dur = get_duration(edited)
    assert abs(dur - 7.0) < 0.1
    if os.path.exists(edited) and edited != sample_wav_path:
        os.remove(edited)


def test_trim_audio_wrapper(sample_wav_path):
    """Test trim_audio backward compatibility wrapper."""
    trimmed = trim_audio(sample_wav_path, start_sec=1.0, end_sec=4.0)
    assert os.path.exists(trimmed)
    dur = get_duration(trimmed)
    assert abs(dur - 3.0) < 0.1
    if os.path.exists(trimmed) and trimmed != sample_wav_path:
        os.remove(trimmed)
