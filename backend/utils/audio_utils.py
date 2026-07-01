"""Audio validation and conversion utilities."""
import os
import io
import wave
import warnings
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


MIN_DURATION_SECONDS = 2.0
MAX_DURATION_SECONDS = 3600.0  # 1 hour
MIN_RMS_THRESHOLD = 0.005       # reject near-silent recordings


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target_sr."""
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def get_duration(file_path: str) -> float:
    """Get duration of audio file in seconds."""
    info = sf.info(file_path)
    return info.duration


def validate_audio(file_path: str) -> Tuple[bool, str]:
    """
    Validate an audio recording.
    Returns (is_valid, reason).
    """
    try:
        audio, sr = load_audio(file_path)
        duration = len(audio) / sr

        if duration < MIN_DURATION_SECONDS:
            return False, f"Recording too short ({duration:.1f}s). Minimum is {MIN_DURATION_SECONDS}s."

        if duration > MAX_DURATION_SECONDS:
            return False, f"Recording too long ({duration:.0f}s). Maximum is {MAX_DURATION_SECONDS}s."

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < MIN_RMS_THRESHOLD:
            return False, f"Recording too quiet (RMS={rms:.4f}). Please speak closer to the microphone."

        return True, "ok"

    except Exception as e:
        return False, f"Could not process audio: {str(e)}"


def convert_to_wav(input_path: str, output_path: str, sr: int = 16000) -> str:
    """Convert any audio format to 16kHz mono WAV."""
    audio, _ = librosa.load(input_path, sr=sr, mono=True)
    sf.write(output_path, audio, sr, subtype="PCM_16")
    return output_path


def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def split_into_chunks(audio: np.ndarray, sr: int, chunk_sec: float = 5.0):
    """Yield (start_sec, chunk_array) for each chunk."""
    chunk_size = int(chunk_sec * sr)
    for i in range(0, len(audio), chunk_size):
        start = i / sr
        yield start, audio[i : i + chunk_size]
