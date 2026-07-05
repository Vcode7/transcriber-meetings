"""
Audio validation and conversion utilities.

All operations are designed to be memory-efficient — large files (multi-hour
recordings) are handled without loading the entire audio into RAM.

Key design decisions:
  - get_duration() uses soundfile.info() which reads only the file header (O(1) RAM)
  - validate_audio() uses sf.info() for duration; RMS sampled from first 30s only
  - convert_to_wav() uses ffmpeg subprocess — streams conversion, no RAM spike
  - There is NO upper-bound duration limit; recordings of any length are accepted
"""
import os
import subprocess
import warnings
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


# ── Constants ──────────────────────────────────────────────────
MIN_DURATION_SECONDS = 2.0       # reject sub-2s clips (too short to transcribe)
MIN_RMS_THRESHOLD    = 0.005     # reject near-silent recordings
RMS_SAMPLE_SECONDS   = 30.0      # number of seconds sampled for RMS check
# No MAX_DURATION_SECONDS — recordings of any length are accepted.


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target_sr.

    NOTE: This loads the ENTIRE file into RAM.  Only use for short clips
    (e.g. voice samples, overlap detector inputs).  For full-recording
    operations use soundfile streaming or ffmpeg.
    """
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def get_duration(file_path: str) -> float:
    """Return duration of audio file in seconds.

    Uses soundfile.info() which reads only the file header — O(1) RAM,
    regardless of file size.
    """
    info = sf.info(file_path)
    return info.duration


def validate_audio(file_path: str) -> Tuple[bool, str]:
    """
    Validate an audio recording.
    Returns (is_valid, reason).

    Memory-efficient implementation:
      - Duration is read from the file header only (sf.info).
      - RMS is computed from the first RMS_SAMPLE_SECONDS only (not the full file).
      - There is NO upper-bound duration limit.
    """
    try:
        info = sf.info(file_path)
        duration = info.duration
        sr = info.samplerate

        if duration < MIN_DURATION_SECONDS:
            return False, f"Recording too short ({duration:.1f}s). Minimum is {MIN_DURATION_SECONDS}s."

        # Compute RMS from first RMS_SAMPLE_SECONDS to avoid loading the whole file.
        sample_frames = int(min(duration, RMS_SAMPLE_SECONDS) * sr)
        with sf.SoundFile(file_path) as f:
            audio_sample = f.read(frames=sample_frames, dtype="float32", always_2d=False)

        # Convert to mono if multi-channel
        if audio_sample.ndim > 1:
            audio_sample = audio_sample.mean(axis=1)

        rms = float(np.sqrt(np.mean(audio_sample ** 2)))
        if rms < MIN_RMS_THRESHOLD:
            return (
                False,
                f"Recording too quiet (RMS={rms:.4f}). Please speak closer to the microphone.",
            )

        return True, "ok"

    except Exception as e:
        return False, f"Could not process audio: {str(e)}"


def convert_to_wav(input_path: str, output_path: str, sr: int = 16000) -> str:
    """
    Convert any audio format to 16kHz mono WAV using ffmpeg.

    ffmpeg streams the conversion without loading the entire file into RAM,
    making this safe for multi-hour recordings.  Falls back to the librosa-
    based method if ffmpeg is not available.

    Parameters
    ----------
    input_path  : source file (any format ffmpeg supports)
    output_path : destination WAV path
    sr          : target sample rate (default 16000)

    Returns
    -------
    output_path on success.  Raises RuntimeError on failure.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",          # overwrite output without asking
                "-i", input_path,        # input file
                "-ar", str(sr),          # resample to target SR
                "-ac", "1",              # mono
                "-f", "wav",             # WAV output
                "-acodec", "pcm_s16le",  # 16-bit PCM
                output_path,
            ],
            capture_output=True,
            timeout=7200,  # 2-hour timeout for very long files
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[-500:]
            raise RuntimeError(f"ffmpeg returned code {result.returncode}: {stderr}")
        return output_path

    except FileNotFoundError:
        # ffmpeg not available — fall back to librosa (loads file into RAM)
        import warnings as _w
        _w.warn(
            "ffmpeg not found — falling back to librosa for WAV conversion. "
            "Large files may use significant RAM.",
            RuntimeWarning,
            stacklevel=2,
        )
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


def split_wav_to_files(
    input_wav: str,
    output_dir: str,
    chunk_sec: float = 600.0,
    prefix: str = "chunk_",
) -> list[tuple[int, float, float, str]]:
    """
    Split a WAV file into fixed-length chunk files using ffmpeg.

    This is memory-efficient — ffmpeg reads and writes the file in a stream;
    the full audio is never loaded into Python memory.

    Parameters
    ----------
    input_wav  : path to the source WAV file
    output_dir : directory to write chunk files into
    chunk_sec  : duration of each chunk in seconds (default 600 = 10 min)
    prefix     : filename prefix for chunk files

    Returns
    -------
    List of (chunk_index, start_sec, end_sec, chunk_path) tuples,
    one per produced chunk file.
    """
    total_duration = get_duration(input_wav)
    os.makedirs(output_dir, exist_ok=True)

    chunks = []
    chunk_index = 0
    start_sec = 0.0

    while start_sec < total_duration:
        end_sec = min(start_sec + chunk_sec, total_duration)
        chunk_path = os.path.join(output_dir, f"{prefix}{chunk_index:04d}.wav")

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_wav,
                "-ss", str(start_sec),
                "-t", str(chunk_sec),
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                "-acodec", "pcm_s16le",
                chunk_path,
            ],
            capture_output=True,
            timeout=300,  # 5-minute timeout per chunk
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[-300:]
            raise RuntimeError(
                f"ffmpeg chunk split failed for chunk {chunk_index} "
                f"(start={start_sec:.1f}s): {stderr}"
            )

        chunks.append((chunk_index, start_sec, end_sec, chunk_path))
        chunk_index += 1
        start_sec = end_sec

    return chunks
