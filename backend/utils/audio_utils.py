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
from typing import Tuple, Optional, List, Sequence

from utils.audio_edit import build_keep_ranges, is_identity, kept_duration

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


def validate_audio(
    file_path: str,
    enabled: bool = True,
    min_duration: float = MIN_DURATION_SECONDS,
    min_rms: float = MIN_RMS_THRESHOLD
) -> Tuple[bool, str]:
    """
    Validate an audio recording.
    Returns (is_valid, reason).

    Memory-efficient implementation:
      - Duration is read from the file header only (sf.info).
      - Multi-window peak 1-second RMS is sampled (Start, Middle, End) so that
        leading silence/pauses at the beginning of a recording do not cause false rejections.
      - If enabled is False, validation is bypassed completely.
    """
    if not enabled:
        return True, "ok"

    try:
        info = sf.info(file_path)
        duration = info.duration
        sr = info.samplerate
        frames_total = info.frames

        if duration < min_duration:
            return False, f"Recording too short ({duration:.1f}s). Minimum is {min_duration}s."

        # Strategic 15-second sampling windows across the recording (Start, Middle, End)
        sample_dur = 15.0
        sample_frames = int(sample_dur * sr)

        offsets_sec = [0.0]
        if duration > 30.0:
            offsets_sec.append(duration / 2.0)
            offsets_sec.append(max(0.0, duration - sample_dur))

        max_rms_found = 0.0

        with sf.SoundFile(file_path) as f:
            for sec in offsets_sec:
                seek_frame = int(sec * sr)
                if seek_frame >= frames_total:
                    continue
                f.seek(seek_frame)
                read_n = min(sample_frames, frames_total - seek_frame)
                audio_chunk = f.read(frames=read_n, dtype="float32", always_2d=False)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.mean(axis=1)

                if len(audio_chunk) == 0:
                    continue

                # Compute 1-second sliding frame RMS values
                win_size = sr
                if len(audio_chunk) <= win_size:
                    chunk_rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                    max_rms_found = max(max_rms_found, chunk_rms)
                else:
                    for i in range(0, len(audio_chunk) - win_size + 1, win_size // 2):
                        frame = audio_chunk[i:i + win_size]
                        frame_rms = float(np.sqrt(np.mean(frame ** 2)))
                        max_rms_found = max(max_rms_found, frame_rms)

                # Early exit if valid speech signal is detected
                if max_rms_found >= min_rms:
                    break

        if max_rms_found < min_rms:
            return (
                False,
                f"Recording too quiet (Peak RMS={max_rms_found:.4f}). Minimum threshold is {min_rms}.",
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
        run_kwargs = {
            "capture_output": True,
            "timeout": 7200,  # 2-hour timeout for very long files
        }
        if os.name == "nt":
            run_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

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
            **run_kwargs
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


def render_keep_ranges(
    wav_path: str,
    keep_ranges: Sequence[Tuple[float, float]],
    output_path: str,
    sr: int = 16000,
) -> str:
    """
    Write the concatenation of ``keep_ranges`` (seconds, source timeline) of
    ``wav_path`` to ``output_path`` as 16 kHz mono PCM WAV.

    Every case — one range or several — goes through ``atrim`` in a single
    filter graph, which is sample-accurate (input ``-ss`` seeking is not
    guaranteed to be for every input), and the result length is verified.
    """
    if not keep_ranges:
        raise ValueError("render_keep_ranges: no audio ranges to keep.")

    n = len(keep_ranges)
    trims = [
        f"atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS"
        for a, b in keep_ranges
    ]
    if n == 1:
        filter_str = f"[0:a]{trims[0]}[outa]"
    else:
        split_labels = "".join(f"[s{i}]" for i in range(n))
        parts = [f"[0:a]asplit={n}{split_labels}"]
        parts += [f"[s{i}]{trims[i]}[a{i}]" for i in range(n)]
        parts.append("".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[outa]")
        filter_str = ";".join(parts)

    run_kwargs: dict = {"capture_output": True, "timeout": 7200}
    if os.name == "nt":
        run_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-filter_complex", filter_str,
        "-map", "[outa]",
        "-ar", str(sr),
        "-ac", "1",
        "-f", "wav",
        "-acodec", "pcm_s16le",
        output_path,
    ]
    result = subprocess.run(cmd, **run_kwargs)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"ffmpeg audio edit failed (code {result.returncode}): {stderr}")

    expected = kept_duration(keep_ranges)
    actual = get_duration(output_path)
    if abs(actual - expected) > max(0.25, expected * 0.005):
        raise RuntimeError(
            f"Edited audio is {actual:.2f}s but the selection is {expected:.2f}s; "
            "refusing to continue with a mis-cut file."
        )
    return output_path


def process_audio_edit(
    wav_path: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    cut_start_sec: Optional[float] = None,
    cut_end_sec: Optional[float] = None,
) -> str:
    """
    Trim and/or cut a WAV file using ffmpeg.

    Supports:
    - Trimming boundary edges: [start_sec, end_sec]; either side may be None,
      meaning "from the beginning" / "to the end of the file".
    - Removing a middle section: [cut_start_sec, cut_end_sec] and seamlessly joining the rest.
    - Combining both: keeps [start_sec, cut_start_sec] and [cut_end_sec, end_sec].

    Returns the path to the edited WAV file (sibling of the original, with _edited suffix),
    or ``wav_path`` itself when the edit keeps the whole file.
    Original file is NOT deleted — caller is responsible for cleanup.
    Raises ValueError for an edit that leaves no audio, RuntimeError if ffmpeg fails.
    """
    file_dur = get_duration(wav_path)
    keep = build_keep_ranges(file_dur, start_sec, end_sec, cut_start_sec, cut_end_sec)
    if is_identity(keep, file_dur):
        return wav_path

    base, ext = os.path.splitext(wav_path)
    edited_path = f"{base}_edited{ext}"
    return render_keep_ranges(wav_path, keep, edited_path)


def trim_audio(wav_path: str, start_sec: float, end_sec: float) -> str:
    """Backward-compatible wrapper for trim_audio."""
    return process_audio_edit(wav_path, start_sec=start_sec, end_sec=end_sec)


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

        run_kwargs = {
            "capture_output": True,
            "timeout": 300,  # 5-minute timeout per chunk
        }
        if os.name == "nt":
            run_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

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
            **run_kwargs
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
