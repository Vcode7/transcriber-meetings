"""
Audio Preprocessing Service

Provides audio cleanup and VAD-based speech segmentation helpers that run
BEFORE WhisperX forced alignment.

Design goals
------------
- Pure numpy/soundfile only for cleanup (no noisereduce, no librosa denoising)
- webrtcvad used opportunistically; falls back to energy-based VAD if not installed
- Original WAV is never modified; preprocessed copies use temp files
- All functions are stateless and safe to call from a thread pool

Preprocessing pipeline (when enabled)
--------------------------------------
1. Load 16 kHz mono WAV
2. Silence trim (leading/trailing)
3. Soft-clipping repair
4. Loudness normalization (peak → -3 dBFS)
5. Light spectral subtraction denoising (pure numpy)
6. Write to temp WAV → return temp path

VAD / alignment chunking
-------------------------
detect_speech_regions() returns a list of (start_sec, end_sec) speech spans.
These can be used by the transcription service to align each speech region
separately and then stitch timestamps back together.
"""
from __future__ import annotations

import logging
import os
import tempfile
import numpy as np
import soundfile as sf
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Hz — everything runs at 16 kHz
FRAME_DURATION_MS = 30        # ms per VAD frame (10 / 20 / 30 allowed by webrtcvad)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # samples per frame

# Energy VAD thresholds
_ENERGY_SPEECH_QUANTILE = 0.25   # frames above this energy quantile = speech
_MIN_SPEECH_SEC = 0.3            # minimum speech region length to keep
_MIN_SILENCE_SEC = 0.5           # minimum silence to split on

# Loudness target
_PEAK_TARGET_DBFS = -3.0         # dBFS


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Audio I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_wav_mono(wav_path: str) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file as mono float32 at native sample rate.
    Multi-channel files are mixed to mono.
    """
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _save_wav(audio: np.ndarray, sr: int, path: str) -> None:
    """Write float32 mono audio to a 16-bit PCM WAV file."""
    # Clip to [-1, 1] before writing to prevent overflow artefacts
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sr, subtype="PCM_16")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Audio cleanup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -45.0) -> np.ndarray:
    """
    Trim leading and trailing silence below `threshold_db` (dBFS).
    Uses a 20ms analysis window.  Falls back to returning the original
    audio unchanged if everything is below the threshold (to avoid
    returning an empty array).
    """
    frame_len = int(sr * 0.02)   # 20 ms frames
    if frame_len <= 0 or len(audio) < frame_len:
        return audio

    threshold_linear = 10 ** (threshold_db / 20.0)

    # Compute RMS per frame
    n_frames = len(audio) // frame_len
    trimmed = audio[: n_frames * frame_len]
    frames = trimmed.reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    speech_frames = np.where(rms > threshold_linear)[0]
    if len(speech_frames) == 0:
        return audio  # don't trim a fully-silent file

    start_sample = speech_frames[0] * frame_len
    end_sample = min((speech_frames[-1] + 1) * frame_len, len(audio))

    trimmed_audio = audio[start_sample:end_sample]
    if len(trimmed_audio) == 0:
        return audio

    trim_secs = (start_sample / sr, (len(audio) - end_sample) / sr)
    logger.debug(
        f"[Preprocess] Silence trim: removed {trim_secs[0]:.2f}s lead, "
        f"{trim_secs[1]:.2f}s tail"
    )
    return trimmed_audio


def _repair_clipping(audio: np.ndarray, clip_threshold: float = 0.98) -> np.ndarray:
    """
    Soft clipping repair: detect saturated samples and apply cubic soft-knee
    compression to reduce the harsh edges.  Leaves non-clipped samples unchanged.
    """
    clipped = np.abs(audio) >= clip_threshold
    n_clipped = int(clipped.sum())
    if n_clipped == 0:
        return audio

    out = audio.copy()
    # Cubic soft knee: maps x→1 smoothly
    mask_pos = (audio > clip_threshold)
    mask_neg = (audio < -clip_threshold)
    # Soft saturation: y = sign(x) * (1 - (1 - |x|/clip_threshold)^3 * (clip_threshold / |x|))
    # Simplified: cubic blend from clip_threshold to 1.0
    for mask, sign in [(mask_pos, 1.0), (mask_neg, -1.0)]:
        if not mask.any():
            continue
        abs_vals = np.abs(audio[mask])
        # Normalize to [0, 1] past threshold
        excess = (abs_vals - clip_threshold) / (1.0 - clip_threshold + 1e-8)
        excess = np.clip(excess, 0.0, 1.0)
        # Cubic attenuation
        factor = 1.0 - 0.3 * excess ** 2
        out[mask] = sign * abs_vals * factor

    logger.debug(f"[Preprocess] Clipping repair: {n_clipped} clipped samples fixed")
    return out


def _normalize_loudness(audio: np.ndarray, target_dbfs: float = _PEAK_TARGET_DBFS) -> np.ndarray:
    """
    Peak-normalize audio to `target_dbfs` dBFS.
    Skips normalization if the audio is near-silent to avoid amplifying noise.
    """
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio  # near-silent — don't amplify

    target_linear = 10 ** (target_dbfs / 20.0)
    gain = target_linear / peak

    # Safety guard: don't amplify by more than 40 dB to prevent noise pumping
    max_gain = 100.0
    if gain > max_gain:
        logger.debug(
            f"[Preprocess] Loudness normalization: peak={20*np.log10(peak):.1f} dBFS, "
            f"gain would be {20*np.log10(gain):.1f} dB — capped at {20*np.log10(max_gain):.0f} dB"
        )
        gain = max_gain

    out = audio * gain
    logger.debug(
        f"[Preprocess] Loudness normalization: peak={20*np.log10(peak):.1f} dBFS → "
        f"{target_dbfs:.1f} dBFS (gain={20*np.log10(gain):.1f} dB)"
    )
    return np.clip(out, -1.0, 1.0)


def _spectral_subtract_denoise(
    audio: np.ndarray,
    sr: int,
    noise_floor_quantile: float = 0.15,
) -> np.ndarray:
    """
    Light spectral subtraction denoising (pure numpy — no external deps).

    Estimates the noise floor from the quietest `noise_floor_quantile` frames
    and subtracts their average magnitude spectrum from each frame.
    Uses overlap-add reconstruction to avoid blocking artefacts.

    This is intentionally gentle: the spectral floor factor is small so that
    speech quality is preserved at the cost of leaving some residual noise.
    """
    frame_len = 512
    hop_len = 256
    n_frames = (len(audio) - frame_len) // hop_len + 1
    if n_frames < 4:
        return audio  # too short to denoise meaningfully

    window = np.hanning(frame_len).astype(np.float32)

    # Build spectrogram
    frames = np.stack(
        [audio[i * hop_len: i * hop_len + frame_len] * window for i in range(n_frames)],
        axis=0,
    )  # (n_frames, frame_len)

    spectra = np.fft.rfft(frames, axis=1)  # (n_frames, frame_len//2 + 1)
    mag = np.abs(spectra)
    phase = np.angle(spectra)

    # Estimate noise floor from quietest frames
    frame_energy = mag.sum(axis=1)
    threshold_energy = np.quantile(frame_energy, noise_floor_quantile)
    noise_frames = frames[frame_energy <= threshold_energy]
    if len(noise_frames) == 0:
        return audio
    noise_mag = np.abs(np.fft.rfft(noise_frames, axis=1)).mean(axis=0)

    # Subtract noise magnitude with over-subtraction factor α=1.5 and spectral floor β=0.02
    alpha = 1.5
    beta = 0.02
    mag_clean = np.maximum(mag - alpha * noise_mag[np.newaxis, :], beta * mag)

    # Reconstruct
    spectra_clean = mag_clean * np.exp(1j * phase)
    frames_clean = np.fft.irfft(spectra_clean, n=frame_len, axis=1)  # (n_frames, frame_len)
    frames_clean = frames_clean * window  # re-apply window for OLA

    # Overlap-add
    out_len = hop_len * (n_frames - 1) + frame_len
    out = np.zeros(out_len, dtype=np.float32)
    for i in range(n_frames):
        out[i * hop_len: i * hop_len + frame_len] += frames_clean[i]

    # Trim to original length
    out = out[: len(audio)]

    logger.debug("[Preprocess] Spectral subtraction denoising applied")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: VAD-based speech region detection
# ─────────────────────────────────────────────────────────────────────────────

def _energy_vad(
    audio: np.ndarray,
    sr: int,
    frame_ms: int = FRAME_DURATION_MS,
    speech_quantile: float = _ENERGY_SPEECH_QUANTILE,
    min_speech_sec: float = _MIN_SPEECH_SEC,
    min_silence_sec: float = _MIN_SILENCE_SEC,
) -> List[Tuple[float, float]]:
    """
    Simple energy-based VAD.

    Labels each frame as speech if its RMS energy is above the
    `speech_quantile`-th percentile of all frame energies.
    Merges adjacent speech frames into regions, applying minimum duration
    filters to remove very short speech/silence bursts.

    Returns list of (start_sec, end_sec) speech regions.
    """
    frame_size = int(sr * frame_ms / 1000)
    if frame_size <= 0 or len(audio) < frame_size:
        # Audio too short for framing — treat entire audio as speech
        return [(0.0, len(audio) / sr)]

    n_frames = len(audio) // frame_size
    frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    if rms.max() < 1e-8:
        return []

    threshold = float(np.quantile(rms, speech_quantile))
    is_speech = rms > threshold

    # Build raw regions
    regions: List[Tuple[float, float]] = []
    in_speech = False
    start = 0
    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            start = i
            in_speech = True
        elif not speech and in_speech:
            regions.append((start * frame_ms / 1000, i * frame_ms / 1000))
            in_speech = False
    if in_speech:
        regions.append((start * frame_ms / 1000, n_frames * frame_ms / 1000))

    # Apply minimum duration filter
    regions = [
        (s, e) for s, e in regions if (e - s) >= min_speech_sec
    ]

    # Merge regions separated by less than min_silence_sec
    merged: List[Tuple[float, float]] = []
    for s, e in regions:
        if merged and (s - merged[-1][1]) < min_silence_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    logger.debug(f"[Preprocess] Energy VAD: found {len(merged)} speech regions")
    return merged


def _webrtcvad_regions(
    audio: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_ms: int = FRAME_DURATION_MS,
    min_speech_sec: float = _MIN_SPEECH_SEC,
    min_silence_sec: float = _MIN_SILENCE_SEC,
) -> List[Tuple[float, float]]:
    """
    webrtcvad-based VAD (used when webrtcvad is installed).

    aggressiveness: 0 (least aggressive) to 3 (most aggressive)
    """
    import webrtcvad  # type: ignore

    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sr * frame_ms / 1000)

    # webrtcvad requires 16-bit PCM bytes
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    n_frames = len(audio) // frame_size
    is_speech = []
    for i in range(n_frames):
        chunk = pcm[i * frame_size * 2: (i + 1) * frame_size * 2]
        try:
            is_speech.append(vad.is_speech(chunk, sr))
        except Exception:
            is_speech.append(False)

    # Convert frame labels to regions (same merging logic as energy VAD)
    regions: List[Tuple[float, float]] = []
    in_speech = False
    start = 0
    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            start = i
            in_speech = True
        elif not speech and in_speech:
            regions.append((start * frame_ms / 1000, i * frame_ms / 1000))
            in_speech = False
    if in_speech:
        regions.append((start * frame_ms / 1000, n_frames * frame_ms / 1000))

    regions = [(s, e) for s, e in regions if (e - s) >= min_speech_sec]
    merged: List[Tuple[float, float]] = []
    for s, e in regions:
        if merged and (s - merged[-1][1]) < min_silence_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    logger.debug(f"[Preprocess] webrtcvad VAD: found {len(merged)} speech regions")
    return merged


def detect_speech_regions(
    wav_path: str,
    aggressiveness: int = 2,
) -> List[Tuple[float, float]]:
    """
    Detect speech regions in a 16 kHz mono WAV file.

    Tries webrtcvad first (more accurate); falls back to energy-based VAD
    if webrtcvad is not installed or fails.

    Args:
        wav_path: Path to a 16 kHz mono WAV file.
        aggressiveness: webrtcvad aggressiveness 0–3 (only used if webrtcvad
                        is available).

    Returns:
        List of (start_sec, end_sec) tuples covering speech regions.
        Empty list if no speech is detected.
    """
    audio, sr = _load_wav_mono(wav_path)

    # Ensure 16 kHz (webrtcvad requirement)
    if sr != SAMPLE_RATE:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        except Exception:
            logger.warning("[Preprocess] Could not resample audio for VAD — using energy VAD directly")
            return _energy_vad(audio, sr)

    try:
        regions = _webrtcvad_regions(audio, sr, aggressiveness=aggressiveness)
        logger.info(f"[Preprocess] Speech detection (webrtcvad): {len(regions)} regions")
        return regions
    except ImportError:
        logger.debug("[Preprocess] webrtcvad not installed — using energy-based VAD")
    except Exception as e:
        logger.warning(f"[Preprocess] webrtcvad failed ({e}) — falling back to energy VAD")

    regions = _energy_vad(audio, sr)
    logger.info(f"[Preprocess] Speech detection (energy VAD): {len(regions)} regions")
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Main preprocessing entry point
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_audio_for_alignment(
    wav_path: str,
    trim_silence: bool = True,
    repair_clipping: bool = True,
    normalize_loudness: bool = True,
    denoise: bool = False,          # off by default — use only if audio is very noisy
    output_path: Optional[str] = None,
) -> str:
    """
    Apply optional audio cleanup steps before WhisperX forced alignment.

    The original WAV is never modified.  A preprocessed copy is written
    to `output_path` (or a temp file if not provided).

    Args:
        wav_path:          Path to the 16 kHz mono input WAV.
        trim_silence:      Trim leading/trailing silence.
        repair_clipping:   Fix soft-clipped samples.
        normalize_loudness: Peak-normalize to -3 dBFS.
        denoise:           Apply light spectral subtraction denoising.
        output_path:       Where to write the preprocessed WAV.
                           If None, a temp file is used (caller is responsible
                           for cleanup).

    Returns:
        Path to the preprocessed WAV file.
    """
    steps_applied = []

    try:
        audio, sr = _load_wav_mono(wav_path)
        original_len = len(audio)

        if trim_silence:
            audio = _trim_silence(audio, sr)
            if len(audio) != original_len:
                steps_applied.append("silence_trim")

        if repair_clipping:
            audio_prev = audio
            audio = _repair_clipping(audio)
            if not np.array_equal(audio, audio_prev):
                steps_applied.append("clip_repair")

        if normalize_loudness:
            audio_prev = audio
            audio = _normalize_loudness(audio)
            if not np.array_equal(audio, audio_prev):
                steps_applied.append("loudness_norm")

        if denoise:
            audio = _spectral_subtract_denoise(audio, sr)
            steps_applied.append("denoise")

    except Exception as e:
        logger.warning(
            f"[Preprocess] Audio preprocessing failed ({e}). "
            "Returning original WAV path unchanged."
        )
        return wav_path

    if not steps_applied:
        logger.info("[Preprocess] No preprocessing steps were needed — using original WAV")
        return wav_path

    # Write preprocessed audio to output path (or temp file)
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_preprocessed.wav")
        os.close(fd)

    try:
        _save_wav(audio, sr, output_path)
        logger.info(
            f"[Preprocess] Audio preprocessing complete. "
            f"Steps applied: {steps_applied}. "
            f"Output: {output_path}"
        )
        return output_path
    except Exception as e:
        logger.warning(f"[Preprocess] Failed to write preprocessed WAV ({e}). Using original.")
        return wav_path


def cleanup_temp_wav(path: str, original_path: str) -> None:
    """
    Delete a preprocessed temp WAV if it differs from the original path.
    Safe to call even if the file does not exist.
    """
    if path and path != original_path:
        try:
            os.unlink(path)
            logger.debug(f"[Preprocess] Deleted temp WAV: {path}")
        except OSError:
            pass
