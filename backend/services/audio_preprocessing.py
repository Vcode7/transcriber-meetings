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


def _apply_selective_loudness_enhancement(
    audio: np.ndarray,
    sr: int = 16000,
    target_dbfs: float = -18.0,
    max_gain_db: float = 14.0,
    noise_floor_db: float = -48.0,
    window_ms: float = 30.0,
    hop_ms: float = 15.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Selective Adaptive Loudness Enhancement.

    Analyzes audio using short overlapping windows (30ms frame, 15ms hop).
    Calculates RMS energy for each window.
    Only quiet speech windows (below target_dbfs and above noise_floor_db) receive additional gain.
    Windows already at or above target_dbfs remain COMPLETELY UNCHANGED (gain = 1.0, 0 dB boost).
    Near-silent / noise floor windows remain UNCHANGED (gain = 1.0) to prevent noise amplification.
    Smooths gain transitions between windows to prevent pumping / pops.
    Applies a peak limiter / clipping protection if necessary after amplification.
    """
    if len(audio) == 0:
        return audio, {
            "avg_input_dbfs": -100.0,
            "quiet_windows_pct": 0.0,
            "avg_gain_applied_db": 0.0,
            "max_gain_applied_db": 0.0,
            "limiter_activated": "No",
        }

    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio))
    if peak < 1e-6:
        return audio, {
            "avg_input_dbfs": -100.0,
            "quiet_windows_pct": 0.0,
            "avg_gain_applied_db": 0.0,
            "max_gain_applied_db": 0.0,
            "limiter_activated": "No",
        }

    frame_len = int(sr * (window_ms / 1000.0))
    hop_len = int(sr * (hop_ms / 1000.0))
    if frame_len <= 0 or len(audio) < frame_len:
        return audio, {
            "avg_input_dbfs": round(20 * np.log10(peak + 1e-7), 1),
            "quiet_windows_pct": 0.0,
            "avg_gain_applied_db": 0.0,
            "max_gain_applied_db": 0.0,
            "limiter_activated": "No",
        }

    n_frames = (len(audio) - frame_len) // hop_len + 1
    target_rms = 10 ** (target_dbfs / 20.0)
    noise_floor_rms = 10 ** (noise_floor_db / 20.0)
    max_gain_linear = 10 ** (max_gain_db / 20.0)

    window_centers = []
    window_gains_linear = []
    window_rms_list = []
    quiet_windows_count = 0

    for i in range(n_frames):
        start_i = i * hop_len
        end_i = start_i + frame_len
        w_audio = audio[start_i:end_i]
        w_rms = float(np.sqrt(np.mean(w_audio ** 2)))
        window_rms_list.append(w_rms)
        center_sample = start_i + frame_len // 2
        window_centers.append(center_sample)

        if w_rms >= target_rms or w_rms <= noise_floor_rms:
            # Already adequate speech volume OR background noise floor -> 0 dB gain boost
            gain = 1.0
        else:
            # Quiet speech window -> boost gain to bring RMS towards target_rms
            quiet_windows_count += 1
            needed_gain = target_rms / max(w_rms, 1e-7)
            gain = min(needed_gain, max_gain_linear)

        window_gains_linear.append(gain)

    avg_input_rms = float(np.mean(window_rms_list))
    avg_input_dbfs = 20 * np.log10(max(avg_input_rms, 1e-7))
    quiet_windows_pct = (quiet_windows_count / max(1, n_frames)) * 100.0

    # Smooth window gains to avoid pumping or audible jumps
    window_gains_linear = np.array(window_gains_linear, dtype=np.float32)
    smooth_kernel_size = 5
    if len(window_gains_linear) >= smooth_kernel_size:
        kernel = np.ones(smooth_kernel_size, dtype=np.float32) / smooth_kernel_size
        window_gains_linear = np.convolve(window_gains_linear, kernel, mode="same")

    sample_indices = np.arange(len(audio))
    sample_gains = np.interp(
        sample_indices,
        np.array(window_centers, dtype=np.float32),
        window_gains_linear,
        left=window_gains_linear[0],
        right=window_gains_linear[-1]
    ).astype(np.float32)

    enhanced = audio * sample_gains

    # Peak Limiter / Soft Knee Clipping Protection
    max_peak = float(np.max(np.abs(enhanced)))
    limiter_activated = False
    if max_peak > 0.95:
        limiter_activated = True
        enhanced = np.clip(enhanced, -0.95, 0.95)

    gains_db = 20 * np.log10(np.maximum(sample_gains, 1.0))
    avg_gain_applied_db = float(np.mean(gains_db))
    max_gain_applied_db = float(np.max(gains_db))

    stats = {
        "avg_input_dbfs": round(avg_input_dbfs, 1),
        "quiet_windows_pct": round(quiet_windows_pct, 1),
        "avg_gain_applied_db": round(avg_gain_applied_db, 1),
        "max_gain_applied_db": round(max_gain_applied_db, 1),
        "limiter_activated": "Yes" if limiter_activated else "No",
    }

    logger.info(
        f"[SelectiveAudioEnhancement] "
        f"AvgInputLoudness={stats['avg_input_dbfs']} dBFS, "
        f"QuietWindows={stats['quiet_windows_pct']}%, "
        f"AvgGainApplied=+{stats['avg_gain_applied_db']} dB, "
        f"MaxGainApplied=+{stats['max_gain_applied_db']} dB, "
        f"PeakLimiterActivated={stats['limiter_activated']}"
    )

    return enhanced, stats


def _normalize_loudness(audio: np.ndarray, target_dbfs: float = _PEAK_TARGET_DBFS) -> np.ndarray:
    """Legacy loudness normalization alias using selective adaptive enhancement."""
    enhanced, _ = _apply_selective_loudness_enhancement(audio, target_dbfs=target_dbfs)
    return enhanced


def _apply_dynamic_range_compression(
    audio: np.ndarray,
    ratio: float = 2.0,
    threshold_db: float = -24.0,
) -> np.ndarray:
    """Legacy dynamic compression alias using selective adaptive enhancement."""
    enhanced, _ = _apply_selective_loudness_enhancement(audio, target_dbfs=-18.0)
    return enhanced



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
    speech_quantile: float = _ENERGY_SPEECH_QUANTILE,
    min_speech_sec: float = _MIN_SPEECH_SEC,
    min_silence_sec: float = _MIN_SILENCE_SEC,
) -> List[Tuple[float, float]]:
    """
    Detect speech regions in a 16 kHz mono WAV file.

    Tries webrtcvad first (more accurate); falls back to energy-based VAD
    if webrtcvad is not installed or fails.
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
            return _energy_vad(audio, sr, speech_quantile=speech_quantile, min_speech_sec=min_speech_sec, min_silence_sec=min_silence_sec)

    try:
        regions = _webrtcvad_regions(
            audio, sr, aggressiveness=aggressiveness,
            min_speech_sec=min_speech_sec, min_silence_sec=min_silence_sec
        )
        logger.info(f"[Preprocess] Speech detection (webrtcvad, agg={aggressiveness}): {len(regions)} regions")
        return regions
    except ImportError:
        logger.debug("[Preprocess] webrtcvad not installed — using energy-based VAD")
    except Exception as e:
        logger.warning(f"[Preprocess] webrtcvad failed ({e}) — falling back to energy VAD")

    regions = _energy_vad(
        audio, sr, speech_quantile=speech_quantile,
        min_speech_sec=min_speech_sec, min_silence_sec=min_silence_sec
    )
    logger.info(f"[Preprocess] Speech detection (energy VAD): {len(regions)} regions")
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# Speech Padding & Segment Merging Helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_speech_padding(
    regions: List[Tuple[float, float]],
    pad_ms: float,
    total_duration_sec: float,
) -> List[Tuple[float, float]]:
    """
    Add pre- and post-segment padding (in ms) to detected speech regions,
    clamping to [0.0, total_duration_sec] and merging overlapping regions.
    """
    if not regions or pad_ms <= 0:
        return regions

    pad_sec = pad_ms / 1000.0
    padded: List[Tuple[float, float]] = []
    for start, end in regions:
        p_start = max(0.0, start - pad_sec)
        p_end = min(total_duration_sec, end + pad_sec)
        padded.append((round(p_start, 3), round(p_end, 3)))

    # Merge overlapping regions resulting from padding
    merged: List[Tuple[float, float]] = []
    for s, e in padded:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged


def merge_speech_segments(
    regions: List[Tuple[float, float]],
    max_merge_silence_ms: float,
) -> List[Tuple[float, float]]:
    """
    Merge nearby speech segments separated by short silence <= max_merge_silence_ms.
    """
    if not regions or max_merge_silence_ms <= 0:
        return regions

    max_silence_sec = max_merge_silence_ms / 1000.0
    merged: List[Tuple[float, float]] = []
    for s, e in regions:
        if merged and (s - merged[-1][1]) <= max_silence_sec:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged


def detect_rejected_low_volume_regions(
    audio: np.ndarray,
    sr: int,
    speech_regions: List[Tuple[float, float]],
    energy_threshold_db: float = -45.0,
    min_duration_sec: float = 2.0,
) -> List[Tuple[float, float]]:
    """
    Analyze rejected (untranscribed/non-speech) audio gaps between speech spans.
    Only gaps with duration >= min_duration_sec (default 2.0s) and audio energy >= energy_threshold_db
    are selected for missing segment recovery. Shorter segments are ignored.
    """
    min_duration_sec = max(0.05, float(min_duration_sec))
    total_sec = len(audio) / sr
    if total_sec <= 0:
        return []

    sorted_speech = sorted(speech_regions, key=lambda x: x[0])
    rejected_spans: List[Tuple[float, float]] = []
    ignored_short_spans: int = 0

    curr = 0.0
    for s, e in sorted_speech:
        gap_dur = s - curr
        if gap_dur >= min_duration_sec:
            rejected_spans.append((curr, s))
        elif gap_dur > 0:
            ignored_short_spans += 1
            logger.debug(
                f"[MissingSegmentRecovery] Ignoring missing span [{curr:.2f}s–{s:.2f}s] "
                f"(duration {gap_dur:.2f}s < threshold {min_duration_sec:.2f}s)"
            )
        curr = max(curr, e)

    tail_gap = total_sec - curr
    if tail_gap >= min_duration_sec:
        rejected_spans.append((curr, total_sec))
    elif tail_gap > 0:
        ignored_short_spans += 1
        logger.debug(
            f"[MissingSegmentRecovery] Ignoring trailing missing span [{curr:.2f}s–{total_sec:.2f}s] "
            f"(duration {tail_gap:.2f}s < threshold {min_duration_sec:.2f}s)"
        )

    logger.info(
        f"[MissingSegmentRecovery] Gap analysis: total_audio={total_sec:.2f}s, "
        f"speech_spans={len(sorted_speech)}, min_segment_threshold={min_duration_sec:.2f}s, "
        f"candidate_gaps={len(rejected_spans)}, ignored_short_gaps={ignored_short_spans}"
    )

    recovered_spans: List[Tuple[float, float]] = []
    threshold_linear = 10 ** (energy_threshold_db / 20.0)

    for r_start, r_end in rejected_spans:
        span_dur = r_end - r_start
        if span_dur < min_duration_sec:
            continue

        s_idx = int(r_start * sr)
        e_idx = int(r_end * sr)
        chunk = audio[s_idx:e_idx]
        if len(chunk) < int(sr * min_duration_sec):
            continue

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        rms_db = 20 * np.log10(rms + 1e-9)

        if rms >= threshold_linear:
            recovered_spans.append((round(r_start, 3), round(r_end, 3)))
            logger.info(
                f"[MissingSegmentRecovery] Candidate span [{r_start:.2f}s–{r_end:.2f}s] "
                f"({span_dur:.2f}s >= {min_duration_sec:.2f}s) passed energy check: "
                f"RMS={rms_db:.1f} dBFS >= {energy_threshold_db:.1f} dBFS → queued for recovery"
            )
        else:
            logger.debug(
                f"[MissingSegmentRecovery] Candidate span [{r_start:.2f}s–{r_end:.2f}s] "
                f"({span_dur:.2f}s) below energy threshold (RMS={rms_db:.1f} dBFS < {energy_threshold_db:.1f} dBFS) → skipped"
            )

    logger.info(
        f"[MissingSegmentRecovery] Evaluation complete: {len(recovered_spans)} missing segment(s) "
        f">= {min_duration_sec:.2f}s qualified for secondary Whisper transcription"
    )
    return recovered_spans


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Main preprocessing entry point
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_audio_for_alignment(
    wav_path: str,
    trim_silence: bool = True,
    repair_clipping: bool = True,
    normalize_loudness: bool = True,
    compress_dynamic_range: bool = True,
    target_dbfs: float = _PEAK_TARGET_DBFS,
    compression_ratio: float = 2.0,
    denoise: bool = False,
    output_path: Optional[str] = None,
) -> str:
    """
    Apply optional audio cleanup steps before WhisperX forced alignment.

    The original WAV is never modified. A preprocessed copy is written
    to `output_path` (or a temp file if not provided).
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

        if normalize_loudness or compress_dynamic_range:
            audio_prev = audio
            audio, stats = _apply_selective_loudness_enhancement(audio, sr=sr, target_dbfs=-18.0)
            if not np.array_equal(audio, audio_prev):
                steps_applied.append(f"selective_adaptive_loudness_enhancement(avg_boost={stats['avg_gain_applied_db']}dB, quiet={stats['quiet_windows_pct']}%)")

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
