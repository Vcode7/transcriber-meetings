"""
Speaker embedding service using SpeechBrain ECAPA-TDNN.

Replaces the previous WeSpeaker CAM++ ONNX implementation.

Model used
----------
SpeechBrain ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation
Time Delay Neural Network)
- Loaded via speechbrain.inference.speaker.EncoderClassifier — fully offline
- Input:  mono 16 kHz waveform (any length ≥ 1 s)
- Output: 192-d L2-normalised speaker embedding
- Model path: MODELS_DIR/ecapa_tdnn/

Preprocessing pipeline
-----------------------
SpeechBrain's EncoderClassifier handles all internal preprocessing:
  - 80-channel Mel filterbank (SpeechBrain default)
  - Mean-variance normalisation
  - ECAPA-TDNN forward pass
  - Embedding extraction (attentive statistics pooling)

This service only performs:
  - Mono mixdown + resample to 16 kHz (before calling encode_batch)
  - VAD-based silence removal (for diarized segments)
  - Windowed averaging for long segments (> 8 s of speech)
  - L2 normalisation of the final embedding

Dimension note
--------------
ECAPA-TDNN produces 192-dimensional embeddings.
Previously enrolled voice profiles with 512-d CAM++ embeddings
are incompatible and will be skipped with a one-time warning.
Users must re-enroll (re-record) all voice profiles after upgrading.

Threshold note
--------------
ECAPA-TDNN cosine similarity scores differ from CAM++ scores.
The default threshold of 0.75 is a reasonable starting point.
See config.SPEAKER_SIMILARITY_THRESHOLD_ECAPA_TDNN.
"""
from __future__ import annotations

import logging
import os
import numpy as np
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Embedding dimension for SpeechBrain ECAPA-TDNN
EMBEDDING_DIM: int = 192
SAMPLE_RATE: int = 16000

# Module-level classifier handle (lazy-loaded)
_ecapa_classifier = None

# Shared device selection
try:
    from services.device_utils import DEVICE as _DEVICE
except Exception:
    _DEVICE = "cpu"


# ── Encoder management ────────────────────────────────────────────────────────

def _ensure_hf_cache(model_dir) -> None:
    """
    Seed the HuggingFace hub cache from the local ecapa_tdnn/ directory.

    SpeechBrain's from_hparams() always fetches files via the HF hub client,
    which attempts to create symlinks on Windows (requires elevated privileges).
    This function pre-populates the HF cache with hard copies so the hub client
    finds the files already in place and skips the symlink/download step.

    This only runs once (idempotent — skips files that are already in cache).
    """
    import shutil
    from pathlib import Path

    # Resolve the HF cache path for this model
    hf_home = Path(os.environ.get(
        "HF_HOME",
        str(Path.home() / ".cache" / "huggingface" / "hub")
    ))
    repo_cache = hf_home / "models--speechbrain--spkrec-ecapa-voxceleb"
    # Find an existing snapshot hash if present, or create a new one
    snapshot_hash = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
    snapshot_dir = repo_cache / "snapshots" / snapshot_hash
    refs_dir = repo_cache / "refs"

    # Write the ref pointer so HF hub resolves "main" → our hash
    refs_dir.mkdir(parents=True, exist_ok=True)
    ref_file = refs_dir / "main"
    if not ref_file.exists():
        ref_file.write_text(snapshot_hash, encoding="utf-8")

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Copy any files from our local model dir that are missing in the cache
    for src_file in model_dir.iterdir():
        if src_file.is_file():
            dst_file = snapshot_dir / src_file.name
            if not dst_file.exists():
                try:
                    shutil.copy2(str(src_file), str(dst_file))
                    logger.debug(f"[Embedding] Seeded HF cache: {src_file.name}")
                except Exception as e:
                    logger.debug(f"[Embedding] Cache seed skipped {src_file.name}: {e}")
    logger.debug(f"[Embedding] HF cache seeded at {snapshot_dir}")


def get_encoder():
    """
    Lazy-load the SpeechBrain ECAPA-TDNN EncoderClassifier.

    Loads the model fully offline from MODELS_DIR/ecapa_tdnn/.
    Raises RuntimeError if the model directory or required files are missing.
    """
    global _ecapa_classifier
    if _ecapa_classifier is not None:
        return _ecapa_classifier

    # Ensure SpeechBrain k2/flair mocks and torchaudio patches are in place
    # before importing EncoderClassifier (importing speechbrain triggers the
    # lazy-import machinery that tries to load the optional k2 package).
    from services.compat import apply_compatibility_patches
    apply_compatibility_patches()

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        raise ImportError(
            "speechbrain is required for ECAPA-TDNN embedding. "
            "Install it with: pip install speechbrain>=1.0.0"
        )

    from services.model_loader import ModelLoader
    model_dir = ModelLoader.get_model_path("ecapa_tdnn")
    if model_dir is None or not model_dir.exists():
        raise RuntimeError(
            "[Embedding] ECAPA-TDNN model directory not found (ecapa_tdnn/).\n"
            "Run tools/download_speaker_models.py to download it."
        )

    required_files = ["hyperparams.yaml", "embedding_model.ckpt"]
    missing = [f for f in required_files if not (model_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"[Embedding] ECAPA-TDNN model is incomplete. "
            f"Missing files: {missing}\n"
            "Run tools/download_speaker_models.py to re-download."
        )

    # Seed HF hub cache from local model dir so SpeechBrain never needs
    # to create symlinks (which require elevated privileges on Windows).
    _ensure_hf_cache(model_dir)

    # Determine run device
    run_device = "cuda" if _DEVICE == "cuda" else "cpu"

    try:
        from services.device_utils import log_gpu_memory
        log_gpu_memory("Pre-load ECAPA-TDNN")
        from speechbrain.utils.fetching import LocalStrategy, FetchConfig

        model_dir_str = str(model_dir)
        _ecapa_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=model_dir_str,
            run_opts={"device": run_device},
            # Use COPY instead of SYMLINK — symlinks require elevated privileges
            # on Windows and are not available in most deployment environments.
            local_strategy=LocalStrategy.COPY,
            # Enforce fully offline inference — no network calls at runtime.
            fetch_config=FetchConfig(allow_network=False),
        )
        logger.info(
            f"[Embedding] SpeechBrain ECAPA-TDNN loaded from {model_dir} "
            f"(device={run_device})"
        )
        log_gpu_memory("Post-load ECAPA-TDNN")
    except Exception as e:
        raise RuntimeError(
            f"[Embedding] Failed to load ECAPA-TDNN model: {e}"
        ) from e


    return _ecapa_classifier


def unload_encoder():
    """Unload the ECAPA-TDNN classifier to free RAM/VRAM."""
    global _ecapa_classifier
    if _ecapa_classifier is not None:
        from services.device_utils import log_gpu_memory
        log_gpu_memory("Pre-unload ECAPA-TDNN")
        logger.info("[Embedding] Unloading SpeechBrain ECAPA-TDNN classifier...")
        del _ecapa_classifier
        _ecapa_classifier = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[Embedding] ECAPA-TDNN classifier unloaded.")
        log_gpu_memory("Post-unload ECAPA-TDNN")



def get_embedding_dim() -> int:
    """Return the dimension of embeddings produced by the current model."""
    return EMBEDDING_DIM


# ── Audio loading helpers ─────────────────────────────────────────────────────

def _load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load mono float32 audio at the target sample rate using torchaudio."""
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(file_path)

    # Mix down to mono if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # Return as 1-D numpy float32 array
    audio = waveform.squeeze(0).numpy().astype(np.float32)
    return audio, sr


def _resample_audio(
    audio: np.ndarray,
    source_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Resample a numpy float32 waveform to target_sr."""
    if source_sr == target_sr:
        return audio
    import torch
    import torchaudio

    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)
    resampled = resampler(waveform).squeeze(0).numpy()
    return resampled.astype(np.float32)


# ── Core embedding extraction ─────────────────────────────────────────────────

def extract_embedding(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    Extract a 192-d L2-normalised speaker embedding from mono audio.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 waveform.
    sr : int
        Sample rate of `audio` (will resample to 16 kHz if needed).

    Returns
    -------
    np.ndarray of shape (192,), or None on failure.
    """
    try:
        import torch
        classifier = get_encoder()

        # Resample to 16 kHz if necessary
        wav = _resample_audio(audio, source_sr=sr, target_sr=SAMPLE_RATE)
        wav = np.asarray(wav, dtype=np.float32)

        # Need at least 1 second of audio for a meaningful embedding
        # if len(wav) < SAMPLE_RATE * 1.0:
        #     return None

        # ECAPA-TDNN expects (batch, samples) torch tensor at 16 kHz
        waveform = torch.from_numpy(wav).unsqueeze(0)  # (1, N)

        with torch.no_grad():
            # encode_batch returns (batch, 1, embedding_dim)
            embedding_tensor = classifier.encode_batch(waveform)

        embedding = embedding_tensor.squeeze().cpu().numpy().astype(np.float32)

        # L2-normalise
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    except Exception as e:
        logger.error(f"[Embedding] extract_embedding failed: {e}")
        return None


def extract_embedding_from_file(file_path: str) -> Optional[np.ndarray]:
    """Load a file and extract a 192-d ECAPA-TDNN speaker embedding."""
    try:
        audio, sr = _load_audio(file_path, target_sr=SAMPLE_RATE)
        return extract_embedding(audio, sr=sr)
    except Exception as e:
        logger.error(f"[Embedding] extract_embedding_from_file failed: {e}")
        return None


# ── VAD-gated embedding helpers ───────────────────────────────────────────────

_ENERGY_VAD_FRAME_MS: int = 30          # ms per energy-VAD analysis frame
_ENERGY_SPEECH_QUANTILE: float = 0.25  # frames above this energy quantile = speech
_MIN_SPEECH_REGION_SEC: float = 0.25   # discard VAD regions shorter than this
_MIN_SILENCE_MERGE_SEC: float = 0.4    # merge adjacent regions separated by less


def _vad_speech_regions(
    audio: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
) -> List[Tuple[float, float]]:
    """
    Detect voiced speech regions in an in-memory waveform.

    Tries webrtcvad (aggressiveness 0-3) first.  Falls back to a simple
    energy-based VAD if webrtcvad is not installed or raises.

    Parameters
    ----------
    audio          : mono float32 waveform at ``sr`` Hz.
    sr             : sample rate (must be 8000, 16000, 32000, or 48000 for
                     webrtcvad; energy VAD works at any rate).
    aggressiveness : webrtcvad aggressiveness (0 = least, 3 = most).

    Returns
    -------
    List of (start_sec, end_sec) speech regions, merged and filtered.
    """
    frame_size = int(sr * _ENERGY_VAD_FRAME_MS / 1000)

    def _build_regions_from_mask(
        is_speech_mask: List[bool],
    ) -> List[Tuple[float, float]]:
        """Convert a per-frame boolean mask to merged (start, end) regions."""
        regions: List[Tuple[float, float]] = []
        in_speech = False
        seg_start = 0
        for i, s in enumerate(is_speech_mask):
            if s and not in_speech:
                seg_start = i
                in_speech = True
            elif not s and in_speech:
                regions.append((
                    seg_start * _ENERGY_VAD_FRAME_MS / 1000,
                    i * _ENERGY_VAD_FRAME_MS / 1000,
                ))
                in_speech = False
        if in_speech:
            regions.append((
                seg_start * _ENERGY_VAD_FRAME_MS / 1000,
                len(is_speech_mask) * _ENERGY_VAD_FRAME_MS / 1000,
            ))

        # Filter out very short regions
        regions = [
            (s, e) for s, e in regions if (e - s) >= _MIN_SPEECH_REGION_SEC
        ]
        # Merge regions separated by short silences
        merged: List[Tuple[float, float]] = []
        for s, e in regions:
            if merged and (s - merged[-1][1]) < _MIN_SILENCE_MERGE_SEC:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        return merged

    # ── Try webrtcvad ─────────────────────────────────────────────────────────
    try:
        import webrtcvad  # type: ignore
        vad = webrtcvad.Vad(aggressiveness)
        n_frames = len(audio) // frame_size
        if n_frames == 0:
            return [(0.0, len(audio) / sr)]
        # webrtcvad expects 16-bit PCM bytes
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        mask = []
        for i in range(n_frames):
            chunk = pcm[i * frame_size * 2: (i + 1) * frame_size * 2]
            try:
                mask.append(vad.is_speech(chunk, sr))
            except Exception:
                mask.append(False)
        regions = _build_regions_from_mask(mask)
        logger.debug(
            f"[Embedding] webrtcvad: {len(regions)} speech regions "
            f"out of {len(mask)} frames"
        )
        return regions
    except ImportError:
        pass  # webrtcvad not installed — fall through to energy VAD
    except Exception as e:
        logger.debug(f"[Embedding] webrtcvad failed ({e}) — using energy VAD")

    # ── Energy VAD fallback ───────────────────────────────────────────────────
    n_frames = len(audio) // frame_size
    if n_frames == 0:
        return [(0.0, len(audio) / sr)]
    frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    if rms.max() < 1e-8:
        return []  # essentially silent
    threshold = float(np.quantile(rms, _ENERGY_SPEECH_QUANTILE))
    mask_e = (rms > threshold).tolist()
    regions = _build_regions_from_mask(mask_e)
    logger.debug(
        f"[Embedding] energy VAD: {len(regions)} speech regions "
        f"out of {n_frames} frames"
    )
    return regions


def _merge_speech_regions(
    audio: np.ndarray,
    sr: int,
    regions: List[Tuple[float, float]],
) -> Optional[np.ndarray]:
    """
    Concatenate sample slices for each (start_sec, end_sec) region.

    Returns a single contiguous float32 waveform, or None if `regions` is
    empty or all slices are empty.
    """
    if not regions:
        return None
    chunks = []
    for start_sec, end_sec in regions:
        s_idx = max(0, int(start_sec * sr))
        e_idx = min(len(audio), int(end_sec * sr))
        if e_idx > s_idx:
            chunks.append(audio[s_idx:e_idx])
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _window_audio(
    audio: np.ndarray,
    sr: int,
    win_sec: float = 4.0,
    hop_sec: float = 3.0,
) -> List[np.ndarray]:
    """
    Split `audio` into overlapping windows of `win_sec` seconds.

    Windows step by `hop_sec` (= win_sec - overlap).  With the defaults
    win_sec=4 and hop_sec=3, consecutive windows overlap by 1 second.

    If the audio is shorter than `win_sec`, returns ``[audio]`` unchanged so
    that the caller always receives at least one window.
    """
    win_samples = int(win_sec * sr)
    hop_samples = int(hop_sec * sr)
    n = len(audio)
    if n <= win_samples:
        return [audio]
    windows: List[np.ndarray] = []
    start = 0
    while start < n:
        end = start + win_samples
        windows.append(audio[start:min(end, n)])
        if end >= n:
            break
        start += hop_samples
    return windows


def vad_extract_speaker_embedding(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    min_speech_sec: float = 2.5,
    max_speech_sec: float = 8.0,
    win_sec: float = 4.0,
    hop_sec: float = 3.0,
) -> Optional[np.ndarray]:
    """
    VAD-gated ECAPA-TDNN speaker embedding extraction.

    Removes silence from the diarized segment before feeding audio to the
    ECAPA-TDNN model.  Long segments are split into overlapping windows and
    their embeddings are averaged, yielding a more robust speaker representation.

    Pipeline
    --------
    1.  Resample `audio` to 16 kHz if needed.
    2.  Run VAD  →  list of (start_sec, end_sec) speech regions.
    3.  Concatenate all speech regions into one contiguous waveform.
    4.  If total speech < ``min_speech_sec``  →  return None.
    5.  If total speech ≤ ``max_speech_sec``  →  ``extract_embedding()`` on
        the whole merged waveform.
    6.  If total speech > ``max_speech_sec``  →  slide overlapping windows
        of ``win_sec`` with hop ``hop_sec``, extract one embedding per window,
        average all window embeddings with ``average_embeddings()``, and
        L2-normalise the result.

    Parameters
    ----------
    audio          : mono float32 waveform.
    sr             : sample rate of `audio` (will be resampled to 16 kHz).
    min_speech_sec : minimum merged speech duration to attempt identification.
    max_speech_sec : above this duration, windowed averaging is used.
    win_sec        : window length in seconds (for windowed mode).
    hop_sec        : window hop in seconds  (win_sec - hop_sec = overlap).

    Returns
    -------
    np.ndarray of shape (192,), L2-normalised, or None if speech is
    insufficient or extraction fails.
    """
    try:
        # Step 0 — resample to SAMPLE_RATE (16 kHz) if needed
        wav = _resample_audio(audio, source_sr=sr, target_sr=SAMPLE_RATE)
        wav = np.asarray(wav, dtype=np.float32)
        working_sr = SAMPLE_RATE

        # Step 1 — run VAD
        regions = _vad_speech_regions(wav, working_sr)
        total_speech_sec = sum(e - s for s, e in regions)
        logger.debug(
            f"[Embedding] VAD: {len(regions)} speech regions, "
            f"total={total_speech_sec:.2f}s (raw segment={len(wav)/working_sr:.2f}s)"
        )

        # Step 2 — merge speech regions into one waveform
        speech_audio = _merge_speech_regions(wav, working_sr, regions)

        if speech_audio is None or len(speech_audio) == 0:
            logger.debug("[Embedding] VAD: no speech detected — skipping segment")
            return None

        actual_speech_sec = len(speech_audio) / working_sr

        # Step 3 — too short to identify reliably
        if actual_speech_sec < min_speech_sec:
            logger.debug(
                f"[Embedding] VAD: merged speech {actual_speech_sec:.2f}s "
                f"< min_speech_sec ({min_speech_sec}s) — skipping"
            )
            return None

        # Step 4 — short enough to embed in one shot
        if actual_speech_sec <= max_speech_sec:
            emb = extract_embedding(speech_audio, sr=working_sr)
            if emb is not None:
                logger.debug(
                    f"[Embedding] VAD single-pass embedding: "
                    f"speech={actual_speech_sec:.2f}s"
                )
            return emb

        # Step 5 — long segment: overlapping windows
        windows = _window_audio(speech_audio, working_sr, win_sec=win_sec, hop_sec=hop_sec)
        window_embeddings = []
        for i, win in enumerate(windows):
            emb_w = extract_embedding(win, sr=working_sr)
            if emb_w is not None:
                window_embeddings.append(emb_w)
        if not window_embeddings:
            logger.debug("[Embedding] VAD windowed: no valid window embeddings")
            return None
        averaged = average_embeddings(window_embeddings)  # already L2-normalised
        logger.debug(
            f"[Embedding] VAD windowed embedding: "
            f"{len(window_embeddings)}/{len(windows)} windows, "
            f"speech={actual_speech_sec:.2f}s"
        )
        return averaged

    except Exception as exc:
        # Safety net — fall back to raw segment embedding so identification
        # still works even if VAD raises an unexpected error.
        logger.warning(
            f"[Embedding] vad_extract_speaker_embedding failed ({exc}); "
            "falling back to extract_embedding on raw segment"
        )
        try:
            return extract_embedding(audio, sr=sr)
        except Exception:
            return None


# ── Similarity helpers (dimension-agnostic) ───────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def best_match_similarity(query: np.ndarray, stored: List[List[float]]) -> float:
    """
    Given a query embedding and a list of stored embeddings,
    return the highest cosine similarity score.
    """
    if not stored:
        return 0.0
    sims = [cosine_similarity(query, np.array(e)) for e in stored]
    return max(sims)


def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """Average multiple embeddings into one representative L2-normalised vector."""
    arr = np.stack(embeddings, axis=0)
    avg = np.mean(arr, axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg
