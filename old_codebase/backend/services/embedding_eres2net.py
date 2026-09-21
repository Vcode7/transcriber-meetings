"""
Speaker embedding service using ERes2Net-Large (3D-Speaker).

Model used
----------
ERes2Net-Large (Emphasized Residual Network with Local & Global Feature Fusion)
- Loaded fully offline from MODELS_DIR/eres2net_large/
- Input:  mono 16 kHz waveform
- Output: L2-normalised speaker embedding (default 512-d)
- Model path: MODELS_DIR/eres2net_large/

This module mirrors the structure of embedding.py (SpeechBrain ECAPA-TDNN)
and follows the exact same model-loading architecture:
- Loads directly from local model files via ModelLoader
- Never initiates network calls, Hugging Face downloads, or external checks at runtime
- Raises clean, descriptive error messages if model files are missing
- Lazy-loads upon first inference request and unloads to free VRAM immediately
"""
from __future__ import annotations

import json
import logging
import os
import numpy as np
import torch
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default embedding dimension for ERes2Net-Large (dynamically updated if model specifies otherwise)
EMBEDDING_DIM: int = 512
SAMPLE_RATE: int = 16000

# Module-level model handle (lazy-loaded)
_eres2net_model = None

# Shared device selection
try:
    from services.device_utils import DEVICE as _DEVICE
except Exception:
    _DEVICE = "cpu"


# ── Encoder management ────────────────────────────────────────────────────────

def get_encoder():
    """
    Lazy-load the ERes2Net-Large speaker verification model.

    Loads the model fully offline from MODELS_DIR/eres2net_large/.
    Raises RuntimeError if the model directory or required files are missing.
    """
    global _eres2net_model, EMBEDDING_DIM
    if _eres2net_model is not None:
        return _eres2net_model

    from services.model_loader import ModelLoader
    model_dir = ModelLoader.get_model_path("eres2net_large")
    if model_dir is None or not model_dir.exists():
        raise RuntimeError(
            "[Embedding] ERes2Net-Large model directory not found (eres2net_large/).\n"
            "Run tools/download_speaker_models.py --with-eres2net to download it."
        )

    # Check for model weights file
    candidates = [
        "eres2net_large_model.ckpt",
        "model.pt",
        "eres2net_large.pt",
        "pytorch_model.bin",
        "model.bin",
    ]
    ckpt_name = None
    for c in candidates:
        if (model_dir / c).exists():
            ckpt_name = c
            break

    if ckpt_name is None:
        raise RuntimeError(
            f"[Embedding] ERes2Net-Large model is incomplete in {model_dir}.\n"
            "Missing weights file (expected eres2net_large_model.ckpt or model.pt).\n"
            "Run tools/download_speaker_models.py --with-eres2net to re-download."
        )

    # Inspect configuration.json if present
    embed_dim = 512
    channels = 64
    config_file = model_dir / "configuration.json"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            mcfg = cfg.get("model", {}).get("model_config", {})
            embed_dim = mcfg.get("embed_dim", embed_dim)
            channels = mcfg.get("channels", channels)
        except Exception as e:
            logger.warning(f"[ERes2Net] Could not parse configuration.json: {e}")

    # Determine run device
    run_device = "cuda" if _DEVICE == "cuda" and torch.cuda.is_available() else "cpu"

    try:
        from services.device_utils import log_gpu_memory
        log_gpu_memory("Pre-load ERes2Net-Large")

        from services.eres2net_arch import OfflineERes2NetEncoder
        _eres2net_model = OfflineERes2NetEncoder(
            model_dir=str(model_dir),
            checkpoint_name=ckpt_name,
            embed_dim=embed_dim,
            channels=channels,
            device=run_device,
        )
        EMBEDDING_DIM = _eres2net_model.embed_dim

        logger.info(
            f"[Embedding] 3D-Speaker ERes2Net-Large loaded from {model_dir} "
            f"(device={run_device}, dim={EMBEDDING_DIM})"
        )
        log_gpu_memory("Post-load ERes2Net-Large")
    except Exception as e:
        raise RuntimeError(
            f"[Embedding] Failed to load ERes2Net-Large model: {e}"
        ) from e

    return _eres2net_model


def unload_encoder():
    """Unload the ERes2Net-Large model to free RAM/VRAM."""
    global _eres2net_model
    if _eres2net_model is not None:
        from services.device_utils import log_gpu_memory
        log_gpu_memory("Pre-unload ERes2Net-Large")
        logger.info("[Embedding] Unloading ERes2Net-Large model...")
        del _eres2net_model
        _eres2net_model = None
        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[Embedding] ERes2Net-Large model unloaded.")
        log_gpu_memory("Post-unload ERes2Net-Large")


def get_embedding_dim() -> int:
    """Return the dimension of embeddings produced by the ERes2Net-Large model."""
    global _eres2net_model, EMBEDDING_DIM
    if _eres2net_model is not None and hasattr(_eres2net_model, "embed_dim"):
        return _eres2net_model.embed_dim
    return EMBEDDING_DIM


# ── Audio loading helpers ─────────────────────────────────────────────────────

def _load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> tuple:
    """Load mono float32 audio at the target sample rate using torchaudio."""
    from services.embedding import _load_audio as _ecapa_load_audio
    return _ecapa_load_audio(file_path, target_sr)


def _resample_audio(
    audio: np.ndarray,
    source_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Resample a numpy float32 waveform to target_sr."""
    from services.embedding import _resample_audio as _ecapa_resample
    return _ecapa_resample(audio, source_sr, target_sr)


# ── Core embedding extraction ─────────────────────────────────────────────────

def extract_embedding(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    Extract an L2-normalised speaker embedding from mono audio.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 waveform.
    sr : int
        Sample rate of `audio` (will resample to 16 kHz if needed).

    Returns
    -------
    np.ndarray of shape (EMBEDDING_DIM,), or None on failure.
    """
    try:
        model = get_encoder()

        # Resample to 16 kHz if necessary
        wav = _resample_audio(audio, source_sr=sr, target_sr=SAMPLE_RATE)
        wav = np.asarray(wav, dtype=np.float32)

        if len(wav) < int(SAMPLE_RATE * 0.2):
            logger.warning("[ERes2Net] Audio segment too short (< 0.2s) for embedding extraction.")
            return None

        # Run direct in-memory offline inference
        embedding = model(wav)

        if embedding is None or len(embedding) == 0:
            logger.error("[ERes2Net] Model returned empty embedding")
            return None

        return embedding.astype(np.float32)

    except Exception as e:
        logger.error(f"[ERes2Net] extract_embedding failed: {e}")
        return None


def extract_embedding_from_file(file_path: str) -> Optional[np.ndarray]:
    """Load a file and extract an ERes2Net speaker embedding."""
    try:
        audio, sr = _load_audio(file_path, target_sr=SAMPLE_RATE)
        return extract_embedding(audio, sr=sr)
    except Exception as e:
        logger.error(f"[ERes2Net] extract_embedding_from_file failed: {e}")
        return None


# ── VAD-gated embedding helpers ───────────────────────────────────────────────

def vad_extract_speaker_embedding(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    min_speech_sec: float = 2.5,
    max_speech_sec: float = 8.0,
    win_sec: float = 4.0,
    hop_sec: float = 3.0,
) -> Optional[np.ndarray]:
    """
    VAD-gated ERes2Net speaker embedding extraction.

    Mirrors the pipeline in embedding.py:
    1. VAD filter to retain only voiced frames.
    2. Windowed extraction with average pooling for longer audio.
    3. L2 normalisation of the output vector.
    """
    from services.embedding import vad_extract_speaker_embedding as _ecapa_vad_extract

    # Run extraction using the ERes2Net extract_embedding function
    target_sr = SAMPLE_RATE
    if sr != target_sr:
        audio = _resample_audio(audio, sr, target_sr)
        sr = target_sr

    try:
        from services.embedding import (
            _run_silero_vad,
            _extract_speech_segments,
            _segment_stats,
        )

        vad_result = _run_silero_vad(audio, sr)
        if vad_result is not None:
            speech_audio = _extract_speech_segments(audio, vad_result, sr)
        else:
            speech_audio = audio

        duration_sec = len(speech_audio) / float(sr)
        if duration_sec < 0.2:
            logger.debug(f"[ERes2Net] Voiced audio too short ({duration_sec:.2f}s), fallback to raw")
            speech_audio = audio
            duration_sec = len(speech_audio) / float(sr)

        if duration_sec <= max_speech_sec:
            return extract_embedding(speech_audio, sr)

        # Sliding window averaging for long audio
        win_samples = int(win_sec * sr)
        hop_samples = int(hop_sec * sr)
        embeddings = []
        for start in range(0, len(speech_audio) - win_samples + 1, hop_samples):
            chunk = speech_audio[start : start + win_samples]
            emb = extract_embedding(chunk, sr)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return extract_embedding(speech_audio[:win_samples], sr)

        mean_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-10:
            mean_emb = mean_emb / norm
        return mean_emb.astype(np.float32)

    except Exception as e:
        logger.warning(f"[ERes2Net] VAD processing encountered error: {e}, falling back to direct extraction")
        return extract_embedding(audio, sr)
