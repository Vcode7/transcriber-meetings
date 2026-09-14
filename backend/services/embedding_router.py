"""
Embedding model router — dispatches embedding operations to the active model.

This module provides a unified API for speaker embedding extraction that
transparently routes to the correct backend (ECAPA-TDNN or ERes2Net-Large)
based on the user's active setting.

When the active model is 'ecapa' (the default), all calls delegate directly
to the existing services.embedding module with zero overhead.

When the active model is 'eres2net_large', calls are routed to
services.embedding_eres2net instead.

Key functions
-------------
- get_active_model()             → current model name from runtime config
- extract_embedding()            → model-aware embedding extraction
- extract_embedding_from_file()  → model-aware file-based extraction
- vad_extract_speaker_embedding()→ model-aware VAD-gated extraction
- get_embedding_dim()            → embedding dimension for active model
- get_profile_embeddings()       → retrieve correct embeddings from a profile dict
- ensure_profile_embeddings()    → lazily generate missing embeddings from saved audio
- unload_active_encoder()        → unload whichever model is currently loaded
"""
from __future__ import annotations

import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

# Valid model identifiers
VALID_MODELS = {"ecapa", "eres2net_large"}

# Runtime cache for the active model setting (avoids repeated DB queries)
_active_model_cache: Optional[str] = None


def get_active_model(user_settings_row: dict = None) -> str:
    """
    Return the active speaker embedding model identifier.

    Checks (in order):
    1. Explicit user_settings_row dict (if provided)
    2. Module-level cache
    3. Falls back to 'ecapa'

    Returns 'ecapa' or 'eres2net_large'.
    """
    global _active_model_cache

    if user_settings_row is not None:
        model = str(user_settings_row.get("speaker_embedding_model", "ecapa")).strip().lower()
        if model in VALID_MODELS:
            _active_model_cache = model
            return model
        return "ecapa"

    if _active_model_cache is not None:
        return _active_model_cache

    return "ecapa"


def set_active_model(model: str) -> None:
    """Set the active model in the module-level cache."""
    global _active_model_cache
    if model in VALID_MODELS:
        _active_model_cache = model
    else:
        logger.warning(f"[EmbeddingRouter] Invalid model '{model}', keeping current setting")


def get_embedding_dim(model: str = None) -> int:
    """Return the embedding dimension for the given (or active) model."""
    m = model or get_active_model()
    if m == "eres2net_large":
        from services.embedding_eres2net import get_embedding_dim as _eres_dim
        return _eres_dim()
    else:
        from services.embedding import EMBEDDING_DIM
        return EMBEDDING_DIM


def extract_embedding(audio: np.ndarray, sr: int = 16000, model: str = None) -> Optional[np.ndarray]:
    """Extract a speaker embedding using the specified (or active) model."""
    m = model or get_active_model()
    if m == "eres2net_large":
        from services.embedding_eres2net import extract_embedding as _extract
    else:
        from services.embedding import extract_embedding as _extract
    return _extract(audio, sr)


def extract_embedding_from_file(file_path: str, model: str = None) -> Optional[np.ndarray]:
    """Extract a speaker embedding from a file using the specified (or active) model."""
    m = model or get_active_model()
    if m == "eres2net_large":
        from services.embedding_eres2net import extract_embedding_from_file as _extract
    else:
        from services.embedding import extract_embedding_from_file as _extract
    return _extract(file_path)


def vad_extract_speaker_embedding(
    audio: np.ndarray, sr: int = 16000, model: str = None, **kwargs
) -> Optional[np.ndarray]:
    """VAD-gated speaker embedding extraction using the specified (or active) model."""
    m = model or get_active_model()
    if m == "eres2net_large":
        from services.embedding_eres2net import vad_extract_speaker_embedding as _vad_extract
    else:
        from services.embedding import vad_extract_speaker_embedding as _vad_extract
    return _vad_extract(audio, sr, **kwargs)


def unload_active_encoder() -> None:
    """Unload whichever embedding model is currently loaded in memory."""
    # Try to unload both — only the loaded one will actually do work
    try:
        from services.embedding import unload_encoder as _unload_ecapa
        _unload_ecapa()
    except Exception:
        pass
    try:
        from services.embedding_eres2net import unload_encoder as _unload_eres2net
        _unload_eres2net()
    except Exception:
        pass


def extract_dual_embeddings(file_path: str) -> tuple[Optional[list], Optional[list]]:
    """
    Extract BOTH ECAPA (192-d) and ERes2Net-Large (512-d) embeddings from an audio file.

    Returns
    -------
    (ecapa_emb, eres2net_emb) : tuple of (list of float | None, list of float | None)
    """
    ecapa_emb = None
    eres2net_emb = None

    # 1. ECAPA-TDNN (192-d)
    try:
        from services.embedding import extract_embedding_from_file as _extract_ecapa
        emb_ecapa = _extract_ecapa(file_path)
        if emb_ecapa is not None:
            emb_list = emb_ecapa.tolist() if hasattr(emb_ecapa, "tolist") else list(emb_ecapa)
            if len(emb_list) == 192:
                ecapa_emb = emb_list
    except Exception as e:
        logger.warning(f"[EmbeddingRouter] Failed to extract ECAPA embedding from {file_path}: {e}")

    # 2. ERes2Net-Large (512-d)
    try:
        from services.embedding_eres2net import extract_embedding_from_file as _extract_eres
        emb_eres = _extract_eres(file_path)
        if emb_eres is not None:
            emb_list = emb_eres.tolist() if hasattr(emb_eres, "tolist") else list(emb_eres)
            if len(emb_list) == 512:
                eres2net_emb = emb_list
    except Exception as e:
        logger.warning(f"[EmbeddingRouter] Failed to extract ERes2Net embedding from {file_path}: {e}")

    return ecapa_emb, eres2net_emb


def validate_embedding_dimension(embedding: Any, expected_model: str) -> bool:
    """Check if embedding vector matches the required dimension for expected_model."""
    expected_dim = get_embedding_dim(expected_model)
    if embedding is None:
        return False
    dim = len(embedding) if hasattr(embedding, "__len__") else 0
    return dim == expected_dim


def get_profile_embeddings(profile: dict, model: str = None) -> list:
    """
    Return the correct embeddings list from a profile dict for the given model.

    Strict rules:
    - For ERes2Net-Large: checks ONLY eres2net_embeddings (must be 512-d).
      Never falls back to ECAPA or legacy 'embeddings'.
    - For ECAPA: checks ecapa_embeddings, or legacy 'embeddings' (must be 192-d).
      Never returns 512-d embeddings.

    Returns an empty list if no compatible embeddings exist.
    """
    m = model or get_active_model()

    if m == "eres2net_large":
        embs = profile.get("eres2net_embeddings")
        if embs is None:
            return []
        if isinstance(embs, str):
            from database import from_json
            embs = from_json(embs, [])
        if not isinstance(embs, list):
            return []
        # Strictly enforce 512-d dimension
        return [e for e in embs if hasattr(e, "__len__") and len(e) == 512]
    else:
        # ECAPA: prefer ecapa_embeddings, fall back to legacy 'embeddings'
        embs = profile.get("ecapa_embeddings")
        if embs is None or embs == "null":
            embs = profile.get("embeddings", [])
        if isinstance(embs, str):
            from database import from_json
            embs = from_json(embs, [])
        if not isinstance(embs, list):
            return []
        # Strictly enforce 192-d dimension
        return [e for e in embs if hasattr(e, "__len__") and len(e) == 192]


def ensure_profile_embeddings(
    profile: dict, model: str = None, save_callback=None
) -> list:
    """
    Ensure a profile has embeddings for the given model.

    If embeddings already exist, return them immediately.
    If not, generate them from the profile's saved audio files.

    Parameters
    ----------
    profile : dict
        Voice profile dict with 'audio_paths', 'embeddings', etc.
    model : str
        Target model ('ecapa' or 'eres2net_large').
    save_callback : callable, optional
        Async callback(profile_id, model, embeddings_list) to persist
        the newly generated embeddings to the database. If None,
        embeddings are returned but not saved.

    Returns
    -------
    list of embedding vectors (as lists of floats), possibly empty.
    """
    import os

    m = model or get_active_model()
    existing = get_profile_embeddings(profile, m)
    if existing:
        return existing

    # No embeddings for this model — generate from saved audio
    audio_paths = profile.get("audio_paths", [])
    if isinstance(audio_paths, str):
        from database import from_json
        audio_paths = from_json(audio_paths, [])

    if not audio_paths:
        profile_label = profile.get("label", "?")
        logger.warning(
            f"[EmbeddingRouter] Profile '{profile_label}' has no saved audio files — "
            f"cannot generate {m} embeddings. User must re-record voice samples."
        )
        return []

    logger.info(
        f"[EmbeddingRouter] Generating {m} embeddings for profile "
        f"'{profile.get('label', '?')}' from {len(audio_paths)} saved audio files"
    )

    embeddings = []
    for fp in audio_paths:
        if not os.path.exists(fp):
            logger.warning(f"[EmbeddingRouter] Audio file not found: {fp}")
            continue
        emb = extract_embedding_from_file(fp, model=m)
        if emb is not None:
            embeddings.append(emb.tolist())

    if embeddings:
        logger.info(
            f"[EmbeddingRouter] Generated {len(embeddings)} {m} embeddings "
            f"for profile '{profile.get('label', '?')}'"
        )
        # Update profile dict in-memory
        if m == "eres2net_large":
            profile["eres2net_embeddings"] = embeddings
        else:
            profile["ecapa_embeddings"] = embeddings
    else:
        logger.warning(
            f"[EmbeddingRouter] Failed to generate any {m} embeddings "
            f"for profile '{profile.get('label', '?')}'"
        )

    return embeddings


def get_effective_threshold(
    user_threshold: float,
    model: str = None,
    use_model_default: bool = True,
) -> tuple:
    """
    Determine the effective similarity threshold for the active model.

    Returns (effective_threshold, reason_string).
    """
    m = model or get_active_model()
    _LEGACY_THRESHOLD = 0.75

    try:
        from config import settings
        if m == "eres2net_large":
            model_default = settings.SPEAKER_SIMILARITY_THRESHOLD_ERES2NET
        else:
            model_default = settings.SPEAKER_SIMILARITY_THRESHOLD_ECAPA_TDNN
    except Exception:
        model_default = 0.55 if m == "eres2net_large" else 0.72

    if use_model_default and abs(user_threshold - _LEGACY_THRESHOLD) < 1e-6:
        model_name = "ERes2Net-Large" if m == "eres2net_large" else "ECAPA-TDNN"
        return model_default, (
            f"model-specific {model_name} default ({model_default}) "
            f"[legacy threshold {_LEGACY_THRESHOLD} overridden]"
        )

    return user_threshold, f"caller-provided ({user_threshold})"
