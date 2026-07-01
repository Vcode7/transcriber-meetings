"""
Speaker embedding service using resemblyzer.
Generates 256-d voice embeddings and computes cosine similarity.
"""
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

_encoder = None
SAMPLE_RATE = 16000

# Shared device selection
try:
    from services.device_utils import DEVICE as _DEVICE
except Exception:
    _DEVICE = "cpu"


def get_encoder():
    global _encoder
    if _encoder is None:
        try:
            from resemblyzer import VoiceEncoder
            # VoiceEncoder accepts device="cuda" or device="cpu"
            _encoder = VoiceEncoder(device=_DEVICE)
            logger.info(f"[Embedding] VoiceEncoder loaded on {_DEVICE}.")
        except TypeError:
            # Older resemblyzer versions don't support the device argument
            from resemblyzer import VoiceEncoder
            _encoder = VoiceEncoder()
            logger.info("[Embedding] VoiceEncoder loaded (no device arg — CPU).")
        except Exception as e:
            logger.error(f"[Embedding] Failed to load VoiceEncoder: {e}")
            raise
    return _encoder


def _load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load mono float32 audio with librosa using keyword-only-safe resampling."""
    import librosa

    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return np.asarray(audio, dtype=np.float32), sr


def _resample_audio(audio: np.ndarray, source_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample audio while staying compatible with librosa >=0.10."""
    if source_sr == target_sr:
        return audio
    print(source_sr,target_sr)
    import librosa

    return librosa.resample(
        y=np.asarray(audio, dtype=np.float32),
        orig_sr=source_sr,
        target_sr=target_sr,
    )


def _preprocess_wav(audio: np.ndarray, source_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Prepare audio for Resemblyzer without calling its librosa positional
    resample path, which breaks with newer librosa versions.
    """
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1 if wav.shape[0] > wav.shape[1] else 0)

    wav = _resample_audio(wav, source_sr, SAMPLE_RATE)

    try:
        from resemblyzer.audio import normalize_volume, trim_long_silences

        wav = normalize_volume(wav, -30, increase_only=True)
        wav = trim_long_silences(wav)
    except Exception as e:
        logger.warning(f"[Embedding] Resemblyzer preprocessing fallback used: {e}")
        peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
        if peak > 0:
            wav = wav / peak * 0.95

    return np.asarray(wav, dtype=np.float32)


def extract_embedding(audio: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
    """
    Extract a 256-d speaker embedding from mono audio at 16kHz.
    Returns None on failure.
    """
    try:
        enc = get_encoder()
        wav = _preprocess_wav(audio, source_sr=sr)
        if len(wav) < SAMPLE_RATE * 1.0:   # need at least 1 second
            return None
        embedding = enc.embed_utterance(wav)
        return embedding
    except Exception as e:
        logger.error(f"[Embedding] extract_embedding failed: {e}")
        return None


def extract_embedding_from_file(file_path: str) -> Optional[np.ndarray]:
    """Load a file and extract embedding."""
    try:
        enc = get_encoder()
        audio, sr = _load_audio(file_path, target_sr=SAMPLE_RATE)
        wav = _preprocess_wav(audio, source_sr=sr)
        if len(wav) < SAMPLE_RATE * 1.0:
            return None
        embedding = enc.embed_utterance(wav)
        return embedding
    except Exception as e:
        logger.error(f"[Embedding] extract_embedding_from_file failed: {e}")
        return None


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
    """Average multiple embeddings into one representative vector."""
    arr = np.stack(embeddings, axis=0)
    avg = np.mean(arr, axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg
