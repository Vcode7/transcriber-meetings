"""
Transcription service using WhisperX.

Drops in as a replacement for the previous faster-whisper service.
Returns the same output schema (segments with word-level timestamps)
plus an aligned result that pipeline.py uses for word→speaker assignment.
"""
import logging
from typing import List, Dict, Any
from config import settings

logger = logging.getLogger(__name__)

_whisperx_model = None
_whisperx_device: str = "cuda"
_whisperx_compute_type: str = "float16"


def _resolve_device() -> tuple[str, str]:
    """
    Resolve device and compute_type.
    - WHISPER_DEVICE="auto"  → use device_utils (cuda if available, else cpu)
    - WHISPER_DEVICE="cuda"  → force CUDA (may fail if unavailable)
    - WHISPER_DEVICE="cpu"   → always CPU
    """
    from services.device_utils import DEVICE as _AUTO_DEVICE
    raw = settings.WHISPER_DEVICE
    device = _AUTO_DEVICE if raw == "auto" else raw
    compute_type = settings.WHISPER_COMPUTE_TYPE
    # float16 only makes sense on CUDA; fall back to int8 on CPU
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"
    return device, compute_type


def get_whisperx_model():
    """Lazy-load the WhisperX model (done once per process)."""
    global _whisperx_model, _whisperx_device, _whisperx_compute_type
    if _whisperx_model is None:
        import whisperx
        _whisperx_device, _whisperx_compute_type = _resolve_device()
        logger.info(
            f"[Transcription] Loading WhisperX model '{settings.WHISPER_MODEL_SIZE}' "
            f"on {_whisperx_device} ({_whisperx_compute_type})"
        )
        _whisperx_model = whisperx.load_model(
            settings.WHISPER_MODEL_SIZE,
            _whisperx_device,
            compute_type=_whisperx_compute_type,
        )
        logger.info("[Transcription] WhisperX model ready ✓")
    return _whisperx_model


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    try:
        if val is None:
            return default
        return round(float(val), 3)
    except (TypeError, ValueError):
        return default


def transcribe(file_path: str, initial_prompt: str = "") -> Dict[str, Any]:
    """
    Transcribe an audio file with WhisperX and run the forced-alignment step
    to obtain precise word-level timestamps.

    Args:
        file_path: Path to a WAV audio file.
        initial_prompt: Optional Whisper initial_prompt for context injection
                        (global prompt + meeting prompt + vocabulary).

    Returns:
        {
          "segments": [
            {
              "start": float,
              "end": float,
              "text": str,
              "words": [{"word": str, "start": float, "end": float, "probability": float}],
              "avg_logprob": float,
            }
          ],
          "language": str,
          "raw_text": str,
          # Extra key consumed by pipeline.py for word→speaker assignment:
          "aligned_result": dict,   # raw WhisperX aligned output
        }
    """
    import whisperx
    import gc

    device, compute_type = _resolve_device()
    model = get_whisperx_model()

    # ── Step 1: Transcribe ────────────────────────────────────
    logger.info(f"[Transcription] Transcribing {file_path} on device={device} ...")
    if initial_prompt:
        logger.info(f"[Transcription] Using initial_prompt ({len(initial_prompt)} chars)")
    try:
        transcribe_kwargs = {"batch_size": 4}
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        raw_result = model.transcribe(file_path, **transcribe_kwargs)
    except TypeError as te:
        # FasterWhisperPipeline.transcribe() in some WhisperX / faster-whisper
        # versions does not accept 'initial_prompt' as a top-level keyword argument.
        # Retry without it — transcription still succeeds, just without the hint.
        if initial_prompt and "initial_prompt" in str(te):
            logger.warning(
                f"[Transcription] initial_prompt not supported by this WhisperX build "
                f"({te}). Retrying without it."
            )
            try:
                raw_result = model.transcribe(file_path, batch_size=4)
            except Exception as e2:
                logger.error(f"[Transcription] model.transcribe() FAILED on retry: {e2}", exc_info=True)
                raise
        else:
            logger.error(f"[Transcription] model.transcribe() FAILED (TypeError): {te}", exc_info=True)
            raise
    except Exception as e:
        logger.error(f"[Transcription] model.transcribe() FAILED: {e}", exc_info=True)
        raise

    language: str = raw_result.get("language", "en")
    logger.info(f"[Transcription] Detected language: {language}, raw segments: {len(raw_result.get('segments', []))}")

    # ── Step 2: Forced alignment (word-level timestamps) ──────
    logger.info("[Transcription] Running forced alignment ...")
    model_a = None
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )
        aligned_result = whisperx.align(
            raw_result["segments"],
            model_a,
            metadata,
            file_path,
            device,
            return_char_alignments=False,
        )
        logger.info(f"[Transcription] Alignment complete ✓ ({len(aligned_result.get('segments', []))} segments)")
    except Exception as e:
        logger.warning(
            f"[Transcription] Alignment failed ({e}). "
            "Falling back to unaligned segments."
        )
        aligned_result = raw_result
    finally:
        # Free alignment model VRAM immediately — we're done with it
        if model_a is not None:
            try:
                del model_a
                import torch
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("[Transcription] Alignment model freed from VRAM")
            except Exception:
                pass

    # ── Step 3: Normalise output schema ───────────────────────
    segments: List[Dict[str, Any]] = []
    raw_parts: List[str] = []

    for seg in aligned_result.get("segments", []):
        seg_start = _safe_float(seg.get("start"), 0.0)
        seg_end = _safe_float(seg.get("end"), seg_start + 0.1)

        words: List[Dict[str, Any]] = []
        for w in seg.get("words", []):
            w_start = _safe_float(w.get("start"), seg_start)
            w_end = _safe_float(w.get("end"), seg_end)
            # WhisperX uses "score" instead of "probability"
            words.append({
                "word": w.get("word", "").strip(),
                "start": w_start,
                "end": w_end,
                # Expose as "probability" to keep downstream code unchanged
                "probability": round(float(w.get("score", w.get("probability", 1.0))), 4),
            })

        text = seg.get("text", "").strip()
        if not text:
            continue  # skip empty segments

        segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": text,
            "words": words,
            "avg_logprob": round(float(seg.get("avg_logprob", 0.0)), 4),
        })
        raw_parts.append(text)

    logger.info(f"[Transcription] Normalized {len(segments)} segments, {sum(len(s['words']) for s in segments)} words total")

    return {
        "segments": segments,
        "language": language,
        "raw_text": " ".join(raw_parts),
        # Pipeline uses this for word-level speaker assignment
        "aligned_result": aligned_result,
    }
