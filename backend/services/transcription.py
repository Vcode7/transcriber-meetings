"""
Transcription service using WhisperX.

Drops in as a replacement for the previous faster-whisper service.
Returns the same output schema (segments with word-level timestamps)
plus an aligned result that pipeline.py uses for word→speaker assignment.

Offline note: whisperx.load_model() and load_align_model() are forced
offline via TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE env vars set in main.py.
The align model (facebook/wav2vec2-base) must be present in the HF cache.
"""
import logging
import os
from pathlib import Path
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


def unload_whisperx_model():
    """Unload the WhisperX model to free RAM/VRAM."""
    global _whisperx_model
    if _whisperx_model is not None:
        logger.info("[Transcription] Unloading WhisperX model...")
        # PyTorch model unloading
        del _whisperx_model
        _whisperx_model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[Transcription] WhisperX model unloaded.")



def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    try:
        if val is None:
            return default
        return round(float(val), 3)
    except (TypeError, ValueError):
        return default


def transcribe(file_path: str, initial_prompt: str = "", language: str = None) -> Dict[str, Any]:
    """
    Transcribe an audio file with WhisperX and run the forced-alignment step
    to obtain precise word-level timestamps.

    Args:
        file_path: Path to a WAV audio file.
        initial_prompt: Optional Whisper initial_prompt for context injection
                        (global prompt + meeting prompt + vocabulary).
        language: Optional detected language code to skip automatic language detection.

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
    if language:
        logger.info(f"[Transcription] Reusing pre-detected language: {language}")

    try:
        transcribe_kwargs = {"batch_size": 4}
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        if language:
            transcribe_kwargs["language"] = language
        raw_result = model.transcribe(file_path, **transcribe_kwargs)
    except TypeError as te:
        # Fallback if initial_prompt or language are not accepted as kwargs
        logger.warning(f"[Transcription] model.transcribe failed with TypeError: {te}. Retrying with basic options.")
        try:
            transcribe_kwargs = {"batch_size": 4}
            if language and "language" not in str(te):
                transcribe_kwargs["language"] = language
            if initial_prompt and "initial_prompt" not in str(te):
                transcribe_kwargs["initial_prompt"] = initial_prompt
            raw_result = model.transcribe(file_path, **transcribe_kwargs)
        except Exception as e2:
            try:
                # absolute minimal fallback
                raw_result = model.transcribe(file_path, batch_size=4)
            except Exception as e3:
                logger.error(f"[Transcription] model.transcribe fallback FAILED: {e3}", exc_info=True)
                raise
    except Exception as e:
        logger.error(f"[Transcription] model.transcribe() FAILED: {e}", exc_info=True)
        raise

    language: str = raw_result.get("language", "en")
    logger.info(f"[Transcription] Detected language: {language}, raw segments: {len(raw_result.get('segments', []))}")

    # ── Step 2: Forced alignment (word-level timestamps) ──────────────────
    logger.info("[Transcription] Running forced alignment ...")
    model_a = None
    aligned_result = raw_result  # default fallback
    try:
        # Resolve model directory: prefer local decrypted .dat, fall back to HF cache.
        hf_home = os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        )
        align_model_dir = hf_home  # default
        try:
            from services.model_loader import ModelLoader
            local_align_path = ModelLoader.get_model_path("align_engine")
            if local_align_path and local_align_path.exists():
                # local_align_path is the snapshot hash dir extracted from align_engine.dat.
                # whisperx.load_align_model() expects cache_dir to be the root of an HF hub
                # cache tree: cache_dir/models--facebook--wav2vec2-base/snapshots/<hash>/
                # Build that layout in the temp dir by symlinking/copying if not already done.
                import shutil
                snap_hash = local_align_path.name  # the hash folder name
                hf_model_id = "models--facebook--wav2vec2-base"
                constructed_cache = local_align_path.parent.parent / "hf_cache"
                target_snap = constructed_cache / hf_model_id / "snapshots" / snap_hash
                if not target_snap.exists() or not any(target_snap.iterdir()):
                    target_snap.mkdir(parents=True, exist_ok=True)
                    # Copy snapshot files into HF-structured cache
                    for item in local_align_path.iterdir():
                        dest = target_snap / item.name
                        if not dest.exists():
                            if item.is_dir():
                                shutil.copytree(str(item), str(dest))
                            else:
                                shutil.copy2(str(item), str(dest))
                    # Write the refs/main pointer
                    refs_dir = constructed_cache / hf_model_id / "refs"
                    refs_dir.mkdir(parents=True, exist_ok=True)
                    (refs_dir / "main").write_text(snap_hash, encoding="utf-8")
                align_model_dir = str(constructed_cache)
                logger.info(f"[Transcription] Using local align model (HF layout): {align_model_dir}")
        except Exception as am_err:
            logger.warning(f"[Transcription] Could not resolve local align model ({am_err}), using HF cache.")


        model_name = "facebook/wav2vec2-base"
        model_cache_only = False
        if align_model_dir != hf_home:
            model_cache_only = True
        elif settings.OFFLINE_MODE:
            model_cache_only = True

        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_name=model_name,
            model_dir=align_model_dir,
            model_cache_only=model_cache_only,
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
            "Falling back to unaligned segments. "
            "Ensure facebook/wav2vec2-base is present as align_engine.dat in MODELS_DIR."
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
