"""
Transcription service using WhisperX.

Drops in as a replacement for the previous faster-whisper service.
Returns the same output schema (segments with word-level timestamps)
plus an aligned result that pipeline.py uses for word→speaker assignment.

Offline note: whisperx.load_model() and load_align_model() are forced
offline via TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE env vars set in main.py.
The align model (facebook/wav2vec2-base) must be present in the HF cache.

Alignment improvements (v2):
-----------------------------
1. **Text normalization** before alignment — expands numbers, contractions,
   acronyms, and strips alignment-hostile punctuation using
   ``services.transcript_normalizer.normalize_segments_for_alignment()``.

2. **Audio preprocessing** before alignment (optional, controlled by
   ``settings.AUDIO_PREPROCESS_BEFORE_ALIGNMENT``) — applies silence trim,
   clipping repair, and loudness normalization using
   ``services.audio_preprocessing.preprocess_audio_for_alignment()``.

3. **VAD-chunked alignment** — speech regions are detected with
   ``services.audio_preprocessing.detect_speech_regions()`` and each region
   is aligned independently.  Timestamps are offset back to global time
   before stitching into the final result.  This avoids feeding long noisy
   spans to wav2vec2 which degrades per-word confidence scores.
   Fallback: if chunked alignment fails for any reason the service falls back
   to the previous single-call alignment path.
"""
from pydantic import root_model
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import settings

logger = logging.getLogger(__name__)

_whisperx_model = None
_whisperx_device: str = "cuda"
_whisperx_compute_type: str = "float16"

_align_model_cache = {}  # Keys: (language_code, device, model_name) -> (model_a, metadata)


def unload_align_model():
    """Unload all cached WhisperX alignment models to free RAM/VRAM."""
    global _align_model_cache
    if _align_model_cache:
        logger.info(f"[Transcription] Unloading {len(_align_model_cache)} cached WhisperX alignment models...")
        for key, val in list(_align_model_cache.items()):
            try:
                model_a, metadata = val
                del model_a
            except Exception:
                pass
        _align_model_cache.clear()
        import gc
        gc.collect()
        try:
            import torch
            device, _ = _resolve_device()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[Transcription] Cached alignment models unloaded.")


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
        # Ensure SpeechBrain k2/flair mocks and torchaudio patches are in place
        # before WhisperX imports its bundled Pyannote VAD (which triggers
        # SpeechBrain lazy-import machinery).
        from services.compat import apply_compatibility_patches
        apply_compatibility_patches()

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


# ─────────────────────────────────────────────────────────────────────────────
# Alignment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_align_model_dir() -> tuple[str, bool]:
    """
    Return (align_model_dir, cache_only) for whisperx.load_align_model().
    Prefers the local decrypted .dat; falls back to the HF cache.
    """
    hf_home = os.environ.get(
        "HF_HOME",
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    )
    align_model_dir = hf_home
    model_cache_only = bool(settings.OFFLINE_MODE)

    try:
        from services.model_loader import ModelLoader
        import shutil
        local_align_path = ModelLoader.get_model_path("align_engine")
        if local_align_path and local_align_path.exists():
            snap_hash = local_align_path.name
            hf_model_id = "models--facebook--wav2vec2-base"
            constructed_cache = local_align_path.parent.parent / "hf_cache"
            target_snap = constructed_cache / hf_model_id / "snapshots" / snap_hash
            if not target_snap.exists() or not any(target_snap.iterdir()):
                target_snap.mkdir(parents=True, exist_ok=True)
                for item in local_align_path.iterdir():
                    dest = target_snap / item.name
                    if not dest.exists():
                        if item.is_dir():
                            shutil.copytree(str(item), str(dest))
                        else:
                            shutil.copy2(str(item), str(dest))
                refs_dir = constructed_cache / hf_model_id / "refs"
                refs_dir.mkdir(parents=True, exist_ok=True)
                (refs_dir / "main").write_text(snap_hash, encoding="utf-8")
            align_model_dir = str(constructed_cache)
            model_cache_only = True
            logger.info(f"[Transcription] Using local align model (HF layout): {align_model_dir}")
    except Exception as am_err:
        logger.warning(f"[Transcription] Could not resolve local align model ({am_err}), using HF cache.")

    return align_model_dir, model_cache_only


def _align_segments_chunked(
    segments: List[Dict[str, Any]],
    audio_path: str,
    model_a,
    metadata: Dict,
    device: str,
    speech_regions: Optional[List[tuple]] = None,
) -> Dict[str, Any]:
    """
    Align *segments* against *audio_path* using VAD-chunked alignment.

    Each speech region is aligned separately so wav2vec2 receives short,
    high-SNR audio slices rather than long noisy spans.  Timestamps in the
    per-chunk aligned results are offset back to global audio time before
    stitching.

    Falls back to a single full-audio alignment call if chunked alignment
    raises an exception.

    Args:
        segments:       Whisper transcription segments (text + coarse timestamps).
        audio_path:     Path to the 16 kHz mono WAV.
        model_a:        Loaded WhisperX alignment model.
        metadata:       Alignment model metadata from load_align_model.
        device:         Torch device string.
        speech_regions: List of (start_sec, end_sec) speech spans. If None,
                        a single full-audio alignment is performed.

    Returns:
        WhisperX-compatible aligned result dict with "segments" key.
    """
    import whisperx
    import soundfile as sf
    import numpy as np
    import tempfile, os

    if not speech_regions:
        logger.info("[Transcription] Chunked alignment: no speech regions — using single-pass alignment")
        return whisperx.align(
            segments, model_a, metadata, audio_path, device,
            return_char_alignments=False,
        )

    logger.info(f"[Transcription] Chunked alignment: {len(speech_regions)} speech regions")

    # Load full audio once
    try:
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except Exception as e:
        logger.warning(f"[Transcription] Chunked alignment: failed to load audio ({e}) — single-pass fallback")
        return whisperx.align(
            segments, model_a, metadata, audio_path, device,
            return_char_alignments=False,
        )

    all_aligned_segments = []
    failed_chunks = 0

    # Map each segment to exactly one speech region to prevent duplicate aligned segments.
    # We assign each segment to the speech region with the maximum overlap. If there's
    # no overlap at all, we assign it to the closest region by midpoint distance.
    region_to_segs = {i: [] for i in range(len(speech_regions))}
    for s in segments:
        s_start = s.get("start", 0.0)
        s_end = s.get("end", s_start + 0.1)
        s_mid = (s_start + s_end) / 2.0

        best_region_idx = -1
        best_overlap = -1.0
        for idx, (r_start, r_end) in enumerate(speech_regions):
            overlap = max(0.0, min(s_end, r_end) - max(s_start, r_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_region_idx = idx

        if best_overlap <= 0.0:
            min_dist = float('inf')
            for idx, (r_start, r_end) in enumerate(speech_regions):
                if s_mid < r_start:
                    dist = r_start - s_mid
                elif s_mid > r_end:
                    dist = s_mid - r_end
                else:
                    dist = 0.0
                if dist < min_dist:
                    min_dist = dist
                    best_region_idx = idx

        if best_region_idx != -1:
            region_to_segs[best_region_idx].append(s)

    for idx, (region_start, region_end) in enumerate(speech_regions):
        region_segs = region_to_segs[idx]
        if not region_segs:
            continue

        # Slice audio for this region
        s_idx = int(region_start * sr)
        e_idx = int(region_end * sr)
        chunk_audio = audio[s_idx:e_idx]
        if len(chunk_audio) < sr * 0.3:
            # Too short to align — carry forward unaligned segments as-is
            all_aligned_segments.extend(region_segs)
            continue

        # Shift segment timestamps to chunk-local time
        offset = region_start
        local_segs = []
        for s in region_segs:
            ls = dict(s)
            ls["start"] = max(0.0, s.get("start", 0.0) - offset)
            ls["end"] = max(0.0, s.get("end", 0.0) - offset)
            if "words" in s and s["words"]:
                ls["words"] = [
                    {**w, "start": max(0.0, w.get("start", 0.0) - offset),
                     "end": max(0.0, w.get("end", 0.0) - offset)}
                    for w in s["words"]
                ]
            local_segs.append(ls)

        # Write chunk to temp WAV
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix="_chunk.wav")
            os.close(fd)
            sf.write(tmp_path, chunk_audio, sr, subtype="PCM_16")

            chunk_aligned = whisperx.align(
                local_segs, model_a, metadata, tmp_path, device,
                return_char_alignments=False,
            )

            # Shift timestamps back to global time
            for aligned_seg in chunk_aligned.get("segments", []):
                gs = dict(aligned_seg)
                gs["start"] = round(gs.get("start", 0.0) + offset, 3)
                gs["end"] = round(gs.get("end", 0.0) + offset, 3)
                if "words" in gs:
                    gs["words"] = [
                        {**w,
                         "start": round(w.get("start", 0.0) + offset, 3),
                         "end": round(w.get("end", 0.0) + offset, 3)}
                        for w in gs["words"]
                    ]
                all_aligned_segments.append(gs)

        except Exception as chunk_err:
            logger.warning(
                f"[Transcription] Chunked alignment: region [{region_start:.2f}s–{region_end:.2f}s] "
                f"failed ({chunk_err}) — using unaligned segments for this region"
            )
            # Fall back to unaligned (global time) segments for this region
            all_aligned_segments.extend(region_segs)
            failed_chunks += 1
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    if not all_aligned_segments:
        logger.warning("[Transcription] Chunked alignment produced no segments — single-pass fallback")
        return whisperx.align(
            segments, model_a, metadata, audio_path, device,
            return_char_alignments=False,
        )

    # Sort by start time to ensure correct ordering after stitching
    all_aligned_segments.sort(key=lambda s: s.get("start", 0.0))

    if failed_chunks:
        logger.warning(f"[Transcription] Chunked alignment: {failed_chunks} chunks fell back to unaligned")

    logger.info(
        f"[Transcription] Chunked alignment complete: {len(all_aligned_segments)} segments "
        f"from {len(speech_regions)} regions"
    )
    return {"segments": all_aligned_segments}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

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

    # ── Step 1: Transcribe ────────────────────────────────────────────────
    logger.info(f"[Transcription] Transcribing {file_path} on device={device} ...")
    if initial_prompt:
        logger.info(f"[Transcription] Using initial_prompt ({len(initial_prompt)} chars)")
    if language:
        logger.info(f"[Transcription] Reusing pre-detected language: {language}")

    transcribe_kwargs = {"batch_size": 8}

    if language:
        transcribe_kwargs["language"] = language
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    try:
        raw_result = model.transcribe(file_path, **transcribe_kwargs)

    except TypeError as e:
        logger.warning(f"Retrying without unsupported kwargs: {e}")

        unsupported = str(e)

        if "initial_prompt" in unsupported:
            transcribe_kwargs.pop("initial_prompt", None)

        if "language" in unsupported:
            transcribe_kwargs.pop("language", None)

        raw_result = model.transcribe(file_path, **transcribe_kwargs)

    language: str = raw_result.get("language", "en")
    logger.info(f"[Transcription] Detected language: {language}, raw segments: {len(raw_result.get('segments', []))}")

    # ── Step 1b: Normalize segments before alignment ──────────────────────
    # Extract vocabulary terms from the initial_prompt for canonical reuse.
    vocab_terms: Optional[List[str]] = None
    if initial_prompt:
        try:
            # Heuristic: extract capitalized words and quoted terms from prompt
            import re
            # Pull any "Terms: X, Y, Z." or capitalized proper nouns
            terms_match = re.search(r"Terms:\s*([^\n]+)", initial_prompt)
            if terms_match:
                vocab_terms = [t.strip().rstrip(".") for t in terms_match.group(1).split(",") if t.strip()]
        except Exception:
            vocab_terms = None

    raw_segments = raw_result.get("segments", [])
    try:
        from services.transcript_normalizer import normalize_segments_for_alignment
        normalized_segments = normalize_segments_for_alignment(raw_segments, vocab_terms)
        
        logger.info(
            f"[Transcription] Text normalization applied to {len(normalized_segments)} segments "
            f"(vocab_terms={len(vocab_terms) if vocab_terms else 0})"
        )
    except Exception as norm_err:
        logger.warning(f"[Transcription] Text normalization failed ({norm_err}) — using raw segments")
        normalized_segments = raw_segments

    # ── Step 1c: Audio preprocessing before alignment ─────────────────────
    alignment_audio_path = file_path
    _preprocessed_path: Optional[str] = None

    # ── Step 2: Forced alignment (word-level timestamps) ──────────────────
    logger.info("[Transcription] Running forced alignment ...")
    model_a = None
    aligned_result = raw_result  # default fallback

    try:
        align_model_dir, model_cache_only = _resolve_align_model_dir()

        model_name = "facebook/wav2vec2-base"
        cache_key = (language, device, model_name)
        if cache_key in _align_model_cache:
            model_a, metadata = _align_model_cache[cache_key]
            logger.info(f"[Transcription] Reusing cached alignment model for key {cache_key}")
        else:
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=device,
                model_name=model_name,
                model_dir=align_model_dir,
                model_cache_only=model_cache_only,
            )
            _align_model_cache[cache_key] = (model_a, metadata)
            logger.info(
                f"[Transcription] Alignment model loaded and cached: {model_name} "
                f"(dir={align_model_dir}, cache_only={model_cache_only})"
            )

        # ── Step 2a: Detect speech regions for chunked alignment ──────────
   
        aligned_result = _align_segments_chunked(
            normalized_segments,
            alignment_audio_path,
            model_a,
            metadata,
            device,
            speech_regions=None,
        )
        logger.info(
            f"[Transcription] Alignment complete ✓ "
            f"({len(aligned_result.get('segments', []))} segments)"
        )

    except Exception as e:
        logger.warning(
            f"[Transcription] Alignment failed ({e}). "
            "Falling back to unaligned segments. "
            "Ensure facebook/wav2vec2-base is present as align_engine in MODELS_DIR."
        )
        aligned_result = raw_result
    finally:
        # Clean up preprocessed temp WAV
        if _preprocessed_path and _preprocessed_path != file_path:
            try:
                from services.audio_preprocessing import cleanup_temp_wav
                cleanup_temp_wav(_preprocessed_path, file_path)
            except Exception:
                pass

    # ── Step 3: Normalise output schema ───────────────────────────────────
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
