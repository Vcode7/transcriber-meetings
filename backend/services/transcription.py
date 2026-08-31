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
from dataclasses import replace, is_dataclass
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
        from services.device_utils import log_gpu_memory
        log_gpu_memory("Pre-unload Alignment Model")
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
        log_gpu_memory("Post-unload Alignment Model")



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
        from services.device_utils import log_gpu_memory
        log_gpu_memory("Pre-load WhisperX")
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
        log_gpu_memory("Post-load WhisperX")
    return _whisperx_model


def unload_whisperx_model():
    """Unload the WhisperX model to free RAM/VRAM."""
    global _whisperx_model
    if _whisperx_model is not None:
        from services.device_utils import log_gpu_memory
        log_gpu_memory("Pre-unload WhisperX")
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
        log_gpu_memory("Post-unload WhisperX")




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
# Parallel Whisper processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_audio_into_chunks(wav_path: str, chunk_minutes: int = 10) -> List[tuple]:
    """
    Split a WAV file into fixed-duration chunks for parallel transcription.

    Returns a list of (temp_wav_path, start_sec) tuples, one per chunk.
    Chunks are written as temporary PCM-16 WAV files (same sample rate as source).
    The caller is responsible for deleting temp files after use.
    """
    import soundfile as sf
    import numpy as np
    import tempfile

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    chunk_samples = int(chunk_minutes * 60 * sr)
    total_samples = len(audio)
    num_chunks = max(1, (total_samples + chunk_samples - 1) // chunk_samples)

    chunks = []
    for i in range(num_chunks):
        s = i * chunk_samples
        e = min(s + chunk_samples, total_samples)
        chunk_audio = audio[s:e]
        start_sec = round(s / sr, 3)

        fd, tmp_path = tempfile.mkstemp(suffix=f"_wchunk{i}.wav")
        os.close(fd)
        sf.write(tmp_path, chunk_audio, sr, subtype="PCM_16")
        chunks.append((tmp_path, start_sec))

    return chunks


def _worker_transcribe(
    wav_path: str,
    initial_prompt: str,
    language: Optional[str],
    user_settings: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Top-level (module-level, picklable) worker function executed inside a
    ProcessPoolExecutor subprocess.

    Loads its own WhisperX model copy, runs transcribe(), then unloads
    the model and clears CUDA cache before returning.

    Returns the same dict schema as transcribe().
    """
    try:
        result = transcribe(
            wav_path,
            initial_prompt=initial_prompt,
            language=language,
            user_settings=user_settings,
        )
        return result
    finally:
        # Always clean up this worker's model copy on exit.
        try:
            unload_whisperx_model()
        except Exception:
            pass
        try:
            unload_align_model()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _offset_segment_timestamps_parallel(seg: Dict[str, Any], offset: float) -> Dict[str, Any]:
    """Shift all timestamps in a transcription segment by *offset* seconds."""
    out = dict(seg)
    out["start"] = round(seg.get("start", 0.0) + offset, 3)
    out["end"] = round(seg.get("end", 0.0) + offset, 3)
    if "words" in seg and seg["words"]:
        out["words"] = [
            {**w,
             "start": round(w.get("start", 0.0) + offset, 3),
             "end": round(w.get("end", 0.0) + offset, 3)}
            for w in seg["words"]
        ]
    return out


def transcribe_parallel(
    file_path: str,
    initial_prompt: str = "",
    language: str = None,
    user_settings: Optional[Dict[str, Any]] = None,
    workers: int = 2,
    chunk_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parallel Whisper transcription: splits *file_path* into fixed-duration
    chunks and processes them concurrently via ProcessPoolExecutor.

    Each worker subprocess loads its own WhisperX model copy, transcribes its
    chunk, unloads the model, and clears CUDA cache before returning.
    After all futures complete, the main process also runs gc + cuda cache
    cleanup to release any residual references.

    Segment timestamps are offset by the chunk's start position so the merged
    result uses global audio time, identical to a sequential transcribe() call.

    Args:
        file_path:      Path to the source WAV file.
        initial_prompt: Whisper initial prompt (applied to every chunk).
        language:       Force-detected language (None = auto-detect on chunk 0).
        user_settings:  Per-user settings dict (passed through to transcribe()).
        workers:        Number of parallel worker processes.
        chunk_minutes:  Length of each audio chunk in minutes.

    Returns:
        {"segments": [...], "language": str, "raw_text": str, "aligned_result": {...}}
    """
    import concurrent.futures
    import gc
    import time

    if chunk_minutes is None:
        chunk_minutes = int((user_settings or {}).get("whisper_parallel_chunk_minutes") or getattr(settings, "WHISPER_PARALLEL_CHUNK_MINUTES", 10))
    chunk_minutes = max(1, int(chunk_minutes))
    workers = max(1, workers)


    logger.info(
        f"[Whisper/Parallel] 🚀 Starting parallel transcription: "
        f"splitting '{os.path.basename(file_path)}' into ~{chunk_minutes}-min chunks "
        f"with {workers} worker(s)"
    )

    # ── Split audio into chunks ──────────────────────────────────────────
    try:
        chunks = _split_audio_into_chunks(file_path, chunk_minutes=chunk_minutes)
    except Exception as e:
        logger.error(f"[Whisper/Parallel] Audio splitting failed ({e}); falling back to sequential transcription.")
        return transcribe(file_path, initial_prompt=initial_prompt, language=language, user_settings=user_settings)

    total = len(chunks)
    logger.info(
        f"[Whisper/Parallel] Split into {total} chunk(s) — "
        f"submitting to {min(workers, total)} worker(s)"
    )

    # ── Submit all chunks to ProcessPoolExecutor ─────────────────────────
    # We use a bounded pool (min(workers, total) so we don't spawn idle processes).
    effective_workers = min(workers, total)
    futures_map: Dict[concurrent.futures.Future, int] = {}  # future → chunk index
    chunk_paths = [c[0] for c in chunks]
    offsets = [c[1] for c in chunks]

    completed = 0
    chunk_results: List[Optional[Dict[str, Any]]] = [None] * total

    t_submit_start = time.monotonic()

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
            for idx, (tmp_wav, start_sec) in enumerate(chunks):
                future = executor.submit(
                    _worker_transcribe,
                    tmp_wav,
                    initial_prompt,
                    language,
                    user_settings,
                )
                futures_map[future] = idx
                logger.info(
                    f"[Whisper/Parallel] Chunk {idx + 1}/{total} → submitted "
                    f"[{start_sec:.1f}s – {start_sec + chunk_minutes * 60:.1f}s]"
                )

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures_map):
                idx = futures_map[future]
                start_sec = offsets[idx]
                try:
                    result = future.result()
                    chunk_results[idx] = result
                    n_segs = len(result.get("segments", []))
                    elapsed = round(time.monotonic() - t_submit_start, 1)
                    completed += 1
                    pct = int(completed / total * 100)
                    logger.info(
                        f"[Whisper/Parallel] Chunk {idx + 1}/{total} ✓ DONE "
                        f"({elapsed}s elapsed, {pct}% overall) — {n_segs} segments"
                    )
                except Exception as e:
                    logger.error(
                        f"[Whisper/Parallel] Chunk {idx + 1}/{total} ✗ FAILED "
                        f"(offset={start_sec:.1f}s): {e}",
                        exc_info=True,
                    )
                    chunk_results[idx] = None

    finally:
        # Delete all temp chunk WAV files regardless of outcome
        for tmp_wav, _ in chunks:
            try:
                if os.path.exists(tmp_wav):
                    os.unlink(tmp_wav)
            except OSError:
                pass

    # ── Main-process VRAM cleanup ────────────────────────────────────────
    logger.info("[Whisper/Parallel] All workers finished. Running main-process VRAM cleanup...")
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("[Whisper/Parallel] CUDA cache cleared ✓")
    except Exception:
        pass

    # ── Merge results in chunk order ─────────────────────────────────────
    merged_segments: List[Dict[str, Any]] = []
    merged_aligned_segs: List[Dict[str, Any]] = []
    merged_raw_parts: List[str] = []
    detected_language: str = language or "en"

    for idx, result in enumerate(chunk_results):
        if result is None:
            logger.warning(f"[Whisper/Parallel] Chunk {idx + 1}/{total} had no result — skipping.")
            continue

        offset = offsets[idx]
        lang = result.get("language", "en")
        if idx == 0 or not language:
            detected_language = lang

        for seg in result.get("segments", []):
            merged_segments.append(_offset_segment_timestamps_parallel(seg, offset))

        aligned = result.get("aligned_result", {})
        for seg in aligned.get("segments", result.get("segments", [])):
            merged_aligned_segs.append(_offset_segment_timestamps_parallel(seg, offset))

        if result.get("raw_text"):
            merged_raw_parts.append(result["raw_text"])

    total_merged = len(merged_segments)
    logger.info(
        f"[Whisper/Parallel] ✅ Merge complete: {total_merged} segments from {total} chunk(s). "
        f"Language={detected_language}. VRAM cleanup complete."
    )

    return {
        "segments": merged_segments,
        "language": detected_language,
        "raw_text": " ".join(merged_raw_parts),
        "aligned_result": {"segments": merged_aligned_segs},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def transcribe(
    file_path: str,
    initial_prompt: str = "",
    language: str = None,
    user_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Transcribe an audio file with WhisperX and run the forced-alignment step
    to obtain precise word-level timestamps. Supports configurable low-volume speech
    preservation features: Audio Normalization, Adaptive VAD, Speech Padding,
    Segment Merging, and Low-Volume Recovery Pass.
    """
    import whisperx
    import gc
    import soundfile as sf
    import numpy as np

    cfg = user_settings or {}

    # Extract pipeline toggles and parameters
    enable_transcription_vad = bool(cfg.get("enable_transcription_vad", cfg.get("enable_vad", getattr(settings, "ENABLE_TRANSCRIPTION_VAD", getattr(settings, "ENABLE_VAD", True)))))
    enable_alignment_vad = bool(cfg.get("enable_alignment_vad", getattr(settings, "ENABLE_ALIGNMENT_VAD", True)))

    enable_audio_norm = bool(cfg.get("enable_audio_normalization", getattr(settings, "ENABLE_AUDIO_NORMALIZATION", True)))
    norm_target_dbfs = float(cfg.get("norm_target_dbfs", getattr(settings, "NORM_TARGET_DBFS", -3.0)))
    norm_compression_ratio = float(cfg.get("norm_compression_ratio", getattr(settings, "NORM_COMPRESSION_RATIO", 2.0)))

    enable_adaptive_vad = bool(cfg.get("enable_adaptive_vad", getattr(settings, "ENABLE_ADAPTIVE_VAD", True)))
    vad_speech_threshold = float(cfg.get("vad_speech_threshold", getattr(settings, "VAD_SPEECH_THRESHOLD", 0.15)))
    vad_silence_threshold = float(cfg.get("vad_silence_threshold", getattr(settings, "VAD_SILENCE_THRESHOLD", 0.10)))
    vad_min_speech_ms = int(cfg.get("vad_min_speech_ms", getattr(settings, "VAD_MIN_SPEECH_MS", 250)))
    vad_min_silence_ms = int(cfg.get("vad_min_silence_ms", getattr(settings, "VAD_MIN_SILENCE_MS", 400)))

    enable_speech_padding = bool(cfg.get("enable_speech_padding", getattr(settings, "ENABLE_SPEECH_PADDING", True)))
    speech_pad_ms = int(cfg.get("speech_pad_ms", getattr(settings, "SPEECH_PAD_MS", 400)))

    enable_segment_merging = bool(cfg.get("enable_speech_segment_merging", getattr(settings, "ENABLE_SPEECH_SEGMENT_MERGING", True)))
    max_merge_silence_ms = int(cfg.get("max_merge_silence_ms", getattr(settings, "MAX_MERGE_SILENCE_MS", 500)))

    enable_low_volume_recovery = bool(cfg.get("enable_low_volume_recovery", getattr(settings, "ENABLE_LOW_VOLUME_RECOVERY", True)))
    recovery_energy_threshold = float(cfg.get("recovery_energy_threshold", getattr(settings, "RECOVERY_ENERGY_THRESHOLD", -45.0)))
    recovery_min_duration_ms = int(cfg.get("recovery_min_duration_ms", getattr(settings, "RECOVERY_MIN_DURATION_MS", 300)))

    whisper_batch_size = int(cfg.get("whisper_batch_size", getattr(settings, "WHISPER_BATCH_SIZE", 8)))
    whisper_batch_size = max(1, min(32, whisper_batch_size))

    logger.info(
        f"[Transcription Settings] "
        f"TranscriptionVAD={'Enabled (Region Processing)' if enable_transcription_vad else 'Disabled (Direct Full Audio)'}, "
        f"AlignmentVAD={'Enabled (Region Slicing)' if enable_alignment_vad else 'Disabled (Full Audio Single Pass)'}, "
        f"AudioNormalization={'Enabled' if enable_audio_norm else 'Disabled'}, "
        f"WhisperBatchSize={whisper_batch_size}"
    )

    device, compute_type = _resolve_device()
    model = get_whisperx_model()

    # ── Step 1: Preprocessing (Audio Normalization & Mild Dynamic Compression)
    alignment_audio_path = file_path
    _preprocessed_path: Optional[str] = None

    if enable_audio_norm:
        try:
            from services.audio_preprocessing import preprocess_audio_for_alignment
            _preprocessed_path = preprocess_audio_for_alignment(
                file_path,
                normalize_loudness=True,
                compress_dynamic_range=True,
                target_dbfs=norm_target_dbfs,
                compression_ratio=norm_compression_ratio,
            )
            if _preprocessed_path and _preprocessed_path != file_path:
                alignment_audio_path = _preprocessed_path
                logger.info(f"[Transcription] Audio normalization & compression applied → {alignment_audio_path}")
        except Exception as norm_err:
            logger.warning(f"[Transcription] Audio normalization failed ({norm_err}) — using raw WAV")

    # ── Step 2: Primary Whisper Transcription Pass ─────────────────────────
    transcribe_target_path = alignment_audio_path if enable_audio_norm else file_path
    if enable_transcription_vad:
        logger.info(f"[Transcription] Primary Whisper Pass: Transcribing {transcribe_target_path} on device={device} (batch_size={whisper_batch_size}) ...")
    else:
        logger.info(f"[Transcription] Primary Whisper Pass (Transcription VAD Disabled): Sending complete audio {transcribe_target_path} directly to Whisper on device={device} (batch_size={whisper_batch_size}) ...")

    prompt_str = initial_prompt.strip() if initial_prompt else None

    # WhisperX FasterWhisperPipeline uses a dataclass `model.options` (TranscriptionOptions)
    # where initial_prompt must be configured, rather than being passed as a kwarg to model.transcribe().
    if hasattr(model, "options") and is_dataclass(model.options):
        model.options = replace(model.options, initial_prompt=prompt_str)
        if prompt_str:
            logger.info(
                f"[Whisper] 🚀 INITIAL PROMPT ACTIVE: Set {len(prompt_str)} chars on WhisperX pipeline options | "
                f"Prompt Preview: {repr(prompt_str[:160])}"
            )
        else:
            logger.info("[Whisper] ℹ️ INITIAL PROMPT INACTIVE: No initial_prompt provided. Pipeline prompt cleared.")
    else:
        if prompt_str:
            logger.info(
                f"[Whisper] 🚀 INITIAL PROMPT ACTIVE: Sending {len(prompt_str)} chars to Whisper | "
                f"Prompt Preview: {repr(prompt_str[:160])}"
            )
        else:
            logger.info("[Whisper] ℹ️ INITIAL PROMPT INACTIVE: No initial_prompt provided. Transcribing standard audio.")

    transcribe_kwargs = {"batch_size": whisper_batch_size}
    if language:
        transcribe_kwargs["language"] = language

    # If the model is not a FasterWhisperPipeline (e.g. raw faster_whisper WhisperModel),
    # pass initial_prompt directly via transcribe_kwargs.
    if prompt_str and not hasattr(model, "options"):
        transcribe_kwargs["initial_prompt"] = prompt_str

    try:
        raw_result = model.transcribe(transcribe_target_path, **transcribe_kwargs)
        if prompt_str:
            logger.info(f"[Whisper] ✓ Successfully executed Whisper transcribe() with initial_prompt ({len(prompt_str)} chars)")
    except TypeError as e:
        logger.warning(f"[Whisper] Retrying without unsupported kwargs: {e}")
        unsupported = str(e)
        if "initial_prompt" in unsupported and "initial_prompt" in transcribe_kwargs:
            transcribe_kwargs.pop("initial_prompt", None)
            if hasattr(model, "options") and is_dataclass(model.options):
                model.options = replace(model.options, initial_prompt=prompt_str)
        if "language" in unsupported:
            transcribe_kwargs.pop("language", None)
        raw_result = model.transcribe(transcribe_target_path, **transcribe_kwargs)

    detected_lang: str = raw_result.get("language", language or "en")
    raw_segments = raw_result.get("segments", [])
    logger.info(f"[Transcription] Primary Whisper Pass complete ✓ Detected language: {detected_lang}, raw segments: {len(raw_segments)}")

    # ── Step 3: Transcription VAD & Low-Volume Recovery Pass ──────────────
    from services.audio_preprocessing import (
        detect_speech_regions, apply_speech_padding, merge_speech_segments,
        detect_rejected_low_volume_regions
    )

    audio_dur = 0.0
    audio_data = None
    sr = 16000
    try:
        audio_data, sr = sf.read(transcribe_target_path, dtype="float32", always_2d=False)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        audio_dur = len(audio_data) / sr
    except Exception:
        pass

    speech_regions = None
    recovered_segments_count = 0

    if enable_transcription_vad:
        logger.info("[Transcription] Transcription VAD: Enabled — Detecting speech regions and applying VAD post-processing...")
        if enable_adaptive_vad:
            speech_regions = detect_speech_regions(
                transcribe_target_path,
                aggressiveness=1,
                speech_quantile=vad_speech_threshold,
                min_speech_sec=vad_min_speech_ms / 1000.0,
                min_silence_sec=vad_min_silence_ms / 1000.0,
            )
        else:
            speech_regions = detect_speech_regions(transcribe_target_path)

        if enable_speech_padding and audio_dur > 0:
            speech_regions = apply_speech_padding(speech_regions, speech_pad_ms, audio_dur)

        if enable_segment_merging:
            speech_regions = merge_speech_segments(speech_regions, max_merge_silence_ms)

        # ── Low-Volume Recovery Pass (Secondary Whisper Pass on Quiet Regions with VAD Bypassed) ──
        if enable_low_volume_recovery and audio_data is not None and len(audio_data) > 0:
            try:
                recovered_spans = detect_rejected_low_volume_regions(
                    audio_data, sr, speech_regions,
                    energy_threshold_db=recovery_energy_threshold,
                    min_duration_sec=recovery_min_duration_ms / 1000.0,
                )
                if recovered_spans:
                    pad_sec = min(0.5, max(0.3, speech_pad_ms / 1000.0))
                    pad_ms = int(pad_sec * 1000)
                    import tempfile
                    from difflib import SequenceMatcher
                    from services.audio_preprocessing import _apply_dynamic_range_compression, _normalize_loudness
                    
                    total_raw_recovered = 0
                    total_merged_recovered = 0

                    for r_start, r_end in recovered_spans:
                        # Extract region with context padding (approx 300-500ms before and after)
                        padded_start = max(0.0, r_start - pad_sec)
                        padded_end = min(audio_dur, r_end + pad_sec)
                        s_idx = int(padded_start * sr)
                        e_idx = int(padded_end * sr)
                        span_audio = audio_data[s_idx:e_idx]
                        if len(span_audio) < int(sr * 0.2):
                            continue
                        
                        # Apply selective adaptive loudness enhancement to recovery clip
                        from services.audio_preprocessing import _apply_selective_loudness_enhancement
                        enhanced_span, rec_stats = _apply_selective_loudness_enhancement(
                            span_audio,
                            sr=sr,
                            target_dbfs=-18.0,
                        )

                        tmp_rec = None
                        try:
                            fd, tmp_rec = tempfile.mkstemp(suffix="_rec.wav")
                            os.close(fd)
                            sf.write(tmp_rec, enhanced_span, sr, subtype="PCM_16")

                            # Direct Whisper pass with VAD explicitly bypassed
                            rec_kwargs = {
                                "batch_size": 1,
                                "language": detected_lang,
                            }
                            if prompt_str and not hasattr(model, "options"):
                                rec_kwargs["initial_prompt"] = prompt_str

                            try:
                                rec_res = model.transcribe(tmp_rec, **rec_kwargs)
                            except TypeError as rec_te:
                                logger.warning(f"[LowVolumeRecovery] Retrying transcribe without unsupported kwargs: {rec_te}")
                                rec_kwargs.pop("language", None)
                                rec_kwargs.pop("initial_prompt", None)
                                rec_res = model.transcribe(tmp_rec, **rec_kwargs)

                            for r_seg in rec_res.get("segments", []):
                                txt = r_seg.get("text", "").strip()
                                if not txt:
                                    continue
                                
                                total_raw_recovered += 1
                                seg_start = round(r_seg.get("start", 0.0) + padded_start, 3)
                                seg_end = round(r_seg.get("end", 0.0) + padded_start, 3)
                                shifted_seg = dict(r_seg)
                                shifted_seg["start"] = seg_start
                                shifted_seg["end"] = seg_end

                                # Deduplicate against primary raw_segments
                                txt_lower = txt.lower()
                                is_dup = False
                                for ex in raw_segments:
                                    ex_txt = ex.get("text", "").strip().lower()
                                    ex_start = ex.get("start", 0.0)
                                    ex_end = ex.get("end", 0.0)
                                    if abs(seg_start - ex_start) < 2.5 or abs(seg_end - ex_end) < 2.5:
                                        if txt_lower in ex_txt or ex_txt in txt_lower:
                                            is_dup = True
                                            break
                                        if SequenceMatcher(None, txt_lower, ex_txt).ratio() > 0.7:
                                            is_dup = True
                                            break

                                if not is_dup:
                                    raw_segments.append(shifted_seg)
                                    total_merged_recovered += 1

                        except Exception as rec_err:
                            logger.debug(f"[LowVolumeRecovery] Recovery pass on span [{r_start:.2f}s-{r_end:.2f}s] failed: {rec_err}")
                        finally:
                            if tmp_rec and os.path.exists(tmp_rec):
                                try:
                                    os.unlink(tmp_rec)
                                except OSError:
                                    pass

                    if total_merged_recovered > 0:
                        raw_segments.sort(key=lambda s: s.get("start", 0.0))

                    logger.info(
                        f"[LowVolumeRecovery]\n"
                        f"  - Candidate Regions: {len(recovered_spans)}\n"
                        f"  - Context Padding: {pad_ms} ms\n"
                        f"  - Audio Enhancement: Selective Adaptive Loudness Enhancement Applied\n"
                        f"    - Avg Input Loudness: {rec_stats['avg_input_dbfs']} dBFS\n"
                        f"    - Quiet Windows Detected: {rec_stats['quiet_windows_pct']}%\n"
                        f"    - Avg Gain Applied: +{rec_stats['avg_gain_applied_db']} dB\n"
                        f"    - Max Gain Applied: +{rec_stats['max_gain_applied_db']} dB\n"
                        f"    - Peak Limiter Activated: {rec_stats['limiter_activated']}\n"
                        f"  - Recovery Whisper Pass: Executed (VAD Bypassed)\n"
                        f"  - Recovery Segments Produced: {total_raw_recovered}\n"
                        f"  - New Segments Merged: {total_merged_recovered}"
                    )
                else:
                    logger.info(
                        f"[LowVolumeRecovery]\n"
                        f"  - Candidate Regions: 0\n"
                        f"  - Context Padding: {speech_pad_ms} ms\n"
                        f"  - Audio Enhancement: Gain + Compression Configured\n"
                        f"  - Recovery Whisper Pass: Skipped (No low-volume candidate regions found above threshold {recovery_energy_threshold} dBFS)"
                    )
            except Exception as low_vol_err:
                logger.warning(f"[LowVolumeRecovery] Recovery pass failed ({low_vol_err}) — keeping primary segments")
    else:
        logger.info("[Transcription] Transcription VAD: Disabled — bypassing speech region detection, padding, merging, and low-volume recovery pass.")

    # ── Step 4: Normalize Text & Run Forced Alignment ─────────────────────
    vocab_terms: Optional[List[str]] = None
    if initial_prompt:
        try:
            import re
            terms_match = re.search(r"Terms:\s*([^\n]+)", initial_prompt)
            if terms_match:
                vocab_terms = [t.strip().rstrip(".") for t in terms_match.group(1).split(",") if t.strip()]
        except Exception:
            vocab_terms = None

    try:
        from services.transcript_normalizer import normalize_segments_for_alignment
        normalized_segments = normalize_segments_for_alignment(raw_segments, vocab_terms)
    except Exception as norm_err:
        logger.warning(f"[Transcription] Text normalization failed ({norm_err}) — using raw segments")
        normalized_segments = raw_segments

    # ── Step 5: Word Alignment VAD & WhisperX Alignment ───────────────────
    if enable_alignment_vad:
        alignment_speech_regions = speech_regions
        if alignment_speech_regions is None:
            alignment_speech_regions = detect_speech_regions(alignment_audio_path)
        logger.info(f"[Alignment] Alignment VAD: Enabled — running region-chunked WhisperX forced alignment on {len(alignment_speech_regions or [])} regions")
    else:
        alignment_speech_regions = None
        logger.info("[Alignment] Alignment VAD: Disabled — running full audio single-pass WhisperX forced alignment")

    model_a = None
    aligned_result = {"segments": raw_segments}

    try:
        align_model_dir, model_cache_only = _resolve_align_model_dir()
        model_name = "facebook/wav2vec2-base"
        cache_key = (detected_lang, device, model_name)

        if cache_key in _align_model_cache:
            model_a, metadata = _align_model_cache[cache_key]
        else:
            from services.device_utils import log_gpu_memory
            log_gpu_memory("Pre-load Alignment Model")
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_lang,
                device=device,
                model_name=model_name,
                model_dir=align_model_dir,
                model_cache_only=model_cache_only,
            )
            _align_model_cache[cache_key] = (model_a, metadata)

        aligned_result = _align_segments_chunked(
            normalized_segments,
            alignment_audio_path,
            model_a,
            metadata,
            device,
            speech_regions=alignment_speech_regions,
        )
        logger.info(f"[Transcription] Forced alignment complete ✓ ({len(aligned_result.get('segments', []))} segments)")
    except Exception as align_err:
        logger.warning(f"[Transcription] Alignment failed ({align_err}). Falling back to raw segments.")
        aligned_result = {"segments": raw_segments}
    finally:
        if _preprocessed_path and _preprocessed_path != file_path:
            try:
                from services.audio_preprocessing import cleanup_temp_wav
                cleanup_temp_wav(_preprocessed_path, file_path)
            except Exception:
                pass

    # ── Step 6: Normalize Output Schema & Log Stage Statistics ────────────
    segments: List[Dict[str, Any]] = []
    raw_parts: List[str] = []

    for seg in aligned_result.get("segments", []):
        seg_start = _safe_float(seg.get("start"), 0.0)
        seg_end = _safe_float(seg.get("end"), seg_start + 0.1)

        words: List[Dict[str, Any]] = []
        for w in seg.get("words", []):
            w_start = _safe_float(w.get("start"), seg_start)
            w_end = _safe_float(w.get("end"), seg_end)
            words.append({
                "word": w.get("word", "").strip(),
                "start": w_start,
                "end": w_end,
                "probability": round(float(w.get("score", w.get("probability", 1.0))), 4),
            })

        text = seg.get("text", "").strip()
        if not text:
            continue

        segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": text,
            "words": words,
            "avg_logprob": round(float(seg.get("avg_logprob", 0.0)), 4),
        })
        raw_parts.append(text)

    total_words_count = sum(len(s["words"]) for s in segments)

    logger.info(
        f"[Transcription Pipeline Final Statistics]\n"
        f"  - Audio Duration: {audio_dur:.2f}s\n"
        f"  - Preprocessing Normalization: {'Applied' if enable_audio_norm else 'Disabled'}\n"
        f"  - Low-Volume Recovery Segments Added: {recovered_segments_count}\n"
        f"  - Final Transcript Segments: {len(segments)} ({total_words_count} total words)"
    )

    return {
        "segments": segments,
        "language": detected_lang,
        "raw_text": " ".join(raw_parts),
        "aligned_result": aligned_result,
    }

