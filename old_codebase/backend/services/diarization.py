"""
Speaker diarization service — 100% offline.

All pyannote models are loaded from MODELS_DIR only. No HuggingFace
downloads are attempted at any point. If required model directories are
missing, initialisation fails with a clear error listing what is absent.

Offline model layout expected under MODELS_DIR:
  audio_context/                              ← complete community-1 snapshot
    config.yaml                               ← pipeline config (uses $model/... paths)
    embedding/pytorch_model.bin               ← CAM++ speaker embedding model
    segmentation/pytorch_model.bin            ← segmentation model
    plda/plda.npz                             ← PLDA model for VBx clustering
    plda/xvec_transform.npz                   ← x-vector transform for VBx

  cam_plus_plus/campplus.onnx                 ← (removed; replaced by ecapa_tdnn/)
  ecapa_tdnn/                                 ← SpeechBrain ECAPA-TDNN (speaker ID)

The community-1 pipeline uses VBxClustering with local PLDA. PLDA must NOT
be set to null — community-1 requires it for correct speaker clustering.
The config.yaml uses $model/... paths which pyannote resolves relative to the
local pipeline directory (audio_context/). No path rewriting is needed.
"""
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from config import settings
from services.compat import apply_compatibility_patches

logger = logging.getLogger(__name__)

_diarization_pipeline = None
_pyannote_available = False

# Import device selection (safe even if torch not installed)
try:
    from services.device_utils import DEVICE as _DEVICE, get_torch_device, log_device_info
except Exception:
    _DEVICE = "cpu"
    def get_torch_device():
        import torch; return torch.device("cpu")
    def log_device_info(): pass




def _check_required_models() -> List[str]:
    """
    Verify all required model files exist inside audio_context/ in MODELS_DIR.
    The community-1 pipeline is a self-contained snapshot: all sub-models live
    under audio_context/ as sub-directories.

    Returns a list of human-readable missing asset descriptions (empty = all present).
    """
    from services.model_loader import ModelLoader
    missing = []

    audio_context_path = ModelLoader.get_model_path("audio_context")
    if audio_context_path is None or not audio_context_path.exists():
        missing.append("  • audio_context/ directory missing — pyannote/speaker-diarization-community-1 snapshot")
        return missing  # everything below depends on this directory

    # Files required inside audio_context/
    required_files = [
        ("config.yaml",                   "pipeline configuration"),
        ("embedding/pytorch_model.bin",   "CAM++ speaker embedding sub-model"),
        ("segmentation/pytorch_model.bin","segmentation sub-model"),
        ("plda/plda.npz",                 "PLDA model for VBx clustering"),
        ("plda/xvec_transform.npz",       "x-vector transform for VBx clustering"),
    ]
    for rel_path, description in required_files:
        full_path = audio_context_path / rel_path
        if not full_path.exists():
            missing.append(f"  • audio_context/{rel_path} missing — {description}")

    return missing


def _try_load_pyannote():
    global _diarization_pipeline, _pyannote_available

    # ── Step 0: Apply early compat patches (before ANY pyannote import) ──────
    # Shared patches: SpeechBrain k2/flair mocks, torchaudio API shims,
    # torchcodec warning suppression. Idempotent — safe to call repeatedly.
    apply_compatibility_patches()

    # ── Step 1: Verify all required model directories/files exist ────────────
    missing = _check_required_models()
    if missing:
        msg = (
            "[Diarization] Cannot load pyannote — the following model assets are "
            f"missing from MODELS_DIR ({settings.MODELS_DIR}):\n"
            + "\n".join(missing)
            + "\n  Run tools/download_speaker_models.py --hf-token <TOKEN> to download and restart."
        )
        logger.error(msg)
        logger.warning("[Diarization] Falling back to energy-based diarization (reduced accuracy).")
        _pyannote_available = False
        return

    try:
        from services.device_utils import log_gpu_memory
        log_gpu_memory("Pre-load Pyannote")
        from services.model_loader import ModelLoader
        audio_context_path = ModelLoader.get_model_path("audio_context")

        # ── Step 2: Import pyannote ───────────────────────────────────────────
        # community-1 config.yaml uses $model/... paths which pyannote resolves
        # relative to audio_context/. No config rewriting is needed.
        logger.info(
            f"[Diarization] Loading pyannote/speaker-diarization-community-1 from "
            f"{audio_context_path} on {_DEVICE} ..."
        )
        from pyannote.audio import Pipeline

        # ── Step 3: Load pipeline from local directory — no internet calls ────
        pipeline = Pipeline.from_pretrained(str(audio_context_path))
        pipeline = pipeline.to(get_torch_device())

        _diarization_pipeline = pipeline
        _pyannote_available = True
        logger.info(
            f"[Diarization] pyannote/speaker-diarization-community-1 ready on {_DEVICE} ✓"
        )
        log_gpu_memory("Post-load Pyannote")


    except Exception as e:
        logger.error(
            f"[Diarization] Pipeline load failed: {e}\n"
            f"  MODELS_DIR = {settings.MODELS_DIR}\n"
            "  Verify audio_context/ contains: config.yaml, embedding/, segmentation/, plda/",
            exc_info=True,
        )
        logger.warning("[Diarization] Falling back to energy-based diarization (reduced accuracy).")
        _pyannote_available = False


def get_diarization_pipeline():
    """Lazy-load pyannote pipeline on demand."""
    global _diarization_pipeline, _pyannote_available
    if _diarization_pipeline is None:
        _try_load_pyannote()
    return _diarization_pipeline


def unload_diarization_pipeline():
    """Unload the Pyannote pipeline to free RAM/VRAM."""
    global _diarization_pipeline, _pyannote_available
    if _diarization_pipeline is not None:
        from services.device_utils import log_gpu_memory
        log_gpu_memory("Pre-unload Pyannote")
        logger.info("[Diarization] Unloading pyannote pipeline...")
        del _diarization_pipeline
        _diarization_pipeline = None
        _pyannote_available = False
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[Diarization] pyannote pipeline unloaded.")
        log_gpu_memory("Post-unload Pyannote")



def init_diarization():
    log_device_info()   # print once: "Using CUDA -- GPU: ..." or "using CPU"
    # Do NOT pre-load the model to save memory until a job requires it.
    logger.info("[Diarization] Lazy diarization initialization set up.")


# ── pyannote diarization ──────────────────────────────────────────────────────

def _load_for_pyannote(file_path: str) -> dict:
    """
    Load audio via soundfile → torch tensor dict.
    Bypasses torchcodec/FFmpeg DLL issues on Windows.
    """
    import torch
    import soundfile as sf
    data, sr = sf.read(file_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (channels, samples)
    return {"waveform": waveform, "sample_rate": sr}


def _detect_overlaps_from_segments(segments: List[Dict]) -> List[Dict]:
    """
    Manual overlap detection: if two segments from DIFFERENT speakers
    overlap in time, mark both as is_overlap=True and append the
    overlapping window (with both speakers listed) to ``overlap_regions``.

    Each segment may accumulate multiple overlap_regions if it overlaps
    with several other speakers.
    """
    n = len(segments)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = segments[i], segments[j]
            if a["speaker"] == b["speaker"]:
                continue
            overlap_start = max(a["start"], b["start"])
            overlap_end   = min(a["end"],   b["end"])
            if overlap_end - overlap_start > 0.1:   # >100 ms overlap
                a["is_overlap"] = True
                b["is_overlap"] = True
                region = {
                    "start": round(overlap_start, 3),
                    "end": round(overlap_end, 3),
                    "speakers": sorted([a["speaker"], b["speaker"]]),
                }
                a.setdefault("overlap_regions", []).append(region)
                b.setdefault("overlap_regions", []).append(region)
    return segments


def _annotation_to_segments(annotation) -> List[Dict[str, Any]]:
    """Convert a pyannote Annotation to our segment list format."""
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
            "is_overlap": False,
        })
    return segments


def _merge_overlaps_into_exclusive(
    exclusive_segments: List[Dict[str, Any]],
    regular_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Stamp ``is_overlap=True`` and ``overlap_regions`` onto any
    exclusive-diarization segment whose time range overlaps (by >100 ms)
    with a region flagged as overlapping in the regular (non-exclusive)
    diarization output.

    We collect all overlap windows AND their speaker lists from the regular
    segments, then apply them to every exclusive segment that intersects.
    This keeps speaker assignments from exclusive_speaker_diarization intact
    while restoring overlap markers (and participants) derived from
    speaker_diarization.
    """
    # Collect overlap windows with speaker lists from the regular diarization
    # Each entry: (start, end, [speaker_ids...])
    overlap_windows: List[Dict[str, Any]] = []
    for seg in regular_segments:
        if seg.get("is_overlap"):
            for region in seg.get("overlap_regions", []):
                overlap_windows.append(region)

    # Deduplicate windows (same start/end/speakers may appear from both segs)
    seen: set = set()
    unique_windows: List[Dict[str, Any]] = []
    for w in overlap_windows:
        key = (w["start"], w["end"], tuple(w["speakers"]))
        if key not in seen:
            seen.add(key)
            unique_windows.append(w)

    if not unique_windows:
        return exclusive_segments  # nothing to mark

    for seg in exclusive_segments:
        for w in unique_windows:
            intersect = min(seg["end"], w["end"]) - max(seg["start"], w["start"])
            if intersect > 0.1:  # >100 ms overlap with an overlap window
                seg["is_overlap"] = True
                seg.setdefault("overlap_regions", []).append(w)

    overlap_count = sum(1 for s in exclusive_segments if s.get("is_overlap"))
    logger.debug(
        f"[Diarization] Overlap merge: {len(unique_windows)} overlap windows → "
        f"{overlap_count}/{len(exclusive_segments)} exclusive segments marked as overlap"
    )
    return exclusive_segments


def _diarize_pyannote(file_path: str) -> List[Dict[str, Any]]:
    try:
        audio_input = _load_for_pyannote(file_path)
    except Exception as e:
        logger.warning(f"[Diarization] soundfile load failed ({e}), using file path.")
        audio_input = file_path

    pipeline = get_diarization_pipeline()
    if pipeline is None:
        raise RuntimeError("Pyannote diarization pipeline is not initialized.")

    diarization = pipeline(audio_input)

    # ── community-1 / pyannote 4.x returns a DiarizeOutput object. ────────
    #
    # Strategy:
    #   • exclusive_speaker_diarization  — one active speaker per timestamp;
    #     used for all transcript speaker assignment (cleaner labels).
    #   • speaker_diarization            — may contain overlapping regions;
    #     used ONLY for overlap detection, then merged back into the
    #     exclusive-based segment list.
    #
    # This gives us clean speaker assignments AND restored overlap markers.

    if hasattr(diarization, "exclusive_speaker_diarization"):
        exclusive_annotation = diarization.exclusive_speaker_diarization
        logger.debug("[Diarization] Using DiarizeOutput.exclusive_speaker_diarization (pyannote 4.x)")
    else:
        exclusive_annotation = None
        logger.debug("[Diarization] exclusive_speaker_diarization not available")

    if hasattr(diarization, "speaker_diarization"):
        regular_annotation = diarization.speaker_diarization
        logger.debug("[Diarization] Also extracted DiarizeOutput.speaker_diarization for overlap detection")
    else:
        regular_annotation = None
        logger.debug("[Diarization] speaker_diarization not available")

    # Resolve which annotation to use for the primary segment list
    if exclusive_annotation is not None:
        primary_annotation = exclusive_annotation
    elif regular_annotation is not None:
        primary_annotation = regular_annotation
        logger.debug("[Diarization] Falling back to speaker_diarization for primary segments (pyannote 4.x fallback)")
    else:
        primary_annotation = diarization
        logger.debug("[Diarization] Using Annotation directly (pyannote 3.x compat)")

    # Build primary segments from the exclusive (or fallback) annotation
    segments = _annotation_to_segments(primary_annotation)

    logger.info(
        f"[Diarization] pyannote/community-1 found {len(segments)} segments, "
        f"speakers: {sorted({s['speaker'] for s in segments})}"
    )

    # ── Overlap detection via regular (non-exclusive) diarization ─────────
    # Run _detect_overlaps_from_segments on the regular annotation so that
    # truly overlapping speech regions are identified, then merge those
    # overlap markers back into the exclusive-based segment list.
    if exclusive_annotation is not None and regular_annotation is not None:
        regular_segments = _annotation_to_segments(regular_annotation)
        regular_segments_with_overlaps = _detect_overlaps_from_segments(regular_segments)
        overlap_count_regular = sum(1 for s in regular_segments_with_overlaps if s.get("is_overlap"))
        logger.info(
            f"[Diarization] Overlap detection on speaker_diarization: "
            f"{overlap_count_regular}/{len(regular_segments)} segments flagged as overlap"
        )
        segments = _merge_overlaps_into_exclusive(segments, regular_segments_with_overlaps)
    else:
        # No separate regular annotation — fall back to detecting overlaps directly
        # on the primary segments (may miss true overlaps if exclusive is used, but
        # better than nothing)
        segments = _detect_overlaps_from_segments(segments)

    return segments


# ── Fallback: energy-based diarization ───────────────────────────────────────

def _diarize_energy(file_path: str, min_seg_dur: float = 1.5) -> List[Dict[str, Any]]:
    """
    Simple energy-based speaker change detection.
    Groups frames by high/low energy and assigns alternating speaker labels.
    Works without HF_TOKEN — used when pyannote is unavailable.
    """
    import librosa
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    hop_length = 512
    frame_length = 2048

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    threshold = np.percentile(rms, 30)
    speech_mask = rms > threshold

    segments = []
    current_speaker = 0
    in_speech = False
    seg_start = 0.0
    SILENCE_GAP = 0.8  # seconds gap to trigger speaker change
    min_frames_silence = int(SILENCE_GAP * sr / hop_length)
    silence_count = 0

    for i, is_speech in enumerate(speech_mask):
        if is_speech:
            if not in_speech:
                seg_start = float(times[i])
                in_speech = True
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_frames_silence:
                    end = float(times[i - min_frames_silence // 2])
                    dur = end - seg_start
                    if dur >= min_seg_dur:
                        segments.append({
                            "start": round(seg_start, 3),
                            "end": round(end, 3),
                            "speaker": f"SPEAKER_{current_speaker:02d}",
                            "is_overlap": False,
                        })
                        current_speaker = (current_speaker + 1) % 2  # alternate
                    in_speech = False
                    silence_count = 0

    # close last segment
    if in_speech:
        end = float(times[-1])
        dur = end - seg_start
        if dur >= min_seg_dur:
            segments.append({
                "start": round(seg_start, 3),
                "end": round(end, 3),
                "speaker": f"SPEAKER_{current_speaker:02d}",
                "is_overlap": False,
            })

    return segments


# ── Public API ────────────────────────────────────────────────────────────────

def diarize(file_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of diarization segments:
    [{"start": float, "end": float, "speaker": str, "is_overlap": bool}]
    """
    min_dur = settings.MIN_SEGMENT_DURATION
    if is_pyannote_available():
        raw = _diarize_pyannote(file_path)
    else:
        raw = _diarize_energy(file_path, min_seg_dur=min_dur)

    # filter out too-short segments
    return [s for s in raw if (s["end"] - s["start"]) >= min_dur]


def is_pyannote_available() -> bool:
    global _pyannote_available
    if _diarization_pipeline is not None:
        return True
    missing = _check_required_models()
    if len(missing) == 0:
        _pyannote_available = True
        return True
    _pyannote_available = False
    return False
