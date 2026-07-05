"""
Speaker diarization service — 100% offline.

All pyannote models are loaded from MODELS_DIR only. No HuggingFace
downloads are attempted at any point. If required model directories are
missing, initialisation fails with a clear error listing what is absent.

Offline model layout expected under MODELS_DIR:
  audio_context/  — pyannote/speaker-diarization-3.1 snapshot (config.yaml)
  voice_segment/  — pyannote/segmentation-3.0 snapshot (pytorch_model.bin)
  wespeaker/      — hbredin/wespeaker-voxceleb-resnet34-LM ONNX snapshot
"""
import logging
import sys
import warnings
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from config import settings

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


# ── Compatibility patches ─────────────────────────────────────

def _suppress_torchcodec_warning():
    """
    torchcodec DLL loading fails on Windows when FFmpeg DLLs are not installed.
    Non-fatal since we use soundfile for audio loading, but produces a
    very long warning block in logs. Suppress it proactively.
    """
    warnings.filterwarnings(
        "ignore",
        message="torchcodec is not installed correctly",
        category=UserWarning,
    )


def _patch_speechbrain():
    """
    Prevent speechbrain 1.x lazy imports from failing when optional
    packages (k2, flair) are not installed. These integrations are not
    needed for pyannote speaker diarization — they are lazily imported
    only when explicitly requested. However, Python's inspect.stack()
    in pytorch-lightning accidentally triggers them during model loading.

    We pre-register mock modules so the lazy loader never attempts the
    real import.  Confirmed fix (tested successfully).
    """
    from unittest.mock import MagicMock

    mocks_needed = [
        "k2",
        "speechbrain.integrations.k2_fsa",
        "speechbrain.integrations.nlp",
        "speechbrain.integrations.nlp.flair_embeddings",
        "flair",
        "flair.data",
        "flair.embeddings",
    ]
    for mod_name in mocks_needed:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()
    logger.debug("[Diarization] speechbrain optional-dependency mocks registered.")


def _patch_torchaudio_backend():
    """Patch missing set_audio_backend on newer torchaudio."""
    try:
        import torchaudio
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend: None
            logger.info("[Diarization] patched torchaudio.set_audio_backend")
    except ImportError:
        pass


def _patch_torchaudio():
    """
    Compatibility shim: newer torchaudio (2.4+) removed AudioMetaData from
    the top-level namespace. pyannote.audio still references it there.
    """
    try:
        import torchaudio
        if hasattr(torchaudio, "AudioMetaData"):
            return  # already fine
        # Walk known relocation paths
        for mod_path, attr in [
            ("torchaudio.backend.common", "AudioMetaData"),
            ("torchaudio._backend",       "AudioMetaData"),
            ("torchaudio.io",             "AudioMetaData"),
        ]:
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, attr)
                torchaudio.AudioMetaData = cls
                logger.info(f"[Diarization] torchaudio.AudioMetaData patched from {mod_path}")
                return
            except Exception:
                continue
        # Last resort: create a namedtuple with the expected fields
        from collections import namedtuple
        torchaudio.AudioMetaData = namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )
        logger.warning("[Diarization] torchaudio.AudioMetaData shimmed with namedtuple.")
    except ImportError:
        pass  # torchaudio not installed at all — that's OK


def _patch_pyannote_plda():
    """
    Bypass ALL PLDA-related network calls in pyannote.audio.

    Root cause:
      speaker_diarization.py does:
          from pyannote.audio.pipelines.utils import get_plda   # line 47
      then calls:
          self._plda = get_plda(plda, token=token, cache_dir=cache_dir)  # line 231

      The default `plda` parameter is a dict that points to a gated HF repo.
      Since the name `get_plda` is bound INTO speaker_diarization.py's namespace
      at import time, patching getter.get_plda alone does NOT fix it.

    Fix:
      This function MUST be called AFTER `from pyannote.audio import Pipeline` so
      that speaker_diarization.py is already in sys.modules. We then replace the
      `get_plda` name in that module's namespace with a no-op mock.
      We also patch utils and getter for belt-and-suspenders coverage.
    """
    try:
        def mock_get_plda(plda, token=None, cache_dir=None):
            """Return None for any plda input — PLDA is unused with AgglomerativeClustering."""
            if plda is not None:
                logger.info(
                    f"[Diarization] Intercepted PLDA load request (plda={type(plda).__name__}) "
                    "— skipping (not needed for AgglomerativeClustering)."
                )
            return None

        patched = []

        # 1. Patch the getter module attribute
        import pyannote.audio.pipelines.utils.getter as getter_mod
        getter_mod.get_plda = mock_get_plda
        patched.append("pyannote.audio.pipelines.utils.getter")

        # 2. Patch the utils package re-export
        import pyannote.audio.pipelines.utils as utils_mod
        utils_mod.get_plda = mock_get_plda
        patched.append("pyannote.audio.pipelines.utils")

        # 3. Patch INSIDE speaker_diarization's own namespace (critical!)
        #    This only works after speaker_diarization.py has been imported.
        import sys
        sd_mod_name = "pyannote.audio.pipelines.speaker_diarization"
        if sd_mod_name in sys.modules:
            sys.modules[sd_mod_name].get_plda = mock_get_plda
            patched.append(sd_mod_name)
        else:
            logger.warning(
                "[Diarization] speaker_diarization not yet imported — "
                "will re-patch after import."
            )

        logger.info(f"[Diarization] PLDA loader patched in: {', '.join(patched)}")
    except Exception as e:
        logger.warning(f"[Diarization] Failed to patch pyannote PLDA loader: {e}")


def _rewrite_config_yaml_with_local_paths(
    audio_context_path: Path,
    voice_segment_path: Path,
    wespeaker_model_path: Path,
) -> Path:
    """
    Read the pyannote pipeline config.yaml and replace all remote HF Hub references
    with absolute local paths. Also sets plda: null to prevent any PLDA download.

    Changes made:
      - embedding: <HF-ID> → <absolute local wespeaker path>
      - segmentation: <HF-ID> → <absolute local voice_segment path>
      - plda: <HF dict or ID> → null  (PLDA unused with AgglomerativeClustering)

    Returns the path to the updated config.yaml.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")

    config_path = audio_context_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"[Diarization] Missing config.yaml at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline_params = config.get("pipeline", {}).get("params", {})
    changed = False

    # ── embedding → local wespeaker path ──────────────────────────────────
    current_embedding = str(pipeline_params.get("embedding", ""))
    wespeaker_abs = str(wespeaker_model_path.resolve())
    if current_embedding != wespeaker_abs:
        pipeline_params["embedding"] = wespeaker_abs
        changed = True
        logger.info(f"[Diarization] config.yaml: embedding '{current_embedding}' → {wespeaker_abs}")

    # ── segmentation → local voice_segment path ───────────────────────────
    current_segmentation = str(pipeline_params.get("segmentation", ""))
    voice_segment_abs = str(voice_segment_path.resolve())
    if current_segmentation != voice_segment_abs:
        pipeline_params["segmentation"] = voice_segment_abs
        changed = True
        logger.info(f"[Diarization] config.yaml: segmentation '{current_segmentation}' → {voice_segment_abs}")

    # ── plda → null (prevents all PLDA network calls) ─────────────────────
    # The default value in SpeakerDiarization.__init__ is a dict pointing to
    # pyannote/speaker-diarization-community-1. Setting plda=null in config.yaml
    # makes from_pretrained pass plda=None to __init__, and our mock_get_plda
    # handles None by returning None immediately.
    current_plda = pipeline_params.get("plda", "__not_set__")
    if current_plda != "__not_set__" and current_plda is not None:
        pipeline_params["plda"] = None
        changed = True
        logger.info(f"[Diarization] config.yaml: plda '{current_plda}' → null (PLDA disabled)")
    elif "plda" not in pipeline_params:
        # Explicitly add it so the default dict in __init__ is overridden
        pipeline_params["plda"] = None
        changed = True
        logger.info("[Diarization] config.yaml: added 'plda: null' to disable PLDA loading")

    if changed:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"[Diarization] config.yaml saved: {config_path}")
    else:
        logger.info("[Diarization] config.yaml already fully configured — no changes needed.")

    return config_path


def _check_required_models() -> List[str]:
    """
    Verify all required diarization model directories AND key files exist in MODELS_DIR.
    Returns a list of human-readable missing asset descriptions (empty = all present).
    """
    from services.model_loader import ModelLoader
    missing = []

    checks = [
        ("audio_context", "config.yaml",        "pyannote diarization pipeline config"),
        ("voice_segment",  "pytorch_model.bin",  "pyannote/segmentation-3.0 weights"),
        ("wespeaker",      "speaker-embedding.onnx", "WeSpeaker ONNX embedding model"),
    ]

    for model_name, required_file, description in checks:
        model_path = ModelLoader.get_model_path(model_name)
        if model_path is None or not model_path.exists():
            missing.append(f"  • {model_name}/ directory missing — {description}")
        elif not (model_path / required_file).exists():
            missing.append(
                f"  • {model_name}/{required_file} missing — {description} "
                f"(directory exists but required file not found)"
            )

    return missing


def _try_load_pyannote():
    global _diarization_pipeline, _pyannote_available

    # ── Step 0: Apply early compat patches (before ANY pyannote import) ──
    _suppress_torchcodec_warning()
    _patch_speechbrain()      # must mock k2/flair before any pyannote import
    _patch_torchaudio()
    _patch_torchaudio_backend()

    # ── Step 1: Verify all required model directories/files exist ──────────
    missing = _check_required_models()
    if missing:
        msg = (
            "[Diarization] Cannot load pyannote — the following model assets are "
            f"missing from MODELS_DIR ({settings.MODELS_DIR}):\n"
            + "\n".join(missing)
            + "\n  Copy the missing model folders into MODELS_DIR and restart."
        )
        logger.error(msg)
        logger.warning("[Diarization] Falling back to energy-based diarization (reduced accuracy).")
        _pyannote_available = False
        return

    try:
        from services.model_loader import ModelLoader

        audio_context_path = ModelLoader.get_model_path("audio_context")
        voice_segment_path  = ModelLoader.get_model_path("voice_segment")
        wespeaker_path      = ModelLoader.get_model_path("wespeaker")
        wespeaker_model_path = wespeaker_path / "speaker-embedding.onnx"

        # ── Step 2: Patch config.yaml before any pyannote import ─────────────
        # Sets embedding/segmentation to absolute local paths and plda=null.
        # Must run before pyannote reads config.yaml in from_pretrained.
        _rewrite_config_yaml_with_local_paths(
            audio_context_path,
            voice_segment_path,
            wespeaker_model_path,
        )

        # ── Step 3: Import pyannote (triggers speaker_diarization.py loading) ─
        logger.info(f"[Diarization] Loading pyannote speaker-diarization-3.1 on {_DEVICE} ...")
        from pyannote.audio import Pipeline

        # ── Step 4: Patch PLDA AFTER import so speaker_diarization's namespace is live ──
        # speaker_diarization.py does `from utils import get_plda` at module load time,
        # binding the name in its own namespace. We MUST patch after it's loaded.
        _patch_pyannote_plda()

        # ── Step 5: Load pipeline from local directory — no internet calls ────
        logger.info(f"[Diarization] Loading pipeline from: {audio_context_path}")
        pipeline = Pipeline.from_pretrained(str(audio_context_path))
        pipeline = pipeline.to(get_torch_device())

        _diarization_pipeline = pipeline
        _pyannote_available = True
        logger.info(f"[Diarization] pyannote speaker-diarization-3.1 ready on {_DEVICE} ✓")

    except Exception as e:
        logger.error(
            f"[Diarization] Pipeline load failed: {e}\n"
            f"  MODELS_DIR = {settings.MODELS_DIR}\n"
            "  Verify audio_context/, voice_segment/, wespeaker/ are present "
            "and contain the expected model files.",
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


def init_diarization():
    log_device_info()   # print once: "Using CUDA -- GPU: ..." or "using CPU"
    # Do NOT pre-load the model to save memory until a job requires it.
    logger.info("[Diarization] Lazy diarization initialization set up.")


# ── pyannote diarization ───────────────────────────────────────

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
    overlap in time, mark both as is_overlap=True.
    """
    n = len(segments)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = segments[i], segments[j]
            if a["speaker"] == b["speaker"]:
                continue
            overlap_start = max(a["start"], b["start"])
            overlap_end   = min(a["end"],   b["end"])
            if overlap_end - overlap_start > 0.1:   # >100ms overlap
                a["is_overlap"] = True
                b["is_overlap"] = True
    return segments


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
    # Support both pyannote.audio 4.x (returns DiarizeOutput) and 3.x (returns Annotation)
    if hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
            "is_overlap": False,
        })

    logger.info(f"[Diarization] pyannote found {len(segments)} segments, speakers: {list({s['speaker'] for s in segments})}")
    return _detect_overlaps_from_segments(segments)


# ── Fallback: energy-based diarization ───────────────────────

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


# ── Public API ─────────────────────────────────────────────────

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

