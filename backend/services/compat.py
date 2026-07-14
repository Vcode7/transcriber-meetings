"""
Shared compatibility patches for SpeechBrain, torchaudio, and torchcodec.

These patches MUST be applied before importing any of the following:
  - whisperx          (bundles Pyannote VAD which triggers SpeechBrain)
  - pyannote.audio
  - speechbrain
  - ECAPA-TDNN encoder

Call ``apply_compatibility_patches()`` as early as possible — typically:
  1. At the top of ``main.py`` (process-wide, before any service is imported).
  2. At the start of each lazy-load function (belt-and-suspenders, since the
     order of first model use at runtime can differ from import order).

All functions in this module are idempotent: calling them multiple times is
safe and cheap.
"""
import logging
import sys
import warnings

logger = logging.getLogger(__name__)

# Track whether we have already applied the patches so repeat calls are no-ops.
_patches_applied: bool = False


def suppress_torchcodec_warning() -> None:
    """
    torchcodec DLL loading fails on Windows when FFmpeg DLLs are not installed.
    Non-fatal since we use soundfile for audio loading, but produces a very
    long warning block in logs. Suppress it proactively.
    """
    warnings.filterwarnings(
        "ignore",
        message="torchcodec is not installed correctly",
        category=UserWarning,
    )


def patch_speechbrain() -> None:
    """
    Prevent SpeechBrain 1.x lazy imports from failing when optional packages
    (k2, flair) are not installed.

    These integrations are not needed for Pyannote speaker diarization or
    ECAPA-TDNN embeddings — they are lazily imported only when explicitly
    requested.  However, Python's ``inspect.stack()`` in pytorch-lightning
    accidentally triggers them during model loading.

    We pre-register mock modules so the lazy loader never attempts the real
    import.  Confirmed fix for:
      ``ImportError: Lazy import of LazyModule(package=None,
        target=speechbrain.integrations.k2_fsa, loaded=False) failed``
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
    registered = []
    for mod_name in mocks_needed:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()
            registered.append(mod_name)

    if registered:
        logger.debug(
            "[Compat] SpeechBrain optional-dependency mocks registered: %s",
            registered,
        )


def patch_torchaudio() -> None:
    """
    Compatibility shim: newer torchaudio (2.4+) removed ``AudioMetaData`` from
    the top-level namespace.  pyannote.audio still references it there.
    """
    try:
        import torchaudio
        if hasattr(torchaudio, "AudioMetaData"):
            return  # already fine

        # Walk known relocation paths
        import importlib
        for mod_path, attr in [
            ("torchaudio.backend.common", "AudioMetaData"),
            ("torchaudio._backend",       "AudioMetaData"),
            ("torchaudio.io",             "AudioMetaData"),
        ]:
            try:
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, attr)
                torchaudio.AudioMetaData = cls
                logger.debug(
                    "[Compat] torchaudio.AudioMetaData patched from %s", mod_path
                )
                return
            except Exception:
                continue

        # Last resort: create a namedtuple with the expected fields
        from collections import namedtuple
        torchaudio.AudioMetaData = namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )
        logger.warning("[Compat] torchaudio.AudioMetaData shimmed with namedtuple.")
    except ImportError:
        pass  # torchaudio not installed — that's OK


def patch_torchaudio_backend() -> None:
    """Patch missing ``set_audio_backend`` on newer torchaudio."""
    try:
        import torchaudio
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend: None
            logger.debug("[Compat] torchaudio.set_audio_backend patched.")
    except ImportError:
        pass


def patch_torch_cuda_memory_apis() -> None:
    """
    Patch deprecated and removed PyTorch CUDA memory management APIs.
    Newer PyTorch versions (2.0+) deprecated `torch.cuda.memory_cached` and
    `torch.cuda.max_memory_cached` in favor of `memory_reserved`.
    Some legacy model packages still call the deprecated ones, causing
    crashes or warnings on newer PyTorch/GPU configurations.
    """
    try:
        import torch
        if hasattr(torch, "cuda"):
            if not hasattr(torch.cuda, "memory_cached"):
                torch.cuda.memory_cached = torch.cuda.memory_reserved
                logger.debug("[Compat] Patched torch.cuda.memory_cached -> memory_reserved")
            if not hasattr(torch.cuda, "max_memory_cached"):
                torch.cuda.max_memory_cached = torch.cuda.max_memory_reserved
                logger.debug("[Compat] Patched torch.cuda.max_memory_cached -> max_memory_reserved")
    except ImportError:
        pass


def apply_compatibility_patches() -> None:
    """
    Apply all compatibility patches in the correct order.

    This function is **idempotent** — subsequent calls are no-ops after the
    first successful application.

    Order matters:
      1. Suppress noisy warnings first (no side-effects).
      2. Mock SpeechBrain optional dependencies BEFORE any pyannote/whisperx
         import so the lazy-import machinery never attempts the real import.
      3. Fix torchaudio API differences (also needed before pyannote).
      4. Patch legacy PyTorch CUDA memory APIs.
    """
    global _patches_applied
    if _patches_applied:
        return

    suppress_torchcodec_warning()
    patch_speechbrain()        # MUST come before any pyannote / whisperx import
    patch_torchaudio()
    patch_torchaudio_backend()
    patch_torch_cuda_memory_apis()

    _patches_applied = True
    logger.debug("[Compat] All compatibility patches applied.")
