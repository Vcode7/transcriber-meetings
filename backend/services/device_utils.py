"""
device_utils.py - Centralized CUDA / CPU device selection for all ML services.

Detects GPU availability once at import time and exposes:
  DEVICE                - "cuda" or "cpu"
  get_torch_device()    - returns torch.device
  pipeline_device_arg() - returns 0 (CUDA) or -1 (CPU) for HuggingFace pipelines
  log_device_info()     - logs a clean startup summary
"""
import logging

logger = logging.getLogger(__name__)

# Resolve once at import time
try:
    import torch as _torch
    _cuda_available: bool = _torch.cuda.is_available()
    _gpu_name: str = _torch.cuda.get_device_name(0) if _cuda_available else ""
    _gpu_count: int = _torch.cuda.device_count() if _cuda_available else 0
    print("cuda", _cuda_available)
except Exception:
    _cuda_available = False
    _gpu_name = ""
    _gpu_count = 0

DEVICE: str = "cuda" if _cuda_available else "cpu"


def get_torch_device():
    """Return torch.device for the selected backend."""
    import torch
    return torch.device(DEVICE)


def pipeline_device_arg() -> int:
    """
    HuggingFace transformers pipeline device argument:
      0  -> first CUDA GPU
     -1  -> CPU
    """
    return 0 if _cuda_available else -1


def log_device_info() -> None:
    """Call once at startup to emit a clear device selection log line."""
    if _cuda_available:
        extra = f" x{_gpu_count}" if _gpu_count > 1 else ""
        logger.info(f"[Device] CUDA available -- using GPU: {_gpu_name}{extra}")
    else:
        logger.info("[Device] CUDA not available -- all ML models will run on CPU.")


def get_loaded_models() -> list:
    """Detect which models are currently loaded in RAM/VRAM by checking references."""
    loaded = []
    import sys
    try:
        if "services.transcription" in sys.modules:
            from services.transcription import _whisperx_model, _align_model_cache
            if _whisperx_model is not None:
                loaded.append("WhisperX")
            if _align_model_cache:
                loaded.append(f"AlignmentCache(count={len(_align_model_cache)})")
    except Exception:
        pass

    try:
        if "services.diarization" in sys.modules:
            from services.diarization import _diarization_pipeline
            if _diarization_pipeline is not None:
                loaded.append("PyannoteDiarization")
    except Exception:
        pass

    try:
        if "services.embedding" in sys.modules:
            from services.embedding import _ecapa_classifier
            if _ecapa_classifier is not None:
                loaded.append("ECAPA-TDNN")
    except Exception:
        pass

    try:
        if "services.ai_provider" in sys.modules:
            from services.ai_provider import QwenProvider
            if QwenProvider._pipeline is not None or QwenProvider._model is not None:
                loaded.append("QwenLLM")
    except Exception:
        pass

    try:
        if "main" in sys.modules:
            import main
            if getattr(main, "_overlap_model", None) is not None:
                loaded.append("OverlapClassifier")
    except Exception:
        pass

    return loaded


def log_gpu_memory(stage: str) -> None:
    """Log VRAM allocation, reservation, free memory, and currently loaded models."""
    try:
        import torch
        loaded_models = get_loaded_models()
        models_str = ", ".join(loaded_models) if loaded_models else "None"
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_mem = free_bytes / (1024 * 1024)
                total_mem = total_bytes / (1024 * 1024)
                logger.info(
                    f"[VRAM Diagnostics] {stage} — "
                    f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, "
                    f"Free: {free_mem:.2f} MB / Total: {total_mem:.2f} MB. "
                    f"Loaded Models: [{models_str}]"
                )
            except Exception:
                logger.info(
                    f"[VRAM Diagnostics] {stage} — "
                    f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB. "
                    f"Loaded Models: [{models_str}]"
                )
        else:
            logger.info(f"[VRAM Diagnostics] {stage} — CUDA not available. Loaded Models: [{models_str}]")
    except Exception as e:
        logger.warning(f"[VRAM Diagnostics] Failed to query GPU memory: {e}")

