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
