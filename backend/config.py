"""Application settings — loaded from .env file."""
import sys
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


def _resolve_base_dir() -> Path:
    """
    Return the application base directory.
    - When running as a PyInstaller .exe: parent of sys.executable
    - When running normally (development): parent of this config.py file
    """
    if getattr(sys, "frozen", False):
        # Running as compiled PyInstaller executable
        return Path(sys.executable).parent
    return Path(__file__).parent


BASE_DIR = _resolve_base_dir()
RUNTIME_DIR = BASE_DIR / "runtime"


class Settings(BaseSettings):
    # SQLite database — relative to runtime/
    DATABASE_URL: str = f"sqlite+aiosqlite:///{(RUNTIME_DIR / 'data' / 'voicesum.db').as_posix()}"

    # JWT — Access token (short-lived, in-memory on client)
    JWT_SECRET: str = "change-me-in-production-use-long-random-string"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 15  # 15 minutes — refresh token extends session

    # Refresh token (long-lived, HttpOnly cookie)
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Environment ("development" | "production") — controls Secure cookie flag
    ENVIRONMENT: str = "development"

    # Rate limiting — login endpoint
    RATE_LIMIT_LOGIN_MAX: int = 5          # max failed attempts
    RATE_LIMIT_LOGIN_WINDOW_SECONDS: int = 300  # 5-minute window
    ACCOUNT_LOCKOUT_SECONDS: int = 900     # 15-minute lockout after max attempts

    # Local Qwen3 4B Instruct (4-bit quantized) — offline inference
    QWEN_MODEL_ID: str = "Qwen/Qwen3-4B"
    QWEN_MAX_NEW_TOKENS: int = 1024
    QWEN_LOAD_IN_4BIT: bool = True  # Requires bitsandbytes; saves ~50% VRAM

    # HuggingFace (optional — enables pyannote diarization + Qwen3 download)
    HF_TOKEN: Optional[str] = ""

    # Audio storage — relative to runtime/
    UPLOAD_DIR: str = str(RUNTIME_DIR / "uploads")

    # Models directory (encrypted .dat files in production)
    MODELS_DIR: str = str(RUNTIME_DIR / "models")

    # Offline mode — when True, never attempt internet downloads
    OFFLINE_MODE: bool = False

    # Speaker identification
    SPEAKER_SIMILARITY_THRESHOLD: float = 0.65
    MIN_SEGMENT_DURATION: float = 1.5  # seconds

    # Transcription
    WHISPER_MODEL_SIZE: str = "medium"
    WHISPER_DEVICE: str = "auto"  # "cuda", "cpu", "auto"
    WHISPER_COMPUTE_TYPE: str = "int8"

    # Word confidence thresholds
    WORD_CONF_LOW: float = 0.7
    WORD_CONF_MID: float = 0.85

    # Overlap detection model (Wav2Vec2-based binary classifier)
    # Default is relative to base dir; override in .env with absolute path if needed
    OVERLAP_MODEL_PATH: str = str(BASE_DIR / "checkpoints" / "overlap_model.pth")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

# Ensure critical runtime directories exist at import time
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(Path(settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")).parent, exist_ok=True)
