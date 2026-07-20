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

def _resolve_runtime_dir() -> Path:
    """
    Return the application runtime directory.
    - When running as a PyInstaller .exe: sibling 'runtime' folder to backend (i.e. sys.executable's grandparent / runtime)
    - When running normally (development): BASE_DIR / "runtime"
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.parent / "runtime"
    return BASE_DIR / "runtime"

RUNTIME_DIR = _resolve_runtime_dir()

def _resolve_models_dir() -> Path:
    # In development mode, check if sibling Application directory has models
    dev_models = BASE_DIR.parent / "Application" / "runtime" / "models"
    if dev_models.is_dir() and not getattr(sys, "frozen", False):
        return dev_models
    dev_models_backend = BASE_DIR.parent / "Application" / "backend" / "runtime" / "models"
    if dev_models_backend.is_dir() and not getattr(sys, "frozen", False):
        return dev_models_backend
    return RUNTIME_DIR / "models"

DEFAULT_MODELS_DIR = _resolve_models_dir()


class Settings(BaseSettings):
    # SQLite database — relative to runtime/
    DATABASE_URL: str = f"sqlite+aiosqlite:///{(RUNTIME_DIR / 'data' / 'voicesum.db').as_posix()}"

    # ── RAG / Text Embedding ──────────────────────────────────────────────────
    # The active embedding model to load from local runtime directory.
    # Supported: "Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B-Instruct-INT8", etc.
    EMBEDDING_MODEL: str = "Qwen3-Embedding-0.6B"

    # Backward-compatible model name. Automatically synchronized with EMBEDDING_MODEL in __init__.
    QWEN_EMBEDDING_MODEL_NAME: Optional[str] = None

    @property
    def QWEN_EMBEDDING_MODEL_DIR(self) -> str:
        """
        Resolved path to the embedding model directory.
        Checks Application/runtime/embeddings/<model_name>/ first, then falls back
        to <runtime_dir>/embeddings/<model_name>/.
        """
        dev_path = BASE_DIR.parent / "Application" / "runtime" / "embeddings" / self.EMBEDDING_MODEL
        if dev_path.is_dir() and not getattr(sys, "frozen", False):
            return str(dev_path)
        return str(RUNTIME_DIR / "embeddings" / self.EMBEDDING_MODEL)

    # FAISS vector store base directory
    VECTOR_STORE_DIR: str = str(RUNTIME_DIR / "vector_store")

    # RAG chunking parameters
    RAG_CHUNK_SIZE: int = 400        # target words per chunk
    RAG_CHUNK_OVERLAP: int = 50      # words of overlap between chunks

    # RAG retrieval — top-K per source
    RAG_RETRIEVAL_K_GLOBAL: int = 2      # global context docs
    RAG_RETRIEVAL_K_MEETING: int = 3     # meeting context attachments
    RAG_RETRIEVAL_K_TRANSCRIPT: int = 10  # transcript chunks
    RAG_RELATIVE_SCORE_CUTOFF: float = 0.01  # similarity score window

    # JWT — Access token (short-lived, in-memory on client)
    JWT_SECRET: str = "change-me-in-production-use-long-random-string"
    JWT_ALGORITHM: str = "HS256"

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

    # Ollama offline fallback settings
    OLLAMA_SERVER_URL: str = "http://localhost:11434"
    OLLAMA_PORT: int = 11434
    OLLAMA_MODEL_PRIORITY: str = "gemma,qwen,llama,deepseek,mistral"

    # Configurable token threshold for switching to Section-wise MoM Generation
    MOM_CONTEXT_TOKEN_THRESHOLD: int = 3000

    # HuggingFace (optional — enables pyannote diarization + Qwen3 download)
    HF_TOKEN: Optional[str] = ""

    # Audio storage — relative to runtime/
    UPLOAD_DIR: str = str(RUNTIME_DIR / "uploads")

    # Models directory (encrypted .dat files in production)
    MODELS_DIR: str = str(DEFAULT_MODELS_DIR)

    # Offline mode — when True, never attempt internet downloads
    OFFLINE_MODE: bool = True


    # Speaker identification
    # --- Embedding model used for speaker ID ---
    # Valid values: "ecapa_tdnn" (others reserved for future)
    SPEAKER_EMBEDDING_MODEL: str = "ecapa_tdnn"

    # --- Similarity thresholds ---
    # SPEAKER_SIMILARITY_THRESHOLD is the user-visible global default.
    # ECAPA-TDNN cosine scores differ from CAM++ — 0.75 is a good starting point.
    SPEAKER_SIMILARITY_THRESHOLD: float = 0.75          # backward-compat user default
    SPEAKER_SIMILARITY_THRESHOLD_ECAPA_TDNN: float = 0.72  # model-specific default

    MIN_SEGMENT_DURATION: float = 1.5  # seconds

    speaker_refinement_margin: float = 0.30  # margins/similarity difference for refinement

    # --- Audio preprocessing before alignment ---
    # When True, a lightweight cleanup pass (silence trim, clipping repair,
    # loudness normalization) is applied to the audio before WhisperX alignment.
    # Set to False to skip preprocessing and use the raw WAV directly.
    AUDIO_PREPROCESS_BEFORE_ALIGNMENT: bool = True

    # Transcription
    WHISPER_MODEL_SIZE: str = "large-v3"
    WHISPER_DEVICE: str = "auto"  # "cuda", "cpu", "auto"
    WHISPER_COMPUTE_TYPE: str = "int8"

    # Word confidence thresholds
    WORD_CONF_LOW: float = 0.7
    WORD_CONF_MID: float = 0.85

    # Minimum average segment confidence for downstream AI processing.
    # Segments whose average word probability < this threshold are excluded
    # from MoM and AI Insights generation (but remain in the transcript).
    MIN_AVG_SEGMENT_CONFIDENCE: float = 0.40

    # Overlap detection model (Wav2Vec2-based binary classifier)
    # Default is relative to base dir; override in .env with absolute path if needed
    OVERLAP_MODEL_PATH: str = str(BASE_DIR / "checkpoints" / "overlap_model.pth")

    def __init__(self, **values):
        super().__init__(**values)
        if self.QWEN_EMBEDDING_MODEL_NAME is not None:
            self.EMBEDDING_MODEL = self.QWEN_EMBEDDING_MODEL_NAME
        else:
            self.QWEN_EMBEDDING_MODEL_NAME = self.EMBEDDING_MODEL

    model_config = {"env_file": str(BASE_DIR / ".env"), "extra": "ignore"}


settings = Settings()

# Ensure critical runtime directories exist at import time
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(Path(settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")).parent, exist_ok=True)
os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)

# Print resolved paths to standard output
print(f"[Config] Resolved MODELS_DIR to: {settings.MODELS_DIR}")
print(f"[Config] Resolved VECTOR_STORE_DIR to: {settings.VECTOR_STORE_DIR}")
print(f"[Config] Resolved QWEN_EMBEDDING_MODEL_DIR to: {settings.QWEN_EMBEDDING_MODEL_DIR}")

