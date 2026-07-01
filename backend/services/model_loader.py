"""
Model Loader — Phase 3
Transparent decryption and loading of encrypted model .dat files.

Usage in backend services:
    from services.model_loader import ModelLoader
    model_dir = ModelLoader.get_model_path("speech_engine")

The loader:
1. Reads the encrypted .dat file from MODELS_DIR
2. Decrypts to a temporary directory under %TEMP%/voicesum_runtime/
3. Returns the path for the caller to use
4. Cleans up temp files when the process exits

In development mode (OFFLINE_MODE=false), falls back to HF cache if .dat not found.
"""
from __future__ import annotations

import atexit
import hashlib
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Magic header written by encrypt_models.py
_MAGIC = b"VSDAT\x01"

# Tracks temp dirs created this session → cleaned up on exit
_temp_dirs: list[Path] = []


def _cleanup_temp_dirs():
    """Remove all temporary decrypted model directories on exit."""
    for d in _temp_dirs:
        try:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                logger.debug(f"[ModelLoader] Cleaned temp dir: {d}")
        except Exception:
            pass


atexit.register(_cleanup_temp_dirs)


def _get_fernet():
    """Load Fernet instance using the embedded key."""
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        raise RuntimeError(
            "cryptography package is required for encrypted model loading. "
            "Install with: pip install cryptography"
        )

    key = _load_key()
    if key is None:
        return None
    return Fernet(key)


def _load_key() -> Optional[bytes]:
    """
    Load the encryption key.
    Search order:
    1. VOICESUM_MODEL_KEY environment variable (base64)
    2. model.key file in MODELS_DIR
    3. model.key file adjacent to executable
    """
    from config import settings

    # From environment (for CI/automated builds)
    env_key = os.environ.get("VOICESUM_MODEL_KEY")
    if env_key:
        return env_key.encode()

    # From MODELS_DIR
    models_dir = Path(settings.MODELS_DIR)
    candidates = [
        models_dir / "model.key",
        models_dir.parent / "model.key",
        Path(sys.executable).parent / "model.key" if getattr(sys, "frozen", False) else None,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            logger.debug(f"[ModelLoader] Using key from: {candidate}")
            return candidate.read_bytes()

    return None


def _get_temp_base() -> Path:
    """Return the base temp directory for decrypted models."""
    base = Path(tempfile.gettempdir()) / "voicesum_runtime"
    base.mkdir(parents=True, exist_ok=True)
    return base


class ModelLoader:
    """
    Transparently loads encrypted model .dat files.
    Falls back to HuggingFace cache in development mode.
    """

    _cache: Dict[str, Path] = {}  # generic_name → decrypted path

    @classmethod
    def get_model_path(cls, generic_name: str) -> Optional[Path]:
        """
        Get the path to a decrypted model directory.
        Returns None if model is not found.
        """
        if generic_name in cls._cache:
            return cls._cache[generic_name]

        # Try encrypted .dat first
        path = cls._try_load_encrypted(generic_name)
        if path:
            cls._cache[generic_name] = path
            return path

        # Fallback: look in MODELS_DIR as plain directory (dev/test)
        path = cls._try_load_plain(generic_name)
        if path:
            cls._cache[generic_name] = path
            return path

        logger.warning(f"[ModelLoader] Model not found: {generic_name}")
        return None

    @classmethod
    def _try_load_encrypted(cls, generic_name: str) -> Optional[Path]:
        """Attempt to decrypt a .dat file and return extracted directory path."""
        from config import settings

        dat_file = Path(settings.MODELS_DIR) / f"{generic_name}.dat"
        if not dat_file.exists():
            return None

        fernet = _get_fernet()
        if fernet is None:
            logger.warning(
                f"[ModelLoader] No encryption key found; cannot decrypt {generic_name}.dat. "
                "Set VOICESUM_MODEL_KEY or place model.key in MODELS_DIR."
            )
            return None

        try:
            logger.info(f"[ModelLoader] Decrypting {generic_name}.dat ...")
            with open(dat_file, "rb") as f:
                header = f.read(6)
                if header != _MAGIC:
                    logger.error(f"[ModelLoader] Invalid magic header in {dat_file}")
                    return None
                encrypted_data = f.read()

            decrypted = fernet.decrypt(encrypted_data)

            # Extract tar.gz to temp directory
            temp_dir = _get_temp_base() / generic_name
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True)

            import io
            with tarfile.open(fileobj=io.BytesIO(decrypted), mode="r:gz") as tar:
                tar.extractall(temp_dir)

            _temp_dirs.append(temp_dir)

            # The tar contains a single root directory
            extracted = list(temp_dir.iterdir())
            if len(extracted) == 1 and extracted[0].is_dir():
                result = extracted[0]
            else:
                result = temp_dir

            logger.info(f"[ModelLoader] {generic_name} ready at {result}")
            return result

        except Exception as e:
            logger.error(f"[ModelLoader] Failed to decrypt {generic_name}: {e}")
            return None

    @classmethod
    def _try_load_plain(cls, generic_name: str) -> Optional[Path]:
        """
        Look for plain (unencrypted) model directory.

        Search order:
        1. MODELS_DIR/<generic_name>           (e.g. runtime/models/nlp_engine)
        2. MODELS_DIR/<generic_name> variants  (underscore, _model, _dir suffixes)
        3. MODELS_DIR/../<generic_name>        (e.g. runtime/nlp_engine)
        4. MODELS_DIR/../<hyphen-name>         (e.g. runtime/nlp-engine)  ← Qwen production path

        The hyphen variant handles models shipped as plain directories whose
        folder name uses a hyphen (e.g. nlp-engine) rather than underscore.
        """
        from config import settings

        models_dir = Path(settings.MODELS_DIR)
        runtime_dir = models_dir.parent  # one level up: runtime/

        # Name variants to try (underscore→hyphen and common suffixes)
        hyphen_name = generic_name.replace("_", "-")  # nlp_engine → nlp-engine
        candidates = [
            # In models/ subdirectory (standard location)
            models_dir / generic_name,
            models_dir / f"{generic_name}_model",
            models_dir / f"{generic_name}_dir",
            models_dir / hyphen_name,
            # One level up in runtime/ (for large models shipped separately)
            runtime_dir / generic_name,
            runtime_dir / hyphen_name,
        ]

        for candidate in candidates:
            if candidate.is_dir():
                logger.info(f"[ModelLoader] Using plain model at {candidate}")
                return candidate

        return None

    @classmethod
    def verify_manifest(cls) -> Dict[str, bool]:
        """
        Verify all models listed in the encrypted manifest are present.
        Returns {generic_name: is_available} for each registered model.
        """
        from config import settings
        from cryptography.fernet import Fernet
        import json

        manifest_path = Path(settings.MODELS_DIR) / "model_manifest.dat"
        if not manifest_path.exists():
            logger.warning("[ModelLoader] No model manifest found.")
            return {}

        fernet = _get_fernet()
        if fernet is None:
            return {}

        try:
            with open(manifest_path, "rb") as f:
                header = f.read(6)
                if header != _MAGIC:
                    return {}
                encrypted = f.read()
            manifest = json.loads(fernet.decrypt(encrypted))
            return {name: (Path(settings.MODELS_DIR) / info["file"]).exists()
                    for name, info in manifest.items()}
        except Exception as e:
            logger.error(f"[ModelLoader] Manifest verification failed: {e}")
            return {}


# ── Install hook: redirect HuggingFace downloads to local cache ──
def setup_offline_hf_environment():
    """
    Configure HuggingFace transformers/datasets to:
    1. Never download files from the internet
    2. Use MODELS_DIR as the cache root (after decryption)

    Call this before importing any HuggingFace library.
    """
    from config import settings

    if not settings.OFFLINE_MODE:
        return  # Only enforce in offline mode

    # Set HF offline flags
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path(settings.MODELS_DIR) / "_hf_home"))

    logger.info("[ModelLoader] Offline mode: HuggingFace downloads disabled.")
