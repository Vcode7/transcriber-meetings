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


def _ensure_wav2vec2_tokenizer_files(target_dir: Path):
    """Write standard wav2vec2-base tokenizer config files to the directory if missing."""
    import json
    files = {
        "vocab.json": {
            "|": 0, "E": 1, "T": 2, "A": 3, "O": 4, "N": 5, "I": 6, "H": 7,
            "S": 8, "R": 9, "D": 10, "L": 11, "U": 12, "M": 13, "W": 14, "C": 15,
            "F": 16, "G": 17, "Y": 18, "P": 19, "B": 20, "V": 21, "K": 22, "'": 23,
            "X": 24, "J": 25, "Q": 26, "Z": 27, "[UNK]": 28, "[PAD]": 29,
        },
        "tokenizer_config.json": {
            "bos_token": "<s>",
            "cls_token": "<s>",
            "eos_token": "</s>",
            "mask_token": "<mask>",
            "model_max_length": 1000000000000000019884624838656,
            "pad_token": "[PAD]",
            "sep_token": "</s>",
            "tokenizer_class": "Wav2Vec2CTCTokenizer",
            "unk_token": "[UNK]",
            "word_delimiter_token": "|"
        },
        "special_tokens_map.json": {
            "bos_token": "<s>",
            "cls_token": "<s>",
            "eos_token": "</s>",
            "mask_token": "<mask>",
            "pad_token": "[PAD]",
            "sep_token": "</s>",
            "unk_token": "[UNK]"
        },
        "preprocessor_config.json": {
            "do_normalize": True,
            "feature_extractor_type": "Wav2Vec2FeatureExtractor",
            "feature_size": 1,
            "padding_side": "right",
            "padding_value": 0.0,
            "processor_class": "Wav2Vec2Processor",
            "return_attention_mask": False,
            "sampling_rate": 16000
        }
    }
    for filename, content in files.items():
        dest = target_dir / filename
        if not dest.exists():
            try:
                dest.write_text(json.dumps(content, indent=2), encoding="utf-8")
                logger.info(f"[ModelLoader] Wrote missing tokenizer file: {filename}")
            except Exception as e:
                logger.error(f"[ModelLoader] Failed to write tokenizer file {filename}: {e}")


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
        print(f"[ModelLoader] Looking for {generic_name}.dat at: {dat_file.resolve()}")
        if not dat_file.exists():
            print(f"[ModelLoader] NOT FOUND: {dat_file.resolve()}")
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

            if generic_name == "align_engine":
                _ensure_wav2vec2_tokenizer_files(result)

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
            print(f"[ModelLoader] Checking candidate plain path: {candidate.resolve()}")
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
            return {name: (Path(settings.MODELS_DIR) / info.get("file", info.get("dir", ""))).exists()
                    for name, info in manifest.items()}
        except Exception as e:
            logger.error(f"[ModelLoader] Manifest verification failed: {e}")
            return {}


# ── Install hook: redirect HuggingFace downloads to local cache ──
def restore_hf_cache():
    """
    Decrypt all models in the manifest and copy/link them to the standard HF cache.
    Allows all third-party libraries (transformers, speechbrain, pyannote, whisperx)
    to load models fully offline without modifying their from_pretrained calls.
    """
    from config import settings
    import json
    from pathlib import Path
    import shutil

    manifest_path = Path(settings.MODELS_DIR) / "model_manifest.dat"
    if not manifest_path.exists():
        logger.info("[ModelLoader] No model manifest found; skipping cache restoration.")
        return

    fernet = _get_fernet()
    if fernet is None:
        logger.warning("[ModelLoader] No encryption key found; skipping cache restoration.")
        return

    try:
        # Resolve standard HF home cache directory
        hf_home = Path(os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface" / "hub")
        )).resolve()

        with open(manifest_path, "rb") as f:
            header = f.read(6)
            if header != _MAGIC:
                logger.error("[ModelLoader] Invalid magic header in manifest")
                return
            encrypted = f.read()

        manifest = json.loads(fernet.decrypt(encrypted))
        logger.info(f"[ModelLoader] Restoring HF cache for {len(manifest)} models...")

        for generic_name, info in manifest.items():
            # Skip nlp_engine if handled separately
            if generic_name == "nlp_engine":
                continue

            dat_file = Path(settings.MODELS_DIR) / info["file"]
            if not dat_file.exists():
                logger.warning(f"[ModelLoader] Model file {info['file']} not found.")
                continue

            # Decrypt the model using existing try_load_encrypted
            decrypted_path = ModelLoader.get_model_path(generic_name)
            if not decrypted_path or not decrypted_path.exists():
                continue

            # The decrypted path points to the snapshots/<hash>/ directory
            snapshot_hash = decrypted_path.name
            original_name = info["original_name"] # e.g. "models--pyannote--speaker-diarization-3.1"

            # Reconstruct the expected HF hub cache directory path
            target_snapshots_dir = hf_home / original_name / "snapshots"
            target_hash_dir = target_snapshots_dir / snapshot_hash

            # If it already exists, verify it contains files (no need to copy again)
            if target_hash_dir.exists() and any(target_hash_dir.iterdir()):
                logger.debug(f"[ModelLoader] Cache already exists for {original_name} at {target_hash_dir}")
                continue

            # Copy or link files from decrypted_path to target_hash_dir
            logger.info(f"[ModelLoader] Syncing {generic_name} to HF cache: {target_hash_dir}")
            target_hash_dir.mkdir(parents=True, exist_ok=True)
            for item in decrypted_path.iterdir():
                dest_item = target_hash_dir / item.name
                if item.is_dir():
                    if dest_item.exists():
                        shutil.rmtree(dest_item)
                    shutil.copytree(str(item), str(dest_item))
                else:
                    shutil.copy2(str(item), str(dest_item))

            # Also create standard ref file if missing
            ref_dir = hf_home / original_name / "refs"
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_file = ref_dir / "main"
            if not ref_file.exists():
                ref_file.write_text(snapshot_hash, encoding="utf-8")

    except Exception as e:
        logger.error(f"[ModelLoader] HF cache restoration failed: {e}", exc_info=True)


# ── Install hook: redirect HuggingFace downloads to local cache ──
def setup_offline_hf_environment():
    """
    Configure HuggingFace transformers/datasets to:
    1. Never download files from the internet
    2. Use the standard HF hub cache (models are already cached there)

    Always enforced — this application is deployed offline and never requires
    internet access. Using setdefault so a developer can override by pre-setting
    the env vars before launching.
    """
    # Enforce offline mode unconditionally
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    logger.info("[ModelLoader] HuggingFace offline mode enforced (no internet calls).")

    # Restore/verify cache directories from local decrypted models
    restore_hf_cache()
