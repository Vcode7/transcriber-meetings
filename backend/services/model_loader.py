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

# Tracks temp dirs created this session (retained for backward compatibility, not used)
_temp_dirs: list[Path] = []


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
    Transparently loads plain model directories.
    """

    _cache: Dict[str, Path] = {}  # generic_name → model path

    @classmethod
    def get_model_path(cls, generic_name: str) -> Optional[Path]:
        """
        Get the path to a model directory.
        Returns None if model is not found.
        """
        if generic_name in cls._cache:
            return cls._cache[generic_name]

        # Look in MODELS_DIR as plain directory (production and dev fallback)
        path = cls._try_load_plain(generic_name)
        if path:
            if generic_name == "align_engine":
                _ensure_wav2vec2_tokenizer_files(path)
            cls._cache[generic_name] = path
            return path

        logger.warning(f"[ModelLoader] Model not found: {generic_name}")
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
        Verify all models listed in the manifest are present.
        Returns {generic_name: is_available} for each registered model.
        """
        from config import settings
        import json

        manifest_path = Path(settings.MODELS_DIR) / "model_manifest.json"
        if not manifest_path.exists():
            logger.warning("[ModelLoader] No model manifest found.")
            return {}

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return {name: (Path(settings.MODELS_DIR) / info.get("dir", info.get("file", ""))).exists()
                    for name, info in manifest.items()}
        except Exception as e:
            logger.error(f"[ModelLoader] Manifest verification failed: {e}")
            return {}


# ── Install hook: redirect HuggingFace downloads to local cache ──
def restore_hf_cache():
    """
    Copy/link plain models in the manifest to the standard HF cache.
    Allows all third-party libraries (transformers, speechbrain, pyannote, whisperx)
    to load models fully offline without modifying their from_pretrained calls.
    """
    from config import settings
    import json
    from pathlib import Path
    import shutil

    manifest_path = Path(settings.MODELS_DIR) / "model_manifest.json"
    if not manifest_path.exists():
        logger.info("[ModelLoader] No model manifest found; skipping cache restoration.")
        return

    try:
        # Resolve standard HF home cache directory
        hf_home = Path(os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface" / "hub")
        )).resolve()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        logger.info(f"[ModelLoader] Restoring HF cache for {len(manifest)} models...")

        for generic_name, info in manifest.items():
            # Skip nlp_engine if handled separately
            if generic_name == "nlp_engine":
                continue

            model_path = ModelLoader.get_model_path(generic_name)
            if not model_path or not model_path.exists():
                logger.warning(f"[ModelLoader] Model directory for {generic_name} not found.")
                continue

            snapshot_hash = info.get("snapshot_hash", "default")
            original_name = info["original_name"] # e.g. "models--pyannote--speaker-diarization-community-1"

            # Reconstruct the expected HF hub cache directory path
            target_snapshots_dir = hf_home / original_name / "snapshots"
            target_hash_dir = target_snapshots_dir / snapshot_hash

            # If it already exists, verify it contains files (no need to copy again)
            if target_hash_dir.exists() and any(target_hash_dir.iterdir()):
                logger.debug(f"[ModelLoader] Cache already exists for {original_name} at {target_hash_dir}")
                continue

            # Copy or link files from model_path to target_hash_dir
            logger.info(f"[ModelLoader] Syncing {generic_name} to HF cache: {target_hash_dir}")
            target_hash_dir.mkdir(parents=True, exist_ok=True)
            for item in model_path.iterdir():
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


def verify_all_models() -> Dict[str, bool]:
    """
    Verify all expected model directories are present in MODELS_DIR.
    Logs a clear summary. Called once at startup.
    Returns {generic_name: is_present} for all known models.
    """
    expected = [
        "speech_engine",   # faster-whisper-large-v3
        "audio_context",   # pyannote/speaker-diarization-community-1 (complete snapshot)
        "ecapa_tdnn",      # SpeechBrain ECAPA-TDNN (speaker identification)
        "align_engine",    # facebook/wav2vec2-base-960h
        "nlp_engine",      # Qwen3-4B
    ]
    results: Dict[str, bool] = {}
    for name in expected:
        path = ModelLoader.get_model_path(name)
        present = path is not None and path.exists()
        results[name] = present

    present_list = [n for n, ok in results.items() if ok]
    missing_list = [n for n, ok in results.items() if not ok]

    if missing_list:
        logger.error(
            f"[ModelLoader] ✕ {len(missing_list)} model(s) MISSING from MODELS_DIR:\n"
            + "\n".join(f"  • {n}" for n in missing_list)
            + "\n  The application may fail or degrade for features requiring these models."
        )
    else:
        logger.info(f"[ModelLoader] ✓ All {len(present_list)} models present in MODELS_DIR.")

    logger.info(
        "[ModelLoader] Model presence summary: "
        + ", ".join(f"{n}={'OK' if ok else 'MISSING'}" for n, ok in results.items())
    )
    return results


# ── Install hook: configure HuggingFace offline mode ──
def setup_offline_hf_environment():
    """
    Configure HuggingFace transformers/datasets to:
    1. Never download files from the internet
    2. Use local model directories for all loads

    Always enforced — this application is deployed offline and never requires
    internet access.
    """
    # Enforce offline mode unconditionally
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    logger.info("[ModelLoader] HuggingFace offline mode enforced (no internet calls).")

    # Restore/verify cache directories from local model folders (for libraries that
    # require HF hub cache layout, such as older whisperx align model loading)
    restore_hf_cache()

    # Verify all models are present and log a startup summary
    verify_all_models()
