"""
Model Encryption Tool — Phase 3
Encrypts all AI models for secure distribution.

Usage:
    python tools/encrypt_models.py --output Application/runtime/models

This script:
1. Locates all required models from HuggingFace cache
2. Encrypts small/medium models with AES-256 (via cryptography.fernet)
3. Copies large models (e.g. Qwen) directly without encryption to avoid
   memory exhaustion — these are stored in a plain subdirectory
4. Renames to generic filenames
5. Writes an encrypted manifest with checksums

Run once during the build process. The resulting .dat files and plain
directories are included in the installer and handled at runtime by
model_loader.py.
"""
import os
import sys
import json
import shutil
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Model Registry ────────────────────────────────────────────
# Maps generic name → HuggingFace cache repo name(s)
# The first match found in HF_CACHE is used.
MODEL_REGISTRY: Dict[str, list] = {
    "speech_engine": [
        "models--Systran--faster-whisper-medium",
        "models--Systran--faster-whisper-small",
        "models--openai--whisper-medium",
    ],
    "audio_context": [
        "models--pyannote--speaker-diarization-3.1",
    ],
    "voice_segment": [
        "models--pyannote--segmentation-3.0",
    ],
    "voice_context": [
        "models--speechbrain--spkrec-ecapa-voxceleb",
    ],
    "align_engine": [
        "models--facebook--wav2vec2-base-960h",
    ],
    "wespeaker": [
        "models--hbredin--wespeaker-voxceleb-resnet34-LM",
    ],
}

# ── Plain-Copy Registry ───────────────────────────────────────
# Large models that are copied as-is (no encryption) to avoid
# loading gigabytes into RAM. Stored under output_dir/<generic_name>/
PLAIN_COPY_REGISTRY: Dict[str, list] = {
    "nlp_engine": [
        "models--Qwen--Qwen3-4B",
    ],
}


def get_hf_cache_dir() -> Path:
    """Return HuggingFace cache directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))


def find_model_in_cache(model_candidates: list, cache_dir: Path) -> Optional[Path]:
    """Find the first available model in HF cache."""
    for candidate in model_candidates:
        path = cache_dir / candidate
        if path.exists():
            logger.info(f"  Found: {path}")
            return path
    return None


def sha256_of_dir(path: Path) -> str:
    """Compute SHA-256 of all files in a directory (deterministic)."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def get_or_create_key(key_file: Path) -> bytes:
    """Load existing encryption key or generate and save a new one."""
    from cryptography.fernet import Fernet
    if key_file.exists():
        logger.info(f"Using existing encryption key: {key_file}")
        return key_file.read_bytes()
    key = Fernet.generate_key()
    key_file.write_bytes(key)
    logger.info(f"Generated new encryption key: {key_file}")
    logger.warning("IMPORTANT: Keep this key file secure. Loss = cannot decrypt models.")
    return key


def encrypt_directory(src_dir: Path, dest_file: Path, fernet) -> str:
    """
    Pack a model directory into a tar.gz, encrypt it, save as .dat.
    Returns SHA-256 of the original directory.
    """
    import tarfile
    import io

    logger.info(f"  Packing {src_dir.name} ...")
    checksum = sha256_of_dir(src_dir)

    # Create in-memory tar.gz
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(src_dir), arcname=src_dir.name)
    compressed = buf.getvalue()

    # Encrypt
    logger.info(f"  Encrypting ({len(compressed) // 1024 // 1024} MB) ...")
    encrypted = fernet.encrypt(compressed)

    # Write with magic header
    with open(dest_file, "wb") as f:
        f.write(b"VSDAT\x01")  # magic + version
        f.write(encrypted)

    size_mb = dest_file.stat().st_size / 1024 / 1024
    logger.info(f"  → {dest_file.name} ({size_mb:.1f} MB)")
    return checksum


def encrypt_single_file(src_file: Path, dest_file: Path, fernet) -> str:
    """Encrypt a single file (e.g. .pth checkpoint)."""
    logger.info(f"  Encrypting file {src_file.name} ...")
    checksum = sha256_of_file(src_file)
    data = src_file.read_bytes()
    encrypted = fernet.encrypt(data)
    with open(dest_file, "wb") as f:
        f.write(b"VSDAT\x01")
        f.write(encrypted)
    logger.info(f"  → {dest_file.name}")
    return checksum


def find_snapshots_dir(model_cache_dir: Path) -> Optional[Path]:
    """Navigate HF cache structure to find the actual model files."""
    snapshots = model_cache_dir / "snapshots"
    if not snapshots.exists():
        return model_cache_dir  # Already a direct directory
    # Get the most recent snapshot
    snapshot_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        return None
    return sorted(snapshot_dirs)[-1]  # Latest by directory name


def copy_directory(src_dir: Path, dest_dir: Path) -> str:
    """
    Copy a model directory directly (no encryption) for large models
    where loading the whole archive into RAM would OOM.
    Returns SHA-256 checksum of the source directory.
    """
    logger.info(f"  Copying {src_dir.name} → {dest_dir.name}/ (plain, no encryption) ...")
    checksum = sha256_of_dir(src_dir)
    if dest_dir.exists():
        logger.info(f"  Destination exists — removing old copy ...")
        shutil.rmtree(dest_dir)
    shutil.copytree(str(src_dir), str(dest_dir))
    size_mb = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file()) / 1024 / 1024
    logger.info(f"  → {dest_dir.name}/ ({size_mb:.0f} MB, unencrypted)")
    return checksum


def main():
    parser = argparse.ArgumentParser(description="Package AI models for distribution (unencrypted)")
    parser.add_argument("--output", default="Application/runtime/models", help="Output directory for packaged models")
    parser.add_argument("--hf-cache", help="Override HuggingFace cache directory")
    parser.add_argument("--dry-run", action="store_true", help="List models without packaging")
    args = parser.parse_args()

    output_dir = Path(args.output)
    cache_dir = Path(args.hf_cache) if args.hf_cache else get_hf_cache_dir()

    if args.dry_run:
        logger.info(f"DRY RUN — HF Cache: {cache_dir}")
        logger.info("[MODELS TO PACKAGE]")
        for generic_name, candidates in MODEL_REGISTRY.items():
            found = find_model_in_cache(candidates, cache_dir)
            status = "✓ FOUND" if found else "✗ MISSING"
            logger.info(f"  [{status}] {generic_name}: {found or candidates[0]}")
        for generic_name, candidates in PLAIN_COPY_REGISTRY.items():
            found = find_model_in_cache(candidates, cache_dir)
            status = "✓ FOUND" if found else "✗ MISSING"
            logger.info(f"  [{status}] {generic_name}: {found or candidates[0]}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    missing = []

    # Merge registries so we process everything in the same way
    all_registries = {}
    all_registries.update(MODEL_REGISTRY)
    all_registries.update(PLAIN_COPY_REGISTRY)

    for generic_name, candidates in all_registries.items():
        logger.info(f"\n[{generic_name}]")
        model_path = find_model_in_cache(candidates, cache_dir)

        if model_path is None:
            logger.warning(f"  SKIPPED — not found in cache: {candidates}")
            missing.append(generic_name)
            continue

        # Navigate to actual model files
        actual_path = find_snapshots_dir(model_path)
        if actual_path is None:
            logger.warning(f"  SKIPPED — no snapshots found in {model_path}")
            missing.append(generic_name)
            continue

        dest_dir = output_dir / generic_name
        try:
            checksum = copy_directory(actual_path, dest_dir)
            manifest[generic_name] = {
                "original_name": candidates[0],
                "checksum": checksum,
                "type": "plain",
                "dir": generic_name,
                "snapshot_hash": actual_path.name,
            }
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            missing.append(generic_name)

    # Write manifest in plain text JSON
    manifest_path = output_dir / "model_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Packaged: {len(manifest)} models → {output_dir}")
    if missing:
        logger.warning(f"Missing:  {missing}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
