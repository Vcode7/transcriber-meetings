"""
Download Whisper large-v3 model directly to Application/runtime/models/speech_engine
and update the model manifest.
"""
import os
import sys
import json
import shutil
import hashlib
from pathlib import Path

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEECH_ENGINE_DIR = PROJECT_ROOT / "Application" / "runtime" / "models" / "speech_engine"
MANIFEST_PATH = PROJECT_ROOT / "Application" / "runtime" / "models" / "model_manifest" / "model_manifest.json" # wait, manifest is under runtime/models/model_manifest.json ?
# Let's check: manifest_path in get_model_path search is Path(settings.MODELS_DIR) / "model_manifest.json"
# settings.MODELS_DIR points to runtime/models
# So manifest is at PROJECT_ROOT / "Application" / "runtime" / "models" / "model_manifest.json"
MANIFEST_PATH = PROJECT_ROOT / "Application" / "runtime" / "models" / "model_manifest.json"

def sha256_of_dir(path: Path) -> str:
    """Compute SHA-256 of all files in a directory (deterministic)."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()

def main():
    print(f"Target directory: {SPEECH_ENGINE_DIR}")
    print(f"Manifest path: {MANIFEST_PATH}")
    
    # 1. Clear speech_engine folder
    if SPEECH_ENGINE_DIR.exists():
        print("Clearing existing speech_engine files...")
        shutil.rmtree(SPEECH_ENGINE_DIR)
    SPEECH_ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Download Systran/faster-whisper-large-v3 via huggingface_hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
        from huggingface_hub import snapshot_download

    print("Downloading Systran/faster-whisper-large-v3 from HF...")
    # download to local_dir
    local_dir_path = snapshot_download(
        repo_id="Systran/faster-whisper-large-v3",
        local_dir=str(SPEECH_ENGINE_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {local_dir_path}")
    
    # 3. Compute checksum of the directory
    print("Computing directory checksum...")
    checksum = sha256_of_dir(SPEECH_ENGINE_DIR)
    print(f"Checksum: {checksum}")
    
    # 4. Resolve the commit hash/snapshot hash
    # huggingface_hub uses a hash folder name inside HF cache, but here we downloaded directly.
    # We can retrieve the commit hash from the .git folder or just use the repo's latest commit hash.
    snapshot_hash = "default_large_v3_hash"
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.repo_info(repo_id="Systran/faster-whisper-large-v3")
        snapshot_hash = info.sha
        print(f"Latest commit hash (sha): {snapshot_hash}")
    except Exception as e:
        print(f"Could not get repo info: {e}")

    # 5. Update the manifest
    if MANIFEST_PATH.exists():
        print("Updating model_manifest.json...")
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        manifest["speech_engine"] = {
            "original_name": "models--Systran--faster-whisper-large-v3",
            "checksum": checksum,
            "type": "plain",
            "dir": "speech_engine",
            "snapshot_hash": snapshot_hash
        }
        
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Manifest updated successfully.")
    else:
        print(f"Warning: manifest not found at {MANIFEST_PATH}")

if __name__ == "__main__":
    main()
