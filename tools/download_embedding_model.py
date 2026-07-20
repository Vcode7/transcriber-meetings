"""
download_embedding_model.py
===========================
Downloads and caches the Qwen3-Embedding-0.6B model into the application's runtime/embeddings/ directory.
Supports downloading from ModelScope (default, recommended for users in China or when HF is unresponsive)
and HuggingFace Hub.

Run once (with internet access) before deploying offline:

    python tools/download_embedding_model.py --provider modelscope

All downloads are idempotent: existing files are verified and skipped.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Resolve project root ───────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

def _resolve_embeddings_dir() -> Path:
    """Return the embeddings directory the backend will read from."""
    # Priority 1: Application/runtime/embeddings
    app_embeds = _PROJECT_ROOT / "Application" / "runtime" / "embeddings"
    if app_embeds.is_dir():
        return app_embeds
    # Priority 2: backend/runtime/embeddings (development)
    return _PROJECT_ROOT / "backend" / "runtime" / "embeddings"

_EMBEDDINGS_DIR = _resolve_embeddings_dir()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Qwen embedding model for offline RAG use."
    )
    parser.add_argument(
        "--model",
        default="Qwen3-Embedding-4B-Instruct-INT8",
        choices=["Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B-Instruct-INT8"],
        help="Embedding model to download (default: Qwen3-Embedding-0.6B)",
    )
    parser.add_argument(
        "--provider",
        default="modelscope",
        choices=["modelscope", "huggingface"],
        help="Model provider/hub to download from (default: modelscope)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Optional HuggingFace access token (only used for huggingface provider)",
    )
    parser.add_argument(
        "--dest-dir",
        default=None,
        help="Override destination directory (default: backend/runtime/embeddings)",
    )
    args = parser.parse_args()

    # Friendly model to repo mapping
    MODEL_MAP = {
        "Qwen3-Embedding-0.6B": {
            "huggingface": "Qwen/Qwen3-Embedding-0.6B",
            "modelscope": "qwen/Qwen3-Embedding-0.6B"
        },
        "Qwen3-Embedding-4B-Instruct-INT8": {
            "huggingface": "Qwen/Qwen3-Embedding-4B",
            "modelscope": "qwen/Qwen3-Embedding-4B"
        }
    }

    # Determine destination dir
    base_dir = Path(args.dest_dir).resolve() if args.dest_dir else _EMBEDDINGS_DIR
    model_name = args.model
    dest = base_dir / model_name
    dest.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target model directory: {dest}")

    # Check if key model files already exist to skip download
    required_files = [
        dest / "config.json",
        dest / "tokenizer.json",
    ]
    missing = [p for p in required_files if not p.exists()]
    if not missing:
        logger.info(f"[SKIP] {model_name} is already present at {dest}")
        sys.exit(0)

    # Resolve repo IDs dynamically based on the model map
    repo_info = MODEL_MAP.get(model_name, {
        "huggingface": f"Qwen/{model_name}",
        "modelscope": f"qwen/{model_name}"
    })
    huggingface_repo_id = repo_info["huggingface"]
    model_scope_id = repo_info["modelscope"]

    if args.provider == "modelscope":
        logger.info(f"[DOWNLOAD] {model_scope_id} from ModelScope → {dest}")
        try:
            from modelscope import snapshot_download
        except ImportError:
            logger.error("modelscope is not installed. Installing it via pip...")
            import subprocess
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "modelscope"], check=True)
                from modelscope import snapshot_download
            except Exception as ex:
                logger.error(f"Failed to install modelscope: {ex}. Please install it manually with: pip install modelscope")
                sys.exit(1)

        try:
            snapshot_download(
                model_id=model_scope_id,
                local_dir=str(dest),
                ignore_file_pattern=["*.gif", "*.png", "*.jpg", "*.md", ".gitattributes"],
            )
            logger.info(f"[OK] {model_name} model successfully cached from ModelScope at {dest} ✓")
        except Exception as e:
            logger.error(f"[ERROR] Failed to download model from ModelScope: {e}")
            sys.exit(1)
            
    else:  # huggingface
        logger.info(f"[DOWNLOAD] {huggingface_repo_id} from HuggingFace → {dest}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
            sys.exit(1)

        try:
            snapshot_download(
                repo_id=huggingface_repo_id,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                token=args.hf_token or None,
                ignore_patterns=["*.gif", "*.png", "*.jpg", "*.md", ".gitattributes"],
            )
            logger.info(f"[OK] {model_name} model successfully cached from HuggingFace at {dest} ✓")
        except Exception as e:
            logger.error(f"[ERROR] Failed to download model from HuggingFace: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
