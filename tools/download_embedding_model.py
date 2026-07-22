"""
download_embedding_model.py
===========================
Downloads and caches the Qwen3-Embedding models into the application's runtime/embeddings/ directory.
If the 4B model is requested, it automatically quantizes it to INT8 in-place using optimum-quanto.
Supports downloading from ModelScope (default) and HuggingFace Hub.

Run once (with internet access) before deploying offline:

    python tools/download_embedding_model.py --provider modelscope

All downloads are idempotent: existing files and quantized status are verified and skipped.
"""

from __future__ import annotations

import argparse
import logging
import os  
import sys
import shutil
import tempfile
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

def _quantize_model_inplace(dest_path: Path) -> None:
    """Quantize the downloaded unquantized float16 weights of Qwen3-Embedding-4B into INT8 in-place."""
    logger.info("Starting INT8 quantization using optimum-quanto...")
    
    # Check/install dependencies
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.error("Missing standard dependencies. Installing torch, transformers, and accelerate via pip...")
        import subprocess
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "transformers", "accelerate"], check=True)
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as ex:
            logger.error(f"Failed to install dependencies: {ex}. Please install manually.")
            sys.exit(1)

    try:
        from optimum.quanto import freeze, qint8, quantize
    except ImportError:
        logger.info("optimum-quanto is not installed. Installing via pip...")
        import subprocess
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "optimum-quanto"], check=True)
            from optimum.quanto import freeze, qint8, quantize
        except Exception as ex:
            logger.error(f"Failed to install optimum-quanto: {ex}. Please install manually with: pip install optimum-quanto")
            sys.exit(1)

    temp_dir = Path(tempfile.mkdtemp(prefix="quantize_temp_"))
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(dest_path),
            local_files_only=True,
            trust_remote_code=True,
        )

        logger.info("Loading model weights into CPU RAM (using float32 for quantization math)...")
        model = AutoModel.from_pretrained(
            str(dest_path),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        logger.info("Quantizing linear layers to INT8...")
        quantize(model, weights=qint8)

        logger.info("Freezing model weights...")
        freeze(model)

        logger.info(f"Saving quantized model temporarily to: {temp_dir}")
        model.save_pretrained(str(temp_dir))
        tokenizer.save_pretrained(str(temp_dir))

        # Copy non-weight config and metadata files that transformers doesn't serialize
        logger.info("Copying sidecar config and tokenizer config files...")
        for item in dest_path.iterdir():
            if item.is_file() and item.suffix not in [".bin", ".safetensors", ".pt", ".pth"]:
                dest_file = temp_dir / item.name
                if not dest_file.exists():
                    shutil.copy2(item, dest_file)

        logger.info("Replacing unquantized weights with INT8 quantized weights in-place...")
        # Remove original float16 weight files
        for item in dest_path.iterdir():
            if item.is_file() and item.suffix in [".bin", ".safetensors", ".pt", ".pth"]:
                item.unlink()

        # Copy all files from temp directory back to dest_path
        for item in temp_dir.iterdir():
            if item.is_file():
                shutil.move(str(item), str(dest_path / item.name))

        # Remove temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Write marker file
        (dest_path / ".quantized_int8").write_text("success")
        logger.info("[SUCCESS] Model quantized to INT8 successfully! ✓")

    except Exception as e:
        logger.error(f"[ERROR] Quantization failed: {e}", exc_info=True)
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Qwen embedding model for offline RAG use."
    )
    parser.add_argument(
        "--model",
        default="Qwen3-Embedding-0.6B",
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
    
    has_weights = any(
        f.suffix in [".safetensors", ".bin", ".pt"]
        for f in dest.iterdir()
    ) if dest.is_dir() else False
    needs_quantization = model_name.endswith("-INT8")
    is_already_quantized = (dest / ".quantized_int8").exists()

    if all(p.exists() for p in required_files) and has_weights and (not needs_quantization or is_already_quantized):
        logger.info(f"[SKIP] {model_name} is already fully downloaded, quantized, and present at {dest} ✓")
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

    # Perform quantization if requested
    if needs_quantization:
        _quantize_model_inplace(dest)

if __name__ == "__main__":
    main()
