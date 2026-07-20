"""
quantize_embedding_model.py
===========================
Quantizes a local text embedding model to INT8 using optimum-quanto.
This script reduces model weight sizes on disk and in memory by ~50%.

Prerequisites:
    pip install optimum-quanto transformers torch accelerate

Usage:
    # Quantize in-place:
    python tools/quantize_embedding_model.py --src-dir Application/runtime/embeddings/Qwen3-Embedding-4B-Instruct-INT8

    # Quantize to a separate directory:
    python tools/quantize_embedding_model.py --src-dir Application/runtime/embeddings/Qwen3-Embedding-4B-Instruct-INT8 --dest-dir Application/runtime/embeddings/Qwen3-Embedding-4B-Instruct-INT8-quantized
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a local text embedding model to INT8 using optimum-quanto."
    )
    parser.add_argument(
        "--src-dir",
        required=True,
        help="Path to the source model directory containing the downloaded unquantized files.",
    )
    parser.add_argument(
        "--dest-dir",
        default=None,
        help="Path to save the quantized model. If omitted, performs in-place quantization.",
    )
    args = parser.parse_args()

    src_path = Path(args.src_dir).resolve()
    if not src_path.exists() or not src_path.is_dir():
        logger.error(f"Source directory does not exist: {src_path}")
        sys.exit(1)

    # Determine destination directory
    in_place = args.dest_dir is None
    if in_place:
        logger.info(f"Performing in-place quantization on: {src_path}")
        # Use a temporary directory for the quantization output
        temp_dir = Path(tempfile.mkdtemp(prefix="quantize_temp_"))
        dest_path = temp_dir
    else:
        dest_path = Path(args.dest_dir).resolve()
        logger.info(f"Quantizing {src_path} -> {dest_path}")
        dest_path.mkdir(parents=True, exist_ok=True)

    # Check dependencies
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.error(
            "Missing dependencies. Please run: pip install torch transformers accelerate"
        )
        sys.exit(1)

    try:
        from optimum.quanto import freeze, qint8, quantize
    except ImportError:
        logger.error(
            "optimum-quanto is required for quantization. Please run: pip install optimum-quanto"
        )
        sys.exit(1)

    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(src_path),
            local_files_only=True,
            trust_remote_code=True,
        )

        logger.info("Loading model weights into CPU RAM...")
        # Load weights on CPU using float32 for clean quantization math
        model = AutoModel.from_pretrained(
            str(src_path),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        logger.info("Quantizing linear layers to INT8...")
        quantize(model, weights=qint8)

        logger.info("Freezing model weights...")
        freeze(model)

        logger.info(f"Saving quantized model to: {dest_path}")
        model.save_pretrained(str(dest_path))
        tokenizer.save_pretrained(str(dest_path))

        # Copy non-weight config and metadata files that transformers doesn't serialize
        logger.info("Copying sidecar configuration and tokenizer configuration files...")
        for item in src_path.iterdir():
            if item.is_file() and item.suffix not in [".bin", ".safetensors", ".pt", ".pth"]:
                dest_file = dest_path / item.name
                if not dest_file.exists():
                    shutil.copy2(item, dest_file)

        # If performing in-place, move temp files back to source directory
        if in_place:
            logger.info("Replacing unquantized weights with INT8 quantized weights in-place...")
            # Remove original float16 weight files
            for item in src_path.iterdir():
                if item.is_file() and item.suffix in [".bin", ".safetensors", ".pt", ".pth"]:
                    logger.debug(f"Removing original weight file: {item.name}")
                    item.unlink()

            # Copy all files from temp directory back to src_path
            for item in temp_dir.iterdir():
                if item.is_file():
                    shutil.move(str(item), str(src_path / item.name))

            # Remove temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info("[SUCCESS] Model quantized to INT8 successfully! ✓")

    except Exception as e:
        logger.error(f"[ERROR] Quantization failed: {e}", exc_info=True)
        if in_place and 'temp_dir' in locals() and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
