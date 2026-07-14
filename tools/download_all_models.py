"""
Download All Required Models -- Build Preparation Step

Downloads every model the application needs to your HuggingFace cache.
Run this ONCE on the build machine before running encrypt_models.py.

Usage:
    python tools/download_all_models.py              # Download everything
    python tools/download_all_models.py --dry-run    # Check what is/isn't cached
    python tools/download_all_models.py --skip whisper alignment  # Skip specific models

After this completes, run:
    python tools/encrypt_models.py

Model list (total ~4 GB download):
  1. faster-whisper/large-v3                     ~3,100 MB  -- speech transcription
  2. pyannote/speaker-diarization-community-1      ~100 MB  -- speaker diarization (complete snapshot)
  3. speechbrain/spkrec-ecapa-voxceleb           ~80 MB   -- speaker ID embedding (ECAPA-TDNN)
  4. facebook/wav2vec2-base-960h                   ~360 MB  -- word-level alignment
  5. Qwen/Qwen3-4B                              ~2,300 MB  -- AI summarization / MoM / insights
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))


@dataclass
class ModelSpec:
    key: str                        # Short identifier (used in --skip)
    hf_id: str                      # HuggingFace repo ID
    description: str                # Human description
    size_mb: int                    # Approximate download size
    cache_dir_name: str             # Expected folder name in HF cache
    download_fn: str                # Which loader function to use
    revision: Optional[str] = None
    extra_kwargs: dict = field(default_factory=dict)


MODELS: List[ModelSpec] = [
    ModelSpec(
        key="whisper",
        hf_id="Systran/faster-whisper-large-v3",
        description="Whisper Large V3 (speech transcription)",
        size_mb=3100,
        cache_dir_name="models--Systran--faster-whisper-large-v3",
        download_fn="snapshot",
    ),
    ModelSpec(
        key="diarization",
        hf_id="pyannote/speaker-diarization-community-1",
        description="pyannote Speaker Diarization Community-1 (complete snapshot)",
        size_mb=100,
        cache_dir_name="models--pyannote--speaker-diarization-community-1",
        download_fn="snapshot",
    ),
    ModelSpec(
        key="ecapa",
        hf_id="speechbrain/spkrec-ecapa-voxceleb",
        description="SpeechBrain ECAPA-TDNN (speaker identification)",
        size_mb=80,
        cache_dir_name="models--speechbrain--spkrec-ecapa-voxceleb",
        download_fn="snapshot",
    ),
    ModelSpec(
        key="alignment",
        hf_id="facebook/wav2vec2-base-960h",
        description="Wav2Vec2 Base 960h CTC (word-level alignment)",
        size_mb=360,
        cache_dir_name="models--facebook--wav2vec2-base-960h",
        download_fn="snapshot",
    ),
    ModelSpec(
        key="qwen3",
        hf_id="Qwen/Qwen3-4B",
        description="Qwen3 4B (AI summarization / MoM / insights)",
        size_mb=2300,
        cache_dir_name="models--Qwen--Qwen3-4B",
        download_fn="causal_lm",
    ),
]


def is_cached(spec: ModelSpec) -> bool:
    """Check if a model is already fully present in HF cache."""
    cache_path = HF_CACHE / spec.cache_dir_name
    if not cache_path.exists():
        return False
    # Check that snapshots exist and contain files
    snapshots = cache_path / "snapshots"
    if snapshots.exists():
        dirs = [d for d in snapshots.iterdir() if d.is_dir()]
        return len(dirs) > 0 and any(
            any(f.is_file() for f in d.rglob("*")) for d in dirs
        )
    # Fallback: directory exists and is non-empty
    return any(cache_path.rglob("*"))


def download_via_snapshot(spec: ModelSpec, hf_token: Optional[str]) -> bool:
    """Download a model using huggingface_hub.snapshot_download."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
        return False

    kwargs = {"repo_id": spec.hf_id, "local_dir_use_symlinks": False}
    if hf_token:
        kwargs["token"] = hf_token
    if spec.revision:
        kwargs["revision"] = spec.revision

    try:
        logger.info(f"  Downloading via snapshot_download ...")
        result = snapshot_download(**kwargs)
        logger.info(f"  Downloaded to: {result}")
        return True
    except Exception as e:
        logger.error(f"  snapshot_download failed: {e}")
        return False


def download_via_causal_lm(spec: ModelSpec, hf_token: Optional[str]) -> bool:
    """
    Download a causal language model (tokenizer + weights).
    Used for Qwen3 4B Instruct and any AutoModelForCausalLM-based model.
    Downloads in float32 (no quantization) so all hardware can load the cache.
    BitsAndBytes 4-bit quantization is applied at runtime, not at download time.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        return False

    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    try:
        logger.info(f"  Downloading tokenizer ...")
        AutoTokenizer.from_pretrained(spec.hf_id, **kwargs)
        logger.info(f"  Downloading model weights (this may take several minutes) ...")
        # low_cpu_mem_usage avoids loading the full model into RAM twice
        AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        logger.info(f"  Qwen3 weights cached ✓")
        return True
    except Exception as e:
        logger.error(f"  causal_lm download failed: {e}")
        return False


def download_via_transformers(spec: ModelSpec, hf_token: Optional[str]) -> bool:
    """Download a Seq2Seq Transformers model (tokenizer + weights). Legacy — kept for compatibility."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        return False

    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    try:
        logger.info(f"  Downloading tokenizer ...")
        AutoTokenizer.from_pretrained(spec.hf_id, **kwargs)
        logger.info(f"  Downloading model weights ...")
        AutoModelForSeq2SeqLM.from_pretrained(spec.hf_id, **kwargs)
        return True
    except Exception as e:
        logger.error(f"  transformers download failed: {e}")
        return False


def print_status_table(specs: List[ModelSpec]):
    """Print a formatted table showing cache status of all models."""
    print()
    print("=" * 78)
    print(f"{'Key':<14} {'Description':<44} {'Size':>7}  {'Status'}")
    print("-" * 78)
    total_missing_mb = 0
    for spec in specs:
        cached = is_cached(spec)
        status = "[OK] CACHED" if cached else "[--] MISSING"
        if not cached:
            total_missing_mb += spec.size_mb
        print(f"{spec.key:<14} {spec.description:<44} {spec.size_mb:>5} MB  {status}")
    print("=" * 78)
    if total_missing_mb > 0:
        print(f"  Total to download: ~{total_missing_mb:,} MB (~{total_missing_mb/1024:.1f} GB)")
    else:
        print("  All models are cached. Ready to encrypt.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download all required AI models to HuggingFace cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show download status without downloading anything",
    )
    parser.add_argument(
        "--skip", nargs="+", metavar="KEY",
        choices=[m.key for m in MODELS],
        help="Skip specific models (e.g. --skip whisper alignment)",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="KEY",
        choices=[m.key for m in MODELS],
        help="Download only specific models (e.g. --only qwen3)",
    )
    parser.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace token for gated models (pyannote requires this)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if already cached",
    )
    args = parser.parse_args()

    # Filter the model list
    specs = MODELS
    if args.only:
        specs = [m for m in MODELS if m.key in args.only]
    if args.skip:
        specs = [m for m in specs if m.key not in args.skip]

    logger.info(f"HuggingFace cache: {HF_CACHE}")

    if not args.hf_token:
        logger.warning(
            "HF_TOKEN not set. pyannote models (diarization, segmentation, wespeaker) "
            "require a token with accepted terms at https://huggingface.co/pyannote"
        )

    # Show status table
    print_status_table(specs)

    if args.dry_run:
        logger.info("Dry run complete. No models were downloaded.")
        return

    # Download missing models
    success = []
    failed = []
    skipped = []

    for spec in specs:
        logger.info(f"\n[{spec.key}] {spec.description} (~{spec.size_mb} MB)")

        if is_cached(spec) and not args.force:
            logger.info(f"  Already cached — skipping. Use --force to re-download.")
            skipped.append(spec.key)
            continue

        logger.info(f"  Starting download ...")
        if spec.download_fn == "causal_lm":
            ok = download_via_causal_lm(spec, args.hf_token)
        elif spec.download_fn == "transformers":
            ok = download_via_transformers(spec, args.hf_token)
        else:
            ok = download_via_snapshot(spec, args.hf_token)

        if ok:
            logger.info(f"  [OK] {spec.key} downloaded successfully")
            success.append(spec.key)
        else:
            logger.error(f"  [FAIL] {spec.key} FAILED")
            failed.append(spec.key)

    # Final summary
    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    if skipped:
        print(f"  Skipped (already cached):  {', '.join(skipped)}")
    if success:
        print(f"  Downloaded successfully:   {', '.join(success)}")
    if failed:
        print(f"  FAILED:                    {', '.join(failed)}")
        print()
        print("  Troubleshooting:")
        if any(k in failed for k in ("diarization", "campplus")):
            print("  -> pyannote community-1 requires HF_TOKEN with accepted terms.")
            print("     Visit: https://huggingface.co/pyannote/speaker-diarization-community-1")
            print("     Accept terms, then pass: --hf-token hf_xxx...")
        if "qwen3" in failed:
            print("  -> Qwen3 download failed. Ensure 'transformers>=4.45' is installed.")
            print("     Run: pip install -U transformers accelerate")
        print()

    if not failed:
        print()
        print("  All models ready. Next step:")
        print("  -> python tools/encrypt_models.py")
    print("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
