"""
download_speaker_models.py
==========================
Downloads and caches all speaker-pipeline models into runtime/models/.

Run once (with internet access + HF token) before deploying offline:

    python tools/download_speaker_models.py --hf-token hf_...

Models downloaded
-----------------
1. pyannote/speaker-diarization-community-1  → runtime/models/audio_context/
   Complete snapshot including all sub-models:
     audio_context/config.yaml                   (pipeline config)
     audio_context/embedding/pytorch_model.bin   (speaker embedding for diarization)
     audio_context/segmentation/pytorch_model.bin (segmentation model)
     audio_context/plda/plda.npz                 (PLDA for VBx clustering)
     audio_context/plda/xvec_transform.npz       (x-vector transform for VBx)

2. SpeechBrain ECAPA-TDNN                    → runtime/models/ecapa_tdnn/
   (speechbrain/spkrec-ecapa-voxceleb — used by embedding.py for speaker ID)
     ecapa_tdnn/hyperparams.yaml
     ecapa_tdnn/embedding_model.ckpt
     ecapa_tdnn/classifier.ckpt
     ecapa_tdnn/mean_var_norm_emb.ckpt

NOTE: The community-1 pipeline bundles its own segmentation and embedding
sub-models as sub-directories. We do NOT download separate wespeaker ONNX
or segmentation-3.0 repos — everything lives inside audio_context/.

All downloads are idempotent: existing files are verified and skipped.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
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

def _resolve_models_dir() -> Path:
    """Return the models directory the backend will actually read from."""
    # Priority 1: Application/runtime/models (packaged release)
    app_models = _PROJECT_ROOT / "Application" / "runtime" / "models"
    if app_models.is_dir():
        return app_models
    # Priority 2: Application/backend/runtime/models (legacy layout)
    app_be_models = _PROJECT_ROOT / "Application" / "backend" / "runtime" / "models"
    if app_be_models.is_dir():
        return app_be_models
    # Priority 3: backend/runtime/models (development)
    return _PROJECT_ROOT / "backend" / "runtime" / "models"

_MODELS_DIR = _resolve_models_dir()


def _models_dir(override: str | None) -> Path:
    if override:
        p = Path(override).resolve()
    else:
        p = _MODELS_DIR
    p.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models directory: {p}")
    return p


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hf_snapshot(
    repo_id: str,
    dest: Path,
    token: str,
    ignore_patterns: list[str] | None = None,
    allow_patterns: list[str] | None = None,
) -> None:
    """Download a full HuggingFace repo snapshot into dest/."""
    logger.info(f"[DOWNLOAD] {repo_id} → {dest}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    kwargs: dict = {
        "repo_id": repo_id,
        "local_dir": str(dest),
        "token": token,
    }
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns

    snapshot_download(**kwargs)
    logger.info(f"[OK] {repo_id} downloaded to {dest}")


# ── Model download functions ───────────────────────────────────────────────────

def download_audio_context(models_dir: Path, token: str) -> None:
    """
    Download the complete pyannote/speaker-diarization-community-1 snapshot
    into audio_context/. This includes ALL sub-models as sub-directories:
      audio_context/config.yaml
      audio_context/embedding/pytorch_model.bin   ← speaker embedding for diarization
      audio_context/segmentation/pytorch_model.bin ← segmentation model
      audio_context/plda/plda.npz                 ← PLDA for VBx clustering
      audio_context/plda/xvec_transform.npz       ← x-vector transform
    """
    dest = models_dir / "audio_context"

    # Check whether all required sub-files already exist
    required = [
        dest / "config.yaml",
        dest / "embedding" / "pytorch_model.bin",
        dest / "segmentation" / "pytorch_model.bin",
        dest / "plda" / "plda.npz",
        dest / "plda" / "xvec_transform.npz",
    ]
    missing = [p for p in required if not p.exists()]
    if not missing:
        logger.info(
            f"[SKIP] pyannote/speaker-diarization-community-1 — all required files present at {dest}"
        )
        return

    logger.info(
        f"[DOWNLOAD] pyannote/speaker-diarization-community-1 → {dest}\n"
        f"  Missing files: {[str(m.relative_to(dest)) for m in missing]}"
    )

    # Download the complete snapshot (includes embedding/, segmentation/, plda/ sub-dirs)
    # Ignore large media/docs that aren't needed at runtime
    _hf_snapshot(
        repo_id="pyannote/speaker-diarization-community-1",
        dest=dest,
        token=token,
        ignore_patterns=["*.gif", "*.png", "*.jpg", "*.md", ".gitattributes"],
    )

    # Verify after download
    still_missing = [p for p in required if not p.exists()]
    if still_missing:
        logger.error(
            f"[ERROR] After download, still missing:\n"
            + "\n".join(f"  {p}" for p in still_missing)
        )
    else:
        logger.info("[OK] audio_context/ — all required sub-model files present ✓")


def download_ecapa_tdnn(models_dir: Path, token: str) -> None:
    """
    Download SpeechBrain ECAPA-TDNN speaker embedding model for speaker identification.

    Source: speechbrain/spkrec-ecapa-voxceleb
    Saved as: runtime/models/ecapa_tdnn/

    Required files (SpeechBrain downloads all of these automatically):
      hyperparams.yaml          — model config
      embedding_model.ckpt      — ECAPA-TDNN encoder weights
      classifier.ckpt           — classification layer weights
      mean_var_norm_emb.ckpt    — embedding normalisation parameters
    """
    dest = models_dir / "ecapa_tdnn"
    required_files = [
        dest / "hyperparams.yaml",
        dest / "embedding_model.ckpt",
    ]
    missing = [p for p in required_files if not p.exists()]
    if not missing:
        logger.info(f"[SKIP] ECAPA-TDNN — all required files present at {dest}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"[DOWNLOAD] speechbrain/spkrec-ecapa-voxceleb → {dest}")

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        logger.error(
            "speechbrain is not installed. Run: pip install speechbrain>=1.0.0"
        )
        sys.exit(1)

    try:
        # Setting source AND savedir to the same local path causes SpeechBrain
        # to download all files into dest/ and then load from there.
        os.environ.setdefault("HF_TOKEN", token)
        EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(dest),
        )
        logger.info(f"[OK] ECAPA-TDNN ready at {dest}")
    except Exception as e:
        logger.error(
            f"Could not download ECAPA-TDNN: {e}\n"
            "Please manually download speechbrain/spkrec-ecapa-voxceleb\n"
            f"and place the files at: {dest}\n"
            "Required: hyperparams.yaml, embedding_model.ckpt, classifier.ckpt, "
            "mean_var_norm_emb.ckpt"
        )
        sys.exit(1)


def verify_models(models_dir: Path) -> bool:
    """Print a summary of all required files and whether they exist."""
    checks = [
        # (relative path inside models_dir, description)
        ("audio_context/config.yaml",                   "community-1 pipeline config"),
        ("audio_context/embedding/pytorch_model.bin",   "community-1 embedding sub-model"),
        ("audio_context/segmentation/pytorch_model.bin","community-1 segmentation sub-model"),
        ("audio_context/plda/plda.npz",                 "community-1 PLDA model"),
        ("audio_context/plda/xvec_transform.npz",       "community-1 x-vector transform"),
        ("ecapa_tdnn/hyperparams.yaml",                 "ECAPA-TDNN config (speaker ID)"),
        ("ecapa_tdnn/embedding_model.ckpt",             "ECAPA-TDNN weights (speaker ID)"),
    ]
    all_ok = True
    print("\n================== Model Verification ==================")
    for rel_path, desc in checks:
        p = models_dir / rel_path
        ok = p.exists()
        status = "OK" if ok else "MISSING"
        print(f"  [{status:<8}]  {rel_path:<45} ({desc})")
        if not ok:
            all_ok = False
    print("========================================================\n")
    return all_ok


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download speaker-pipeline models for offline use."
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Override destination models directory (default: backend/runtime/models)",
    )
    parser.add_argument(
        "--skip-diarization",
        action="store_true",
        help="Skip downloading pyannote community-1 diarization snapshot (audio_context/)",
    )
    parser.add_argument(
        "--skip-ecapa",
        action="store_true",
        help="Skip downloading SpeechBrain ECAPA-TDNN embedding model (ecapa_tdnn/)",
    )
    args = parser.parse_args()

    token = args.hf_token
    if not token:
        logger.warning(
            "No HF_TOKEN provided. Downloads from gated HuggingFace repos will fail.\n"
            "Set --hf-token or export HF_TOKEN=hf_..."
        )

    mdir = _models_dir(args.models_dir)

    if not args.skip_diarization:
        logger.info("=== Step 1/2: pyannote/speaker-diarization-community-1 (complete snapshot) ===")
        download_audio_context(mdir, token)
    else:
        logger.info("[SKIP] Diarization models (--skip-diarization)")

    if not args.skip_ecapa:
        logger.info("=== Step 2/2: SpeechBrain ECAPA-TDNN (speaker identification) ===")
        download_ecapa_tdnn(mdir, token)
    else:
        logger.info("[SKIP] ECAPA-TDNN model (--skip-ecapa)")

    ok = verify_models(mdir)
    if ok:
        logger.info("All speaker models ready. Backend can now run fully offline.")
    else:
        logger.error("Some models are missing — see above. Re-run this script to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
