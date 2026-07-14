#!/usr/bin/env python3
"""
validate_embedding.py
=====================
Validate the SpeechBrain ECAPA-TDNN speaker embedding pipeline.

Usage
-----
    # From the project root:
    python tools/validate_embedding.py path/to/audio.wav

    # Test with two separate files for cross-speaker comparison:
    python tools/validate_embedding.py path/to/speaker_a.wav path/to/speaker_b.wav

What this script checks
-----------------------
1. Audio loads and resamples to 16 kHz correctly.
2. SpeechBrain ECAPA-TDNN model loads from ecapa_tdnn/ without internet.
3. Embedding shape is (192,) — 192-dimensional ECAPA output.
4. Embedding norm ≈ 1.0 (L2-normalised).
5. Two runs of the same audio produce identical embeddings (deterministic).
6. Optionally: cosine similarity between two different audio files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── Add the backend directory to sys.path ─────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _check_imports() -> None:
    """Fail early with a helpful message if key dependencies are missing."""
    missing = []
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        missing.append("torchaudio (pip install torchaudio)")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch (pip install torch)")
    try:
        import speechbrain  # noqa: F401
    except ImportError:
        missing.append("speechbrain (pip install speechbrain>=1.0.0)")
    if missing:
        print("ERROR: Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


def validate_single(wav_path: str) -> np.ndarray:
    """Run the full pipeline on a single WAV file and print diagnostics."""
    from services.embedding import (
        EMBEDDING_DIM,
        SAMPLE_RATE,
        _load_audio,
        extract_embedding,
        get_encoder,
    )

    print(f"\n{'='*60}")
    print(f"  File: {wav_path}")
    print(f"{'='*60}")

    # ── Step 1: Load audio ────────────────────────────────────────────────────
    audio, sr = _load_audio(wav_path, target_sr=SAMPLE_RATE)
    duration_sec = len(audio) / SAMPLE_RATE
    print(f"\n[Audio]")
    print(f"  Sample rate : {sr} Hz")
    print(f"  Samples     : {len(audio)}")
    print(f"  Duration    : {duration_sec:.2f} s")
    print(f"  dtype       : {audio.dtype}")
    print(f"  Amplitude   : [{audio.min():.4f}, {audio.max():.4f}]")

    if duration_sec < 1.0:
        print("\nWARNING: Audio is shorter than 1 second — embedding will be None.")

    # ── Step 2: Load / verify encoder ────────────────────────────────────────
    print(f"\n[Model]")
    try:
        encoder = get_encoder()
        print(f"  ✓ SpeechBrain ECAPA-TDNN loaded")
        print(f"  Expected embedding dim: {EMBEDDING_DIM}")
    except Exception as e:
        print(f"  ERROR: Could not load ECAPA-TDNN model: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # ── Step 3: Extract embedding (run 1) ─────────────────────────────────────
    embedding1 = extract_embedding(audio, sr=SAMPLE_RATE)
    print(f"\n[Embedding — Run 1]")
    if embedding1 is None:
        print("  ERROR: extract_embedding returned None. Check model path and audio length.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    print(f"  Shape       : {embedding1.shape}  (should be ({EMBEDDING_DIM},))")
    print(f"  dtype       : {embedding1.dtype}  (should be float32)")
    emb_norm1 = float(np.linalg.norm(embedding1))
    print(f"  L2 norm     : {emb_norm1:.6f}  (should be ≈ 1.0 after L2-normalisation)")
    if abs(emb_norm1 - 1.0) > 0.01:
        print(f"  WARNING: Embedding norm is not ≈ 1.0")
    else:
        print(f"  ✓ Embedding is L2-normalised")

    if embedding1.shape != (EMBEDDING_DIM,):
        print(f"  ERROR: Expected shape ({EMBEDDING_DIM},), got {embedding1.shape}")
    else:
        print(f"  ✓ Embedding dimension = {EMBEDDING_DIM}")

    # ── Step 4: Extract embedding (run 2) — determinism check ─────────────────
    embedding2 = extract_embedding(audio, sr=SAMPLE_RATE)
    print(f"\n[Embedding — Run 2 (determinism check)]")
    if embedding2 is None:
        print("  ERROR: Second run returned None.")
        return embedding1

    max_diff = float(np.abs(embedding1 - embedding2).max())
    cosine_self = float(
        np.dot(embedding1, embedding2) /
        (np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-9)
    )
    print(f"  Max element diff : {max_diff:.2e}")
    print(f"  Cosine self-sim  : {cosine_self:.6f}  (should be ≈ 1.0)")
    if max_diff < 1e-5:
        print(f"  ✓ Effectively deterministic")
    else:
        print(f"  WARNING: Non-deterministic output — check model/device settings.")

    return embedding1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate SpeechBrain ECAPA-TDNN speaker embedding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "wav",
        nargs="+",
        help="Path(s) to WAV file(s). Pass two files to compare embeddings.",
    )
    args = parser.parse_args()

    _check_imports()

    embeddings = []
    for wav_path in args.wav:
        if not Path(wav_path).exists():
            print(f"ERROR: File not found: {wav_path}")
            sys.exit(1)
        emb = validate_single(wav_path)
        embeddings.append((wav_path, emb))

    # ── Cross-file comparison (if two files given) ────────────────────────────
    if len(embeddings) == 2:
        path_a, emb_a = embeddings[0]
        path_b, emb_b = embeddings[1]
        cosine = float(
            np.dot(emb_a, emb_b) /
            (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9)
        )
        print(f"\n{'='*60}")
        print(f"  Cross-file Cosine Similarity")
        print(f"{'='*60}")
        print(f"  File A: {path_a}")
        print(f"  File B: {path_b}")
        print(f"  Cosine similarity: {cosine:.4f}")
        if cosine > 0.75:
            print(f"  → Likely SAME speaker (threshold 0.75)")
        elif cosine > 0.65:
            print(f"  → Possibly same speaker (marginal — threshold 0.65–0.75)")
        else:
            print(f"  → Likely DIFFERENT speakers (cosine < 0.65)")

    print(f"\n{'='*60}")
    print("  Validation complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
