"""
fix_align_tokenizer.py
─────────────────────────────────────────────────────────────────────────────
One-time fix for the alignment model tokenizer files missing from align_engine.dat.

Run this ONCE on any machine where the VoiceSum backend is installed:
    python fix_align_tokenizer.py

What it does:
  1. Finds the voicesum_runtime/hf_cache snapshot directory
  2. Writes the 4 missing facebook/wav2vec2-base tokenizer files there
     (vocab.json, tokenizer_config.json, special_tokens_map.json, preprocessor_config.json)
  3. These files persist across backend restarts (the hf_cache dir is not wiped)

Why needed:
  The align_engine.dat was packed with only model.safetensors (weights), but
  Wav2Vec2Processor.from_pretrained also needs the tokenizer config files.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import tempfile
from pathlib import Path

# The snapshot hash that align_engine.dat extracts to
SNAP_HASH = "f30924bbbfbf6058e04aab549531ea55c74928d0"

# Build the target path
snap_dir = (
    Path(tempfile.gettempdir())
    / "voicesum_runtime"
    / "hf_cache"
    / "models--facebook--wav2vec2-base"
    / "snapshots"
    / SNAP_HASH
)

# ── Tokenizer file contents for facebook/wav2vec2-base ─────────────────────
# Standard static files for the wav2vec2-base CTC tokenizer.
# Vocabulary: 28 characters + [UNK] + [PAD]  (verified against HF repo)

VOCAB = {
    "|": 0, "E": 1, "T": 2, "A": 3, "O": 4, "N": 5, "I": 6, "H": 7,
    "S": 8, "R": 9, "D": 10, "L": 11, "U": 12, "M": 13, "W": 14, "C": 15,
    "F": 16, "G": 17, "Y": 18, "P": 19, "B": 20, "V": 21, "K": 22, "'": 23,
    "X": 24, "J": 25, "Q": 26, "Z": 27, "[UNK]": 28, "[PAD]": 29,
}

VOCAB_JSON = json.dumps(VOCAB, indent=2)

TOKENIZER_CONFIG = json.dumps({
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
}, indent=2)

SPECIAL_TOKENS_MAP = json.dumps({
    "bos_token": "<s>",
    "cls_token": "<s>",
    "eos_token": "</s>",
    "mask_token": "<mask>",
    "pad_token": "[PAD]",
    "sep_token": "</s>",
    "unk_token": "[UNK]"
}, indent=2)

PREPROCESSOR_CONFIG = json.dumps({
    "do_normalize": True,
    "feature_extractor_type": "Wav2Vec2FeatureExtractor",
    "feature_size": 1,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "Wav2Vec2Processor",
    "return_attention_mask": False,
    "sampling_rate": 16000
}, indent=2)

FILES = {
    "vocab.json": VOCAB_JSON,
    "tokenizer_config.json": TOKENIZER_CONFIG,
    "special_tokens_map.json": SPECIAL_TOKENS_MAP,
    "preprocessor_config.json": PREPROCESSOR_CONFIG,
}


def main():
    print(f"Target snapshot directory:\n  {snap_dir}\n")

    if not snap_dir.exists():
        print("WARNING: Snapshot directory does not exist yet.")
        print("  This means the backend hasn't run yet (align_engine.dat not decrypted).")
        print("  Steps:")
        print("  1. Start the backend and wait for it to fully load (all models decrypted).")
        print("  2. Then run this script again — it will find the directory.")
        return

    already_have = [f for f in FILES if (snap_dir / f).exists()]
    to_write = [f for f in FILES if f not in already_have]

    if already_have:
        print(f"Already present: {', '.join(already_have)}")

    if not to_write:
        print("OK - All tokenizer files already present — nothing to do!")
        print("  The alignment model should work on next backend startup.")
        return

    snap_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in FILES.items():
        if filename in to_write:
            dest = snap_dir / filename
            dest.write_text(content, encoding="utf-8")
            print(f"  Written: {dest.name}")

    print(f"\nDone! {len(to_write)} tokenizer file(s) added.")
    print("  Restart the backend — alignment should now work correctly.")
    print()
    print("Final snapshot contents:")
    for f in sorted(snap_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:40s}  {size:>10,} bytes")


if __name__ == "__main__":
    main()
