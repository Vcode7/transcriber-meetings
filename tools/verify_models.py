from pathlib import Path

models = Path("backend/runtime/models")

checks = [
    ("audio_context/config.yaml",                   "pipeline config"),
    ("audio_context/embedding/pytorch_model.bin",   "embedding sub-model"),
    ("audio_context/segmentation/pytorch_model.bin","segmentation sub-model"),
    ("audio_context/plda/plda.npz",                 "PLDA model"),
    ("audio_context/plda/xvec_transform.npz",       "x-vector transform"),
    ("ecapa_tdnn/hyperparams.yaml",                 "ECAPA-TDNN config"),
    ("ecapa_tdnn/embedding_model.ckpt",             "ECAPA-TDNN weights"),
]

print()
print("Model Verification")
print("=" * 70)

all_ok = True

for rel, desc in checks:
    p = models / rel
    ok = p.exists()
    status = "OK" if ok else "MISSING"
    print(f"[{status:<8}] {rel:<45} ({desc})")
    if not ok:
        all_ok = False

print("=" * 70)
print("Result:", "READY" if all_ok else "INCOMPLETE - run download_speaker_models.py")