from pathlib import Path

def _resolve_models_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    app_models = root / "Application" / "runtime" / "models"
    if app_models.is_dir():
        return app_models
    return root / "backend" / "runtime" / "models"

models = _resolve_models_dir()

checks = [
    ("audio_context/config.yaml",                   "pipeline config"),
    ("audio_context/embedding/pytorch_model.bin",   "embedding sub-model"),
    ("audio_context/segmentation/pytorch_model.bin","segmentation sub-model"),
    ("audio_context/plda/plda.npz",                 "PLDA model"),
    ("audio_context/plda/xvec_transform.npz",       "x-vector transform"),
    ("ecapa_tdnn/hyperparams.yaml",                 "ECAPA-TDNN config"),
    ("ecapa_tdnn/embedding_model.ckpt",             "ECAPA-TDNN weights"),
]

optional_checks = [
    ("eres2net_large/configuration.json",           "ERes2Net-Large config (optional)"),
    ("eres2net_large/eres2net_large_model.ckpt",    "ERes2Net-Large weights (optional)"),
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

for rel, desc in optional_checks:
    p = models / rel
    ok = p.exists()
    status = "OK" if ok else "OPTIONAL"
    print(f"[{status:<8}] {rel:<45} ({desc})")

print("=" * 70)
print("Result:", "READY" if all_ok else "INCOMPLETE - run download_speaker_models.py")