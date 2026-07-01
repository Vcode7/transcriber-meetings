import os
import zipfile
from huggingface_hub import snapshot_download, login

# ============================================================
# Configuration
# ============================================================

HF_TOKEN = ""

MODEL_ID = "Qwen/Qwen3-4B"

DOWNLOAD_DIR = "models"
MODEL_DIR = os.path.join(DOWNLOAD_DIR, "Qwen3-4B")
ZIP_FILE = "Qwen3-4B.zip"

# ============================================================
# Login
# ============================================================

login(token=HF_TOKEN)

# ============================================================
# Download Model
# ============================================================

print(f"Downloading {MODEL_ID}...")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
    token=HF_TOKEN,
)

print("Download completed.")

# ============================================================
# Create ZIP
# ============================================================

print("Creating ZIP archive...")

with zipfile.ZipFile(
    ZIP_FILE,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
) as zipf:
    for root, _, files in os.walk(MODEL_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            archive_name = os.path.relpath(file_path, DOWNLOAD_DIR)
            zipf.write(file_path, archive_name)

print("ZIP created successfully!")

# ============================================================
# Show Sizes
# ============================================================

model_size = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, files in os.walk(MODEL_DIR)
    for f in files
)

zip_size = os.path.getsize(ZIP_FILE)

print(f"\nModel Size : {model_size / (1024**3):.2f} GB")
print(f"ZIP Size   : {zip_size / (1024**3):.2f} GB")