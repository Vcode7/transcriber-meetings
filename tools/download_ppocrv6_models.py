"""
download_ppocrv6_models.py — Download PP-OCRv6 Medium ONNX models into download/ppocrv6/.

Downloads:
  1. PaddlePaddle/PP-OCRv6_medium_det_onnx -> PP-OCRv6_medium_det.onnx & inference_det.yml
  2. PaddlePaddle/PP-OCRv6_medium_rec_onnx -> PP-OCRv6_medium_rec.onnx & inference_rec.yml
  3. ppocrv6_dict.txt (extracted from inference_rec.yml)
  4. ch_ppocr_mobile_v2.0_cls.onnx (direction classifier)

Run once during setup:
    python tools/download_ppocrv6_models.py
"""
import os
import sys
import shutil
import urllib.request
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = ROOT_DIR / "download" / "ppocrv6"

HF_BASE = "https://huggingface.co"
DET_REPO = "PaddlePaddle/PP-OCRv6_medium_det_onnx"
REC_REPO = "PaddlePaddle/PP-OCRv6_medium_rec_onnx"
CLS_URL = "https://github.com/RapidAI/RapidOCR/raw/main/python/rapidocr_onnxruntime/models/ch_ppocr_mobile_v2.0_cls_infer.onnx"

def download_file(url: str, dest_path: Path, desc: str = ""):
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[SKIP] {dest_path.name} already exists ({dest_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return
    print(f"[DOWNLOAD] {desc or dest_path.name} from {url}...")
    temp_path = dest_path.with_suffix(".tmp")
    
    # Try huggingface_hub first if available for HF urls
    if "huggingface.co" in url:
        try:
            from huggingface_hub import hf_hub_download
            parts = url.split("huggingface.co/")[1].split("/resolve/main/")
            if len(parts) == 2:
                repo_id, filename = parts[0], parts[1]
                print(f"  Using huggingface_hub: repo={repo_id}, file={filename}")
                cached_file = hf_hub_download(repo_id=repo_id, filename=filename)
                shutil.copy(cached_file, dest_path)
                print(f"[OK] {dest_path.name} saved ({dest_path.stat().st_size / 1024 / 1024:.2f} MB)")
                return
        except Exception as hf_err:
            print(f"  huggingface_hub download attempt failed ({hf_err}), falling back to direct stream...")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    temp_path.rename(dest_path)
    print(f"[OK] {dest_path.name} saved ({dest_path.stat().st_size / 1024 / 1024:.2f} MB)")

def extract_dictionary(rec_yml_path: Path, dict_dest_path: Path):
    if dict_dest_path.exists() and dict_dest_path.stat().st_size > 1000:
        print(f"[SKIP] {dict_dest_path.name} already exists")
        return
    print(f"[EXTRACT] Extracting character dictionary from {rec_yml_path.name}...")
    with open(rec_yml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    character_dict = data.get("PostProcess", {}).get("character_dict", [])
    if not character_dict:
        raise ValueError("character_dict not found in inference_rec.yml!")
    
    with open(dict_dest_path, "w", encoding="utf-8") as f:
        for char in character_dict:
            f.write(f"{char}\n")
    print(f"[OK] Extracted {len(character_dict)} characters to {dict_dest_path.name}")

def copy_or_download_cls(cls_dest: Path):
    if cls_dest.exists() and cls_dest.stat().st_size > 1000:
        print(f"[SKIP] {cls_dest.name} already exists")
        return
    # Check if local rapidocr_onnxruntime has it
    try:
        import rapidocr_onnxruntime
        pkg_dir = Path(rapidocr_onnxruntime.__file__).parent / "models"
        candidate = pkg_dir / "ch_ppocr_mobile_v2.0_cls_infer.onnx"
        if candidate.exists() and candidate.stat().st_size > 1000:
            print(f"[COPY] Found local cls model in rapidocr_onnxruntime: {candidate}")
            shutil.copy(candidate, cls_dest)
            print(f"[OK] Copied cls model to {cls_dest.name}")
            return
    except Exception:
        pass
    
    # Download cls model
    cls_hf_url = "https://huggingface.co/PaddlePaddle/ch_ppocr_mobile_v2.0_cls_infer/resolve/main/inference.onnx"
    try:
        download_file(cls_hf_url, cls_dest, desc="CLS Model from HuggingFace")
    except Exception:
        download_file(CLS_URL, cls_dest, desc="CLS Model from GitHub")

def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {DOWNLOAD_DIR}")

    # 1. Detection model & config
    det_onnx_url = f"{HF_BASE}/{DET_REPO}/resolve/main/inference.onnx"
    det_yml_url = f"{HF_BASE}/{DET_REPO}/resolve/main/inference.yml"
    download_file(det_onnx_url, DOWNLOAD_DIR / "PP-OCRv6_medium_det.onnx", "PP-OCRv6 Medium Det ONNX")
    download_file(det_yml_url, DOWNLOAD_DIR / "inference_det.yml", "Det Config")

    # 2. Recognition model & config
    rec_onnx_url = f"{HF_BASE}/{REC_REPO}/resolve/main/inference.onnx"
    rec_yml_url = f"{HF_BASE}/{REC_REPO}/resolve/main/inference.yml"
    download_file(rec_onnx_url, DOWNLOAD_DIR / "PP-OCRv6_medium_rec.onnx", "PP-OCRv6 Medium Rec ONNX")
    download_file(rec_yml_url, DOWNLOAD_DIR / "inference_rec.yml", "Rec Config")

    # 3. Extract dictionary
    extract_dictionary(DOWNLOAD_DIR / "inference_rec.yml", DOWNLOAD_DIR / "ppocrv6_dict.txt")

    # 4. Text classifier
    copy_or_download_cls(DOWNLOAD_DIR / "ch_ppocr_mobile_v2.0_cls.onnx")

    print("\n--- Model Verification ---")
    files = [
        "PP-OCRv6_medium_det.onnx",
        "PP-OCRv6_medium_rec.onnx",
        "ch_ppocr_mobile_v2.0_cls.onnx",
        "ppocrv6_dict.txt",
        "inference_det.yml",
        "inference_rec.yml"
    ]
    all_ok = True
    for fname in files:
        fpath = DOWNLOAD_DIR / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            print(f"  [OK] {fname}: {fpath.stat().st_size:,} bytes")
        else:
            print(f"  [FAIL] Missing {fname}")
            all_ok = False

    if all_ok:
        print("\nAll PP-OCRv6 Medium model files successfully stored in download/ppocrv6/!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
