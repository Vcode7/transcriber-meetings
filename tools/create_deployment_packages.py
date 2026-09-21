"""
Deployment Packaging Tool for VoiceSum.
Creates:
  deployment/frontend.zip
  deployment/backend.zip
  deployment/backend_venv.zip
  deployment/license.key

Enforces strict exclusion rules and verifies the integrity of created packages.
"""
import os
from pathlib import Path
import sys
import zipfile
import shutil

ROOT_DIR = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = ROOT_DIR / "deployment"
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_DIR = ROOT_DIR / "backend"
ADMIN_DIR = ROOT_DIR / "admin_license"

sys.path.insert(0, str(ROOT_DIR))
from admin_license.generate_license import create_signed_license


def log(msg: str):
    print(f"[Packaging] {msg}", flush=True)


def init_deployment_dir():
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Target deployment directory: {DEPLOYMENT_DIR.resolve()}")


def generate_client_license():
    log("Step 1: Generating signed client license (Expiry: 2026-10-15)...")
    lic_path = DEPLOYMENT_DIR / "license.key"
    create_signed_license(
        customer="Offline Client Demo",
        license_type="demo",
        expiry_date="2026-10-15",
        machine_id=None,
        features=["all"],
        output_path=lic_path,
    )
    # Also ensure backend has a copy for development testing
    shutil.copy2(lic_path, BACKEND_DIR / "license.key")
    log("Signed license.key generated successfully.")


def package_frontend():
    log("Step 2: Packaging frontend.zip (including node_modules, source, configs)...")
    out_zip = DEPLOYMENT_DIR / "frontend.zip"
    if out_zip.exists():
        out_zip.unlink()

    ignored_root_dirs = {".git", ".pytest_cache", "dist"}
    ignored_sub_dirs = {".git", ".pytest_cache"}
    ignored_extensions = {".log", ".tmp"}

    file_count = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(FRONTEND_DIR):
            rel_root = Path(root).relative_to(FRONTEND_DIR)
            if rel_root == Path("."):
                dirs[:] = [d for d in dirs if d not in ignored_root_dirs]
            else:
                dirs[:] = [d for d in dirs if d not in ignored_sub_dirs]

            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in ignored_extensions:
                    continue
                if f.startswith("vite-ui-check"):
                    continue

                arcname = Path("frontend") / rel_root / f
                zf.write(p, arcname.as_posix())
                file_count += 1

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    log(f"frontend.zip created: {file_count} files, {size_mb:.2f} MB")


def package_backend():
    log("Step 3: Packaging backend.zip (source, config, license verifier, public key)...")
    out_zip = DEPLOYMENT_DIR / "backend.zip"
    if out_zip.exists():
        out_zip.unlink()

    # Explicit exclusions
    exclude_dirs = {
        "venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        "debug_ocr",
        "tests",
        "admin_license",
    }
    exclude_extensions = {
        ".pyc",
        ".pyo",
        ".log",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".docx",
        ".txt",
    }
    exclude_files = {
        "license.key",          # Must NOT be inside backend.zip
        "private_key.pem",      # Must NEVER be distributed
        "voicesum.log",
        "backend_debug.txt",
        "voicesum_diagnostics.json",
        "required-runtime-version.txt",
    }

    # Files to explicitly include if in root of backend:
    allowed_root_files = {
        "main.py",
        "config.py",
        "database.py",
        "license.py",
        "public_key.pem",
        "update_prompts.py",
        ".env",
        ".env.example",
        "requirements.txt",
        "run_backend.bat",
        "backend.spec",
    }

    allowed_subdirs = {
        "models",
        "routers",
        "services",
        "tasks",
        "utils",
    }

    file_count = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Root files
        for f in allowed_root_files:
            fp = BACKEND_DIR / f
            if fp.is_file():
                zf.write(fp, (Path("backend") / f).as_posix())
                file_count += 1

        # 2. Subdirectories
        for sdir in allowed_subdirs:
            dp = BACKEND_DIR / sdir
            if not dp.is_dir():
                continue
            for root, dirs, files in os.walk(dp):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                rel_root = Path(root).relative_to(BACKEND_DIR)
                for file in files:
                    fp = Path(root) / file
                    if fp.name in exclude_files:
                        continue
                    if fp.suffix.lower() in exclude_extensions:
                        continue
                    # Never include private key
                    if "private" in fp.name.lower():
                        continue
                    zf.write(fp, (Path("backend") / rel_root / file).as_posix())
                    file_count += 1

        # 3. Create clean runtime placeholder directories in archive
        for pdir in ["runtime/models", "runtime/uploads", "runtime/data"]:
            zfi = zipfile.ZipInfo(f"backend/{pdir}/")
            zf.writestr(zfi, "")

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    log(f"backend.zip created: {file_count} files, {size_mb:.2f} MB")


def package_backend_venv():
    log("Step 4: Packaging backend_venv.zip (existing virtual environment)...")
    out_zip = DEPLOYMENT_DIR / "backend_venv.zip"
    if out_zip.exists():
        out_zip.unlink()

    venv_src = BACKEND_DIR / "venv"
    if not venv_src.is_dir():
        raise RuntimeError(f"Virtual environment not found at: {venv_src}")

    exclude_dirs = {"__pycache__", ".pytest_cache"}
    exclude_extensions = {".log", ".tmp"}

    file_count = 0
    # Store with standard deflate compression
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for root, dirs, files in os.walk(venv_src):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            rel_root = Path(root).relative_to(venv_src)

            for f in files:
                fp = Path(root) / f
                if fp.suffix.lower() in exclude_extensions:
                    continue
                # Ensure no private key leaks
                if "private_key" in f.lower() or "admin_license" in str(fp).lower():
                    continue

                arcname = Path("venv") / rel_root / f
                zf.write(fp, arcname.as_posix())
                file_count += 1
                if file_count % 5000 == 0:
                    log(f"  Packaged {file_count} files into backend_venv.zip...")

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    log(f"backend_venv.zip created: {file_count} files, {size_mb:.2f} MB ({size_mb/1024:.2f} GB)")


def verify_packages():
    log("Step 5: Verifying all packages and enforcing security constraints...")
    
    # 1. Verify license.key exists and is outside backend.zip
    lic_file = DEPLOYMENT_DIR / "license.key"
    assert lic_file.exists(), "license.key missing in deployment!"
    
    # 2. Check backend.zip
    backend_zip = DEPLOYMENT_DIR / "backend.zip"
    with zipfile.ZipFile(backend_zip, "r") as zf:
        names = zf.namelist()
        for name in names:
            lower = name.lower()
            assert "private_key" not in lower and "admin_license" not in lower, f"SECURITY VIOLATION: Private key found in backend.zip: {name}"
            assert "license.key" not in lower, f"license.key must NOT be inside backend.zip: {name}"
            assert "ffmpeg.exe" not in lower, f"FFmpeg found in backend.zip: {name}"
            assert not lower.endswith(".safetensors"), f"Model weights found in backend.zip: {name}"
            assert not lower.endswith(".db"), f"Database file found in backend.zip: {name}"
        assert "backend/license.py" in names, "backend/license.py missing in backend.zip!"
        assert "backend/main.py" in names, "backend/main.py missing in backend.zip!"
        assert "backend/public_key.pem" in names, "backend/public_key.pem missing in backend.zip!"
        assert "backend/run_backend.bat" in names, "backend/run_backend.bat missing in backend.zip!"
    log("backend.zip verification PASSED (Zero private keys, zero models, zero databases).")

    # 3. Check frontend.zip
    frontend_zip = DEPLOYMENT_DIR / "frontend.zip"
    with zipfile.ZipFile(frontend_zip, "r") as zf:
        names = zf.namelist()
        for name in names:
            lower = name.lower()
            assert "private_key" not in lower and not lower.endswith(".pem"), f"SECURITY VIOLATION: Private key found in frontend.zip: {name}"
            assert "admin_license" not in lower, f"Admin directory found in frontend.zip: {name}"
        assert "frontend/package.json" in names, "package.json missing in frontend.zip!"
        assert any(n.startswith("frontend/node_modules/") for n in names), "node_modules missing in frontend.zip!"
        assert "frontend/node_modules/vite/dist/node/cli.js" in names, "vite/dist/node/cli.js missing in frontend.zip!"
        assert "frontend/src/App.tsx" in names, "src/App.tsx missing in frontend.zip!"
        assert "frontend/run_frontend.bat" in names, "run_frontend.bat missing in frontend.zip!"
    log("frontend.zip verification PASSED (node_modules complete, source complete).")

    # 4. Check backend_venv.zip
    venv_zip = DEPLOYMENT_DIR / "backend_venv.zip"
    with zipfile.ZipFile(venv_zip, "r") as zf:
        names = zf.namelist()
        for name in names:
            lower = name.lower()
            assert "private_key" not in lower and "admin_license" not in lower, f"SECURITY VIOLATION: Private key found in backend_venv.zip: {name}"
            assert "admin_license" not in lower, f"Admin directory found in backend_venv.zip: {name}"
        assert any("site-packages" in n for n in names), "site-packages missing in backend_venv.zip!"
        assert any("fastapi" in n for n in names), "FastAPI missing in backend_venv.zip!"
        assert any("torch" in n for n in names), "PyTorch missing in backend_venv.zip!"
    log("backend_venv.zip verification PASSED (venv dependencies complete).")
    
    log("ALL PACKAGING VERIFICATIONS SUCCEEDED!")


def main():
    init_deployment_dir()
    if not (DEPLOYMENT_DIR / "license.key").exists():
        generate_client_license()
    if not (DEPLOYMENT_DIR / "backend.zip").exists():
        package_backend()
    if not (DEPLOYMENT_DIR / "frontend.zip").exists():
        package_frontend()
    if not (DEPLOYMENT_DIR / "backend_venv.zip").exists():
        package_backend_venv()
    verify_packages()
    log("Packaging workflow completed successfully!")


if __name__ == "__main__":
    main()
