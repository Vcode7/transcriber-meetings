# make_app_pyz.py -- Package the backend source into a portable app.pyz
#
# Usage:
#     python tools/make_app_pyz.py
#     python tools/make_app_pyz.py --src backend/ --out Application/backend/app.pyz
#
# The resulting app.pyz is executed by the runtime Python:
#     runtime\python\python.exe backend\app.pyz
#
# This runs main:app via uvicorn. The __main__.py entry point inside the
# zip means:  python app.pyz   works without any -m flag.
import argparse
import sys
import zipfile
import os
import shutil
from pathlib import Path

# Directories/files to exclude from the zip
EXCLUDE_DIRS = {
    "venv", "__pycache__", ".git", ".mypy_cache", ".pytest_cache",
    "node_modules", "dist", "build", ".tox",
    # Runtime data — should NOT be in the source zip
    "checkpoints",    # large model weights (handled separately in Step 9)
    "uploads",        # user audio files
    "runtime",        # development runtime symlink / data dir
    "data",           # SQLite DB
}
EXCLUDE_EXTS = {
    ".pyc", ".pyo", ".pyd",               # bytecode / compiled extensions
    ".log", ".db", ".db-shm", ".db-wal",  # runtime state
    ".zip", ".tar.gz",                     # archives
    ".pth",  ".pt", ".ckpt",              # model weights
    ".wav", ".mp3", ".mp4", ".m4a",       # audio/video
    ".dat",                                # encrypted model files
}
EXCLUDE_FILES = {".env", ".env.example", ".gitignore"}



def iter_backend_files(src: Path):
    """Yield (arcname, abs_path) for every file to include in the pyz."""
    for root, dirs, files in os.walk(src):
        # Prune excluded dirs in-place (os.walk respects this)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        root_path = Path(root)
        for fname in files:
            if fname in EXCLUDE_FILES:
                continue
            if Path(fname).suffix in EXCLUDE_EXTS:
                continue
            abs_path = root_path / fname
            # arcname is relative to src (so backend/routers/audio.py -> routers/audio.py)
            arcname = abs_path.relative_to(src).as_posix()
            yield arcname, abs_path


MAIN_PY = '''\
"""
app.pyz entry point — invoked when the pyz is run directly.
Equivalent to: uvicorn main:app --host 127.0.0.1 --port 8000
"""
import sys
import os
import runpy

# app.pyz itself is the first element of sys.path when run directly.
# Nothing else needed — all source modules are inside the zip.

def _run():
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        log_level="warning",
        access_log=False,
    )

if __name__ == "__main__":
    _run()
'''


def build_pyz(src: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.zip")

    print(f"  Packaging {src} -> {out}")
    file_count = 0

    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Write __main__.py entry point
        zf.writestr("__main__.py", MAIN_PY)

        for arcname, abs_path in iter_backend_files(src):
            zf.write(abs_path, arcname)
            file_count += 1
            if file_count % 50 == 0:
                print(f"    ...{file_count} files", end="\r")

    # Rename to final output (atomic-ish on Windows)
    if out.exists():
        out.unlink()
    shutil.move(str(tmp), str(out))

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  Done: {file_count} files, {size_mb:.1f} MB -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Package backend source into app.pyz")
    parser.add_argument("--src",  default="backend",
                        help="Backend source directory (default: backend/)")
    parser.add_argument("--out",  default=r"Application\backend\app.pyz",
                        help=r"Output path (default: Application\backend\app.pyz)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    src = (project_root / args.src).resolve()
    out = (project_root / args.out).resolve()

    if not src.is_dir():
        print(f"ERROR: Source directory not found: {src}", file=sys.stderr)
        sys.exit(1)

    build_pyz(src, out)


if __name__ == "__main__":
    main()
