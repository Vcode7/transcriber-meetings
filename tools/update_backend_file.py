"""
tools/update_backend_file.py
=============================
Fast file updater for Application/backend/.

Instead of re-running a full 10-15 minute PyInstaller build when you only edit
a Python file (e.g. services/ai_provider.py or routers/global_context_router.py),
this script updates the target file(s) instantly into Application/backend/
or triggers a fast 5-second incremental PyInstaller update.

Usage
-----
1. Update a specific file instantly (0.5 sec):
   python tools/update_backend_file.py services/ai_provider.py

2. Sync all modified backend python files (0.5 sec):
   python tools/update_backend_file.py --sync-all

3. Run fast incremental PyInstaller build (~5-10 sec):
   python tools/update_backend_file.py --incremental
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
APP_BACKEND_DIR = ROOT_DIR / "Application" / "backend"
INTERNAL_DIR = APP_BACKEND_DIR / "_internal"


def copy_file(rel_path_str: str) -> bool:
    """Copy a relative file from backend/ to Application/backend/ & _internal/."""
    src = BACKEND_DIR / rel_path_str
    if not src.exists():
        print(f"ERROR: Source file does not exist: {src}")
        return False

    rel_path = Path(rel_path_str)

    # Destination paths in Application/backend
    dests = [
        APP_BACKEND_DIR / rel_path,
        INTERNAL_DIR / rel_path,
    ]

    copied_any = False
    for dest in dests:
        if dest.parent.exists() or dest.parent.parent.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"[OK] Copied {rel_path_str} -> {dest}")
            copied_any = True

    if not copied_any:
        print(f"WARNING: Application/backend directory not found at {APP_BACKEND_DIR}. Build backend first.")
        return False

    return True


def sync_all_modified() -> int:
    """Sync all .py files under backend/ to Application/backend/."""
    count = 0
    for py_file in BACKEND_DIR.glob("**/*.py"):
        if "venv" in py_file.parts or "__pycache__" in py_file.parts:
            continue
        rel = py_file.relative_to(BACKEND_DIR).as_posix()
        if copy_file(rel):
            count += 1
    return count


def run_incremental_build() -> bool:
    """Run fast incremental PyInstaller build using existing build/backend_work cache."""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "tools/backend.spec",
        "--distpath", "Application/",
        "--workpath", "build/backend_work",
        "--noconfirm"
    ]
    print(f"\n[Incremental Build] Running: {' '.join(cmd)}\n")
    res = subprocess.run(cmd, cwd=ROOT_DIR)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Fast update for specific Python files in Application/backend."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Relative path to Python file inside backend/ (e.g. services/ai_provider.py)"
    )
    parser.add_argument(
        "--sync-all",
        action="store_true",
        help="Copy all modified Python files from backend/ to Application/backend/"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Run fast incremental PyInstaller build (~5-10 seconds)"
    )

    args = parser.parse_args()

    if args.file:
        copy_file(args.file)
        if args.incremental:
            run_incremental_build()
    elif args.sync_all:
        count = sync_all_modified()
        print(f"\n[OK] Synced {count} files to Application/backend/")
        if args.incremental:
            run_incremental_build()
    elif args.incremental:
        run_incremental_build()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
