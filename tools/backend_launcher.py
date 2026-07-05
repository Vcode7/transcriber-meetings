"""
backend_launcher.py — Tiny runtime bridge for the AI Meeting Transcriber backend.

This is compiled into backend.exe via PyInstaller (backend_launcher.spec).
It has ZERO ML dependencies — it compiles in seconds and weighs ~10 MB.

Responsibilities:
1. Locate the VoiceSum runtime Python (embeddable Python 3.12 in %ProgramData%\VoiceSum\runtime\)
2. Construct the correct PYTHONPATH / environment
3. Launch app.pyz via the runtime Python as a subprocess
4. Forward stdout/stderr to backend.log (same directory as this exe)
5. Exit with the same return code as the app.pyz process

The runtime location is searched in priority order:
  a. VOICESUM_RUNTIME env var (override for dev/testing)
  b. %ProgramData%\VoiceSum\runtime\   (production — shared, all users)
  c. %LOCALAPPDATA%\VoiceSum\runtime\  (fallback — per-user install)
  d. <exe_dir>\..\runtime_pkg\         (portable / USB stick mode)
"""
from __future__ import annotations

import os
import subprocess
import sys
import logging
from pathlib import Path

# ── Locate self ────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    EXE_DIR = Path(sys.executable).parent          # Application/backend/
    APP_DIR  = EXE_DIR.parent                      # Application/
else:
    EXE_DIR = Path(__file__).parent.parent / "Application" / "backend"
    APP_DIR  = EXE_DIR.parent

APP_PYZ   = EXE_DIR / "app.pyz"
BACKEND_LOG = APP_DIR / "backend.log"

# ── Runtime search locations (in priority order) ───────────────
RUNTIME_SEARCH = [
    os.environ.get("VOICESUM_RUNTIME", ""),
    str(Path(os.environ.get("ProgramData", "C:/ProgramData")) / "VoiceSum" / "runtime"),
    str(Path(os.environ.get("LOCALAPPDATA", "")) / "VoiceSum" / "runtime")
        if os.environ.get("LOCALAPPDATA") else "",
    str(APP_DIR / "runtime_pkg"),                  # portable / side-by-side
]

logging.basicConfig(
    filename=str(BACKEND_LOG),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger("backend_launcher")


def find_runtime() -> Path | None:
    """Return the first valid runtime directory, or None."""
    for candidate in RUNTIME_SEARCH:
        if not candidate:
            continue
        p = Path(candidate)
        py = p / "python" / "python.exe"
        if py.is_file():
            return p
    return None


def main() -> int:
    logger.info("backend_launcher starting")
    logger.info(f"  EXE_DIR : {EXE_DIR}")
    logger.info(f"  APP_PYZ : {APP_PYZ}")

    # ── Locate runtime ─────────────────────────────────────────
    runtime = find_runtime()
    if runtime is None:
        msg = (
            "VoiceSum Runtime not found.\n"
            "Please install VoiceSum-Runtime before running this application.\n\n"
            "Searched:\n" + "\n".join(f"  {p}" for p in RUNTIME_SEARCH if p)
        )
        logger.error(msg)
        # Write to stdout so launcher.exe can capture it
        print(msg, flush=True)
        return 1

    python_exe = runtime / "python" / "python.exe"
    site_packages = runtime / "python" / "Lib" / "site-packages"
    logger.info(f"  Runtime : {runtime}")
    logger.info(f"  Python  : {python_exe}")

    # ── Verify app.pyz ─────────────────────────────────────────
    if not APP_PYZ.is_file():
        msg = f"app.pyz not found: {APP_PYZ}\nReinstall the application."
        logger.error(msg)
        print(msg, flush=True)
        return 1

    # ── Build environment for the subprocess ───────────────────
    env = os.environ.copy()

    # Point Python at our site-packages
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(site_packages) + (os.pathsep + existing_pp if existing_pp else "")

    # Ensure ffmpeg/ffprobe in runtime are on PATH
    env["PATH"] = str(runtime) + os.pathsep + env.get("PATH", "")

    # Critical: tell the backend to run from EXE_DIR so relative paths
    # (.env, voicesum.db, uploads/) resolve correctly.
    env["VOICESUM_BASE_DIR"] = str(EXE_DIR)

    # Offline mode
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"

    # ── Launch app.pyz ─────────────────────────────────────────
    cmd = [str(python_exe), str(APP_PYZ)]
    logger.info(f"  CMD     : {' '.join(cmd)}")

    # On Windows, suppress the console window for the child process
    si = None
    cf = 0
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        cf = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(
        cmd,
        cwd=str(EXE_DIR),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        startupinfo=si,
        creationflags=cf,
    )

    logger.info(f"  PID     : {proc.pid}")
    rc = proc.wait()
    logger.info(f"  Exit    : {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
