"""
Launcher — Phase 7
Main entry point for the AI Meeting Transcriber desktop application.

Responsibilities:
1. Verify license (30-day demo)
2. Verify runtime folders exist
3. Show splash screen
4. Start backend.exe without console window
5. Poll GET /health until HTTP 200 (no fixed sleep)
6. Launch frontend.exe
7. Monitor frontend — on exit, terminate backend + cleanup

Build to launcher.exe with:
    pyinstaller launcher.spec
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
import psutil

# ── Paths ─────────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).parent
else:
    # Development: launcher/ is one level below project root
    APP_DIR = Path(__file__).parent.parent

# PyInstaller places backend.exe inside Application/backend/
BACKEND_EXE = APP_DIR / "backend" / "backend.exe"

# Electron win-unpacked output — exe name matches AppName in package.json
# Primary: frontend/win-unpacked/AI Meeting Transcriber.exe
# Fallback: any .exe found at win-unpacked root
FRONTEND_DIR = APP_DIR / "frontend"
FRONTEND_WIN_UNPACKED = FRONTEND_DIR / "win-unpacked"

RUNTIME_DIR = APP_DIR / "runtime"

# Development fallbacks (when running as .py files)
BACKEND_DEV_CMD = [
    sys.executable, "-m", "uvicorn", "main:app",
    "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"
]
FRONTEND_DEV_CMD = ["npm", "run", "dev", "--prefix", str(APP_DIR / "frontend")]

BACKEND_HEALTH_URL = "http://127.0.0.1:8000/health"
BACKEND_HEALTH_TIMEOUT = 500  # max seconds to wait for backend (PyTorch cold-start is slow)
BACKEND_HEALTH_INTERVAL = .5  # poll every 500ms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(APP_DIR / "launcher.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── Subprocess helpers ────────────────────────────────────────
def _no_console_kwargs() -> dict:
    """Return subprocess kwargs that hide console windows on Windows."""
    if sys.platform == "win32":
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = subprocess.SW_HIDE
        return {
            "startupinfo": info,
            "creationflags": subprocess.CREATE_NO_WINDOW,
        }
    return {}


# Path where backend stdout+stderr is written for diagnostics
BACKEND_LOG = APP_DIR / "backend.log"


def start_backend() -> subprocess.Popen:
    """Start backend.exe (or dev server) without a visible console."""
    if BACKEND_EXE.exists():
        cmd = [str(BACKEND_EXE)]
        # PyInstaller backend.exe must run from its own directory so that
        # _internal/ and .env are resolved correctly via relative paths.
        cwd = str(BACKEND_EXE.parent)
        logger.info(f"Starting backend: {BACKEND_EXE}")
    else:
        # Development mode
        cmd = BACKEND_DEV_CMD
        cwd = str(APP_DIR / "backend")
        logger.info("Dev mode: starting uvicorn")

    # Write backend output to backend.log so errors are always visible
    backend_log_file = open(BACKEND_LOG, "w", encoding="utf-8", buffering=1)
    logger.info(f"Backend stdout/stderr → {BACKEND_LOG}")

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=backend_log_file,
        stderr=backend_log_file,
        **_no_console_kwargs(),
    )
    # Keep the file handle alive (proc holds a ref); also stream to launcher.log
    proc._backend_log_file = backend_log_file
    return proc


def wait_for_backend(timeout: float = BACKEND_HEALTH_TIMEOUT,
                     on_progress: Optional[callable] = None) -> bool:
    """
    Poll /health endpoint until HTTP 200 or timeout.
    Returns True if backend became ready within timeout.
    Never uses a fixed sleep.
    """
    deadline = time.monotonic() + timeout
    attempts = 0

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BACKEND_HEALTH_URL, timeout=2) as resp:
                if resp.status == 200:
                    logger.info(f"Backend ready after {attempts} polls ✓")
                    return True
        except (urllib.error.URLError, OSError):
            pass  # Still starting

        attempts += 1
        if on_progress:
            elapsed = time.monotonic() - (deadline - timeout)
            on_progress(elapsed)

        time.sleep(BACKEND_HEALTH_INTERVAL)

    return False


def launch_frontend() -> subprocess.Popen:
    """Launch the Electron frontend from frontend/win-unpacked/."""
    # Production: look in frontend/win-unpacked/ for any .exe
    if FRONTEND_WIN_UNPACKED.exists():
        exe_candidates = list(FRONTEND_WIN_UNPACKED.glob("*.exe"))
        if exe_candidates:
            # Prefer one that is NOT 'update.exe'
            app_exes = [e for e in exe_candidates if e.name.lower() != "update.exe"]
            frontend_exe = app_exes[0] if app_exes else exe_candidates[0]
            logger.info(f"Launching frontend: {frontend_exe}")
            return subprocess.Popen(
                [str(frontend_exe)],
                cwd=str(FRONTEND_WIN_UNPACKED),
                **_no_console_kwargs(),
            )

    # Legacy / dev fallback
    legacy_exe = APP_DIR / "frontend.exe"
    if legacy_exe.exists():
        logger.info(f"Launching frontend (legacy path): {legacy_exe}")
        return subprocess.Popen([str(legacy_exe)], cwd=str(APP_DIR), **_no_console_kwargs())

    # Development fallback: open browser
    import webbrowser
    webbrowser.open("http://localhost:5173")
    logger.info("Dev mode: opened browser at http://localhost:5173")
    return subprocess.Popen(["cmd", "/c", "pause"], **_no_console_kwargs())


def cleanup(backend_proc: Optional[subprocess.Popen], temp_dirs: list):
    """Terminate backend and clean temp decrypted model files."""
    if backend_proc and backend_proc.poll() is None:
        logger.info("Terminating backend...")
        try:
            backend_proc.terminate()
            backend_proc.wait(timeout=10)
        except Exception:
            try:
                backend_proc.kill()
            except Exception:
                pass

    # Clean temp decrypted model files
    import shutil
    for d in temp_dirs:
        p = Path(d)
        if p.exists():
            try:
                shutil.rmtree(p, ignore_errors=True)
                logger.info(f"Cleaned temp: {p}")
            except Exception:
                pass

    logger.info("Cleanup complete.")


def verify_runtime():
    """Check that required runtime directories exist."""
    required = [RUNTIME_DIR]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Required runtime directories missing:\n" + "\n".join(missing) +
            "\n\nPlease reinstall the application."
        )


def main():
    import tempfile

    backend_proc: Optional[subprocess.Popen] = None
    frontend_proc: Optional[subprocess.Popen] = None
    temp_cleanup_dirs = [Path(tempfile.gettempdir()) / "voicesum_runtime"]

    # Import splash on main thread
    sys.path.insert(0, str(Path(__file__).parent))
    from splash import SplashScreen
    from license_manager import verify_license

    splash = SplashScreen()

    def _worker():
        nonlocal backend_proc, frontend_proc

        try:
            # ── Step 0: License ─────────────────────────────
            splash.update_step(0)
            time.sleep(0.3)

            valid, msg = verify_license()
            if not valid:
                splash.close_with_error("Demo Expired", msg)
                return

            # ── Step 1: Runtime check ─────────────────────
            splash.update_step(1)
            try:
                verify_runtime()
            except RuntimeError as e:
                splash.close_with_error("Runtime Error", str(e))
                return

            time.sleep(0.5)

            # ── Step 2–3: Starting backend ─────────────────
            splash.update_step(2, "Starting AI Engine…")
            backend_proc = start_backend()
            time.sleep(0.8)

            splash.update_step(3, "Loading Voice Models…")

            def _on_progress(elapsed: float):
                pct = min(int(elapsed / BACKEND_HEALTH_TIMEOUT * 100), 95)
                splash._progress = 0.5 + pct / 200.0  # 50%–95%

            ready = wait_for_backend(on_progress=_on_progress)

            if not ready:
                # Check if backend crashed vs timed out
                rc = backend_proc.poll()
                status = f"Exit code {rc}" if rc is not None else "Timed out (still running but /health never responded)"

                # Read last 1500 chars from backend.log for the error dialog
                tail = ""
                try:
                    if BACKEND_LOG.exists():
                        text = BACKEND_LOG.read_text(encoding="utf-8", errors="replace")
                        tail = text[-1500:] if len(text) > 1500 else text
                except Exception:
                    pass

                logger.error(f"Backend failed: {status}")
                if tail:
                    logger.error(f"Backend log tail:\n{tail}")

                splash.close_with_error(
                    "Backend Failed to Start",
                    f"{status}\n\nSee backend.log for details:\n{tail[:400]}"
                )
                cleanup(backend_proc, temp_cleanup_dirs)
                return

            # ── Step 4: Ready ─────────────────────────────
            splash.update_step(4, "Services Ready!")
            time.sleep(0.6)

            # ── Step 5: Launch frontend ────────────────────
            splash.update_step(5, "Launching Application…")
            time.sleep(0.4)

            frontend_proc = launch_frontend()
            time.sleep(1.5)

            # Close splash
            splash.close()

            # ── Monitor frontend ───────────────────────────
            # Since Electron may immediately spawn child processes and detach,
            # monitor by checking if the process name is still active.
            time.sleep(2)
            exe_candidates = list(FRONTEND_WIN_UNPACKED.glob("*.exe"))
            app_exes = [e for e in exe_candidates if e.name.lower() != "update.exe"]
            exe_name = app_exes[0].name if app_exes else (exe_candidates[0].name if exe_candidates else "AI Meeting Transcriber.exe")
            
            logger.info(f"Monitoring frontend process: {exe_name}")
            
            # Wait for the process to appear (max 15 seconds)
            process_found = False
            for _ in range(15):
                for proc in psutil.process_iter(['name']):
                    try:
                        if proc.info['name'] == exe_name:
                            process_found = True
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if process_found:
                    break
                time.sleep(1)
            
            if not process_found:
                logger.warning(f"Could not find running process {exe_name} by name. Falling back to process handle wait.")
                frontend_proc.wait()
            else:
                # Keep loop running as long as the process is alive
                while True:
                    running = False
                    for proc in psutil.process_iter(['name']):
                        try:
                            if proc.info['name'] == exe_name:
                                running = True
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    if not running:
                        logger.info("Frontend process exited.")
                        break
                    time.sleep(2)

        except Exception as e:
            logger.error(f"Launcher error: {e}", exc_info=True)
            splash.close_with_error("Launcher Error", str(e))

        finally:
            cleanup(backend_proc, temp_cleanup_dirs)

    # Run worker in background thread; splash mainloop on main thread
    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()
    splash.run_mainloop()  # blocks until splash is closed

    # Wait for cleanup
    worker_thread.join(timeout=15)
    logger.info("Launcher exited cleanly.")


if __name__ == "__main__":
    main()
