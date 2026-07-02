"""
updater.py -- Client-Side Offline Patch Applier
================================================

Reads a patch ZIP created by create_patch.py and applies it to the installed
Application/ directory.

How it works:
1. User opens updater.exe (or updater.py)
2. A file-picker dialog opens asking for the patch ZIP
3. Reads patch_manifest.json from the ZIP
4. For each file in the manifest:
   - Skips protected paths (database, models, config, recordings, logs)
   - Backs up the original file to runtime/backup_<timestamp>/
   - Replaces with the patched version
   - Verifies SHA-256 after copy
5. Shows a result summary

This tool is COMPLETELY OFFLINE.
No internet. No package manager. No automatic downloads.
Only stdlib + tkinter (bundled with Python/PyInstaller).

Build with:
    pyinstaller tools/updater.spec

The resulting Application/updater.exe is standalone.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tkinter as tk
import zipfile
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List

# -- Protected paths: NEVER overwrite these on the client machine -------------
PROTECTED_PREFIXES: List[str] = [
    "runtime/data/",
    "runtime/uploads/",
    "runtime/models/",
    "runtime/nlp-engine/",
    "runtime/backup_",
    "backend/.env",
    "logs/",
    "runtime/_hf_home/",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_protected(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/").lstrip("/")
    for prefix in PROTECTED_PREFIXES:
        if norm == prefix.rstrip("/") or norm.startswith(prefix):
            return True
    return False


def _find_app_root() -> Path:
    """
    Locate the Application/ root directory.
    When running as updater.exe placed inside Application/, the root is the
    parent of the executable.
    When running as updater.py from tools/, the root is ../Application/.
    """
    if getattr(sys, "frozen", False):
        # Running as PyInstaller exe: exe lives in Application/
        return Path(sys.executable).parent.resolve()
    else:
        # Running as script: tools/ is one level below project root
        tools_dir = Path(__file__).parent.resolve()
        # Try Application/ sibling of project root
        candidate = tools_dir.parent / "Application"
        if candidate.is_dir():
            return candidate
        # Fallback: ask user
        return Path.cwd()


class UpdaterApp(tk.Tk):
    """Main updater GUI window."""

    def __init__(self):
        super().__init__()
        self.title("AI Meeting Transcriber -- Offline Updater")
        self.geometry("700x500")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        self._app_root = _find_app_root()
        self._patch_path: str = ""
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        FONT_HEADING = ("Segoe UI", 13, "bold")
        FONT_BODY    = ("Segoe UI", 10)
        FONT_MONO    = ("Consolas", 9)
        BG           = "#1e1e2e"
        FG           = "#cdd6f4"
        ACCENT       = "#89b4fa"
        BTN_BG       = "#313244"
        BTN_FG       = "#cdd6f4"

        # -- Title -------------------------------------------------------
        tk.Label(self, text="AI Meeting Transcriber", font=FONT_HEADING,
                 bg=BG, fg=ACCENT).pack(pady=(18, 0))
        tk.Label(self, text="Offline Update Installer", font=FONT_BODY,
                 bg=BG, fg=FG).pack(pady=(2, 14))

        # -- App root display --------------------------------------------
        root_frame = tk.Frame(self, bg=BG)
        root_frame.pack(fill="x", padx=20, pady=(0, 6))
        tk.Label(root_frame, text="Install location:", font=FONT_BODY,
                 bg=BG, fg=FG, width=16, anchor="w").pack(side="left")
        self._root_lbl = tk.Label(root_frame, text=str(self._app_root),
                                  font=FONT_MONO, bg=BTN_BG, fg=ACCENT,
                                  anchor="w", padx=6, relief="flat")
        self._root_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(root_frame, text="Change...", font=FONT_BODY,
                  bg=BTN_BG, fg=BTN_FG, relief="flat", cursor="hand2",
                  command=self._pick_app_root).pack(side="left", padx=(6, 0))

        # -- Patch file picker -------------------------------------------
        patch_frame = tk.Frame(self, bg=BG)
        patch_frame.pack(fill="x", padx=20, pady=6)
        tk.Label(patch_frame, text="Patch file (.zip):", font=FONT_BODY,
                 bg=BG, fg=FG, width=16, anchor="w").pack(side="left")
        self._patch_entry = tk.Entry(patch_frame, font=FONT_MONO,
                                     bg=BTN_BG, fg=ACCENT,
                                     insertbackground=ACCENT, relief="flat")
        self._patch_entry.pack(side="left", fill="x", expand=True)
        tk.Button(patch_frame, text="Browse...", font=FONT_BODY,
                  bg=BTN_BG, fg=BTN_FG, relief="flat", cursor="hand2",
                  command=self._pick_patch).pack(side="left", padx=(6, 0))

        # -- Log area ---------------------------------------------------
        tk.Label(self, text="Update log:", font=FONT_BODY,
                 bg=BG, fg=FG, anchor="w").pack(fill="x", padx=22, pady=(10, 2))
        self._log = scrolledtext.ScrolledText(
            self, font=FONT_MONO, bg="#11111b", fg="#a6e3a1",
            insertbackground="#a6e3a1", relief="flat", state="disabled", height=15
        )
        self._log.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # -- Progress bar -----------------------------------------------
        self._progress = ttk.Progressbar(self, mode="determinate")
        self._progress.pack(fill="x", padx=20, pady=(0, 4))

        # -- Buttons ----------------------------------------------------
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(pady=(0, 16))
        self._apply_btn = tk.Button(
            btn_frame, text="Apply Update", font=("Segoe UI", 11, "bold"),
            bg="#89b4fa", fg="#1e1e2e", relief="flat", padx=24, pady=8,
            cursor="hand2", command=self._apply_update
        )
        self._apply_btn.pack(side="left", padx=8)
        tk.Button(btn_frame, text="Exit", font=FONT_BODY,
                  bg=BTN_BG, fg=BTN_FG, relief="flat", padx=16, pady=8,
                  cursor="hand2", command=self.quit).pack(side="left", padx=8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log_line(self, msg: str, color: str = "#a6e3a1"):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.configure(state="disabled")
        self._log.see("end")
        self.update_idletasks()

    def _pick_patch(self):
        path = filedialog.askopenfilename(
            title="Select patch ZIP",
            filetypes=[("Patch ZIP", "*.zip"), ("All files", "*.*")],
        )
        if path:
            self._patch_entry.delete(0, "end")
            self._patch_entry.insert(0, path)

    def _pick_app_root(self):
        path = filedialog.askdirectory(title="Select Application/ install directory")
        if path:
            self._app_root = Path(path)
            self._root_lbl.configure(text=str(self._app_root))

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------
    def _apply_update(self):
        patch_path = self._patch_entry.get().strip()
        if not patch_path:
            messagebox.showerror("No patch selected", "Please select a patch ZIP file first.")
            return

        patch_file = Path(patch_path)
        if not patch_file.exists():
            messagebox.showerror("File not found", f"Patch file not found:\n{patch_file}")
            return

        if not self._app_root.is_dir():
            messagebox.showerror("Invalid directory",
                                 f"Application directory not found:\n{self._app_root}")
            return

        self._apply_btn.configure(state="disabled")
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

        try:
            self._run_update(patch_file)
        except Exception as e:
            self._log_line(f"\n[ERROR] Unexpected error: {e}", color="#f38ba8")
            messagebox.showerror("Update failed", str(e))
        finally:
            self._apply_btn.configure(state="normal")
            self._progress["value"] = 0

    def _run_update(self, patch_file: Path):
        self._log_line(f"Opening patch: {patch_file.name}")

        with zipfile.ZipFile(patch_file, "r") as zf:
            # Read manifest
            try:
                manifest_data = zf.read("patch_manifest.json")
            except KeyError:
                self._log_line("[ERROR] Invalid patch: patch_manifest.json not found.", "#f38ba8")
                messagebox.showerror("Invalid patch", "This ZIP is not a valid patch file.")
                return

            manifest: Dict = json.loads(manifest_data)
            files: List[Dict] = manifest.get("files", [])
            v_from = manifest.get("version_from", "?")
            v_to   = manifest.get("version_to", "?")

            self._log_line(f"Patch: v{v_from} -> v{v_to}  |  {len(files)} file(s)")
            self._log_line(f"Created: {manifest.get('created_at', 'unknown')}")
            self._log_line("-" * 60)

            if not files:
                self._log_line("[INFO] Patch contains no files. Nothing to do.")
                messagebox.showinfo("Nothing to update", "The patch contains no file changes.")
                return

            # Create backup directory
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self._app_root / "runtime" / f"backup_{ts}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            self._log_line(f"Backup dir: runtime/backup_{ts}/")
            self._log_line("")

            self._progress["maximum"] = len(files)
            self._progress["value"] = 0

            ok_count = 0
            skip_count = 0
            fail_count = 0

            for i, item in enumerate(files):
                rel   = item["path"]
                sha   = item["sha256"]
                arc   = f"files/{rel}"
                dest  = self._app_root / rel.replace("/", os.sep)

                self._progress["value"] = i + 1

                # Skip protected paths
                if _is_protected(rel):
                    self._log_line(f"  [SKIP]     {rel}  (protected path)")
                    skip_count += 1
                    continue

                # Backup existing file
                if dest.exists():
                    bak = backup_dir / rel.replace("/", os.sep)
                    bak.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dest, bak)

                # Extract
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    data = zf.read(arc)
                    dest.write_bytes(data)

                    # Verify hash
                    actual = _sha256(dest)
                    if actual != sha:
                        self._log_line(
                            f"  [FAIL]     {rel}  (hash mismatch after copy!)", "#f38ba8"
                        )
                        fail_count += 1
                    else:
                        status = item.get("status", "updated")
                        self._log_line(f"  [{status.upper():8s}] {rel}")
                        ok_count += 1

                except Exception as e:
                    self._log_line(f"  [ERROR]    {rel}: {e}", "#f38ba8")
                    fail_count += 1

            self._log_line("")
            self._log_line("=" * 60)
            self._log_line(f"Done.  {ok_count} updated  |  {skip_count} skipped  |  {fail_count} failed")

            if fail_count == 0:
                self._log_line("Update applied successfully. Please restart the application.", "#89b4fa")
                messagebox.showinfo(
                    "Update complete",
                    f"Update applied successfully!\n\n"
                    f"{ok_count} file(s) updated.\n\n"
                    "Please restart the AI Meeting Transcriber application.",
                )
            else:
                self._log_line(f"WARNING: {fail_count} file(s) failed to update.", "#fab387")
                messagebox.showwarning(
                    "Partial update",
                    f"{ok_count} file(s) updated, but {fail_count} file(s) failed.\n"
                    f"Check the log for details.\n"
                    f"Backup saved to: runtime/backup_{ts}/",
                )


def main():
    app = UpdaterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
