"""
Splash Screen — Phase 10
Professional loading splash window shown while backend starts.
Uses tkinter (built into Python standard library — no extra deps).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
import threading
import time
from typing import Optional, Callable

# ── Color palette ─────────────────────────────────────────────
BG_COLOR = "#0b0d17"          # near-black navy
CARD_COLOR = "#12151f"        # slightly lighter card
ACCENT = "#6366f1"            # indigo accent
ACCENT_GLOW = "#818cf8"       # lighter indigo
TEXT_WHITE = "#f0f2ff"        # off-white
TEXT_MUTED = "#8892b0"        # muted text
SUCCESS = "#22d3a8"           # teal-green
ERROR_COLOR = "#ff6b6b"       # soft red
PROGRESS_BG = "#1e2235"       # progress track
PROGRESS_FG = "#6366f1"       # progress fill


class SplashScreen:
    """
    Frameless splash window with animated progress steps.
    Runs on the main thread; update_step() is thread-safe.
    """

    STEPS = [
        ("🔑", "Verifying License…"),
        ("🧠", "Initializing AI Engine…"),
        ("🎙️", "Loading Speech Processing…"),
        ("🎵", "Preparing Voice Models…"),
        ("🚀", "Starting Services…"),
        ("✅", "Ready!"),
    ]

    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # hide until ready

        self._current_step = 0
        self._progress = 0.0
        self._status_text = "Starting…"
        self._error_text: Optional[str] = None
        self._done = False
        self._closed = False

        self._build_ui()

    def _build_ui(self):
        root = self.root
        root.title("AI Meeting Transcriber")
        root.overrideredirect(True)          # frameless window
        root.attributes("-topmost", True)    # always on top during load
        root.configure(bg=BG_COLOR)
        root.resizable(False, False)

        W, H = 480, 340
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{W}x{H}+{(sw - W)//2}+{(sh - H)//2}")

        # ── Outer frame ──────────────────────────────────────
        outer = tk.Frame(root, bg=BG_COLOR, padx=2, pady=2)
        outer.pack(fill="both", expand=True)

        # Gradient-like border effect
        border = tk.Frame(outer, bg=ACCENT, padx=1, pady=1)
        border.pack(fill="both", expand=True)

        inner = tk.Frame(border, bg=BG_COLOR)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        # ── Logo area ─────────────────────────────────────────
        logo_frame = tk.Frame(inner, bg=BG_COLOR, pady=24)
        logo_frame.pack(fill="x")

        # App icon (Unicode emoji as substitute)
        icon_lbl = tk.Label(
            logo_frame, text="🎙️",
            bg=BG_COLOR, fg=TEXT_WHITE,
            font=("Segoe UI Emoji", 36),
        )
        icon_lbl.pack()

        title_lbl = tk.Label(
            logo_frame, text="AI Meeting Transcriber",
            bg=BG_COLOR, fg=TEXT_WHITE,
            font=("Segoe UI", 18, "bold"),
        )
        title_lbl.pack(pady=(6, 0))

        subtitle_lbl = tk.Label(
            logo_frame, text="Secure Offline Intelligence",
            bg=BG_COLOR, fg=TEXT_MUTED,
            font=("Segoe UI", 10),
        )
        subtitle_lbl.pack()

        # ── Separator ─────────────────────────────────────────
        sep = tk.Frame(inner, bg=ACCENT, height=1)
        sep.pack(fill="x", padx=32, pady=(8, 16))

        # ── Status area ───────────────────────────────────────
        status_frame = tk.Frame(inner, bg=BG_COLOR)
        status_frame.pack(fill="x", padx=32)

        self._step_icon = tk.Label(
            status_frame, text="⏳",
            bg=BG_COLOR, fg=ACCENT_GLOW,
            font=("Segoe UI Emoji", 14),
            width=2,
        )
        self._step_icon.pack(side="left")

        self._status_lbl = tk.Label(
            status_frame, text="Starting…",
            bg=BG_COLOR, fg=TEXT_WHITE,
            font=("Segoe UI", 11),
            anchor="w",
        )
        self._status_lbl.pack(side="left", padx=(8, 0))

        # ── Progress bar ──────────────────────────────────────
        prog_frame = tk.Frame(inner, bg=BG_COLOR, pady=12)
        prog_frame.pack(fill="x", padx=32)

        self._prog_canvas = tk.Canvas(
            prog_frame, bg=PROGRESS_BG, height=6,
            highlightthickness=0, bd=0,
        )
        self._prog_canvas.pack(fill="x")
        self._prog_bar = self._prog_canvas.create_rectangle(
            0, 0, 0, 6, fill=PROGRESS_FG, outline="",
        )

        # ── Bottom label ──────────────────────────────────────
        self._bottom_lbl = tk.Label(
            inner, text="Please wait…",
            bg=BG_COLOR, fg=TEXT_MUTED,
            font=("Segoe UI", 9),
        )
        self._bottom_lbl.pack(pady=(0, 20))

        # ── Error display (hidden initially) ─────────────────
        self._error_lbl = tk.Label(
            inner, text="",
            bg=BG_COLOR, fg=ERROR_COLOR,
            font=("Segoe UI", 9),
            wraplength=400, justify="center",
        )
        self._error_lbl.pack(pady=(0, 4))

        # Start progress animation
        self._animate_progress()

        root.update_idletasks()
        root.deiconify()  # show the window

    def _animate_progress(self):
        """Smooth progress bar animation — runs every 50ms."""
        if self._closed:
            return
        try:
            canvas = self._prog_canvas
            w = canvas.winfo_width()
            if w > 1:
                fill_w = int(w * min(self._progress, 1.0))
                canvas.coords(self._prog_bar, 0, 0, fill_w, 6)
            self.root.after(50, self._animate_progress)
        except tk.TclError:
            pass  # window destroyed

    def update_step(self, step_index: int, message: Optional[str] = None):
        """Thread-safe: move to the given step."""
        self._current_step = step_index
        icon, default_msg = self.STEPS[step_index] if step_index < len(self.STEPS) else ("⏳", "Working…")
        text = message or default_msg
        progress = (step_index + 1) / len(self.STEPS)

        def _do():
            try:
                self._status_lbl.config(text=text)
                self._step_icon.config(text=icon)
                self._progress = progress
                if step_index == len(self.STEPS) - 1:
                    self._status_lbl.config(fg=SUCCESS)
                    self._step_icon.config(fg=SUCCESS)
                    self._bottom_lbl.config(text="Launching…")
            except tk.TclError:
                pass

        self.root.after(0, _do)

    def show_error(self, title: str, message: str):
        """Show an error message in the splash (non-fatal) or close with error."""
        def _do():
            try:
                self._status_lbl.config(text=title, fg=ERROR_COLOR)
                self._step_icon.config(text="❌", fg=ERROR_COLOR)
                self._bottom_lbl.config(text="")
                self._error_lbl.config(text=message)
                self._progress = 0
            except tk.TclError:
                pass
        self.root.after(0, _do)

    def close(self):
        """Close the splash screen."""
        self._closed = True
        try:
            self.root.after(300, self.root.destroy)
        except tk.TclError:
            pass

    def close_with_error(self, title: str, message: str):
        """Show error, then add Exit button."""
        self.show_error(title, message)

        def _add_exit():
            try:
                btn = tk.Button(
                    self.root, text="Exit",
                    bg=ACCENT, fg=TEXT_WHITE,
                    font=("Segoe UI", 10, "bold"),
                    relief="flat", padx=20, pady=6,
                    cursor="hand2",
                    command=self.root.destroy,
                )
                btn.pack(pady=10)
            except tk.TclError:
                pass

        self.root.after(200, _add_exit)

    def run_mainloop(self):
        """Run tk mainloop (must be called from main thread)."""
        self.root.mainloop()
