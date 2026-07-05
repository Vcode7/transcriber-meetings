# -*- mode: python ; coding: utf-8 -*-
"""
backend_launcher.spec ? PyInstaller spec for the tiny backend launcher.

This produces backend.exe ? a small (~10 MB) executable that:
  1. Finds the VoiceSum runtime Python (embeddable Python 3.12)
  2. Runs app.pyz via that Python

NO ML dependencies are included. Compile time is seconds, not minutes.

Build:
    pyinstaller tools/backend_launcher.spec --distpath Application/backend_launcher_dist --workpath build/bl_work --noconfirm
Then copy Application/backend_launcher_dist/backend/backend.exe -> Application/backend/backend.exe
"""
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH).parent          # project root
TOOLS_DIR = ROOT / "tools"
ASSETS_DIR = ROOT / "assets"

a = Analysis(
    [str(TOOLS_DIR / "backend_launcher.py")],
    pathex=[str(TOOLS_DIR)],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Only stdlib ? no ML packages
        "subprocess",
        "logging",
        "os",
        "sys",
        "pathlib",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Explicitly exclude all heavy ML packages so PyInstaller
        # does not accidentally pull them in from the active venv.
        "torch", "torchaudio", "torchvision",
        "transformers", "whisperx", "faster_whisper",
        "pyannote", "speechbrain", "resemblyzer",
        "librosa", "soundfile", "scipy", "numpy",
        "sklearn", "pandas", "matplotlib",
        "bitsandbytes", "accelerate", "ctranslate2",
        "onnxruntime", "tokenizers", "sentencepiece",
        "huggingface_hub",
        # Dev tools
        "IPython", "jupyter", "pytest", "sphinx",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="backend",                  # Output: backend.exe
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                   # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ASSETS_DIR / "icon.ico") if (ASSETS_DIR / "icon.ico").exists() else None,
)
