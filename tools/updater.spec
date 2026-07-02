# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for updater.exe
Build: pyinstaller tools/updater.spec --distpath Application/ --workpath build/updater_work

The resulting Application/updater.exe is a single standalone executable.
Place it in the Application/ root so it can locate the install directory
automatically (it resolves the app root as its own parent directory).
"""
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH).parent   # project root

a = Analysis(
    [str(ROOT / "tools" / "updater.py")],
    pathex=[str(ROOT / "tools")],
    binaries=[],
    datas=[],
    hiddenimports=[
        "tkinter",
        "tkinter.filedialog",
        "tkinter.messagebox",
        "tkinter.scrolledtext",
        "tkinter.ttk",
        "hashlib",
        "zipfile",
        "json",
        "shutil",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "numpy",
        "scipy",
        "torch",
        "transformers",
        "PIL",
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
    name="updater",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    # onefile=True -- single .exe, no folder
    console=False,           # windowed app (tkinter GUI)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,               # add icon path here if you have one
)
