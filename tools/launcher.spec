from pathlib import Path
# SPECPATH is always the directory containing this .spec file (tools/).
# The project root is one level up.
ROOT = Path(SPECPATH).parent
LAUNCHER_DIR = ROOT / "launcher"
ASSETS_DIR = ROOT / "assets"

block_cipher = None

a = Analysis(
    [str(LAUNCHER_DIR / "launcher.py")],
    pathex=[str(LAUNCHER_DIR)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "tkinter",
        "tkinter.font",
        "tkinter.ttk",
        # splash.py and license_manager.py are in the same directory as launcher.py
        # pathex ensures they are found as top-level modules
        "splash",
        "license_manager",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name="launcher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,           # No console window — splash is the UI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ASSETS_DIR / "icon.ico"),
)
