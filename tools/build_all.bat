@echo off
REM ============================================================
REM  AI Meeting Transcriber — Full Build Script
REM  Run from project root: tools\build_all.bat
REM
REM  IMPORTANT: AI models are NOT bundled in the installer.
REM  They are distributed separately:
REM    - Encrypted models (.dat)  → Application\runtime\models\
REM    - Qwen3 nlp-engine folder  → Application\runtime\nlp-engine\
REM  These must be copied manually AFTER build before creating installer.
REM ============================================================
setlocal enabledelayedexpansion

set PROJECT_ROOT=%~dp0..
set TOOLS_DIR=%~dp0
set APP_DIR=%PROJECT_ROOT%\Application

echo.
echo ============================================================
echo  AI Meeting Transcriber — Build Pipeline
echo ============================================================
echo.

REM ── Check prerequisites ──────────────────────────────────────
echo [1/7] Checking prerequisites...

where python >nul 2>&1 || (
    echo ERROR: Python not found in PATH
    exit /b 1
)

where node >nul 2>&1 || (
    echo ERROR: Node.js not found in PATH
    exit /b 1
)

where npm >nul 2>&1 || (
    echo ERROR: npm not found in PATH
    exit /b 1
)

echo      Python, Node.js, npm: OK

REM ── Create Application directory structure ──────────────────
echo [2/7] Creating output directory structure...
if not exist "%APP_DIR%" mkdir "%APP_DIR%"
if not exist "%APP_DIR%\runtime" mkdir "%APP_DIR%\runtime"
if not exist "%APP_DIR%\runtime\models" mkdir "%APP_DIR%\runtime\models"
if not exist "%APP_DIR%\runtime\nlp-engine" mkdir "%APP_DIR%\runtime\nlp-engine"
if not exist "%APP_DIR%\runtime\data" mkdir "%APP_DIR%\runtime\data"
if not exist "%APP_DIR%\runtime\uploads" mkdir "%APP_DIR%\runtime\uploads"
if not exist "%APP_DIR%\assets" mkdir "%APP_DIR%\assets"
echo      Directory structure created.
echo.
echo      NOTE: Models are NOT bundled in the installer.
echo      After build, place the following manually:
echo        Application\runtime\models\     ^<-- encrypted .dat model files
echo        Application\runtime\nlp-engine\ ^<-- Qwen3-4B plain model folder
echo.

REM ── Install Python dependencies ──────────────────────────────
echo [3/7] Installing Python build dependencies...
pip install pyinstaller cryptography httpx --quiet
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install Python build dependencies
    exit /b 1
)
echo      Python build deps: OK

REM ── Check encrypted models (dry-run only) ───────────────────
echo [4/7] Checking AI model availability...
python "%TOOLS_DIR%\download_all_models.py" --dry-run
echo.
echo      To package models into Application\runtime\models\:
echo        python tools\download_all_models.py --hf-token YOUR_TOKEN
echo        python tools\encrypt_models.py --output Application\runtime\models
echo.
echo      For the Qwen3 nlp-engine model (plain, no encryption):
echo        Copy the Qwen3-4B folder to: Application\runtime\nlp-engine\
echo.

REM ── Build React frontend ─────────────────────────────────────
echo [5/7] Building React frontend...
cd /d "%PROJECT_ROOT%\frontend"
call npm install --legacy-peer-deps --silent
if %ERRORLEVEL% neq 0 (
    echo ERROR: npm install failed
    exit /b 1
)
call npm run build
if %ERRORLEVEL% neq 0 (
    echo ERROR: Frontend build failed
    exit /b 1
)
echo      Frontend built successfully.

REM ── Package Electron ─────────────────────────────────────────
echo      Packaging Electron frontend...
cd /d "%PROJECT_ROOT%\frontend-electron"
call npm install --silent
if %ERRORLEVEL% neq 0 (
    echo ERROR: Electron npm install failed
    exit /b 1
)
if exist dist rmdir /S /Q dist
mkdir dist
xcopy /Y /E /Q "%PROJECT_ROOT%\frontend\dist\*" dist\ >nul
call npm run dist:dir
if %ERRORLEVEL% neq 0 (
    echo WARNING: Electron packaging failed. Check output above.
) else (
    echo      Electron packaged successfully.
    echo      Output: Application\frontend\win-unpacked\
)

REM ── Build PyInstaller backend ────────────────────────────────
echo [6/7] Building backend executable (PyInstaller)...
cd /d "%PROJECT_ROOT%"

set PYINSTALLER_CMD=pyinstaller
if exist "%PROJECT_ROOT%\backend\venv\Scripts\pyinstaller.exe" (
    set PYINSTALLER_CMD="%PROJECT_ROOT%\backend\venv\Scripts\pyinstaller.exe"
    echo      Using virtual environment PyInstaller
)

%PYINSTALLER_CMD% "%TOOLS_DIR%\backend.spec" --distpath "%APP_DIR%" --workpath "build\backend_work" --noconfirm
if %ERRORLEVEL% neq 0 (
    echo ERROR: Backend build failed
    exit /b 1
)
echo      Backend built: Application\backend\backend.exe

REM ── Bundle FFmpeg if available ───────────────────────────────
where ffmpeg.exe >temp_ffmpeg.txt 2>nul
if %ERRORLEVEL% equ 0 (
    set /p FFMPEG_PATH=<temp_ffmpeg.txt
    del temp_ffmpeg.txt
    copy /Y "!FFMPEG_PATH!" "%APP_DIR%\backend\ffmpeg.exe" >nul
    echo      Bundled ffmpeg.exe
)
where ffprobe.exe >temp_ffprobe.txt 2>nul
if %ERRORLEVEL% equ 0 (
    set /p FFPROBE_PATH=<temp_ffprobe.txt
    del temp_ffprobe.txt
    copy /Y "!FFPROBE_PATH!" "%APP_DIR%\backend\ffprobe.exe" >nul
    echo      Bundled ffprobe.exe
)

REM ── Build launcher ───────────────────────────────────────────
echo      Building launcher executable...
%PYINSTALLER_CMD% "%TOOLS_DIR%\launcher.spec" --distpath "%APP_DIR%" --workpath "build\launcher_work" --noconfirm
if %ERRORLEVEL% neq 0 (
    echo ERROR: Launcher build failed
    exit /b 1
)
echo      Launcher built: Application\launcher.exe

REM ── Copy default .env config ─────────────────────────────────
copy /Y "%PROJECT_ROOT%\backend\.env.example" "%APP_DIR%\backend\.env" >nul 2>&1
echo      Copied .env.example → Application\backend\.env

REM ── Copy checkpoints if available ─────────────────────────────
if exist "%PROJECT_ROOT%\backend\checkpoints" (
    xcopy /Y /E /I /Q "%PROJECT_ROOT%\backend\checkpoints\*" "%APP_DIR%\backend\checkpoints\" >nul 2>&1
    echo      Copied checkpoints → Application\backend\checkpoints\
)

REM ── Copy assets ──────────────────────────────────────────────
xcopy /Y /E /Q "%PROJECT_ROOT%\assets\*" "%APP_DIR%\assets\" >nul 2>&1

REM ── Verify models before installer ───────────────────────────
echo.
echo [7/7] Pre-installer model verification...
echo.
if exist "%APP_DIR%\runtime\models\speech_engine.dat" (
    echo      [OK] speech_engine.dat
) else (
    echo      [MISSING] speech_engine.dat
)
if exist "%APP_DIR%\runtime\models\align_engine.dat" (
    echo      [OK] align_engine.dat
) else (
    echo      [MISSING] align_engine.dat
)
if exist "%APP_DIR%\runtime\models\voice_context.dat" (
    echo      [OK] voice_context.dat
) else (
    echo      [MISSING] voice_context.dat
)
if exist "%APP_DIR%\runtime\nlp-engine\" (
    echo      [OK] nlp-engine/ (Qwen3)
) else (
    echo      [MISSING] nlp-engine/ — copy Qwen3-4B folder here before running installer
)
echo.

REM ── Run Inno Setup installer ─────────────────────────────────
set ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe
if not exist "!ISCC_PATH!" (
    echo WARNING: Inno Setup not found at !ISCC_PATH!
    echo          Download from https://jrsoftware.org/isdl.php
    echo          Skipping installer creation.
    echo.
    echo          To build manually after installing Inno Setup:
    echo            "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\setup.iss
) else (
    echo Building installer (Inno Setup)...
    "!ISCC_PATH!" "%PROJECT_ROOT%\installer\setup.iss"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Inno Setup failed
        exit /b 1
    )
    echo      Setup.exe created in installer\dist\
)

echo.
echo ============================================================
echo  BUILD COMPLETE
echo  Application\     — packaged application (place models here)
echo  installer\dist\  — Setup.exe installer (excludes models)
echo.
echo  REMINDER: Before distributing the installer, ensure:
echo    1. Application\runtime\models\*.dat  are populated
echo    2. Application\runtime\nlp-engine\   is populated
echo ============================================================
echo.

cd /d "%PROJECT_ROOT%"
endlocal
