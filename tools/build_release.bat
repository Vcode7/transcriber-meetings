@echo off
setlocal enabledelayedexpansion

set PROJECT_ROOT=%~dp0..
set TOOLS_DIR=%~dp0
set NEW_APP_DIR=%PROJECT_ROOT%\Application
set OLD_INSTALLER=%PROJECT_ROOT%\installer\dist\Setup_AIMeetingTranscriber_v1.0.0.exe
set OLD_APP_DIR=%PROJECT_ROOT%\build\old_v1.0.0_temp

echo ============================================================
echo  AI Meeting Transcriber -- Automated Release Builder
echo ============================================================
echo.

REM ── 1. Silently extract old v1.0.0 installer ─────────────────
echo [1/3] Silently installing old v1.0.0 to temp folder...
if not exist "!OLD_INSTALLER!" (
    echo ERROR: Old v1.0.0 installer not found at: !OLD_INSTALLER!
    exit /b 1
)

if exist "%OLD_APP_DIR%" rmdir /S /Q "%OLD_APP_DIR%"
mkdir "%OLD_APP_DIR%"

echo Running silent installation...
start /wait "" "%OLD_INSTALLER%" /VERYSILENT /SUPPRESSMSGBOXES /DIR="%OLD_APP_DIR%"

if !ERRORLEVEL! neq 0 (
    echo ERROR: Silent installation of v1.0.0 failed.
    exit /b 1
)
echo [OK] Old v1.0.0 extracted to temporary directory.
echo.

REM ── 2. Compile Inno Setup installer ──────────────────────────
echo [2/3] Compiling Setup Installer (Inno Setup)...
set ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe
if exist "!ISCC_PATH!" (
    "!ISCC_PATH!" "!PROJECT_ROOT!\installer\setup.iss"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Inno Setup compilation failed.
        exit /b 1
    )
    echo [OK] Setup_v1.0.1.exe created in installer\dist\
) else (
    echo WARNING: Inno Setup compiler not found at: !ISCC_PATH!
    echo          Skipping installer compilation.
)
echo.

REM ── 3. Create Incremental Update Patch ───────────────────────
echo [3/3] Creating Incremental Update Patch...
set PATCH_OUT_ZIP=%PROJECT_ROOT%\patch_v1.0.0_to_v1.0.1.zip
echo Creating patch from "%OLD_APP_DIR%" to "%NEW_APP_DIR%"...
"%PROJECT_ROOT%\backend\venv\Scripts\python.exe" "%TOOLS_DIR%\create_patch.py" --old "%OLD_APP_DIR%" --new "%NEW_APP_DIR%" --out "%PATCH_OUT_ZIP%" --version-from 1.0.0 --version-to 1.0.1

if !ERRORLEVEL! neq 0 (
    echo ERROR: Patch creation failed.
    exit /b 1
)
echo [OK] Patch ZIP created.
echo.

REM ── 4. Clean up temporary old directory ──────────────────────
echo Cleaning up temporary folders...
if exist "%OLD_APP_DIR%" rmdir /S /Q "%OLD_APP_DIR%"
echo.

echo ============================================================
echo  RELEASE PACKAGING COMPLETE
echo  1. Setup Installer : installer\dist\Setup_AIMeetingTranscriber_v1.0.1.exe
echo  2. Update Patch    : %PATCH_OUT_ZIP%
echo ============================================================
pause
