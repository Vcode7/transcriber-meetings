@echo off
REM ============================================================
REM  VoiceSum ? Runtime Package Builder
REM
REM  Builds the embeddable Python 3.12 + ML dependencies package.
REM  Run this ONCE whenever ML dependencies change (torch, whisperx, etc.)
REM  Result: tools\runtime_dist\VoiceSum-Runtime-2.0.exe
REM
REM  Usage:
REM    tools\build_runtime.bat              -- full build
REM    tools\build_runtime.bat --deps-only  -- reinstall pip deps only (skip py download)
REM    tools\build_runtime.bat --iss-only   -- rebuild installer only (skip everything else)
REM
REM  Requirements on build machine:
REM    - Internet access (downloads Python embeddable zip + torch from PyPI)
REM    - Inno Setup 6 installed at default path
REM    - ffmpeg.exe / ffprobe.exe on PATH (bundled into runtime)
REM    - At least 10 GB free disk in build\runtime-pkg\
REM
REM  Output layout in build\runtime-pkg\:
REM    python\
REM        python.exe
REM        python312.dll
REM        ...
REM        Lib\
REM            site-packages\
REM                torch\
REM                torchaudio\
REM                whisperx\
REM                pyannote\
REM                ... (all ML deps)
REM    ffmpeg.exe
REM    ffprobe.exe
REM    runtime-version.txt   ("2.0")
REM ============================================================
setlocal enabledelayedexpansion

REM ?? Configuration ????????????????????????????????????????????
set RUNTIME_VERSION=2.0
set PY_VERSION=3.12.3
set PY_EMBED_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-embed-amd64.zip
set GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py

set PROJECT_ROOT=%~dp0..
set TOOLS_DIR=%~dp0
set BUILD_DIR=%PROJECT_ROOT%\build\runtime-pkg
set PYTHON_DIR=%BUILD_DIR%\python
set SITE_PACKAGES=%PYTHON_DIR%\Lib\site-packages
set ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe

REM ?? Parse arguments ??????????????????????????????????????????
set DEPS_ONLY=0
set ISS_ONLY=0
if "%1"=="--deps-only" set DEPS_ONLY=1
if "%1"=="--iss-only"  set ISS_ONLY=1

echo.
echo ============================================================
echo  VoiceSum Runtime Builder  v%RUNTIME_VERSION%
echo  Python %PY_VERSION%  +  PyTorch 2.8.0+cu128  +  CUDA 12.8
echo ============================================================
echo.

REM ?? Check prerequisites ??????????????????????????????????????
where python >nul 2>&1 || ( echo ERROR: Python not found in PATH. & exit /b 1 )
where curl   >nul 2>&1 || ( echo ERROR: curl not found in PATH. It ships with Windows 10 1803+. & exit /b 1 )
echo   Prerequisites: OK

if "%ISS_ONLY%"=="1" goto BUILD_INSTALLER

REM ?? Create build directory ????????????????????????????????????
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

if "%DEPS_ONLY%"=="1" goto INSTALL_DEPS

REM ?? Download embeddable Python ????????????????????????????????
echo.
echo [R1] Downloading Python %PY_VERSION% embeddable zip...
set PY_ZIP=%BUILD_DIR%\python-embed.zip
if not exist "%PY_ZIP%" (
    curl -L -o "%PY_ZIP%" "%PY_EMBED_URL%"
    if %ERRORLEVEL% neq 0 ( echo ERROR: Download failed. & exit /b 1 )
) else (
    echo   [SKIP] Already downloaded: %PY_ZIP%
)

REM ?? Extract embeddable Python ?????????????????????????????????
echo.
echo [R2] Extracting embeddable Python...
if exist "%PYTHON_DIR%" rmdir /S /Q "%PYTHON_DIR%"
mkdir "%PYTHON_DIR%"
powershell -NoProfile -Command "Expand-Archive -Path '%PY_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
if %ERRORLEVEL% neq 0 ( echo ERROR: Extraction failed. & exit /b 1 )
echo   Extracted to: %PYTHON_DIR%

REM ?? Enable site-packages in embeddable Python ?????????????????
REM The ._pth file in embeddable Python disables site.py by default.
REM We must uncomment "import site" and add Lib/site-packages to the path.
echo.
echo [R3] Enabling site-packages in embeddable Python...

REM Find the ._pth file (e.g. python312._pth)
set PTH_FILE=
for %%F in ("%PYTHON_DIR%\python*._pth") do set PTH_FILE=%%F

if "!PTH_FILE!"=="" (
    echo   ERROR: Could not find ._pth file in %PYTHON_DIR%
    exit /b 1
)
echo   Found: !PTH_FILE!

REM Create a new ._pth with site-packages path and "import site" uncommented
(
    echo .
    echo Lib\site-packages
    echo import site
) > "!PTH_FILE!"

REM Create the Lib\site-packages directory
if not exist "%SITE_PACKAGES%" mkdir "%SITE_PACKAGES%"
echo   site-packages enabled.

REM ?? Install pip into embeddable Python ???????????????????????
echo.
echo [R4] Installing pip into embeddable Python...
set GET_PIP_FILE=%BUILD_DIR%\get-pip.py
if not exist "%GET_PIP_FILE%" (
    curl -L -o "%GET_PIP_FILE%" "%GET_PIP_URL%"
    if %ERRORLEVEL% neq 0 ( echo ERROR: get-pip.py download failed. & exit /b 1 )
)
"%PYTHON_DIR%\python.exe" "%GET_PIP_FILE%" --no-warn-script-location
if %ERRORLEVEL% neq 0 ( echo ERROR: pip install failed. & exit /b 1 )
echo   pip installed.

:INSTALL_DEPS
REM ?? Install ML dependencies ???????????????????????????????????
echo.
echo [R5] Installing ML dependencies (this takes 20-60 minutes)...
echo   Target: %SITE_PACKAGES%
echo.
echo   Step 5a: PyTorch 2.8.0 + CUDA 12.8...
"%PYTHON_DIR%\python.exe" -m pip install ^
    torch==2.8.0 torchaudio==2.8.0 ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    --target "%SITE_PACKAGES%" ^
    --no-warn-script-location
if %ERRORLEVEL% neq 0 ( echo ERROR: PyTorch install failed. & exit /b 1 )

echo.
echo   Step 5b: Core ML libraries...
"%PYTHON_DIR%\python.exe" -m pip install ^
    faster-whisper ^
    whisperx ^
    pyannote.audio ^
    resemblyzer ^
    speechbrain ^
    --target "%SITE_PACKAGES%" ^
    --no-warn-script-location
if %ERRORLEVEL% neq 0 ( echo ERROR: Core ML deps install failed. & exit /b 1 )

echo.
echo   Step 5c: NLP + transformers...
"%PYTHON_DIR%\python.exe" -m pip install ^
    transformers ^
    accelerate ^
    bitsandbytes ^
    sentencepiece ^
    optimum ^
    --target "%SITE_PACKAGES%" ^
    --no-warn-script-location
if %ERRORLEVEL% neq 0 ( echo ERROR: Transformers install failed. & exit /b 1 )

echo.
echo   Step 5d: Backend / API dependencies...
"%PYTHON_DIR%\python.exe" -m pip install ^
    fastapi uvicorn[standard] ^
    sqlalchemy aiosqlite ^
    python-jose[cryptography] passlib[bcrypt] ^
    python-multipart aiofiles ^
    httpx pydantic pydantic-settings python-dotenv ^
    librosa soundfile ^
    scipy numpy pandas scikit-learn ^
    reportlab python-docx pymupdf ^
    cryptography ^
    psutil ^
    --target "%SITE_PACKAGES%" ^
    --no-warn-script-location
if %ERRORLEVEL% neq 0 ( echo ERROR: Backend deps install failed. & exit /b 1 )

echo.
echo   [R5 DONE] All ML dependencies installed.

REM ?? Bundle ffmpeg / ffprobe ???????????????????????????????????
echo.
echo [R6] Bundling ffmpeg and ffprobe...
where ffmpeg.exe >tmp_ff.txt 2>nul
if %ERRORLEVEL% equ 0 (
    set /p FFP=<tmp_ff.txt
    copy /Y "!FFP!" "%BUILD_DIR%\ffmpeg.exe" >nul
    echo   Copied ffmpeg.exe from !FFP!
) else (
    echo   WARNING: ffmpeg.exe not found on PATH. Add it manually to %BUILD_DIR%\
)
del tmp_ff.txt >nul 2>&1

where ffprobe.exe >tmp_ffp.txt 2>nul
if %ERRORLEVEL% equ 0 (
    set /p FFPB=<tmp_ffp.txt
    copy /Y "!FFPB!" "%BUILD_DIR%\ffprobe.exe" >nul
    echo   Copied ffprobe.exe from !FFPB!
) else (
    echo   WARNING: ffprobe.exe not found on PATH. Add it manually to %BUILD_DIR%\
)
del tmp_ffp.txt >nul 2>&1

REM ?? Write version stamp ???????????????????????????????????????
echo.
echo [R7] Stamping runtime-version.txt = %RUNTIME_VERSION%...
echo %RUNTIME_VERSION%> "%BUILD_DIR%\runtime-version.txt"
echo   Written: %BUILD_DIR%\runtime-version.txt

:BUILD_INSTALLER
REM ?? Build Inno Setup runtime installer ???????????????????????
echo.
echo [R8] Building VoiceSum-Runtime-%RUNTIME_VERSION%.exe (Inno Setup)...
if not exist "%ISCC_PATH%" (
    echo   WARNING: Inno Setup not found at %ISCC_PATH%
    echo   Download: https://jrsoftware.org/isdl.php
    echo   Run manually: "%ISCC_PATH%" "%TOOLS_DIR%runtime.iss"
    goto DONE
)

REM Inno Setup runs relative to its own .iss file location
cd /d "%TOOLS_DIR%"
"%ISCC_PATH%" "runtime.iss"
if %ERRORLEVEL% neq 0 ( echo ERROR: Inno Setup failed. & exit /b 1 )
cd /d "%PROJECT_ROOT%"

echo.
echo   Output: tools\runtime_dist\VoiceSum-Runtime-%RUNTIME_VERSION%.exe

:DONE
echo.
echo ============================================================
echo  RUNTIME BUILD COMPLETE
echo.
echo  Deliverable:  tools\runtime_dist\VoiceSum-Runtime-%RUNTIME_VERSION%.exe
echo  Install on client: run the exe (installs to %%ProgramData%%\VoiceSum\runtime\)
echo.
echo  Registry keys written on client:
echo    HKLM\SOFTWARE\VoiceSum\Runtime\Version     = "%RUNTIME_VERSION%"
echo    HKLM\SOFTWARE\VoiceSum\Runtime\InstallPath = "C:\ProgramData\VoiceSum\runtime"
echo ============================================================
echo.

cd /d "%PROJECT_ROOT%"
endlocal
