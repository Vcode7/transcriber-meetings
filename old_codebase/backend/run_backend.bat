@echo off
setlocal enabledelayedexpansion
title VoiceSum Backend (Offline)

echo ========================================================
echo Starting VoiceSum Backend (Offline Mode)
echo ========================================================
echo.

cd /d "%~dp0"

:: 1. Locate Virtual Environment
set "VENV_DIR="
if exist "..\venv\Scripts\python.exe" (
    set "VENV_DIR=..\venv"
) else if exist ".\venv\Scripts\python.exe" (
    set "VENV_DIR=.\venv"
)

if "%VENV_DIR%"=="" (
    echo [ERROR] Could not find virtual environment folder 'venv'.
    echo Please make sure backend_venv.zip was extracted so that 'venv' exists
    echo either inside 'backend\venv' or as sibling '..\venv'.
    echo.
    pause
    exit /b 1
)

:: 2. Check for license.key
if not exist "license.key" (
    if exist "..\license.key" (
        copy "..\license.key" "license.key" >nul 2>&1
    )
)

if not exist "license.key" (
    echo [WARNING] 'license.key' was not found in the backend directory.
    echo Please ensure license.key is placed next to backend files or in the parent folder.
    echo.
)

:: 3. Check / Auto-repair pyvenv.cfg if host python path differs
if exist "%VENV_DIR%\pyvenv.cfg" (
    "%VENV_DIR%\Scripts\python.exe" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo [INFO] Detected host Python environment change. Auto-configuring pyvenv.cfg...
        for /f "delims=" %%I in ('where python 2^>nul') do (
            set "HOST_PY=%%~dpI"
            goto :found_host_py
        )
        :found_host_py
        if defined HOST_PY (
            :: Strip trailing backslash
            if "!HOST_PY:~-1!"=="\" set "HOST_PY=!HOST_PY:~0,-1!"
            echo [INFO] Binding venv to host Python at: !HOST_PY!
            (
                echo home = !HOST_PY!
                echo include-system-site-packages = false
                echo version = 3.12.3
            ) > "%VENV_DIR%\pyvenv.cfg"
        )
    )
)

:: 4. Start VoiceSum Backend Server
echo [INFO] Starting Uvicorn on http://127.0.0.1:8000 ...
"%VENV_DIR%\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000

if errorlevel 1 (
    echo.
    echo [ERROR] Backend exited with an error code.
    pause
)
