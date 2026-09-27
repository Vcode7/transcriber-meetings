@echo off
setlocal
cd /d "%~dp0.."
echo ========================================================
echo  VoiceSum Incremental Source Code Patch Generator
echo ========================================================
echo.
python tools\create_code_patch.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Patch generation failed.
    pause
    exit /b 1
)
echo.
if "%1"=="" (
    pause
)
