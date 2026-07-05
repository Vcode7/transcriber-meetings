@echo off
REM ============================================================
REM  AI Meeting Transcriber - Step-by-Step App Build Script
REM  Run from project root:   tools\build_steps.bat STEP
REM
REM  This script builds the APPLICATION package only.
REM  It does NOT build the VoiceSum Runtime.
REM  For the runtime, use:  tools\build_runtime.bat
REM
REM  STEPS (run individually or together with ALL):
REM    1  - Check prerequisites (Python, Node, npm)
REM    2  - Verify AI model assets in Application\runtime\
REM    3  - Create output directory structure
REM    4  - Build React frontend   (npm run build)
REM    5  - Package Electron wrapper
REM    6  - Build backend.exe (tiny runtime-launcher, ~10 MB, compiles fast)
REM    7  - Build launcher.exe
REM    8  - Build updater.exe
REM    9  - Copy .env, checkpoints, assets, required-runtime-version.txt
REM    10 - Package backend source -> Application\backend\app.pyz  (fast)
REM    11 - Build VoiceSum-Setup-X.Y.Z.exe  (Inno Setup, lightweight)
REM
REM  RUN ALL STEPS:
REM    tools\build_steps.bat ALL
REM
REM  NOTE: The VoiceSum Runtime (Python + torch + CUDA) is built separately:
REM    tools\build_runtime.bat
REM    (Only needed when ML dependencies change)
REM ============================================================
setlocal enabledelayedexpansion

set PROJECT_ROOT=%~dp0..
set TOOLS_DIR=%~dp0
set APP_DIR=%PROJECT_ROOT%\Application
set VENV_DIR=%PROJECT_ROOT%\backend\venv
set ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe

set STEP=%1
if "%STEP%"=="" (
    echo.
    echo  Usage:  tools\build_steps.bat [1-11 ^| ALL]
    echo.
    exit /b 1
)

REM Resolve PyInstaller path (prefer venv version)
set PYINSTALLER_CMD=pyinstaller
if exist "%VENV_DIR%\Scripts\pyinstaller.exe" (
    set PYINSTALLER_CMD="%VENV_DIR%\Scripts\pyinstaller.exe"
)

if "%STEP%"=="ALL" goto ALL_STEPS
if "%STEP%"=="1"   goto STEP1
if "%STEP%"=="2"   goto STEP2
if "%STEP%"=="3"   goto STEP3
if "%STEP%"=="4"   goto STEP4
if "%STEP%"=="5"   goto STEP5
if "%STEP%"=="6"   goto STEP6
if "%STEP%"=="7"   goto STEP7
if "%STEP%"=="8"   goto STEP8
if "%STEP%"=="9"   goto STEP9
if "%STEP%"=="10"  goto STEP10
if "%STEP%"=="11"  goto STEP11
echo ERROR: Unknown step "%STEP%". Valid: 1-11 or ALL.
exit /b 1

REM ============================================================
REM  STEP 1 - Check prerequisites
REM ============================================================
:STEP1
echo.
echo [STEP 1/11] Checking prerequisites...
where python >nul 2>&1 || ( echo   ERROR: Python not found in PATH. & exit /b 1 )
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo   Python  : %%v
where node >nul 2>&1 || ( echo   ERROR: Node.js not found in PATH. & exit /b 1 )
for /f "tokens=*" %%v in ('node --version') do echo   Node.js : %%v
where npm >nul 2>&1 || ( echo   ERROR: npm not found in PATH. & exit /b 1 )
for /f "tokens=*" %%v in ('npm --version') do echo   npm     : %%v
if exist "%VENV_DIR%\Scripts\python.exe" ( echo   venv    : OK ) else (
    echo   WARNING: backend\venv not found. PyInstaller steps may fail.
)
echo.
echo   [STEP 1 DONE]
if "%STEP%"=="ALL" goto STEP2
goto END

REM ============================================================
REM  STEP 2 - Verify model assets
REM ============================================================
:STEP2
echo.
echo [STEP 2/11] Verifying model assets in Application\runtime\...
set MISSING=0
if exist "%APP_DIR%\runtime\models\speech_engine.dat" ( echo   [OK]      speech_engine.dat ) else ( echo   [MISSING] speech_engine.dat & set MISSING=1 )
if exist "%APP_DIR%\runtime\models\align_engine.dat"  ( echo   [OK]      align_engine.dat  ) else ( echo   [MISSING] align_engine.dat  & set MISSING=1 )
if exist "%APP_DIR%\runtime\models\voice_context.dat" ( echo   [OK]      voice_context.dat ) else ( echo   [MISSING] voice_context.dat & set MISSING=1 )
if exist "%APP_DIR%\runtime\nlp-engine\"              ( echo   [OK]      nlp-engine\       ) else ( echo   [MISSING] nlp-engine\  -- copy Qwen3 folder here & set MISSING=1 )
if "!MISSING!"=="1" ( echo. & echo   ERROR: Missing model assets. & exit /b 1 )
echo.
echo   [STEP 2 DONE] All model assets present.
if "%STEP%"=="ALL" goto STEP3
goto END

REM ============================================================
REM  STEP 3 - Create output directories
REM ============================================================
:STEP3
echo.
echo [STEP 3/11] Creating output directory structure...
if not exist "%APP_DIR%"                    mkdir "%APP_DIR%"
if not exist "%APP_DIR%\runtime"            mkdir "%APP_DIR%\runtime"
if not exist "%APP_DIR%\runtime\models"     mkdir "%APP_DIR%\runtime\models"
if not exist "%APP_DIR%\runtime\nlp-engine" mkdir "%APP_DIR%\runtime\nlp-engine"
if not exist "%APP_DIR%\runtime\data"       mkdir "%APP_DIR%\runtime\data"
if not exist "%APP_DIR%\runtime\uploads"    mkdir "%APP_DIR%\runtime\uploads"
if not exist "%APP_DIR%\assets"             mkdir "%APP_DIR%\assets"
if not exist "%APP_DIR%\backend"            mkdir "%APP_DIR%\backend"
echo   Directories ready.
echo.
echo   [STEP 3 DONE]
if "%STEP%"=="ALL" goto STEP4
goto END

REM ============================================================
REM  STEP 4 - Build React frontend
REM ============================================================
:STEP4
echo.
echo [STEP 4/11] Building React frontend...
cd /d "%PROJECT_ROOT%\frontend"
call npm install --legacy-peer-deps --silent
if %ERRORLEVEL% neq 0 ( echo   ERROR: npm install failed. & exit /b 1 )
call npm run build
if %ERRORLEVEL% neq 0 ( echo   ERROR: npm run build failed. & exit /b 1 )
cd /d "%PROJECT_ROOT%"
echo   Output: frontend\dist\
echo.
echo   [STEP 4 DONE]
if "%STEP%"=="ALL" goto STEP5
goto END

REM ============================================================
REM  STEP 5 - Package Electron wrapper
REM ============================================================
:STEP5
echo.
echo [STEP 5/11] Packaging Electron wrapper...
cd /d "%PROJECT_ROOT%\frontend-electron"
call npm install --silent
if %ERRORLEVEL% neq 0 ( echo   ERROR: Electron npm install failed. & exit /b 1 )
if exist dist rmdir /S /Q dist
mkdir dist
xcopy /Y /E /Q "%PROJECT_ROOT%\frontend\dist\*" dist\ >nul
call npm run dist:dir
if %ERRORLEVEL% neq 0 ( echo   WARNING: Electron packaging failed. ) else ( echo   Output: Application\frontend\win-unpacked\ )
cd /d "%PROJECT_ROOT%"
echo.
echo   [STEP 5 DONE]
if "%STEP%"=="ALL" goto STEP6
goto END

REM ============================================================
REM  STEP 6 - Build backend.exe (tiny runtime-launcher)
REM           NO ML dependencies - compiles in seconds (~10 MB)
REM ============================================================
:STEP6
echo.
echo [STEP 6/11] Building backend.exe (tiny runtime-launcher)...
echo   This step is FAST (no ML deps). Expected output: ~10 MB.
echo   Using: !PYINSTALLER_CMD!
echo.
cd /d "%PROJECT_ROOT%"
!PYINSTALLER_CMD! "%TOOLS_DIR%\backend_launcher.spec" ^
    --distpath "%APP_DIR%\backend_launcher_dist" ^
    --workpath "build\bl_work" ^
    --noconfirm
if %ERRORLEVEL% neq 0 ( echo   ERROR: backend.exe build failed. & exit /b 1 )

REM The spec produces a onefile exe named backend.exe
REM (backend_launcher.spec uses runtime_tmpdir=None and no COLLECT -> onefile)
if exist "%APP_DIR%\backend_launcher_dist\backend.exe" (
    copy /Y "%APP_DIR%\backend_launcher_dist\backend.exe" "%APP_DIR%\backend\backend.exe" >nul
) else if exist "%APP_DIR%\backend_launcher_dist\backend\backend.exe" (
    copy /Y "%APP_DIR%\backend_launcher_dist\backend\backend.exe" "%APP_DIR%\backend\backend.exe" >nul
) else (
    echo   ERROR: backend.exe not found in distpath. Check spec output above.
    exit /b 1
)

echo   Output: Application\backend\backend.exe
for %%F in ("%APP_DIR%\backend\backend.exe") do echo   Size  : %%~zF bytes
echo.
echo   [STEP 6 DONE]
if "%STEP%"=="ALL" goto STEP7
goto END

REM ============================================================
REM  STEP 7 - Build launcher.exe
REM ============================================================
:STEP7
echo.
echo [STEP 7/11] Building launcher.exe...
cd /d "%PROJECT_ROOT%"
!PYINSTALLER_CMD! "%TOOLS_DIR%\launcher.spec" ^
    --distpath "%APP_DIR%" ^
    --workpath "build\launcher_work" ^
    --noconfirm
if %ERRORLEVEL% neq 0 ( echo   ERROR: launcher.exe build failed. & exit /b 1 )
echo   Output: Application\launcher.exe
echo.
echo   [STEP 7 DONE]
if "%STEP%"=="ALL" goto STEP8
goto END

REM ============================================================
REM  STEP 8 - Build updater.exe
REM ============================================================
:STEP8
echo.
echo [STEP 8/11] Building updater.exe...
cd /d "%PROJECT_ROOT%"
!PYINSTALLER_CMD! "%TOOLS_DIR%\updater.spec" ^
    --distpath "%APP_DIR%" ^
    --workpath "build\updater_work" ^
    --noconfirm
if %ERRORLEVEL% neq 0 ( echo   WARNING: updater.exe build failed (non-fatal). ) else ( echo   Output: Application\updater.exe )
echo.
echo   [STEP 8 DONE]
if "%STEP%"=="ALL" goto STEP9
goto END

REM ============================================================
REM  STEP 9 - Copy configs, checkpoints, assets
REM ============================================================
:STEP9
echo.
echo [STEP 9/11] Copying configs, checkpoints, assets...

REM .env default
copy /Y "%PROJECT_ROOT%\backend\.env.example" "%APP_DIR%\backend\.env" >nul 2>&1
echo   Copied: .env.example -> Application\backend\.env

REM Required runtime version file
copy /Y "%PROJECT_ROOT%\backend\required-runtime-version.txt" "%APP_DIR%\required-runtime-version.txt" >nul 2>&1
echo   Copied: required-runtime-version.txt -> Application\

REM Checkpoints
if exist "%PROJECT_ROOT%\backend\checkpoints" (
    xcopy /Y /E /I /Q "%PROJECT_ROOT%\backend\checkpoints\*" "%APP_DIR%\backend\checkpoints\" >nul 2>&1
    echo   Copied: backend\checkpoints\
)

REM Assets
if exist "%PROJECT_ROOT%\assets" (
    xcopy /Y /E /Q "%PROJECT_ROOT%\assets\*" "%APP_DIR%\assets\" >nul 2>&1
    echo   Copied: assets\
)

echo.
echo   [STEP 9 DONE]
if "%STEP%"=="ALL" goto STEP10
goto END

REM ============================================================
REM  STEP 10 - Package backend source -> app.pyz
REM            This is FAST: just zips Python source files.
REM            Run this step for every code change.
REM ============================================================
:STEP10
echo.
echo [STEP 10/11] Packaging backend source -> Application\backend\app.pyz...
echo   (Fast - no PyInstaller, just zipping Python source)

python "%TOOLS_DIR%\make_app_pyz.py" ^
    --src backend ^
    --out "Application\backend\app.pyz"

if %ERRORLEVEL% neq 0 ( echo   ERROR: app.pyz packaging failed. & exit /b 1 )

for %%F in ("%APP_DIR%\backend\app.pyz") do echo   Size  : %%~zF bytes
echo.
echo   [STEP 10 DONE]
if "%STEP%"=="ALL" goto STEP11
goto END

REM ============================================================
REM  STEP 11 - Build Inno Setup app installer (lightweight)
REM ============================================================
:STEP11
echo.
echo [STEP 11/11] Building Inno Setup app installer...
if not exist "%ISCC_PATH%" (
    echo   WARNING: Inno Setup not found at %ISCC_PATH%
    echo   Download: https://jrsoftware.org/isdl.php
    echo   Manual:   "%ISCC_PATH%" "%PROJECT_ROOT%\installer\setup.iss"
    if "%STEP%"=="ALL" goto BUILD_COMPLETE
    goto END
)
"%ISCC_PATH%" "%PROJECT_ROOT%\installer\setup.iss"
if %ERRORLEVEL% neq 0 ( echo   ERROR: Inno Setup failed. & exit /b 1 )
echo   Output: installer\dist\Setup.exe
echo.
echo   [STEP 11 DONE]
if "%STEP%"=="ALL" goto BUILD_COMPLETE
goto END

REM ============================================================
:ALL_STEPS
echo.
echo ============================================================
echo  AI Meeting Transcriber - Full App Build (ALL steps 1-11)
echo  Runtime package is built separately: tools\build_runtime.bat
echo ============================================================
goto STEP1

:BUILD_COMPLETE
echo.
echo ============================================================
echo  APP BUILD COMPLETE
echo.
echo  Application\backend\backend.exe  -- tiny runtime-launcher (~10 MB)
echo  Application\backend\app.pyz      -- backend Python source
echo  Application\launcher.exe         -- app launcher
echo  installer\dist\Setup.exe         -- lightweight app installer
echo.
echo  Prereq on client:  VoiceSum-Runtime-2.0.exe  must be installed first.
echo  Client install order:
echo    1. VoiceSum-Runtime-2.0.exe   (from tools\build_runtime.bat)
echo    2. VoiceSum-Setup-X.Y.Z.exe   (from this script)
echo ============================================================
echo.

:END
cd /d "%PROJECT_ROOT%"
endlocal
