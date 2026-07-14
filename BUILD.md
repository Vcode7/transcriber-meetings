# AI Meeting Transcriber — Build & Distribution Guide

> Complete guide for building the production Windows installer.

---

## Overview

```
Source Code
    │
    ├─ Phase 1: Build React Frontend
    ├─ Phase 2: Package Electron (→ Application/frontend/win-unpacked/)
    ├─ Phase 3: Compile Backend  (PyInstaller → Application/backend/)
    ├─ Phase 4: Compile Launcher (PyInstaller → Application/launcher.exe)
    ├─ Phase 5: Place Models     (encrypted .dat + nlp-engine/ separately)
    └─ Phase 6: Build Installer  (Inno Setup → installer/dist/Setup.exe)
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | Active venv with requirements.txt |
| Node.js | 18+ | https://nodejs.org |
| PyInstaller | Latest | `pip install pyinstaller` |
| cryptography | Latest | `pip install cryptography` |
| bitsandbytes | Latest | `pip install bitsandbytes accelerate` |
| Electron + electron-builder | Latest | `npm install` in `frontend-electron/` |
| Inno Setup | 6.x | https://jrsoftware.org/isdl.php |

---

## Application Structure (after build)

```
Application/
├── launcher.exe              ← Client opens this ONLY
├── backend/                  ← PyInstaller-compiled backend
│   ├── backend.exe
│   ├── .env                  ← Copied from .env.example at build time
│   └── _internal/            ← Bundled Python runtime + ML libraries
├── frontend/                 ← Electron win-unpacked output
│   ├── AI Meeting Transcriber.exe
│   └── resources/app.asar
├── runtime/                  ← Created by installer; models placed separately
│   ├── models/               ← Packaged model folders (NOT in installer)
│   │   ├── speech_engine/       (faster-whisper/medium)
│   │   ├── audio_context/       (pyannote/speaker-diarization-community-1 snapshot)
│   │   ├── ecapa_tdnn/          (SpeechBrain ECAPA-TDNN speaker embedding)
│   │   ├── align_engine/        (wav2vec2-base alignment)
│   │   └── model_manifest.json  (manifest mapping and checksums)
│   ├── nlp-engine/           ← Qwen3-4B plain folder (NOT in installer)
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── model-*.safetensors
│   ├── data/
│   │   └── voicesum.db          (SQLite — auto-created on first run)
│   └── uploads/                 (audio files — auto-created)
├── assets/
│   └── icon.ico
└── license.dat                  (auto-created on first run)
```

> **Important**: The `runtime/models/` and `runtime/nlp-engine/` directories are
> **NOT bundled in the installer**. They must be placed manually after installation
> (or delivered via a separate media package). The installer creates the empty
> directory structure automatically.

---

## Step-by-Step Build

### Step 1 — Prepare Python Environment

```powershell
# Activate your project venv (in backend/)
backend\venv\Scripts\activate

# Install build dependencies
pip install pyinstaller cryptography httpx bitsandbytes accelerate
```

### Step 2 — Download & Package Models (Run ONCE on build machine)

#### 2a — Download encrypted models to HF cache
```powershell
# Requires HuggingFace token for pyannote models
python tools/download_all_models.py --hf-token YOUR_HF_TOKEN
```

#### 2b — Encrypt & package models
```powershell
# Creates Application/runtime/models/*.dat
python tools/encrypt_models.py --output Application/runtime/models

# Verify output (dry-run check)
python tools/encrypt_models.py --dry-run
```

> **Security**: Keep `tools/model.key` **private**. Loss = cannot decrypt packaged models.
> The key file is NOT included in the installer.

#### 2c — Place Qwen3 nlp-engine (plain, not encrypted)

The Qwen3-4B model is **not encrypted** and is too large to load from memory.
It must be placed as a plain directory:

```powershell
# Copy the Qwen3-4B model folder to:
Application\runtime\nlp-engine\

# The folder should contain:
#   config.json, tokenizer.json, tokenizer_config.json,
#   generation_config.json, model-*.safetensors, vocab.json
```

If you need to download it first:
```powershell
python tools/download_qwen.py
```

### Step 3 — Build React Frontend

```powershell
cd frontend
npm install --legacy-peer-deps
$env:VITE_ELECTRON="true"; npm run build  # Electron mode
cd ..
```

### Step 4 — Package Electron Frontend

```powershell
cd frontend-electron
npm install
npm run dist:dir      # → Application/frontend/win-unpacked/
cd ..
```

### Step 5 — Compile Backend (PyInstaller)

```powershell
pyinstaller tools/backend.spec `
    --distpath Application/ `
    --workpath build/backend_work `
    --noconfirm

# Result: Application/backend/backend.exe
```

> ⚠️ First run takes 5–15 minutes. Bundles full Python runtime + all ML libraries.

### Step 6 — Compile Launcher (PyInstaller)

```powershell
pyinstaller tools/launcher.spec `
    --distpath Application/ `
    --workpath build/launcher_work `
    --noconfirm

# Result: Application/launcher.exe
```

### Step 7 — Build Installer (Inno Setup)

```powershell
# After installing Inno Setup 6:
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer/setup.iss

# Result: installer/dist/Setup_AIMeetingTranscriber_v1.0.0.exe
```

> **Note**: The installer does NOT include models. It creates the empty `runtime/`
> directory structure. Models must be placed separately after installation.

---

## One-Command Build

```powershell
tools\build_all.bat
```

---

## Model Distribution

Models are distributed **separately** from the installer because of their large size (~4–5 GB).

| Model | Location after install | Size | Notes |
|---|---|---|---|
| speech_engine.dat (whisper-medium) | `runtime/models/` | ~300 MB | Encrypted |
| audio_context.dat (pyannote) | `runtime/models/` | ~10 MB | Encrypted |
| voice_segment.dat (segmentation) | `runtime/models/` | ~10 MB | Encrypted |
| voice_context.dat (speechbrain) | `runtime/models/` | ~85 MB | Encrypted |
| align_engine.dat (wav2vec2) | `runtime/models/` | ~725 MB | Encrypted |
| wespeaker.dat | `runtime/models/` | ~25 MB | Encrypted |
| nlp-engine/ (Qwen3-4B) | `runtime/nlp-engine/` | ~2.3 GB | **Plain, not encrypted** |

After the installer runs, the technician copies the models into the installation directory:
```
C:\Program Files\AI Meeting Transcriber\runtime\models\    ← .dat files
C:\Program Files\AI Meeting Transcriber\runtime\nlp-engine\ ← Qwen3 folder
```

---

## License System

### How It Works

1. **First launch**: `launcher.exe` creates `license.dat` containing:
   - `install_date`: Today's date
   - `machine_id`: SHA-256 hash of (hostname + CPU + disk serial)
   - `expires_date`: install_date + 30 days
   - `last_run_date`: Updated every launch
   - `signature`: HMAC-SHA256 with app secret + machine ID

2. **Every launch**: Verifies:
   - Signature matches current machine
   - `current_date > last_run_date` (clock rollback detection)
   - `current_date <= expires_date` (expiry check)

3. **On expiry**: Shows:
   > "This demo version has expired. Please contact the developer."
   > Then exits gracefully.

### License File Location

```
C:\Program Files\AI Meeting Transcriber\license.dat
```

### Resetting for Testing

Delete `license.dat` and relaunch — a fresh 30-day license is created.

---

## Model Encryption

### Encryption Algorithm

- **Cipher**: AES-128-CBC (Fernet = AES-CBC + HMAC-SHA256)
- **Key**: Fernet-generated random 32-byte key stored in `tools/model.key`
- **Format**: Each `.dat` file = `VSDAT\x01` header + Fernet-encrypted tar.gz
- **Decryption**: At runtime, `model_loader.py` decrypts to `%TEMP%\voicesum_runtime\<name>\`
- **Cleanup**: Temp dir is deleted when the backend process exits

### Qwen3 (nlp-engine) — No Encryption

Qwen3-4B is **not encrypted** because loading a ~2.3 GB file into RAM for decryption
would exceed available memory on most systems. It is stored as a plain directory and
loaded directly by `transformers` with `local_files_only=True`.

### Model Name Mapping

| Generic Name | Original Model | Type |
|---|---|---|
| `speech_engine/` | Systran/faster-whisper-medium | Plain directory |
| `audio_context/` | pyannote/speaker-diarization-community-1 (complete snapshot) | Plain directory |
| `ecapa_tdnn/` | speechbrain/spkrec-ecapa-voxceleb (ECAPA-TDNN) | Plain directory |
| `align_engine/` | facebook/wav2vec2-base | Plain directory |
| `nlp-engine/` | Qwen/Qwen3-4B | Plain directory |

---

## AI Architecture

### Offline Features (no internet required)

| Feature | Provider |
|---|---|
| Transcription | WhisperX (speech_engine/) |
| Speaker Diarization | pyannote (audio_context/) |
| Speaker Identification | SpeechBrain ECAPA-TDNN (ecapa_tdnn/) |
| Meeting Summary | QwenProvider (runtime/nlp-engine/) |
| Key Points | QwenProvider |
| Action Items | QwenProvider |
| Minutes of Meeting | QwenProvider |
| AI Chat Assistant | QwenProvider (offline) |

### Qwen3 Loading Flow

```
Backend startup
    │
    ├─ warm_up_model() called
    ├─ QwenProvider._get_pipeline()
    │       ├─ ModelLoader.get_model_path("nlp_engine")
    │       │       └─ Checks: runtime/models/nlp_engine/
    │       │                  runtime/models/nlp-engine/
    │       │                  runtime/nlp_engine/
    │       │                  runtime/nlp-engine/  ← FOUND (production)
    │       └─ AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True)
    └─ Model loaded ✓ (no internet required)
```

---

## Launcher Workflow

```
launcher.exe launched
    │
    ├─ Show splash screen (tkinter, frameless, dark theme)
    ├─ Step 1: Verify license.dat
    │       └─ If expired → show error message + Exit button
    ├─ Step 2: Verify runtime/ directory exists
    ├─ Step 3: Start backend.exe (no console window)
    ├─ Step 4: Poll GET http://127.0.0.1:8000/health every 500ms
    │       └─ Max wait: 120 seconds
    │       └─ On timeout → show error + terminate backend
    ├─ Step 5: "Services Ready!" splash update
    ├─ Step 6: Launch frontend.exe (from frontend/win-unpacked/)
    ├─ Close splash
    └─ Monitor frontend PID
            └─ On frontend exit:
                ├─ Terminate backend.exe
                └─ Delete %TEMP%\voicesum_runtime\
```

---

## Troubleshooting

### Backend fails to start

Check `launcher.log` in the installation directory.

Common causes:
- Port 8000 already in use → `netstat -ano | findstr :8000`
- Missing `runtime/models/*.dat` → copy model files to runtime/models/
- Missing `runtime/nlp-engine/` → copy Qwen3-4B folder to runtime/nlp-engine/
- Antivirus blocking backend.exe → Add exception

### Qwen3 fails to load

Shown in backend log as `[QwenAI] Local nlp-engine model not found`.

Fix: Ensure `runtime/nlp-engine/` contains all Qwen3-4B files:
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `generation_config.json`
- `model-*.safetensors`

### Pyannote fails to load

The diarization service falls back to energy-based diarization automatically.
Expected on systems without HF_TOKEN or if the encrypted model is missing.

### Models decrypt slowly

First launch may take 30–60 seconds for large models (speech_engine ~300 MB compressed).
Subsequent launches are instant (temp files cached until process exit).

---

## Security Notes

- Model `.dat` files are encrypted and cannot be used without `model.key`
- `model.key` is NEVER shipped with the installer
- Decrypted models exist only in `%TEMP%` during runtime
- Qwen3 (`nlp-engine/`) is distributed as a plain folder — protect access accordingly
- License is machine-bound — cannot be copied to another computer
- No Python source `.py` files in the distributed application
- No React/TypeScript source files in the distributed application

---

## File Sizes (estimated)

| Component | Size |
|---|---|
| speech_engine.dat (whisper-medium) | ~300 MB |
| audio_context.dat (pyannote) | ~10 MB |
| align_engine.dat (wav2vec2) | ~725 MB |
| voice_context.dat (speechbrain) | ~85 MB |
| nlp-engine/ (Qwen3-4B) | ~2,300 MB |
| Other models | ~60 MB |
| Backend runtime (PyInstaller) | ~800 MB |
| Frontend (Electron) | ~180 MB |
| **Installer .exe (excludes models)** | **~1.0 GB** |
| **Total installed (with models)** | **~4.5 GB** |

> Note: The installer itself only bundles the application (~1 GB). Models (~3.5 GB)
> are delivered separately and placed in the runtime/ directory manually.
