# VoiceSum Offline Client Deployment Guide

This guide explains how to extract, configure, and run the **VoiceSum Meeting Transcriber** application on an offline client machine with zero internet connectivity.

---

## 1. Package Inventory

You should have received the following four files:

| File | Description |
| :--- | :--- |
| **`frontend.zip`** | Complete React/Vite frontend source code, assets, and offline `node_modules/`. |
| **`backend.zip`** | FastAPI backend application source, configuration, and cryptographic license verification logic. |
| **`backend_venv.zip`** | Complete Python virtual environment (`venv/`) pre-packaged with PyTorch, CUDA, WhisperX, and Transformers. |
| **`license.key`** | Standalone cryptographically signed demo license file (expires **2026-10-15**). |

---

## 2. Client System Prerequisites

Ensure the target machine has the following installed before setup:

1. **Base Python 3.12 (64-bit)**:
   * Installed and accessible in your system `PATH`.
   * Standard Python virtual environments on Windows rely on a host Python installation for base libraries and `python312.dll`.
2. **Node.js & npm**:
   * Node.js v18+ and npm installed and in system `PATH`.
3. **FFmpeg**:
   * `ffmpeg.exe` and `ffprobe.exe` must be in system `PATH` (or located in your executable search path).
4. **Microsoft Visual C++ 2015–2022 Redistributable (x64)**:
   * Required by PyTorch, torchaudio, onnxruntime, soundfile, and PyMuPDF.
5. **AI Models**:
   * Pre-downloaded speech and NLP model assets placed in the runtime directory.
6. **Hardware (Optional GPU Acceleration)**:
   * **NVIDIA GPU**: Driver version $\ge$ 550.x (CUDA 12.x compatible).
   * **CPU Fallback**: If no NVIDIA GPU is detected, the application automatically runs in CPU mode.
7. **Network Ports**:
   * Port `8000` (Backend API) and Port `8080` (Frontend UI) must be free.

---

## 3. Extraction & Directory Setup

Choose an installation folder (e.g. `C:\VoiceSum` or `D:\Client`).

Extract all three archives into the target folder so they form sibling directories:

### Extracted Directory Layout:
```text
Client/
│
├── frontend/
│   ├── node_modules/
│   ├── src/
│   ├── package.json
│   ├── run_frontend.bat
│   └── ...
│
├── backend/
│   ├── routers/
│   ├── services/
│   ├── license.py
│   ├── public_key.pem
│   ├── license.key             <-- Place license.key here
│   ├── run_backend.bat
│   └── ...
│
└── venv/
    ├── Scripts/
    ├── Lib/
    └── pyvenv.cfg
```

### Extraction Commands:

**Option A (Fastest — Windows built-in `tar`):**
```powershell
# Open terminal in the folder containing the zips:
tar.exe -xf frontend.zip
tar.exe -xf backend.zip
tar.exe -xf backend_venv.zip

# Place license.key in backend folder:
Copy-Item license.key backend\license.key
```

**Option B (Standard PowerShell `Expand-Archive` or Right-Click -> Extract All):**
```powershell
Expand-Archive -Path frontend.zip -DestinationPath .\
Expand-Archive -Path backend.zip -DestinationPath .\
Expand-Archive -Path backend_venv.zip -DestinationPath .\

Copy-Item license.key backend\license.key
```

---

## 4. Starting the Application

### Step 1 — Launch Backend:
Open a terminal or command prompt, navigate to `backend/`, and run:
```bat
cd backend
run_backend.bat
```
*What this does:*
* Checks for the virtual environment (`..\venv` or `.\venv`).
* Automatically adjusts `pyvenv.cfg` to point to the client's local Python 3.12 if needed.
* Verifies `license.key`.
* Starts Uvicorn server on `http://127.0.0.1:8000`.

*Alternative manual command:*
```bat
cd backend
..\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

---

### Step 2 — Launch Frontend:
Open a second terminal or command prompt, navigate to `frontend/`, and run:
```bat
cd frontend
run_frontend.bat
```
*What this does:*
* Runs `npm run dev` in offline mode (uses pre-packaged `node_modules/` without downloading anything).
* Starts Vite dev server on `http://localhost:8080/`.

*Alternative manual command:*
```bat
cd frontend
npm run dev
```

---

### Step 3 — Access Web UI:
Open any modern web browser (Chrome, Edge, Firefox) and navigate to:
```text
http://localhost:8080
```

---

## 5. License Management

* **Current License Expiry**: **`2026-10-15`**.
* **License Location**: `Client/backend/license.key` (or `Client/license.key`).
* **Cryptographic Signing**: The license is signed using Ed25519 asymmetric cryptography. **Do not manually edit any values (e.g. expiry date) inside `license.key`**; any modification will invalidate the cryptographic signature and block the application.
* **Replacing or Renewing License**: When you receive an updated `license.key` from your administrator, overwrite `backend/license.key` and restart the backend.
* **Obtaining Machine ID (for Node-Locked Licenses)**:
  If your license is locked to your specific hardware, run the following command in PowerShell and share the result with your administrator:
  ```powershell
  (Get-CimInstance -Class Win32_ComputerSystemProduct).UUID
  ```

---

## 6. Verification & Health Checks

### Check Backend Health & License Status:
In your browser or PowerShell:
```powershell
curl http://127.0.0.1:8000/health
```
**Expected Response:**
```json
{
  "status": "ok",
  "license_valid": true,
  "license_expiry": "2026-10-15",
  "license_message": "Valid",
  "offline_mode": true
}
```

### Test GPU & CUDA Availability:
From the `backend/` directory:
```powershell
..\venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 7. Troubleshooting & FAQs

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **"No Python at '...' "** | Python path in `pyvenv.cfg` points to build machine. | Run `run_backend.bat` — it automatically detects your host Python 3.12 and updates `pyvenv.cfg`. |
| **"License file 'license.key' not found"** | `license.key` is missing or placed in the wrong folder. | Ensure `license.key` is placed inside `backend/` or in the root `Client/` folder. |
| **"Signature verification failed"** | `license.key` was altered, corrupted, or re-saved with modified text. | Restore the original unmodified `license.key` provided by the administrator. |
| **"Demo license expired"** | System date is past `2026-10-15`. | Contact administrator for a renewed license key. |
| **"System clock rollback detected"** | System clock was set backward or prior to the license issuance date. | Restore the correct current date and time on your operating system. |
| **Port 8000 / 8080 already in use** | Another process is occupying the port. | Terminate the process using the port or reconfigure the port via `.env` / `vite.config.ts`. |
| **FFmpeg not found errors** | FFmpeg executables are not in system `PATH`. | Add the folder containing `ffmpeg.exe` and `ffprobe.exe` to Windows System Environment Variables `Path`. |
