# 🎙️ Transcriber Meeting — AI-Powered Local Meeting & Speech Intelligence Platform

A 100% offline, privacy-first full-stack application for recording, transcribing, diarizing, and analyzing multi-speaker conversations. Features SpeechBrain ECAPA-TDNN speaker voice identification, local Qwen3 4B LLM intelligence, local Qwen3 RAG vector embeddings, visual video slide OCR, and cross-platform Electron desktop integration.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![React](https://img.shields.io/badge/react-18+-61dafb.svg)
![TypeScript](https://img.shields.io/badge/typescript-5+-3178c6.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.100+-009688.svg)
![Electron](https://img.shields.io/badge/electron-30+-47848F.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## ✨ System Architecture & Key Features

### 🎤 1. Audio & Video Ingestion
- **Live Microphone Recording**: Browser-based audio capture with crash recovery check and un-saved session restoration.
- **System & Tab Audio Capture**: Capture audio from specific browser tabs or system-wide audio output.
- **Audio File Upload**: Upload pre-recorded audio files (`WAV`, `MP3`, `M4A`, `AAC`, `FLAC`, etc.).
- **Video File Upload & Audio Extraction**: Video uploading (`MP4`, `MKV`, `AVI`) with automated audio extraction and video frame processing.

### 🗣️ 2. Speech Recognition & Voice Identification
- **Local Faster-Whisper Engine**: High-performance local speech recognition powered by `faster-whisper` (default model `large-v3`) with configurable compute types (`int8`, `float16`) and `cuda`/`cpu` hardware controls.
- **SpeechBrain ECAPA-TDNN Voice Embeddings**: Extracts 192-dimensional L2-normalized speaker embeddings (SpeechBrain ECAPA-TDNN architecture) for multi-sample centroid voice profile enrollment and cross-session speaker re-identification.
- **Speaker-Level Diarization & Identification**: Group-level diarization segment aggregation ensuring a single, stable speaker label per diarized speaker.
- **Dual-Engine Diarization**: Neural `pyannote.audio` diarization (optional with HF token) and adaptive VAD energy-based fallback diarization.
- **Overlap Detection**: Wav2Vec2-based binary classifier for detecting overlapping speech segments.

### 🧠 3. 100% Offline AI Summarization & LLM Intelligence
- **Local Qwen3 4B Instruct**: Fully offline, 4-bit quantized resident LLM (`Qwen/Qwen3-4B`) for fast, local inference without external API calls.
- **Ollama Offline Fallback**: Configurable fallback server supporting local Ollama models (`gemma`, `qwen`, `llama`, `deepseek`, `mistral`).
- **Minutes of Meeting (MoM)**: Automated extraction of executive summaries, discussion topics, outcomes, conclusions, and next steps.
- **Record of Meeting (RoM)**: Comprehensive deep meeting report breakdown with agenda tracking, key takeaways, and decision trees.
- **Custom Prompt Templates & Raw MoM Lab**: Pre-built and user-defined prompt templates, plus an interactive prompt engineering lab.
- **Attachment Context Processing**: Attach background documents (`PDF`, `DOCX`, `PPTX`, `TXT`, `MD`, `PNG`, `JPG`, `CSV`, `XLSX`) to automatically enrich MoM generation with context.

### 🖼️ 4. Video Keyframe Analysis & OCR Slide Processing
- **Keyframe Slide Extraction**: Automatically detects visual transitions and extracts slides from uploaded video recordings.
- **Visual OCR Engine**: Tesseract / Paddle OCR extracts slide text from screen recordings or visual presentations and merges visual context into meeting transcripts.

### 🔍 5. Hybrid RAG & Collection AI
- **Qwen3 Vector Embeddings**: Local `Qwen3-Embedding-0.6B` / `Qwen3-Embedding-4B-Instruct-INT8` embeddings combined with SQLite / FAISS vector stores.
- **Hybrid Retrieval (BM25 + Dense Vectors)**: Combines BM25 lexical keyword search with dense vector embeddings for high-precision context retrieval.
- **Meeting Collections & Global Context**: Index global background documents or group recordings into collections for cross-meeting Q&A ("Chat with Collections").

### 📖 6. Custom Vocabulary & Technical Glossary
- **Domain Glossary**: Add specialized industry jargon, employee names, and acronyms.
- **Automated Term Extraction**: Extracts technical vocabulary from transcripts and uploaded documents.
- **Speech & LLM Injection**: Injects custom dictionary words into speech recognition initial prompts and LLM context to prevent misspellings.

### 📊 7. Analytics & Interactive Transcript Viewer
- **Talk-Time & Sentiment Analytics**: Visual breakdown of total duration, speaker talk-time percentages, sentiment indicators, and audio quality scores.
- **Synchronized Transcript Player**: Audio playback synchronization, speaker filtering, inline transcript editing, and segment splitting/merging.
- **Multi-Format Export**: Export transcripts and reports to `PDF`, `Word (DOCX)`, `TXT`, `JSON`, `SRT`, and `VTT`.

### 💻 8. Desktop App & Offline Backend Launcher
- **Electron Desktop Container**: Desktop wrapper (`frontend-electron`) with native system tray controls.
- **Standalone Backend Launcher**: Self-contained backend executable bundled via PyInstaller (`tools/backend.spec`).

---

## 🛠️ Tech Stack Breakdown

### Backend (Python)
- **Framework**: FastAPI with `asyncio` & Pydantic settings
- **Database**: SQLite (`voicesum.db`) via `sqlalchemy` / `aiosqlite`
- **Speech Recognition**: `faster-whisper`
- **Speaker Embedding**: `speechbrain` ECAPA-TDNN (192-d)
- **Diarization**: `pyannote.audio` / VAD fallback
- **Offline LLM**: `Qwen/Qwen3-4B` (4-bit quantized) / Ollama fallback
- **RAG & Vector Embeddings**: `Qwen3-Embedding-0.6B`, BM25 lexical retriever, SQLite / FAISS vector store
- **OCR Engine**: Tesseract / Paddle OCR

### Frontend (React & TypeScript)
- **Framework**: React 18, Vite, TypeScript
- **Styling**: Tailwind CSS + shadcn/ui
- **State Management**: Zustand
- **Routing & Networking**: React Router (HashRouter), Axios

### Desktop App (Electron)
- **Container**: Electron (`frontend-electron`)
- **PyInstaller Spec**: `tools/backend.spec` launcher

---

## 📁 Project Structure

```
Transcriber-meeting-initial/
├── backend/                    # FastAPI Backend Application
│   ├── main.py                 # FastAPI application entrypoint
│   ├── config.py               # Application & runtime config (Qwen3, ECAPA-TDNN, SQLite)
│   ├── database.py             # SQLite database connection & ORM models
│   ├── routers/                # API Endpoints
│   │   ├── audio.py            # Audio recording & upload endpoints
│   │   ├── auth.py             # User authentication & token refresh
│   │   ├── voice.py            # Speaker voice profile management
│   │   ├── history.py          # Session history & transcript viewer
│   │   ├── mom_router.py       # Minutes of Meeting (MoM) generation
│   │   ├── rom_router.py       # Record of Meeting (RoM) endpoints
│   │   ├── raw_mom_router.py   # Raw MoM Lab testing environment
│   │   ├── video_router.py     # Video upload & OCR slide processing
│   │   ├── attachments_router.py # Agenda & context file uploads
│   │   ├── collections_router.py # Meeting collections management
│   │   ├── collection_ai_router.py # Collection-level RAG & Q&A
│   │   ├── dictionary_router.py  # Domain glossary & custom terms
│   │   ├── global_context_router.py # Global document repository
│   │   ├── prompt_templates_router.py # Prompt template management
│   │   ├── analytics_router.py  # Dashboard statistics & talk-time metrics
│   │   └── settings_router.py  # Hardware, model, & API key settings
│   └── services/               # Core Business Logic & AI Engines
│       ├── transcription.py    # faster-whisper engine
│       ├── embedding.py        # SpeechBrain ECAPA-TDNN (192-d) embeddings
│       ├── identification.py   # Speaker identification & centroid matching
│       ├── diarization.py      # pyannote & VAD diarization
│       ├── ai_provider.py      # Local Qwen3 4B Instruct LLM engine
│       ├── text_embedding_service.py # Local Qwen3 text embeddings
│       ├── rag_pipeline.py     # BM25 + Vector RAG search pipeline
│       ├── collection_ai_service.py # RAG Q&A orchestrator
│       ├── ocr_engine.py       # OCR slide text extraction
│       └── video_processing_service.py # Keyframe extraction
│
├── frontend/                   # React + Vite Frontend Web Application
│   ├── src/
│   │   ├── api/                # Axios client setup
│   │   ├── components/         # UI components & transcript viewers
│   │   ├── pages/              # Main view pages
│   │   │   ├── Record.tsx      # Live microphone recording page
│   │   │   ├── TabAudio.tsx    # Tab/System audio recording page
│   │   │   ├── Upload.tsx      # Audio file upload page
│   │   │   ├── VideoUpload.tsx # Video upload & slide OCR page
│   │   │   ├── History.tsx     # Session list & search
│   │   │   ├── HistoryDetail.tsx # Synchronized transcript player
│   │   │   ├── MomPage.tsx     # MoM generation & export view
│   │   │   ├── RomPage.tsx     # Record of Meeting view
│   │   │   ├── RawMomLab.tsx   # Custom prompt laboratory
│   │   │   ├── GlobalContext.tsx # Global document repository
│   │   │   ├── Dictionary.tsx  # Vocabulary management page
│   │   │   ├── AddVoice.tsx    # Voice profile setup
│   │   │   ├── Dashboard.tsx   # Analytics & meeting overview
│   │   │   └── Settings.tsx    # App & model configuration
│   │   └── store/              # Zustand global state stores
│   ├── package.json
│   └── vite.config.ts
│
├── frontend-electron/          # Electron Desktop App Wrapper
│   ├── main.js                 # Electron main process & tray icon
│   └── package.json
│
├── launcher/                   # Desktop Backend Launcher Utility
├── tools/                      # PyInstaller Build Specs & Packaging Tools
└── README.md                   # Project Documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.9+**
- **Node.js 18+** & **npm**
- **FFmpeg** installed and added to system `PATH`
- *(Optional)* **HuggingFace Token** (for `pyannote.audio` diarization)
- *(Optional)* **Ollama** installed locally for alternative offline models

---

### 1️⃣ Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

Start the FastAPI backend server:
```bash
uvicorn main:app --reload --port 8000
```
- **Backend API**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`

---

### 2️⃣ Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
- **Frontend App**: `http://localhost:5173`

---

### 3️⃣ Electron Desktop Setup (Optional)

```bash
# Navigate to the frontend-electron directory
cd frontend-electron

# Install dependencies
npm install

# Launch Electron desktop application
npm start
```

---

## ⚙️ Configuration Reference (`backend/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEAKER_EMBEDDING_MODEL` | `ecapa_tdnn` | Active speaker embedding model (SpeechBrain ECAPA-TDNN) |
| `SPEAKER_SIMILARITY_THRESHOLD_ECAPA_TDNN` | `0.72` | Cosine similarity threshold for ECAPA-TDNN speaker matching |
| `QWEN_MODEL_ID` | `Qwen/Qwen3-4B` | Local offline 4-bit quantized resident LLM |
| `EMBEDDING_MODEL` | `Qwen3-Embedding-0.6B` | Local text embedding model for RAG vector search |
| `OLLAMA_SERVER_URL` | `http://localhost:11434` | URL of local Ollama server fallback |
| `WHISPER_MODEL_SIZE` | `large-v3` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `WHISPER_DEVICE` | `auto` | Execution target (`auto`, `cuda`, `cpu`) |
| `OFFLINE_MODE` | `True` | Completely offline execution mode |

---

## 📝 License

This project is licensed under the **MIT License**.
