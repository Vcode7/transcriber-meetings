# Settings, Database Schema & Deployment Architecture

## 1. Relational Persistence: SQLite Schema (`voicesum.db`)

All application state, user accounts, meetings, transcripts, and model configurations are stored locally in `backend/voicesum.db` using **SQLAlchemy Async + aiosqlite**.

### 1.1 Core Tables Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Primary SQLite Relational Schema                    │
├──────────────────────────┬─────────────────────────────────────────────┤
│ recordings               │ Stores audio/video meetings, transcripts,   │
│                          │ duration, status, and ROM JSON.             │
├──────────────────────────┼─────────────────────────────────────────────┤
│ manual_moms              │ Stores pure text/document manual MoMs       │
│                          │ (.docx, .pdf) uploaded for AI Chat RAG.     │
├──────────────────────────┼─────────────────────────────────────────────┤
│ rom_metadata             │ Revision history and finalized ROM points.  │
├──────────────────────────┼─────────────────────────────────────────────┤
│ minutes_of_meeting       │ Traditional corporate MoM format records.   │
├──────────────────────────┼─────────────────────────────────────────────┤
│ user_settings            │ Global inference, VAD, and model thresholds.│
├──────────────────────────┼─────────────────────────────────────────────┤
│ speaker_profiles         │ Enrolled voice signatures and centroids.    │
├──────────────────────────┼─────────────────────────────────────────────┤
│ meeting_collections      │ Organizational project folders & groups.    │
├──────────────────────────┼─────────────────────────────────────────────┤
│ meeting_collection_items │ Many-to-many link between meetings & groups.│
├──────────────────────────┼─────────────────────────────────────────────┤
│ ai_chat_sessions         │ Conversational threads & user queries.      │
├──────────────────────────┼─────────────────────────────────────────────┤
│ ai_chat_messages         │ Chat turn history and grounded citations.   │
├──────────────────────────┼─────────────────────────────────────────────┤
│ arena_battles            │ A/B blind testing votes and Elo metrics.    │
└──────────────────────────┴─────────────────────────────────────────────┘
```

### 1.2 Table Definitions (`database.py`)

#### `recordings`
```sql
CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT DEFAULT NULL,
    file_path TEXT NOT NULL,
    duration REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'pending',
    progress TEXT,
    transcript TEXT NOT NULL DEFAULT '[]',
    raw_text TEXT,
    summary TEXT,
    short_summary TEXT,
    detailed_summary TEXT,
    key_points TEXT NOT NULL DEFAULT '[]',
    action_items TEXT NOT NULL DEFAULT '[]',
    speakers_detected TEXT NOT NULL DEFAULT '[]',
    language TEXT DEFAULT 'en',
    error_message TEXT,
    created_at TEXT NOT NULL,
    processed_at TEXT,
    source_type TEXT DEFAULT 'audio',
    rom_data TEXT,
    video_transcript TEXT
);
```

#### `manual_moms`
```sql
CREATE TABLE IF NOT EXISTS manual_moms (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT NOT NULL,
    meeting_date TEXT,
    file_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_type TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    summary TEXT,
    speakers TEXT NOT NULL DEFAULT '[]',
    agendas TEXT NOT NULL DEFAULT '[]',
    action_items TEXT NOT NULL DEFAULT '[]',
    decisions TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    chunks_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

---

## 2. User Settings & Tunable Hyperparameters (`routers/settings_router.py`)

All engine hyperparameters are configurable through the Settings UI:

| Setting Key | Default Value | Description |
|---|---|---|
| `speaker_similarity_threshold` | `0.75` | Minimum cosine similarity required to auto-identify an enrolled speaker. |
| `enable_vad` | `1` (True) | Enables Voice Activity Detection to purge silence prior to Whisper. |
| `vad_speech_threshold` | `0.15` | Energy probability threshold for detecting active speech. |
| `norm_target_dbfs` | `-3.0` | Target peak decibels for audio normalization. |
| `whisper_parallel_processing` | `1` (True) | Splits files longer than 10 mins into parallel worker chunks. |
| `whisper_parallel_chunk_minutes`| `10` | Size of parallel audio slices in minutes. |
| `use_ollama` | `0` (False) | Toggles external local Ollama server vs in-process HuggingFace Qwen3-4B. |
| `ollama_server_url` | `http://localhost:11434` | Endpoint for Ollama instance. |
| `ollama_num_ctx` | `32768` | Dynamic context window allocation for large transcript batches. |
| `max_tokens_detailed_summary` | `3000` | Token limit for comprehensive meeting summaries. |
| `max_tokens_rom_version_short` | `4096` | Token budget for Short ROM generation. |
| `reasoning_mode_short_rom` | `0` (False) | Enables chain-of-thought verification step before emitting ROM JSON. |

---

## 3. Application Packaging & Distribution

### 3.1 PyInstaller Spec (`tools/backend.spec`)
- Bundles Python 3.12, PyTorch, CTranslate2, Faster-Whisper, Pyannote, and FastAPI into a single executable `backend.exe`.
- Dynamically excludes unnecessary cloud drivers while bundling offline torch DLLs, CUDA runtimes, and local assets.

### 3.2 Electron Desktop Shell (`frontend-electron/`)
- Wraps the React web build (`frontend/dist/`) in a lightweight native Chromium window.
- Spawns and manages the lifecycle of the local FastAPI backend on `http://127.0.0.1:8000`.
- Native system tray integration, microphone hardware permission prompts, and offline file dialogs.

### 3.3 Zero-Downtime Code Patch Builder (`tools/create_code_patch.py`)
- Enables differential code updates without re-downloading multi-gigabyte neural model weights.
- Hashes changed files, validates dependencies, generates `patch_manifest.json`, and packages a cryptographically validated zip archive.
- The built-in updater (`tools/updater.py`) verifies SHA-256 digests and hot-applies updates automatically.
