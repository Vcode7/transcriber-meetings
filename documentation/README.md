# VoiceSum System Documentation Index

Welcome to the comprehensive technical documentation for the **VoiceSum / AI Meeting Transcriber** platform. Each document provides deep architectural flows, code references, database schemas, and mathematical algorithms for its respective domain.

---

## Documentation Directory Map

| Document | Core Subject | Primary Code Files |
|---|---|---|
| [**00. Application Overview**](file:///e:/projects/Transcriber-meeting-initial/documentation/00_APPLICATION_OVERVIEW.md) | End-to-end architecture, technology stack, directory map, offline security | `main.py`, `config.py`, `database.py` |
| [**01. Transcription & Diarization**](file:///e:/projects/Transcriber-meeting-initial/documentation/01_TRANSCRIPTION_AND_DIARIZATION.md) | Audio normalization, VAD, Faster-Whisper, Pyannote diarization, WhisperX word alignment, ECAPA2 voice profile matching | `tasks/pipeline.py`, `services/transcription.py`, `services/diarization.py`, `services/identification.py` |
| [**02. Record of Meeting (ROM) Pipeline**](file:///e:/projects/Transcriber-meeting-initial/documentation/02_ROM_SYSTEM_AND_PIPELINE.md) | Short/Medium/Long ROM hierarchy, factual extraction, timeline anchoring, ChromaDB sync, interactive editor | `routers/rom_router.py`, `services/rom_service.py`, `frontend/src/pages/RomPage.tsx` |
| [**03. AI Services & Local LLM**](file:///e:/projects/Transcriber-meeting-initial/documentation/03_AI_SERVICES_AND_LOCAL_LLM.md) | Local Qwen3-4B-Instruct, Ollama fallback, Qwen3-Embedding-0.6B, ChromaDB vector store, PaddleOCR PP-OCRv6 | `services/llm.py`, `services/ai_provider.py`, `services/text_embedding_service.py`, `services/ocr_engine.py` |
| [**04. AI Chat & Multi-Document RAG**](file:///e:/projects/Transcriber-meeting-initial/documentation/04_AI_CHAT_AND_RETRIEVAL.md) | Multi-tier retrieval hierarchy (Manual MoMs, ROM points, Slides, Transcripts), streaming responses, citations, document isolation | `routers/ai_chat_router.py`, `services/ai_chat_service.py`, `services/manual_mom_service.py`, `frontend/src/pages/AIChat.tsx` |
| [**05. Model Training & Fine-Tuning**](file:///e:/projects/Transcriber-meeting-initial/documentation/05_MODEL_TRAINING_AND_FINE_TUNING.md) | Stage 1/2/3 fine-tuning, QLoRA adapter training, dataset curation, variant activation, Model Arena blind benchmarking | `routers/training_routes.py`, `routers/rom_training_router.py`, `routers/arena_router.py`, `frontend/src/pages/ModelArena.tsx` |
| [**06. Recording Sessions & History**](file:///e:/projects/Transcriber-meeting-initial/documentation/06_RECORDINGS_AND_HISTORY.md) | History listing, authenticated audio streaming, synchronized word-highlight player, speaker reassignment, collections | `routers/history.py`, `frontend/src/pages/History.tsx`, `frontend/src/pages/HistoryDetail.tsx` |
| [**07. Video Meeting Analysis**](file:///e:/projects/Transcriber-meeting-initial/documentation/07_VIDEO_MEETING_ANALYSIS.md) | Video stream splitting, keyframe sampling, slide transition detection, visual OCR text extraction, multimodal alignment | `routers/video_router.py`, `services/video_processing_service.py`, `frontend/src/pages/VideoUpload.tsx` |
| [**08. Speaker Management & Voice Profiles**](file:///e:/projects/Transcriber-meeting-initial/documentation/08_SPEAKER_MANAGEMENT_AND_PROFILES.md) | Voice enrollment, ECAPA2 centroid extraction, cosine similarity, global speaker sync, calendar-scoped matching | `routers/voice.py`, `services/identification.py`, `services/speaker_sync.py`, `services/voice_profile_service.py` |
| [**09. Settings, Database & Deployment**](file:///e:/projects/Transcriber-meeting-initial/documentation/09_SETTINGS_DATABASE_AND_DEPLOYMENT.md) | SQLite schema definitions, engine hyperparameters, Electron desktop shell, PyInstaller spec, zero-downtime updater | `database.py`, `routers/settings_router.py`, `tools/backend.spec`, `tools/create_code_patch.py` |
