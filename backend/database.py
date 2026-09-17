"""
Database layer — SQLite via aiosqlite + SQLAlchemy async.

Replaces the previous MongoDB/Motor implementation.
All nested JSON data (transcript, embeddings, etc.) is stored as JSON strings.
IDs are UUID strings throughout.
"""
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

from config import settings

logger = logging.getLogger(__name__)

engine = None
AsyncSessionLocal = None


async def connect_db():
    """Create the SQLite engine and initialise all tables."""
    global engine, AsyncSessionLocal

    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                hashed_password TEXT NOT NULL,
                needs_setup INTEGER NOT NULL DEFAULT 1,
                own_profile_id TEXT,
                locked_until TEXT,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                refresh_token_hash TEXT NOT NULL,
                device_name TEXT,
                ip_address TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                last_used TEXT NOT NULL,
                is_revoked INTEGER NOT NULL DEFAULT 0
            )
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                success INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                label TEXT NOT NULL,
                embeddings TEXT NOT NULL DEFAULT '[]',
                sample_count INTEGER NOT NULL DEFAULT 0,
                is_self INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))

        await conn.execute(text("""
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
                processed_at TEXT
            )
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                speaker_similarity_threshold REAL NOT NULL DEFAULT 0.75,
                word_conf_low REAL NOT NULL DEFAULT 0.7,
                word_conf_mid REAL NOT NULL DEFAULT 0.85,
                min_segment_duration REAL NOT NULL DEFAULT 1.5,
                use_ollama INTEGER NOT NULL DEFAULT 0,
                ollama_server_url TEXT NOT NULL DEFAULT 'http://localhost:11434',
                ollama_port INTEGER NOT NULL DEFAULT 11434,
                ollama_model_priority TEXT NOT NULL DEFAULT 'gemma,qwen,llama,deepseek,mistral',
                rag_chunk_size INTEGER NOT NULL DEFAULT 400,
                rag_chunk_overlap INTEGER NOT NULL DEFAULT 50,
                rag_retrieval_k_global INTEGER NOT NULL DEFAULT 2,
                rag_retrieval_k_meeting INTEGER NOT NULL DEFAULT 3,
                rag_retrieval_k_transcript INTEGER NOT NULL DEFAULT 10,
                rag_max_collection_context INTEGER NOT NULL DEFAULT 10,
                rag_relative_score_cutoff REAL NOT NULL DEFAULT 0.01,
                generate_mom_auto INTEGER NOT NULL DEFAULT 1,
                embedding_model TEXT NOT NULL DEFAULT 'Qwen3-Embedding-0.6B',
                ollama_num_ctx INTEGER NOT NULL DEFAULT 32768,
                ollama_dynamic_ctx INTEGER NOT NULL DEFAULT 1,
                ollama_think INTEGER NOT NULL DEFAULT 0,
                ollama_temperature REAL NOT NULL DEFAULT 0.0,
                ollama_top_p REAL NOT NULL DEFAULT 0.9,
                ollama_top_k INTEGER NOT NULL DEFAULT 40,
                ollama_repeat_penalty REAL NOT NULL DEFAULT 1.15,
                ollama_seed INTEGER NOT NULL DEFAULT -1,
                ollama_stop TEXT NOT NULL DEFAULT '',
                ollama_keep_alive TEXT NOT NULL DEFAULT '5m',
                ollama_num_thread INTEGER NOT NULL DEFAULT 0,
                ollama_num_gpu INTEGER NOT NULL DEFAULT -1,
                max_tokens_mom INTEGER NOT NULL DEFAULT 1500,
                max_tokens_mom_merge INTEGER NOT NULL DEFAULT 3072,
                max_tokens_mom_extract_actions INTEGER NOT NULL DEFAULT 4096,
                max_tokens_raw_mom_to_mom INTEGER NOT NULL DEFAULT 3000,
                max_tokens_raw_mom_extraction INTEGER NOT NULL DEFAULT 1024,
                max_tokens_raw_mom_repair INTEGER NOT NULL DEFAULT 1024,
                max_tokens_agenda_compress INTEGER NOT NULL DEFAULT 2000,
                max_tokens_reference_compress INTEGER NOT NULL DEFAULT 2000,
                max_tokens_agenda_from_summary INTEGER NOT NULL DEFAULT 1024,
                max_tokens_executive_summary INTEGER NOT NULL DEFAULT 700,
                max_tokens_short_summary INTEGER NOT NULL DEFAULT 120,
                max_tokens_detailed_summary INTEGER NOT NULL DEFAULT 3000,
                max_tokens_chunk_summary INTEGER NOT NULL DEFAULT 256,
                max_tokens_key_points INTEGER NOT NULL DEFAULT 1028,
                max_tokens_action_items INTEGER NOT NULL DEFAULT 1028,
                max_tokens_key_decisions INTEGER NOT NULL DEFAULT 1028,
                max_tokens_speaker_summary INTEGER NOT NULL DEFAULT 200,
                max_tokens_speaker_key_points INTEGER NOT NULL DEFAULT 350,
                max_tokens_speaker_action_items INTEGER NOT NULL DEFAULT 250,
                max_tokens_collection_chat INTEGER NOT NULL DEFAULT 1500,
                max_tokens_collection_compare INTEGER NOT NULL DEFAULT 1500,
                max_tokens_collection_topic_growth INTEGER NOT NULL DEFAULT 1500,
                max_tokens_vocab_extractor INTEGER NOT NULL DEFAULT 512,
                enable_vad INTEGER NOT NULL DEFAULT 1,
                enable_transcription_vad INTEGER NOT NULL DEFAULT 1,
                enable_alignment_vad INTEGER NOT NULL DEFAULT 1,
                enable_audio_normalization INTEGER NOT NULL DEFAULT 1,
                norm_target_dbfs REAL NOT NULL DEFAULT -3.0,
                norm_compression_ratio REAL NOT NULL DEFAULT 2.0,
                enable_adaptive_vad INTEGER NOT NULL DEFAULT 1,
                vad_speech_threshold REAL NOT NULL DEFAULT 0.15,
                vad_silence_threshold REAL NOT NULL DEFAULT 0.10,
                vad_min_speech_ms INTEGER NOT NULL DEFAULT 250,
                vad_min_silence_ms INTEGER NOT NULL DEFAULT 400,
                enable_speech_padding INTEGER NOT NULL DEFAULT 1,
                speech_pad_ms INTEGER NOT NULL DEFAULT 400,
                enable_speech_segment_merging INTEGER NOT NULL DEFAULT 1,
                max_merge_silence_ms INTEGER NOT NULL DEFAULT 500,
                enable_low_volume_recovery INTEGER NOT NULL DEFAULT 1,
                recovery_energy_threshold REAL NOT NULL DEFAULT -45.0,
                recovery_min_duration_ms INTEGER NOT NULL DEFAULT 300,
                missing_segment_min_duration_sec REAL NOT NULL DEFAULT 2.0,
                whisper_parallel_processing INTEGER NOT NULL DEFAULT 1,
                whisper_parallel_chunk_minutes INTEGER NOT NULL DEFAULT 10,
                rom_action_generation_chunk_size INTEGER NOT NULL DEFAULT 10,
                rom_pipeline_mode TEXT NOT NULL DEFAULT 'base',
                restrict_reassignment_to_meeting_speakers INTEGER NOT NULL DEFAULT 0,
                max_tokens_rom_version_short INTEGER NOT NULL DEFAULT 4096,
                max_tokens_rom_version_medium INTEGER NOT NULL DEFAULT 4096,
                updated_at TEXT NOT NULL
            )

        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS minutes_of_meeting (
                id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                title TEXT,
                date TEXT,
                duration REAL DEFAULT 0,
                planned_start_time TEXT,
                actual_start_time TEXT,
                planned_end_time TEXT,
                actual_end_time TEXT,
                participants TEXT NOT NULL DEFAULT '[]',
                introduction TEXT,
                points_discussed TEXT NOT NULL DEFAULT '[]',
                action_items TEXT NOT NULL DEFAULT '[]',
                conclusion TEXT,
                -- legacy columns kept for existing data / rollback compatibility
                agenda_items TEXT NOT NULL DEFAULT '[]',
                discussion_summary TEXT,
                decisions TEXT NOT NULL DEFAULT '[]',
                risks_concerns TEXT NOT NULL DEFAULT '[]',
                next_steps TEXT NOT NULL DEFAULT '[]',
                next_meeting_date TEXT,
                versions TEXT NOT NULL DEFAULT '[]',
                is_draft INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))

        # ── Add new MOM columns to existing DBs (idempotent) ────────────
        for _col, _def in [
            ("planned_start_time", "TEXT"),
            ("actual_start_time",  "TEXT"),
            ("planned_end_time",   "TEXT"),
            ("actual_end_time",    "TEXT"),
            ("introduction",       "TEXT"),
            ("points_discussed",   "TEXT NOT NULL DEFAULT '[]'"),
            ("conclusion",         "TEXT"),
        ]:
            try:
                await conn.execute(text(
                    f"ALTER TABLE minutes_of_meeting ADD COLUMN {_col} {_def}"
                ))
            except Exception:
                pass  # column already exists


        # ── Global prompt per user (auto-saved) ─────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS global_prompts (
                user_id    TEXT PRIMARY KEY,
                prompt     TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
        """))

        # ── Shortcut dictionary (abbreviation → full form) ────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS shortcut_dictionary (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                shortcut   TEXT NOT NULL,
                full_form  TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))

        # ── Technical vocabulary (domain words for Whisper prompt) ────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS technical_vocabulary (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                word       TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """))

        # Indexes
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_refresh_token ON sessions(refresh_token_hash)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_login_attempts_ip ON login_attempts(ip_address)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_login_attempts_email ON login_attempts(email)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_recordings_user_id ON recordings(user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_voice_profiles_user_id ON voice_profiles(user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_mom_recording_id ON minutes_of_meeting(recording_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_shortcuts_user_id ON shortcut_dictionary(user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_vocab_user_id ON technical_vocabulary(user_id)"))

        # ── stage2_edit_history ──────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stage2_edit_history (
                id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                change_type TEXT NOT NULL,
                point_ids_affected TEXT NOT NULL DEFAULT '[]',
                before_state TEXT NOT NULL DEFAULT '[]',
                after_state TEXT NOT NULL DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                is_reverted INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                reverted_at TEXT
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_stage2_edits_recording ON stage2_edit_history(recording_id)"))

        # ── arena_drafts — Model Arena draft prompt storage ─────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS arena_drafts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                prompt_key TEXT NOT NULL,
                draft_name TEXT NOT NULL DEFAULT '',
                template TEXT NOT NULL,
                rule_states TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_arena_drafts_user ON arena_drafts(user_id)"))

        # ── arena_history — Model Arena generation history storage ──────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS arena_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                prompt_key TEXT NOT NULL,
                meeting_id TEXT NOT NULL,
                recording_name TEXT NOT NULL DEFAULT '',
                model_name TEXT NOT NULL,
                window_index INTEGER,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_arena_history_user_pm ON arena_history(user_id, prompt_key, meeting_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_arena_history_created ON arena_history(created_at DESC)"))

        # ── recording_chunks — per-chunk transcription results ──────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS recording_chunks (
                id           TEXT PRIMARY KEY,
                recording_id TEXT,
                chunk_index  INTEGER NOT NULL,
                chunk_start_sec REAL NOT NULL DEFAULT 0.0,
                chunk_end_sec   REAL NOT NULL DEFAULT 0.0,
                file_path    TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                transcript   TEXT NOT NULL DEFAULT '[]',
                raw_text     TEXT,
                aligned_result TEXT,
                error_message  TEXT,
                created_at   TEXT NOT NULL
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chunks_recording_id ON recording_chunks(recording_id)"
        ))

        # ── Processing Analytics — one row per completed pipeline job ──────────
        # Imported here to avoid circular imports (analytics.py imports database.py)
        from services.analytics import CREATE_TABLE_SQL, ADD_INDEX_SQL, ADD_INDEX_COMPLETED_SQL
        await conn.execute(text(CREATE_TABLE_SQL))
        await conn.execute(text(ADD_INDEX_SQL))
        await conn.execute(text(ADD_INDEX_COMPLETED_SQL))

        # ── Migration: add short_summary / detailed_summary columns if missing ──
        for col in ("short_summary", "detailed_summary"):
            try:
                await conn.execute(text(f"ALTER TABLE recordings ADD COLUMN {col} TEXT"))
            except Exception:
                pass  # column already exists

        # ── Migration: add advanced-options columns to recordings if missing ──
        for col_def in (
            "title TEXT DEFAULT NULL",
            "meeting_prompt TEXT DEFAULT ''",
            "participant_voice_ids TEXT DEFAULT '[]'",
            "use_vocabulary INTEGER DEFAULT 0",
            "speaker_summary TEXT DEFAULT NULL",
        ):
            try:
                await conn.execute(text(f"ALTER TABLE recordings ADD COLUMN {col_def}"))
            except Exception:
                pass  # column already exists

        # ── Migration: add Ollama & RAG columns to user_settings if missing ──
        for col_name, col_type in [
            ("use_ollama", "INTEGER NOT NULL DEFAULT 0"),
            ("ollama_server_url", "TEXT NOT NULL DEFAULT 'http://localhost:11434'"),
            ("ollama_port", "INTEGER NOT NULL DEFAULT 11434"),
            ("ollama_model_priority", "TEXT NOT NULL DEFAULT 'gemma,qwen,llama,deepseek,mistral'"),
            ("rag_chunk_size", "INTEGER NOT NULL DEFAULT 400"),
            ("rag_chunk_overlap", "INTEGER NOT NULL DEFAULT 50"),
            ("rag_retrieval_k_global", "INTEGER NOT NULL DEFAULT 2"),
            ("rag_retrieval_k_meeting", "INTEGER NOT NULL DEFAULT 3"),
            ("rag_retrieval_k_transcript", "INTEGER NOT NULL DEFAULT 10"),
            ("rag_max_collection_context", "INTEGER NOT NULL DEFAULT 10"),
            ("rag_relative_score_cutoff", "REAL NOT NULL DEFAULT 0.01"),
            ("generate_mom_auto", "INTEGER NOT NULL DEFAULT 1"),
            ("ollama_num_ctx", "INTEGER NOT NULL DEFAULT 32768"),
            ("ollama_dynamic_ctx", "INTEGER NOT NULL DEFAULT 1"),
            ("ollama_think", "INTEGER NOT NULL DEFAULT 0"),
            ("ollama_temperature", "REAL NOT NULL DEFAULT 0.0"),
            ("ollama_top_p", "REAL NOT NULL DEFAULT 0.9"),
            ("ollama_top_k", "INTEGER NOT NULL DEFAULT 40"),
            ("ollama_repeat_penalty", "REAL NOT NULL DEFAULT 1.15"),
            ("ollama_seed", "INTEGER NOT NULL DEFAULT -1"),
            ("ollama_stop", "TEXT NOT NULL DEFAULT ''"),
            ("ollama_keep_alive", "TEXT NOT NULL DEFAULT '5m'"),
            ("ollama_num_thread", "INTEGER NOT NULL DEFAULT 0"),
            ("ollama_num_gpu", "INTEGER NOT NULL DEFAULT -1"),
            ("max_tokens_mom", "INTEGER NOT NULL DEFAULT 1500"),
            ("max_tokens_mom_merge", "INTEGER NOT NULL DEFAULT 3072"),
            ("max_tokens_raw_mom_to_mom", "INTEGER NOT NULL DEFAULT 3000"),
            ("max_tokens_raw_mom_extraction", "INTEGER NOT NULL DEFAULT 1024"),
            ("max_tokens_raw_mom_repair", "INTEGER NOT NULL DEFAULT 1024"),
            ("max_tokens_agenda_compress", "INTEGER NOT NULL DEFAULT 2000"),
            ("max_tokens_reference_compress", "INTEGER NOT NULL DEFAULT 2000"),
            ("max_tokens_agenda_from_summary", "INTEGER NOT NULL DEFAULT 1024"),
            ("max_tokens_executive_summary", "INTEGER NOT NULL DEFAULT 700"),
            ("max_tokens_short_summary", "INTEGER NOT NULL DEFAULT 120"),
            ("max_tokens_detailed_summary", "INTEGER NOT NULL DEFAULT 3000"),
            ("max_tokens_chunk_summary", "INTEGER NOT NULL DEFAULT 256"),
            ("max_tokens_key_points", "INTEGER NOT NULL DEFAULT 1028"),
            ("max_tokens_action_items", "INTEGER NOT NULL DEFAULT 1028"),
            ("max_tokens_key_decisions", "INTEGER NOT NULL DEFAULT 1028"),
            ("max_tokens_speaker_summary", "INTEGER NOT NULL DEFAULT 200"),
            ("max_tokens_speaker_key_points", "INTEGER NOT NULL DEFAULT 350"),
            ("max_tokens_speaker_action_items", "INTEGER NOT NULL DEFAULT 250"),
            ("max_tokens_collection_chat", "INTEGER NOT NULL DEFAULT 1500"),
            ("max_tokens_collection_compare", "INTEGER NOT NULL DEFAULT 1500"),
            ("max_tokens_collection_topic_growth", "INTEGER NOT NULL DEFAULT 1500"),
            ("max_tokens_vocab_extractor", "INTEGER NOT NULL DEFAULT 512"),
            ("embedding_model", "TEXT NOT NULL DEFAULT 'Qwen3-Embedding-0.6B'"),
            ("restrict_reassignment_to_meeting_speakers", "INTEGER NOT NULL DEFAULT 0"),
            ("max_tokens_rom_version_short", "INTEGER NOT NULL DEFAULT 4096"),
            ("max_tokens_rom_version_medium", "INTEGER NOT NULL DEFAULT 4096"),
        ]:
            try:
                await conn.execute(text(f"ALTER TABLE user_settings ADD COLUMN {col_name} {col_type}"))
            except Exception:
                pass  # column already exists

        for col_def in (
            "chunk_ids TEXT DEFAULT '[]'",
            "is_chunked INTEGER NOT NULL DEFAULT 0",
            "speaker_mappings TEXT DEFAULT '{}'",
        ):
            try:
                await conn.execute(text(f"ALTER TABLE recordings ADD COLUMN {col_def}"))
            except Exception:
                pass  # column already exists

        # ── Migration: add context_summary caching columns to recordings ─────
        # context_summary       — compressed hierarchical summary used by all AI tasks.
        # context_summary_hash  — MD5 of raw_text at time of generation; used to detect
        #                         stale context when the transcript is edited.
        for col_def in (
            "context_summary TEXT DEFAULT NULL",
            "context_summary_hash TEXT DEFAULT NULL",
        ):
            try:
                await conn.execute(text(f"ALTER TABLE recordings ADD COLUMN {col_def}"))
            except Exception:
                pass  # column already exists

        # ── Migration: add chunk_summary to recording_chunks ─────────────────
        # Stores the compressed LLM summary of each 10-minute chunk so the
        # finalize pipeline can merge them without re-reading the full transcript.
        try:
            await conn.execute(text(
                "ALTER TABLE recording_chunks ADD COLUMN chunk_summary TEXT DEFAULT NULL"
            ))
        except Exception:
            pass  # column already exists

        # ── Migration: add speaker_reid_at to recordings ──────────────────────
        # Timestamp of the most recent "Re-run Speaker Identification" operation.
        try:
            await conn.execute(text(
                "ALTER TABLE recordings ADD COLUMN speaker_reid_at TEXT DEFAULT NULL"
            ))
        except Exception:
            pass  # column already exists

        # ── Migration: Add Agenda/Context summary columns to recordings table ─
        for col_def in (
            "agenda_summary TEXT DEFAULT NULL",
            "agenda_summary_hash TEXT DEFAULT NULL",
            "reference_summary TEXT DEFAULT NULL",
            "reference_summary_hash TEXT DEFAULT NULL",
            "parsed_agenda_json TEXT DEFAULT NULL",
        ):
            try:
                await conn.execute(text(f"ALTER TABLE recordings ADD COLUMN {col_def}"))
            except Exception:
                pass  # column already exists

        # ── Migration: Create recording_attachments table ─────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS recording_attachments (
                id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                type TEXT NOT NULL, -- 'agenda' or 'context'
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_attachments_recording ON recording_attachments(recording_id)"
        ))

        # ── Global Context Documents — org-wide knowledge base ──────────────────
        # Uploaded once by the user; chunks are embedded into a per-user FAISS index
        # and retrieved during Raw MoM generation for any meeting.
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS global_context_documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                relative_path TEXT DEFAULT '',
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                embedded INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_global_ctx_user ON global_context_documents(user_id)"
        ))

        try:
            await conn.execute(text("ALTER TABLE global_context_documents ADD COLUMN relative_path TEXT DEFAULT ''"))
        except Exception:
            pass  # column already exists

        # ── RAG pipeline columns on recordings (idempotent migrations) ───────────
        # raw_mom                   — JSON output of the Raw MoM pipeline
        # transcript_embedded       — 1 if transcript chunks have been embedded into FAISS
        # meeting_context_embedded  — 1 if meeting context attachments have been embedded
        for col_def in (
            "raw_mom TEXT DEFAULT NULL",
            "transcript_embedded INTEGER NOT NULL DEFAULT 0",
            "meeting_context_embedded INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                await conn.execute(text(f"ALTER TABLE recordings ADD COLUMN {col_def}"))
            except Exception:
                pass  # column already exists

        # ── Migration: Add ROM (Record of Meeting) data column ────────────────
        try:
            await conn.execute(text("ALTER TABLE recordings ADD COLUMN rom_data TEXT DEFAULT NULL"))
        except Exception:
            pass  # column already exists

        # ── ROM Metadata table & migration (for standalone ROM queries) ───────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rom_metadata (
                id              TEXT PRIMARY KEY,
                recording_id    TEXT NOT NULL UNIQUE,
                user_id         TEXT NOT NULL,
                rom_data        TEXT DEFAULT NULL,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_rom_metadata_recording_id ON rom_metadata(recording_id)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_rom_metadata_user_id ON rom_metadata(user_id)"
        ))

        # Backfill rom_metadata from existing recordings.rom_data for existing databases
        try:
            await conn.execute(text("""
                INSERT OR IGNORE INTO rom_metadata (id, recording_id, user_id, rom_data, created_at, updated_at)
                SELECT id, id, user_id, rom_data, created_at, created_at
                FROM recordings
                WHERE rom_data IS NOT NULL AND rom_data != '' AND rom_data != '{}'
            """))
        except Exception as bf_err:
            logger.warning(f"[DB] rom_metadata backfill check (non-fatal): {bf_err}")

        # ── Migration: Add Video support columns ──────────────────────────────
        # video_transcript — JSON array of merged OCR blocks: [{start, end, text}]
        #                    populated asynchronously after video upload; NULL for audio-only recordings.
        # source_type      — 'audio' (default) or 'video'; used to activate video OCR context in ROM Stage 1.
        for col_def in (
            "video_transcript TEXT DEFAULT NULL",
            "source_type TEXT NOT NULL DEFAULT 'audio'",
        ):
            try:
                await conn.execute(text(f"ALTER TABLE recordings ADD COLUMN {col_def}"))
            except Exception:
                pass  # column already exists


        # ── Migration: Add ROM settings columns to user_settings ─────────────
        for col_name, col_type in [
            ("rom_transcript_window", "REAL NOT NULL DEFAULT 2.0"),
            ("rom_meeting_top_k", "INTEGER NOT NULL DEFAULT 5"),
            ("rom_global_top_k", "INTEGER NOT NULL DEFAULT 3"),
            ("rom_windows_per_batch", "INTEGER NOT NULL DEFAULT 5"),
            ("rom_parallel_window_processing", "INTEGER NOT NULL DEFAULT 2"),
            ("rom_separate_action_extraction", "INTEGER NOT NULL DEFAULT 0"),
            ("rom_action_generation_chunk_size", "INTEGER NOT NULL DEFAULT 10"),
            ("rom_stage2_process_all_together", "INTEGER NOT NULL DEFAULT 0"),
            ("rom_min_similarity_threshold", "REAL DEFAULT 0.80"),
            ("rom_pipeline_mode", "TEXT NOT NULL DEFAULT 'base'"),
            ("whisper_batch_size", "INTEGER NOT NULL DEFAULT 8"),
            ("max_tokens_rom_discussion", "INTEGER NOT NULL DEFAULT 4096"),
            ("max_tokens_rom_discussion_no_actions", "INTEGER NOT NULL DEFAULT 4096"),
            ("max_tokens_rom_action_extraction", "INTEGER NOT NULL DEFAULT 2048"),
            ("max_tokens_mom_extract_actions", "INTEGER NOT NULL DEFAULT 4096"),
            ("max_tokens_stage1_json_repair", "INTEGER NOT NULL DEFAULT 4548"),
            ("max_tokens_mom_action_regen", "INTEGER NOT NULL DEFAULT 4048"),
            ("max_tokens_rom_polish", "INTEGER NOT NULL DEFAULT 4096"),
            ("max_tokens_rom_enhance_window", "INTEGER NOT NULL DEFAULT 4096"),
            ("max_tokens_rom_deduplicate", "INTEGER NOT NULL DEFAULT 2048"),
            ("max_tokens_rom_agenda", "INTEGER NOT NULL DEFAULT 2048"),
            ("max_tokens_rom_mom_expansion", "INTEGER NOT NULL DEFAULT 3000"),
            ("max_tokens_rom_agenda_assign_batch", "INTEGER NOT NULL DEFAULT 4096"),
            ("max_tokens_rom_agenda_doc_points", "INTEGER NOT NULL DEFAULT 1024"),
        ]:
            try:
                await conn.execute(text(f"ALTER TABLE user_settings ADD COLUMN {col_name} {col_type}"))
            except Exception:
                pass  # column already exists


        # ── Migration: Add Low-Volume Speech Transcription columns to user_settings ─
        for col_name, col_type in [
            ("enable_vad", "INTEGER NOT NULL DEFAULT 1"),
            ("enable_transcription_vad", "INTEGER NOT NULL DEFAULT 1"),
            ("enable_alignment_vad", "INTEGER NOT NULL DEFAULT 1"),
            ("enable_audio_normalization", "INTEGER NOT NULL DEFAULT 1"),
            ("norm_target_dbfs", "REAL NOT NULL DEFAULT -3.0"),
            ("norm_compression_ratio", "REAL NOT NULL DEFAULT 2.0"),
            ("enable_adaptive_vad", "INTEGER NOT NULL DEFAULT 1"),
            ("vad_speech_threshold", "REAL NOT NULL DEFAULT 0.15"),
            ("vad_silence_threshold", "REAL NOT NULL DEFAULT 0.10"),
            ("vad_min_speech_ms", "INTEGER NOT NULL DEFAULT 250"),
            ("vad_min_silence_ms", "INTEGER NOT NULL DEFAULT 400"),
            ("enable_speech_padding", "INTEGER NOT NULL DEFAULT 1"),
            ("speech_pad_ms", "INTEGER NOT NULL DEFAULT 400"),
            ("enable_speech_segment_merging", "INTEGER NOT NULL DEFAULT 1"),
            ("max_merge_silence_ms", "INTEGER NOT NULL DEFAULT 500"),
            ("enable_low_volume_recovery", "INTEGER NOT NULL DEFAULT 1"),
            ("recovery_energy_threshold", "REAL NOT NULL DEFAULT -45.0"),
            ("recovery_min_duration_ms", "INTEGER NOT NULL DEFAULT 300"),
            ("enable_audio_validation", "INTEGER NOT NULL DEFAULT 1"),
            ("min_audio_duration_seconds", "REAL NOT NULL DEFAULT 2.0"),
            ("min_audio_rms_threshold", "REAL NOT NULL DEFAULT 0.003"),
            ("whisper_parallel_processing", "INTEGER NOT NULL DEFAULT 1"),
            ("whisper_parallel_chunk_minutes", "INTEGER NOT NULL DEFAULT 10"),
            ("parallel_transcription_diarization", "INTEGER NOT NULL DEFAULT 0"),
        ]:
            try:
                await conn.execute(text(f"ALTER TABLE user_settings ADD COLUMN {col_name} {col_type}"))
            except Exception:
                pass  # column already exists


        # ── Migration: Missing Transcription Recovery ────────────────────────────
        # raw_transcript — stores JSON of aligned transcript segments after whisperx,
        #                  before diarization; used for the optional pipeline-pause
        #                  Missing Transcription Recovery feature.
        try:
            await conn.execute(text("ALTER TABLE recordings ADD COLUMN raw_transcript TEXT DEFAULT NULL"))
        except Exception:
            pass  # column already exists

        # missing_transcript_recovery_enabled — user toggle for missing transcript recovery
        try:
            await conn.execute(text(
                "ALTER TABLE user_settings ADD COLUMN missing_transcript_recovery_enabled "
                "INTEGER NOT NULL DEFAULT 0"
            ))
        except Exception:
            pass  # column already exists

        # missing_segment_min_duration_sec — minimum segment length threshold for missing speech recovery
        try:
            await conn.execute(text(
                "ALTER TABLE user_settings ADD COLUMN missing_segment_min_duration_sec "
                "REAL NOT NULL DEFAULT 2.0"
            ))
        except Exception:
            pass  # column already exists

        # ── Migration: Add multi-model speaker embedding columns to voice_profiles ──
        # audio_paths         — JSON list of saved audio WAV file paths for this profile
        # recordings          — JSON list of recording objects: [{id, filename, file_path, duration, created_at, metadata}]
        # ecapa_embeddings    — JSON list of ECAPA-TDNN 192-d embeddings (dedicated column)
        # eres2net_embeddings — JSON list of ERes2Net-Large 512-d embeddings
        for col_def in (
            "audio_paths TEXT DEFAULT '[]'",
            "recordings TEXT DEFAULT '[]'",
            "ecapa_embeddings TEXT DEFAULT NULL",
            "eres2net_embeddings TEXT DEFAULT NULL",
        ):
            try:
                await conn.execute(text(f"ALTER TABLE voice_profiles ADD COLUMN {col_def}"))
            except Exception:
                pass  # column already exists

        # Backfill: copy existing embeddings → ecapa_embeddings for profiles that
        # have embeddings but no ecapa_embeddings yet (one-time migration).
        try:
            await conn.execute(text(
                "UPDATE voice_profiles SET ecapa_embeddings = embeddings "
                "WHERE ecapa_embeddings IS NULL AND embeddings IS NOT NULL AND embeddings != '[]'"
            ))
        except Exception:
            pass  # safe to skip

        # speaker_embedding_model — active embedding model preference per user
        try:
            await conn.execute(text(
                "ALTER TABLE user_settings ADD COLUMN speaker_embedding_model "
                "TEXT NOT NULL DEFAULT 'ecapa'"
            ))
        except Exception:
            pass  # column already exists


        # ── Prompt Templates — system-wide, shared by all users ─────────────────
        # Stores custom overrides for every AI prompt used in the application.
        # When a key is absent the backend falls back to the hardcoded default.
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prompt_templates (
                key        TEXT PRIMARY KEY,
                template   TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))

        # ── Meeting Collections — organizational folders for recordings ────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS meeting_collections (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                name          TEXT NOT NULL,
                description   TEXT DEFAULT '',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                UNIQUE(user_id, name)
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_collections_user ON meeting_collections(user_id)"
        ))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS meeting_collection_items (
                id              TEXT PRIMARY KEY,
                collection_id   TEXT NOT NULL,
                meeting_id      TEXT NOT NULL,
                display_order   INTEGER NOT NULL DEFAULT 0,
                added_at        TEXT NOT NULL,
                UNIQUE(collection_id, meeting_id)
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_collection_items_coll ON meeting_collection_items(collection_id)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_collection_items_meeting ON meeting_collection_items(meeting_id)"
        ))

        # ── Collection AI Chat Messages — per-collection chat history ──────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS collection_chat_messages (
                id              TEXT PRIMARY KEY,
                collection_id   TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                message_type    TEXT NOT NULL DEFAULT 'chat',
                metadata        TEXT DEFAULT '{}',
                created_at      TEXT NOT NULL
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chat_collection ON collection_chat_messages(collection_id)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chat_user ON collection_chat_messages(user_id)"
        ))

        # ── Standalone AI Chat Messages ─────────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ai_chat_messages (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                metadata    TEXT DEFAULT '{}',
                created_at  TEXT NOT NULL
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_ai_chat_user ON ai_chat_messages(user_id)"
        ))


        # ── Training & Optimization Tables ────────────────────────────────────────
        # These tables support the Training & Optimization feature module.
        # The existing pipeline tables are NOT modified by this feature.
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id          TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                description     TEXT,
                status          TEXT NOT NULL DEFAULT 'pending',
                stage_configs   TEXT NOT NULL DEFAULT '[]',
                dataset_id      TEXT,
                created_at      TEXT NOT NULL,
                started_at      TEXT,
                completed_at    TEXT,
                error           TEXT,
                logs            TEXT NOT NULL DEFAULT '[]',
                progress        TEXT,
                artifacts       TEXT NOT NULL DEFAULT '[]',
                eval_results    TEXT NOT NULL DEFAULT '{}'
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_training_jobs_user ON training_jobs(user_id)"
        ))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS training_datasets (
                dataset_id          TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                stages              TEXT NOT NULL DEFAULT '[]',
                source_type         TEXT NOT NULL DEFAULT 'meeting',
                source_meeting_id   TEXT,
                total_samples       INTEGER NOT NULL DEFAULT 0,
                samples_by_stage    TEXT NOT NULL DEFAULT '{}',
                manual_mom          TEXT,
                created_at          TEXT NOT NULL
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_training_datasets_user ON training_datasets(user_id)"
        ))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rom_training_data (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                recording_title TEXT,
                agenda_id TEXT NOT NULL,
                agenda_title TEXT NOT NULL,
                long_rom_points TEXT NOT NULL DEFAULT '[]',
                manual_mom_points TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rom_training_variants (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                base_model TEXT NOT NULL,
                training_type TEXT NOT NULL DEFAULT 'new',
                parent_variant_id TEXT,
                adapter_path TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                config TEXT NOT NULL DEFAULT '{}',
                dataset_size INTEGER NOT NULL DEFAULT 0,
                test_size INTEGER NOT NULL DEFAULT 0,
                test_results TEXT NOT NULL DEFAULT '[]',
                accuracy TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """))

        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_rom_training_data_user ON rom_training_data(user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_rom_training_data_recording ON rom_training_data(recording_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_rom_training_variants_user ON rom_training_variants(user_id)"))

        for col_name, col_type in [
            ("rom_short_model_mode", "TEXT NOT NULL DEFAULT 'base'"),
            ("rom_short_model_variant_id", "TEXT"),
        ]:
            try:
                await conn.execute(text(f"ALTER TABLE user_settings ADD COLUMN {col_name} {col_type}"))
            except Exception:
                pass

        # Fixes users who have an old 30-day cookie that has already expired.
        # Sessions are only revoked by explicit logout, never by time expiry.
        try:
            await conn.execute(text(
                "UPDATE sessions SET expires_at = '2125-01-01T00:00:00+00:00' "
                "WHERE is_revoked = 0"
            ))
            logger.info("[DB] Extended all active sessions to year 2125.")
        except Exception as ext_err:
            logger.warning(f"[DB] Could not extend sessions (non-fatal): {ext_err}")

        # ── Correction & Acronym Validation Tables ────────────────────────
        # transcript_corrections — LLM-proposed ASR corrections + user decisions per recording
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS transcript_corrections (
                id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                proposed_corrections TEXT NOT NULL DEFAULT '[]',
                user_decisions TEXT NOT NULL DEFAULT '[]',
                applied INTEGER NOT NULL DEFAULT 0,
                original_transcript TEXT DEFAULT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_transcript_corrections_recording ON transcript_corrections(recording_id)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_transcript_corrections_user ON transcript_corrections(user_id)"
        ))

        # transcript_acronyms — detected acronyms + user decisions per recording
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS transcript_acronyms (
                id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                detected_acronyms TEXT NOT NULL DEFAULT '[]',
                user_decisions TEXT NOT NULL DEFAULT '[]',
                applied INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_transcript_acronyms_recording ON transcript_acronyms(recording_id)"
        ))

        # acronym_dictionary — persistent per-user acronym → full-form dictionary
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS acronym_dictionary (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                acronym TEXT NOT NULL,
                full_form TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, acronym)
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_acronym_dict_user ON acronym_dictionary(user_id)"
        ))

        # ── Embedding Model Variants ─────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS embedding_variants (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                name            TEXT NOT NULL UNIQUE,
                base_model      TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                document_count  INTEGER NOT NULL DEFAULT 0,
                document_names  TEXT NOT NULL DEFAULT '[]',
                chunk_count     INTEGER NOT NULL DEFAULT 0,
                status          TEXT NOT NULL DEFAULT 'completed',
                model_path      TEXT NOT NULL,
                size_bytes      INTEGER NOT NULL DEFAULT 0,
                metrics         TEXT NOT NULL DEFAULT '{}',
                is_active       INTEGER NOT NULL DEFAULT 0
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_embedding_variants_user ON embedding_variants(user_id)"
        ))


        # ── Verification: ensure core and newly added tables exist ─────────
        verification = await conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('recordings', 'rom_metadata', 'user_settings', 'minutes_of_meeting')"
        ))
        verified_tables = {row[0] for row in verification.fetchall()}
        logger.info(f"[DB] Verified core tables present: {verified_tables}")
        if 'rom_metadata' not in verified_tables:
            logger.error("[DB] CRITICAL: rom_metadata table was not found after initialization!")

    logger.info(f"[DB] SQLite database ready: {settings.DATABASE_URL}")


async def close_db():
    global engine
    if engine:
        await engine.dispose()
        logger.info("[DB] SQLite connection closed.")


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager yielding an AsyncSession.
    Guarantees session rollback on error and proper session closure on exit.
    Usage: `async with get_db_context() as db:`
    """
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding an AsyncSession.
    FastAPI automatically handles entering and exiting this async generator per request,
    guaranteeing connection checkin even on error or early return.
    """
    async with get_db_context() as session:
        yield session


# ── JSON helpers ──────────────────────────────────────────────────────────────

def to_json(value: Any) -> str:
    """Serialize a Python object to a JSON string for storage."""
    return json.dumps(value, default=str)


def from_json(value: str | None, default=None):
    """Deserialize a JSON string from DB. Returns default if None/empty or parsed value is None."""
    if value is None or not str(value).strip():
        return default
    try:
        res = json.loads(value)
        return default if res is None else res
    except (json.JSONDecodeError, TypeError):
        return default



def dt_to_str(dt: datetime | None) -> str | None:
    """Convert datetime to ISO string for SQLite storage."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def str_to_dt(s: str | None) -> datetime | None:
    """Parse an ISO datetime string from SQLite into a timezone-aware datetime."""
    if s is None:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None
