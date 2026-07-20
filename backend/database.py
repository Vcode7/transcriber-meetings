"""
Database layer — SQLite via aiosqlite + SQLAlchemy async.

Replaces the previous MongoDB/Motor implementation.
All nested JSON data (transcript, embeddings, etc.) is stored as JSON strings.
IDs are UUID strings throughout.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any

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
                rag_relative_score_cutoff REAL NOT NULL DEFAULT 0.01,
                generate_mom_auto INTEGER NOT NULL DEFAULT 1,
                ollama_num_ctx INTEGER NOT NULL DEFAULT 32768,
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
            ("rag_relative_score_cutoff", "REAL NOT NULL DEFAULT 0.01"),
            ("generate_mom_auto", "INTEGER NOT NULL DEFAULT 1"),
            ("ollama_num_ctx", "INTEGER NOT NULL DEFAULT 32768"),
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
        ]:
            try:
                await conn.execute(text(f"ALTER TABLE user_settings ADD COLUMN {col_name} {col_type}"))
            except Exception:
                pass  # column already exists

        for col_def in (
            "chunk_ids TEXT DEFAULT '[]'",
            "is_chunked INTEGER NOT NULL DEFAULT 0",
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


    logger.info(f"[DB] SQLite database ready: {settings.DATABASE_URL}")


async def close_db():
    global engine
    if engine:
        await engine.dispose()
        logger.info("[DB] SQLite connection closed.")


def get_db():
    """Return a new AsyncSession as an async context manager."""
    return AsyncSessionLocal()


# ── JSON helpers ──────────────────────────────────────────────────────────────

def to_json(value: Any) -> str:
    """Serialize a Python object to a JSON string for storage."""
    return json.dumps(value, default=str)


def from_json(value: str | None, default=None):
    """Deserialize a JSON string from DB. Returns default if None/empty."""
    if value is None:
        return default
    try:
        return json.loads(value)
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
