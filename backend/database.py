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
            col_name = col_def.split()[0]
            try:
                await conn.execute(text(f"ALTER TABLE recordings ADD COLUMN {col_def}"))
            except Exception:
                pass  # column already exists

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
