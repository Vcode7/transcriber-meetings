"""
Dictionary Service

Async CRUD helpers for shortcut_dictionary and technical_vocabulary tables.
All functions accept an AsyncSession and return plain dicts for JSON serialisation.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from database import dt_to_str

logger = logging.getLogger(__name__)


# ── Shortcuts ─────────────────────────────────────────────────────────────────

async def list_shortcuts(db: AsyncSession, user_id: str) -> List[dict]:
    r = await db.execute(
        text("SELECT * FROM shortcut_dictionary WHERE user_id = :uid ORDER BY shortcut ASC"),
        {"uid": user_id},
    )
    return [dict(row) for row in r.mappings().fetchall()]


async def create_shortcut(
    db: AsyncSession,
    user_id: str,
    shortcut: str,
    full_form: str,
) -> dict:
    sid = str(uuid.uuid4())
    now = dt_to_str(datetime.now(timezone.utc))
    await db.execute(
        text("""
            INSERT INTO shortcut_dictionary (id, user_id, shortcut, full_form, created_at, updated_at)
            VALUES (:id, :uid, :shortcut, :full_form, :now, :now)
        """),
        {"id": sid, "uid": user_id, "shortcut": shortcut.strip(), "full_form": full_form.strip(), "now": now},
    )
    await db.commit()
    return {"id": sid, "user_id": user_id, "shortcut": shortcut.strip(),
            "full_form": full_form.strip(), "created_at": now, "updated_at": now}


async def update_shortcut(
    db: AsyncSession,
    shortcut_id: str,
    user_id: str,
    shortcut: Optional[str] = None,
    full_form: Optional[str] = None,
) -> Optional[dict]:
    now = dt_to_str(datetime.now(timezone.utc))
    parts = ["updated_at = :now"]
    params: dict = {"id": shortcut_id, "uid": user_id, "now": now}
    if shortcut is not None:
        parts.append("shortcut = :shortcut")
        params["shortcut"] = shortcut.strip()
    if full_form is not None:
        parts.append("full_form = :full_form")
        params["full_form"] = full_form.strip()
    set_clause = ", ".join(parts)
    r = await db.execute(
        text(f"UPDATE shortcut_dictionary SET {set_clause} WHERE id = :id AND user_id = :uid"),
        params,
    )
    await db.commit()
    if r.rowcount == 0:
        return None
    r2 = await db.execute(
        text("SELECT * FROM shortcut_dictionary WHERE id = :id"),
        {"id": shortcut_id},
    )
    row = r2.mappings().fetchone()
    return dict(row) if row else None


async def delete_shortcut(db: AsyncSession, shortcut_id: str, user_id: str) -> bool:
    r = await db.execute(
        text("DELETE FROM shortcut_dictionary WHERE id = :id AND user_id = :uid"),
        {"id": shortcut_id, "uid": user_id},
    )
    await db.commit()
    return r.rowcount > 0


async def bulk_create_shortcuts(
    db: AsyncSession,
    user_id: str,
    entries: List[dict],
) -> int:
    """Bulk insert shortcuts, skipping duplicates (same shortcut per user)."""
    now = dt_to_str(datetime.now(timezone.utc))
    count = 0
    for entry in entries:
        shortcut = str(entry.get("short") or entry.get("shortcut", "")).strip()
        full_form = str(entry.get("full") or entry.get("full_form", "")).strip()
        if not shortcut or not full_form:
            continue
        # Skip if already exists
        ex = await db.execute(
            text("SELECT id FROM shortcut_dictionary WHERE user_id = :uid AND shortcut = :s"),
            {"uid": user_id, "s": shortcut},
        )
        if ex.fetchone():
            continue
        await db.execute(
            text("""
                INSERT INTO shortcut_dictionary (id, user_id, shortcut, full_form, created_at, updated_at)
                VALUES (:id, :uid, :s, :f, :now, :now)
            """),
            {"id": str(uuid.uuid4()), "uid": user_id, "s": shortcut, "f": full_form, "now": now},
        )
        count += 1
    await db.commit()
    return count


# ── Technical Vocabulary ──────────────────────────────────────────────────────

async def list_vocabulary(db: AsyncSession, user_id: str) -> List[dict]:
    r = await db.execute(
        text("SELECT * FROM technical_vocabulary WHERE user_id = :uid ORDER BY word ASC"),
        {"uid": user_id},
    )
    return [dict(row) for row in r.mappings().fetchall()]


async def create_vocab_word(db: AsyncSession, user_id: str, word: str) -> Optional[dict]:
    word = word.strip()
    if not word:
        return None
    # Duplicate check
    ex = await db.execute(
        text("SELECT id FROM technical_vocabulary WHERE user_id = :uid AND LOWER(word) = LOWER(:w)"),
        {"uid": user_id, "w": word},
    )
    if ex.fetchone():
        return None  # already exists
    vid = str(uuid.uuid4())
    now = dt_to_str(datetime.now(timezone.utc))
    await db.execute(
        text("INSERT INTO technical_vocabulary (id, user_id, word, created_at) VALUES (:id, :uid, :w, :now)"),
        {"id": vid, "uid": user_id, "w": word, "now": now},
    )
    await db.commit()
    return {"id": vid, "user_id": user_id, "word": word, "created_at": now}


async def delete_vocab_word(db: AsyncSession, vocab_id: str, user_id: str) -> bool:
    r = await db.execute(
        text("DELETE FROM technical_vocabulary WHERE id = :id AND user_id = :uid"),
        {"id": vocab_id, "uid": user_id},
    )
    await db.commit()
    return r.rowcount > 0


async def bulk_create_vocabulary(
    db: AsyncSession,
    user_id: str,
    words: List[str],
) -> int:
    """Bulk insert vocabulary words, skipping duplicates (case-insensitive)."""
    now = dt_to_str(datetime.now(timezone.utc))
    count = 0
    for word in words:
        word = word.strip()
        if not word:
            continue
        ex = await db.execute(
            text("SELECT id FROM technical_vocabulary WHERE user_id = :uid AND LOWER(word) = LOWER(:w)"),
            {"uid": user_id, "w": word},
        )
        if ex.fetchone():
            continue
        await db.execute(
            text("INSERT INTO technical_vocabulary (id, user_id, word, created_at) VALUES (:id, :uid, :w, :now)"),
            {"id": str(uuid.uuid4()), "uid": user_id, "w": word, "now": now},
        )
        count += 1
    await db.commit()
    return count


# ── Global Prompt ─────────────────────────────────────────────────────────────

async def get_global_prompt(db: AsyncSession, user_id: str) -> str:
    r = await db.execute(
        text("SELECT prompt FROM global_prompts WHERE user_id = :uid"),
        {"uid": user_id},
    )
    row = r.fetchone()
    return row[0] if row else ""


async def save_global_prompt(db: AsyncSession, user_id: str, prompt: str) -> None:
    now = dt_to_str(datetime.now(timezone.utc))
    r = await db.execute(
        text("UPDATE global_prompts SET prompt = :p, updated_at = :now WHERE user_id = :uid"),
        {"p": prompt, "now": now, "uid": user_id},
    )
    if r.rowcount == 0:
        await db.execute(
            text("INSERT INTO global_prompts (user_id, prompt, updated_at) VALUES (:uid, :p, :now)"),
            {"uid": user_id, "p": prompt, "now": now},
        )
    await db.commit()
