"""Settings router — get/update user threshold settings."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text

from database import get_db, dt_to_str
from routers.auth import get_current_user
from models.settings import UserSettingsUpdate

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("")
async def get_settings(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM user_settings WHERE user_id = :uid"),
            {"uid": user_id},
        )
        doc = r.mappings().fetchone()

    if not doc:
        return {
            "speaker_similarity_threshold": 0.75,
            "word_conf_low": 0.7,
            "word_conf_mid": 0.85,
            "min_segment_duration": 1.5,
        }
    return {
        "speaker_similarity_threshold": doc.get("speaker_similarity_threshold", 0.75),
        "word_conf_low": doc.get("word_conf_low", 0.7),
        "word_conf_mid": doc.get("word_conf_mid", 0.85),
        "min_segment_duration": doc.get("min_segment_duration", 1.5),
        "updated_at": doc.get("updated_at"),
    }


@router.put("")
async def update_settings(
    body: UserSettingsUpdate,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]
    now = datetime.now(timezone.utc)
    patch = {k: v for k, v in body.model_dump().items() if v is not None}

    if not patch:
        return {"message": "Nothing to update."}

    # Build SET clause dynamically from provided fields
    set_parts = [f"{k} = :{k}" for k in patch]
    set_parts.append("updated_at = :updated_at")
    set_clause = ", ".join(set_parts)

    params = {**patch, "user_id": user_id, "updated_at": dt_to_str(now)}

    async with get_db() as db:
        # Try update first
        r = await db.execute(
            text(f"UPDATE user_settings SET {set_clause} WHERE user_id = :user_id"),
            params,
        )
        if r.rowcount == 0:
            # Row doesn't exist yet — insert with defaults
            await db.execute(
                text("""
                    INSERT INTO user_settings (user_id, speaker_similarity_threshold, word_conf_low,
                        word_conf_mid, min_segment_duration, updated_at)
                    VALUES (:user_id, :threshold, :low, :mid, :min_dur, :updated_at)
                """),
                {
                    "user_id": user_id,
                    "threshold": patch.get("speaker_similarity_threshold", 0.75),
                    "low": patch.get("word_conf_low", 0.7),
                    "mid": patch.get("word_conf_mid", 0.85),
                    "min_dur": patch.get("min_segment_duration", 1.5),
                    "updated_at": dt_to_str(now),
                },
            )
        await db.commit()

    return {"message": "Settings updated.", **patch}
