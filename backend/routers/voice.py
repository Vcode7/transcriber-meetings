"""Voice profile router — onboarding samples + add-voice + manage profiles."""
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import text

from database import get_db, dt_to_str, to_json, from_json
from routers.auth import get_current_user
from utils.storage import save_upload, delete_file
from utils.audio_utils import validate_audio, convert_to_wav
from services.embedding import extract_embedding_from_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])


# ── Upload a single voice sample ─────────────────────────────
@router.post("/sample")
async def upload_voice_sample(
    file: UploadFile = File(...),
    label: str = Form("self"),
    sample_index: int = Form(0),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload one voice sample. Returns the saved file path.
    Client calls this 1-3 times, then calls /finalize-setup or /add-profile.
    """
    user_id = current_user["id"]
    raw_path = await save_upload(file, user_id, prefix=f"vs_{sample_index}_")

    # Convert to WAV 16 kHz
    wav_path = raw_path.rsplit(".", 1)[0] + "_16k.wav"
    try:
        convert_to_wav(raw_path, wav_path)
    except Exception as e:
        delete_file(raw_path)
        raise HTTPException(status_code=422, detail=f"Audio conversion failed: {e}")

    delete_file(raw_path)  # keep only converted

    # Validate
    valid, reason = validate_audio(wav_path)
    if not valid:
        delete_file(wav_path)
        raise HTTPException(status_code=422, detail=reason)

    return {"file_path": wav_path, "sample_index": sample_index, "label": label}


# ── Finalize onboarding setup ─────────────────────────────────
@router.post("/finalize-setup")
async def finalize_setup(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """
    Body: {"file_paths": [...], "label": "My Name"}
    Generates embeddings from samples and stores as the user's own voice profile.
    Marks needs_setup = False.
    """
    user_id = current_user["id"]
    file_paths = body.get("file_paths", [])
    label = body.get("label", current_user.get("name", "Me"))

    if not file_paths:
        raise HTTPException(status_code=400, detail="No voice sample files provided.")

    embeddings = []
    for fp in file_paths:
        emb = extract_embedding_from_file(fp)
        if emb is not None:
            embeddings.append(emb.tolist())

    if not embeddings:
        raise HTTPException(status_code=422, detail="Could not extract embeddings from samples. Please re-record.")

    now = datetime.now(timezone.utc)
    profile_id = str(uuid.uuid4())

    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO voice_profiles (id, user_id, label, embeddings, sample_count, is_self, created_at, updated_at)
                VALUES (:id, :user_id, :label, :embeddings, :sample_count, 1, :created_at, :updated_at)
            """),
            {
                "id": profile_id,
                "user_id": user_id,
                "label": label,
                "embeddings": to_json(embeddings),
                "sample_count": len(file_paths),
                "created_at": dt_to_str(now),
                "updated_at": dt_to_str(now),
            },
        )
        # Mark setup complete
        await db.execute(
            text("UPDATE users SET needs_setup = 0, own_profile_id = :pid WHERE id = :id"),
            {"pid": profile_id, "id": user_id},
        )
        await db.commit()

    return {
        "profile_id": profile_id,
        "label": label,
        "embedding_count": len(embeddings),
        "message": "Voice profile created. Setup complete!",
    }


# ── Skip onboarding voice profiling ───────────────────────────
@router.post("/skip-setup")
async def skip_setup(current_user: dict = Depends(get_current_user)):
    """
    Skips the voice setup onboarding step.
    Marks needs_setup = False in the database for the user.
    """
    user_id = current_user["id"]
    async with get_db() as db:
        await db.execute(
            text("UPDATE users SET needs_setup = 0 WHERE id = :id"),
            {"id": user_id},
        )
        await db.commit()
    return {"message": "Voice profile setup skipped."}



# ── Add an extra voice profile ────────────────────────────────
@router.post("/add-profile")
async def add_voice_profile(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """Body: {"file_paths": [...], "label": "Alice"}"""
    user_id = current_user["id"]
    file_paths = body.get("file_paths", [])
    label = body.get("label", "").strip()

    if not label:
        raise HTTPException(status_code=400, detail="Label is required.")
    if not file_paths:
        raise HTTPException(status_code=400, detail="No file paths provided.")

    embeddings = []
    for fp in file_paths:
        emb = extract_embedding_from_file(fp)
        if emb is not None:
            embeddings.append(emb.tolist())

    if not embeddings:
        raise HTTPException(status_code=422, detail="Could not extract embeddings. Please re-record with clearer audio.")

    now = datetime.now(timezone.utc)
    profile_id = str(uuid.uuid4())

    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO voice_profiles (id, user_id, label, embeddings, sample_count, is_self, created_at, updated_at)
                VALUES (:id, :user_id, :label, :embeddings, :sample_count, 0, :created_at, :updated_at)
            """),
            {
                "id": profile_id,
                "user_id": user_id,
                "label": label,
                "embeddings": to_json(embeddings),
                "sample_count": len(file_paths),
                "created_at": dt_to_str(now),
                "updated_at": dt_to_str(now),
            },
        )
        await db.commit()

    return {
        "profile_id": profile_id,
        "label": label,
        "embedding_count": len(embeddings),
    }


# ── List all profiles ─────────────────────────────────────────
@router.get("/profiles")
async def list_profiles(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM voice_profiles WHERE user_id = :uid ORDER BY created_at DESC LIMIT 50"),
            {"uid": user_id},
        )
        profiles = r.mappings().fetchall()

    return [
        {
            "id": p["id"],
            "label": p["label"],
            "sample_count": p.get("sample_count", 0),
            "is_self": bool(p.get("is_self", False)),
            "created_at": p["created_at"],
            "updated_at": p["updated_at"],
        }
        for p in profiles
    ]


# ── Rename profile ────────────────────────────────────────────
@router.put("/profiles/{profile_id}")
async def update_profile(
    profile_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]
    now = datetime.now(timezone.utc)

    async with get_db() as db:
        r = await db.execute(
            text("UPDATE voice_profiles SET label = :label, updated_at = :updated_at WHERE id = :id AND user_id = :uid"),
            {"label": body.get("label", ""), "updated_at": dt_to_str(now), "id": profile_id, "uid": user_id},
        )
        await db.commit()
        if r.rowcount == 0:
            raise HTTPException(status_code=404, detail="Profile not found.")

    return {"message": "Profile updated."}


# ── Delete profile ────────────────────────────────────────────
@router.delete("/profiles/{profile_id}")
async def delete_profile(
    profile_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]

    async with get_db() as db:
        r = await db.execute(
            text("DELETE FROM voice_profiles WHERE id = :id AND user_id = :uid"),
            {"id": profile_id, "uid": user_id},
        )
        if r.rowcount == 0:
            raise HTTPException(status_code=404, detail="Profile not found.")

        # If deleted own profile, mark needs_setup again
        if str(current_user.get("own_profile_id")) == profile_id:
            await db.execute(
                text("UPDATE users SET needs_setup = 1, own_profile_id = NULL WHERE id = :id"),
                {"id": user_id},
            )
        await db.commit()

    return {"message": "Profile deleted."}
