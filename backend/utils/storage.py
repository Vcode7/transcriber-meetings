"""File storage helpers for audio uploads."""
import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile
from config import settings


def get_upload_dir() -> Path:
    p = Path(settings.UPLOAD_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_user_dir(user_id: str) -> Path:
    p = get_upload_dir() / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_voice_dir(user_id: str) -> Path:
    p = get_user_dir(user_id) / "voice_samples"
    p.mkdir(parents=True, exist_ok=True)
    return p


async def save_upload(upload: UploadFile, user_id: str, prefix: str = "") -> str:
    """Save an uploaded file and return its path."""
    ext = Path(upload.filename).suffix or ".wav"
    filename = f"{prefix}{uuid.uuid4().hex}{ext}"
    dest = get_user_dir(user_id) / filename
    async with aiofiles.open(dest, "wb") as f:
        content = await upload.read()
        await f.write(content)
    return str(dest)


async def save_bytes(data: bytes, user_id: str, prefix: str = "", ext: str = ".wav") -> str:
    """Save raw bytes to a file and return path."""
    filename = f"{prefix}{uuid.uuid4().hex}{ext}"
    dest = get_user_dir(user_id) / filename
    async with aiofiles.open(dest, "wb") as f:
        await f.write(data)
    return str(dest)


def delete_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
