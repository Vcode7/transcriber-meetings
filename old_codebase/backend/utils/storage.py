"""
File storage helpers for audio uploads.

All file saves use streaming (64 KB chunks) so that large uploads — including
multi-hour recordings — never load the full file into RAM. Memory usage stays
constant regardless of file size, limited only by available disk space.
"""
import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile
from config import settings

# Streaming chunk size — 64 KB is a good balance between syscall count and RAM use.
_STREAM_CHUNK_SIZE = 64 * 1024  # 64 KB


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
    """
    Stream an uploaded file to disk and return its path.

    Uses 64 KB chunks to avoid loading the entire file into RAM.
    Works correctly for uploads of any size — limited only by available disk space.
    """
    ext = Path(upload.filename).suffix or ".wav"
    filename = f"{prefix}{uuid.uuid4().hex}{ext}"
    dest = get_user_dir(user_id) / filename
    async with aiofiles.open(dest, "wb") as f:
        while True:
            chunk = await upload.read(_STREAM_CHUNK_SIZE)
            if not chunk:
                break
            await f.write(chunk)
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
