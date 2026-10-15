"""
rom_staging.py — Files chosen in the ROM settings panel before the meeting exists.

The Upload / Record / Tab Recording pages pick meeting-context files and the agenda document
before a recording row exists, so they are staged per user under
``UPLOAD_DIR/rom_staging/{user_id}/{staged_id}/`` and referenced from the meeting's
``rom_settings``. The background ROM run moves them onto the recording.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from typing import Any, Dict, Optional

from config import settings

logger = logging.getLogger(__name__)

STAGING_SUBDIR = "rom_staging"
EXTRACTED_FILE = "extracted.json"
MAX_AGE_SEC = 7 * 24 * 3600
_STAGED_ID = re.compile(r"^[0-9a-f]{32}$")


def _user_dir(user_id: str) -> str:
    safe_user = re.sub(r"[^A-Za-z0-9_-]", "_", user_id)
    return os.path.join(settings.UPLOAD_DIR, STAGING_SUBDIR, safe_user)


def staged_dir(user_id: str, staged_id: str) -> Optional[str]:
    if not _STAGED_ID.match(staged_id or ""):
        return None
    path = os.path.join(_user_dir(user_id), staged_id)
    return path if os.path.isdir(path) else None


def safe_filename(filename: str) -> str:
    name = os.path.basename(filename or "document").strip() or "document"
    return re.sub(r"[^\w.\- ()]", "_", name)[:180]


def cleanup_old(user_id: str) -> None:
    """Remove staged files older than a week (never attached to a meeting)."""
    root = _user_dir(user_id)
    if not os.path.isdir(root):
        return
    cutoff = time.time() - MAX_AGE_SEC
    for name in os.listdir(root):
        path = os.path.join(root, name)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def stage_file(user_id: str, filename: str, content: bytes) -> str:
    cleanup_old(user_id)
    staged_id = uuid.uuid4().hex
    path = os.path.join(_user_dir(user_id), staged_id)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, safe_filename(filename)), "wb") as fh:
        fh.write(content)
    return staged_id


def staged_file(user_id: str, staged_id: str) -> Optional[str]:
    """Path of the staged upload itself (not the extraction JSON)."""
    path = staged_dir(user_id, staged_id)
    if not path:
        return None
    for name in os.listdir(path):
        if name != EXTRACTED_FILE:
            return os.path.join(path, name)
    return None


def write_extracted(user_id: str, staged_id: str, payload: Dict[str, Any]) -> None:
    path = staged_dir(user_id, staged_id)
    if path:
        with open(os.path.join(path, EXTRACTED_FILE), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)


def read_extracted(user_id: str, staged_id: str) -> Optional[Dict[str, Any]]:
    path = staged_dir(user_id, staged_id)
    if not path or not os.path.isfile(os.path.join(path, EXTRACTED_FILE)):
        return None
    try:
        with open(os.path.join(path, EXTRACTED_FILE), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning(f"[ROMStaging] Could not read extraction for {staged_id}: {exc}")
        return None


def delete_staged(user_id: str, staged_id: str) -> bool:
    path = staged_dir(user_id, staged_id)
    if not path:
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True
