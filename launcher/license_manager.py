"""
License Manager — Phase 6
30-day demo license with machine binding and clock-rollback detection.

License file (license.dat) is created on first run and verified on every launch.
It is stored adjacent to the launcher executable.

License format (JSON, HMAC-signed):
{
    "install_date": "2026-06-25",
    "machine_id": "<sha256 of machine identifiers>",
    "last_run_date": "2026-06-25",
    "expires_date": "2026-07-25",
    "signature": "<hmac-sha256>"
}
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import platform
import sys
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
DEMO_DURATION_DAYS = 30
LICENSE_FILENAME = "license.dat"

# App secret — combined with machine ID for signing
# In production, this would be obfuscated or compiled in
_APP_SECRET = b"VoiceSum-Demo-2026-SecureKey-X9$kP2"


def _get_license_path() -> Path:
    """Return path to license.dat adjacent to the executable or script."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / LICENSE_FILENAME
    return Path(__file__).parent / LICENSE_FILENAME


def _get_machine_id() -> str:
    """
    Generate a stable machine identifier from hardware characteristics.
    Combines: hostname + OS + CPU info + volume serial (Windows).
    """
    parts = []

    # Hostname
    parts.append(platform.node())

    # OS
    parts.append(platform.system() + platform.release())

    # CPU
    try:
        parts.append(platform.processor())
    except Exception:
        pass

    # Windows volume serial number (most stable identifier)
    if platform.system() == "Windows":
        run_kwargs = {
            "capture_output": True,
            "text": True,
            "creationflags": 0x08000000  # CREATE_NO_WINDOW
        }
        try:
            result = subprocess.run(
                ["wmic", "diskdrive", "get", "SerialNumber"],
                timeout=5,
                **run_kwargs
            )
            if result.returncode == 0:
                serial = result.stdout.strip().split("\n")[-1].strip()
                if serial:
                    parts.append(serial)
        except Exception:
            pass

        # Also try volume label
        try:
            result = subprocess.run(
                ["vol", "C:"],
                shell=True,
                timeout=3,
                **run_kwargs
            )
            if result.returncode == 0:
                parts.append(result.stdout.strip())
        except Exception:
            pass

    raw = "|".join(parts)
    machine_id = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return machine_id


def _sign(data: dict, machine_id: str) -> str:
    """Create HMAC-SHA256 signature over license data."""
    key = hashlib.sha256(_APP_SECRET + machine_id.encode()).digest()
    payload = json.dumps({k: v for k, v in data.items() if k != "signature"}, sort_keys=True)
    sig = hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()
    return sig


def _verify_signature(data: dict, machine_id: str) -> bool:
    """Verify HMAC signature of license data."""
    expected = _sign(data, machine_id)
    actual = data.get("signature", "")
    return hmac.compare_digest(expected, actual)


def create_license() -> dict:
    """Create a new license file on first run."""
    machine_id = _get_machine_id()
    today = date.today()
    expires = today + timedelta(days=DEMO_DURATION_DAYS)

    data = {
        "install_date": today.isoformat(),
        "machine_id": machine_id,
        "last_run_date": today.isoformat(),
        "expires_date": expires.isoformat(),
        "version": "1.0",
    }
    data["signature"] = _sign(data, machine_id)

    license_path = _get_license_path()
    with open(license_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"[License] Created license. Expires: {expires.isoformat()}")
    return data


def _update_last_run(data: dict, machine_id: str):
    """Update last_run_date in license file."""
    data["last_run_date"] = date.today().isoformat()
    data["signature"] = _sign(data, machine_id)
    license_path = _get_license_path()
    with open(license_path, "w") as f:
        json.dump(data, f, indent=2)


class LicenseError(Exception):
    """Raised when license validation fails."""
    pass


def verify_license() -> Tuple[bool, str]:
    """
    Verify the license on startup.

    Returns:
        (True, "") if valid
        (False, error_message) if invalid/expired

    Raises LicenseError for critical failures.
    """
    license_path = _get_license_path()

    # ── First run: create license ──────────────────────────────
    if not license_path.exists():
        logger.info("[License] First run detected — creating license.")
        create_license()
        return True, ""

    # ── Load license ──────────────────────────────────────────
    try:
        with open(license_path) as f:
            data = json.load(f)
    except Exception as e:
        return False, f"License file is corrupted: {e}"

    # ── Verify machine binding ─────────────────────────────────
    machine_id = _get_machine_id()
    stored_machine_id = data.get("machine_id", "")

    if not _verify_signature(data, machine_id):
        # Try stored machine_id in case machine changed slightly
        if not _verify_signature(data, stored_machine_id):
            return False, (
                "License validation failed.\n\n"
                "This may happen if the license was transferred to a different computer.\n"
                "Please contact the developer for a new license."
            )

    # ── Parse dates ───────────────────────────────────────────
    try:
        today = date.today()
        install_date = date.fromisoformat(data["install_date"])
        expires_date = date.fromisoformat(data["expires_date"])
        last_run_date = date.fromisoformat(data.get("last_run_date", data["install_date"]))
    except (KeyError, ValueError) as e:
        return False, f"License file is malformed: {e}"

    # ── Clock rollback detection ───────────────────────────────
    if today < last_run_date:
        return False, (
            "System clock has been rolled back.\n\n"
            f"Last used: {last_run_date.isoformat()}\n"
            f"Current date: {today.isoformat()}\n\n"
            "Please correct your system clock and try again."
        )

    # ── Expiry check ──────────────────────────────────────────
    if today > expires_date:
        days_expired = (today - expires_date).days
        return False, (
            f"This demo version has expired.\n\n"
            f"Expiry date: {expires_date.isoformat()}\n"
            f"Expired {days_expired} day(s) ago.\n\n"
            "Please contact the developer to obtain a new license."
        )

    # ── Days remaining ────────────────────────────────────────
    days_remaining = (expires_date - today).days
    if days_remaining <= 5:
        logger.warning(f"[License] Demo expires in {days_remaining} day(s).")

    # ── Update last run ───────────────────────────────────────
    _update_last_run(data, machine_id)

    logger.info(f"[License] Valid ✓  Expires: {expires_date.isoformat()} ({days_remaining} days remaining)")
    return True, ""


def get_license_info() -> Optional[dict]:
    """Return license info dict or None if not found."""
    license_path = _get_license_path()
    if not license_path.exists():
        return None
    try:
        with open(license_path) as f:
            data = json.load(f)
        today = date.today()
        expires = date.fromisoformat(data["expires_date"])
        data["days_remaining"] = max(0, (expires - today).days)
        data["expired"] = today > expires
        return data
    except Exception:
        return None


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    valid, msg = verify_license()
    if valid:
        info = get_license_info()
        print(f"License valid. Days remaining: {info.get('days_remaining')}")
    else:
        print(f"License INVALID: {msg}")
