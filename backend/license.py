"""
License validation for the application.

To update the expiry date for a future release, change LICENSE_EXPIRY_DATE only.
No other code needs to be touched.
"""
from datetime import date

# ── CONFIGURATION ──────────────────────────────────────────────
# The application is licensed until (and including) this date.
# Beginning the following day the app will refuse to operate.
LICENSE_EXPIRY_DATE: date = date(2026, 7, 30)

# Message shown to users after expiry
LICENSE_EXPIRED_MESSAGE: str = (
    "This application license expired on 30 July 2026. "
    "Please contact the administrator for a renewed version."
)

# HTTP 503 JSON body returned by the middleware after expiry
LICENSE_EXPIRED_BODY: dict = {
    "error": "license_expired",
    "expired_on": LICENSE_EXPIRY_DATE.isoformat(),
    "message": LICENSE_EXPIRED_MESSAGE,
}


def check_license() -> tuple[bool, str]:
    """
    Check whether the application license is still valid.

    Returns
    -------
    (True, "")          — license is valid
    (False, message)    — license has expired; message contains the human-readable reason
    """
    today = date.today()
    if today <= LICENSE_EXPIRY_DATE:
        return True, ""
    days_expired = (today - LICENSE_EXPIRY_DATE).days
    return (
        False,
        f"{LICENSE_EXPIRED_MESSAGE} (expired {days_expired} day(s) ago)",
    )
