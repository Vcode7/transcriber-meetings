"""
Auth router — production-grade session management.

Architecture:
  - Access token (JWT, 15 min) → returned in response body, stored in memory on client
  - Refresh token (32-byte random, 30 days) → HttpOnly SameSite cookie, hashed in DB
  - Sessions table tracks every active device session
  - Rate limiting and account lockout via login_attempts table
"""
import hashlib
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Cookie, HTTPException, Depends, Request, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import text

from database import get_db, dt_to_str, str_to_dt
from models.user import UserCreate, UserLogin
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

# ── Crypto helpers ─────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

REFRESH_COOKIE = "vs_refresh"


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_token(raw: str) -> str:
    """SHA-256 hex digest — fast, one-way, no need for bcrypt on random tokens."""
    return hashlib.sha256(raw.encode()).hexdigest()


def create_access_token(user_id: str, session_id: str) -> str:
    """Short-lived JWT carrying user_id and session_id for revocation checks."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": user_id, "sid": session_id, "exp": expire},
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )


def create_refresh_token() -> tuple[str, str]:
    """Return (raw_token, hashed_token). Raw goes to cookie; hash goes to DB."""
    raw = os.urandom(32).hex()  # 64-char hex string
    return raw, hash_token(raw)


def set_refresh_cookie(response: Response, raw_token: str, expires_at: datetime) -> None:
    """Set HttpOnly refresh token cookie with proper security flags."""
    max_age = int((expires_at - datetime.now(timezone.utc)).total_seconds())
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=raw_token,
        max_age=max_age,
        expires=expires_at.strftime("%a, %d %b %Y %H:%M:%S GMT"),
        httponly=True,
        samesite="lax",
        secure=(settings.ENVIRONMENT == "production"),
        path="/",
    )


def clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=REFRESH_COOKIE,
        httponly=True,
        samesite="lax",
        secure=(settings.ENVIRONMENT == "production"),
        path="/",
    )


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _get_device_name(request: Request) -> str:
    ua = request.headers.get("User-Agent", "")
    if "Chrome" in ua and "Edg" not in ua:
        browser = "Chrome"
    elif "Firefox" in ua:
        browser = "Firefox"
    elif "Safari" in ua and "Chrome" not in ua:
        browser = "Safari"
    elif "Edg" in ua:
        browser = "Edge"
    else:
        browser = "Browser"

    if "Windows" in ua:
        os_name = "Windows"
    elif "Mac" in ua:
        os_name = "macOS"
    elif "Linux" in ua:
        os_name = "Linux"
    elif "Android" in ua:
        os_name = "Android"
    elif "iPhone" in ua or "iPad" in ua:
        os_name = "iOS"
    else:
        os_name = "Unknown OS"

    return f"{browser} on {os_name}"


# ── Rate limiting & lockout ────────────────────────────────────

async def _check_rate_limit(email: str, ip: str) -> None:
    """Raise 429 if too many failed attempts; raise 423 if account locked."""
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(seconds=settings.RATE_LIMIT_LOGIN_WINDOW_SECONDS)

    async with get_db() as db:
        # Check account lockout on user row
        row = await db.execute(
            text("SELECT locked_until FROM users WHERE email = :email"),
            {"email": email.lower()},
        )
        user_row = row.fetchone()
        if user_row and user_row.locked_until:
            locked_until = str_to_dt(user_row.locked_until)
            if locked_until and now < locked_until:
                remaining = int((locked_until - now).total_seconds())
                raise HTTPException(
                    status_code=423,
                    detail=f"Account temporarily locked. Try again in {remaining // 60 + 1} minutes.",
                )

        # Count recent IP failures
        r = await db.execute(
            text("""
                SELECT COUNT(*) FROM login_attempts
                WHERE ip_address = :ip AND success = 0 AND created_at >= :window
            """),
            {"ip": ip, "window": dt_to_str(window_start)},
        )
        ip_count = r.scalar()
        if ip_count >= settings.RATE_LIMIT_LOGIN_MAX * 2:
            raise HTTPException(
                status_code=429,
                detail="Too many login attempts from this IP. Try again later.",
            )


async def _record_attempt(email: str, ip: str, success: bool) -> None:
    """Record a login attempt and apply lockout on max failures."""
    now = datetime.now(timezone.utc)
    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO login_attempts (email, ip_address, success, created_at)
                VALUES (:email, :ip, :success, :created_at)
            """),
            {"email": email.lower(), "ip": ip, "success": 1 if success else 0, "created_at": dt_to_str(now)},
        )

        if not success:
            window_start = now - timedelta(seconds=settings.RATE_LIMIT_LOGIN_WINDOW_SECONDS)
            r = await db.execute(
                text("""
                    SELECT COUNT(*) FROM login_attempts
                    WHERE email = :email AND success = 0 AND created_at >= :window
                """),
                {"email": email.lower(), "window": dt_to_str(window_start)},
            )
            failures = r.scalar()
            if failures >= settings.RATE_LIMIT_LOGIN_MAX:
                locked_until = now + timedelta(seconds=settings.ACCOUNT_LOCKOUT_SECONDS)
                await db.execute(
                    text("""
                        UPDATE users SET locked_until = :locked_until, failed_login_attempts = :fails
                        WHERE email = :email
                    """),
                    {"locked_until": dt_to_str(locked_until), "fails": failures, "email": email.lower()},
                )
        else:
            # Clear lockout on successful login
            await db.execute(
                text("UPDATE users SET locked_until = NULL, failed_login_attempts = 0 WHERE email = :email"),
                {"email": email.lower()},
            )

        await db.commit()


# ── Session helpers ────────────────────────────────────────────

async def _create_session(user_id: str, raw_refresh: str, request: Request) -> dict:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    session = {
        "id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "user_id": user_id,
        "refresh_token_hash": hash_token(raw_refresh),
        "device_name": _get_device_name(request),
        "ip_address": _get_client_ip(request),
        "created_at": dt_to_str(now),
        "expires_at": dt_to_str(expires_at),
        "last_used": dt_to_str(now),
        "is_revoked": 0,
    }
    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO sessions (id, session_id, user_id, refresh_token_hash, device_name,
                    ip_address, created_at, expires_at, last_used, is_revoked)
                VALUES (:id, :session_id, :user_id, :refresh_token_hash, :device_name,
                    :ip_address, :created_at, :expires_at, :last_used, :is_revoked)
            """),
            session,
        )
        await db.commit()
    session["expires_at"] = str_to_dt(session["expires_at"])
    return session


# ── Dependency: current user ───────────────────────────────────

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> dict:
    """Validate access token and ensure session is not revoked."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        session_id: str = payload.get("sid")
        if not user_id or not session_id:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    now = datetime.now(timezone.utc)
    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM sessions WHERE session_id = :sid"),
            {"sid": session_id},
        )
        session = r.mappings().fetchone()
        if not session or session["is_revoked"]:
            raise credentials_exception

        r2 = await db.execute(
            text("SELECT * FROM users WHERE id = :id"),
            {"id": user_id},
        )
        user = r2.mappings().fetchone()
        if not user:
            raise credentials_exception

    return dict(user)


# ── Shared response builder ────────────────────────────────────

def _user_payload(user: dict) -> dict:
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "needs_setup": bool(user.get("needs_setup", True)),
        "own_profile_id": user.get("own_profile_id"),
    }


async def _issue_tokens_and_respond(
    response: Response,
    user: dict,
    request: Request,
) -> dict:
    """Create a session, set cookie, return access token."""
    user_id = user["id"]
    raw_refresh, _ = create_refresh_token()
    session = await _create_session(user_id, raw_refresh, request)
    access_token = create_access_token(user_id, session["session_id"])
    set_refresh_cookie(response, raw_refresh, session["expires_at"])
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.JWT_EXPIRE_MINUTES * 60,
        "user": _user_payload(user),
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

# ── Register ─────────────────────────────────────────────────
@router.post("/register", status_code=201)
async def register(body: UserCreate, request: Request, response: Response):
    async with get_db() as db:
        r = await db.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": body.email.lower()},
        )
        if r.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered.")

    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    hashed = hash_password(body.password)
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO users (id, name, email, hashed_password, needs_setup, own_profile_id,
                    locked_until, failed_login_attempts, created_at)
                VALUES (:id, :name, :email, :hashed_password, 1, NULL, NULL, 0, :created_at)
            """),
            {"id": user_id, "name": body.name.strip(), "email": body.email.lower(),
             "hashed_password": hashed, "created_at": dt_to_str(now)},
        )
        # Bootstrap default settings
        await db.execute(
            text("""
                INSERT INTO user_settings (user_id, speaker_similarity_threshold, word_conf_low,
                    word_conf_mid, min_segment_duration, updated_at)
                VALUES (:user_id, :threshold, :low, :mid, :min_dur, :updated_at)
            """),
            {
                "user_id": user_id,
                "threshold": settings.SPEAKER_SIMILARITY_THRESHOLD,
                "low": settings.WORD_CONF_LOW,
                "mid": settings.WORD_CONF_MID,
                "min_dur": settings.MIN_SEGMENT_DURATION,
                "updated_at": dt_to_str(now),
            },
        )
        await db.commit()

    user_doc = {
        "id": user_id, "name": body.name.strip(), "email": body.email.lower(),
        "needs_setup": 1, "own_profile_id": None,
    }
    return await _issue_tokens_and_respond(response, user_doc, request)


# ── Login ────────────────────────────────────────────────────
@router.post("/login")
async def login(body: UserLogin, request: Request, response: Response):
    ip = _get_client_ip(request)
    await _check_rate_limit(body.email, ip)

    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM users WHERE email = :email"),
            {"email": body.email.lower()},
        )
        user = r.mappings().fetchone()

    if not user or not verify_password(body.password, user["hashed_password"]):
        if user:
            await _record_attempt(body.email, ip, success=False)
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    await _record_attempt(body.email, ip, success=True)
    return await _issue_tokens_and_respond(response, dict(user), request)


# ── Refresh ──────────────────────────────────────────────────
@router.post("/refresh")
async def refresh(request: Request, response: Response):
    """
    Silent token refresh. Reads refresh token from HttpOnly cookie, validates against
    DB session, issues new access token and rotates the refresh token.
    """
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    if not raw_refresh:
        raise HTTPException(status_code=401, detail="No refresh token.")

    token_hash = hash_token(raw_refresh)
    now = datetime.now(timezone.utc)

    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM sessions WHERE refresh_token_hash = :hash"),
            {"hash": token_hash},
        )
        session = r.mappings().fetchone()
        if not session:
            raise HTTPException(status_code=401, detail="Invalid refresh token.")

        if session["is_revoked"]:
            logger.warning(f"[AUTH] Revoked refresh token reused for user {session['user_id']} — revoking all sessions.")
            await db.execute(
                text("UPDATE sessions SET is_revoked = 1 WHERE user_id = :uid"),
                {"uid": session["user_id"]},
            )
            await db.commit()
            clear_refresh_cookie(response)
            raise HTTPException(status_code=401, detail="Session revoked. Please log in again.")

        expires_at = str_to_dt(session["expires_at"])
        if expires_at and now >= expires_at:
            clear_refresh_cookie(response)
            raise HTTPException(status_code=401, detail="Refresh token expired. Please log in again.")

        # Rotate refresh token
        new_raw, new_hash = create_refresh_token()
        new_expires = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

        await db.execute(
            text("""
                UPDATE sessions SET refresh_token_hash = :hash, expires_at = :exp, last_used = :lu
                WHERE session_id = :sid
            """),
            {"hash": new_hash, "exp": dt_to_str(new_expires), "lu": dt_to_str(now), "sid": session["session_id"]},
        )

        r2 = await db.execute(
            text("SELECT * FROM users WHERE id = :id"),
            {"id": session["user_id"]},
        )
        user = r2.mappings().fetchone()
        await db.commit()

    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    access_token = create_access_token(session["user_id"], session["session_id"])
    set_refresh_cookie(response, new_raw, new_expires)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.JWT_EXPIRE_MINUTES * 60,
        "user": _user_payload(dict(user)),
    }


# ── Logout ───────────────────────────────────────────────────
@router.post("/logout")
async def logout(request: Request, response: Response):
    """Revoke current session and clear cookie."""
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    if raw_refresh:
        token_hash = hash_token(raw_refresh)
        async with get_db() as db:
            await db.execute(
                text("UPDATE sessions SET is_revoked = 1 WHERE refresh_token_hash = :hash"),
                {"hash": token_hash},
            )
            await db.commit()
    clear_refresh_cookie(response)
    return {"detail": "Logged out successfully."}


# ── Me ───────────────────────────────────────────────────────
@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {
        **_user_payload(current_user),
        "created_at": current_user["created_at"],
    }


# ── Sessions list ────────────────────────────────────────────
@router.get("/sessions")
async def list_sessions(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Return all active (non-revoked, non-expired) sessions for current user."""
    now = datetime.now(timezone.utc)
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    current_hash = hash_token(raw_refresh) if raw_refresh else None

    async with get_db() as db:
        r = await db.execute(
            text("""
                SELECT * FROM sessions
                WHERE user_id = :uid AND is_revoked = 0 AND expires_at > :now
            """),
            {"uid": current_user["id"], "now": dt_to_str(now)},
        )
        rows = r.mappings().fetchall()

    sessions = []
    for s in rows:
        sessions.append({
            "session_id": s["session_id"],
            "device_name": s["device_name"],
            "ip_address": s["ip_address"],
            "created_at": s["created_at"],
            "last_used": s["last_used"],
            "expires_at": s["expires_at"],
            "is_current": s["refresh_token_hash"] == current_hash,
        })

    # Current session first, then by last_used desc
    sessions.sort(key=lambda x: (not x["is_current"], x["last_used"]), reverse=False)
    return {"sessions": sessions}


# ── Revoke session ───────────────────────────────────────────
@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: str,
    response: Response,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Revoke a specific session by session_id (must belong to current user)."""
    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM sessions WHERE session_id = :sid AND user_id = :uid"),
            {"sid": session_id, "uid": current_user["id"]},
        )
        session = r.mappings().fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")

        await db.execute(
            text("UPDATE sessions SET is_revoked = 1 WHERE session_id = :sid"),
            {"sid": session_id},
        )
        await db.commit()

    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    if raw_refresh and hash_token(raw_refresh) == session["refresh_token_hash"]:
        clear_refresh_cookie(response)
        return {"detail": "Session revoked. You have been logged out.", "self": True}

    return {"detail": "Session revoked.", "self": False}


# ── OAuth2 form token (Swagger UI only) ─────────────────────
@router.post("/token", include_in_schema=False)
async def token_form(form: OAuth2PasswordRequestForm = Depends(), request: Request = None, response: Response = None):
    async with get_db() as db:
        r = await db.execute(
            text("SELECT * FROM users WHERE email = :email"),
            {"email": form.username.lower()},
        )
        user = r.mappings().fetchone()
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    user_id = user["id"]
    session_id = str(uuid.uuid4())  # ephemeral session for Swagger
    token = create_access_token(user_id, session_id)
    return {"access_token": token, "token_type": "bearer"}
