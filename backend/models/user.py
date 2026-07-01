from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ── Voice Profile ─────────────────────────────────────────────
class VoiceProfile(BaseModel):
    id: Optional[str] = None
    user_id: str
    label: str  # "John", "Alice", etc.
    embeddings: List[List[float]]  # list of 256-d vectors
    sample_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"arbitrary_types_allowed": True}


class VoiceProfileOut(BaseModel):
    id: str
    label: str
    sample_count: int
    created_at: datetime
    updated_at: datetime


# ── User ──────────────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserInDB(BaseModel):
    id: Optional[str] = None
    name: str
    email: str
    hashed_password: str
    needs_setup: bool = True  # True until onboarding complete
    own_profile_id: Optional[str] = None  # The user's own voice profile
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"arbitrary_types_allowed": True}


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    needs_setup: bool
    own_profile_id: Optional[str] = None
    created_at: datetime
