"""Pydantic models for the Dictionary & Prompt system."""
from pydantic import BaseModel, Field
from typing import Optional


# ── Shortcut Dictionary ───────────────────────────────────────────────────────

class ShortcutCreate(BaseModel):
    shortcut: str = Field(..., min_length=1, max_length=50)
    full_form: str = Field(..., min_length=1, max_length=500)


class ShortcutUpdate(BaseModel):
    shortcut: Optional[str] = Field(None, min_length=1, max_length=50)
    full_form: Optional[str] = Field(None, min_length=1, max_length=500)


class ShortcutImport(BaseModel):
    """Bulk import payload — list of {shortcut, full_form} dicts."""
    entries: list[dict]


# ── Technical Vocabulary ──────────────────────────────────────────────────────

class VocabCreate(BaseModel):
    word: str = Field(..., min_length=1, max_length=200)


class VocabBulkCreate(BaseModel):
    """Bulk save after user review of extracted words."""
    words: list[str]


# ── Document Import ───────────────────────────────────────────────────────────

class AIExtractionResult(BaseModel):
    """Shape returned by AI extraction endpoint (before user saves)."""
    technical_words: list[str] = []
    shortcuts: list[dict] = []  # [{short, full}]


# ── Global Prompt ─────────────────────────────────────────────────────────────

class GlobalPromptUpdate(BaseModel):
    prompt: str = Field(..., max_length=4000)
