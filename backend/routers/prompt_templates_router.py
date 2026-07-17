"""
Prompt Templates Router — CRUD for system-wide AI prompt templates.

All endpoints require authentication (any logged-in user can manage prompts,
since the templates are system-wide / shared).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from database import get_db
from routers.auth import get_current_user
from services import prompt_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prompt-templates", tags=["prompt-templates"])


# ── Pydantic models ───────────────────────────────────────────────────────────

class PromptUpdate(BaseModel):
    template: str

    @field_validator("template")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Template cannot be blank")
        return v


class PromptImport(BaseModel):
    version: int = 1
    templates: dict[str, str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
async def list_all_prompts(current_user: dict = Depends(get_current_user)):
    """
    Return metadata + current template for every prompt.
    Each item includes: key, name, category, description, variables,
    template (active), default_template, is_modified, updated_at.
    """
    async with get_db() as db:
        return await prompt_service.list_prompts(db)


@router.get("/export")
async def export_all_prompts(current_user: dict = Depends(get_current_user)):
    """Download all active templates as a JSON export."""
    async with get_db() as db:
        data = await prompt_service.export_prompts(db)
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": "attachment; filename=prompt_templates.json"},
    )


@router.get("/{key}")
async def get_single_prompt(key: str, current_user: dict = Depends(get_current_user)):
    """Return the active template (custom or default) for a single key."""
    if key not in prompt_service.VALID_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown prompt key: {key!r}")
    async with get_db() as db:
        template = await prompt_service.get_prompt(db, key)
    default = prompt_service._defaults().get(key, "")
    return {
        "key": key,
        "template": template,
        "is_modified": template != default,
        "default_template": default,
    }


@router.put("/{key}")
async def save_prompt(
    key: str,
    body: PromptUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Save a custom template for key. Takes effect immediately (no restart)."""
    if key not in prompt_service.VALID_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown prompt key: {key!r}")
    async with get_db() as db:
        await prompt_service.set_prompt(db, key, body.template)
    return {"message": f"Prompt '{key}' saved.", "key": key, "length": len(body.template)}


@router.delete("/{key}")
async def reset_single_prompt(key: str, current_user: dict = Depends(get_current_user)):
    """Reset a single prompt to its hardcoded default."""
    if key not in prompt_service.VALID_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown prompt key: {key!r}")
    async with get_db() as db:
        await prompt_service.reset_prompt(db, key)
    return {"message": f"Prompt '{key}' reset to default.", "key": key}


@router.delete("")
async def reset_all_prompts(current_user: dict = Depends(get_current_user)):
    """Reset ALL prompts to hardcoded defaults."""
    async with get_db() as db:
        await prompt_service.reset_all_prompts(db)
    return {"message": "All prompts reset to defaults."}


@router.post("/import")
async def import_prompts(
    body: PromptImport,
    current_user: dict = Depends(get_current_user),
):
    """Import prompt templates from a JSON export payload."""
    async with get_db() as db:
        result = await prompt_service.import_prompts(db, body.model_dump())
    return {
        "message": f"Import complete: {result['imported']} imported, {result['skipped']} skipped.",
        **result,
    }
