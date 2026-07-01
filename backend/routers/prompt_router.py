"""Global Prompt router — get/save per-user transcription prompt."""
from fastapi import APIRouter, Depends
from database import get_db
from routers.auth import get_current_user
from models.dictionary import GlobalPromptUpdate
from services.dictionary_service import get_global_prompt, save_global_prompt

router = APIRouter(prefix="/prompt", tags=["prompt"])


@router.get("/global")
async def get_prompt(current_user: dict = Depends(get_current_user)):
    """Return the current user's global transcription prompt."""
    user_id = current_user["id"]
    async with get_db() as db:
        prompt = await get_global_prompt(db, user_id)
    return {"prompt": prompt}


@router.put("/global")
async def save_prompt(
    body: GlobalPromptUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Upsert the global transcription prompt (auto-save)."""
    user_id = current_user["id"]
    async with get_db() as db:
        await save_global_prompt(db, user_id, body.prompt)
    return {"message": "Prompt saved.", "prompt": body.prompt}
