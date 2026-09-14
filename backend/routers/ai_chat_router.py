"""
AI Chat Router — Standalone AI Chat endpoints for the top-level AI Chat tab.

Retrieval source: Stage 2 Long Changes points from ChromaDB.
Completely isolated from Collection AI Chat.
All responses streamed via Server-Sent Events (SSE).
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text

from database import get_db_context, dt_to_str, from_json
from routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai-chat", tags=["ai-chat"])


# ── Pydantic Models ──────────────────────────────────────────────────────────

class AIChatRequest(BaseModel):
    message: str
    selected_meeting_id: Optional[str] = None
    max_results: Optional[int] = 8


# ── Chat History Helpers ─────────────────────────────────────────────────────

async def _get_chat_history(user_id: str, limit: int = 20) -> List[dict]:
    """Fetch recent AI Chat messages."""
    async with get_db_context() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT id, role, content, metadata, created_at "
                    "FROM ai_chat_messages "
                    "WHERE user_id = :uid "
                    "ORDER BY created_at DESC LIMIT :limit"
                ),
                {"uid": user_id, "limit": limit},
            )
        ).fetchall()

        messages = [
            {
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "metadata": from_json(r[3], {}),
                "created_at": r[4],
            }
            for r in reversed(rows)
        ]
        return messages


async def _save_message(
    user_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> str:
    """Save an AI Chat message. Returns the message ID."""
    msg_id = str(uuid.uuid4())
    now = dt_to_str(datetime.now(timezone.utc))

    async with get_db_context() as session:
        await session.execute(
            text(
                "INSERT INTO ai_chat_messages "
                "(id, user_id, role, content, metadata, created_at) "
                "VALUES (:id, :uid, :role, :content, :meta, :now)"
            ),
            {
                "id": msg_id,
                "uid": user_id,
                "role": role,
                "content": content,
                "meta": json.dumps(metadata or {}, ensure_ascii=False),
                "now": now,
            },
        )
        await session.commit()

    # Prune old messages (keep max 200 per user)
    async with get_db_context() as session:
        await session.execute(
            text(
                "DELETE FROM ai_chat_messages WHERE id IN ("
                "  SELECT id FROM ai_chat_messages "
                "  WHERE user_id = :uid "
                "  ORDER BY created_at DESC LIMIT -1 OFFSET 200"
                ")"
            ),
            {"uid": user_id},
        )
        await session.commit()

    return msg_id


# ── SSE helper ───────────────────────────────────────────────────────────────

def _sse_event(data: str, event: str = "chunk") -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/send")
async def ai_chat_send(
    body: AIChatRequest,
    user=Depends(get_current_user),
):
    """
    Send a message to the AI Chat.
    Returns a streaming SSE response.
    """
    user_id = user["id"]
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    max_results = body.max_results or 8
    selected_meeting_id = body.selected_meeting_id

    # Load chat history for context
    chat_history = await _get_chat_history(user_id, limit=10)

    # Save user message
    await _save_message(user_id, "user", message)

    async def generate():
        import asyncio

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_inference():
            try:
                from services.ai_chat_service import (
                    triage_query,
                    search_stage2_points,
                    extract_meeting_sources,
                    build_answer_prompt,
                    stream_inference,
                )

                if selected_meeting_id:
                    # User selected a specific meeting — skip triage, always retrieve
                    search_query = message
                    context_required = True
                else:
                    # Run triage to determine if context is needed
                    context_required, triage_detail = triage_query(message, chat_history)
                    search_query = triage_detail if context_required else ""

                if not context_required:
                    # Direct answer from triage
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", triage_detail))
                    meta = {"meeting_sources": [], "query_type": "direct"}
                    loop.call_soon_threadsafe(queue.put_nowait, ("meta", meta))
                    loop.call_soon_threadsafe(queue.put_nowait, ("save", triage_detail, meta))
                else:
                    # Search Stage 2 points
                    results = search_stage2_points(
                        query=search_query,
                        user_id=user_id,
                        meeting_id=selected_meeting_id,
                        max_results=max_results,
                    )

                    # Fallback: if triage query yielded 0 results, retry with raw query
                    if not results and search_query != message:
                        results = search_stage2_points(
                            query=message,
                            user_id=user_id,
                            meeting_id=selected_meeting_id,
                            max_results=max_results,
                        )

                    meeting_sources = extract_meeting_sources(results)

                    # Build prompt and run inference with real-time chunk streaming
                    prompt = build_answer_prompt(
                        question=message,
                        context_results=results,
                        chat_history=chat_history,
                    )

                    def on_chunk(text_chunk: str):
                        if text_chunk:
                            loop.call_soon_threadsafe(queue.put_nowait, ("chunk", text_chunk))

                    full_response = stream_inference(
                        prompt=prompt,
                        max_new_tokens=1500,
                        chunk_callback=on_chunk,
                    )

                    meta = {
                        "meeting_sources": meeting_sources,
                        "query_type": "retrieval",
                    }
                    loop.call_soon_threadsafe(queue.put_nowait, ("meta", meta))
                    loop.call_soon_threadsafe(queue.put_nowait, ("save", full_response, meta))

            except Exception as e:
                logger.error(f"[AIChat] Chat failed: {e}", exc_info=True)
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        task = loop.run_in_executor(None, _run_inference)

        try:
            while True:
                item = await queue.get()
                event_type = item[0]

                if event_type == "chunk":
                    yield _sse_event(item[1], "chunk")
                elif event_type == "meta":
                    yield _sse_event(json.dumps(item[1]), "metadata")
                elif event_type == "save":
                    try:
                        await _save_message(user_id, "assistant", item[1], metadata=item[2])
                    except Exception as err:
                        logger.error(f"[AIChat] Failed to save assistant message: {err}")
                elif event_type == "error":
                    yield _sse_event(f"⚠️ Error: {item[1]}", "error")
                elif event_type == "done":
                    break
        finally:
            await task

        yield _sse_event("", "done")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/meetings")
async def get_meetings_with_stage2(
    user=Depends(get_current_user),
):
    """Return all meetings for the user, annotated with has_stage2 boolean."""
    import asyncio
    user_id = user["id"]

    def _load_chroma():
        from services.ai_chat_service import get_stage2_meeting_ids, get_available_meetings
        stage2_mids = get_stage2_meeting_ids(user_id)
        chroma_meetings = get_available_meetings(user_id)
        return stage2_mids, chroma_meetings

    loop = asyncio.get_event_loop()
    stage2_mids, chroma_meetings = await loop.run_in_executor(None, _load_chroma)

    async with get_db_context() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT id, filename, title, created_at, status "
                    "FROM recordings "
                    "WHERE user_id = :uid "
                    "ORDER BY created_at DESC"
                ),
                {"uid": user_id},
            )
        ).mappings().fetchall()

        meetings = []
        seen_ids = set()
        for r in rows:
            mid = r["id"]
            seen_ids.add(mid)
            meetings.append({
                "id": mid,
                "name": r.get("title") or r.get("filename") or "Meeting",
                "date": r.get("created_at") or "",
                "has_stage2": mid in stage2_mids,
            })

        # Include any ChromaDB meetings that might not be in the SQLite recordings table
        for cm in chroma_meetings:
            cid = cm["id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                meetings.append({
                    "id": cid,
                    "name": cm["name"],
                    "date": cm.get("date", ""),
                    "has_stage2": True,
                })

        meetings.sort(key=lambda m: m.get("date", ""), reverse=True)
        return meetings


@router.get("/history")
async def get_chat_history_endpoint(
    user=Depends(get_current_user),
):
    """Load AI Chat history."""
    messages = await _get_chat_history(user["id"], limit=100)
    return messages


@router.delete("/history")
async def clear_chat_history_endpoint(
    user=Depends(get_current_user),
):
    """Clear all AI Chat history."""
    user_id = user["id"]
    async with get_db_context() as session:
        result = await session.execute(
            text(
                "DELETE FROM ai_chat_messages WHERE user_id = :uid"
            ),
            {"uid": user_id},
        )
        await session.commit()
    return {"message": f"Cleared {result.rowcount} messages"}
