"""
Collection AI Router — AI Chat endpoints scoped to a meeting collection.

Provides three AI capabilities:
  1. Free-form Q&A (chat) with RAG over all meetings in a collection
  2. Meeting comparison reports (select two meetings)
  3. Topic growth tracking across meetings

All responses are streamed via Server-Sent Events (SSE).
Chat history is persisted per-collection in SQLite.
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

from database import get_db, dt_to_str, from_json
from routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/collections/{collection_id}/ai", tags=["collection-ai"])


# ── Pydantic Models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    max_context: Optional[int] = 10

class CompareRequest(BaseModel):
    meeting_id_a: str
    meeting_id_b: str

class TopicGrowthRequest(BaseModel):
    topic: str

class ExportRequest(BaseModel):
    content: str
    filename: Optional[str] = "report"


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _verify_collection_ownership(collection_id: str, user_id: str) -> dict:
    """Verify the collection exists and belongs to the user. Returns collection row."""
    async with get_db() as session:
        coll = (
            await session.execute(
                text("SELECT id, name FROM meeting_collections WHERE id = :cid AND user_id = :uid"),
                {"cid": collection_id, "uid": user_id},
            )
        ).fetchone()
        if not coll:
            raise HTTPException(status_code=404, detail="Collection not found")
        return {"id": coll[0], "name": coll[1]}


async def _get_collection_meetings(collection_id: str) -> List[dict]:
    """Fetch all meetings in a collection with their metadata."""
    async with get_db() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT r.id, r.filename, r.created_at, r.duration, r.status, "
                    "r.transcript_embedded, r.context_summary, LENGTH(r.transcript) "
                    "FROM meeting_collection_items ci "
                    "JOIN recordings r ON r.id = ci.meeting_id "
                    "WHERE ci.collection_id = :cid "
                    "ORDER BY r.created_at ASC"
                ),
                {"cid": collection_id},
            )
        ).fetchall()

        return [
            {
                "id": r[0],
                "filename": r[1],
                "created_at": r[2],
                "duration": r[3] or 0,
                "status": r[4],
                "transcript_embedded": r[5],
                "context_summary": r[6],
                "has_transcript": r[7] is not None and r[7] > 2,
            }
            for r in rows
        ]


async def _get_chat_history(collection_id: str, user_id: str, limit: int = 20) -> List[dict]:
    """Fetch recent chat messages for a collection."""
    async with get_db() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT id, role, content, message_type, metadata, created_at "
                    "FROM collection_chat_messages "
                    "WHERE collection_id = :cid AND user_id = :uid "
                    "ORDER BY created_at DESC LIMIT :limit"
                ),
                {"cid": collection_id, "uid": user_id, "limit": limit},
            )
        ).fetchall()

        messages = [
            {
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "message_type": r[3],
                "metadata": from_json(r[4], {}),
                "created_at": r[5],
            }
            for r in reversed(rows)  # Reverse to get chronological order
        ]
        return messages


async def _save_message(
    collection_id: str,
    user_id: str,
    role: str,
    content: str,
    message_type: str = "chat",
    metadata: dict | None = None,
) -> str:
    """Save a chat message to the database. Returns the message ID."""
    msg_id = str(uuid.uuid4())
    now = dt_to_str(datetime.now(timezone.utc))

    async with get_db() as session:
        await session.execute(
            text(
                "INSERT INTO collection_chat_messages "
                "(id, collection_id, user_id, role, content, message_type, metadata, created_at) "
                "VALUES (:id, :cid, :uid, :role, :content, :type, :meta, :now)"
            ),
            {
                "id": msg_id,
                "cid": collection_id,
                "uid": user_id,
                "role": role,
                "content": content,
                "type": message_type,
                "meta": json.dumps(metadata or {}, ensure_ascii=False),
                "now": now,
            },
        )
        await session.commit()

    # Prune old messages (keep max 200 per collection per user)
    async with get_db() as session:
        await session.execute(
            text(
                "DELETE FROM collection_chat_messages WHERE id IN ("
                "  SELECT id FROM collection_chat_messages "
                "  WHERE collection_id = :cid AND user_id = :uid "
                "  ORDER BY created_at DESC LIMIT -1 OFFSET 200"
                ")"
            ),
            {"cid": collection_id, "uid": user_id},
        )
        await session.commit()

    return msg_id


# ── SSE helper ───────────────────────────────────────────────────────────────

def _sse_event(data: str, event: str = "chunk") -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/chat")
async def collection_chat(
    collection_id: str,
    body: ChatRequest,
    user=Depends(get_current_user),
):
    """
    Send a message to the collection AI chat.
    Returns a streaming SSE response.
    """
    user_id = user["id"]
    await _verify_collection_ownership(collection_id, user_id)

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    max_context = body.max_context or 10

    # Fetch collection meetings
    meetings = await _get_collection_meetings(collection_id)
    if not meetings:
        raise HTTPException(status_code=400, detail="Collection has no meetings")

    # Check if any meetings are transcribed (meaning they have transcripts in DB)
    transcribed_meetings = [m for m in meetings if m.get("has_transcript")]
    if not transcribed_meetings:
        raise HTTPException(
            status_code=400,
            detail="No meetings in this collection have been transcribed. "
                   "Transcribe meetings first."
        )

    # Load chat history for context
    chat_history = await _get_chat_history(collection_id, user_id, limit=10)

    # Save user message
    await _save_message(collection_id, user_id, "user", message, "chat")

    # Prepare meeting metadata
    meeting_ids = [m["id"] for m in transcribed_meetings]
    meeting_meta = {m["id"]: m for m in transcribed_meetings}

    async def generate():
        """SSE generator that runs RAG + LLM in a thread."""
        import asyncio

        def _run_inference():
            from services.collection_ai_service import (
                ensure_meetings_are_embedded,
                retrieve_collection_context_two_stage,
                build_chat_prompt,
                run_collection_ai_inference_streaming,
            )

            # Ensure all meetings in collection are embedded in FAISS
            ensure_meetings_are_embedded(meeting_ids, user_id)

            # Two-Stage RAG retrieval & Triage (Stage 1 Context Planning -> Stage 2 Vector Search + Fallback)
            ctx, plan_or_direct, context_required = retrieve_collection_context_two_stage(
                meeting_ids=meeting_ids,
                meeting_meta=meeting_meta,
                query=message,
                user_id=user_id,
                chat_history=chat_history,
                max_context=max_context,
                k_per_meeting=5,
                max_total_chars=20000,
            )

            if not context_required:
                # Direct answer returned from Stage 1 (e.g. general greeting/knowledge) — skip Stage 2 RAG
                full_response = plan_or_direct
                chunks_collected = [full_response]
                cited = []
                meeting_names = {}
                plan_used = "No context required (Direct Answer)"
            else:
                # Build prompt with retrieved context
                prompt = build_chat_prompt(
                    question=message,
                    context=ctx,
                    chat_history=chat_history,
                )

                # Collect cited meetings
                cited = list({c.meeting_id for c in ctx.chunks if c.meeting_id != "global"})
                meeting_names = {c.meeting_id: c.meeting_name for c in ctx.chunks if c.meeting_id != "global"}

                # Run inference (collects full response for saving)
                chunks_collected = []

                def on_chunk(text_chunk):
                    chunks_collected.append(text_chunk)

                run_collection_ai_inference_streaming(
                    prompt=prompt,
                    max_new_tokens=1500,
                    chunk_callback=on_chunk,
                    task_key="collection_chat",
                )

                full_response = "".join(chunks_collected)
                plan_used = plan_or_direct

            return full_response, chunks_collected, cited, meeting_names, plan_used

        loop = asyncio.get_event_loop()
        try:
            full_response, chunks, cited, meeting_names, plan_used = await loop.run_in_executor(
                None, _run_inference
            )

            # Stream chunks to client
            for chunk in chunks:
                yield _sse_event(chunk, "chunk")

            # Send metadata
            meta = {
                "cited_meetings": cited,
                "meeting_names": meeting_names,
                "retrieval_plan": plan_used,
                "max_context": max_context,
            }
            yield _sse_event(json.dumps(meta), "metadata")

            # Save assistant response
            await _save_message(
                collection_id, user_id, "assistant", full_response, "chat",
                metadata=meta,
            )

        except Exception as e:
            logger.error(f"[CollectionAI] Chat failed: {e}", exc_info=True)
            yield _sse_event(f"⚠️ Error: {str(e)}", "error")

        yield _sse_event("", "done")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/compare")
async def collection_compare(
    collection_id: str,
    body: CompareRequest,
    user=Depends(get_current_user),
):
    """
    Compare two meetings within a collection.
    Returns a streaming SSE response with a structured comparison report.
    """
    user_id = user["id"]
    await _verify_collection_ownership(collection_id, user_id)

    meetings = await _get_collection_meetings(collection_id)
    meeting_map = {m["id"]: m for m in meetings}

    if body.meeting_id_a not in meeting_map:
        raise HTTPException(status_code=400, detail="Meeting A not found in this collection")
    if body.meeting_id_b not in meeting_map:
        raise HTTPException(status_code=400, detail="Meeting B not found in this collection")
    if body.meeting_id_a == body.meeting_id_b:
        raise HTTPException(status_code=400, detail="Cannot compare a meeting with itself")

    meta_a = meeting_map[body.meeting_id_a]
    meta_b = meeting_map[body.meeting_id_b]

    if not meta_a.get("has_transcript") or not meta_b.get("has_transcript"):
        raise HTTPException(
            status_code=400,
            detail="Both meetings must have a transcript before comparison."
        )

    # Save user message
    compare_msg = f"Compare: {meta_a['filename']} vs {meta_b['filename']}"
    await _save_message(collection_id, user_id, "user", compare_msg, "comparison")

    async def generate():
        import asyncio

        def _run_comparison():
            from services.collection_ai_service import (
                ensure_meetings_are_embedded,
                get_meeting_full_context,
                build_comparison_prompt,
                run_collection_ai_inference_streaming,
            )

            # Ensure both meetings are embedded in FAISS
            ensure_meetings_are_embedded([body.meeting_id_a, body.meeting_id_b], user_id)

            # Get broad context from both meetings
            chunks_a = get_meeting_full_context(body.meeting_id_a, meta_a, max_chars=12000)
            chunks_b = get_meeting_full_context(body.meeting_id_b, meta_b, max_chars=12000)

            if not chunks_a and not chunks_b:
                return "⚠️ No transcript content was found in these meetings.", [], [], {}

            # Build prompt
            prompt = build_comparison_prompt(
                meeting_a_name=meta_a["filename"],
                meeting_a_date=meta_a.get("created_at", ""),
                meeting_a_chunks=chunks_a,
                meeting_b_name=meta_b["filename"],
                meeting_b_date=meta_b.get("created_at", ""),
                meeting_b_chunks=chunks_b,
            )

            # Run inference
            chunks_collected = []
            def on_chunk(text_chunk):
                chunks_collected.append(text_chunk)

            run_collection_ai_inference_streaming(
                prompt=prompt,
                max_new_tokens=2000,
                chunk_callback=on_chunk,
                task_key="collection_compare",
            )

            full_response = "".join(chunks_collected)
            cited = [body.meeting_id_a, body.meeting_id_b]
            meeting_names = {
                body.meeting_id_a: meta_a["filename"],
                body.meeting_id_b: meta_b["filename"],
            }
            return full_response, chunks_collected, cited, meeting_names

        loop = asyncio.get_event_loop()
        try:
            full_response, chunks, cited, meeting_names = await loop.run_in_executor(
                None, _run_comparison
            )

            for chunk in chunks:
                yield _sse_event(chunk, "chunk")

            meta = {"cited_meetings": cited, "meeting_names": meeting_names}
            yield _sse_event(json.dumps(meta), "metadata")

            await _save_message(
                collection_id, user_id, "assistant", full_response, "comparison",
                metadata=meta,
            )

        except Exception as e:
            logger.error(f"[CollectionAI] Comparison failed: {e}", exc_info=True)
            yield _sse_event(f"⚠️ Error: {str(e)}", "error")

        yield _sse_event("", "done")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/topic-growth")
async def collection_topic_growth(
    collection_id: str,
    body: TopicGrowthRequest,
    user=Depends(get_current_user),
):
    """
    Track a topic's evolution across all meetings in a collection.
    Returns a streaming SSE response with a chronological report.
    """
    user_id = user["id"]
    await _verify_collection_ownership(collection_id, user_id)

    topic = body.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    # Fetch collection meetings
    meetings = await _get_collection_meetings(collection_id)
    if not meetings:
        raise HTTPException(status_code=400, detail="Collection has no meetings")

    # Check if any meetings are transcribed (meaning they have transcripts in DB)
    transcribed_meetings = [m for m in meetings if m.get("has_transcript")]
    if not transcribed_meetings:
        raise HTTPException(
            status_code=400,
            detail="No meetings in this collection have been transcribed. "
                   "Transcribe meetings first."
        )

    # Save user message
    await _save_message(collection_id, user_id, "user", f"Topic Growth: {topic}", "topic_growth")

    meeting_ids = [m["id"] for m in transcribed_meetings]

    async def generate():
        import asyncio

        def _run_topic_growth():
            from services.collection_ai_service import (
                ensure_meetings_are_embedded,
                retrieve_meeting_context,
                build_topic_growth_prompt,
                run_collection_ai_inference_streaming,
            )

            # Ensure all meetings in collection are embedded in FAISS
            ensure_meetings_are_embedded(meeting_ids, user_id)

            # Search each meeting for the topic
            meetings_data = []
            for m in transcribed_meetings:
                chunks = retrieve_meeting_context(
                    meeting_id=m["id"],
                    meeting_meta=m,
                    query=topic,
                    user_id=user_id,
                    k=5,
                    max_chars=5000,
                )
                if chunks:
                    meetings_data.append({
                        "name": m["filename"],
                        "date": m.get("created_at", ""),
                        "meeting_id": m["id"],
                        "chunks": chunks,
                    })

            if not meetings_data:
                return (
                    f"⚠️ The topic \"{topic}\" was not found in any meetings in this collection.",
                    [f"⚠️ The topic \"{topic}\" was not found in any meetings in this collection."],
                    [], {},
                )

            # Build prompt
            prompt = build_topic_growth_prompt(topic=topic, meetings_data=meetings_data)

            # Run inference
            chunks_collected = []
            def on_chunk(text_chunk):
                chunks_collected.append(text_chunk)

            run_collection_ai_inference_streaming(
                prompt=prompt,
                max_new_tokens=2000,
                chunk_callback=on_chunk,
                task_key="collection_topic_growth",
            )

            full_response = "".join(chunks_collected)
            cited = [md["meeting_id"] for md in meetings_data]
            meeting_names = {md["meeting_id"]: md["name"] for md in meetings_data}
            return full_response, chunks_collected, cited, meeting_names

        loop = asyncio.get_event_loop()
        try:
            full_response, chunks, cited, meeting_names = await loop.run_in_executor(
                None, _run_topic_growth
            )

            for chunk in chunks:
                yield _sse_event(chunk, "chunk")

            meta = {"cited_meetings": cited, "meeting_names": meeting_names}
            yield _sse_event(json.dumps(meta), "metadata")

            await _save_message(
                collection_id, user_id, "assistant", full_response, "topic_growth",
                metadata=meta,
            )

        except Exception as e:
            logger.error(f"[CollectionAI] Topic growth failed: {e}", exc_info=True)
            yield _sse_event(f"⚠️ Error: {str(e)}", "error")

        yield _sse_event("", "done")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Chat History Endpoints ───────────────────────────────────────────────────

@router.get("/history")
async def get_chat_history(
    collection_id: str,
    user=Depends(get_current_user),
):
    """Load chat history for a collection."""
    user_id = user["id"]
    await _verify_collection_ownership(collection_id, user_id)
    messages = await _get_chat_history(collection_id, user_id, limit=100)
    return messages


@router.delete("/history")
async def clear_chat_history(
    collection_id: str,
    user=Depends(get_current_user),
):
    """Clear all chat history for a collection."""
    user_id = user["id"]
    await _verify_collection_ownership(collection_id, user_id)

    async with get_db() as session:
        result = await session.execute(
            text(
                "DELETE FROM collection_chat_messages "
                "WHERE collection_id = :cid AND user_id = :uid"
            ),
            {"cid": collection_id, "uid": user_id},
        )
        await session.commit()

    return {"message": f"Cleared {result.rowcount} messages"}


# ── Export Endpoint ──────────────────────────────────────────────────────────

@router.post("/export")
async def export_report(
    collection_id: str,
    body: ExportRequest,
    user=Depends(get_current_user),
):
    """Export AI-generated content as a downloadable Markdown file."""
    user_id = user["id"]
    await _verify_collection_ownership(collection_id, user_id)

    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="No content to export")

    filename = f"{body.filename or 'report'}.md"

    return StreamingResponse(
        iter([content.encode("utf-8")]),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
