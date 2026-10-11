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
from typing import Optional, List, Any, Dict

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
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
    session_id: Optional[str] = "default"


class AIChatSessionCreate(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = "New Chat"
    selected_meeting_id: Optional[str] = None


class AIChatSessionUpdate(BaseModel):
    title: Optional[str] = None
    selected_meeting_id: Optional[str] = None


# ── Chat History Helpers ─────────────────────────────────────────────────────

async def _get_chat_history(
    user_id: str,
    session_id: Optional[str] = None,
    limit: int = 200,
) -> List[dict]:
    """Fetch recent AI Chat messages for a specific session."""
    async with get_db_context() as session:
        if session_id and session_id != "default":
            try:
                rows = (
                    await session.execute(
                        text(
                            "SELECT id, role, content, metadata, created_at "
                            "FROM ai_chat_messages "
                            "WHERE user_id = :uid AND session_id = :sid "
                            "ORDER BY created_at DESC LIMIT :limit"
                        ),
                        {"uid": user_id, "sid": session_id, "limit": limit},
                    )
                ).fetchall()
            except Exception:
                rows = []
        elif session_id == "default":
            try:
                rows = (
                    await session.execute(
                        text(
                            "SELECT id, role, content, metadata, created_at "
                            "FROM ai_chat_messages "
                            "WHERE user_id = :uid AND (session_id = 'default' OR session_id IS NULL) "
                            "ORDER BY created_at DESC LIMIT :limit"
                        ),
                        {"uid": user_id, "limit": limit},
                    )
                ).fetchall()
            except Exception:
                rows = []
        else:
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
    session_id: str = "default",
) -> str:
    """Save an AI Chat message. Ensures session exists and returns message ID."""
    msg_id = str(uuid.uuid4())
    now = dt_to_str(datetime.now(timezone.utc))

    async with get_db_context() as session:
        # 1. Ensure the session exists in ai_chat_sessions so messages are never orphaned
        try:
            s_row = (
                await session.execute(
                    text("SELECT id FROM ai_chat_sessions WHERE id = :sid AND user_id = :uid"),
                    {"sid": session_id, "uid": user_id},
                )
            ).fetchone()
            if not s_row:
                auto_title = (content[:36].strip() + ("..." if len(content) > 36 else "")) if role == "user" else "Chat"
                await session.execute(
                    text(
                        "INSERT INTO ai_chat_sessions (id, user_id, title, selected_meeting_id, created_at, updated_at) "
                        "VALUES (:sid, :uid, :title, NULL, :now, :now)"
                    ),
                    {"sid": session_id, "uid": user_id, "title": auto_title or "Chat", "now": now},
                )
            else:
                await session.execute(
                    text("UPDATE ai_chat_sessions SET updated_at = :now WHERE id = :sid AND user_id = :uid"),
                    {"now": now, "sid": session_id, "uid": user_id},
                )
        except Exception as e:
            logger.warning(f"[AIChat] Error checking/updating session {session_id}: {e}")

        # 2. Insert message
        try:
            await session.execute(
                text(
                    "INSERT INTO ai_chat_messages "
                    "(id, user_id, role, content, metadata, created_at, session_id) "
                    "VALUES (:id, :uid, :role, :content, :meta, :now, :sid)"
                ),
                {
                    "id": msg_id,
                    "uid": user_id,
                    "role": role,
                    "content": content,
                    "meta": json.dumps(metadata or {}, ensure_ascii=False),
                    "now": now,
                    "sid": session_id,
                },
            )
        except Exception:
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

    # 3. Prune old messages PER SESSION (keep max 300 per session to avoid wiping other chats)
    try:
        async with get_db_context() as session:
            await session.execute(
                text(
                    "DELETE FROM ai_chat_messages WHERE id IN ("
                    "  SELECT id FROM ai_chat_messages "
                    "  WHERE user_id = :uid AND session_id = :sid "
                    "  ORDER BY created_at DESC LIMIT -1 OFFSET 300"
                    ")"
                ),
                {"uid": user_id, "sid": session_id},
            )
            await session.commit()
    except Exception as e:
        logger.warning(f"[AIChat] Failed to prune messages for session {session_id}: {e}")

    return msg_id


# ── SSE helper ───────────────────────────────────────────────────────────────

def _sse_event(data: str, event: str = "chunk") -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Meeting Context Helper ───────────────────────────────────────────────────

async def _load_user_available_meetings(user_id: str) -> List[Dict[str, Any]]:
    """Return all meetings for the user, annotated with has_final_rom and has_stage2 booleans."""
    import asyncio
    from services.ai_chat_service import (
        get_stage2_meeting_ids,
        get_final_rom_meeting_ids,
        get_available_meetings,
    )

    def _load_chroma():
        try:
            final_mids = get_final_rom_meeting_ids(user_id)
            stage2_mids = get_stage2_meeting_ids(user_id)
            chroma_meetings = get_available_meetings(user_id)
            return final_mids, stage2_mids, chroma_meetings
        except Exception as e:
            logger.warning(f"[AIChat] Error querying ChromaDB meeting collections for user {user_id}: {e}")
            return set(), set(), []

    loop = asyncio.get_running_loop()
    final_mids, stage2_mids, chroma_meetings = await loop.run_in_executor(None, _load_chroma)

    async with get_db_context() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT id, filename, title, created_at, status, source_type "
                    "FROM recordings "
                    "WHERE user_id = :uid "
                    "ORDER BY created_at DESC"
                ),
                {"uid": user_id},
            )
        ).mappings().fetchall()

        # Also query manual_moms table directly for completeness
        mom_rows = (
            await session.execute(
                text(
                    "SELECT id, filename, title, meeting_date, created_at "
                    "FROM manual_moms "
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
            is_manual = bool(r.get("source_type") == "manual_mom" or str(mid).startswith("mom_"))
            meetings.append({
                "id": mid,
                "name": r.get("title") or r.get("filename") or "Meeting",
                "date": r.get("created_at") or "",
                "has_stage2": mid in stage2_mids,
                "has_final_rom": (mid in final_mids) or is_manual,
                "is_manual_mom": is_manual,
            })

        # Include manual_moms rows if any were not in recordings table
        for mr in mom_rows:
            mom_id = mr["id"]
            if mom_id not in seen_ids:
                seen_ids.add(mom_id)
                meetings.append({
                    "id": mom_id,
                    "name": mr.get("title") or mr.get("filename") or "Manual MoM",
                    "date": mr.get("meeting_date") or mr.get("created_at") or "",
                    "has_stage2": False,
                    "has_final_rom": True,
                    "is_manual_mom": True,
                })

        # Include any ChromaDB meetings that might not be in the SQLite recordings table
        for cm in chroma_meetings:
            cid = cm["id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                is_man = bool(cm.get("is_manual_mom") or str(cid).startswith("mom_"))
                meetings.append({
                    "id": cid,
                    "name": cm["name"],
                    "date": cm.get("date", ""),
                    "has_stage2": cm.get("has_stage2", False),
                    "has_final_rom": cm.get("has_final_rom", False) or is_man,
                    "is_manual_mom": is_man,
                })

        meetings.sort(key=lambda m: m.get("date", ""), reverse=True)
        logger.info(
            f"[AIChat] Loaded {len(meetings)} available meetings for user {user_id} "
            f"(Final ROM in Chroma: {len(final_mids)}, Stage 2 in Chroma: {len(stage2_mids)})"
        )
        return meetings


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
    session_id = body.session_id or "default"

    # Load chat history for context (scoped to session)
    chat_history = await _get_chat_history(user_id, session_id=session_id, limit=10)

    # Save user message
    await _save_message(user_id, "user", message, session_id=session_id)

    # Load available meetings for triage and context retrieval
    available_meetings = await _load_user_available_meetings(user_id)

    # Auto-title session if it's currently named "New Chat"
    try:
        async with get_db_context() as sess:
            s_row = (
                await sess.execute(
                    text("SELECT title FROM ai_chat_sessions WHERE id = :sid AND user_id = :uid"),
                    {"sid": session_id, "uid": user_id},
                )
            ).fetchone()
            if s_row and (not s_row[0] or s_row[0].strip().lower() in ("new chat", "untitled")):
                auto_title = message[:38].strip()
                if len(message) > 38:
                    auto_title += "..."
                await sess.execute(
                    text("UPDATE ai_chat_sessions SET title = :t WHERE id = :sid AND user_id = :uid"),
                    {"t": auto_title, "sid": session_id, "uid": user_id},
                )
                await sess.commit()
    except Exception:
        pass

    async def generate():
        import asyncio

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_inference():
            try:
                from services.ai_chat_service import (
                    triage_query,
                    get_direct_meeting_context,
                    search_meeting_points,
                    is_detailed_query,
                    extract_meeting_sources,
                    build_answer_prompt,
                    stream_inference,
                )

                meeting_sources = []
                query_type = "retrieval"
                is_detailed = is_detailed_query(message)

                if selected_meeting_id:
                    # User explicitly selected a specific meeting from the dropdown UI
                    target_mid = selected_meeting_id
                    m_name = next(
                        (m["name"] for m in available_meetings if m["id"] == target_mid),
                        "Selected Meeting",
                    )
                    logger.info(
                        f"[AIChat] User dropdown selection: '{m_name}' (ID: {target_mid}). "
                        f"Retrieving direct meeting context (is_detailed={is_detailed})."
                    )
                    # Fetch direct meeting context adhering to Priority 1 (Final ROM)
                    results = get_direct_meeting_context(user_id, target_mid, is_detailed=is_detailed)
                    query_type = "direct_meeting"

                    # Fallback to similarity search filtered to this meeting if direct is empty
                    if not results:
                        logger.info(
                            f"[AIChat] Direct context empty for {target_mid}. "
                            "Falling back to similarity search on this meeting."
                        )
                        results = search_meeting_points(
                            query=message,
                            user_id=user_id,
                            meeting_id=target_mid,
                            max_results=max_results,
                            is_detailed=is_detailed,
                        )
                        query_type = "retrieval"

                    if not results:
                        logger.warning(
                            f"[AIChat Context Warning] No context retrieved for selected meeting '{m_name}' (ID: {target_mid}). "
                            f"Reason: Direct retrieval found 0 points in ChromaDB and SQLite, and similarity search matched 0 points."
                        )

                    meeting_sources = extract_meeting_sources(results) or [
                        {"id": target_mid, "name": m_name, "date": ""}
                    ]

                else:
                    # Let the AI triage decide which approach is appropriate
                    triage_res = triage_query(
                        question=message,
                        chat_history=chat_history,
                        available_meetings=available_meetings,
                        user_id=user_id,
                    )

                    if not triage_res.context_required:
                        # Direct answer from triage (greeting, general Q, or available meeting list)
                        logger.info(f"[AIChat] Triage determined no meeting context required for query: '{message[:60]}'")
                        loop.call_soon_threadsafe(queue.put_nowait, ("chunk", triage_res.detail))
                        meta = {"meeting_sources": [], "query_type": "direct"}
                        loop.call_soon_threadsafe(queue.put_nowait, ("meta", meta))
                        # Persist assistant response directly so client disconnect never drops it
                        fut = asyncio.run_coroutine_threadsafe(
                            _save_message(user_id, "assistant", triage_res.detail, metadata=meta, session_id=session_id),
                            loop,
                        )
                        try:
                            fut.result(timeout=10.0)
                        except Exception as e:
                            logger.error(f"[AIChat] Failed to save triage assistant message: {e}")
                        return

                    elif triage_res.retrieval_mode == "direct_meeting" and triage_res.target_meeting_id:
                        target_mid = triage_res.target_meeting_id
                        target_name = (
                            triage_res.target_meeting_name
                            or next(
                                (m["name"] for m in available_meetings if m["id"] == target_mid),
                                "Meeting",
                            )
                        )
                        logger.info(
                            f"[AIChat] AI chose Direct Meeting Selection: '{target_name}' (ID: {target_mid}). "
                            f"Fetching context directly (is_detailed={is_detailed})."
                        )

                        # Fetch direct meeting context adhering to Priority 1 (Final ROM)
                        results = get_direct_meeting_context(user_id, target_mid, is_detailed=is_detailed)
                        query_type = "direct_meeting"

                        # Fallback to similarity search if direct context returned empty
                        if not results:
                            logger.info(
                                f"[AIChat] Direct context returned empty for {target_mid}. "
                                "Falling back to similarity search."
                            )
                            results = search_meeting_points(
                                query=triage_res.detail or message,
                                user_id=user_id,
                                meeting_id=target_mid,
                                max_results=max_results,
                                is_detailed=is_detailed,
                            )
                            if not results:
                                results = search_meeting_points(
                                    query=message,
                                    user_id=user_id,
                                    meeting_id=target_mid,
                                    max_results=max_results,
                                    is_detailed=is_detailed,
                                )
                            query_type = "retrieval"

                        if not results:
                            logger.warning(
                                f"[AIChat Context Warning] No context retrieved for AI-targeted meeting '{target_name}' (ID: {target_mid}). "
                                f"Reason: Neither Direct retrieval nor Similarity search matched discussion points for this meeting."
                            )

                        meeting_sources = extract_meeting_sources(results) or [
                            {"id": target_mid, "name": target_name, "date": ""}
                        ]

                    else:
                        # Semantic / similarity search across meetings prioritizing Final ROM
                        search_query = triage_res.detail or message
                        logger.info(f"[AIChat] AI chose Semantic Search for query: '{search_query[:60]}...' (is_detailed={is_detailed})")
                        results = search_meeting_points(
                            query=search_query,
                            user_id=user_id,
                            max_results=max_results,
                            is_detailed=is_detailed,
                        )
                        # Fallback if triage query yielded 0 results
                        if not results and search_query != message:
                            logger.info(f"[AIChat] Triage search returned 0 results. Retrying with raw query: '{message[:60]}'")
                            results = search_meeting_points(
                                query=message,
                                user_id=user_id,
                                max_results=max_results,
                                is_detailed=is_detailed,
                            )
                        if not results:
                            logger.warning(
                                f"[AIChat Context Warning] No context retrieved for semantic search query: '{message[:80]}'. "
                                f"Reason: ChromaDB vector collections for user {user_id} contain 0 matching points above threshold "
                                f"or vector store has not been indexed for these meetings."
                            )
                        query_type = "retrieval"
                        meeting_sources = extract_meeting_sources(results)

                if results:
                    logger.info(f"[AIChat Context] Proceeding to inference with {len(results)} context points from {len(meeting_sources)} meetings.")
                else:
                    logger.warning(
                        f"[AIChat Context Warning] Proceeding to inference with ZERO context points for user query: '{message[:80]}'. "
                        f"The LLM will be instructed to inform the user that no relevant discussion points were found."
                    )

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
                    "query_type": query_type,
                    "retrieval_mode": query_type,
                }
                loop.call_soon_threadsafe(queue.put_nowait, ("meta", meta))

                # Persist assistant response directly so client disconnect never drops it
                fut = asyncio.run_coroutine_threadsafe(
                    _save_message(user_id, "assistant", full_response, metadata=meta, session_id=session_id),
                    loop,
                )
                try:
                    fut.result(timeout=10.0)
                except Exception as e:
                    logger.error(f"[AIChat] Failed to save assistant message: {e}")

            except Exception as e:
                logger.error(f"[AIChat Error] Chat processing failed: {e}", exc_info=True)
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
                    pass
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
    return await _load_user_available_meetings(user["id"])


# ── Session Management Endpoints ─────────────────────────────────────────────

@router.get("/sessions")
async def get_chat_sessions_endpoint(
    user=Depends(get_current_user),
):
    """Load user's chat sessions, auto-healing any orphaned sessions that contain messages."""
    user_id = user["id"]
    now = dt_to_str(datetime.now(timezone.utc))
    async with get_db_context() as session:
        try:
            # Auto-backfill any session_id found in ai_chat_messages that is missing in ai_chat_sessions
            orphaned = (
                await session.execute(
                    text(
                        "SELECT m.session_id, MIN(m.created_at) as first_created, MAX(m.created_at) as last_created "
                        "FROM ai_chat_messages m "
                        "LEFT JOIN ai_chat_sessions s ON m.session_id = s.id AND m.user_id = s.user_id "
                        "WHERE m.user_id = :uid AND m.session_id IS NOT NULL AND s.id IS NULL "
                        "GROUP BY m.session_id"
                    ),
                    {"uid": user_id},
                )
            ).fetchall()

            for orphan in orphaned:
                o_sid = orphan[0]
                if not o_sid:
                    continue
                o_created = orphan[1] or now
                o_updated = orphan[2] or now
                first_msg_row = (
                    await session.execute(
                        text(
                            "SELECT content FROM ai_chat_messages "
                            "WHERE user_id = :uid AND session_id = :sid AND role = 'user' "
                            "ORDER BY created_at ASC LIMIT 1"
                        ),
                        {"uid": user_id, "sid": o_sid},
                    )
                ).fetchone()
                o_title = "Chat"
                if first_msg_row and first_msg_row[0]:
                    o_title = first_msg_row[0][:36].strip()
                    if len(first_msg_row[0]) > 36:
                        o_title += "..."
                await session.execute(
                    text(
                        "INSERT INTO ai_chat_sessions (id, user_id, title, selected_meeting_id, created_at, updated_at) "
                        "VALUES (:sid, :uid, :title, NULL, :c_at, :u_at)"
                    ),
                    {
                        "sid": o_sid,
                        "uid": user_id,
                        "title": o_title or "Chat",
                        "c_at": o_created,
                        "u_at": o_updated,
                    },
                )
            if orphaned:
                await session.commit()

            rows = (
                await session.execute(
                    text(
                        "SELECT id, title, selected_meeting_id, created_at, updated_at "
                        "FROM ai_chat_sessions "
                        "WHERE user_id = :uid "
                        "ORDER BY updated_at DESC"
                    ),
                    {"uid": user_id},
                )
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "title": r[1],
                    "selected_meeting_id": r[2],
                    "created_at": r[3],
                    "updated_at": r[4],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"Failed to query ai_chat_sessions: {e}")
            return []


@router.post("/sessions")
async def create_chat_session_endpoint(
    body: AIChatSessionCreate,
    user=Depends(get_current_user),
):
    """Create or register a chat session."""
    user_id = user["id"]
    sid = (body.id or "").strip() or str(uuid.uuid4())
    now = dt_to_str(datetime.now(timezone.utc))
    title = (body.title or "New Chat").strip()

    async with get_db_context() as session:
        try:
            existing = (
                await session.execute(
                    text("SELECT id FROM ai_chat_sessions WHERE id = :id AND user_id = :uid"),
                    {"id": sid, "uid": user_id},
                )
            ).fetchone()
            if existing:
                await session.execute(
                    text(
                        "UPDATE ai_chat_sessions SET title = :title, selected_meeting_id = :smid, updated_at = :now "
                        "WHERE id = :id AND user_id = :uid"
                    ),
                    {
                        "id": sid,
                        "uid": user_id,
                        "title": title,
                        "smid": body.selected_meeting_id,
                        "now": now,
                    },
                )
            else:
                await session.execute(
                    text(
                        "INSERT INTO ai_chat_sessions (id, user_id, title, selected_meeting_id, created_at, updated_at) "
                        "VALUES (:id, :uid, :title, :smid, :now, :now)"
                    ),
                    {
                        "id": sid,
                        "uid": user_id,
                        "title": title,
                        "smid": body.selected_meeting_id,
                        "now": now,
                    },
                )
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to create chat session: {e}")
            raise HTTPException(status_code=500, detail="Failed to create session")

    return {
        "id": sid,
        "title": title,
        "selected_meeting_id": body.selected_meeting_id,
        "created_at": now,
        "updated_at": now,
    }


@router.patch("/sessions/{session_id}")
async def update_chat_session_endpoint(
    session_id: str,
    body: AIChatSessionUpdate,
    user=Depends(get_current_user),
):
    """Update chat session title or focused meeting."""
    user_id = user["id"]
    now = dt_to_str(datetime.now(timezone.utc))
    updates = []
    params: dict = {"sid": session_id, "uid": user_id, "now": now}
    if body.title is not None:
        updates.append("title = :title")
        params["title"] = body.title.strip()
    if body.selected_meeting_id is not None:
        updates.append("selected_meeting_id = :smid")
        params["smid"] = body.selected_meeting_id or None

    if not updates:
        return {"status": "noop"}

    updates.append("updated_at = :now")
    query = f"UPDATE ai_chat_sessions SET {', '.join(updates)} WHERE id = :sid AND user_id = :uid"
    async with get_db_context() as session:
        await session.execute(text(query), params)
        await session.commit()
    return {"status": "ok"}


@router.delete("/sessions/{session_id}")
async def delete_chat_session_endpoint(
    session_id: str,
    user=Depends(get_current_user),
):
    """Delete a chat session and all its messages."""
    user_id = user["id"]
    async with get_db_context() as session:
        try:
            await session.execute(
                text("DELETE FROM ai_chat_messages WHERE user_id = :uid AND session_id = :sid"),
                {"uid": user_id, "sid": session_id},
            )
        except Exception:
            pass
        try:
            await session.execute(
                text("DELETE FROM ai_chat_sessions WHERE user_id = :uid AND id = :sid"),
                {"uid": user_id, "sid": session_id},
            )
        except Exception:
            pass
        await session.commit()
    return {"status": "ok"}


@router.get("/history")
async def get_chat_history_endpoint(
    session_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Load AI Chat history for a session."""
    messages = await _get_chat_history(user["id"], session_id=session_id, limit=300)
    return messages


@router.delete("/history")
async def clear_chat_history_endpoint(
    session_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Clear AI Chat history for a specific session or all."""
    user_id = user["id"]
    async with get_db_context() as session:
        if session_id and session_id != "default":
            result = await session.execute(
                text(
                    "DELETE FROM ai_chat_messages WHERE user_id = :uid AND session_id = :sid"
                ),
                {"uid": user_id, "sid": session_id},
            )
        elif session_id == "default":
            result = await session.execute(
                text(
                    "DELETE FROM ai_chat_messages WHERE user_id = :uid AND (session_id = 'default' OR session_id IS NULL)"
                ),
                {"uid": user_id},
            )
        else:
            result = await session.execute(
                text(
                    "DELETE FROM ai_chat_messages WHERE user_id = :uid"
                ),
                {"uid": user_id},
            )
        await session.commit()
    return {"message": f"Cleared {result.rowcount} messages"}


# ── Meeting Resync Helpers & Endpoints ───────────────────────────────────────

def _safe_parse_rom_data(raw: Any) -> dict:
    """Safely parse rom_data, handling JSON strings, double-JSON, None, etc."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                return {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _extract_stage2_points(rom_data: dict) -> List[dict]:
    """Extract valid Stage 2 discussion points from parsed rom_data."""
    if not isinstance(rom_data, dict):
        return []
    stage2 = rom_data.get("stage2")
    raw_points = []
    if isinstance(stage2, dict):
        raw_points = (
            stage2.get("polished_points")
            or stage2.get("discussion_points")
            or stage2.get("points")
            or []
        )
    elif isinstance(stage2, list):
        raw_points = stage2

    valid_points = []
    if isinstance(raw_points, list):
        for p in raw_points:
            if isinstance(p, dict):
                text = (p.get("polished_text") or p.get("discussion_point") or "").strip()
                if text:
                    valid_points.append(p)
            elif isinstance(p, str) and p.strip():
                valid_points.append({"polished_text": p.strip()})
    return valid_points


def _extract_final_rom_points(rom_dict: Optional[dict]) -> List[dict]:
    """Extract flattened discussion points from Final ROM agendas."""
    if not isinstance(rom_dict, dict):
        return []
    final_rom = rom_dict.get("final_rom") or rom_dict.get("stage3") or {}
    agendas = final_rom.get("agendas") or []
    flat = []
    for a_idx, ag in enumerate(agendas):
        if not isinstance(ag, dict):
            continue
        ag_id = str(ag.get("agenda_id") or f"agenda_{a_idx}")
        ag_title = str(ag.get("title") or ag.get("agenda_title") or "").strip()
        pts = ag.get("discussion_points") or []
        for p in pts:
            if isinstance(p, dict):
                text = (p.get("text") or p.get("polished_text") or p.get("discussion_point") or "").strip()
                if text:
                    flat.append({
                        "id": p.get("id"),
                        "text": text,
                        "agenda_id": ag_id,
                        "agenda_title": ag_title,
                        "speakers": p.get("speakers") or p.get("speaker"),
                        "action_owner": p.get("action_owner"),
                    })
            elif isinstance(p, str) and p.strip():
                flat.append({
                    "id": str(uuid.uuid4()),
                    "text": p.strip(),
                    "agenda_id": ag_id,
                    "agenda_title": ag_title,
                    "speakers": None,
                    "action_owner": None,
                })
    return flat


@router.get("/syncable-meetings")
async def get_syncable_meetings(
    user=Depends(get_current_user),
):
    """
    List all meetings for the user, checking both Final ROM and Stage 2 discussion points,
    annotated with their ChromaDB index status for both collections.
    """
    import asyncio
    from services.vector_store import get_stage2_points_store, get_final_rom_points_store

    user_id = user["id"]

    # 1. Fetch all recordings with their ROM data from database
    async with get_db_context() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT r.id, r.filename, r.title, r.created_at, "
                    "       COALESCE(rm.rom_data, r.rom_data) as rom_data "
                    "FROM recordings r "
                    "LEFT JOIN rom_metadata rm ON rm.recording_id = r.id "
                    "WHERE r.user_id = :uid "
                    "ORDER BY r.created_at DESC"
                ),
                {"uid": user_id},
            )
        ).mappings().fetchall()

    if not rows:
        return []

    # 2. Extract Stage 2 and Final ROM points for each meeting
    candidates = []
    for r in rows:
        mid = r["id"]
        rom_dict = _safe_parse_rom_data(r.get("rom_data"))
        s2_pts = _extract_stage2_points(rom_dict)
        final_pts = _extract_final_rom_points(rom_dict)
        candidates.append({
            "id": mid,
            "name": r.get("title") or r.get("filename") or "Meeting",
            "date": r.get("created_at") or "",
            "stage2_points_count": len(s2_pts),
            "final_rom_points_count": len(final_pts),
            "has_stage2": len(s2_pts) > 0,
            "has_final_rom": len(final_pts) > 0,
        })

    # 3. Query ChromaDB to check indexed counts for both collections
    def _check_chroma_counts():
        s2_counts = {}
        final_counts = {}
        try:
            s2_store = get_stage2_points_store(user_id)
            s2_store.load_or_create()
            if s2_store._collection is not None:
                res = s2_store._collection.get(include=["metadatas"])
                metadatas = res.get("metadatas", [])
                for meta in metadatas:
                    if meta:
                        mid = meta.get("meeting_id") or meta.get("recording_id")
                        if mid:
                            s2_counts[mid] = s2_counts.get(mid, 0) + 1
        except Exception as e:
            logger.warning(f"[AIChat] Failed to check Stage 2 ChromaDB counts: {e}")

        try:
            final_store = get_final_rom_points_store(user_id)
            final_store.load_or_create()
            if final_store._collection is not None:
                res = final_store._collection.get(include=["metadatas"])
                metadatas = res.get("metadatas", [])
                for meta in metadatas:
                    if meta:
                        mid = meta.get("meeting_id") or meta.get("recording_id")
                        if mid:
                            final_counts[mid] = final_counts.get(mid, 0) + 1
        except Exception as e:
            logger.warning(f"[AIChat] Failed to check Final ROM ChromaDB counts: {e}")

        return s2_counts, final_counts

    loop = asyncio.get_running_loop()
    s2_counts, final_counts = await loop.run_in_executor(None, _check_chroma_counts)

    results = []
    for c in candidates:
        mid = c["id"]
        s2_total = c["stage2_points_count"]
        s2_indexed = s2_counts.get(mid, 0)
        final_total = c["final_rom_points_count"]
        final_indexed = final_counts.get(mid, 0)

        # Stage 2 sync status
        if s2_total == 0:
            s2_status = "not_generated"
        elif s2_indexed == 0:
            s2_status = "not_indexed"
        elif s2_indexed < s2_total:
            s2_status = "partial"
        else:
            s2_status = "synced"

        # Final ROM sync status
        if final_total == 0:
            final_status = "not_generated"
        elif final_indexed == 0:
            final_status = "not_indexed"
        elif final_indexed < final_total:
            final_status = "partial"
        else:
            final_status = "synced"

        # A meeting needs sync if either Final ROM or Stage 2 has generated points that are not fully indexed
        needs_sync = False
        if final_total > 0 and final_indexed < final_total:
            needs_sync = True
        if s2_total > 0 and s2_indexed < s2_total:
            needs_sync = True

        # Total points & indexed points (for summary / backwards compat)
        total_pts = final_total + s2_total
        indexed_pts = final_indexed + s2_indexed
        if total_pts == 0:
            overall_status = "not_indexed"
        elif not needs_sync:
            overall_status = "synced"
        elif indexed_pts == 0:
            overall_status = "not_indexed"
        else:
            overall_status = "partial"

        results.append({
            "id": mid,
            "name": c["name"],
            "date": c["date"],
            # Stage 2
            "stage2_points_count": s2_total,
            "indexed_stage2_count": s2_indexed,
            "stage2_sync_status": s2_status,
            "has_stage2": c["has_stage2"],
            # Final ROM
            "final_rom_points_count": final_total,
            "indexed_final_rom_count": final_indexed,
            "final_rom_sync_status": final_status,
            "has_final_rom": c["has_final_rom"],
            # Summary / backward compat
            "indexed_points_count": s2_indexed,
            "needs_sync": needs_sync,
            "sync_status": overall_status,
        })

    # Sort: meetings needing sync first, then date descending
    results.sort(key=lambda m: (1 if m["needs_sync"] else 0, m["date"] or ""), reverse=True)
    return results


@router.post("/resync-meeting/{recording_id}")
async def resync_meeting(
    recording_id: str,
    force: bool = False,
    user=Depends(get_current_user),
):
    """
    Resync both Final ROM points and Stage 2 points for a meeting into ChromaDB.
    - If Final ROM points exist, indexes them into collection `final_rom_{user_id}`.
    - If Stage 2 points exist, indexes them into collection `stage2_points_{user_id}`.
    - If force=False, only missing points are added; if force=True, previous vectors are purged and re-indexed.
    - Prevents duplicates and maintains strict retrieval priority.
    """
    import asyncio
    from services.rom_service import rom_service

    user_id = user["id"]

    # 1. Fetch meeting and ROM data
    async with get_db_context() as session:
        row = (
            await session.execute(
                text(
                    "SELECT r.id, r.filename, r.title, r.created_at, "
                    "       COALESCE(rm.rom_data, r.rom_data) as rom_data "
                    "FROM recordings r "
                    "LEFT JOIN rom_metadata rm ON rm.recording_id = r.id "
                    "WHERE r.id = :rid AND r.user_id = :uid"
                ),
                {"rid": recording_id, "uid": user_id},
            )
        ).mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Meeting not found")

    rom_dict = _safe_parse_rom_data(row.get("rom_data"))
    s2_points = _extract_stage2_points(rom_dict)
    final_points = _extract_final_rom_points(rom_dict)

    if not s2_points and not final_points:
        raise HTTPException(
            status_code=400,
            detail="No Stage 2 or Final ROM discussion points found for this meeting",
        )

    meeting_name = row.get("title") or row.get("filename") or "Meeting"
    created_val = row.get("created_at")
    meeting_date = str(created_val)[:10] if created_val else ""

    loop = asyncio.get_running_loop()

    def _sync_in_background():
        res_final = {"total_found": 0, "already_indexed": 0, "newly_indexed": 0, "skipped_failed": 0}
        res_s2 = {"total_found": 0, "already_indexed": 0, "newly_indexed": 0, "skipped_failed": 0}

        # 1. Sync Final ROM if available
        if final_points:
            final_rom_obj = rom_dict.get("final_rom") or rom_dict.get("stage3") or {}
            res_final = rom_service.index_final_rom_points_in_chromadb(
                recording_id=recording_id,
                user_id=user_id,
                final_rom=final_rom_obj,
                meeting_name=meeting_name,
                meeting_date=meeting_date,
                version=rom_dict.get("final_rom_active_version") or "long",
                only_missing=not force,
                return_details=True,
            )

        # 2. Sync Stage 2 if available
        if s2_points:
            res_s2 = rom_service.index_stage2_points_in_chromadb(
                recording_id=recording_id,
                user_id=user_id,
                points=s2_points,
                meeting_name=meeting_name,
                meeting_date=meeting_date,
                only_missing=not force,
                return_details=True,
            )

        return res_final, res_s2

    final_res, s2_res = await loop.run_in_executor(None, _sync_in_background)

    total_found = final_res.get("total_found", 0) + s2_res.get("total_found", 0)
    already_indexed = final_res.get("already_indexed", 0) + s2_res.get("already_indexed", 0)
    newly_indexed = final_res.get("newly_indexed", 0) + s2_res.get("newly_indexed", 0)
    skipped_failed = final_res.get("skipped_failed", 0) + s2_res.get("skipped_failed", 0)

    msg_parts = []
    if final_points:
        msg_parts.append(
            f"Final ROM: {final_res.get('newly_indexed', 0)} new, {final_res.get('already_indexed', 0)} existing"
        )
    if s2_points:
        msg_parts.append(
            f"Stage 2: {s2_res.get('newly_indexed', 0)} new, {s2_res.get('already_indexed', 0)} existing"
        )

    return {
        "status": "success",
        "recording_id": recording_id,
        "meeting_name": meeting_name,
        "total_found": total_found,
        "already_indexed": already_indexed,
        "newly_indexed": newly_indexed,
        "skipped_failed": skipped_failed,
        "final_rom_result": final_res,
        "stage2_result": s2_res,
        "message": f"Successfully synced — {', '.join(msg_parts)}.",
    }


# ── Manual MoM Upload & Management Endpoints ─────────────────────────────────

@router.post("/manual-mom/upload")
async def upload_manual_moms(
    files: List[UploadFile] = File(...),
    user=Depends(get_current_user),
):
    """
    Upload multiple manual MoM files at once (.pdf, .docx, .doc, .txt, .md, .pptx, etc.).
    Extracts content, detects metadata, chunks, and ingests directly into ChromaDB.
    """
    from services.manual_mom_service import process_manual_mom_file

    results = []
    errors = []

    for file in files:
        filename = file.filename or "uploaded_mom.txt"
        try:
            content = await file.read()
            if not content:
                errors.append({"filename": filename, "error": "File is empty"})
                continue

            item = await process_manual_mom_file(
                user_id=user["id"],
                file_bytes=content,
                filename=filename,
            )
            results.append(item)
        except Exception as e:
            logger.error(f"[Manual MoM Router] Error processing {filename}: {e}", exc_info=True)
            errors.append({"filename": filename, "error": str(e)})

    return {
        "status": "success" if results else "error",
        "results": results,
        "success_count": len(results),
        "fail_count": len(errors),
        "errors": errors,
    }


@router.get("/manual-mom/list")
async def get_manual_moms_list(user=Depends(get_current_user)):
    """List all manual MoM uploads for the current user."""
    from services.manual_mom_service import list_manual_moms
    return await list_manual_moms(user["id"])


@router.get("/manual-mom/{mom_id}")
async def get_manual_mom_by_id(mom_id: str, user=Depends(get_current_user)):
    """Get detailed extracted content, metadata, and chunks for a manual MoM."""
    from services.manual_mom_service import get_manual_mom_detail
    detail = await get_manual_mom_detail(user["id"], mom_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Manual MoM not found")
    return detail


@router.delete("/manual-mom/{mom_id}")
async def delete_manual_mom_endpoint(mom_id: str, user=Depends(get_current_user)):
    """Delete a manual MoM and remove its vectors from ChromaDB."""
    from services.manual_mom_service import delete_manual_mom
    success = await delete_manual_mom(user["id"], mom_id)
    return {"status": "success", "deleted": success, "mom_id": mom_id}


@router.post("/manual-mom/{mom_id}/reindex")
async def reindex_manual_mom_endpoint(mom_id: str, user=Depends(get_current_user)):
    """Re-chunk and re-embed a manual MoM in ChromaDB."""
    from services.manual_mom_service import reindex_manual_mom
    try:
        return await reindex_manual_mom(user["id"], mom_id)
    except Exception as e:
        logger.error(f"[Manual MoM Router] Reindex failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

