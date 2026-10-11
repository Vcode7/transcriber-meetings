"""
Unit test for AI Chat Sessions and independent conversation history.
"""
import os
import sys
import uuid
import pytest
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import connect_db
from routers.ai_chat_router import (
    _get_chat_history,
    _save_message,
    create_chat_session_endpoint,
    get_chat_sessions_endpoint,
    update_chat_session_endpoint,
    delete_chat_session_endpoint,
    get_chat_history_endpoint,
    clear_chat_history_endpoint,
    AIChatSessionCreate,
    AIChatSessionUpdate,
)


@pytest.mark.asyncio
async def test_ai_chat_sessions_isolation():
    await connect_db()
    test_user_id = f"test-user-{uuid.uuid4()}"
    mock_user = {"id": test_user_id}

    # 1. Create two distinct sessions
    sess1 = await create_chat_session_endpoint(
        AIChatSessionCreate(title="Project Discussion"),
        user=mock_user,
    )
    sess2 = await create_chat_session_endpoint(
        AIChatSessionCreate(title="Financial Review"),
        user=mock_user,
    )

    s1_id = sess1["id"]
    s2_id = sess2["id"]
    assert s1_id != s2_id

    # 2. Add messages to Session 1
    await _save_message(test_user_id, "user", "What is the project deadline?", session_id=s1_id)
    await _save_message(test_user_id, "assistant", "The project deadline is Friday.", session_id=s1_id)

    # 3. Add messages to Session 2
    await _save_message(test_user_id, "user", "What is the total budget allocated?", session_id=s2_id)
    await _save_message(test_user_id, "assistant", "The total budget is $50,000.", session_id=s2_id)

    # 4. Verify Session 1 only has Session 1's messages
    hist1 = await _get_chat_history(test_user_id, session_id=s1_id)
    assert len(hist1) == 2
    assert "project deadline" in hist1[0]["content"]
    assert "Friday" in hist1[1]["content"]

    # 5. Verify Session 2 only has Session 2's messages
    hist2 = await _get_chat_history(test_user_id, session_id=s2_id)
    assert len(hist2) == 2
    assert "total budget" in hist2[0]["content"]
    assert "$50,000" in hist2[1]["content"]

    # 6. Verify list sessions endpoint returns both
    all_sessions = await get_chat_sessions_endpoint(user=mock_user)
    session_ids = [s["id"] for s in all_sessions]
    assert s1_id in session_ids
    assert s2_id in session_ids

    # 7. Update Session 1 title
    await update_chat_session_endpoint(s1_id, AIChatSessionUpdate(title="Updated Project Plan"), user=mock_user)
    all_sessions_after_update = await get_chat_sessions_endpoint(user=mock_user)
    updated_s1 = next(s for s in all_sessions_after_update if s["id"] == s1_id)
    assert updated_s1["title"] == "Updated Project Plan"

    # 8. Delete Session 1 and verify Session 2 remains intact
    await delete_chat_session_endpoint(s1_id, user=mock_user)
    remaining_sessions = await get_chat_sessions_endpoint(user=mock_user)
    remaining_ids = [s["id"] for s in remaining_sessions]
    assert s1_id not in remaining_ids
    assert s2_id in remaining_ids

    # Messages in Session 1 should be cleaned up
    hist1_after_del = await _get_chat_history(test_user_id, session_id=s1_id)
    assert len(hist1_after_del) == 0

    # Messages in Session 2 still intact
    hist2_still_there = await _get_chat_history(test_user_id, session_id=s2_id)
    assert len(hist2_still_there) == 2

    print("\n[SUCCESS] AI Chat Sessions isolation & CRUD tests passed!")


@pytest.mark.asyncio
async def test_ai_chat_session_reload_and_persistence():
    """Verify reload persistence, client ID forwarding, and auto-healing of orphaned sessions."""
    await connect_db()
    test_user_id = f"test-reload-{uuid.uuid4()}"
    mock_user = {"id": test_user_id}

    # 1. Client creates a session with custom client-side ID
    client_sess_id = f"chat-{uuid.uuid4()}"
    created = await create_chat_session_endpoint(
        AIChatSessionCreate(id=client_sess_id, title="Frontend Session"),
        user=mock_user,
    )
    assert created["id"] == client_sess_id

    # 2. Save user and assistant messages for this session
    await _save_message(test_user_id, "user", "Hello assistant", session_id=client_sess_id)
    await _save_message(test_user_id, "assistant", "Hello! How can I help you?", session_id=client_sess_id)

    # 3. Simulate reload: query history from DB
    reloaded_history = await get_chat_history_endpoint(session_id=client_sess_id, user=mock_user)
    assert len(reloaded_history) == 2
    assert reloaded_history[0]["role"] == "user"
    assert reloaded_history[0]["content"] == "Hello assistant"
    assert reloaded_history[1]["role"] == "assistant"
    assert reloaded_history[1]["content"] == "Hello! How can I help you?"

    # 4. Save a message to an unindexed session (orphaned session ID)
    orphaned_sid = f"chat-orphan-{uuid.uuid4()}"
    await _save_message(test_user_id, "user", "What was discussed in marketing sync?", session_id=orphaned_sid)

    # 5. Calling get_chat_sessions_endpoint auto-heals and backfills the orphaned session
    sessions_list = await get_chat_sessions_endpoint(user=mock_user)
    session_ids = [s["id"] for s in sessions_list]
    assert orphaned_sid in session_ids
    assert client_sess_id in session_ids

    # Check that auto-healed session extracted the title from user message
    orphan_sess = next(s for s in sessions_list if s["id"] == orphaned_sid)
    assert "marketing sync" in orphan_sess["title"].lower()

    # 6. Verify cross-session pruning does NOT wipe other sessions
    # Add messages to orphaned_sid
    for i in range(10):
        await _save_message(test_user_id, "user", f"Message {i}", session_id=orphaned_sid)

    # Verify client_sess_id still has its original 2 messages intact
    client_history = await get_chat_history_endpoint(session_id=client_sess_id, user=mock_user)
    assert len(client_history) == 2

    print("\n[SUCCESS] AI Chat reload persistence & auto-healing tests passed!")
