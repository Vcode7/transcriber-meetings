"""
Unit tests for the Model Arena history endpoints and persistence.
"""
import json
import uuid
from datetime import datetime
import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text

from database import connect_db, get_db_context
from routers.arena_router import router as arena_router
from routers.auth import get_current_user

test_app = FastAPI()
test_app.include_router(arena_router)


@pytest.mark.asyncio
async def test_arena_history_list_and_persistence():
    """Verify that arena history entries can be saved, listed, and filtered."""
    await connect_db()
    test_user = {"id": "test_history_user", "email": "test@example.com"}
    test_app.dependency_overrides[get_current_user] = lambda: test_user

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Clear any preexisting test entries
        await client.delete("/arena/history?prompt_key=rom_version_short&meeting_id=rec-test-1")

        # Insert 2 test generation results directly into arena_history
        h1_id = str(uuid.uuid4())
        h2_id = str(uuid.uuid4())
        now1 = datetime.utcnow().isoformat()
        now2 = datetime.utcnow().isoformat()

        res1_data = {
            "id": h1_id,
            "prompt_key": "rom_version_short",
            "meeting_id": "rec-test-1",
            "model": "llama3.2",
            "output": "Run 1 output",
            "duration_ms": 1200,
            "tokens_eval": 150
        }
        res2_data = {
            "id": h2_id,
            "prompt_key": "rom_version_short",
            "meeting_id": "rec-test-1",
            "model": "mistral:latest",
            "output": "Run 2 output with new rules",
            "duration_ms": 1800,
            "tokens_eval": 210
        }

        async with get_db_context() as db:
            q = text("""
                INSERT INTO arena_history (id, user_id, prompt_key, meeting_id, recording_name, model_name, window_index, result_json, created_at)
                VALUES (:id, :uid, :prompt_key, :meeting_id, :rec_name, :model_name, :w_idx, :res_json, :created_at)
            """)
            await db.execute(q, {
                "id": h1_id, "uid": test_user["id"], "prompt_key": "rom_version_short",
                "meeting_id": "rec-test-1", "rec_name": "Meeting 1", "model_name": "llama3.2",
                "w_idx": None, "res_json": json.dumps(res1_data), "created_at": now1
            })
            await db.execute(q, {
                "id": h2_id, "uid": test_user["id"], "prompt_key": "rom_version_short",
                "meeting_id": "rec-test-1", "rec_name": "Meeting 1", "model_name": "mistral:latest",
                "w_idx": None, "res_json": json.dumps(res2_data), "created_at": now2
            })
            await db.commit()

        # 1. Fetch history for this prompt & meeting: BOTH should be returned (not overwritten)
        res = await client.get("/arena/history?prompt_key=rom_version_short&meeting_id=rec-test-1")
        assert res.status_code == 200
        items = res.json()
        assert len(items) >= 2
        item_ids = [it["id"] for it in items]
        assert h1_id in item_ids
        assert h2_id in item_ids

        # Verify full details exist in data
        h2_item = next(it for it in items if it["id"] == h2_id)
        assert h2_item["data"]["output"] == "Run 2 output with new rules"
        assert h2_item["data"]["model"] == "mistral:latest"
        assert h2_item["data"]["tokens_eval"] == 210

        # 2. Delete single history entry
        del_res = await client.delete(f"/arena/history/{h1_id}")
        assert del_res.status_code == 200

        res_after_del = await client.get("/arena/history?prompt_key=rom_version_short&meeting_id=rec-test-1")
        item_ids_after = [it["id"] for it in res_after_del.json()]
        assert h1_id not in item_ids_after
        assert h2_id in item_ids_after

        # 3. Clear remaining history
        clear_res = await client.delete("/arena/history?prompt_key=rom_version_short&meeting_id=rec-test-1")
        assert clear_res.status_code == 200

        res_after_clear = await client.get("/arena/history?prompt_key=rom_version_short&meeting_id=rec-test-1")
        assert len(res_after_clear.json()) == 0

    test_app.dependency_overrides.clear()
