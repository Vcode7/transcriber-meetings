"""
Unit tests for the Model Arena router endpoints and prompt parser.
"""
import pytest
from httpx import AsyncClient, ASGITransport

from main import app
from database import connect_db
from routers.auth import get_current_user


@pytest.mark.asyncio
async def test_arena_prompt_groups_endpoint():
    """Verify that the /arena/prompt-groups endpoint returns properly categorized stages."""
    app.dependency_overrides[get_current_user] = lambda: {"id": "test_user_id", "email": "test@example.com"}
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/arena/prompt-groups")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        group_names = [g["group"] for g in data]
        assert "Stage 1" in group_names
        assert "Stage 2" in group_names
        assert "Stage 3" in group_names
        assert "Final ROM" in group_names
        assert "MoM" in group_names

        # Verify Stage 1 has expected prompt keys
        stage1 = next(g for g in data if g["group"] == "Stage 1")
        stage1_keys = [k["key"] for k in stage1["keys"]]
        assert "rom_discussion_embedded" in stage1_keys
        assert "rom_discussion" in stage1_keys

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_arena_drafts_crud():
    """Verify that arena drafts can be listed, created, and deleted."""
    await connect_db()
    app.dependency_overrides[get_current_user] = lambda: {"id": "test_user_id", "email": "test@example.com"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Create a draft
        post_res = await client.post("/arena/drafts", json={
            "prompt_key": "rom_discussion_embedded",
            "draft_name": "Test Draft 1",
            "template": "Custom test template instruction...",
            "rule_states": {}
        })
        assert post_res.status_code == 200
        draft_id = post_res.json().get("id")
        assert draft_id is not None

        # 2. List drafts
        list_res = await client.get("/arena/drafts")
        assert list_res.status_code == 200
        drafts = list_res.json()
        assert any(d["id"] == draft_id for d in drafts)

        # 3. Delete draft
        del_res = await client.delete(f"/arena/drafts/{draft_id}")
        assert del_res.status_code == 200

        # Verify deleted
        list_res_after = await client.get("/arena/drafts")
        assert not any(d["id"] == draft_id for d in list_res_after.json())

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_arena_dependency_validation():
    """Verify that /arena/test rejects requests with missing dependencies."""
    await connect_db()
    app.dependency_overrides[get_current_user] = lambda: {"id": "test_user_id", "email": "test@example.com"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test with nonexistent recording
        res = await client.post("/arena/test", json={
            "prompt_key": "rom_discussion_embedded",
            "recording_id": "nonexistent_id",
            "model_name": "qwen2.5",
            "prompt_text": "Test prompt"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is False
        assert "Recording not found" in data["error"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_arena_with_string_rom_data():
    """Verify that string/null rom_data in database does not cause AttributeError in stage-data or test."""
    from database import get_db_context
    from sqlalchemy import text
    import uuid
    from unittest.mock import patch, MagicMock

    await connect_db()
    test_user_id = "test_user_str_rom"
    test_rec_id = f"test_rec_{uuid.uuid4().hex[:8]}"
    app.dependency_overrides[get_current_user] = lambda: {"id": test_user_id, "email": "str_rom@example.com"}

    async with get_db_context() as db:
        # Insert user settings
        await db.execute(text("""
            INSERT OR REPLACE INTO user_settings (user_id, ollama_server_url, ollama_model_priority, updated_at)
            VALUES (:uid, 'http://localhost:11434', 'qwen2.5', CURRENT_TIMESTAMP)
        """), {"uid": test_user_id})
        # Insert recording with stringified null rom_data
        await db.execute(text("""
            INSERT INTO recordings (id, user_id, filename, file_path, status, transcript, rom_data, created_at)
            VALUES (:id, :uid, 'test.wav', '/tmp/test.wav', 'done', 
                    '[{"start": 0.0, "end": 2.5, "speaker": "SPEAKER_01", "text": "Hello team"}]', 
                    '"\"null\""', CURRENT_TIMESTAMP)
        """), {"id": test_rec_id, "uid": test_user_id})
        await db.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Test GET /arena/meetings/{id}/stage-data
        stage_res = await client.get(f"/arena/meetings/{test_rec_id}/stage-data")
        assert stage_res.status_code == 200
        stage_data = stage_res.json()
        assert stage_data["has_transcript"] is True
        assert stage_data["has_stage1_points"] is False

        # 2. Test POST /arena/test with stage 1 prompt (mocking urllib.request.urlopen)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"message": {"content": "Points extracted"}, "total_duration": 1000000}'
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = await client.post("/arena/test", json={
                "prompt_key": "rom_discussion_embedded",
                "recording_id": test_rec_id,
                "model_name": "qwen2.5",
                "prompt_text": "Extract points from: {window_text}"
            })
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["output"] == "Points extracted"

    # Cleanup
    async with get_db_context() as db:
        await db.execute(text("DELETE FROM recordings WHERE id = :id"), {"id": test_rec_id})
        await db.execute(text("DELETE FROM user_settings WHERE user_id = :uid"), {"uid": test_user_id})
        await db.commit()

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_arena_short_rom_loads_points_from_meeting():
    """Verify that rom_version_short in Model Arena loads actual points and agendas from the meeting."""
    import json
    import uuid
    from unittest.mock import patch, MagicMock
    from database import get_db_context
    from sqlalchemy import text

    await connect_db()
    test_user_id = "test_user_short_rom"
    test_rec_id = f"test_rec_{uuid.uuid4().hex[:8]}"
    app.dependency_overrides[get_current_user] = lambda: {"id": test_user_id, "email": "short_rom@example.com"}

    meeting_rom_data = {
        "stage1": {
            "discussion_points": [
                {"id": "p1", "text": "Discussion on Q3 budget allocations.", "speaker": "Alice"}
            ]
        },
        "stage2": {
            "polished_points": [
                {
                    "id": "p1_polished",
                    "polished_text": "Alice discussed Q3 budget allocations and requested finance review.",
                    "speaker": "Alice",
                    "action_owner": "Alice"
                },
                {
                    "id": "p2_polished",
                    "polished_text": "Bob assigned Alice to prepare the finalized budget proposal.",
                    "speaker": "Bob",
                    "action_owner": "Alice"
                }
            ]
        },
        "stage3": {
            "agendas": [{"agenda_id": "A1", "title": "Budget Planning"}]
        },
        "final_rom_versions": {
            "long": {
                "agendas": [
                    {
                        "agenda_id": "A1",
                        "title": "Budget Planning",
                        "discussion_points": [
                            {
                                "id": "p1_polished",
                                "polished_text": "Alice discussed Q3 budget allocations and requested finance review.",
                                "speaker": "Alice",
                                "action_owner": "Alice"
                            },
                            {
                                "id": "p2_polished",
                                "polished_text": "Bob assigned Alice to prepare the finalized budget proposal.",
                                "speaker": "Bob",
                                "action_owner": "Alice"
                            }
                        ]
                    }
                ]
            }
        }
    }

    async with get_db_context() as db:
        await db.execute(text("""
            INSERT OR REPLACE INTO user_settings (user_id, ollama_server_url, ollama_model_priority, updated_at)
            VALUES (:uid, 'http://localhost:11434', 'qwen2.5', CURRENT_TIMESTAMP)
        """), {"uid": test_user_id})

        await db.execute(text("""
            INSERT INTO recordings (id, user_id, filename, file_path, status, transcript, rom_data, created_at)
            VALUES (:id, :uid, 'budget_meeting.wav', '/tmp/budget.wav', 'done', 
                    '[{"start": 0.0, "end": 10.0, "speaker": "Alice", "text": "Discussed Q3 budget."}]', 
                    :rom_data, CURRENT_TIMESTAMP)
        """), {"id": test_rec_id, "uid": test_user_id, "rom_data": json.dumps(meeting_rom_data)})
        await db.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Verify stage-data reports stage 2 points and final ROM
        stage_res = await client.get(f"/arena/meetings/{test_rec_id}/stage-data")
        assert stage_res.status_code == 200
        stage_info = stage_res.json()
        assert stage_info["has_stage2_points"] is True
        assert stage_info["stage2_point_count"] == 2
        assert stage_info["has_final_rom"] is True

        # 2. Test rom_version_short execution in arena
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "message": {"content": '["Alice to submit Q3 budget proposal by Friday."]'},
            "total_duration": 1500000,
            "eval_count": 10
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        prompt_template = "AGENDA: {agenda_title}\nPOINTS:\n{points_json}\n{rules_section}{mandatory_section}"

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = await client.post("/arena/test", json={
                "prompt_key": "rom_version_short",
                "recording_id": test_rec_id,
                "model_name": "qwen2.5",
                "prompt_text": prompt_template
            })
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True

            input_ctx = data.get("input_context", {})
            assert input_ctx.get("{agenda_title}") == "Budget Planning"

            pts_json = input_ctx.get("{points_json}")
            assert pts_json is not None
            parsed_pts = json.loads(pts_json)
            assert len(parsed_pts) == 2
            assert parsed_pts[0]["speaker"] == "Alice"
            assert "Q3 budget" in parsed_pts[0]["discussion_point"]
            assert parsed_pts[0]["action_owner"] == "Alice"

            # Check prompt was substituted without un-replaced placeholders
            assert "{agenda_title}" not in data["final_prompt"]
            assert "{points_json}" not in data["final_prompt"]
            assert "Budget Planning" in data["final_prompt"]
            assert "Alice" in data["final_prompt"]

    # Cleanup
    async with get_db_context() as db:
        await db.execute(text("DELETE FROM recordings WHERE id = :id"), {"id": test_rec_id})
        await db.execute(text("DELETE FROM user_settings WHERE user_id = :uid"), {"uid": test_user_id})
        await db.commit()

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_arena_every_prompt_has_inputs_loaded():
    """Verify that EVERY prompt registered in prompt_service has its inputs fully loaded in Model Arena."""
    import json
    import uuid
    from unittest.mock import patch, MagicMock
    from database import get_db_context
    from sqlalchemy import text
    from services.prompt_service import PROMPT_META, get_prompt_sync

    await connect_db()
    test_user_id = "test_user_all_prompts"
    test_rec_id = f"test_rec_{uuid.uuid4().hex[:8]}"
    app.dependency_overrides[get_current_user] = lambda: {"id": test_user_id, "email": "all_prompts@example.com"}

    test_rom = {
        "stage1": {
            "discussion_points": [
                {"id": "p1", "text": "Point 1 discussion on architecture.", "speaker": "Ruben"}
            ]
        },
        "stage2": {
            "polished_points": [
                {"id": "p1", "polished_text": "Point 1 polished text.", "speaker": "Ruben", "action_owner": "Ruben"}
            ]
        },
        "final_rom": {
            "agendas": [
                {
                    "agenda_id": "A1",
                    "title": "Architecture Overview",
                    "discussion_points": [
                        {"id": "p1", "polished_text": "Point 1 polished text.", "speaker": "Ruben", "action_owner": "Ruben"}
                    ]
                }
            ]
        }
    }

    async with get_db_context() as db:
        await db.execute(text("""
            INSERT OR REPLACE INTO user_settings (user_id, ollama_server_url, ollama_model_priority, updated_at)
            VALUES (:uid, 'http://localhost:11434', 'qwen2.5', CURRENT_TIMESTAMP)
        """), {"uid": test_user_id})

        await db.execute(text("""
            INSERT INTO recordings (id, user_id, filename, file_path, status, transcript, rom_data, created_at)
            VALUES (:id, :uid, 'test_all.wav', '/tmp/test_all.wav', 'done', 
                    '[{"start": 0.0, "end": 10.0, "speaker": "Ruben", "text": "Welcome to the architecture review."}]', 
                    :rom_data, CURRENT_TIMESTAMP)
        """), {"id": test_rec_id, "uid": test_user_id, "rom_data": json.dumps(test_rom)})
        await db.commit()

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "message": {"content": "Sample model response."},
        "total_duration": 1000000,
        "eval_count": 5
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("urllib.request.urlopen", return_value=mock_resp):
            for meta in PROMPT_META:
                prompt_key = meta["key"]
                template = get_prompt_sync(prompt_key)

                res = await client.post("/arena/test", json={
                    "prompt_key": prompt_key,
                    "recording_id": test_rec_id,
                    "model_name": "qwen2.5",
                    "prompt_text": template
                })
                assert res.status_code == 200, f"Failed for {prompt_key}: {res.text}"
                data = res.json()
                assert data["success"] is True, f"Prompt {prompt_key} failed: {data}"

                final_prompt = data.get("final_prompt", "")
                input_context = data.get("input_context", {})

                # Verify all declared variables for this prompt are present in input_context
                for var in meta.get("variables", []):
                    assert var in input_context, f"Variable {var} missing from input_context for {prompt_key}"
                    val = input_context[var]
                    # Verify variable is not an empty string (except optional sections)
                    if var not in ("{rules_section}", "{mandatory_section}", "{chat_history_section}", "{previous_context}", "{video_context_section}", "{reference_examples_section}"):
                        assert val != "", f"Variable {var} is unexpectedly empty for {prompt_key}"

    # Cleanup
    async with get_db_context() as db:
        await db.execute(text("DELETE FROM recordings WHERE id = :id"), {"id": test_rec_id})
        await db.execute(text("DELETE FROM user_settings WHERE user_id = :uid"), {"uid": test_user_id})
        await db.commit()

    app.dependency_overrides.clear()

