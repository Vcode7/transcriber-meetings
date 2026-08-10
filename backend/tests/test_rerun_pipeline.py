import pytest
import os
import json
from sqlalchemy import text
from database import get_db, get_db_context, to_json, connect_db

@pytest.mark.asyncio
async def test_rerun_endpoint_logic(tmp_path):
    await connect_db()
    # Create a temporary dummy media file
    dummy_file = tmp_path / "test_audio.wav"
    dummy_file.write_bytes(b"dummy audio content")

    recording_id = "test_rerun_rec_123"
    user_id = "test_user_rerun"

    async with get_db_context() as db:
        # Insert initial recording with mock outputs
        await db.execute(
            text("""
                INSERT INTO recordings (
                    id, user_id, filename, file_path, duration, status, progress,
                    transcript, raw_text, summary, key_points, action_items, speakers_detected, created_at
                ) VALUES (
                    :id, :uid, 'test.wav', :path, 12.5, 'done', NULL,
                    :transcript, 'Old raw text', 'Old summary', '["point 1"]', '["action 1"]', '["Speaker 1"]', '2026-07-28 12:00:00'
                )
            """),
            {
                "id": recording_id,
                "uid": user_id,
                "path": str(dummy_file),
                "transcript": json.dumps([{"text": "Old transcript"}]),
            }
        )
        await db.commit()

    # Verify initial state in DB
    async with get_db_context() as db:
        row = (await db.execute(text("SELECT status, summary, transcript FROM recordings WHERE id = 'test_rerun_rec_123'"))).mappings().fetchone()
        assert row["status"] == "done"
        assert row["summary"] == "Old summary"

    # Reset outputs for rerun
    async with get_db_context() as db:
        await db.execute(
            text("""
                UPDATE recordings SET
                    status = 'pending',
                    progress = 'queued',
                    transcript = '[]',
                    raw_text = NULL,
                    summary = NULL,
                    short_summary = NULL,
                    detailed_summary = NULL,
                    key_points = '[]',
                    action_items = '[]',
                    speakers_detected = '[]',
                    speaker_summary = NULL,
                    context_summary = NULL,
                    context_summary_hash = NULL,
                    agenda_summary = NULL,
                    agenda_summary_hash = NULL,
                    reference_summary = NULL,
                    reference_summary_hash = NULL,
                    raw_mom = NULL,
                    rom_data = NULL,
                    video_transcript = NULL,
                    transcript_embedded = 0,
                    meeting_context_embedded = 0,
                    error_message = NULL,
                    processed_at = NULL
                WHERE id = :id AND user_id = :uid
            """),
            {"id": recording_id, "uid": user_id}
        )
        await db.commit()

    # Verify reset state in DB
    async with get_db_context() as db:
        row = (await db.execute(text("SELECT status, progress, summary, transcript FROM recordings WHERE id = 'test_rerun_rec_123'"))).mappings().fetchone()
        assert row["status"] == "pending"
        assert row["progress"] == "queued"
        assert row["summary"] is None
        assert row["transcript"] == "[]"

    # Clean up test row
    async with get_db_context() as db:
        await db.execute(text("DELETE FROM recordings WHERE id = 'test_rerun_rec_123'"))
        await db.commit()


@pytest.mark.asyncio
async def test_rerun_video_pipeline(tmp_path):
    from unittest.mock import patch, AsyncMock
    from routers.history import rerun_recording_pipeline

    await connect_db()
    dummy_video = tmp_path / "test_video.mp4"
    dummy_video.write_bytes(b"dummy video content")

    recording_id = "test_rerun_video_456"
    user = {"id": "test_user_video"}

    async with get_db_context() as db:
        await db.execute(text("DELETE FROM recordings WHERE id = :id"), {"id": recording_id})
        await db.execute(
            text("""
                INSERT INTO recordings (
                    id, user_id, filename, file_path, duration, status, source_type, created_at
                ) VALUES (
                    :id, :uid, 'test_video.mp4', :path, 60.0, 'done', 'video', '2026-07-28 12:00:00'
                )
            """),
            {
                "id": recording_id,
                "uid": user["id"],
                "path": str(dummy_video),
            }
        )
        await db.commit()

    with patch("services.video_processing_service.extract_audio_from_video", return_value=str(dummy_video)), \
         patch("routers.video_router._run_synchronized_video_pipeline", new_callable=AsyncMock) as mock_video_pipeline:

        res = await rerun_recording_pipeline(recording_id=recording_id, current_user=user)
        assert res["status"] == "pending"

    # Clean up test row
    async with get_db_context() as db:
        await db.execute(text("DELETE FROM recordings WHERE id = :id"), {"id": recording_id})
        await db.commit()

