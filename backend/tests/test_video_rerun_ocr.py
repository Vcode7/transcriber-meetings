import pytest
import pytest_asyncio
import asyncio
import os
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from sqlalchemy import text

from database import connect_db, close_db, get_db, get_db_context, to_json, from_json
from routers.video_router import rerun_video_ocr, get_ocr_status, get_video_transcript, _run_ocr_task

@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    await connect_db()
    yield



@pytest.mark.asyncio
async def test_rerun_video_ocr_missing_recording():
    """Verify 404 error returned when attempting to rerun OCR for non-existent recording."""
    user = {"id": "user_test_123"}
    with pytest.raises(HTTPException) as exc_info:
        await rerun_video_ocr(recording_id="non_existent_rec_id", current_user=user)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_rerun_video_ocr_missing_file(tmp_path):
    """Verify 400 error returned when video file path does not exist on disk."""
    user = {"id": "user_test_456"}
    rec_id = "rec_missing_video_file"

    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO recordings (id, user_id, filename, file_path, duration, status, source_type, created_at)
                VALUES (:id, :uid, 'test.mp4', :fp, 10.0, 'done', 'video', '2026-07-27T00:00:00Z')
            """),
            {"id": rec_id, "uid": user["id"], "fp": str(tmp_path / "non_existent_video.mp4")},
        )
        await db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await rerun_video_ocr(recording_id=rec_id, current_user=user)
    assert exc_info.value.status_code == 400
    assert "Video source file is no longer available" in exc_info.value.detail

    # Clean up DB
    async with get_db_context() as db:
        await db.execute(text("DELETE FROM recordings WHERE id = :id"), {"id": rec_id})
        await db.commit()


@pytest.mark.asyncio
async def test_rerun_video_ocr_preserves_audio_and_summary_data(tmp_path):
    """Verify that re-running Video OCR updates video_transcript while leaving audio transcript and AI summary untouched."""
    user = {"id": "user_test_789"}
    rec_id = "rec_video_rerun_test"

    video_file = tmp_path / "sample_video.mp4"
    video_file.write_bytes(b"fake_video_bytes")

    initial_audio_transcript = [{"start": 0.0, "end": 5.0, "text": "Welcome to the meeting", "speaker": "Speaker 1"}]
    initial_summary = "This is a test summary."
    initial_key_points = ["Point 1", "Point 2"]
    old_video_transcript = [{"start": 0.0, "end": 10.0, "text": "Old Slide Text"}]

    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO recordings (
                    id, user_id, filename, file_path, duration, status, source_type,
                    transcript, summary, key_points, video_transcript, created_at
                )
                VALUES (
                    :id, :uid, 'sample_video.mp4', :fp, 15.0, 'done', 'video',
                    :trans, :sum, :kp, :vt, '2026-07-27T00:00:00Z'
                )
            """),
            {
                "id": rec_id,
                "uid": user["id"],
                "fp": str(video_file),
                "trans": to_json(initial_audio_transcript),
                "sum": initial_summary,
                "kp": to_json(initial_key_points),
                "vt": to_json(old_video_transcript),
            },
        )
        await db.commit()

    # Mock the OCR pipeline extraction & merging functions
    mock_raw_ocr = [{"start_time": 2.0, "text": "New Slide Title: AntiGravity Architecture"}]
    mock_merged = [{"start": 2.0, "end": 15.0, "text": "New Slide Title: AntiGravity Architecture"}]

    with patch("routers.video_router.extract_video_ocr_timeline", return_value=mock_raw_ocr), \
         patch("routers.video_router.merge_ocr_results", return_value=mock_merged):

        # Execute OCR task directly
        await _run_ocr_task(rec_id, str(video_file), 15.0)

    # Verify DB state after OCR task completion
    async with get_db_context() as db:
        r = await db.execute(text("SELECT * FROM recordings WHERE id = :id"), {"id": rec_id})
        rec = r.mappings().fetchone()

    assert rec is not None
    # 1. Spoken audio transcript and AI insights MUST remain untouched
    assert from_json(rec["transcript"]) == initial_audio_transcript
    assert rec["summary"] == initial_summary
    assert from_json(rec["key_points"]) == initial_key_points

    # 2. video_transcript MUST be updated with the newly generated OCR blocks
    updated_vt = from_json(rec["video_transcript"])
    assert updated_vt == mock_merged
    assert updated_vt != old_video_transcript

    # Clean up DB
    async with get_db_context() as db:
        await db.execute(text("DELETE FROM recordings WHERE id = :id"), {"id": rec_id})
        await db.commit()
