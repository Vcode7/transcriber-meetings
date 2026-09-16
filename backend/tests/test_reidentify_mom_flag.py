import pytest
import json
from unittest.mock import patch, AsyncMock
from sqlalchemy import text
from database import connect_db, get_db_context
from tasks.rereid_pipeline import _run_reidentify_impl
from routers.history import ReidentifySpeakersRequest


@pytest.mark.asyncio
async def test_reidentify_request_model():
    """Verify ReidentifySpeakersRequest model validation."""
    req_default = ReidentifySpeakersRequest()
    assert req_default.regenerate_mom is False

    req_false = ReidentifySpeakersRequest(regenerate_mom=False)
    assert req_false.regenerate_mom is False

    req_true = ReidentifySpeakersRequest(regenerate_mom=True)
    assert req_true.regenerate_mom is True


@pytest.mark.asyncio
async def test_rereid_pipeline_never_auto_regenerates_mom_on_speaker_change(tmp_path):
    """Verify that by default (even when speakers change and generate_mom_auto=1), MoM is NEVER regenerated."""
    await connect_db()
    dummy_audio = tmp_path / "test_audio.wav"
    dummy_audio.write_bytes(b"dummy")

    recording_id = "test_reid_rec_never_auto_mom"
    user_id = "test_user_reid_never_mom"

    # Set up DB record with old speakers
    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT OR REPLACE INTO recordings (
                    id, user_id, filename, file_path, duration, status, progress,
                    transcript, raw_text, speakers_detected, created_at
                ) VALUES (
                    :id, :uid, 'test.wav', :path, 10.0, 'done', NULL,
                    :transcript, 'Hello world', '["Old Speaker 1"]', '2026-09-15 12:00:00'
                )
            """),
            {
                "id": recording_id,
                "uid": user_id,
                "path": str(dummy_audio),
                "transcript": json.dumps([
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Hello world",
                        "words": [{"word": "Hello", "start": 0.0, "end": 1.0, "probability": 0.9}],
                        "speaker_label": "Old Speaker 1",
                    }
                ]),
            }
        )
        # Even with generate_mom_auto = 1 in settings
        await db.execute(
            text("""
                INSERT OR REPLACE INTO user_settings (user_id, generate_mom_auto, updated_at)
                VALUES (:uid, 1, '2026-09-15 12:00:00')
            """),
            {"uid": user_id}
        )
        await db.commit()

    with patch("tasks.rereid_pipeline.diarize", return_value=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]), \
         patch("tasks.rereid_pipeline.identify_speakers", return_value=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "speaker_label": "New Speaker 2", "similarity": 0.9}]), \
         patch("tasks.rereid_pipeline.refine_transcript_speakers_with_ecapa", side_effect=lambda **kwargs: kwargs.get("speaker_segments", [])), \
         patch("tasks.rereid_pipeline._regenerate_mom", new_callable=AsyncMock) as mock_regen_mom:

        # Default run_reidentify with regenerate_mom=False (default behavior during speaker re-run)
        await _run_reidentify_impl(
            recording_id=recording_id,
            file_path=str(dummy_audio),
            user_id=user_id,
            regenerate_mom=False,
        )

        # MoM regeneration must NEVER have been called!
        mock_regen_mom.assert_not_called()


@pytest.mark.asyncio
async def test_rereid_pipeline_triggers_mom_only_when_explicitly_requested(tmp_path):
    """Verify that MoM is regenerated ONLY when explicitly requested with regenerate_mom=True."""
    await connect_db()
    dummy_audio = tmp_path / "test_audio.wav"
    dummy_audio.write_bytes(b"dummy")

    recording_id = "test_reid_rec_explicit_true"
    user_id = "test_user_reid_explicit"

    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT OR REPLACE INTO recordings (
                    id, user_id, filename, file_path, duration, status, progress,
                    transcript, raw_text, speakers_detected, created_at
                ) VALUES (
                    :id, :uid, 'test.wav', :path, 10.0, 'done', NULL,
                    :transcript, 'Hello world', '["Old Speaker 1"]', '2026-09-15 12:00:00'
                )
            """),
            {
                "id": recording_id,
                "uid": user_id,
                "path": str(dummy_audio),
                "transcript": json.dumps([
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Hello world",
                        "words": [{"word": "Hello", "start": 0.0, "end": 1.0, "probability": 0.9}],
                        "speaker_label": "Old Speaker 1",
                    }
                ]),
            }
        )
        await db.commit()

    with patch("tasks.rereid_pipeline.diarize", return_value=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]), \
         patch("tasks.rereid_pipeline.identify_speakers", return_value=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "speaker_label": "New Speaker 2", "similarity": 0.9}]), \
         patch("tasks.rereid_pipeline.refine_transcript_speakers_with_ecapa", side_effect=lambda **kwargs: kwargs.get("speaker_segments", [])), \
         patch("tasks.rereid_pipeline.asyncio.create_task") as mock_create_task:

        # Run reidentify with explicit regenerate_mom=True
        await _run_reidentify_impl(
            recording_id=recording_id,
            file_path=str(dummy_audio),
            user_id=user_id,
            regenerate_mom=True,
        )

        assert mock_create_task.called
