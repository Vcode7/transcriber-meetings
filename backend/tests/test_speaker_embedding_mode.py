"""
test_speaker_embedding_mode.py — Unit tests for switchable speaker embedding modes (ECAPA vs ERes2Net-Large).
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from models.settings import UserSettings, UserSettingsUpdate
from services import embedding_router
from services import embedding_eres2net
from config import settings


def test_speaker_embedding_setting_defaults():
    """Verify default speaker_embedding_model is 'ecapa'."""
    us = UserSettings(user_id="test_user")
    assert us.speaker_embedding_model == "ecapa"


def test_speaker_embedding_setting_update():
    """Verify UserSettingsUpdate accepts 'eres2net_large'."""
    update = UserSettingsUpdate(speaker_embedding_model="eres2net_large")
    assert update.speaker_embedding_model == "eres2net_large"


def test_embedding_router_get_and_set_active_model():
    """Test embedding router active model resolution."""
    embedding_router.set_active_model("ecapa")
    assert embedding_router.get_active_model() == "ecapa"

    embedding_router.set_active_model("eres2net_large")
    assert embedding_router.get_active_model() == "eres2net_large"

    # With user_settings row override
    assert embedding_router.get_active_model({"speaker_embedding_model": "ecapa"}) == "ecapa"
    assert embedding_router.get_active_model({"speaker_embedding_model": "eres2net_large"}) == "eres2net_large"

    # Reset
    embedding_router.set_active_model("ecapa")


def test_embedding_router_get_embedding_dim():
    """Verify embedding dimensions: 192 for ECAPA, 512 for ERes2Net-Large."""
    assert embedding_router.get_embedding_dim("ecapa") == 192
    assert embedding_router.get_embedding_dim("eres2net_large") == 512


def test_embedding_router_get_profile_embeddings():
    """Test retrieving model-specific embeddings from profile dict."""
    ecapa_vec = [0.1] * 192
    eres_vec = [0.2] * 512

    profile = {
        "id": "prof_1",
        "label": "Alice",
        "embeddings": [ecapa_vec],
        "ecapa_embeddings": [ecapa_vec],
        "eres2net_embeddings": [eres_vec],
    }

    # ECAPA should get ecapa_embeddings
    embs_ecapa = embedding_router.get_profile_embeddings(profile, model="ecapa")
    assert len(embs_ecapa) == 1
    assert len(embs_ecapa[0]) == 192

    # ERes2Net should get eres2net_embeddings
    embs_eres = embedding_router.get_profile_embeddings(profile, model="eres2net_large")
    assert len(embs_eres) == 1
    assert len(embs_eres[0]) == 512

    # Fallback to legacy embeddings for ECAPA if ecapa_embeddings is missing
    legacy_profile = {
        "id": "prof_legacy",
        "label": "Bob",
        "embeddings": [ecapa_vec],
    }
    embs_legacy = embedding_router.get_profile_embeddings(legacy_profile, model="ecapa")
    assert len(embs_legacy) == 1
    assert len(embs_legacy[0]) == 192

    # ERes2Net with legacy profile should return empty
    embs_eres_missing = embedding_router.get_profile_embeddings(legacy_profile, model="eres2net_large")
    assert embs_eres_missing == []


def test_embedding_router_get_effective_threshold():
    """Verify threshold resolution for both models."""
    # When default legacy 0.75 is passed, model defaults are used
    ecapa_thresh, _ = embedding_router.get_effective_threshold(0.75, model="ecapa", use_model_default=True)
    assert ecapa_thresh == settings.SPEAKER_SIMILARITY_THRESHOLD_ECAPA_TDNN

    eres_thresh, _ = embedding_router.get_effective_threshold(0.75, model="eres2net_large", use_model_default=True)
    assert eres_thresh == settings.SPEAKER_SIMILARITY_THRESHOLD_ERES2NET

    # Custom threshold overrides defaults
    custom_thresh, _ = embedding_router.get_effective_threshold(0.60, model="eres2net_large", use_model_default=True)
    assert custom_thresh == 0.60


def test_ensure_profile_embeddings_lazy_generation(tmp_path):
    """Verify ensure_profile_embeddings lazily generates embeddings when missing."""
    dummy_wav = tmp_path / "sample.wav"
    dummy_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    mock_vec = np.ones(512, dtype=np.float32)
    mock_vec /= np.linalg.norm(mock_vec)

    profile = {
        "id": "prof_test",
        "label": "Test User",
        "audio_paths": [str(dummy_wav)],
        "embeddings": [[0.1] * 192],
    }

    with patch("services.embedding_router.extract_embedding_from_file", return_value=mock_vec):
        res = embedding_router.ensure_profile_embeddings(profile, model="eres2net_large")
        assert len(res) == 1
        assert len(res[0]) == 512
        assert "eres2net_embeddings" in profile


def test_identification_service_dimension_and_routing():
    """Verify identification service correctly validates dimensions according to active model."""
    from services.identification import _check_embedding_dim

    # 192-d valid for ecapa
    valid_ecapa = np.zeros(192)
    assert _check_embedding_dim(valid_ecapa, "test", expected_dim=192) is True

    # 512-d valid for eres2net
    valid_eres = np.zeros(512)
    assert _check_embedding_dim(valid_eres, "test", expected_dim=512) is True

    # Mismatched dimensions return False
    assert _check_embedding_dim(valid_ecapa, "test", expected_dim=512) is False
    assert _check_embedding_dim(valid_eres, "test", expected_dim=192) is False


def test_eres2net_missing_model_error_message():
    """Verify that attempting to load ERes2Net when files are missing raises a clear actionable error."""
    embedding_eres2net.unload_encoder()
    with patch("services.model_loader.ModelLoader.get_model_path", return_value=None):
        with pytest.raises(RuntimeError) as exc_info:
            embedding_eres2net.get_encoder()
        assert "ERes2Net-Large model directory not found" in str(exc_info.value)
        assert "--with-eres2net" in str(exc_info.value)


def test_validate_embedding_dimension():
    """Verify validate_embedding_dimension helper."""
    assert embedding_router.validate_embedding_dimension([0.1] * 192, "ecapa") is True
    assert embedding_router.validate_embedding_dimension([0.1] * 512, "ecapa") is False
    assert embedding_router.validate_embedding_dimension([0.1] * 512, "eres2net_large") is True
    assert embedding_router.validate_embedding_dimension([0.1] * 192, "eres2net_large") is False
    assert embedding_router.validate_embedding_dimension(None, "ecapa") is False


def test_extract_dual_embeddings(tmp_path):
    """Verify dual embedding extraction produces both ECAPA and ERes2Net embeddings."""
    dummy_wav = tmp_path / "dual_test.wav"
    dummy_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    mock_ecapa = np.ones(192, dtype=np.float32)
    mock_eres = np.ones(512, dtype=np.float32)

    with patch("services.embedding.extract_embedding_from_file", return_value=mock_ecapa), \
         patch("services.embedding_eres2net.extract_embedding_from_file", return_value=mock_eres):
        ecapa_emb, eres_emb = embedding_router.extract_dual_embeddings(str(dummy_wav))
        assert ecapa_emb is not None
        assert len(ecapa_emb) == 192
        assert eres_emb is not None
        assert len(eres_emb) == 512


def test_identify_speakers_strict_isolation_eres2net():
    """Verify that in eres2net mode, profiles lacking eres2net embeddings are strictly skipped."""
    from services.identification import identify_speakers

    # Query embedding is 512-dim (eres2net)
    query_emb = np.ones(512, dtype=np.float32)
    query_emb /= np.linalg.norm(query_emb)

    # Profile 1 has only ECAPA (192-dim)
    # Profile 2 has ERes2Net (512-dim) matching query
    p1_ecapa = np.ones(192, dtype=np.float32).tolist()
    p2_eres = query_emb.tolist()

    profiles = [
        {"id": "p1", "label": "Bob_Legacy", "ecapa_embeddings": [p1_ecapa], "eres2net_embeddings": []},
        {"id": "p2", "label": "Alice_Modern", "ecapa_embeddings": [p1_ecapa], "eres2net_embeddings": [p2_eres]},
    ]

    with patch("services.identification.load_audio", return_value=(np.zeros(32000), 16000)), \
         patch("services.embedding_router.vad_extract_speaker_embedding", return_value=query_emb):
        # In eres2net_large mode:
        # Profile 1 should be skipped because it has no 512-d embeddings
        # Profile 2 matches query
        identified = identify_speakers(
            file_path="dummy.wav",
            diarization_segments=[{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0}],
            voice_profiles=profiles,
            similarity_threshold=0.5,
            embedding_model="eres2net_large",
        )
        assert identified[0]["speaker_label"] == "Alice_Modern"
        assert identified[0]["speaker_profile_id"] == "p2"

        # If only Profile 1 is present, it must be skipped, never fallback to ECAPA
        identified_none = identify_speakers(
            file_path="dummy.wav",
            diarization_segments=[{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0}],
            voice_profiles=[profiles[0]],
            similarity_threshold=0.5,
            embedding_model="eres2net_large",
        )
        assert identified_none[0]["speaker_label"] == "Speaker 1"
        assert identified_none[0]["speaker_profile_id"] is None


def test_identify_speakers_strict_isolation_ecapa():
    """Verify that in ecapa mode, profiles lacking ecapa embeddings are strictly skipped."""
    from services.identification import identify_speakers

    query_emb = np.ones(192, dtype=np.float32)
    query_emb /= np.linalg.norm(query_emb)

    p1_eres = np.ones(512, dtype=np.float32).tolist()
    p2_ecapa = query_emb.tolist()

    profiles = [
        {"id": "p1", "label": "Bob_EResOnly", "ecapa_embeddings": [], "eres2net_embeddings": [p1_eres]},
        {"id": "p2", "label": "Alice_ECAPA", "ecapa_embeddings": [p2_ecapa], "eres2net_embeddings": [p1_eres]},
    ]

    with patch("services.identification.load_audio", return_value=(np.zeros(32000), 16000)), \
         patch("services.embedding_router.vad_extract_speaker_embedding", return_value=query_emb):
        identified = identify_speakers(
            file_path="dummy.wav",
            diarization_segments=[{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0}],
            voice_profiles=profiles,
            similarity_threshold=0.5,
            embedding_model="ecapa",
        )
        assert identified[0]["speaker_label"] == "Alice_ECAPA"
        assert identified[0]["speaker_profile_id"] == "p2"

        identified_none = identify_speakers(
            file_path="dummy.wav",
            diarization_segments=[{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0}],
            voice_profiles=[profiles[0]],
            similarity_threshold=0.5,
            embedding_model="ecapa",
        )
        assert identified_none[0]["speaker_label"] == "Speaker 1"
        assert identified_none[0]["speaker_profile_id"] is None


def test_voice_add_profile_missing_audio_rejected():
    """Verify that creating a profile without audio returns HTTP 400."""
    from fastapi.testclient import TestClient
    from main import app
    from routers.auth import get_current_user

    mock_user = {"id": "test_user_strict", "email": "test@test.com", "name": "Test"}
    app.dependency_overrides[get_current_user] = lambda: mock_user
    with TestClient(app) as client:
        res = client.post("/voice/add-profile", json={"label": "Alice", "file_paths": []})
        assert res.status_code == 400
        assert "No audio files provided" in res.json().get("detail", "")
    app.dependency_overrides.clear()


def test_voice_sample_missing_audio_rejected():
    """Verify that creating a sample without audio returns HTTP 400 or 422."""
    from fastapi.testclient import TestClient
    from main import app
    from routers.auth import get_current_user

    mock_user = {"id": "test_user_strict", "email": "test@test.com", "name": "Test"}
    app.dependency_overrides[get_current_user] = lambda: mock_user
    with TestClient(app) as client:
        # Missing file field entirely triggers FastAPI validation error
        res = client.post("/voice/sample", data={"label": "self"})
        assert res.status_code in (400, 422)
    app.dependency_overrides.clear()


def test_voice_profile_recordings_lifecycle(tmp_path):
    """Verify voice recording preservation, playback, and removal."""
    import uuid
    import asyncio
    import json
    from fastapi.testclient import TestClient
    from main import app
    from routers.auth import get_current_user
    from database import get_db_context
    from sqlalchemy import text

    mock_user = {"id": "test_user_rec_lifecycle", "email": "test@test.com", "name": "Test"}
    app.dependency_overrides[get_current_user] = lambda: mock_user

    # Create dummy audio file
    test_audio = tmp_path / "test_sample.wav"
    test_audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    profile_id = str(uuid.uuid4())
    rec_id = str(uuid.uuid4())

    recordings = [{
        "id": rec_id,
        "filename": "test_sample.wav",
        "file_path": str(test_audio),
        "duration": 5.0,
        "created_at": "2026-09-14T00:00:00Z",
    }]
    ecapa_embs = [[0.1] * 192]
    eres_embs = [[0.2] * 512]

    # Insert test profile into database
    async def _setup_profile():
        async with get_db_context() as db:
            await db.execute(
                text("""
                    INSERT OR REPLACE INTO voice_profiles
                        (id, user_id, label, embeddings, ecapa_embeddings, eres2net_embeddings, recordings, audio_paths, sample_count, is_self, created_at, updated_at)
                    VALUES
                        (:id, :user_id, :label, :embeddings, :ecapa_embeddings, :eres2net_embeddings, :recordings, :audio_paths, 1, 0, '2026-09-14T00:00:00Z', '2026-09-14T00:00:00Z')
                """),
                {
                    "id": profile_id,
                    "user_id": mock_user["id"],
                    "label": "Test Speaker",
                    "embeddings": json.dumps(ecapa_embs),
                    "ecapa_embeddings": json.dumps(ecapa_embs),
                    "eres2net_embeddings": json.dumps(eres_embs),
                    "recordings": json.dumps(recordings),
                    "audio_paths": json.dumps([str(test_audio)]),
                }
            )
            await db.commit()

    asyncio.run(_setup_profile())

    with TestClient(app) as client:
        # 1. Verify GET /voice/profiles returns recordings and tags
        list_res = client.get("/voice/profiles")
        assert list_res.status_code == 200
        profs = [p for p in list_res.json() if p["id"] == profile_id]
        assert len(profs) == 1
        assert profs[0]["has_ecapa"] is True
        assert profs[0]["has_eres2net"] is True
        assert profs[0]["has_audio"] is True
        assert len(profs[0]["recordings"]) == 1

        # 2. Verify audio playback endpoint
        audio_res = client.get(f"/voice/profiles/{profile_id}/recordings/{rec_id}/audio")
        assert audio_res.status_code == 200

        # 3. Verify remove recording endpoint
        del_rec_res = client.delete(f"/voice/profiles/{profile_id}/recordings/{rec_id}")
        assert del_rec_res.status_code == 200
        del_rec_data = del_rec_res.json()
        assert del_rec_data["sample_count"] == 0

        # 4. Clean up profile
        del_prof = client.delete(f"/voice/profiles/{profile_id}")
        assert del_prof.status_code == 200

    app.dependency_overrides.clear()


def test_train_from_transcript_success_dual_embeddings_and_recording_preservation(tmp_path):
    """Verify /voice/train-from-transcript creates dual embeddings, persists recordings, and cleans temporary files only on success."""
    import os
    import json
    from fastapi.testclient import TestClient
    from main import app
    from routers.auth import get_current_user
    from database import get_db_context
    from sqlalchemy import text

    mock_user = {"id": "test_train_user_1", "email": "train1@test.com", "name": "Train User 1"}
    app.dependency_overrides[get_current_user] = lambda: mock_user

    # Create temporary source sample audio file
    tmp_sample = tmp_path / "sample_extract_1.wav"
    tmp_sample.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    assert tmp_sample.exists()

    mock_ecapa = [0.11] * 192
    mock_eres = [0.22] * 512

    with patch("services.embedding_router.extract_dual_embeddings", return_value=(mock_ecapa, mock_eres)), \
         patch("services.speaker_sync.sync_global_speaker_rename", return_value={"transcript": [{"speaker_label": "Vikas"}]}), \
         TestClient(app) as client:

        payload = {
            "recording_id": "rec_dummy_123",
            "speaker_label": "Speaker 1",
            "new_label": "Vikas",
            "sample_paths": [str(tmp_sample)],
        }
        res = client.post("/voice/train-from-transcript", json=payload)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()

        assert data["new_label"] == "Vikas"
        assert data["updated_segment_count"] == 1
        assert data["embedding_count"] == 1
        assert data["has_ecapa"] is True
        assert data["has_eres2net"] is True
        assert data["sample_count"] == 1
        profile_id = data["profile_id"]

        # Temporary sample must be deleted AFTER successful training
        assert not os.path.exists(str(tmp_sample)), "Temporary sample file should be cleaned up after success"

        # Verify DB records and permanent audio storage
        import asyncio
        async def _check_db():
            async with get_db_context() as db:
                r = await db.execute(
                    text("SELECT * FROM voice_profiles WHERE id = :id AND user_id = :uid"),
                    {"id": profile_id, "uid": mock_user["id"]},
                )
                prof = r.mappings().fetchone()
                assert prof is not None
                assert prof["label"] == "Vikas"

                recs = json.loads(prof["recordings"])
                assert len(recs) == 1
                perm_path = recs[0]["file_path"]
                assert os.path.exists(perm_path), f"Permanent audio file {perm_path} must exist on disk"

                ecapa_stored = json.loads(prof["ecapa_embeddings"])
                assert len(ecapa_stored) == 1
                assert len(ecapa_stored[0]) == 192

                eres_stored = json.loads(prof["eres2net_embeddings"])
                assert len(eres_stored) == 1
                assert len(eres_stored[0]) == 512

        asyncio.run(_check_db())

        # Cleanup created profile
        client.delete(f"/voice/profiles/{profile_id}")

    app.dependency_overrides.clear()


def test_train_from_transcript_failure_preserves_source_audio(tmp_path):
    """Verify /voice/train-from-transcript failure DOES NOT delete temporary source audio."""
    import os
    from fastapi.testclient import TestClient
    from main import app
    from routers.auth import get_current_user

    mock_user = {"id": "test_train_user_fail", "email": "train_fail@test.com", "name": "Train Fail User"}
    app.dependency_overrides[get_current_user] = lambda: mock_user

    tmp_sample = tmp_path / "sample_preserve_on_fail.wav"
    tmp_sample.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    assert tmp_sample.exists()

    mock_ecapa = [0.11] * 192
    mock_eres = [0.22] * 512

    # Simulate failure during speaker rename / DB step
    with patch("services.embedding_router.extract_dual_embeddings", return_value=(mock_ecapa, mock_eres)), \
         patch("services.speaker_sync.sync_global_speaker_rename", side_effect=RuntimeError("Simulated sync error")), \
         TestClient(app) as client:

        payload = {
            "recording_id": "rec_dummy_fail",
            "speaker_label": "Speaker 1",
            "new_label": "Alice Fail",
            "sample_paths": [str(tmp_sample)],
        }
        res = client.post("/voice/train-from-transcript", json=payload)
        assert res.status_code == 500

        # CRITICAL: source audio must NOT be deleted on failure
        assert os.path.exists(str(tmp_sample)), "Source audio file must be preserved when an error occurs"

    app.dependency_overrides.clear()


def test_train_from_transcript_embedding_extraction_failure_preserves_source_audio(tmp_path):
    """Verify that if embedding extraction fails, 422 is returned and source audio is preserved."""
    import os
    from fastapi.testclient import TestClient
    from main import app
    from routers.auth import get_current_user

    mock_user = {"id": "test_train_user_bad_audio", "email": "bad_audio@test.com", "name": "Bad Audio User"}
    app.dependency_overrides[get_current_user] = lambda: mock_user

    tmp_sample = tmp_path / "sample_bad_speech.wav"
    tmp_sample.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    assert tmp_sample.exists()

    # Embedding router cannot extract embeddings
    with patch("services.embedding_router.extract_dual_embeddings", return_value=(None, None)), \
         TestClient(app) as client:

        payload = {
            "recording_id": "rec_dummy_bad",
            "speaker_label": "Speaker 1",
            "new_label": "Bob Bad",
            "sample_paths": [str(tmp_sample)],
        }
        res = client.post("/voice/train-from-transcript", json=payload)
        assert res.status_code == 422
        assert "Could not extract voice embeddings" in res.json().get("detail", "")

        # Source audio must NOT be deleted
        assert os.path.exists(str(tmp_sample)), "Source audio file must be preserved on extraction failure"

    app.dependency_overrides.clear()


def test_train_from_transcript_update_existing_profile(tmp_path):
    """Verify training an existing profile merges recordings and dual embeddings without deleting prior data."""
    import os
    import json
    import uuid
    import asyncio
    from fastapi.testclient import TestClient
    from main import app
    from routers.auth import get_current_user
    from database import get_db_context
    from sqlalchemy import text

    mock_user = {"id": "test_train_user_existing", "email": "train_exist@test.com", "name": "Existing User"}
    app.dependency_overrides[get_current_user] = lambda: mock_user

    existing_audio = tmp_path / "existing_rec.wav"
    existing_audio.write_bytes(b"RIFF....")
    existing_pid = str(uuid.uuid4())
    existing_rec_id = str(uuid.uuid4())

    # Pre-populate an existing profile
    async def _setup_existing():
        async with get_db_context() as db:
            await db.execute(
                text("""
                    INSERT OR REPLACE INTO voice_profiles
                        (id, user_id, label, embeddings, ecapa_embeddings, eres2net_embeddings, recordings, audio_paths, sample_count, is_self, created_at, updated_at)
                    VALUES
                        (:id, :user_id, :label, :embeddings, :ecapa_embeddings, :eres2net_embeddings, :recordings, :audio_paths, 1, 0, '2026-09-14T00:00:00Z', '2026-09-14T00:00:00Z')
                """),
                {
                    "id": existing_pid,
                    "user_id": mock_user["id"],
                    "label": "Original Label",
                    "embeddings": json.dumps([[0.1] * 192]),
                    "ecapa_embeddings": json.dumps([[0.1] * 192]),
                    "eres2net_embeddings": json.dumps([[0.2] * 512]),
                    "recordings": json.dumps([{"id": existing_rec_id, "filename": "existing_rec.wav", "file_path": str(existing_audio), "duration": 3.0}]),
                    "audio_paths": json.dumps([str(existing_audio)]),
                }
            )
            await db.commit()

    asyncio.run(_setup_existing())

    new_sample = tmp_path / "new_training_sample.wav"
    new_sample.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    mock_ecapa_new = [0.33] * 192
    mock_eres_new = [0.44] * 512

    with patch("services.embedding_router.extract_dual_embeddings", return_value=(mock_ecapa_new, mock_eres_new)), \
         patch("services.speaker_sync.sync_global_speaker_rename", return_value={"transcript": [{"speaker_label": "Renamed"}]}), \
         TestClient(app) as client:

        payload = {
            "recording_id": "rec_dummy_append",
            "speaker_label": "Speaker 1",
            "new_label": "Renamed",
            "sample_paths": [str(new_sample)],
            "profile_id": existing_pid,
        }
        res = client.post("/voice/train-from-transcript", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["profile_id"] == existing_pid
        assert data["new_label"] == "Renamed"
        assert data["embedding_count"] == 2  # 1 original + 1 new
        assert data["sample_count"] == 2

        # Verify new sample is deleted and existing audio is untouched
        assert not os.path.exists(str(new_sample))
        assert os.path.exists(str(existing_audio))

        # Check DB
        async def _check_db():
            async with get_db_context() as db:
                r = await db.execute(
                    text("SELECT * FROM voice_profiles WHERE id = :id AND user_id = :uid"),
                    {"id": existing_pid, "uid": mock_user["id"]},
                )
                prof = r.mappings().fetchone()
                assert prof["label"] == "Renamed"
                assert len(json.loads(prof["recordings"])) == 2
                assert len(json.loads(prof["ecapa_embeddings"])) == 2
                assert len(json.loads(prof["eres2net_embeddings"])) == 2

        asyncio.run(_check_db())

        # Cleanup
        client.delete(f"/voice/profiles/{existing_pid}")

    app.dependency_overrides.clear()


def test_train_from_transcript_existing_name_without_profile_id_appends_sample(tmp_path):
    """Verify that specifying an existing name (e.g. 'p3') without profile_id automatically appends as an additional sample instead of returning 409."""
    import os
    import json
    import uuid
    import asyncio
    from fastapi.testclient import TestClient
    from main import app
    from routers.auth import get_current_user
    from database import get_db_context
    from sqlalchemy import text

    mock_user = {"id": "test_user_p3", "email": "p3@test.com", "name": "P3 User"}
    app.dependency_overrides[get_current_user] = lambda: mock_user

    existing_audio = tmp_path / "p3_orig.wav"
    existing_audio.write_bytes(b"RIFF....")
    p3_pid = str(uuid.uuid4())
    p3_rec_id = str(uuid.uuid4())

    # Pre-populate existing profile named 'p3'
    async def _setup_p3():
        async with get_db_context() as db:
            await db.execute(
                text("""
                    INSERT OR REPLACE INTO voice_profiles
                        (id, user_id, label, embeddings, ecapa_embeddings, eres2net_embeddings, recordings, audio_paths, sample_count, is_self, created_at, updated_at)
                    VALUES
                        (:id, :user_id, :label, :embeddings, :ecapa_embeddings, :eres2net_embeddings, :recordings, :audio_paths, 1, 0, '2026-09-14T00:00:00Z', '2026-09-14T00:00:00Z')
                """),
                {
                    "id": p3_pid,
                    "user_id": mock_user["id"],
                    "label": "p3",
                    "embeddings": json.dumps([[0.1] * 192]),
                    "ecapa_embeddings": json.dumps([[0.1] * 192]),
                    "eres2net_embeddings": json.dumps([[0.2] * 512]),
                    "recordings": json.dumps([{"id": p3_rec_id, "filename": "p3_orig.wav", "file_path": str(existing_audio), "duration": 3.0}]),
                    "audio_paths": json.dumps([str(existing_audio)]),
                }
            )
            await db.commit()

    asyncio.run(_setup_p3())

    new_sample = tmp_path / "p3_new_sample.wav"
    new_sample.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    mock_ecapa_new = [0.35] * 192
    mock_eres_new = [0.45] * 512

    with patch("services.embedding_router.extract_dual_embeddings", return_value=(mock_ecapa_new, mock_eres_new)), \
         patch("services.speaker_sync.sync_global_speaker_rename", return_value={"transcript": [{"speaker_label": "p3"}]}), \
         TestClient(app) as client:

        # Notice: NO profile_id is provided in payload! User simply typed 'p3'
        payload = {
            "recording_id": "rec_p3_session",
            "speaker_label": "Speaker 2",
            "new_label": "p3",
            "sample_paths": [str(new_sample)],
        }
        res = client.post("/voice/train-from-transcript", json=payload)
        assert res.status_code == 200, f"Expected 200 append, got {res.status_code}: {res.text}"
        data = res.json()

        assert data["profile_id"] == p3_pid
        assert data["new_label"] == "p3"
        assert data["embedding_count"] == 2
        assert data["sample_count"] == 2

        # Check DB that p3 was appended to
        async def _verify_p3():
            async with get_db_context() as db:
                r = await db.execute(
                    text("SELECT * FROM voice_profiles WHERE id = :id AND user_id = :uid"),
                    {"id": p3_pid, "uid": mock_user["id"]},
                )
                prof = r.mappings().fetchone()
                assert prof is not None
                assert prof["label"] == "p3"
                assert len(json.loads(prof["recordings"])) == 2
                assert len(json.loads(prof["ecapa_embeddings"])) == 2
                assert len(json.loads(prof["eres2net_embeddings"])) == 2

        asyncio.run(_verify_p3())

        # Cleanup
        client.delete(f"/voice/profiles/{p3_pid}")

    app.dependency_overrides.clear()




