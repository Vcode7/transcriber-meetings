import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from services.identification import refine_transcript_speakers_with_ecapa


def test_refinement_no_evidence_overrides_generic_label():
    """
    Test that when an original label has no enrolled profile and no other segments
    to form a conversation centroid (no evidence), original_label_sim defaults to 0.0.
    This allows an enrolled speaker match with realistic confidence (e.g. 0.65)
    to override the generic label (0.65 - 0.0 = 0.65 > 0.30 margin), whereas with
    the old 1.0 behavior it would have been rejected.
    """
    # 1 segment with generic label "Speaker 1"
    speaker_segments = [
        {"start": 0.0, "end": 2.0, "speaker": "Speaker 1", "speaker_label": "Speaker 1"}
    ]

    # Mock audio load to return dummy audio
    mock_audio = np.zeros(32000, dtype=np.float32)
    sr = 16000

    # Dummy embedding for segment (192-d for ECAPA)
    emb = np.zeros(192, dtype=np.float32)
    emb[0] = 1.0

    # Voice profile for Alice with an embedding that yields ~0.65 cosine similarity
    alice_emb = np.zeros(192, dtype=np.float32)
    alice_emb[0] = 0.65
    alice_emb[1] = np.sqrt(1.0 - 0.65**2)

    voice_profiles = [
        {
            "_id": "alice_id",
            "label": "Alice",
            "ecapa_embeddings": [alice_emb.tolist()],
        }
    ]

    with patch("services.identification.load_audio", return_value=(mock_audio, sr)), \
         patch("services.identification.embedding_router.vad_extract_speaker_embedding", return_value=emb), \
         patch("services.identification.embedding_router.extract_embedding", return_value=emb):

        refined = refine_transcript_speakers_with_ecapa(
            file_path="dummy.wav",
            speaker_segments=speaker_segments,
            voice_profiles=voice_profiles,
            embedding_model="ecapa_tdnn",
            refinement_margin=0.30,
            refinement_high_threshold=0.82,
            refinement_min_threshold=0.20,
        )

        assert len(refined) == 1
        # Alice should have replaced Speaker 1 because 0.65 > 0.20 and 0.65 - 0.0 > 0.30
        assert refined[0]["speaker_label"] == "Alice"
        assert refined[0]["speaker_profile_id"] == "alice_id"
        assert refined[0]["similarity"] == 0.65


def test_refinement_configurable_thresholds():
    """
    Test that custom refinement thresholds (margin, high_threshold, min_threshold)
    are respected when passed to refine_transcript_speakers_with_ecapa.
    """
    speaker_segments = [
        {"start": 0.0, "end": 2.0, "speaker": "Speaker 1", "speaker_label": "Speaker 1"}
    ]

    mock_audio = np.zeros(32000, dtype=np.float32)
    sr = 16000

    emb = np.zeros(192, dtype=np.float32)
    emb[0] = 1.0

    # Match similarity of 0.50
    alice_emb = np.zeros(192, dtype=np.float32)
    alice_emb[0] = 0.50
    alice_emb[1] = np.sqrt(1.0 - 0.50**2)

    voice_profiles = [
        {
            "_id": "alice_id",
            "label": "Alice",
            "ecapa_embeddings": [alice_emb.tolist()],
        }
    ]

    with patch("services.identification.load_audio", return_value=(mock_audio, sr)), \
         patch("services.identification.embedding_router.vad_extract_speaker_embedding", return_value=emb), \
         patch("services.identification.embedding_router.extract_embedding", return_value=emb):

        # Test with strict margin = 0.60: 0.50 - 0.0 = 0.50 <= 0.60 -> Rejected!
        rejected = refine_transcript_speakers_with_ecapa(
            file_path="dummy.wav",
            speaker_segments=[dict(speaker_segments[0])],
            voice_profiles=voice_profiles,
            embedding_model="ecapa_tdnn",
            refinement_margin=0.60,
            refinement_high_threshold=0.82,
            refinement_min_threshold=0.20,
        )
        assert rejected[0]["speaker_label"] == "Speaker 1"

        # Test with high_threshold = 0.50: 0.50 >= 0.50 -> Accepted via high_threshold!
        accepted = refine_transcript_speakers_with_ecapa(
            file_path="dummy.wav",
            speaker_segments=[dict(speaker_segments[0])],
            voice_profiles=voice_profiles,
            embedding_model="ecapa_tdnn",
            refinement_margin=0.60,
            refinement_high_threshold=0.50,
            refinement_min_threshold=0.20,
        )
        assert accepted[0]["speaker_label"] == "Alice"


def test_refinement_existing_enrolled_speaker_margin_logic():
    """
    Test that when an existing segment has an enrolled profile (Bob):
    - Alice (0.65) vs Bob (0.58): margin is 0.07 <= 0.30 -> Bob remains Bob (protected).
    - Alice (0.65) vs Bob (0.25): margin is 0.40 > 0.30 -> Alice overrides Bob.
    """
    mock_audio = np.zeros(32000, dtype=np.float32)
    sr = 16000

    emb = np.zeros(192, dtype=np.float32)
    emb[0] = 1.0

    # Alice embedding (sim = 0.65)
    alice_emb = np.zeros(192, dtype=np.float32)
    alice_emb[0] = 0.65
    alice_emb[1] = np.sqrt(1.0 - 0.65**2)

    # Bob embedding (close match: sim = 0.58)
    bob_close_emb = np.zeros(192, dtype=np.float32)
    bob_close_emb[0] = 0.58
    bob_close_emb[1] = np.sqrt(1.0 - 0.58**2)

    # Bob embedding (weak match: sim = 0.25)
    bob_weak_emb = np.zeros(192, dtype=np.float32)
    bob_weak_emb[0] = 0.25
    bob_weak_emb[1] = np.sqrt(1.0 - 0.25**2)

    # Case 1: Close match -> Bob protected
    seg1 = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01", "speaker_label": "Bob", "speaker_profile_id": "bob_id"}]
    profiles_case1 = [
        {"_id": "alice_id", "label": "Alice", "ecapa_embeddings": [alice_emb.tolist()]},
        {"_id": "bob_id", "label": "Bob", "ecapa_embeddings": [bob_close_emb.tolist()]},
    ]

    with patch("services.identification.load_audio", return_value=(mock_audio, sr)), \
         patch("services.identification.embedding_router.vad_extract_speaker_embedding", return_value=emb), \
         patch("services.identification.embedding_router.extract_embedding", return_value=emb):

        res1 = refine_transcript_speakers_with_ecapa(
            file_path="dummy.wav",
            speaker_segments=seg1,
            voice_profiles=profiles_case1,
            embedding_model="ecapa_tdnn",
            refinement_margin=0.30,
        )
        assert res1[0]["speaker_label"] == "Bob"

    # Case 2: Weak match -> Alice overrides Bob
    seg2 = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01", "speaker_label": "Bob", "speaker_profile_id": "bob_id"}]
    profiles_case2 = [
        {"_id": "alice_id", "label": "Alice", "ecapa_embeddings": [alice_emb.tolist()]},
        {"_id": "bob_id", "label": "Bob", "ecapa_embeddings": [bob_weak_emb.tolist()]},
    ]

    with patch("services.identification.load_audio", return_value=(mock_audio, sr)), \
         patch("services.identification.embedding_router.vad_extract_speaker_embedding", return_value=emb), \
         patch("services.identification.embedding_router.extract_embedding", return_value=emb):

        res2 = refine_transcript_speakers_with_ecapa(
            file_path="dummy.wav",
            speaker_segments=seg2,
            voice_profiles=profiles_case2,
            embedding_model="ecapa_tdnn",
            refinement_margin=0.30,
        )
        assert res2[0]["speaker_label"] == "Alice"
        assert res2[0]["speaker_profile_id"] == "alice_id"


def test_reassignment_restricted_to_meeting_speakers_blocks_external_profiles():
    """
    Test that when restrict_to_meeting_speakers is True:
    - Reassignment can match only against the speaker profiles assigned to that meeting (Vikas, Rahul).
    - Even if an external profile (Alice) has much higher similarity (0.95 vs 0.70), Alice is rejected.
    """
    # Meeting segments: S1 -> Vikas, S3 -> Rahul
    mock_audio = np.zeros(32000, dtype=np.float32)
    sr = 16000

    emb = np.zeros(192, dtype=np.float32)
    emb[0] = 1.0

    # Alice (external profile, sim = 0.95)
    alice_emb = np.zeros(192, dtype=np.float32)
    alice_emb[0] = 0.95
    alice_emb[1] = np.sqrt(1.0 - 0.95**2)

    # Vikas (meeting profile, sim = 0.70)
    vikas_emb = np.zeros(192, dtype=np.float32)
    vikas_emb[0] = 0.70
    vikas_emb[1] = np.sqrt(1.0 - 0.70**2)

    # Rahul (meeting profile, sim = 0.40)
    rahul_emb = np.zeros(192, dtype=np.float32)
    rahul_emb[0] = 0.40
    rahul_emb[1] = np.sqrt(1.0 - 0.40**2)

    speaker_segments = [
        {"start": 0.0, "end": 2.0, "speaker": "S1", "speaker_label": "Vikas", "speaker_profile_id": "vikas_id"},
        {"start": 2.0, "end": 4.0, "speaker": "S3", "speaker_label": "Rahul", "speaker_profile_id": "rahul_id"},
        # Segment 3 has generic Speaker 2 (no evidence, baseline sim = 0.0)
        {"start": 4.0, "end": 6.0, "speaker": "S2", "speaker_label": "Speaker 2", "speaker_profile_id": None},
    ]

    # 10 profiles exist in database, including Alice (external)
    voice_profiles = [
        {"_id": "alice_id", "label": "Alice", "ecapa_embeddings": [alice_emb.tolist()]},
        {"_id": "vikas_id", "label": "Vikas", "ecapa_embeddings": [vikas_emb.tolist()]},
        {"_id": "rahul_id", "label": "Rahul", "ecapa_embeddings": [rahul_emb.tolist()]},
    ] + [
        {"_id": f"dummy_{i}", "label": f"Dummy {i}", "ecapa_embeddings": [np.zeros(192).tolist()]}
        for i in range(10)
    ]

    with patch("services.identification.load_audio", return_value=(mock_audio, sr)), \
         patch("services.identification.embedding_router.vad_extract_speaker_embedding", side_effect=[vikas_emb, rahul_emb, emb]), \
         patch("services.identification.embedding_router.extract_embedding", return_value=emb):

        refined = refine_transcript_speakers_with_ecapa(
            file_path="dummy.wav",
            speaker_segments=[dict(s) for s in speaker_segments],
            voice_profiles=voice_profiles,
            embedding_model="ecapa_tdnn",
            refinement_margin=0.20,
            refinement_high_threshold=0.80,
            refinement_min_threshold=0.20,
            restrict_to_meeting_speakers=True,
        )

        # Segment 3: Even though Alice has 0.95 similarity, Alice is external!
        # Vikas is the best MEETING profile (0.70 > 0.20), so Segment 3 matches Vikas.
        assert refined[2]["speaker_label"] == "Vikas"
        assert refined[2]["speaker_profile_id"] == "vikas_id"
        assert refined[2]["speaker"] == "S1"  # S1 ↔ Vikas mapping preserved


def test_reassignment_preserves_speaker_id_mapping():
    """
    Test that when a segment is reassigned between meeting speakers:
    - If a segment originally marked as S1 (Vikas) is determined to be Rahul,
      its speaker ID is updated to S3 (the speaker ID mapped to Rahul in this meeting).
    - Words in the segment also have their speaker and speaker_label updated to match.
    """
    mock_audio = np.zeros(32000, dtype=np.float32)
    sr = 16000

    emb = np.zeros(192, dtype=np.float32)
    emb[0] = 1.0

    # Segment audio strongly matches Rahul (sim = 0.85)
    rahul_emb = np.zeros(192, dtype=np.float32)
    rahul_emb[0] = 0.85
    rahul_emb[1] = np.sqrt(1.0 - 0.85**2)

    # Vikas embedding (weak match: sim = 0.20)
    vikas_emb = np.zeros(192, dtype=np.float32)
    vikas_emb[0] = 0.20
    vikas_emb[1] = np.sqrt(1.0 - 0.20**2)

    speaker_segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "S1",
            "speaker_label": "Vikas",
            "speaker_profile_id": "vikas_id",
            "words": [{"word": "Hello", "speaker": "S1", "speaker_label": "Vikas"}],
        },
        {
            "start": 2.0,
            "end": 4.0,
            "speaker": "S3",
            "speaker_label": "Rahul",
            "speaker_profile_id": "rahul_id",
            "words": [{"word": "Hi", "speaker": "S3", "speaker_label": "Rahul"}],
        },
    ]

    voice_profiles = [
        {"_id": "vikas_id", "label": "Vikas", "ecapa_embeddings": [vikas_emb.tolist()]},
        {"_id": "rahul_id", "label": "Rahul", "ecapa_embeddings": [rahul_emb.tolist()]},
    ]

    with patch("services.identification.load_audio", return_value=(mock_audio, sr)), \
         patch("services.identification.embedding_router.vad_extract_speaker_embedding", return_value=rahul_emb), \
         patch("services.identification.embedding_router.extract_embedding", return_value=rahul_emb):

        # Reassign with restrict_to_meeting_speakers=True
        # Segment 0 (originally S1 / Vikas) will be overridden by Rahul (1.0 - 0.20 = 0.80 > 0.30 margin)
        refined = refine_transcript_speakers_with_ecapa(
            file_path="dummy.wav",
            speaker_segments=[dict(s, words=[dict(w) for w in s["words"]]) for s in speaker_segments],
            voice_profiles=voice_profiles,
            embedding_model="ecapa_tdnn",
            refinement_margin=0.30,
            restrict_to_meeting_speakers=True,
        )

        # Segment 0 was reassigned to Rahul
        assert refined[0]["speaker_label"] == "Rahul"
        assert refined[0]["speaker_profile_id"] == "rahul_id"
        # Crucial: speaker ID must be S3 (the speaker ID mapped to Rahul in this meeting)
        assert refined[0]["speaker"] == "S3"
        # Words must be updated as well
        assert refined[0]["words"][0]["speaker"] == "S3"
        assert refined[0]["words"][0]["speaker_label"] == "Rahul"

        # Segment 1 remains Rahul with S3
        assert refined[1]["speaker_label"] == "Rahul"
        assert refined[1]["speaker"] == "S3"


def test_reassignment_unrestricted_allows_external_profiles():
    """
    Test that when restrict_to_meeting_speakers is False (the default):
    - External profiles CAN be matched and override meeting speakers as before.
    """
    mock_audio = np.zeros(32000, dtype=np.float32)
    sr = 16000

    emb = np.zeros(192, dtype=np.float32)
    emb[0] = 1.0

    # Alice (external profile, sim = 0.95)
    alice_emb = np.zeros(192, dtype=np.float32)
    alice_emb[0] = 0.95
    alice_emb[1] = np.sqrt(1.0 - 0.95**2)

    # Vikas (meeting profile, sim = 0.70)
    vikas_emb = np.zeros(192, dtype=np.float32)
    vikas_emb[0] = 0.70
    vikas_emb[1] = np.sqrt(1.0 - 0.70**2)

    # Rahul (meeting profile, sim = 0.40)
    rahul_emb = np.zeros(192, dtype=np.float32)
    rahul_emb[0] = 0.40
    rahul_emb[1] = np.sqrt(1.0 - 0.40**2)

    speaker_segments = [
        {"start": 0.0, "end": 2.0, "speaker": "S1", "speaker_label": "Vikas", "speaker_profile_id": "vikas_id"},
        {"start": 2.0, "end": 4.0, "speaker": "S3", "speaker_label": "Rahul", "speaker_profile_id": "rahul_id"},
        {"start": 4.0, "end": 6.0, "speaker": "S2", "speaker_label": "Speaker 2", "speaker_profile_id": None},
    ]

    voice_profiles = [
        {"_id": "alice_id", "label": "Alice", "ecapa_embeddings": [alice_emb.tolist()]},
        {"_id": "vikas_id", "label": "Vikas", "ecapa_embeddings": [vikas_emb.tolist()]},
        {"_id": "rahul_id", "label": "Rahul", "ecapa_embeddings": [rahul_emb.tolist()]},
    ]

    with patch("services.identification.load_audio", return_value=(mock_audio, sr)), \
         patch("services.identification.embedding_router.vad_extract_speaker_embedding", side_effect=[vikas_emb, rahul_emb, emb]), \
         patch("services.identification.embedding_router.extract_embedding", return_value=emb):

        refined = refine_transcript_speakers_with_ecapa(
            file_path="dummy.wav",
            speaker_segments=[dict(s) for s in speaker_segments],
            voice_profiles=voice_profiles,
            embedding_model="ecapa_tdnn",
            refinement_margin=0.20,
            refinement_high_threshold=0.80,
            refinement_min_threshold=0.20,
            restrict_to_meeting_speakers=False,
        )

        # In unrestricted mode, Alice (0.95) wins over Vikas (0.70) for Segment 2
        assert refined[2]["speaker_label"] == "Alice"
        assert refined[2]["speaker_profile_id"] == "alice_id"


def test_user_settings_restrict_reassignment_model():
    """
    Test that restrict_reassignment_to_meeting_speakers is False by default
    in UserSettings, and can be updated via UserSettingsUpdate.
    """
    from models.settings import UserSettings, UserSettingsUpdate

    s = UserSettings(user_id="user_123")
    assert s.restrict_reassignment_to_meeting_speakers is False

    s_on = UserSettings(user_id="user_123", restrict_reassignment_to_meeting_speakers=True)
    assert s_on.restrict_reassignment_to_meeting_speakers is True

    update = UserSettingsUpdate(restrict_reassignment_to_meeting_speakers=True)
    assert update.restrict_reassignment_to_meeting_speakers is True


@pytest.mark.asyncio
async def test_settings_router_get_and_save_restrict_reassignment():
    """
    Test that get_settings returns restrict_reassignment_to_meeting_speakers=False by default,
    and update_settings correctly persists and returns the boolean value.
    """
    from routers.settings_router import get_settings, update_settings
    from models.settings import UserSettingsUpdate
    from unittest.mock import AsyncMock, patch, MagicMock

    mock_user = {"id": "user_abc"}

    # Mock DB execute returning None for user_settings (first time user)
    mock_db = MagicMock()
    mock_result = MagicMock()
    mock_result.mappings.return_value.fetchone.return_value = None
    mock_db.execute = AsyncMock(return_value=mock_result)

    with patch("routers.settings_router.get_db_context") as mock_ctx:
        mock_ctx.return_value.__aenter__.return_value = mock_db
        mock_ctx.return_value.__aexit__ = AsyncMock()

        settings_dict = await get_settings(current_user=mock_user)
        assert "restrict_reassignment_to_meeting_speakers" in settings_dict
        assert settings_dict["restrict_reassignment_to_meeting_speakers"] is False

    # Test saving with True
    mock_update_result = MagicMock()
    mock_update_result.rowcount = 1
    mock_db.execute = AsyncMock(return_value=mock_update_result)
    mock_db.commit = AsyncMock()

    with patch("routers.settings_router.get_db_context") as mock_ctx:
        mock_ctx.return_value.__aenter__.return_value = mock_db
        mock_ctx.return_value.__aexit__ = AsyncMock()

        patch_body = UserSettingsUpdate(restrict_reassignment_to_meeting_speakers=True)
        res = await update_settings(body=patch_body, current_user=mock_user)
        assert res["restrict_reassignment_to_meeting_speakers"] is True

