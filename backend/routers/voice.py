"""Voice profile router — onboarding samples + add-voice + manage profiles."""
import logging
import uuid
from datetime import datetime, timezone
import os

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Response
from sqlalchemy import text

from database import get_db, get_db_context, dt_to_str, to_json, from_json
from routers.auth import get_current_user
from utils.storage import save_upload, delete_file
from utils.audio_utils import validate_audio, convert_to_wav
from services.embedding import extract_embedding_from_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])


# ── Upload a single voice sample ─────────────────────────────
@router.post("/sample")
async def upload_voice_sample(
    file: UploadFile = File(...),
    label: str = Form("self"),
    sample_index: int = Form(0),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    """
    Upload one voice sample. Returns the saved file path.
    Client calls this 1-3 times, then calls /finalize-setup or /add-profile.
    """
    user_id = current_user["id"]
    raw_path = await save_upload(file, user_id, prefix=f"vs_{sample_index}_")

    # Convert to WAV 16 kHz
    wav_path = raw_path.rsplit(".", 1)[0] + "_16k.wav"
    try:
        convert_to_wav(raw_path, wav_path)
    except Exception as e:
        delete_file(raw_path)
        raise HTTPException(status_code=422, detail=f"Audio conversion failed: {e}")

    delete_file(raw_path)  # keep only converted

    # Validate with user settings
    enabled = True
    min_dur = 2.0
    min_rms = 0.003
    try:
        r = await db.execute(
            text("SELECT enable_audio_validation, min_audio_duration_seconds, min_audio_rms_threshold FROM user_settings WHERE user_id = :uid"),
            {"uid": user_id}
        )
        row = r.mappings().fetchone()
        if row:
            if row.get("enable_audio_validation") is not None: enabled = bool(row["enable_audio_validation"])
            if row.get("min_audio_duration_seconds") is not None: min_dur = float(row["min_audio_duration_seconds"])
            if row.get("min_audio_rms_threshold") is not None: min_rms = float(row["min_audio_rms_threshold"])
    except Exception:
        pass

    valid, reason = validate_audio(wav_path, enabled=enabled, min_duration=min_dur, min_rms=min_rms)
    if not valid:
        delete_file(wav_path)
        raise HTTPException(status_code=422, detail=reason)

    return {"file_path": wav_path, "sample_index": sample_index, "label": label}


# ── Finalize onboarding setup ─────────────────────────────────
@router.post("/finalize-setup")
async def finalize_setup(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """
    Body: {"file_paths": [...], "label": "My Name"}
    Generates embeddings from samples and stores as the user's own voice profile.
    Marks needs_setup = False.
    """
    user_id = current_user["id"]
    file_paths = body.get("file_paths", [])
    label = body.get("label", current_user.get("name", "Me"))

    if not file_paths:
        raise HTTPException(status_code=400, detail="No voice sample files provided.")

    embeddings = []
    for fp in file_paths:
        emb = extract_embedding_from_file(fp)
        if emb is not None:
            embeddings.append(emb.tolist())

    if not embeddings:
        raise HTTPException(status_code=422, detail="Could not extract embeddings from samples. Please re-record.")

    now = datetime.now(timezone.utc)
    profile_id = str(uuid.uuid4())

    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO voice_profiles (id, user_id, label, embeddings, sample_count, is_self, created_at, updated_at)
                VALUES (:id, :user_id, :label, :embeddings, :sample_count, 1, :created_at, :updated_at)
            """),
            {
                "id": profile_id,
                "user_id": user_id,
                "label": label,
                "embeddings": to_json(embeddings),
                "sample_count": len(file_paths),
                "created_at": dt_to_str(now),
                "updated_at": dt_to_str(now),
            },
        )
        # Mark setup complete
        await db.execute(
            text("UPDATE users SET needs_setup = 0, own_profile_id = :pid WHERE id = :id"),
            {"pid": profile_id, "id": user_id},
        )
        await db.commit()

    return {
        "profile_id": profile_id,
        "label": label,
        "embedding_count": len(embeddings),
        "message": "Voice profile created. Setup complete!",
    }


# ── Skip onboarding voice profiling ───────────────────────────
@router.post("/skip-setup")
async def skip_setup(current_user: dict = Depends(get_current_user)):
    """
    Skips the voice setup onboarding step.
    Marks needs_setup = False in the database for the user.
    """
    user_id = current_user["id"]
    async with get_db_context() as db:
        await db.execute(
            text("UPDATE users SET needs_setup = 0 WHERE id = :id"),
            {"id": user_id},
        )
        await db.commit()
    return {"message": "Voice profile setup skipped."}



# ── Add an extra voice profile ────────────────────────────────
@router.post("/add-profile")
async def add_voice_profile(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """Body: {"file_paths": [...], "label": "Alice"}"""
    user_id = current_user["id"]
    file_paths = body.get("file_paths", [])
    label = body.get("label", "").strip()

    if not label:
        raise HTTPException(status_code=400, detail="Label is required.")
    if not file_paths:
        raise HTTPException(status_code=400, detail="No file paths provided.")

    embeddings = []
    for fp in file_paths:
        emb = extract_embedding_from_file(fp)
        if emb is not None:
            embeddings.append(emb.tolist())

    if not embeddings:
        raise HTTPException(status_code=422, detail="Could not extract embeddings. Please re-record with clearer audio.")

    now = datetime.now(timezone.utc)
    profile_id = str(uuid.uuid4())

    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO voice_profiles (id, user_id, label, embeddings, sample_count, is_self, created_at, updated_at)
                VALUES (:id, :user_id, :label, :embeddings, :sample_count, 0, :created_at, :updated_at)
            """),
            {
                "id": profile_id,
                "user_id": user_id,
                "label": label,
                "embeddings": to_json(embeddings),
                "sample_count": len(file_paths),
                "created_at": dt_to_str(now),
                "updated_at": dt_to_str(now),
            },
        )
        await db.commit()

    return {
        "profile_id": profile_id,
        "label": label,
        "embedding_count": len(embeddings),
    }


# ── Bulk Folder Import for Voice Profiles ──────────────────────
@router.post("/bulk-folder-import")
async def bulk_folder_import_voices(
    files: List[UploadFile] = File(...),
    relative_paths: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Import an entire root folder containing speaker subfolders.
    Folder structure:
      Root/
        Speaker1/
          sample1.wav
          sample2.wav
        Speaker2/
          sample1.wav
    Subfolder name is treated as the speaker label.
    """
    user_id = current_user["id"]
    import json as _json
    import tempfile
    import shutil
    from pathlib import Path
    from services.embedding import extract_embedding_from_file

    try:
        rel_paths = _json.loads(relative_paths)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid relative_paths JSON string.")

    if len(files) != len(rel_paths):
        raise HTTPException(status_code=400, detail="Mismatch between files and relative_paths counts.")

    supported_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".webm"}

    # Speaker -> list of temp file paths
    speaker_samples: dict[str, list[str]] = {}
    skipped_files: list[dict] = []

    temp_dir = Path(tempfile.mkdtemp(prefix="voice_bulk_"))

    try:
        for idx, upload in enumerate(files):
            rel_path = rel_paths[idx]
            filename = upload.filename or "sample.wav"
            parts = [p.strip() for p in rel_path.replace("\\", "/").split("/") if p.strip()]

            # In HTML5 webkitRelativePath, paths are e.g. "RootFolder/Speaker1/sample.wav" (3+ parts)
            # Loose files directly under RootFolder have 2 parts ("RootFolder/sample.wav") and are skipped
            if len(parts) >= 3:
                speaker_label = parts[-2]
            else:
                skipped_files.append({
                    "filename": filename,
                    "relative_path": rel_path,
                    "reason": "File is directly in root folder without a speaker subfolder"
                })
                continue

            ext = os.path.splitext(filename.lower())[1]
            if ext not in supported_exts:
                skipped_files.append({
                    "filename": filename,
                    "relative_path": rel_path,
                    "reason": f"Unsupported audio format '{ext}'"
                })
                continue

            # Save sample to temp file
            data = await upload.read()
            safe_speaker_dir = temp_dir / "".join(c for c in speaker_label if c.isalnum() or c in (" ", "_", "-")).strip()
            safe_speaker_dir.mkdir(parents=True, exist_ok=True)

            sample_file = safe_speaker_dir / f"{uuid.uuid4().hex}_{filename}"
            with open(sample_file, "wb") as f:
                f.write(data)

            if speaker_label not in speaker_samples:
                speaker_samples[speaker_label] = []
            speaker_samples[speaker_label].append(str(sample_file))

        speaker_results: list[dict] = []
        now = datetime.now(timezone.utc)

        for speaker_label, sample_paths in speaker_samples.items():
            embeddings = []
            for sample_path in sample_paths:
                emb = extract_embedding_from_file(sample_path)
                if emb is not None:
                    embeddings.append(emb.tolist())

            if not embeddings:
                speaker_results.append({
                    "speaker": speaker_label,
                    "status": "failed",
                    "error": "No clear voice embeddings could be extracted from audio samples",
                    "samples_trained": 0,
                })
                continue

            # Save or update profile in DB
            async with get_db_context() as db:
                r = await db.execute(
                    text("SELECT id, embeddings FROM voice_profiles WHERE user_id = :uid AND label = :label LIMIT 1"),
                    {"uid": user_id, "label": speaker_label},
                )
                existing = r.mappings().fetchone()

                if existing:
                    profile_id = existing["id"]
                    existing_embs = from_json(existing["embeddings"], [])
                    if existing_embs and embeddings:
                        new_dim = len(embeddings[0])
                        compatible = [e for e in existing_embs if len(e) == new_dim]
                        merged = compatible + embeddings
                    else:
                        merged = existing_embs + embeddings

                    await db.execute(
                        text("""
                            UPDATE voice_profiles
                            SET embeddings = :embeddings, sample_count = :count, updated_at = :updated_at
                            WHERE id = :id AND user_id = :uid
                        """),
                        {
                            "embeddings": to_json(merged),
                            "count": len(merged),
                            "updated_at": dt_to_str(now),
                            "id": profile_id,
                            "uid": user_id,
                        },
                    )
                    await db.commit()
                    speaker_results.append({
                        "speaker": speaker_label,
                        "status": "success",
                        "profile_id": profile_id,
                        "samples_trained": len(embeddings),
                        "action": "updated",
                    })
                else:
                    profile_id = str(uuid.uuid4())
                    await db.execute(
                        text("""
                            INSERT INTO voice_profiles (id, user_id, label, embeddings, sample_count, is_self, created_at, updated_at)
                            VALUES (:id, :user_id, :label, :embeddings, :count, 0, :created_at, :updated_at)
                        """),
                        {
                            "id": profile_id,
                            "user_id": user_id,
                            "label": speaker_label,
                            "embeddings": to_json(embeddings),
                            "count": len(embeddings),
                            "created_at": dt_to_str(now),
                            "updated_at": dt_to_str(now),
                        },
                    )
                    await db.commit()
                    speaker_results.append({
                        "speaker": speaker_label,
                        "status": "success",
                        "profile_id": profile_id,
                        "samples_trained": len(embeddings),
                        "action": "created",
                    })

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    successful = [s for s in speaker_results if s["status"] == "success"]
    failed = [s for s in speaker_results if s["status"] == "failed"]

    return {
        "total_speakers": len(speaker_samples),
        "successful_speakers": len(successful),
        "failed_speakers": len(failed),
        "speaker_results": speaker_results,
        "skipped_files": skipped_files,
    }


# ── List all profiles ─────────────────────────────────────────
@router.get("/profiles")
async def list_profiles(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT * FROM voice_profiles WHERE user_id = :uid ORDER BY created_at DESC LIMIT 50"),
            {"uid": user_id},
        )
        profiles = r.mappings().fetchall()

    return [
        {
            "id": p["id"],
            "label": p["label"],
            "sample_count": p.get("sample_count", 0),
            "is_self": bool(p.get("is_self", False)),
            "created_at": p["created_at"],
            "updated_at": p["updated_at"],
        }
        for p in profiles
    ]


# ── Rename profile ────────────────────────────────────────────
@router.put("/profiles/{profile_id}")
async def update_profile(
    profile_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]
    now = datetime.now(timezone.utc)

    async with get_db_context() as db:
        r = await db.execute(
            text("UPDATE voice_profiles SET label = :label, updated_at = :updated_at WHERE id = :id AND user_id = :uid"),
            {"label": body.get("label", ""), "updated_at": dt_to_str(now), "id": profile_id, "uid": user_id},
        )
        await db.commit()
        if r.rowcount == 0:
            raise HTTPException(status_code=404, detail="Profile not found.")

    return {"message": "Profile updated."}


# ── Delete profile ────────────────────────────────────────────
@router.delete("/profiles/{profile_id}")
async def delete_profile(
    profile_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]

    async with get_db_context() as db:
        # Fetch profile label before deleting to scrub stale speaker_mappings
        p_res = await db.execute(
            text("SELECT label FROM voice_profiles WHERE id = :id AND user_id = :uid"),
            {"id": profile_id, "uid": user_id},
        )
        p_row = p_res.mappings().fetchone()
        deleted_label = p_row["label"] if p_row else None

        r = await db.execute(
            text("DELETE FROM voice_profiles WHERE id = :id AND user_id = :uid"),
            {"id": profile_id, "uid": user_id},
        )
        if r.rowcount == 0:
            raise HTTPException(status_code=404, detail="Profile not found.")

        # Scrub deleted profile label from stored recordings.speaker_mappings
        if deleted_label:
            recs_res = await db.execute(
                text("SELECT id, speaker_mappings FROM recordings WHERE user_id = :uid AND speaker_mappings IS NOT NULL"),
                {"uid": user_id},
            )
            for rec_row in recs_res.mappings().fetchall():
                sm_raw = rec_row.get("speaker_mappings")
                sm = from_json(sm_raw, {}) if isinstance(sm_raw, str) else (sm_raw or {})
                if isinstance(sm, dict):
                    cleaned_sm = {k: v for k, v in sm.items() if k != deleted_label and v != deleted_label}
                    if len(cleaned_sm) != len(sm):
                        await db.execute(
                            text("UPDATE recordings SET speaker_mappings = :sm WHERE id = :id AND user_id = :uid"),
                            {"sm": to_json(cleaned_sm), "id": rec_row["id"], "uid": user_id},
                        )

        # If deleted own profile, mark needs_setup again
        if str(current_user.get("own_profile_id")) == profile_id:
            await db.execute(
                text("UPDATE users SET needs_setup = 1, own_profile_id = NULL WHERE id = :id"),
                {"id": user_id},
            )
        await db.commit()

    return {"message": "Profile deleted."}


# ── Check if a label is already in use ───────────────────────────────────────
@router.get("/check-label")
async def check_label(
    label: str,
    exclude_id: str = "",
    current_user: dict = Depends(get_current_user),
):
    """
    Check if a voice profile label is already in use by this user.
    Pass exclude_id to ignore a specific profile (useful when renaming).
    Returns { exists: bool, profile_id?: str }
    """
    user_id = current_user["id"]
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT id FROM voice_profiles WHERE user_id = :uid AND label = :label LIMIT 1"),
            {"uid": user_id, "label": label.strip()},
        )
        row = r.mappings().fetchone()

    if row and str(row["id"]) != exclude_id:
        return {"exists": True, "profile_id": row["id"]}
    return {"exists": False}


# ── Extract voice samples from a recording for a given speaker ────────────────
@router.post("/extract-samples")
async def extract_voice_samples(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """
    Extract 3-5 high-quality audio slices for a given speaker from a recording.

    Body: { "recording_id": str, "speaker_label": str, "max_samples": int (default 5) }

    Selection criteria:
     - segment duration >= 2.0s
     - no is_overlap
     - best avg_logprob (highest speech clarity)

    Uses ffmpeg to slice the recording file. Returns temp file paths + metadata.
    Caller must DELETE these files via /voice/delete-sample or they are cleaned up
    automatically after /voice/train-from-transcript is called.
    """
    import json as _json
    import subprocess
    import os

    user_id = current_user["id"]
    recording_id = body.get("recording_id", "")
    speaker_label = body.get("speaker_label", "")
    max_samples: int = int(body.get("max_samples", 5))

    if not recording_id or not speaker_label:
        raise HTTPException(status_code=422, detail="recording_id and speaker_label are required.")

    # Load recording
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT file_path, transcript FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    file_path = rec.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk.")

    transcript: list = _json.loads(rec.get("transcript") or "[]")

    # Filter segments for this speaker
    MIN_DURATION = 2.0
    candidates = [
        seg for seg in transcript
        if seg.get("speaker_label") == speaker_label
        and (seg.get("end", 0) - seg.get("start", 0)) >= MIN_DURATION
        and not seg.get("is_overlap", False)
    ]

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No suitable segments found for speaker '{speaker_label}'. "
                   "Segments must be at least 2s long and not overlap.",
        )

    # Sort by avg_logprob descending (higher = more confident/clear speech)
    # Fall back to duration if logprob not available
    candidates.sort(
        key=lambda s: (s.get("avg_logprob", -1.0), s.get("end", 0) - s.get("start", 0)),
        reverse=True,
    )
    selected = candidates[:max_samples]

    # Extract audio slices using ffmpeg
    from utils.storage import get_user_dir
    sample_dir = get_user_dir(user_id) / "voice_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, seg in enumerate(selected):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start + 3))
        duration = round(end - start, 2)

        out_filename = f"vt_{recording_id[:8]}_spk{i}_{int(start*100)}.wav"
        out_path = str(sample_dir / out_filename)

        try:
            run_kwargs = {
                "capture_output": True,
                "check": True,
                "timeout": 30,
            }
            if os.name == "nt":
                run_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", file_path,
                    "-ss", str(start),
                    "-t", str(duration),
                    "-ar", "16000",
                    "-ac", "1",
                    "-vn",
                    out_path,
                ],
                **run_kwargs
            )
            samples.append({
                "file_path": out_path,
                "start": start,
                "end": end,
                "duration": duration,
                "segment_text": seg.get("text", "")[:80],
            })
            logger.info(f"[ExtractSamples] Extracted sample {i+1}: {out_path} ({duration:.1f}s)")
        except subprocess.CalledProcessError as e:
            logger.warning(f"[ExtractSamples] ffmpeg failed for segment {i}: {e.stderr.decode()[:200]}")
        except Exception as e:
            logger.warning(f"[ExtractSamples] Sample {i} extraction failed: {e}")

    if not samples:
        raise HTTPException(status_code=500, detail="Failed to extract any audio samples.")

    return {"samples": samples, "speaker_label": speaker_label, "recording_id": recording_id}


# ── Train / update a voice profile from extracted samples ─────────────────────
@router.post("/train-from-transcript")
async def train_from_transcript(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """
    Train or update a voice profile using extracted sample files.
    Also relabels all matching segments in the transcript.

    Body: {
        "recording_id": str,
        "speaker_label": str,        # original label in transcript (e.g. "Speaker 1")
        "new_label": str,            # human name to assign (e.g. "Vikas")
        "sample_paths": [str],       # file paths from /voice/extract-samples
        "profile_id": str | null     # if set: update existing profile
    }

    Returns: { profile_id, new_label, updated_segment_count }
    Sample files are deleted after training.
    """
    import json as _json
    import os

    user_id = current_user["id"]
    recording_id = body.get("recording_id", "")
    speaker_label = body.get("speaker_label", "").strip()
    new_label = body.get("new_label", "").strip()
    sample_paths: list = body.get("sample_paths", [])
    existing_profile_id = body.get("profile_id") or None

    if not recording_id or not speaker_label or not new_label:
        raise HTTPException(status_code=422, detail="recording_id, speaker_label and new_label are required.")
    if not sample_paths:
        raise HTTPException(status_code=422, detail="At least one sample_path is required.")

    # Check uniqueness of new_label (skip if it belongs to the profile being updated)
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT id FROM voice_profiles WHERE user_id = :uid AND label = :label LIMIT 1"),
            {"uid": user_id, "label": new_label},
        )
        dup = r.mappings().fetchone()
    if dup and str(dup["id"]) != str(existing_profile_id or ""):
        raise HTTPException(status_code=409, detail=f"The name '{new_label}' is already used by another profile.")

    # Extract embeddings from each valid sample
    from services.embedding import extract_embedding_from_file
    embeddings = []
    for fp in sample_paths:
        if not os.path.exists(fp):
            logger.warning(f"[TrainFromTranscript] Sample file not found, skipping: {fp}")
            continue
        emb = extract_embedding_from_file(fp)
        if emb is not None:
            embeddings.append(emb.tolist())

    if not embeddings:
        raise HTTPException(
            status_code=422,
            detail="Could not extract voice embeddings from the samples. "
                   "Please ensure the samples contain clear speech.",
        )

    now = datetime.now(timezone.utc)

    async with get_db_context() as db:
        if existing_profile_id:
            # Update existing profile: merge embeddings, update label
            r = await db.execute(
                text("SELECT embeddings FROM voice_profiles WHERE id = :id AND user_id = :uid"),
                {"id": existing_profile_id, "uid": user_id},
            )
            prof = r.mappings().fetchone()
            if prof:
                existing_embs = from_json(prof["embeddings"], [])
                # Filter out stale embeddings with a different dimension (e.g. old 256-d
                # Resemblyzer vectors) — mixing dimensions breaks cosine similarity.
                if existing_embs and embeddings:
                    new_dim = len(embeddings[0])
                    compatible = [e for e in existing_embs if len(e) == new_dim]
                    if len(compatible) < len(existing_embs):
                        logger.warning(
                            f"[TrainFromTranscript] Filtered out "
                            f"{len(existing_embs) - len(compatible)} stale embeddings "
                            f"with wrong dimension from profile {existing_profile_id}. "
                            f"Expected {new_dim}-d, keeping only matching ones."
                        )
                    merged = compatible + embeddings
                else:
                    merged = existing_embs + embeddings
                await db.execute(
                    text("""
                        UPDATE voice_profiles
                        SET label = :label, embeddings = :embeddings,
                            sample_count = :count, updated_at = :updated_at
                        WHERE id = :id AND user_id = :uid
                    """),
                    {
                        "label": new_label,
                        "embeddings": to_json(merged),
                        "count": len(merged),
                        "updated_at": dt_to_str(now),
                        "id": existing_profile_id,
                        "uid": user_id,
                    },
                )
                profile_id = existing_profile_id
                logger.info(f"[TrainFromTranscript] Updated profile {profile_id} with {len(embeddings)} new embeddings")
            else:
                raise HTTPException(status_code=404, detail="Existing profile not found.")
        else:
            # Create new profile
            profile_id = str(uuid.uuid4())
            await db.execute(
                text("""
                    INSERT INTO voice_profiles
                        (id, user_id, label, embeddings, sample_count, is_self, created_at, updated_at)
                    VALUES
                        (:id, :user_id, :label, :embeddings, :count, 0, :created_at, :updated_at)
                """),
                {
                    "id": profile_id,
                    "user_id": user_id,
                    "label": new_label,
                    "embeddings": to_json(embeddings),
                    "count": len(embeddings),
                    "created_at": dt_to_str(now),
                    "updated_at": dt_to_str(now),
                },
            )
            logger.info(f"[TrainFromTranscript] Created new profile {profile_id} for label '{new_label}'")

        # Relabel matching segments and synchronize speaker mappings everywhere
        from services.speaker_sync import sync_global_speaker_rename
        updated_rec = await sync_global_speaker_rename(db, recording_id, user_id, {speaker_label: new_label})
        updated_count = len(updated_rec.get("transcript", [])) if updated_rec else 0

        await db.commit()

    # Auto-delete sample files
    for fp in sample_paths:
        try:
            if fp and os.path.exists(fp):
                os.remove(fp)
                logger.info(f"[TrainFromTranscript] Deleted sample: {fp}")
        except Exception as e:
            logger.warning(f"[TrainFromTranscript] Could not delete sample {fp}: {e}")

    logger.info(
        f"[TrainFromTranscript] Done — profile={profile_id}, label='{new_label}', "
        f"segments_updated={updated_count}, embeddings={len(embeddings)}"
    )

    # speaker_sync already updated the transcript, speakers_detected, speaker
    # mappings, and any existing MoM text fields (participants, action items)
    # inline — no background AI pipeline is needed.

    return {
        "profile_id": profile_id,
        "new_label": new_label,
        "updated_segment_count": updated_count,
        "embedding_count": len(embeddings),
    }


# ── Serve extracted sample audio file (for playback in the modal) ─────────────
@router.get("/sample-audio")
async def get_sample_audio(
    file_path: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Serve an extracted voice sample WAV file for playback.
    Only serves files that belong to the requesting user's upload directory.
    """
    import os
    from fastapi.responses import FileResponse
    from utils.storage import get_user_dir

    user_id = current_user["id"]

    # Security: ensure the requested path is within this user's directory
    user_dir = str(get_user_dir(user_id).resolve())
    requested = str(os.path.realpath(file_path))

    if not requested.startswith(user_dir):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not os.path.exists(requested):
        raise HTTPException(status_code=404, detail="Sample file not found.")

    return FileResponse(requested, media_type="audio/wav")

