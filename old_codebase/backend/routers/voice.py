"""Voice profile router — onboarding samples + add-voice + manage profiles."""
import logging
import uuid
import shutil
from datetime import datetime, timezone
import os
from pathlib import Path

import time
from typing import List, Optional, Tuple, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import FileResponse
from sqlalchemy import text

from database import get_db, get_db_context, dt_to_str, to_json, from_json
from routers.auth import get_current_user
from utils.storage import save_upload, delete_file, get_user_dir
from utils.audio_utils import validate_audio, convert_to_wav
from services.embedding import extract_embedding_from_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])


def _get_audio_duration(file_path: str) -> float:
    """Read audio file duration in seconds."""
    try:
        import soundfile as sf
        info = sf.info(file_path)
        return round(float(info.duration), 2)
    except Exception:
        pass
    try:
        import wave
        with wave.open(file_path, "r") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return round(frames / float(rate), 2)
    except Exception:
        pass
    return 0.0


def _save_profile_recordings(
    user_id: str,
    profile_id: str,
    source_paths: list,
    original_filenames: list = None,
    source_type: str = "upload",
) -> tuple[list[dict], list[str], list[list[float]], list[list[float]]]:
    """
    Copy voice sample WAV files to permanent profile audio storage,
    create rich recording metadata objects, and extract BOTH ECAPA (192-d)
    and ERes2Net-Large (512-d) embeddings.

    Destination: {UPLOAD_DIR}/{user_id}/voice_profiles_audio/{profile_id}/
    Returns: (recordings_list, audio_paths_list, ecapa_embeddings, eres2net_embeddings)
    """
    dest_dir = get_user_dir(user_id) / "voice_profiles_audio" / profile_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    recordings_list: list[dict] = []
    audio_paths_list: list[str] = []
    ecapa_embeddings: list[list[float]] = []
    eres2net_embeddings: list[list[float]] = []

    from services.embedding_router import extract_dual_embeddings

    now_iso = dt_to_str(datetime.now(timezone.utc))

    for i, src in enumerate(source_paths):
        if not src or not os.path.exists(src):
            continue
        rec_id = str(uuid.uuid4())
        ext = Path(src).suffix or ".wav"
        dest_filename = f"rec_{int(time.time())}_{rec_id[:8]}{ext}"
        dest = dest_dir / dest_filename

        try:
            shutil.copy2(src, str(dest))
            dur = _get_audio_duration(str(dest))
            orig_name = (
                original_filenames[i]
                if (original_filenames and i < len(original_filenames) and original_filenames[i])
                else f"recording_{i+1}{ext}"
            )

            rec_obj = {
                "id": rec_id,
                "filename": orig_name,
                "file_path": str(dest),
                "duration": dur,
                "created_at": now_iso,
                "metadata": {
                    "source": source_type,
                },
            }

            # Extract dual embeddings (ECAPA 192-d + ERes2Net 512-d)
            ecapa_emb, eres_emb = extract_dual_embeddings(str(dest))
            if ecapa_emb is not None:
                ecapa_embeddings.append(ecapa_emb)
            if eres_emb is not None:
                eres2net_embeddings.append(eres_emb)

            recordings_list.append(rec_obj)
            audio_paths_list.append(str(dest))
        except Exception as e:
            logger.warning(f"[Voice] Failed to process audio {src} -> {dest}: {e}")

    # Immediately unload embedding models to free VRAM
    try:
        from services.embedding_router import unload_active_encoder
        unload_active_encoder()
    except Exception:
        pass

    return recordings_list, audio_paths_list, ecapa_embeddings, eres2net_embeddings


def _save_profile_audio(user_id: str, profile_id: str, source_paths: list) -> list:
    """Backwards-compatible wrapper returning only audio file paths."""
    _, paths, _, _ = _save_profile_recordings(user_id, profile_id, source_paths)
    return paths


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
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Audio file is required and cannot be empty.")

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
        raise HTTPException(status_code=400, detail="No voice sample files provided. You must provide an audio recording to create a voice profile.")

    now = datetime.now(timezone.utc)
    profile_id = str(uuid.uuid4())

    # Save audio recordings permanently with metadata and dual embeddings
    new_recs, new_paths, ecapa_embs, eres_embs = _save_profile_recordings(
        user_id, profile_id, file_paths, source_type="onboarding"
    )

    if not ecapa_embs and not eres_embs:
        raise HTTPException(status_code=422, detail="Could not extract voice embeddings from samples. Please re-record with clearer speech.")

    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO voice_profiles (id, user_id, label, embeddings, ecapa_embeddings, eres2net_embeddings, recordings, audio_paths, sample_count, is_self, created_at, updated_at)
                VALUES (:id, :user_id, :label, :embeddings, :ecapa_embeddings, :eres2net_embeddings, :recordings, :audio_paths, :sample_count, 1, :created_at, :updated_at)
            """),
            {
                "id": profile_id,
                "user_id": user_id,
                "label": label,
                "embeddings": to_json(ecapa_embs),
                "ecapa_embeddings": to_json(ecapa_embs),
                "eres2net_embeddings": to_json(eres_embs) if eres_embs else None,
                "recordings": to_json(new_recs),
                "audio_paths": to_json(new_paths),
                "sample_count": len(new_recs),
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
        "embedding_count": len(ecapa_embs),
        "has_ecapa": len(ecapa_embs) > 0,
        "has_eres2net": len(eres_embs) > 0,
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
        raise HTTPException(status_code=400, detail="No audio files provided. You must provide an audio recording to create a voice profile.")

    # Check if a profile with this label already exists for the user
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT id, recordings, audio_paths, ecapa_embeddings, eres2net_embeddings, embeddings FROM voice_profiles WHERE user_id = :uid AND LOWER(TRIM(label)) = LOWER(TRIM(:label)) LIMIT 1"),
            {"uid": user_id, "label": label},
        )
        existing = r.mappings().fetchone()

    target_pid = existing["id"] if existing else str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Save audio files to permanent storage for future embedding generation and playback
    new_recs, new_paths, ecapa_embs, eres_embs = _save_profile_recordings(
        user_id, target_pid, file_paths, source_type="add_profile"
    )

    if not ecapa_embs and not eres_embs:
        raise HTTPException(status_code=422, detail="Could not extract voice embeddings from the audio. Please provide clearer speech.")

    async with get_db_context() as db:
        if existing:
            existing_recs = from_json(existing.get("recordings", "[]"), [])
            existing_paths = from_json(existing.get("audio_paths", "[]"), [])
            existing_ecapa = from_json(existing.get("ecapa_embeddings") or existing.get("embeddings", "[]"), [])
            existing_eres = from_json(existing.get("eres2net_embeddings", "[]"), [])

            merged_recs = existing_recs + new_recs
            merged_paths = existing_paths + new_paths
            merged_ecapa = [e for e in existing_ecapa if len(e) == 192] + ecapa_embs
            merged_eres = [e for e in existing_eres if len(e) == 512] + eres_embs

            await db.execute(
                text("""
                    UPDATE voice_profiles
                    SET recordings = :recordings,
                        audio_paths = :audio_paths,
                        ecapa_embeddings = :ecapa_embeddings,
                        eres2net_embeddings = :eres2net_embeddings,
                        embeddings = :embeddings,
                        sample_count = :count,
                        updated_at = :updated_at
                    WHERE id = :id AND user_id = :uid
                """),
                {
                    "recordings": to_json(merged_recs),
                    "audio_paths": to_json(merged_paths),
                    "ecapa_embeddings": to_json(merged_ecapa),
                    "eres2net_embeddings": to_json(merged_eres) if merged_eres else None,
                    "embeddings": to_json(merged_ecapa),
                    "count": len(merged_recs),
                    "updated_at": dt_to_str(now),
                    "id": target_pid,
                    "uid": user_id,
                },
            )
            profile_id = target_pid
            total_embeddings = len(merged_ecapa)
            total_sample_count = len(merged_recs)
        else:
            profile_id = target_pid
            total_embeddings = len(ecapa_embs)
            total_sample_count = len(new_recs)
            await db.execute(
                text("""
                    INSERT INTO voice_profiles (id, user_id, label, embeddings, ecapa_embeddings, eres2net_embeddings, recordings, audio_paths, sample_count, is_self, created_at, updated_at)
                    VALUES (:id, :user_id, :label, :embeddings, :ecapa_embeddings, :eres2net_embeddings, :recordings, :audio_paths, :sample_count, 0, :created_at, :updated_at)
                """),
                {
                    "id": profile_id,
                    "user_id": user_id,
                    "label": label,
                    "embeddings": to_json(ecapa_embs),
                    "ecapa_embeddings": to_json(ecapa_embs),
                    "eres2net_embeddings": to_json(eres_embs) if eres_embs else None,
                    "recordings": to_json(new_recs),
                    "audio_paths": to_json(new_paths),
                    "sample_count": total_sample_count,
                    "created_at": dt_to_str(now),
                    "updated_at": dt_to_str(now),
                },
            )
        await db.commit()

    return {
        "profile_id": profile_id,
        "label": label,
        "embedding_count": total_embeddings,
        "has_ecapa": len(ecapa_embs) > 0,
        "has_eres2net": len(eres_embs) > 0,
        "sample_count": total_sample_count,
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
            # Check if profile already exists for this speaker
            async with get_db_context() as db:
                r = await db.execute(
                    text("SELECT id, embeddings, ecapa_embeddings, eres2net_embeddings, recordings, audio_paths FROM voice_profiles WHERE user_id = :uid AND label = :label LIMIT 1"),
                    {"uid": user_id, "label": speaker_label},
                )
                existing = r.mappings().fetchone()

            target_pid = existing["id"] if existing else str(uuid.uuid4())

            # Save recordings permanently and extract dual embeddings
            new_recs, new_paths, new_ecapa, new_eres = _save_profile_recordings(
                user_id, target_pid, sample_paths, source_type="bulk_import"
            )

            if not new_ecapa and not new_eres:
                speaker_results.append({
                    "speaker": speaker_label,
                    "status": "failed",
                    "error": "No clear voice embeddings could be extracted from audio samples",
                    "samples_trained": 0,
                })
                continue

            async with get_db_context() as db:
                if existing:
                    existing_recs = from_json(existing.get("recordings", "[]"), [])
                    existing_paths = from_json(existing.get("audio_paths", "[]"), [])
                    existing_ecapa = from_json(existing.get("ecapa_embeddings") or existing.get("embeddings", "[]"), [])
                    existing_eres = from_json(existing.get("eres2net_embeddings", "[]"), [])

                    merged_recs = existing_recs + new_recs
                    merged_paths = existing_paths + new_paths
                    merged_ecapa = [e for e in existing_ecapa if len(e) == 192] + new_ecapa
                    merged_eres = [e for e in existing_eres if len(e) == 512] + new_eres

                    await db.execute(
                        text("""
                            UPDATE voice_profiles
                            SET recordings = :recordings,
                                audio_paths = :audio_paths,
                                ecapa_embeddings = :ecapa_embeddings,
                                eres2net_embeddings = :eres2net_embeddings,
                                embeddings = :embeddings,
                                sample_count = :count,
                                updated_at = :updated_at
                            WHERE id = :id AND user_id = :uid
                        """),
                        {
                            "recordings": to_json(merged_recs),
                            "audio_paths": to_json(merged_paths),
                            "ecapa_embeddings": to_json(merged_ecapa),
                            "eres2net_embeddings": to_json(merged_eres) if merged_eres else None,
                            "embeddings": to_json(merged_ecapa),
                            "count": len(merged_recs),
                            "updated_at": dt_to_str(now),
                            "id": target_pid,
                            "uid": user_id,
                        },
                    )
                    await db.commit()
                    speaker_results.append({
                        "speaker": speaker_label,
                        "status": "success",
                        "profile_id": target_pid,
                        "samples_trained": len(new_recs),
                        "action": "updated",
                    })
                else:
                    await db.execute(
                        text("""
                            INSERT INTO voice_profiles (id, user_id, label, embeddings, ecapa_embeddings, eres2net_embeddings, recordings, audio_paths, sample_count, is_self, created_at, updated_at)
                            VALUES (:id, :user_id, :label, :embeddings, :ecapa_embeddings, :eres2net_embeddings, :recordings, :audio_paths, :count, 0, :created_at, :updated_at)
                        """),
                        {
                            "id": target_pid,
                            "user_id": user_id,
                            "label": speaker_label,
                            "embeddings": to_json(new_ecapa),
                            "ecapa_embeddings": to_json(new_ecapa),
                            "eres2net_embeddings": to_json(new_eres) if new_eres else None,
                            "recordings": to_json(new_recs),
                            "audio_paths": to_json(new_paths),
                            "count": len(new_recs),
                            "created_at": dt_to_str(now),
                            "updated_at": dt_to_str(now),
                        },
                    )
                    await db.commit()
                    speaker_results.append({
                        "speaker": speaker_label,
                        "status": "success",
                        "profile_id": target_pid,
                        "samples_trained": len(new_recs),
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

    result = []
    for p in profiles:
        # 1. Parse recordings
        raw_recs = p.get("recordings")
        recs = from_json(raw_recs, []) if isinstance(raw_recs, str) else (raw_recs or [])
        if not recs:
            # Check legacy audio_paths
            raw_paths = p.get("audio_paths")
            paths = from_json(raw_paths, []) if isinstance(raw_paths, str) else (raw_paths or [])
            recs = [
                {
                    "id": f"rec_{i}",
                    "filename": Path(fp).name,
                    "file_path": fp,
                    "duration": _get_audio_duration(fp) if os.path.exists(fp) else 0.0,
                    "created_at": p["created_at"],
                }
                for i, fp in enumerate(paths)
                if fp and os.path.exists(fp)
            ]

        # Valid recordings on disk
        valid_recs = [r for r in recs if r.get("file_path") and os.path.exists(r["file_path"])]
        has_audio = len(valid_recs) > 0

        # Check embeddings
        ecapa_raw = p.get("ecapa_embeddings")
        if ecapa_raw is None or ecapa_raw == "null":
            ecapa_raw = p.get("embeddings")
        ecapa_list = from_json(ecapa_raw, []) if isinstance(ecapa_raw, str) else (ecapa_raw or [])
        has_ecapa = any(hasattr(e, "__len__") and len(e) == 192 for e in ecapa_list)

        eres_raw = p.get("eres2net_embeddings")
        eres_list = from_json(eres_raw, []) if isinstance(eres_raw, str) else (eres_raw or [])
        has_eres2net = any(hasattr(e, "__len__") and len(e) == 512 for e in eres_list)

        result.append({
            "id": p["id"],
            "label": p["label"],
            "sample_count": len(valid_recs) if valid_recs else p.get("sample_count", 0),
            "is_self": bool(p.get("is_self", False)),
            "created_at": p["created_at"],
            "updated_at": p["updated_at"],
            "has_ecapa": has_ecapa,
            "has_eres2net": has_eres2net,
            "has_audio": has_audio,
            "recordings": [
                {
                    "id": r["id"],
                    "filename": r.get("filename") or Path(r.get("file_path", "audio.wav")).name,
                    "duration": r.get("duration", 0.0),
                    "created_at": r.get("created_at", p["created_at"]),
                }
                for r in valid_recs
            ],
        })

    return result


# ── Serve profile recording audio for playback ────────────────────────
@router.get("/profiles/{profile_id}/recordings/{recording_id}/audio")
async def get_profile_recording_audio(
    profile_id: str,
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Stream a saved voice recording audio file for in-browser playback."""
    user_id = current_user["id"]
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT recordings, audio_paths FROM voice_profiles WHERE id = :id AND user_id = :uid"),
            {"id": profile_id, "uid": user_id},
        )
        row = r.mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Profile not found.")

    raw_recs = row.get("recordings")
    recs = from_json(raw_recs, []) if isinstance(raw_recs, str) else (raw_recs or [])
    target_path = None

    for r in recs:
        if str(r.get("id")) == str(recording_id):
            target_path = r.get("file_path")
            break

    if not target_path:
        raw_paths = row.get("audio_paths")
        paths = from_json(raw_paths, []) if isinstance(raw_paths, str) else (raw_paths or [])
        if recording_id.isdigit():
            idx = int(recording_id)
            if 0 <= idx < len(paths):
                target_path = paths[idx]
        elif recording_id.startswith("rec_") and recording_id[4:].isdigit():
            idx = int(recording_id[4:])
            if 0 <= idx < len(paths):
                target_path = paths[idx]

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Recording audio file not found on disk.")

    return FileResponse(target_path, media_type="audio/wav", filename=Path(target_path).name)


# ── Add a new recording to an existing profile (Add More / Train) ─────
@router.post("/profiles/{profile_id}/recordings")
async def add_recording_to_profile(
    profile_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    """
    Upload and attach an additional voice recording to an existing profile.
    Generates BOTH ECAPA and ERes2Net-Large embeddings, and stores the audio file permanently.
    """
    user_id = current_user["id"]
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Audio file is required and cannot be empty.")

    r = await db.execute(
        text("SELECT * FROM voice_profiles WHERE id = :id AND user_id = :uid"),
        {"id": profile_id, "uid": user_id},
    )
    prof = r.mappings().fetchone()
    if not prof:
        raise HTTPException(status_code=404, detail="Profile not found.")

    # Save uploaded file to temp and convert to 16 kHz WAV
    raw_path = await save_upload(file, user_id, prefix="add_rec_")
    wav_path = raw_path.rsplit(".", 1)[0] + "_16k.wav"
    try:
        convert_to_wav(raw_path, wav_path)
    except Exception as e:
        delete_file(raw_path)
        raise HTTPException(status_code=422, detail=f"Audio conversion failed: {e}")
    delete_file(raw_path)

    # Validate audio quality
    valid, reason = validate_audio(wav_path)
    if not valid:
        delete_file(wav_path)
        raise HTTPException(status_code=422, detail=reason)

    # Process and save permanently with dual embeddings
    new_recs, new_paths, new_ecapa, new_eres = _save_profile_recordings(
        user_id, profile_id, [wav_path], [file.filename], source_type="add_recording"
    )
    delete_file(wav_path)

    if not new_recs:
        raise HTTPException(status_code=422, detail="Failed to save voice recording.")

    # Merge into existing profile
    existing_recs = from_json(prof.get("recordings", "[]"), [])
    existing_paths = from_json(prof.get("audio_paths", "[]"), [])
    existing_ecapa = from_json(prof.get("ecapa_embeddings") or prof.get("embeddings", "[]"), [])
    existing_eres = from_json(prof.get("eres2net_embeddings", "[]"), [])

    merged_recs = existing_recs + new_recs
    merged_paths = existing_paths + new_paths
    merged_ecapa = [e for e in existing_ecapa if len(e) == 192] + new_ecapa
    merged_eres = [e for e in existing_eres if len(e) == 512] + new_eres

    now = datetime.now(timezone.utc)
    await db.execute(
        text("""
            UPDATE voice_profiles
            SET recordings = :recordings,
                audio_paths = :audio_paths,
                ecapa_embeddings = :ecapa_embeddings,
                eres2net_embeddings = :eres2net_embeddings,
                embeddings = :embeddings,
                sample_count = :count,
                updated_at = :updated_at
            WHERE id = :id AND user_id = :uid
        """),
        {
            "recordings": to_json(merged_recs),
            "audio_paths": to_json(merged_paths),
            "ecapa_embeddings": to_json(merged_ecapa),
            "eres2net_embeddings": to_json(merged_eres) if merged_eres else None,
            "embeddings": to_json(merged_ecapa),
            "count": len(merged_recs),
            "updated_at": dt_to_str(now),
            "id": profile_id,
            "uid": user_id,
        },
    )
    await db.commit()

    return {
        "message": "Recording added successfully.",
        "recording": new_recs[0],
        "has_ecapa": len(merged_ecapa) > 0,
        "has_eres2net": len(merged_eres) > 0,
        "sample_count": len(merged_recs),
    }


# ── Remove a specific recording from a profile ────────────────────────
@router.delete("/profiles/{profile_id}/recordings/{recording_id}")
async def remove_profile_recording(
    profile_id: str,
    recording_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    """
    Remove a specific voice recording from a profile.
    Deletes the audio file and recomputes embeddings.
    """
    user_id = current_user["id"]
    r = await db.execute(
        text("SELECT * FROM voice_profiles WHERE id = :id AND user_id = :uid"),
        {"id": profile_id, "uid": user_id},
    )
    prof = r.mappings().fetchone()
    if not prof:
        raise HTTPException(status_code=404, detail="Profile not found.")

    existing_recs = from_json(prof.get("recordings", "[]"), [])
    existing_paths = from_json(prof.get("audio_paths", "[]"), [])

    # Find target recording
    target_idx = None
    target_file = None

    for i, rec in enumerate(existing_recs):
        if str(rec.get("id")) == str(recording_id):
            target_idx = i
            target_file = rec.get("file_path")
            break

    if target_idx is None and recording_id.isdigit():
        idx = int(recording_id)
        if 0 <= idx < len(existing_recs):
            target_idx = idx
            target_file = existing_recs[idx].get("file_path")
        elif 0 <= idx < len(existing_paths):
            target_file = existing_paths[idx]

    if target_file and os.path.exists(target_file):
        try:
            os.remove(target_file)
        except Exception as e:
            logger.warning(f"[Voice] Could not delete file {target_file}: {e}")

    # Remove from recordings and audio_paths
    if target_idx is not None and target_idx < len(existing_recs):
        existing_recs.pop(target_idx)
    if target_file and target_file in existing_paths:
        existing_paths.remove(target_file)

    # Recompute dual embeddings for remaining valid recordings
    from services.embedding_router import extract_dual_embeddings
    recomputed_ecapa = []
    recomputed_eres = []
    for rec in existing_recs:
        fp = rec.get("file_path")
        if fp and os.path.exists(fp):
            ec, er = extract_dual_embeddings(fp)
            if ec is not None:
                recomputed_ecapa.append(ec)
            if er is not None:
                recomputed_eres.append(er)

    try:
        from services.embedding_router import unload_active_encoder
        unload_active_encoder()
    except Exception:
        pass

    now = datetime.now(timezone.utc)
    await db.execute(
        text("""
            UPDATE voice_profiles
            SET recordings = :recordings,
                audio_paths = :audio_paths,
                ecapa_embeddings = :ecapa_embeddings,
                eres2net_embeddings = :eres2net_embeddings,
                embeddings = :embeddings,
                sample_count = :count,
                updated_at = :updated_at
            WHERE id = :id AND user_id = :uid
        """),
        {
            "recordings": to_json(existing_recs),
            "audio_paths": to_json(existing_paths),
            "ecapa_embeddings": to_json(recomputed_ecapa),
            "eres2net_embeddings": to_json(recomputed_eres) if recomputed_eres else None,
            "embeddings": to_json(recomputed_ecapa),
            "count": len(existing_recs),
            "updated_at": dt_to_str(now),
            "id": profile_id,
            "uid": user_id,
        },
    )
    await db.commit()

    return {
        "message": "Recording removed.",
        "sample_count": len(existing_recs),
        "has_ecapa": len(recomputed_ecapa) > 0,
        "has_eres2net": len(recomputed_eres) > 0,
        "has_audio": len(existing_recs) > 0,
    }


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
            text("SELECT id FROM voice_profiles WHERE user_id = :uid AND LOWER(TRIM(label)) = LOWER(TRIM(:label)) LIMIT 1"),
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

    # Check uniqueness of new_label (or find existing profile to append to)
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT id FROM voice_profiles WHERE user_id = :uid AND LOWER(TRIM(label)) = LOWER(TRIM(:label)) LIMIT 1"),
            {"uid": user_id, "label": new_label},
        )
        dup = r.mappings().fetchone()

    if dup:
        if not existing_profile_id:
            # If no profile_id was explicitly provided, automatically append as an additional sample to the existing profile!
            existing_profile_id = str(dup["id"])
            logger.info(f"[TrainFromTranscript] Name '{new_label}' matches existing profile {existing_profile_id} — treating as additional sample.")
        elif str(dup["id"]) != str(existing_profile_id):
            # Only conflict if renaming an existing profile into a DIFFERENT existing profile's name
            raise HTTPException(status_code=409, detail=f"The name '{new_label}' is already used by another profile.")

    target_profile_id = existing_profile_id or str(uuid.uuid4())

    # Permanently save audio recordings and generate BOTH ECAPA and ERes2Net embeddings
    new_recs, new_paths, ecapa_embs, eres_embs = _save_profile_recordings(
        user_id, target_profile_id, sample_paths, source_type="train_from_transcript"
    )

    if not ecapa_embs and not eres_embs:
        # Clean up copied files since profile was not created
        for p in new_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        raise HTTPException(
            status_code=422,
            detail="Could not extract voice embeddings from the samples. "
                   "Please ensure the samples contain clear speech.",
        )

    now = datetime.now(timezone.utc)

    try:
        async with get_db_context() as db:
            if existing_profile_id:
                # Update existing profile: merge recordings and both embedding models
                r = await db.execute(
                    text("SELECT recordings, audio_paths, embeddings, ecapa_embeddings, eres2net_embeddings FROM voice_profiles WHERE id = :id AND user_id = :uid"),
                    {"id": existing_profile_id, "uid": user_id},
                )
                prof = r.mappings().fetchone()
                if prof:
                    existing_recs = from_json(prof.get("recordings", "[]"), [])
                    existing_paths = from_json(prof.get("audio_paths", "[]"), [])
                    existing_ecapa = from_json(prof.get("ecapa_embeddings") or prof.get("embeddings", "[]"), [])
                    existing_eres = from_json(prof.get("eres2net_embeddings", "[]"), [])

                    merged_recs = existing_recs + new_recs
                    merged_paths = existing_paths + new_paths
                    merged_ecapa = [e for e in existing_ecapa if len(e) == 192] + ecapa_embs
                    merged_eres = [e for e in existing_eres if len(e) == 512] + eres_embs

                    await db.execute(
                        text("""
                            UPDATE voice_profiles
                            SET label = :label,
                                recordings = :recordings,
                                audio_paths = :audio_paths,
                                ecapa_embeddings = :ecapa_embeddings,
                                eres2net_embeddings = :eres2net_embeddings,
                                embeddings = :embeddings,
                                sample_count = :count,
                                updated_at = :updated_at
                            WHERE id = :id AND user_id = :uid
                        """),
                        {
                            "label": new_label,
                            "recordings": to_json(merged_recs),
                            "audio_paths": to_json(merged_paths),
                            "ecapa_embeddings": to_json(merged_ecapa),
                            "eres2net_embeddings": to_json(merged_eres) if merged_eres else None,
                            "embeddings": to_json(merged_ecapa),
                            "count": len(merged_recs),
                            "updated_at": dt_to_str(now),
                            "id": existing_profile_id,
                            "uid": user_id,
                        },
                    )
                    profile_id = existing_profile_id
                    total_embeddings = len(merged_ecapa)
                    total_sample_count = len(merged_recs)
                    has_ecapa = len(merged_ecapa) > 0
                    has_eres2net = len(merged_eres) > 0
                    logger.info(f"[TrainFromTranscript] Updated profile {profile_id} with {len(new_recs)} new recordings")
                else:
                    raise HTTPException(status_code=404, detail="Existing profile not found.")
            else:
                # Create new profile with recordings and dual embeddings
                profile_id = target_profile_id
                total_embeddings = len(ecapa_embs)
                total_sample_count = len(new_recs)
                has_ecapa = len(ecapa_embs) > 0
                has_eres2net = len(eres_embs) > 0

                await db.execute(
                    text("""
                        INSERT INTO voice_profiles
                            (id, user_id, label, embeddings, ecapa_embeddings, eres2net_embeddings, recordings, audio_paths, sample_count, is_self, created_at, updated_at)
                        VALUES
                            (:id, :user_id, :label, :embeddings, :ecapa_embeddings, :eres2net_embeddings, :recordings, :audio_paths, :count, 0, :created_at, :updated_at)
                    """),
                    {
                        "id": profile_id,
                        "user_id": user_id,
                        "label": new_label,
                        "embeddings": to_json(ecapa_embs),
                        "ecapa_embeddings": to_json(ecapa_embs),
                        "eres2net_embeddings": to_json(eres_embs) if eres_embs else None,
                        "recordings": to_json(new_recs),
                        "audio_paths": to_json(new_paths),
                        "count": len(new_recs),
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
    except HTTPException:
        # Re-raise explicit HTTP exceptions (e.g. 404, 422) after cleaning up newly copied recordings
        for p in new_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        raise
    except Exception as e:
        # If DB or rename fails, clean up newly copied permanent recordings to prevent orphans
        for p in new_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        logger.error(f"[TrainFromTranscript] Failed to train voice profile: {e}", exc_info=True)
        # Do NOT delete source recordings on failure!
        raise HTTPException(status_code=500, detail=f"Failed to complete voice training: {e}")

    # Only delete temporary sample files AFTER permanent recording storage and profile/embedding updates succeed
    for fp in sample_paths:
        try:
            if fp and os.path.exists(fp):
                os.remove(fp)
                logger.info(f"[TrainFromTranscript] Deleted temporary sample: {fp}")
        except Exception as e:
            logger.warning(f"[TrainFromTranscript] Could not delete temporary sample {fp}: {e}")

    logger.info(
        f"[TrainFromTranscript] Done — profile={profile_id}, label='{new_label}', "
        f"segments_updated={updated_count}, ecapa_embeddings={len(ecapa_embs)}, "
        f"eres2net_embeddings={len(eres_embs)}"
    )

    # speaker_sync already updated the transcript, speakers_detected, speaker
    # mappings, and any existing MoM text fields (participants, action items)
    # inline — no background AI pipeline is needed.

    return {
        "profile_id": profile_id,
        "new_label": new_label,
        "updated_segment_count": updated_count,
        "embedding_count": total_embeddings,
        "has_ecapa": has_ecapa,
        "has_eres2net": has_eres2net,
        "sample_count": total_sample_count,
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

