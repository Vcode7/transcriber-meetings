"""Audio upload/record router — accepts files, triggers pipeline."""
import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Form
from sqlalchemy import text

from database import get_db, get_db_context, dt_to_str, to_json
from routers.auth import get_current_user
from utils.storage import save_upload, delete_file
from utils.audio_utils import validate_audio, convert_to_wav, get_duration, render_keep_ranges
from utils.audio_edit import build_keep_ranges, is_identity, rescale_for_source_duration
from tasks.pipeline import run_pipeline, run_finalize_pipeline
from tasks.chunk_pipeline import run_chunk_pipeline
from tasks.upload_chunk_pipeline import run_upload_chunk_pipeline, UPLOAD_CHUNK_THRESHOLD_SEC

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["audio"])


async def _validate_audio_for_user(wav_path: str, user_id: str, db) -> tuple[bool, str]:
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
            if row.get("enable_audio_validation") is not None:
                enabled = bool(row["enable_audio_validation"])
            if row.get("min_audio_duration_seconds") is not None:
                min_dur = float(row["min_audio_duration_seconds"])
            if row.get("min_audio_rms_threshold") is not None:
                min_rms = float(row["min_audio_rms_threshold"])
    except Exception as e:
        logger.warning(f"Could not load audio validation settings: {e}")
    return validate_audio(wav_path, enabled=enabled, min_duration=min_dur, min_rms=min_rms)


def _rom_settings_columns(rom_settings: Optional[str]) -> dict:
    """recordings.rom_settings / rom_pipeline_status for the ROM settings chosen at capture time."""
    from models.rom_pipeline import parse_rom_settings
    from tasks.rom_auto_pipeline import initial_status_json
    parsed = parse_rom_settings(rom_settings)
    if parsed is None:
        return {"rom_settings": None, "rom_pipeline_status": None}
    return {"rom_settings": parsed.model_dump_json(), "rom_pipeline_status": initial_status_json(parsed)}


def _clean_expected_speakers(value: Optional[int]) -> Optional[int]:
    """A positive speaker count, or None (unknown)."""
    try:
        n = int(value) if value is not None else 0
    except (TypeError, ValueError):
        return None
    return n if 0 < n <= 50 else None


# ── Trim / middle-cut ──────────────────────────────────────────────────────

def build_edit_request(
    trim_start_sec: Optional[float],
    trim_end_sec: Optional[float],
    cut_start_sec: Optional[float],
    cut_end_sec: Optional[float],
    source_duration_sec: Optional[float],
) -> Optional[dict]:
    """
    Canonical form of the trim/cut a client asked for, or None for "process
    the whole file".  Each trim boundary is independent: an omitted start means
    the beginning and an omitted end means the true end of the decoded audio,
    so a boundary the user never moved can't clip audio when the browser's
    duration estimate was short.  Stored with the recording and used for
    upload de-duplication, so values are rounded to be JSON-stable.
    """
    def r(v: Optional[float]) -> Optional[float]:
        return None if v is None else round(float(v), 3)

    has_cut = cut_start_sec is not None and cut_end_sec is not None and cut_end_sec > cut_start_sec
    if trim_start_sec is None and trim_end_sec is None and not has_cut:
        return None
    return {
        "trim_start_sec": r(trim_start_sec),
        "trim_end_sec": r(trim_end_sec),
        "cut_start_sec": r(cut_start_sec) if has_cut else None,
        "cut_end_sec": r(cut_end_sec) if has_cut else None,
        "source_duration_sec": r(source_duration_sec),
    }


def apply_edit_request(
    wav_path: str,
    request: Optional[dict],
    reference_duration: Optional[float] = None,
    log_tag: str = "[Audio]",
) -> Tuple[str, Optional[dict]]:
    """
    Apply a client trim/cut request to a freshly converted 16 kHz WAV.

    Returns ``(wav_path_to_process, audio_edit)`` where ``audio_edit`` is the
    record persisted in ``recordings.audio_edit`` (None when no edit was
    requested).  ``audio_edit["keep_ranges"]`` is None when the selection
    covers the whole file.  On success with a new file the source WAV is
    deleted.  An impossible selection or a failed cut raises HTTP 422 — the
    user asked for specific audio, so silently processing the full recording
    would be wrong — and leaves ``wav_path`` for the caller to clean up.

    ``reference_duration`` is the duration of the timeline the browser
    trimmer showed (e.g. the video container); defaults to the decoded audio.
    """
    if request is None:
        return wav_path, None

    decoded = get_duration(wav_path)
    points, scale = rescale_for_source_duration(
        {k: request[k] for k in ("trim_start_sec", "trim_end_sec", "cut_start_sec", "cut_end_sec")},
        request.get("source_duration_sec"),
        reference_duration or decoded,
    )
    if scale != 1.0:
        logger.warning(
            f"{log_tag} Browser reported {request.get('source_duration_sec')}s but decoded audio is "
            f"{decoded:.3f}s; rescaling trim points by {scale:.4f}"
        )

    try:
        keep = build_keep_ranges(
            decoded,
            points["trim_start_sec"], points["trim_end_sec"],
            points["cut_start_sec"], points["cut_end_sec"],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid trim selection: {e}")

    audio_edit = {
        "request": request,
        "decoded_duration": round(decoded, 3),
        "scale": round(scale, 6),
        "keep_ranges": None,
        "output_duration": round(decoded, 3),
        "audio_path": None,
    }
    if is_identity(keep, decoded):
        logger.info(f"{log_tag} Trim selection covers the whole file — processing it unedited")
        return wav_path, audio_edit

    base, ext = os.path.splitext(wav_path)
    edited_path = f"{base}_edited{ext}"
    try:
        render_keep_ranges(wav_path, keep, edited_path)
    except Exception as e:
        delete_file(edited_path)
        logger.error(f"{log_tag} Audio edit failed for keep ranges {keep}: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Audio trim failed: {e}")

    delete_file(wav_path)
    audio_edit.update({
        "keep_ranges": [[round(a, 3), round(b, 3)] for a, b in keep],
        "output_duration": round(get_duration(edited_path), 3),
        "audio_path": edited_path,
    })
    logger.info(
        f"{log_tag} Edited audio: keep {audio_edit['keep_ranges']} of {decoded:.3f}s "
        f"→ {audio_edit['output_duration']}s ({edited_path})"
    )
    return edited_path, audio_edit


def keep_ranges_of(audio_edit_json: Optional[str]) -> Optional[List[Tuple[float, float]]]:
    """Keep ranges from a stored ``recordings.audio_edit`` value (None = unedited)."""
    try:
        ranges = (json.loads(audio_edit_json) if audio_edit_json else {}).get("keep_ranges")
    except (TypeError, ValueError, AttributeError):
        return None
    return [(float(a), float(b)) for a, b in ranges] if ranges else None


async def find_duplicate_upload(db, user_id: str, filename: Optional[str], edit_request: Optional[dict]):
    """
    An active recording of the same file submitted in the last 20 seconds with
    the same trim selection (guards against double-submits).  A different
    selection is a new request, not a duplicate.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=20)
    res = await db.execute(
        text("""
            SELECT id, status, progress, audio_edit
            FROM recordings
            WHERE user_id = :user_id
              AND filename = :filename
              AND status IN ('pending', 'processing', 'transcript_ready', 'pending_transcript_review')
              AND created_at >= :cutoff
            ORDER BY created_at DESC
            LIMIT 5
        """),
        {"user_id": user_id, "filename": filename, "cutoff": dt_to_str(cutoff)},
    )
    for row in res.mappings().all():
        try:
            stored = (json.loads(row["audio_edit"]) if row["audio_edit"] else {}).get("request")
        except (TypeError, ValueError, AttributeError):
            stored = None
        if stored == edit_request:
            return row
    return None


async def _create_recording_and_run(
    file_path: str,
    original_filename: str,
    user_id: str,
    background_tasks: BackgroundTasks,
    meeting_prompt: str = "",
    participant_voice_ids: Optional[List[str]] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
    rom_settings: Optional[str] = None,
    expected_speakers: Optional[int] = None,
    audio_edit: Optional[dict] = None,
) -> str:
    """Create recording row and schedule pipeline."""
    try:
        duration = get_duration(file_path)
    except Exception:
        duration = 0.0

    recording_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    pv_json = to_json(participant_voice_ids or [])

    logger.info(f"[Audio] Creating recording {recording_id} for user {user_id}, file={file_path}, duration={duration:.1f}s")

    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO recordings (
                    id, user_id, filename, file_path, duration, status, progress,
                    transcript, raw_text, summary, key_points, action_items, speakers_detected,
                    language, error_message, created_at, processed_at,
                    meeting_prompt, participant_voice_ids, use_vocabulary,
                    rom_settings, rom_pipeline_status, expected_speakers, audio_edit
                )
                VALUES (
                    :id, :user_id, :filename, :file_path, :duration, 'pending', 'queued',
                    '[]', NULL, NULL, '[]', '[]', '[]', 'en', NULL, :created_at, NULL,
                    :meeting_prompt, :participant_voice_ids, :use_vocabulary,
                    :rom_settings, :rom_pipeline_status, :expected_speakers, :audio_edit
                )
            """),
            {
                "id": recording_id,
                "user_id": user_id,
                "filename": original_filename,
                "file_path": file_path,
                "duration": duration,
                "created_at": dt_to_str(now),
                "meeting_prompt": meeting_prompt or "",
                "participant_voice_ids": pv_json,
                "use_vocabulary": 1 if use_vocabulary else 0,
                "expected_speakers": _clean_expected_speakers(expected_speakers),
                "audio_edit": to_json(audio_edit) if audio_edit else None,
                **_rom_settings_columns(rom_settings),
            },
        )
        await db.commit()

    logger.info(f"[Audio] Recording row created: {recording_id}. Scheduling pipeline...")
    import asyncio as _asyncio
    from tasks.pipeline import register_task
    task = _asyncio.create_task(
        run_pipeline(
            recording_id,
            file_path,
            user_id,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=participant_voice_ids or [],
            use_vocabulary=use_vocabulary,
            speaker_summary=speaker_summary,
        )
    )
    register_task(recording_id, task)
    logger.info(f"[Audio] Pipeline task scheduled for {recording_id}")
    return recording_id


async def _create_recording_and_run_chunked(
    file_path: str,
    original_filename: str,
    user_id: str,
    background_tasks: BackgroundTasks,
    duration: float,
    meeting_prompt: str = "",
    participant_voice_ids: Optional[List[str]] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
    rom_settings: Optional[str] = None,
    expected_speakers: Optional[int] = None,
    audio_edit: Optional[dict] = None,
) -> str:
    """Create recording row and schedule chunked upload pipeline."""
    recording_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    pv_json = to_json(participant_voice_ids or [])

    logger.info(f"[Audio] Creating chunked recording {recording_id} for user {user_id}, duration={duration:.1f}s")

    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO recordings (
                    id, user_id, filename, file_path, duration, status, progress,
                    transcript, raw_text, summary, key_points, action_items, speakers_detected,
                    language, error_message, created_at, processed_at,
                    meeting_prompt, participant_voice_ids, use_vocabulary,
                    rom_settings, rom_pipeline_status, expected_speakers, audio_edit
                )
                VALUES (
                    :id, :user_id, :filename, :file_path, :duration, 'pending', 'queued',
                    '[]', NULL, NULL, '[]', '[]', '[]', 'en', NULL, :created_at, NULL,
                    :meeting_prompt, :participant_voice_ids, :use_vocabulary,
                    :rom_settings, :rom_pipeline_status, :expected_speakers, :audio_edit
                )
            """),
            {
                "id": recording_id,
                "user_id": user_id,
                "filename": original_filename,
                "file_path": file_path,
                "duration": duration,
                "created_at": dt_to_str(now),
                "meeting_prompt": meeting_prompt or "",
                "participant_voice_ids": pv_json,
                "use_vocabulary": 1 if use_vocabulary else 0,
                "expected_speakers": _clean_expected_speakers(expected_speakers),
                "audio_edit": to_json(audio_edit) if audio_edit else None,
                **_rom_settings_columns(rom_settings),
            },
        )
        await db.commit()

    logger.info(f"[Audio] Recording row created: {recording_id}. Scheduling chunked pipeline...")
    import asyncio as _asyncio
    from tasks.pipeline import register_task
    task = _asyncio.create_task(
        run_upload_chunk_pipeline(
            recording_id,
            file_path,
            user_id,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=participant_voice_ids or [],
            use_vocabulary=use_vocabulary,
            speaker_summary=speaker_summary,
        )
    )
    register_task(recording_id, task)
    return recording_id


@router.post("/upload")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    meeting_prompt: Optional[str] = Form(default=""),
    participant_voice_ids: Optional[str] = Form(default="[]"),  # JSON array string
    use_vocabulary: Optional[bool] = Form(default=False),
    speaker_summary: Optional[bool] = Form(default=False),
    rom_settings: Optional[str] = Form(default=None),  # JSON RomPipelineSettings (models/rom_pipeline.py)
    expected_speakers: Optional[int] = Form(default=None),  # known speaker count → pyannote num_speakers
    trim_start_sec: Optional[float] = Form(default=None),
    trim_end_sec: Optional[float] = Form(default=None),
    cut_start_sec: Optional[float] = Form(default=None),
    cut_end_sec: Optional[float] = Form(default=None),
    source_duration_sec: Optional[float] = Form(default=None),  # duration the browser trimmer saw
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    """Upload a pre-recorded audio file for processing."""
    import json as _json
    user_id = current_user["id"]
    logger.info(f"[Audio] Upload request from user {user_id}, filename={file.filename}")
    edit_request = build_edit_request(trim_start_sec, trim_end_sec, cut_start_sec, cut_end_sec, source_duration_sec)

    # ── Deduplication check ────────────────────────────────────────────────
    # If the user submitted the exact same filename with the same trim selection
    # within the last 20 seconds and that recording is still active, return the
    # existing recording ID to prevent double-submissions/duplicate history entries.
    existing_row = await find_duplicate_upload(db, user_id, file.filename, edit_request)
    if existing_row:
        logger.info(f"[Audio] Deduplicating upload: reusing active recording {existing_row['id']} for {file.filename}")
        return {
            "recording_id": existing_row["id"],
            "status": existing_row["status"],
            "message": "Reusing existing active recording. Processing already in progress.",
        }

    # Parse participant_voice_ids JSON string
    try:
        pv_ids: List[str] = _json.loads(participant_voice_ids or "[]")
    except Exception:
        pv_ids = []

    raw_path = await save_upload(file, user_id, prefix="rec_")

    # Convert to WAV
    wav_path = raw_path.rsplit(".", 1)[0] + "_proc.wav"
    try:
        convert_to_wav(raw_path, wav_path)
        delete_file(raw_path)
    except Exception as e:
        logger.error(f"[Audio] Audio conversion failed: {e}", exc_info=True)
        delete_file(raw_path)
        raise HTTPException(status_code=422, detail=f"Audio conversion failed: {e}")

    # ── Optional trim / middle-cut (before validation: validate what we process) ──
    try:
        wav_path, audio_edit = apply_edit_request(wav_path, edit_request, log_tag="[Audio] Upload")
    except HTTPException:
        delete_file(wav_path)
        raise

    valid, reason = await _validate_audio_for_user(wav_path, user_id, db)
    if not valid:
        logger.warning(f"[Audio] Audio validation failed: {reason}")
        delete_file(wav_path)
        raise HTTPException(status_code=422, detail=reason)

    # Get duration (header-only read — no RAM spike regardless of file size)
    duration = get_duration(wav_path)
    logger.info(f"[Audio] Upload validated OK: {wav_path}, duration={duration:.1f}s")

    # ── Auto-route: large uploads use the chunked pipeline ────────────────
    if duration > UPLOAD_CHUNK_THRESHOLD_SEC:
        logger.info(
            f"[Audio] Upload duration {duration:.0f}s exceeds threshold "
            f"{UPLOAD_CHUNK_THRESHOLD_SEC:.0f}s — routing to chunked pipeline"
        )
        recording_id = await _create_recording_and_run_chunked(
            file_path=wav_path,
            original_filename=file.filename,
            user_id=user_id,
            background_tasks=background_tasks,
            duration=duration,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=pv_ids,
            use_vocabulary=use_vocabulary or False,
            speaker_summary=speaker_summary or False,
            rom_settings=rom_settings,
            expected_speakers=expected_speakers,
            audio_edit=audio_edit,
        )
    else:
        recording_id = await _create_recording_and_run(
            file_path=wav_path,
            original_filename=file.filename,
            user_id=user_id,
            background_tasks=background_tasks,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=pv_ids,
            use_vocabulary=use_vocabulary or False,
            speaker_summary=speaker_summary or False,
            rom_settings=rom_settings,
            expected_speakers=expected_speakers,
            audio_edit=audio_edit,
        )

    return {
        "recording_id": recording_id,
        "status": "pending",
        "message": "Audio uploaded. Processing started in background.",
    }


# ── Submit raw recorded audio (from browser MediaRecorder) ────
@router.post("/record")
async def submit_recording(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    meeting_prompt: Optional[str] = Form(default=""),
    participant_voice_ids: Optional[str] = Form(default="[]"),
    use_vocabulary: Optional[bool] = Form(default=False),
    speaker_summary: Optional[bool] = Form(default=False),
    rom_settings: Optional[str] = Form(default=None),  # JSON RomPipelineSettings (models/rom_pipeline.py)
    expected_speakers: Optional[int] = Form(default=None),  # known speaker count → pyannote num_speakers
    trim_start_sec: Optional[float] = Form(default=None),
    trim_end_sec: Optional[float] = Form(default=None),
    cut_start_sec: Optional[float] = Form(default=None),
    cut_end_sec: Optional[float] = Form(default=None),
    source_duration_sec: Optional[float] = Form(default=None),  # duration the browser trimmer saw
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    """
    Accept audio blob from browser MediaRecorder (webm/ogg/wav).
    Converts to 16kHz WAV and queues for processing.
    """
    import json as _json
    user_id = current_user["id"]
    logger.info(f"[Audio] Record submission from user {user_id}, content_type={file.content_type}")
    edit_request = build_edit_request(trim_start_sec, trim_end_sec, cut_start_sec, cut_end_sec, source_duration_sec)

    try:
        pv_ids: List[str] = _json.loads(participant_voice_ids or "[]")
    except Exception:
        pv_ids = []

    raw_path = await save_upload(file, user_id, prefix="live_")
    wav_path = raw_path.rsplit(".", 1)[0] + "_proc.wav"

    try:
        convert_to_wav(raw_path, wav_path)
        delete_file(raw_path)
    except Exception as e:
        logger.error(f"[Audio] Conversion failed: {e}", exc_info=True)
        delete_file(raw_path)
        raise HTTPException(status_code=422, detail=f"Conversion failed: {e}")

    # ── Optional trim / middle-cut (before validation: validate what we process) ──
    try:
        wav_path, audio_edit = apply_edit_request(wav_path, edit_request, log_tag="[Audio] Record")
    except HTTPException:
        delete_file(wav_path)
        raise

    valid, reason = await _validate_audio_for_user(wav_path, user_id, db)
    if not valid:
        logger.warning(f"[Audio] Audio validation failed: {reason}")
        delete_file(wav_path)
        raise HTTPException(status_code=422, detail=reason)

    # Get duration (header-only read — no RAM spike regardless of file size)
    duration = get_duration(wav_path)
    logger.info(f"[Audio] Recording validated OK: {wav_path}, duration={duration:.1f}s")

    # ── Auto-route: large recordings use the chunked pipeline ─────────────
    if duration > UPLOAD_CHUNK_THRESHOLD_SEC:
        logger.info(
            f"[Audio] Recording duration {duration:.0f}s exceeds threshold "
            f"{UPLOAD_CHUNK_THRESHOLD_SEC:.0f}s — routing to chunked pipeline"
        )
        recording_id = await _create_recording_and_run_chunked(
            file_path=wav_path,
            original_filename=file.filename or "live_recording.wav",
            user_id=user_id,
            background_tasks=background_tasks,
            duration=duration,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=pv_ids,
            use_vocabulary=use_vocabulary or False,
            speaker_summary=speaker_summary or False,
            rom_settings=rom_settings,
            expected_speakers=expected_speakers,
            audio_edit=audio_edit,
        )
    else:
        recording_id = await _create_recording_and_run(
            file_path=wav_path,
            original_filename=file.filename or "live_recording.wav",
            user_id=user_id,
            background_tasks=background_tasks,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=pv_ids,
            use_vocabulary=use_vocabulary or False,
            speaker_summary=speaker_summary or False,
            rom_settings=rom_settings,
            expected_speakers=expected_speakers,
            audio_edit=audio_edit,
        )

    return {
        "recording_id": recording_id,
        "status": "pending",
        "message": "Recording received. Processing started.",
    }


# ── Submit a background chunk during recording ────────────────
@router.post("/chunk")
async def submit_chunk(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_index: int = Form(...),
    chunk_start_sec: float = Form(default=0.0),
    chunk_end_sec: float = Form(default=0.0),
    meeting_prompt: Optional[str] = Form(default=""),
    use_vocabulary: Optional[bool] = Form(default=False),
    current_user: dict = Depends(get_current_user),
):
    """
    Accept a single 10-minute audio chunk submitted during a long recording.
    Saves the chunk, schedules transcription-only background processing,
    and returns a chunk_id for tracking.
    """
    user_id = current_user["id"]
    chunk_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    logger.info(
        f"[Audio] Chunk submission from user {user_id}, "
        f"chunk_index={chunk_index}, start={chunk_start_sec:.1f}s, end={chunk_end_sec:.1f}s"
    )

    raw_path = await save_upload(file, user_id, prefix=f"chunk_{chunk_index}_")
    wav_path = raw_path.rsplit(".", 1)[0] + "_proc.wav"

    try:
        convert_to_wav(raw_path, wav_path)
        delete_file(raw_path)
    except Exception as e:
        logger.error(f"[Audio] Chunk conversion failed: {e}", exc_info=True)
        delete_file(raw_path)
        raise HTTPException(status_code=422, detail=f"Chunk conversion failed: {e}")

    async with get_db_context() as db:
        valid, reason = await _validate_audio_for_user(wav_path, user_id, db)
        if not valid:
            delete_file(wav_path)
            raise HTTPException(status_code=422, detail=reason)

    # Create chunk row in DB
    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO recording_chunks (
                    id, recording_id, chunk_index, chunk_start_sec, chunk_end_sec,
                    file_path, status, transcript, raw_text, aligned_result, created_at
                ) VALUES (
                    :id, NULL, :chunk_index, :chunk_start_sec, :chunk_end_sec,
                    :file_path, 'pending', '[]', NULL, NULL, :created_at
                )
            """),
            {
                "id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_start_sec": chunk_start_sec,
                "chunk_end_sec": chunk_end_sec,
                "file_path": wav_path,
                "created_at": dt_to_str(now),
            },
        )
        await db.commit()

    # Schedule transcription-only pipeline for this chunk
    background_tasks.add_task(
        run_chunk_pipeline,
        chunk_id=chunk_id,
        chunk_wav_path=wav_path,
        user_id=user_id,
        meeting_prompt=meeting_prompt or "",
        use_vocabulary=use_vocabulary or False,
    )

    logger.info(f"[Audio] Chunk {chunk_id} (index={chunk_index}) queued for background transcription")
    return {"chunk_id": chunk_id, "status": "pending"}


# ── Finalize a chunked recording (called when recording stops) ─
@router.post("/record-finalize")
async def finalize_recording(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_ids: Optional[str] = Form(default="[]"),         # JSON array of chunk IDs
    meeting_prompt: Optional[str] = Form(default=""),
    participant_voice_ids: Optional[str] = Form(default="[]"),
    use_vocabulary: Optional[bool] = Form(default=False),
    speaker_summary: Optional[bool] = Form(default=False),
    rom_settings: Optional[str] = Form(default=None),  # JSON RomPipelineSettings (models/rom_pipeline.py)
    expected_speakers: Optional[int] = Form(default=None),  # known speaker count → pyannote num_speakers
    trim_start_sec: Optional[float] = Form(default=None),
    trim_end_sec: Optional[float] = Form(default=None),
    cut_start_sec: Optional[float] = Form(default=None),
    cut_end_sec: Optional[float] = Form(default=None),
    source_duration_sec: Optional[float] = Form(default=None),  # duration the browser trimmer saw
    current_user: dict = Depends(get_current_user),
):
    """
    Called when the user stops a long recording (>= 10 min).
    Accepts the full audio blob + list of already-submitted chunk IDs.
    Saves the full audio, creates the recording row, then schedules
    the finalize pipeline which: waits for chunks, merges transcripts,
    runs diarization on full audio, assigns speakers, generates AI insights.

    The chunks were transcribed on the untrimmed timeline while recording, so
    a trim/cut is applied to the full audio here and its keep ranges are
    handed to the pipeline, which remaps the chunk transcripts onto it.
    """
    user_id = current_user["id"]
    logger.info(f"[Audio] Finalize request from user {user_id}")
    edit_request = build_edit_request(trim_start_sec, trim_end_sec, cut_start_sec, cut_end_sec, source_duration_sec)

    # Parse chunk_ids JSON
    try:
        parsed_chunk_ids: List[str] = json.loads(chunk_ids or "[]")
    except Exception:
        parsed_chunk_ids = []

    # Parse participant_voice_ids JSON
    try:
        pv_ids: List[str] = json.loads(participant_voice_ids or "[]")
    except Exception:
        pv_ids = []

    # Save the full audio blob
    raw_path = await save_upload(file, user_id, prefix="final_")
    wav_path = raw_path.rsplit(".", 1)[0] + "_proc.wav"

    try:
        convert_to_wav(raw_path, wav_path)
        delete_file(raw_path)
    except Exception as e:
        logger.error(f"[Audio] Finalize audio conversion failed: {e}", exc_info=True)
        delete_file(raw_path)
        raise HTTPException(status_code=422, detail=f"Audio conversion failed: {e}")

    # ── Optional trim / middle-cut (before validation: validate what we process) ──
    try:
        wav_path, audio_edit = apply_edit_request(wav_path, edit_request, log_tag="[Audio] Finalize")
    except HTTPException:
        delete_file(wav_path)
        raise
    edit_keep_ranges = (audio_edit or {}).get("keep_ranges")

    async with get_db_context() as db:
        valid, reason = await _validate_audio_for_user(wav_path, user_id, db)
        if not valid:
            delete_file(wav_path)
            raise HTTPException(status_code=422, detail=reason)

    # Get duration
    try:
        duration = get_duration(wav_path)
    except Exception:
        duration = 0.0

    # Create recording row
    recording_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO recordings (
                    id, user_id, filename, file_path, duration, status, progress,
                    transcript, raw_text, summary, key_points, action_items, speakers_detected,
                    language, error_message, created_at, processed_at,
                    meeting_prompt, participant_voice_ids, use_vocabulary,
                    chunk_ids, is_chunked, rom_settings, rom_pipeline_status, expected_speakers,
                    audio_edit
                ) VALUES (
                    :id, :user_id, :filename, :file_path, :duration, 'pending', 'queued',
                    '[]', NULL, NULL, '[]', '[]', '[]', 'en', NULL, :created_at, NULL,
                    :meeting_prompt, :participant_voice_ids, :use_vocabulary,
                    :chunk_ids, 1, :rom_settings, :rom_pipeline_status, :expected_speakers,
                    :audio_edit
                )
            """),
            {
                "id": recording_id,
                "user_id": user_id,
                "filename": file.filename or "recording.wav",
                "file_path": wav_path,
                "duration": duration,
                "created_at": dt_to_str(now),
                "meeting_prompt": meeting_prompt or "",
                "participant_voice_ids": json.dumps(pv_ids),
                "use_vocabulary": 1 if use_vocabulary else 0,
                "chunk_ids": json.dumps(parsed_chunk_ids),
                "expected_speakers": _clean_expected_speakers(expected_speakers),
                "audio_edit": to_json(audio_edit) if audio_edit else None,
                **_rom_settings_columns(rom_settings),
            },
        )
        await db.commit()

    # Link chunks to this recording_id
    if parsed_chunk_ids:
        async with get_db_context() as db:
            for cid in parsed_chunk_ids:
                await db.execute(
                    text("UPDATE recording_chunks SET recording_id = :rid WHERE id = :cid"),
                    {"rid": recording_id, "cid": cid},
                )
            await db.commit()

    # Schedule finalize pipeline
    import asyncio as _asyncio
    from tasks.pipeline import register_task
    task = _asyncio.create_task(
        run_finalize_pipeline(
            recording_id=recording_id,
            full_wav_path=wav_path,
            chunk_ids=parsed_chunk_ids,
            user_id=user_id,
            meeting_prompt=meeting_prompt or "",
            participant_voice_ids=pv_ids,
            use_vocabulary=use_vocabulary or False,
            speaker_summary=speaker_summary or False,
            edit_keep_ranges=edit_keep_ranges,
        )
    )
    register_task(recording_id, task)

    logger.info(
        f"[Audio] Finalize pipeline scheduled: recording={recording_id}, "
        f"chunks={len(parsed_chunk_ids)}, duration={duration:.1f}s"
    )
    return {
        "recording_id": recording_id,
        "status": "pending",
        "message": "Recording finalized. Processing started.",
    }


# ── Poll job status ───────────────────────────────────────────
@router.get("/jobs/{recording_id}")
async def get_job_status(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Poll the status of a recording job. Returns full result when done."""
    user_id = current_user["id"]

    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT * FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    import json
    resp = {
        "job_id": recording_id,
        "status": rec["status"],
        "progress": rec.get("progress", ""),
        "duration": rec.get("duration", 0),
        "created_at": rec["created_at"],
    }

    if rec["status"] == "done":
        resp["result"] = {
            "transcript": json.loads(rec["transcript"] or "[]"),
            "raw_text": rec.get("raw_text", ""),
            "summary": rec.get("summary", ""),
            "short_summary": rec.get("short_summary", ""),
            "detailed_summary": rec.get("detailed_summary", ""),
            "key_points": json.loads(rec["key_points"] or "[]"),
            "action_items": json.loads(rec["action_items"] or "[]"),
            "speakers_detected": json.loads(rec["speakers_detected"] or "[]"),
            "language": rec.get("language", "en"),
            "processed_at": rec.get("processed_at"),
            "speaker_summary": json.loads(rec["speaker_summary"]) if rec.get("speaker_summary") else None,
        }
    elif rec["status"] == "transcript_ready":
        # Phase 1 done — return transcript immediately while Qwen3 runs in background
        resp["ai_generating"] = True
        resp["result"] = {
            "transcript": json.loads(rec["transcript"] or "[]"),
            "raw_text": rec.get("raw_text", ""),
            "speakers_detected": json.loads(rec["speakers_detected"] or "[]"),
            "language": rec.get("language", "en"),
            # AI fields not yet available
            "summary": "",
            "short_summary": "",
            "detailed_summary": "",
            "key_points": [],
            "action_items": [],
        }
    elif rec["status"] == "error":
        resp["error"] = rec.get("error_message", "Unknown error.")
        logger.warning(f"[Audio] Job {recording_id} is in error state: {resp['error']}")

    return resp


@router.post("/jobs/{recording_id}/cancel")
async def cancel_job(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Cancel an active transcription/processing job.
    Terminates the background worker task, cleans up RAM/VRAM, and marks job cancelled.
    """
    user_id = current_user["id"]
    logger.info(f"[Audio] Cancel request received for recording={recording_id} from user={user_id}")

    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT id, status, file_path FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    if rec["status"] in ("done", "error", "cancelled"):
        return {"status": rec["status"], "message": f"Job is already completed with status: {rec['status']}"}

    # Cancel the active task cooperatively
    from tasks.pipeline import cancel_task
    task_cancelled = await cancel_task(recording_id)

    # Force update the status in the database to 'cancelled' (in case the task wasn't running or unregistered)
    async with get_db_context() as db:
        await db.execute(
            text("UPDATE recordings SET status = 'cancelled', progress = NULL, error_message = 'Cancelled by user' WHERE id = :id"),
            {"id": recording_id}
        )
        await db.commit()

    # Unload models to free GPU/CPU memory
    from tasks.pipeline import unload_all_models
    unload_all_models()

    logger.info(f"[Audio] Cancel request completed for recording={recording_id}. Active task found/cancelled={task_cancelled}")
    return {"status": "cancelled", "message": "Processing cancelled successfully."}


# ── List all active jobs for user ─────────────────────────────
@router.get("/jobs")
async def list_active_jobs(
    current_user: dict = Depends(get_current_user),
):
    """
    Return all non-completed recordings for the authenticated user.
    Used by the frontend on startup to reconnect to in-flight jobs
    that were started before the current page session.

    Only returns recordings in status: pending, processing, transcript_ready.
    (Done/error recordings are accessible through the History endpoints.)
    """
    user_id = current_user["id"]
    from tasks.pipeline import active_tasks
    import datetime

    async with get_db_context() as db:
        r = await db.execute(
            text("""
                SELECT id, filename, status, progress, duration, created_at, meeting_prompt
                FROM recordings
                WHERE user_id = :uid
                  AND status IN ('pending', 'processing', 'transcript_ready', 'pending_transcript_review')
                ORDER BY created_at DESC
                LIMIT 20
            """),
            {"uid": user_id},
        )
        rows = r.mappings().fetchall()

        stale_ids = []
        jobs = []
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        for rec in rows:
            rec_id = rec["id"]
            rec_status = rec["status"]
            is_running = rec_id in active_tasks

            # If job is pending or processing but has no active asyncio task running
            if rec_status in ("pending", "processing") and not is_running:
                created_at_val = rec.get("created_at")
                created_at_dt = None
                if isinstance(created_at_val, datetime.datetime):
                    created_at_dt = created_at_val
                elif isinstance(created_at_val, str):
                    try:
                        created_at_dt = datetime.datetime.fromisoformat(created_at_val.replace("Z", "+00:00"))
                    except Exception:
                        pass
                
                age_seconds = (now_utc - created_at_dt).total_seconds() if created_at_dt else 999
                if age_seconds > 10:
                    stale_ids.append(rec_id)
                    continue

            jobs.append({
                "job_id": rec_id,
                "filename": rec.get("filename", ""),
                "status": rec_status,
                "progress": rec.get("progress", ""),
                "duration": rec.get("duration", 0),
                "created_at": rec["created_at"],
            })

        if stale_ids:
            logger.info(f"[Audio] Reconciler cleaning up {len(stale_ids)} stale/unresolved jobs: {stale_ids}")
            for sid in stale_ids:
                await db.execute(
                    text("UPDATE recordings SET status = 'error', error_message = 'Job interrupted due to server restart or shutdown.' WHERE id = :id"),
                    {"id": sid}
                )
            await db.commit()

    logger.debug(f"[Audio] Active jobs for user {user_id}: {len(jobs)}")
    return {"jobs": jobs}


# ── Missing Transcription Recovery endpoints ──────────────────

@router.get("/{recording_id}/transcript/raw")
async def get_raw_transcript(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Return the raw aligned transcript segments saved after transcription+alignment,
    before any speaker diarization.  Only available when job status is
    'pending_transcript_review' or after pipeline has passed that point.
    """
    import json as _json
    user_id = current_user["id"]
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT id, status, raw_transcript FROM recordings WHERE id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    raw = rec.get("raw_transcript") or "[]"
    try:
        segments = _json.loads(raw)
    except Exception:
        segments = []

    return {
        "recording_id": recording_id,
        "status": rec["status"],
        "segments": segments,
    }


from pydantic import BaseModel as _BaseModel
from typing import List as _List

class TranscriptCorrection(_BaseModel):
    start: float
    end: float
    text: str

class SubmitCorrectionsRequest(_BaseModel):
    corrections: _List[TranscriptCorrection]


@router.post("/{recording_id}/transcript/corrections")
async def submit_transcript_corrections(
    recording_id: str,
    req: SubmitCorrectionsRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Merge user-supplied transcript corrections into the raw_transcript and
    set the recording status back to 'processing' so the pipeline resumes.

    Each correction is inserted as a new segment at the correct chronological
    position in the raw_transcript array.
    """
    import json as _json
    user_id = current_user["id"]

    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT id, status, raw_transcript FROM recordings WHERE id = :rid AND user_id = :uid"),
            {"rid": recording_id, "uid": user_id},
        )
        rec = r.mappings().fetchone()

    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found.")

    if rec["status"] not in ("pending_transcript_review", "processing", "transcript_ready", "done"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot apply transcript corrections for job in status: {rec['status']}"
        )

    # Load existing segments
    try:
        segments = _json.loads(rec.get("raw_transcript") or "[]")
    except Exception:
        segments = []

    # Build new correction segments
    correction_segments = []
    for corr in req.corrections:
        c_text = (corr.text or "").strip()
        if c_text:
            correction_segments.append({
                "start": corr.start,
                "end": corr.end,
                "text": c_text,
                "words": [],  # will be synthesized with proper timestamps by sanitize_and_merge_segments
                "manually_added": True,
            })

    from services.transcript_normalizer import sanitize_and_merge_segments
    sanitized_segments = sanitize_and_merge_segments(segments, correction_segments)

    merged_json = _json.dumps(sanitized_segments, ensure_ascii=False)

    # Write updated transcript and resume pipeline
    async with get_db_context() as db:
        await db.execute(
            text("""
                UPDATE recordings
                SET status = 'processing',
                    progress = 'diarizing',
                    raw_transcript = :rt
                WHERE id = :rid
            """),
            {"rt": merged_json, "rid": recording_id},
        )
        await db.commit()

    logger.info(
        f"[Audio] Transcript corrections submitted for {recording_id}: "
        f"{len(correction_segments)} correction(s) merged without overlap; {len(sanitized_segments)} total segments; pipeline resumed."
    )
    return {
        "status": "resumed",
        "corrections_merged": len(correction_segments),
        "total_segments": len(sanitized_segments),
    }

