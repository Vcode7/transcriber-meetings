"""
upload_chunk_pipeline.py — Auto-chunked processing for large uploaded audio files.

When a user uploads an audio file longer than UPLOAD_CHUNK_THRESHOLD_SEC (default
3600s / 60 min) the regular run_pipeline is replaced by this function which:

  1. Splits the uploaded WAV into 10-minute chunks on disk using ffmpeg
     (memory-efficient: no audio loaded into Python RAM).
  2. Creates a recording_chunks row for each chunk.
  3. Transcribes each chunk sequentially via run_chunk_pipeline.
  4. Once all chunks are transcribed, calls run_finalize_pipeline which:
       - Diarizes the full audio ONCE (consistent speaker timeline)
       - Runs speaker identification globally across the full audio
       - Merges timestamps correctly across chunk boundaries
       - Generates MoM and stores the unified transcript

This approach is identical to what happens with live long recordings, ensuring
the same code path handles merging, speaker consistency, and AI generation.

Memory guarantees:
  - Each chunk WAV is ~30 MB for 10 minutes of 16 kHz mono audio.
  - Chunks are deleted immediately after transcription.
  - The full-audio WAV is kept for diarization (needed for speaker assignment).
"""
import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import text

from database import get_db, dt_to_str, to_json
from utils.audio_utils import split_wav_to_files
from tasks.chunk_pipeline import run_chunk_pipeline
from tasks.pipeline import run_finalize_pipeline

logger = logging.getLogger(__name__)

# Chunk duration — must match the live-recording chunk size so the finalize
# pipeline logic (wait times, merging) behaves identically.
UPLOAD_CHUNK_DURATION_SEC: float = 600.0   # 10 minutes per chunk

# Files shorter than this are processed by run_pipeline (no splitting needed).
# Uploads longer than this are auto-split and routed through run_finalize_pipeline.
# Default: 3600s (60 min). A user uploading a 90-minute meeting gets split into
# nine 10-minute chunks automatically.
UPLOAD_CHUNK_THRESHOLD_SEC: float = 3600.0  # 60 minutes


async def run_upload_chunk_pipeline(
    recording_id: str,
    full_wav_path: str,
    user_id: str,
    meeting_prompt: str = "",
    participant_voice_ids: Optional[List[str]] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
) -> None:
    try:
        await _run_upload_chunk_pipeline_impl(
            recording_id=recording_id,
            full_wav_path=full_wav_path,
            user_id=user_id,
            meeting_prompt=meeting_prompt,
            participant_voice_ids=participant_voice_ids,
            use_vocabulary=use_vocabulary,
            speaker_summary=speaker_summary,
        )
    except asyncio.CancelledError:
        logger.info(f"[UploadChunkPipeline] {recording_id} — Task was CANCELLED.")
        try:
            async with get_db() as db:
                await db.execute(
                    text("UPDATE recordings SET status='cancelled', progress=NULL, error_message='Cancelled by user' WHERE id=:rid"),
                    {"rid": recording_id}
                )
                await db.commit()
        except Exception as e:
            logger.error(f"[UploadChunkPipeline] {recording_id} — Failed to update cancelled status in DB: {e}")
        raise
    finally:
        from tasks.pipeline import unregister_task, unload_all_models
        unregister_task(recording_id)
        unload_all_models()


async def _run_upload_chunk_pipeline_impl(
    recording_id: str,
    full_wav_path: str,
    user_id: str,
    meeting_prompt: str = "",
    participant_voice_ids: Optional[List[str]] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
) -> None:
    """
    Auto-split a large uploaded WAV and process it through the finalize pipeline.

    This is the entry point for uploads > UPLOAD_CHUNK_THRESHOLD_SEC.
    After this function returns, the recording will have a complete transcript,
    diarization, speaker labels, and MoM in the DB.
    """
    participant_voice_ids = participant_voice_ids or []
    logger.info(
        f"[UploadChunkPipeline] ===== START recording={recording_id} "
        f"file={full_wav_path} ====="
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error(f"[UploadChunkPipeline] {recording_id} — No running event loop!")
        return

    # ── Step 1: Split WAV into chunk files on disk ──────────────────────────
    logger.info(f"[UploadChunkPipeline] {recording_id} — Splitting WAV into 10-min chunks")

    upload_dir = os.path.dirname(full_wav_path)
    chunk_dir = os.path.join(upload_dir, f"_chunks_{recording_id}")

    try:
        try:
            chunk_files: list[tuple[int, float, float, str]] = await loop.run_in_executor(
                None,
                lambda: split_wav_to_files(
                    input_wav=full_wav_path,
                    output_dir=chunk_dir,
                    chunk_sec=UPLOAD_CHUNK_DURATION_SEC,
                    prefix="uc_",
                ),
            )
        except Exception as e:
            logger.error(
                f"[UploadChunkPipeline] {recording_id} — WAV split failed: {e}",
                exc_info=True,
            )
            async with get_db() as db:
                await db.execute(
                    text(
                        "UPDATE recordings SET status='error', error_message=:msg "
                        "WHERE id=:rid"
                    ),
                    {"msg": f"Audio split failed: {e}", "rid": recording_id},
                )
                await db.commit()
            return

        logger.info(
            f"[UploadChunkPipeline] {recording_id} — "
            f"Produced {len(chunk_files)} chunks in {chunk_dir}"
        )

        # ── Step 2: Insert recording_chunks rows ────────────────────────────────
        now = datetime.now(timezone.utc)
        chunk_ids: List[str] = []

        for chunk_index, start_sec, end_sec, chunk_path in chunk_files:
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)

            async with get_db() as db:
                await db.execute(
                    text("""
                        INSERT INTO recording_chunks (
                            id, recording_id, chunk_index, chunk_start_sec, chunk_end_sec,
                            file_path, status, transcript, raw_text, aligned_result, created_at
                        ) VALUES (
                            :id, :recording_id, :chunk_index, :chunk_start_sec, :chunk_end_sec,
                            :file_path, 'pending', '[]', NULL, NULL, :created_at
                        )
                    """),
                    {
                        "id": chunk_id,
                        "recording_id": recording_id,
                        "chunk_index": chunk_index,
                        "chunk_start_sec": start_sec,
                        "chunk_end_sec": end_sec,
                        "file_path": chunk_path,
                        "created_at": dt_to_str(now),
                    },
                )
                await db.commit()

        logger.info(
            f"[UploadChunkPipeline] {recording_id} — "
            f"Inserted {len(chunk_ids)} recording_chunks rows"
        )

        # ── Step 3: Transcribe each chunk sequentially ──────────────────────────
        # Sequential processing keeps VRAM usage bounded — only one Whisper inference
        # runs at a time.  For N chunks of 10 min each, peak RAM is ~1 chunk.
        logger.info(
            f"[UploadChunkPipeline] {recording_id} — "
            f"Transcribing {len(chunk_ids)} chunks sequentially"
        )

        for chunk_id, (chunk_index, start_sec, end_sec, chunk_path) in zip(
            chunk_ids, chunk_files
        ):
            logger.info(
                f"[UploadChunkPipeline] {recording_id} — "
                f"Transcribing chunk {chunk_index + 1}/{len(chunk_files)} "
                f"({start_sec:.0f}s – {end_sec:.0f}s)"
            )
            try:
                await run_chunk_pipeline(
                    chunk_id=chunk_id,
                    chunk_wav_path=chunk_path,
                    user_id=user_id,
                    meeting_prompt=meeting_prompt,
                    use_vocabulary=use_vocabulary,
                )
            except Exception as e:
                logger.error(
                    f"[UploadChunkPipeline] {recording_id} — "
                    f"Chunk {chunk_index} transcription failed (non-fatal): {e}",
                    exc_info=True,
                )
                # Mark chunk as error so finalize pipeline skips it gracefully.
                async with get_db() as db:
                    await db.execute(
                        text(
                            "UPDATE recording_chunks SET status='error' WHERE id=:cid"
                        ),
                        {"cid": chunk_id},
                    )
                    await db.commit()
            finally:
                # Delete chunk file immediately after transcription to free disk space.
                try:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                        logger.debug(
                            f"[UploadChunkPipeline] Deleted chunk file: {chunk_path}"
                        )
                except OSError as rm_err:
                    logger.warning(
                        f"[UploadChunkPipeline] Could not delete chunk file "
                        f"{chunk_path}: {rm_err}"
                    )

        logger.info(
            f"[UploadChunkPipeline] {recording_id} — "
            f"All chunks transcribed. Starting finalize pipeline."
        )

        # ── Step 4: Run finalize pipeline ───────────────────────────────────────
        await run_finalize_pipeline(
            recording_id=recording_id,
            full_wav_path=full_wav_path,
            chunk_ids=chunk_ids,
            user_id=user_id,
            meeting_prompt=meeting_prompt,
            participant_voice_ids=participant_voice_ids,
            use_vocabulary=use_vocabulary,
            speaker_summary=speaker_summary,
        )

        logger.info(
            f"[UploadChunkPipeline] {recording_id} — "
            f"===== COMPLETE ====="
        )
    finally:
        # Clean up chunk directory strictly (even on exit or failure)
        import shutil
        if os.path.exists(chunk_dir):
            try:
                shutil.rmtree(chunk_dir, ignore_errors=True)
                logger.info(f"[UploadChunkPipeline] Cleaned up temporary directory: {chunk_dir}")
            except Exception as e:
                logger.warning(f"[UploadChunkPipeline] Failed to clean up temp dir {chunk_dir}: {e}")
