"""
Chunk pipeline — transcription-only for a single audio chunk.

During a long recording every completed 10-minute segment is passed here.
This function ONLY performs:
  1. Whisper transcription + forced word alignment
  2. Language detection (cached on first chunk)
  3. Persisting the result to recording_chunks

All of the following are intentionally deferred to run_finalize_pipeline:
  - Speaker diarization (runs once on the full audio)
  - Speaker identification
  - Chunk/context summary generation (runs in parallel with diarization)
  - Minutes of Meeting generation

This keeps VRAM usage bounded during live recording sessions: only one
Whisper inference runs at a time and no LLM is loaded until finalize.
"""
import json
import logging
import asyncio
from typing import Optional

from sqlalchemy import text
from database import get_db, to_json
from services.transcription import transcribe
from services.prompt_builder import build_whisper_prompt
from services.dictionary_service import get_global_prompt, list_vocabulary

logger = logging.getLogger(__name__)


async def _update_chunk_status(chunk_id: str, status: str, extra: dict = None):
    patch = {"status": status}
    if extra:
        patch.update(extra)
    set_parts = [f"{k} = :{k}" for k in patch]
    set_clause = ", ".join(set_parts)
    try:
        async with get_db() as db:
            await db.execute(
                text(f"UPDATE recording_chunks SET {set_clause} WHERE id = :chunk_id"),
                {**patch, "chunk_id": chunk_id},
            )
            await db.commit()
    except Exception as e:
        logger.error(f"[ChunkPipeline] chunk {chunk_id} — DB status update FAILED: {e}", exc_info=True)


async def run_chunk_pipeline(
    chunk_id: str,
    chunk_wav_path: str,
    user_id: str,
    meeting_prompt: str = "",
    use_vocabulary: bool = False,
):
    """
    Transcribe a single audio chunk and persist the result.

    Runs: transcription + forced word alignment + incremental chunk summary.
    Speaker diarization is deferred to run_finalize_pipeline() which
    runs once on the full audio after recording stops.
    """
    logger.info(f"[ChunkPipeline] ===== START chunk={chunk_id} file={chunk_wav_path} =====")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error(f"[ChunkPipeline] {chunk_id} — No running event loop!")
        return

    # ── Build Whisper initial_prompt ─────────────────────────────────────
    initial_prompt = ""
    try:
        async with get_db() as db:
            global_prompt = await get_global_prompt(db, user_id)
            vocab_items = await list_vocabulary(db, user_id) if use_vocabulary else []
        vocab_words = [item["word"] for item in vocab_items]
        initial_prompt = build_whisper_prompt(
            global_prompt=global_prompt,
            meeting_prompt=meeting_prompt,
            vocabulary=vocab_words,
            use_vocabulary=use_vocabulary,
        )
    except Exception as e:
        logger.warning(f"[ChunkPipeline] {chunk_id} — Prompt build failed (non-fatal): {e}")

    # ── Fetch chunk metadata and check for cached language ───────────────
    recording_id = None
    chunk_index = 0
    detected_language = None

    try:
        async with get_db() as db:
            row = await db.execute(
                text("SELECT recording_id, chunk_index FROM recording_chunks WHERE id = :cid"),
                {"cid": chunk_id}
            )
            chunk_info = row.fetchone()
            if chunk_info:
                recording_id = chunk_info[0]
                chunk_index = chunk_info[1]

        if recording_id:
            async with get_db() as db:
                r = await db.execute(
                    text("SELECT status FROM recordings WHERE id = :rid"),
                    {"rid": recording_id}
                )
                rec_row = r.fetchone()
                if rec_row and rec_row[0] == "cancelled":
                    logger.info(f"[ChunkPipeline] Parent recording {recording_id} is CANCELLED. Aborting chunk processing.")
                    try:
                        if os.path.exists(chunk_wav_path):
                            os.remove(chunk_wav_path)
                    except Exception:
                        pass
                    return

        if recording_id and chunk_index > 0:
            async with get_db() as db:
                r = await db.execute(
                    text("SELECT language FROM recordings WHERE id = :rid"),
                    {"rid": recording_id}
                )
                rec_row = r.fetchone()
                if rec_row and rec_row[0]:
                    detected_language = rec_row[0]
    except Exception as e:
        logger.warning(f"[ChunkPipeline] {chunk_id} — Failed to fetch metadata, cancel status, or language cache: {e}")

    # ── Transcribe + align ───────────────────────────────────────────────
    try:
        t_result = await loop.run_in_executor(
            None,
            lambda: transcribe(chunk_wav_path, initial_prompt=initial_prompt, language=detected_language),
        )
    except Exception as e:
        logger.error(f"[ChunkPipeline] {chunk_id} — Transcription FAILED: {e}", exc_info=True)
        await _update_chunk_status(chunk_id, "error", {"error_message": str(e)})
        return

    segments = t_result.get("segments", [])
    raw_text = t_result.get("raw_text", "")
    aligned_result = t_result.get("aligned_result", {"segments": segments})
    language = t_result.get("language", "en")

    logger.info(f"[ChunkPipeline] {chunk_id} — Transcription OK: {len(segments)} segments, language={language}")

    # Save detected language on first chunk so later chunks can reuse it
    if recording_id and chunk_index == 0:
        try:
            async with get_db() as db:
                await db.execute(
                    text("UPDATE recordings SET language = :lang WHERE id = :rid"),
                    {"lang": language, "rid": recording_id}
                )
                await db.commit()
            logger.info(f"[ChunkPipeline] {recording_id} — Cached detected language '{language}' in recordings DB.")
        except Exception as db_err:
            logger.warning(f"[ChunkPipeline] {recording_id} — Failed to cache language to DB: {db_err}")

    # ── Persist chunk result ─────────────────────────────────────────────
    # Note: chunk_summary is intentionally NOT generated here.
    # It will be generated later in run_finalize_pipeline, in parallel
    # with diarization, so no LLM is loaded during the recording session.
    extra: dict = {
        "transcript": json.dumps(segments, default=str),
        "raw_text": raw_text,
        "aligned_result": json.dumps(aligned_result, default=str),
    }

    await _update_chunk_status(chunk_id, "done", extra)

    logger.info(f"[ChunkPipeline] ===== COMPLETE chunk={chunk_id} =====")
