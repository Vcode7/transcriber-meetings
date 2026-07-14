"""
Re-identification pipeline — re-run diarization + speaker ID on a completed recording.

This task:
  1. Re-diarizes the audio (full run via pyannote or energy fallback).
  2. Loads the user's current voice profiles from DB.
  3. Re-runs speaker identification (ECAPA-TDNN + VAD-gated matching).
  4. Re-assigns speaker labels to the existing word-level timestamps
     (reconstructed from the stored transcript — no re-transcription).
  5. Writes the updated transcript + speakers_detected back to the DB.
  6. Optionally regenerates MoM if speaker names changed.
  7. On any failure: leaves the existing transcript and status unchanged.

Isolation guarantees:
  - Uses a separate task key  \"reid_{recording_id}\" in active_tasks
    so it never collides with a live transcription task.
  - Does NOT import or modify run_pipeline / _run_pipeline_impl.
  - Does NOT re-run WhisperX transcription or forced alignment.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from config import settings
from database import get_db, dt_to_str, to_json, from_json
from services.diarization import diarize, is_pyannote_available
from services.identification import identify_speakers, refine_transcript_speakers_with_ecapa
from tasks.pipeline import (
    active_tasks,
    unregister_task,
    unload_all_models,
    _convert_diar_to_whisperx_format,
    _post_process_whisperx_segments,
    _assign_speakers_to_words_manual,
    _resegment_by_word_speakers,
    _filter_high_confidence_segments,
    _build_and_store_context_summary,
    _update_status_safe,
    _recover_gpu_and_run_llm,
)

logger = logging.getLogger(__name__)

# ── Task key prefix so reid tasks don't shadow live transcription tasks ───────
_REID_PREFIX = "reid_"


def _reid_key(recording_id: str) -> str:
    return f"{_REID_PREFIX}{recording_id}"


def register_reid_task(recording_id: str, task: asyncio.Task) -> None:
    key = _reid_key(recording_id)
    active_tasks[key] = task
    logger.info(f"[ReID] Registered task key={key}")


def unregister_reid_task(recording_id: str) -> None:
    key = _reid_key(recording_id)
    active_tasks.pop(key, None)
    logger.info(f"[ReID] Unregistered task key={key}")


def is_reid_running(recording_id: str) -> bool:
    key = _reid_key(recording_id)
    t = active_tasks.get(key)
    return t is not None and not t.done()


def _transcript_to_aligned_result(
    transcript: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Reconstruct a WhisperX-compatible aligned_result dict from the stored
    transcript segments.

    WhisperX assign_word_speakers() reads  result["segments"]  where each
    segment has  {"start", "end", "text", "words": [{"word", "start", "end",
    "probability"/"score"}, ...]}  — exactly the schema we persist.

    We do NOT need to re-run transcription or alignment; the stored word
    timestamps are already forced-aligned.
    """
    segments = []
    for seg in transcript:
        # Copy the segment but remove speaker-specific fields so
        # assign_word_speakers can write fresh speaker annotations.
        words = []
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "start": float(w.get("start", seg.get("start", 0.0))),
                "end": float(w.get("end", seg.get("end", 0.0))),
                # WhisperX uses "score"; we also keep "probability" for our own code.
                "score": float(w.get("probability", w.get("score", 1.0))),
                "probability": float(w.get("probability", w.get("score", 1.0))),
            })
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": seg.get("text", "").strip(),
            "words": words,
            "avg_logprob": float(seg.get("avg_logprob", 0.0)),
        })
    return {"segments": segments}


async def run_reidentify_pipeline(
    recording_id: str,
    file_path: str,
    user_id: str,
) -> None:
    """
    Top-level entry point — wraps _run_reidentify_impl with error handling,
    task cleanup, and model unloading.
    """
    try:
        await _run_reidentify_impl(recording_id, file_path, user_id)
    except asyncio.CancelledError:
        logger.info(f"[ReID] {recording_id} — Task CANCELLED.")
        try:
            async with get_db() as db:
                # Restore status to 'done' so the recording stays accessible.
                await db.execute(
                    text(
                        "UPDATE recordings SET status='done', progress=NULL, "
                        "error_message='Re-identification cancelled' WHERE id=:rid"
                    ),
                    {"rid": recording_id},
                )
                await db.commit()
        except Exception as e:
            logger.error(f"[ReID] {recording_id} — Failed to restore status after cancel: {e}")
        raise
    finally:
        unregister_reid_task(recording_id)
        # Only unload diarization + embedding; leave Qwen loaded if active.
        try:
            from services.diarization import unload_diarization_pipeline
            unload_diarization_pipeline()
        except Exception:
            pass
        try:
            from services.embedding import unload_encoder
            unload_encoder()
        except Exception:
            pass
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


async def _run_reidentify_impl(
    recording_id: str,
    file_path: str,
    user_id: str,
) -> None:
    """Core re-identification logic."""
    logger.info(f"[ReID] ===== START recording_id={recording_id} =====")
    _t_start = time.monotonic()
    loop = asyncio.get_running_loop()

    # ── Snapshot the existing transcript (safety net) ─────────────────────────
    original_transcript: Optional[str] = None
    original_speakers: Optional[str] = None
    original_status: str = "done"
    raw_text: str = ""
    try:
        async with get_db() as db:
            r = await db.execute(
                text(
                    "SELECT transcript, speakers_detected, status, raw_text "
                    "FROM recordings WHERE id = :rid"
                ),
                {"rid": recording_id},
            )
            row = r.mappings().fetchone()
        if row:
            original_transcript = row["transcript"]
            original_speakers = row["speakers_detected"]
            original_status = row.get("status") or "done"
            raw_text = row.get("raw_text") or ""
    except Exception as e:
        logger.error(f"[ReID] {recording_id} — Could not snapshot existing data: {e}")

    transcript_list: List[Dict[str, Any]] = from_json(original_transcript, [])
    if not transcript_list:
        logger.error(f"[ReID] {recording_id} — No stored transcript to re-identify. Aborting.")
        await _restore_status(recording_id, original_status, "No transcript found for re-identification.")
        return

    # ── Stage 1: Diarization ──────────────────────────────────────────────────
    logger.info(f"[ReID] {recording_id} — STAGE 1: Diarization")
    await _update_status_safe(recording_id, "processing", {"progress": "diarizing"})

    try:
        diar_segs = await loop.run_in_executor(None, diarize, file_path)
        logger.info(
            f"[ReID] {recording_id} — Diarization OK: {len(diar_segs)} segments "
            f"({'pyannote' if is_pyannote_available() else 'energy'})"
        )
    except Exception as e:
        logger.error(f"[ReID] {recording_id} — Diarization FAILED: {e}", exc_info=True)
        await _restore_status(recording_id, original_status, f"Diarization failed: {e}")
        return

    # ── Stage 2: Load voice profiles ──────────────────────────────────────────
    logger.info(f"[ReID] {recording_id} — STAGE 2: Loading voice profiles")
    try:
        async with get_db() as db:
            r = await db.execute(
                text("SELECT * FROM voice_profiles WHERE user_id = :uid LIMIT 100"),
                {"uid": user_id},
            )
            raw_profiles = r.mappings().fetchall()
        voice_profiles = []
        for p in raw_profiles:
            profile = dict(p)
            profile["embeddings"] = from_json(p["embeddings"], [])
            voice_profiles.append(profile)
        logger.info(f"[ReID] {recording_id} — {len(voice_profiles)} voice profiles loaded")
    except Exception as e:
        logger.error(f"[ReID] {recording_id} — Voice profile load FAILED: {e}", exc_info=True)
        voice_profiles = []

    # Load user similarity threshold
    threshold = settings.SPEAKER_SIMILARITY_THRESHOLD
    try:
        async with get_db() as db:
            r = await db.execute(
                text("SELECT speaker_similarity_threshold FROM user_settings WHERE user_id = :uid"),
                {"uid": user_id},
            )
            row = r.mappings().fetchone()
        if row and row.get("speaker_similarity_threshold") is not None:
            threshold = float(row["speaker_similarity_threshold"])
    except Exception:
        pass
    logger.info(f"[ReID] {recording_id} — Similarity threshold: {threshold}")

    # ── Stage 3: Speaker identification ───────────────────────────────────────
    logger.info(f"[ReID] {recording_id} — STAGE 3: Speaker identification")
    await _update_status_safe(recording_id, "processing", {"progress": "identifying_speakers"})

    try:
        identified_segs = await loop.run_in_executor(
            None,
            lambda: identify_speakers(
                file_path=file_path,
                diarization_segments=diar_segs,
                voice_profiles=voice_profiles,
                similarity_threshold=threshold,
            ),
        )
        logger.info(f"[ReID] {recording_id} — Speaker ID OK: {len(identified_segs)} segments")
    except Exception as e:
        logger.error(f"[ReID] {recording_id} — Speaker ID FAILED: {e}", exc_info=True)
        logger.warning(f"[ReID] {recording_id} — Falling back to generic speaker labels")
        identified_segs = [
            {**seg, "speaker_label": f"Speaker {i+1}", "speaker_profile_id": None, "similarity": 0.0}
            for i, seg in enumerate(diar_segs)
        ]

    # ── Stage 4: Word→speaker assignment ──────────────────────────────────────
    logger.info(f"[ReID] {recording_id} — STAGE 4: Word-speaker assignment")
    await _update_status_safe(recording_id, "processing", {"progress": "updating_transcript"})

    # Reconstruct aligned_result from stored transcript (avoids re-transcription)
    aligned_result = _transcript_to_aligned_result(transcript_list)

    speaker_segments: List[Dict[str, Any]] = []
    assignment_method = "whisperx"
    try:
        import whisperx  # type: ignore
        diar_df = _convert_diar_to_whisperx_format(identified_segs)
        if diar_df is not None and not diar_df.empty:
            wx_assigned = whisperx.assign_word_speakers(diar_df, aligned_result, fill_nearest=True)
            speaker_segments = _post_process_whisperx_segments(wx_assigned, identified_segs)
            speaker_segments = _resegment_by_word_speakers(speaker_segments)
            logger.info(
                f"[ReID] {recording_id} — whisperx.assign_word_speakers OK → "
                f"{len(speaker_segments)} segments after resegmentation"
            )
        else:
            raise ValueError("Empty diarization dataframe")
    except Exception as e:
        logger.warning(
            f"[ReID] {recording_id} — whisperx.assign_word_speakers failed ({e}); "
            "using manual assignment"
        )
        assignment_method = "manual"
        speaker_segments = _assign_speakers_to_words_manual(aligned_result, identified_segs)
        speaker_segments = _resegment_by_word_speakers(speaker_segments)

    # ── ECAPA refinement pass on final segments ──
    logger.info(f"[ReID] {recording_id} — Running ECAPA refinement pass on re-segmented transcript")
    speaker_segments = refine_transcript_speakers_with_ecapa(
        file_path=file_path,
        speaker_segments=speaker_segments,
        voice_profiles=voice_profiles,
        similarity_threshold=threshold,
        use_model_default_threshold=True,
    )

    # ── Stage 5: Build final segments ─────────────────────────────────────────
    final_segments = []
    for seg in speaker_segments:
        final_segments.append({
            "speaker_label": seg.get("speaker_label") or seg.get("speaker") or "Speaker 1",
            "speaker_profile_id": seg.get("speaker_profile_id"),
            "speaker": seg.get("speaker"),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "words": seg.get("words", []),
            "is_overlap": seg.get("is_overlap", False),
            "overlap_regions": seg.get("overlap_regions", []),
            # Preserve original avg_logprob if present
            "avg_logprob": seg.get("avg_logprob", 0.0),
        })

    speakers_detected = list({
        s["speaker_label"]
        for s in final_segments
        if not s.get("is_overlap") and s["speaker_label"] not in ("Unknown",)
    })
    logger.info(f"[ReID] {recording_id} — Speakers detected: {speakers_detected}")

    # ── Stage 6: Persist updated transcript ───────────────────────────────────
    logger.info(f"[ReID] {recording_id} — STAGE 6: Persisting updated transcript")
    now = datetime.now(timezone.utc)
    try:
        async with get_db() as db:
            await db.execute(
                text(
                    "UPDATE recordings SET "
                    "  transcript = :transcript, "
                    "  speakers_detected = :speakers_detected, "
                    "  status = 'done', "
                    "  progress = NULL, "
                    "  speaker_reid_at = :reid_at "
                    "WHERE id = :rid"
                ),
                {
                    "transcript": to_json(final_segments),
                    "speakers_detected": to_json(speakers_detected),
                    "reid_at": dt_to_str(now),
                    "rid": recording_id,
                },
            )
            await db.commit()
        logger.info(f"[ReID] {recording_id} — Transcript updated in DB ✓")
    except Exception as e:
        logger.error(f"[ReID] {recording_id} — DB write FAILED: {e}", exc_info=True)
        # Restore original status so the recording remains accessible
        await _restore_status(recording_id, original_status, f"Transcript update failed: {e}")
        return

    # ── Stage 7: Conditionally regenerate MoM (non-blocking) ─────────────────
    old_speakers = set(json.loads(original_speakers or "[]"))
    new_speakers = set(speakers_detected)
    speakers_changed = old_speakers != new_speakers

    if speakers_changed:
        logger.info(
            f"[ReID] {recording_id} — Speaker names changed "
            f"({old_speakers} → {new_speakers}). Regenerating MoM."
        )
        asyncio.create_task(
            _regenerate_mom(recording_id, user_id, final_segments, raw_text, speakers_detected, loop)
        )
    else:
        logger.info(f"[ReID] {recording_id} — Speakers unchanged — skipping MoM regeneration.")

    elapsed = round(time.monotonic() - _t_start, 2)
    logger.info(f"[ReID] ===== COMPLETE recording_id={recording_id} in {elapsed}s =====")


async def _restore_status(recording_id: str, original_status: str, reason: str) -> None:
    """Restore the recording's status to its pre-reid value on failure."""
    try:
        async with get_db() as db:
            await db.execute(
                text(
                    "UPDATE recordings SET status = :st, progress = NULL "
                    "WHERE id = :rid"
                ),
                {"st": original_status, "rid": recording_id},
            )
            await db.commit()
        logger.info(f"[ReID] {recording_id} — Status restored to '{original_status}' after error: {reason}")
    except Exception as e:
        logger.error(f"[ReID] {recording_id} — Could not restore status: {e}")


async def _regenerate_mom(
    recording_id: str,
    user_id: str,
    final_segments: List[Dict[str, Any]],
    raw_text: str,
    speakers_detected: List[str],
    loop,
) -> None:
    """
    Regenerate Minutes of Meeting after speaker names changed.
    Runs as a fire-and-forget task — failures are logged but do not affect
    the recording's status.

    Includes self-contained GPU recovery: if context_summary or mom_data
    come back empty (even without an exception), the function unloads
    non-LLM models, purges CUDA memory, and retries generation.
    """
    try:
        from services.llm import generate_mom

        conf_threshold = settings.MIN_AVG_SEGMENT_CONFIDENCE
        filtered = _filter_high_confidence_segments(final_segments, conf_threshold)
        logger.info(
            f"[ReID/MoM] {recording_id} — MoM input: "
            f"{len(filtered)}/{len(final_segments)} high-confidence segments"
        )

        # Build context summary
        try:
            context_summary = await _build_and_store_context_summary(
                recording_id, filtered, raw_text, loop
            )
        except Exception as ctx_err:
            logger.warning(
                f"[ReID/MoM] {recording_id} — Initial context summary generation failed: {ctx_err}. "
                "Triggering VRAM recovery and retry..."
            )
            context_summary = None

        # Recover when result is empty/None (with or without exception)
        if not (context_summary and context_summary.strip()):
            if context_summary is not None:
                logger.warning(
                    f"[ReID/MoM] {recording_id} — Context summary returned empty without exception. "
                    "Triggering VRAM recovery and retry..."
                )
            context_summary = await _recover_gpu_and_run_llm(
                recording_id=recording_id,
                task_name="context_summary (ReID/MoM)",
                fn_to_run=lambda: _build_and_store_context_summary(
                    recording_id, filtered, raw_text, loop
                )
            )

        if context_summary and context_summary.strip():
            logger.info(
                f"[ReID/MoM] {recording_id} — Context summary ready ({len(context_summary.split())} words)"
            )
        else:
            logger.warning(
                f"[ReID/MoM] {recording_id} — Context summary empty after recovery. "
                "Will generate MoM without context."
            )

        async with get_db() as db:
            r = await db.execute(
                text("SELECT filename, created_at, duration FROM recordings WHERE id = :rid"),
                {"rid": recording_id},
            )
            rec_meta = r.mappings().fetchone()

        recording_meta = {
            "filename": rec_meta["filename"] if rec_meta else "Meeting",
            "created_at": rec_meta["created_at"] if rec_meta else "",
            "duration": rec_meta["duration"] if rec_meta else 0,
            "speakers_detected": speakers_detected,
        }

        # Generate MoM
        try:
            mom_data = await loop.run_in_executor(
                None,
                lambda: generate_mom(
                    filtered, recording_meta,
                    context=context_summary or None,
                ),
            )
        except Exception as mom_err:
            logger.warning(
                f"[ReID/MoM] {recording_id} — Initial MoM generation failed: {mom_err}. "
                "Triggering VRAM recovery and retry..."
            )
            mom_data = None

        # Recover when MoM is missing or clearly invalid (empty points_discussed)
        _mom_invalid = (
            not mom_data
            or not mom_data.get("points_discussed")
        )
        if _mom_invalid:
            if mom_data is not None:
                logger.warning(
                    f"[ReID/MoM] {recording_id} — MoM returned empty/invalid without exception. "
                    "Triggering VRAM recovery and retry..."
                )
            mom_data = await _recover_gpu_and_run_llm(
                recording_id=recording_id,
                task_name="Minutes of Meeting (ReID/MoM)",
                fn_to_run=lambda: loop.run_in_executor(
                    None,
                    lambda: generate_mom(
                        filtered, recording_meta,
                        context=context_summary or None,
                    )
                )
            )

        if mom_data:
            import uuid
            mom_id = str(uuid.uuid4())
            now_str = dt_to_str(datetime.now(timezone.utc))
            async with get_db() as db:
                # Upsert: delete old MoM for this recording then insert fresh
                await db.execute(
                    text("DELETE FROM minutes_of_meeting WHERE recording_id = :rid"),
                    {"rid": recording_id},
                )
                await db.execute(
                    text("""
                        INSERT INTO minutes_of_meeting (
                            id, recording_id, user_id,
                            title, date, duration,
                            participants, introduction,
                            points_discussed, action_items, conclusion,
                            agenda_items, discussion_summary, decisions,
                            risks_concerns, next_steps, versions,
                            is_draft, created_at, updated_at
                        ) VALUES (
                            :id, :rid, :uid,
                            :title, :date, :duration,
                            :participants, :intro,
                            :points, :actions, :conclusion,
                            '[]', NULL, '[]',
                            '[]', '[]', '[]',
                            0, :now, :now
                        )
                    """),
                    {
                        "id": mom_id,
                        "rid": recording_id,
                        "uid": user_id,
                        "title": mom_data.get("title", ""),
                        "date": mom_data.get("date", ""),
                        "duration": recording_meta["duration"],
                        "participants": to_json(mom_data.get("participants", [])),
                        "intro": mom_data.get("introduction", ""),
                        "points": to_json(mom_data.get("points_discussed", [])),
                        "actions": to_json(mom_data.get("action_items", [])),
                        "conclusion": mom_data.get("conclusion", ""),
                        "now": now_str,
                    },
                )
                await db.commit()
            logger.info(f"[ReID/MoM] {recording_id} — MoM regenerated ✓")
        else:
            logger.warning(f"[ReID/MoM] {recording_id} — generate_mom returned empty data after all recovery attempts")

    except Exception as e:
        logger.warning(
            f"[ReID/MoM] {recording_id} — MoM regeneration failed (non-fatal): {e}",
            exc_info=True,
        )
    finally:
        try:
            from services.ai_provider import QwenProvider
            QwenProvider.unload_model()
        except Exception:
            pass
