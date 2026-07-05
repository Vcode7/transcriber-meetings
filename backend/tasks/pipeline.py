"""
Background pipeline: transcribe → diarize → identify speakers → generate MoM → save.

WhisperX integration:
  - transcribe() returns an `aligned_result` dict consumed by whisperx.assign_word_speakers()
  - assign_word_speakers() annotates every word with the speaker from pyannote diarization
  - We then run our voice-profile identification on top to map pyannote IDs → human names

Phase 2 now generates the Minutes of Meeting (MoM) automatically.
AI Insights (summary, key points, action items) are triggered manually by the user.
"""
import json
import logging
import asyncio
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any
from sqlalchemy import text
import hashlib
from database import get_db, dt_to_str, to_json, from_json
from services.transcription import transcribe
from services.diarization import diarize, is_pyannote_available
from services.identification import identify_speakers
from services.llm import (
    generate_summary, generate_key_points, generate_action_items,
    generate_short_summary, generate_detailed_summary,
    generate_speaker_summaries, generate_mom, build_context_summary,
)
from services.prompt_builder import build_whisper_prompt
from services.dictionary_service import get_global_prompt, list_vocabulary
from config import settings
logger = logging.getLogger(__name__)

active_tasks: Dict[str, asyncio.Task] = {}

def register_task(recording_id: str, task: asyncio.Task):
    active_tasks[recording_id] = task
    logger.info(f"[Pipeline] Registered active task for recording={recording_id}. Total active tasks: {len(active_tasks)}")

def unregister_task(recording_id: str):
    active_tasks.pop(recording_id, None)
    logger.info(f"[Pipeline] Unregistered task for recording={recording_id}. Total active tasks: {len(active_tasks)}")

async def cancel_task(recording_id: str) -> bool:
    task = active_tasks.get(recording_id)
    if task:
        logger.info(f"[Pipeline] Cancelling active task for recording={recording_id}...")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[Pipeline] Exception during task cancellation of {recording_id}: {e}")
        finally:
            active_tasks.pop(recording_id, None)
        return True
    return False




def unload_all_models():
    """Unload all models (WhisperX, Pyannote Diarization, VoiceEncoder, Qwen LLM, Overlap Model) to release RAM/VRAM."""
    logger.info("[Pipeline] Initiating global AI model memory cleanup...")
    
    # 1. Unload WhisperX
    try:
        from services.transcription import unload_whisperx_model
        unload_whisperx_model()
    except Exception as e:
        logger.warning(f"[Pipeline] Failed to unload WhisperX model: {e}")
        
    # 2. Unload Pyannote Diarization
    try:
        from services.diarization import unload_diarization_pipeline
        unload_diarization_pipeline()
    except Exception as e:
        logger.warning(f"[Pipeline] Failed to unload pyannote diarization pipeline: {e}")
        
    # 3. Unload Resemblyzer Encoder
    try:
        from services.embedding import unload_encoder
        unload_encoder()
    except Exception as e:
        logger.warning(f"[Pipeline] Failed to unload speaker encoder: {e}")
        
    # 4. Unload Qwen LLM
    try:
        from services.ai_provider import QwenProvider
        QwenProvider.unload_model()
    except Exception as e:
        logger.warning(f"[Pipeline] Failed to unload Qwen LLM: {e}")
        
    # 5. Unload Overlap Model
    try:
        from main import unload_overlap_model
        unload_overlap_model()
    except Exception as e:
        logger.warning(f"[Pipeline] Failed to unload overlap model: {e}")
        
    # Force garbage collection and CUDA cache empty
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"[Pipeline] CUDA empty_cache called. Current VRAM allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
    except Exception:
        pass
    logger.info("[Pipeline] Global AI model memory cleanup complete.")


# ── Confidence threshold (from config) ────────────────────────────────────────
# Imported lazily inside functions to avoid circular imports at module level.

def _filter_high_confidence_segments(
    segments: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    Return only segments whose average word-level confidence >= threshold.
    Segments with no word data are always included (no basis to exclude).
    Used to avoid feeding garbled/low-quality text to MoM and AI Insights.
    """
    filtered = []
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            # No word-level data — include unconditionally
            filtered.append(seg)
            continue
        probs = [
            w.get("probability", w.get("score", 1.0))
            for w in words
            if w.get("probability") is not None or w.get("score") is not None
        ]
        if not probs:
            filtered.append(seg)
            continue
        avg = sum(probs) / len(probs)
        if avg >= threshold:
            filtered.append(seg)
    return filtered


def _raw_text_hash(raw_text: str) -> str:
    """MD5 hex digest of raw_text — used to detect stale context_summary."""
    return hashlib.md5((raw_text or "").encode("utf-8", errors="replace")).hexdigest()


async def _build_and_store_context_summary(
    recording_id: str,
    filtered_segments: List[Dict[str, Any]],
    raw_text: str,
    loop,
) -> str:
    """
    Build the context_summary for a recording and persist it in the DB.

    Steps:
      1. Compute MD5 of raw_text to use as a staleness-check hash.
      2. Check if the DB already has a valid (non-stale) context_summary.
      3. If so, return it without any inference.
      4. Otherwise run build_context_summary() via executor and store the result.

    Returns the context_summary string (may be empty on failure).
    """
    current_hash = _raw_text_hash(raw_text)

    # Check for an existing valid context
    try:
        async with get_db() as db:
            row = await db.execute(
                text(
                    "SELECT context_summary, context_summary_hash "
                    "FROM recordings WHERE id = :rid"
                ),
                {"rid": recording_id},
            )
            existing = row.mappings().fetchone()
        if (
            existing
            and existing["context_summary"]
            and existing["context_summary_hash"] == current_hash
        ):
            logger.info(
                f"[Pipeline] {recording_id} — Reusing cached context_summary "
                f"(hash={current_hash[:8]}…)"
            )
            return existing["context_summary"]
    except Exception as e:
        logger.warning(f"[Pipeline] {recording_id} — Could not check cached context (non-fatal): {e}")

    # Build fresh context
    logger.info(f"[Pipeline] {recording_id} — Building context_summary from {len(filtered_segments)} segments…")
    try:
        ctx = await loop.run_in_executor(
            None, lambda: build_context_summary(filtered_segments)
        )
    except Exception as e:
        logger.warning(f"[Pipeline] {recording_id} — build_context_summary failed (non-fatal): {e}")
        return ""

    # Persist to DB
    if ctx:
        try:
            async with get_db() as db:
                await db.execute(
                    text(
                        "UPDATE recordings SET "
                        "context_summary = :ctx, context_summary_hash = :h "
                        "WHERE id = :rid"
                    ),
                    {"ctx": ctx, "h": current_hash, "rid": recording_id},
                )
                await db.commit()
            logger.info(
                f"[Pipeline] {recording_id} — context_summary stored "
                f"({len(ctx.split())} words, hash={current_hash[:8]}…) ✓"
            )
        except Exception as e:
            logger.warning(f"[Pipeline] {recording_id} — Failed to store context_summary (non-fatal): {e}")

    return ctx or ""


# ── Analytics helpers ──────────────────────────────────────────────────────

def _file_size_bytes(path: str) -> int | None:
    """Return file size in bytes, or None on failure."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _word_confidence_stats(segments: List[Dict[str, Any]]):
    """
    Return (avg_conf, min_conf, total_words) across all word-level
    probability scores in the segment list.
    """
    all_probs = [
        w.get("probability", w.get("score", 1.0))
        for seg in segments
        for w in seg.get("words", [])
        if w.get("probability") is not None or w.get("score") is not None
    ]
    if not all_probs:
        return None, None, 0
    avg_conf = round(sum(all_probs) / len(all_probs), 4)
    min_conf = round(min(all_probs), 4)
    return avg_conf, min_conf, len(all_probs)


def _count_total_words(segments: List[Dict[str, Any]]) -> int:
    return sum(len(seg.get("words", [])) for seg in segments)


async def _emit_analytics(metrics: Dict[str, Any]) -> None:
    """Fire-and-forget analytics write — any exception is silently swallowed."""
    try:
        from services.analytics import record_pipeline_analytics
        await record_pipeline_analytics(metrics)
    except Exception as exc:
        logger.warning(f"[Pipeline] Analytics emit failed (non-fatal): {exc}")


def _convert_diar_to_whisperx_format(
    diar_segments: List[Dict[str, Any]]
) -> Any:
    """
    Convert our pyannote diarization output to the format whisperx.assign_word_speakers
    expects.  WhisperX wants a pandas DataFrame (or object with an .itertracks()-like
    interface), but its internal `assign_word_speakers` actually just needs a list of
    dicts with keys: segment, label.

    We replicate the minimal DataFrame structure whisperx expects by wrapping in a
    simple object — or we can use the pandas route which is more robust.
    """
    try:
        import pandas as pd
        rows = []
        for seg in diar_segments:
            rows.append({
                "segment": {"start": seg["start"], "end": seg["end"]},
                "label": seg["speaker"],
                "speaker": seg["speaker"],
            })
        return pd.DataFrame(rows)
    except ImportError:
        return None


def _assign_speakers_to_words_manual(
    aligned_result: Dict[str, Any],
    identified_segs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Fallback word→speaker assignment (used if whisperx.assign_word_speakers
    is not available or fails).

    For every word, find the identified diarization segment with the greatest
    time overlap and assign its speaker_label.
    """
    out_segments = []
    for seg in aligned_result.get("segments", []):
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Find best segment match by time overlap
        best_label = "Unknown"
        best_profile_id = None
        best_is_overlap = False
        best_overlap = 0.0
        for s in identified_segs:
            ov = max(0.0, min(seg_end, s["end"]) - max(seg_start, s["start"]))
            if ov > best_overlap:
                best_overlap = ov
                best_label = s.get("speaker_label", "Unknown")
                best_profile_id = s.get("speaker_profile_id")
                best_is_overlap = s.get("is_overlap", False)

        # Per-word speaker (fine-grained)
        enriched_words = []
        for w in seg.get("words", []):
            w_start = w.get("start", seg_start)
            w_end = w.get("end", seg_end)
            w_label = best_label
            w_profile_id = best_profile_id
            for s in identified_segs:
                ov = max(0.0, min(w_end, s["end"]) - max(w_start, s["start"]))
                if ov > 0.0:
                    w_label = s.get("speaker_label", w_label)
                    w_profile_id = s.get("speaker_profile_id", w_profile_id)
                    break
            enriched_words.append({
                "word": w.get("word", "").strip(),
                "start": float(w.get("start", seg_start)),
                "end": float(w.get("end", seg_end)),
                "probability": float(
                    w.get("probability", w.get("score", 1.0))
                ),
                "speaker_label": w_label,
            })

        out_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg.get("text", "").strip(),
            "words": enriched_words,
            "avg_logprob": seg.get("avg_logprob", 0.0),
            "speaker_label": best_label,
            "speaker_profile_id": best_profile_id,
            "is_overlap": best_is_overlap,
        })

    return out_segments


async def _update_status_safe(recording_id: str, status_val: str, extra: dict = None):
    """
    Update recording status in DB. Uses its own DB session to avoid
    cross-session state corruption.
    """
    patch = {"status": status_val}
    if extra:
        patch.update(extra)

    set_parts = [f"{k} = :{k}" for k in patch]
    set_clause = ", ".join(set_parts)

    try:
        async with get_db() as db:
            await db.execute(
                text(f"UPDATE recordings SET {set_clause} WHERE id = :recording_id"),
                {**patch, "recording_id": recording_id},
            )
            await db.commit()
        logger.info(f"[Pipeline] {recording_id} — status → {status_val} (extra={list(extra.keys()) if extra else []})")
    except Exception as e:
        logger.error(f"[Pipeline] {recording_id} — DB status update FAILED: {e}", exc_info=True)


async def run_pipeline(
    recording_id: str,
    file_path: str,
    user_id: str,
    meeting_prompt: str = "",
    participant_voice_ids: List[str] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
):
    try:
        await _run_pipeline_impl(
            recording_id=recording_id,
            file_path=file_path,
            user_id=user_id,
            meeting_prompt=meeting_prompt,
            participant_voice_ids=participant_voice_ids,
            use_vocabulary=use_vocabulary,
            speaker_summary=speaker_summary,
        )
    except asyncio.CancelledError:
        logger.info(f"[Pipeline] {recording_id} — Task was CANCELLED.")
        try:
            async with get_db() as db:
                await db.execute(
                    text("UPDATE recordings SET status='cancelled', progress=NULL, error_message='Cancelled by user' WHERE id=:rid"),
                    {"rid": recording_id}
                )
                await db.commit()
        except Exception as e:
            logger.error(f"[Pipeline] {recording_id} — Failed to update cancelled status in DB: {e}")
        raise
    finally:
        unregister_task(recording_id)
        unload_all_models()


async def _run_pipeline_impl(
    recording_id: str,
    file_path: str,
    user_id: str,
    meeting_prompt: str = "",
    participant_voice_ids: List[str] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
):
    """
    Full async pipeline for a single recording.

    Phase 1 (fast): Transcription → Diarization → Speaker ID → save transcript
                    Sets status = 'transcript_ready' so frontend can show transcript immediately.
    Phase 2 (slow): Qwen3 AI insights → set status = 'done'
    """
    participant_voice_ids = participant_voice_ids or []
    logger.info(f"[Pipeline] ===== START recording_id={recording_id} file={file_path} =====")

    # ── Analytics accumulators ─────────────────────────────────────────────
    _pipeline_start = time.monotonic()
    _job_created_at = dt_to_str(datetime.now(timezone.utc))
    _analytics: Dict[str, Any] = {
        "recording_id": recording_id,
        "user_id": user_id,
        "pipeline_type": "single",
        "source_type": "upload",
        "file_size_bytes": _file_size_bytes(file_path),
        "whisper_device": settings.WHISPER_DEVICE,
        "whisper_compute_type": settings.WHISPER_COMPUTE_TYPE,
        "whisper_model_size": settings.WHISPER_MODEL_SIZE,
        "similarity_threshold": settings.SPEAKER_SIMILARITY_THRESHOLD,
        "pyannote_available": is_pyannote_available(),
        "use_vocabulary": use_vocabulary,
        "meeting_prompt_chars": len(meeting_prompt) if meeting_prompt else 0,
        "job_created_at": _job_created_at,
        "final_status": "error",  # default; overwritten on success
    }
    # Detect source type from file_path prefix
    _fname = os.path.basename(file_path)
    if _fname.startswith("live_"):
        _analytics["source_type"] = "live"
    elif _fname.startswith("rec_"):
        _analytics["source_type"] = "upload"

    try:
        # Get the running event loop — works correctly in all Python 3.10+ async contexts
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error(f"[Pipeline] {recording_id} — No running event loop! Cannot start pipeline.")
        return

    # ───────────────────────────────────────────────────────────────────────
    # PHASE 0: Build Whisper initial_prompt
    # ───────────────────────────────────────────────────────────────────────

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
        logger.info(
            f"[Pipeline] {recording_id} — Built initial_prompt "
            f"({len(initial_prompt)} chars, {len(vocab_words)} vocab terms)"
        )
        _analytics["vocab_term_count"] = len(vocab_words)
        _analytics["initial_prompt_chars"] = len(initial_prompt)
    except Exception as e:
        logger.warning(f"[Pipeline] {recording_id} — Prompt build failed (non-fatal): {e}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1: Transcription → Diarization → Speaker ID → Save transcript
    # ─────────────────────────────────────────────────────────────────────

    try:
        # ── Stage 1: Transcription (WhisperX + alignment) ─────────────
        logger.info(f"[Pipeline] {recording_id} — STAGE 1: Transcribing {file_path}")
        await _update_status_safe(recording_id, "processing", {"progress": "transcribing"})

        _t0_transcription = time.monotonic()
        try:
            t_result = await loop.run_in_executor(
                None,
                lambda: transcribe(file_path, initial_prompt=initial_prompt),
            )
        except Exception as e:
            logger.error(f"[Pipeline] {recording_id} — Transcription FAILED: {e}", exc_info=True)
            _analytics["error_stage"] = "transcription"
            _analytics["error_message"] = str(e)
            _analytics["total_pipeline_sec"] = round(time.monotonic() - _pipeline_start, 3)
            await _emit_analytics(_analytics)
            await _update_status_safe(recording_id, "error", {"error_message": f"Transcription failed: {str(e)}"})
            unload_all_models()
            return
        _analytics["transcription_sec"] = round(time.monotonic() - _t0_transcription, 3)

        transcript_segs = t_result["segments"]
        raw_text = t_result["raw_text"]
        language = t_result.get("language", "en")
        aligned_result = t_result.get("aligned_result", {"segments": transcript_segs})
        # Check whether forced alignment actually ran (aligned_result differs from raw segments)
        _alignment_used = aligned_result is not t_result and aligned_result != {"segments": transcript_segs}
        _avg_conf, _min_conf, _word_cnt = _word_confidence_stats(transcript_segs)
        _analytics.update({
            "language_detected": language,
            "transcript_segment_count": len(transcript_segs),
            "transcript_word_count": _word_cnt or _count_total_words(transcript_segs),
            "avg_word_confidence": _avg_conf,
            "min_word_confidence": _min_conf,
            "alignment_used": _alignment_used,
        })
        logger.info(f"[Pipeline] {recording_id} — Transcription OK: {len(transcript_segs)} segments, lang={language}")

        # ── Stage 2: Diarization ──────────────────────────────────────
        logger.info(f"[Pipeline] {recording_id} — STAGE 2: Diarizing")
        await _update_status_safe(recording_id, "processing", {"progress": "diarizing"})
        _t0_diarization = time.monotonic()
        diar_segs = await loop.run_in_executor(None, diarize, file_path)
        _analytics["diarization_sec"] = round(time.monotonic() - _t0_diarization, 3)
        _analytics["diarization_engine"] = "pyannote" if is_pyannote_available() else "energy"
        _analytics["diar_raw_segment_count"] = len(diar_segs)
        _analytics["diar_overlap_segment_count"] = sum(1 for s in diar_segs if s.get("is_overlap"))
        _analytics["diar_unique_speakers"] = len({s["speaker"] for s in diar_segs})
        logger.info(f"[Pipeline] {recording_id} — Diarization OK: {len(diar_segs)} segments")

        # ── Stage 3: Load voice profiles ──────────────────────────────
        logger.info(f"[Pipeline] {recording_id} — STAGE 3: Loading voice profiles for user {user_id}")
        if participant_voice_ids:
            logger.info(f"[Pipeline] {recording_id} — Filtering to {len(participant_voice_ids)} selected voice profiles")
        try:
            async with get_db() as db:
                r = await db.execute(
                    text("SELECT * FROM voice_profiles WHERE user_id = :uid LIMIT 100"),
                    {"uid": user_id},
                )
                raw_profiles = r.mappings().fetchall()
        except Exception as e:
            logger.error(f"[Pipeline] {recording_id} — Voice profile load FAILED: {e}", exc_info=True)
            raw_profiles = []

        voice_profiles = []
        for p in raw_profiles:
            profile = dict(p)
            profile["embeddings"] = from_json(p["embeddings"], [])
            voice_profiles.append(profile)

        # Filter to selected participants (strict) if specified
        if participant_voice_ids:
            voice_profiles = [
                vp for vp in voice_profiles
                if vp.get("id") in participant_voice_ids
            ]
            logger.info(
                f"[Pipeline] {recording_id} — After participant filter: {len(voice_profiles)} voice profiles"
            )
        else:
            logger.info(f"[Pipeline] {recording_id} — Loaded {len(voice_profiles)} voice profiles")

        try:
            async with get_db() as db:
                r = await db.execute(
                    text("SELECT * FROM user_settings WHERE user_id = :uid"),
                    {"uid": user_id},
                )
                user_settings_row = r.mappings().fetchone()
        except Exception as e:
            logger.error(f"[Pipeline] {recording_id} — User settings load FAILED: {e}", exc_info=True)
            user_settings_row = None

        threshold = (
            float(user_settings_row["speaker_similarity_threshold"])
            if user_settings_row and user_settings_row.get("speaker_similarity_threshold") is not None
            else settings.SPEAKER_SIMILARITY_THRESHOLD
        )
        logger.info(f"[Pipeline] {recording_id} — Speaker similarity threshold: {threshold}")

        # ── Stage 4: Speaker identification ───────────────────────────
        logger.info(f"[Pipeline] {recording_id} — STAGE 4: Identifying speakers")
        await _update_status_safe(recording_id, "processing", {"progress": "identifying_speakers"})
        _analytics["voice_profiles_loaded"] = len(voice_profiles)
        _analytics["similarity_threshold"] = threshold

        _t0_speaker_id = time.monotonic()
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
        except Exception as e:
            logger.error(f"[Pipeline] {recording_id} — Speaker identification FAILED: {e}", exc_info=True)
            # Non-fatal: build generic identified segs from diar_segs
            logger.warning(f"[Pipeline] {recording_id} — Falling back to generic speaker labels")
            identified_segs = []
            for i, seg in enumerate(diar_segs):
                identified_segs.append({
                    **seg,
                    "speaker_label": f"Speaker {i+1}",
                    "speaker_profile_id": None,
                    "similarity": 0.0,
                })
        _analytics["speaker_id_sec"] = round(time.monotonic() - _t0_speaker_id, 3)
        # Compute speaker ID quality metrics
        _sims = [s.get("similarity", 0.0) for s in identified_segs if not s.get("is_overlap")]
        _identified = sum(1 for s in identified_segs if s.get("speaker_profile_id") is not None)
        _analytics["speaker_identified_count"] = _identified
        _analytics["speaker_unidentified_count"] = len(identified_segs) - _identified
        _analytics["avg_speaker_similarity"] = round(sum(_sims) / len(_sims), 4) if _sims else None

        logger.info(f"[Pipeline] {recording_id} — Speaker ID OK: {len(identified_segs)} identified segments")

        # ── Stage 5: Word-level speaker assignment (WhisperX-native) ──
        logger.info(f"[Pipeline] {recording_id} — STAGE 5: Word→speaker assignment")
        _speaker_assignment_method = "whisperx"
        try:
            import whisperx
            diar_df = _convert_diar_to_whisperx_format(identified_segs)
            if diar_df is not None and not diar_df.empty:
                wx_assigned = whisperx.assign_word_speakers(diar_df, aligned_result)
                speaker_segments = _post_process_whisperx_segments(wx_assigned, identified_segs)
                logger.info(f"[Pipeline] {recording_id} — whisperx.assign_word_speakers succeeded")
            else:
                raise ValueError("Empty diarization dataframe — using manual assignment")
        except Exception as e:
            logger.warning(
                f"[Pipeline] {recording_id} — whisperx.assign_word_speakers failed ({e}), "
                "falling back to manual assignment."
            )
            _speaker_assignment_method = "manual"
            speaker_segments = _assign_speakers_to_words_manual(aligned_result, identified_segs)
        _analytics["speaker_assignment_method"] = _speaker_assignment_method

        logger.info(f"[Pipeline] {recording_id} — Speaker segments: {len(speaker_segments)}")

        # ── Stage 6: Build final segments ─────────────────────────────
        logger.info(f"[Pipeline] {recording_id} — STAGE 6: Building final segments")
        final_segments = []
        for seg in speaker_segments:
            words = seg.get("words", [])
            final_segments.append({
                "speaker_label": seg.get("speaker_label", "Unknown"),
                "speaker_profile_id": seg.get("speaker_profile_id"),
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "words": words,
                "is_overlap": seg.get("is_overlap", False),
            })

        speakers_detected = list({
            s["speaker_label"]
            for s in final_segments
            if s["speaker_label"] not in ("Unknown", "[Multiple Speakers]")
        })
        _analytics["final_segment_count"] = len(final_segments)
        _analytics["speakers_detected_count"] = len(speakers_detected)
        _analytics["overlap_segments_in_final"] = sum(1 for s in final_segments if s.get("is_overlap"))
        logger.info(f"[Pipeline] {recording_id} — Final: {len(final_segments)} segments, speakers={speakers_detected}")

        # ── Phase 1 complete: Save transcript immediately ──────────────
        # Frontend can now show transcript while Qwen3 runs in the background.
        logger.info(f"[Pipeline] {recording_id} — PHASE 1 COMPLETE: Saving transcript to DB")
        try:
            transcript_json = to_json(final_segments)
            speakers_json = to_json(speakers_detected)
            async with get_db() as db:
                await db.execute(
                    text("""
                        UPDATE recordings SET
                            status = 'transcript_ready', progress = 'generating_mom',
                            transcript = :transcript, raw_text = :raw_text, language = :language,
                            speakers_detected = :speakers_detected
                        WHERE id = :recording_id
                    """),
                    {
                        "transcript": transcript_json,
                        "raw_text": raw_text,
                        "language": language,
                        "speakers_detected": speakers_json,
                        "recording_id": recording_id,
                    },
                )
                await db.commit()
            logger.info(f"[Pipeline] {recording_id} — Transcript saved to DB ✓ (status=transcript_ready)")
        except Exception as e:
            logger.error(f"[Pipeline] {recording_id} — CRITICAL: Transcript DB save FAILED: {e}", exc_info=True)
            _analytics["error_stage"] = "transcript_save"
            _analytics["error_message"] = str(e)
            _analytics["total_pipeline_sec"] = round(time.monotonic() - _pipeline_start, 3)
            await _emit_analytics(_analytics)
            await _update_status_safe(recording_id, "error", {"error_message": f"Transcript save failed: {str(e)}"})
            unload_all_models()
            return

    except Exception as e:
        logger.error(f"[Pipeline] {recording_id} — PHASE 1 FAILED: {e}", exc_info=True)
        _analytics["error_stage"] = "phase1"
        _analytics["error_message"] = str(e)
        _analytics["total_pipeline_sec"] = round(time.monotonic() - _pipeline_start, 3)
        await _emit_analytics(_analytics)
        await _update_status_safe(recording_id, "error", {"error_message": f"Pipeline Phase 1 failed: {str(e)}"})
        unload_all_models()
        return

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2: Generate Minutes of Meeting (MoM) — Qwen3 — transcript already saved/visible.
    # AI Insights (summary, key points, action items) are now manual (user-triggered).
    # A failure here does NOT hide the transcript.
    # ─────────────────────────────────────────────────────────────────────

    logger.info(f"[Pipeline] {recording_id} — PHASE 2 START: Generating Minutes of Meeting (Qwen3 4B)")

    _t0_mom = time.monotonic()
    try:
        # Filter out low-confidence segments before feeding to MoM generation
        conf_threshold = settings.MIN_AVG_SEGMENT_CONFIDENCE
        filtered_for_mom = _filter_high_confidence_segments(final_segments, conf_threshold)
        logger.info(
            f"[Pipeline] {recording_id} — MoM input: {len(filtered_for_mom)}/{len(final_segments)} segments "
            f"(confidence >= {conf_threshold})"
        )

        # Build (or reuse cached) context_summary — this is the compressed
        # hierarchical summary used by ALL AI tasks (MoM, insights, PDF).
        # Building it once here means generate_mom skips its own
        # _hierarchical_summarize() call, saving significant LLM inference.
        _t0_ctx = time.monotonic()
        context_summary = await _build_and_store_context_summary(
            recording_id, filtered_for_mom, raw_text, loop
        )
        _analytics["context_summary_sec"] = round(time.monotonic() - _t0_ctx, 3)
        logger.info(
            f"[Pipeline] {recording_id} — Context summary ready "
            f"({len(context_summary.split())} words, took {_analytics['context_summary_sec']:.1f}s)"
        )

        # Fetch recording metadata for MoM
        async with get_db() as db:
            _rec_row = await db.execute(
                text("SELECT filename, created_at, duration FROM recordings WHERE id = :rid"),
                {"rid": recording_id},
            )
            _rec_meta = _rec_row.mappings().fetchone()

        recording_meta = {
            "filename": _rec_meta["filename"] if _rec_meta else "Meeting Notes",
            "created_at": _rec_meta["created_at"] if _rec_meta else "",
            "duration": _rec_meta["duration"] if _rec_meta else 0,
            "speakers_detected": speakers_detected,
        }

        mom_data = await loop.run_in_executor(
            None,
            lambda: generate_mom(
                filtered_for_mom, recording_meta,
                context=context_summary or None,
            )
        )
        logger.info(f"[Pipeline] {recording_id} — MoM generated: {mom_data.get('title', '')}")

        # Upsert MoM into minutes_of_meeting table
        now_mom = datetime.now(timezone.utc)
        mom_id = str(uuid.uuid4())
        initial_version = [{"version": 1, "data": mom_data, "saved_at": dt_to_str(now_mom)}]

        async with get_db() as db:
            _existing = await db.execute(
                text("SELECT id FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
                {"rid": recording_id, "uid": user_id},
            )
            existing_mom = _existing.fetchone()

            if existing_mom:
                await db.execute(
                    text("""
                        UPDATE minutes_of_meeting SET
                            title = :title, date = :date, duration = :duration,
                            planned_start_time = :planned_start_time,
                            actual_start_time = :actual_start_time,
                            participants = :participants,
                            introduction = :introduction,
                            points_discussed = :points_discussed,
                            action_items = :action_items,
                            conclusion = :conclusion,
                            versions = :versions, is_draft = 0, updated_at = :updated_at
                        WHERE recording_id = :rid AND user_id = :uid
                    """),
                    {
                        "title": mom_data.get("title", ""),
                        "date": mom_data.get("date", ""),
                        "duration": mom_data.get("duration", 0),
                        "planned_start_time": mom_data.get("planned_start_time", ""),
                        "actual_start_time": mom_data.get("actual_start_time", ""),
                        "participants": to_json(mom_data.get("participants", [])),
                        "introduction": mom_data.get("introduction", ""),
                        "points_discussed": to_json(mom_data.get("points_discussed", [])),
                        "action_items": to_json(mom_data.get("action_items", [])),
                        "conclusion": mom_data.get("conclusion", ""),
                        "versions": to_json(initial_version),
                        "updated_at": dt_to_str(now_mom),
                        "rid": recording_id,
                        "uid": user_id,
                    },
                )
            else:
                await db.execute(
                    text("""
                        INSERT INTO minutes_of_meeting (
                            id, recording_id, user_id, title, date, duration,
                            planned_start_time, actual_start_time,
                            participants, introduction, points_discussed,
                            action_items, conclusion,
                            versions, is_draft, created_at, updated_at
                        )
                        VALUES (
                            :id, :rid, :uid, :title, :date, :duration,
                            :planned_start_time, :actual_start_time,
                            :participants, :introduction, :points_discussed,
                            :action_items, :conclusion,
                            :versions, 0, :created_at, :updated_at
                        )
                    """),
                    {
                        "id": mom_id,
                        "rid": recording_id,
                        "uid": user_id,
                        "title": mom_data.get("title", ""),
                        "date": mom_data.get("date", ""),
                        "duration": mom_data.get("duration", 0),
                        "planned_start_time": mom_data.get("planned_start_time", ""),
                        "actual_start_time": mom_data.get("actual_start_time", ""),
                        "participants": to_json(mom_data.get("participants", [])),
                        "introduction": mom_data.get("introduction", ""),
                        "points_discussed": to_json(mom_data.get("points_discussed", [])),
                        "action_items": to_json(mom_data.get("action_items", [])),
                        "conclusion": mom_data.get("conclusion", ""),
                        "versions": to_json(initial_version),
                        "created_at": dt_to_str(now_mom),
                        "updated_at": dt_to_str(now_mom),
                    },
                )
            await db.commit()
        logger.info(f"[Pipeline] {recording_id} — MoM saved to DB ✓")

    except Exception as e:
        logger.warning(
            f"[Pipeline] {recording_id} — MoM generation failed (non-fatal, transcript is already saved): {e}",
            exc_info=True
        )
    _analytics["mom_generation_sec"] = round(time.monotonic() - _t0_mom, 3)

    # ── Stage 8: Mark recording done (no AI insight fields saved here) ────
    logger.info(f"[Pipeline] {recording_id} — STAGE 8: Marking recording done")
    try:
        now = datetime.now(timezone.utc)
        async with get_db() as db:
            await db.execute(
                text("""
                    UPDATE recordings SET
                        status = 'done', progress = 'done',
                        processed_at = :processed_at
                    WHERE id = :recording_id
                """),
                {
                    "processed_at": dt_to_str(now),
                    "recording_id": recording_id,
                },
            )
            await db.commit()
        logger.info(f"[Pipeline] {recording_id} — ===== PIPELINE COMPLETE ✓ =====")
    except Exception as e:
        logger.error(f"[Pipeline] {recording_id} — Final DB update FAILED: {e}", exc_info=True)
        try:
            await _update_status_safe(recording_id, "done", {"processed_at": dt_to_str(datetime.now(timezone.utc))})
        except Exception as e2:
            logger.error(f"[Pipeline] {recording_id} — Even fallback done-update FAILED: {e2}")

    # ── Analytics: emit record (always runs, never raises) ─────────────
    _analytics["final_status"] = "done"
    _analytics["total_pipeline_sec"] = round(time.monotonic() - _pipeline_start, 3)
    try:
        async with get_db() as db:
            _dr = await db.execute(
                text("SELECT duration FROM recordings WHERE id = :rid"),
                {"rid": recording_id},
            )
            _dur_row = _dr.mappings().fetchone()
            if _dur_row:
                _analytics["audio_duration_sec"] = _dur_row["duration"]
    except Exception:
        pass
    await _emit_analytics(_analytics)
    unload_all_models()


def _post_process_whisperx_segments(
    wx_result: Dict[str, Any],
    identified_segs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Map WhisperX speaker IDs (SPEAKER_00, SPEAKER_01…) from assign_word_speakers
    back to human-readable labels from our voice-profile identification.
    """
    id_to_label: Dict[str, str] = {}
    id_to_profile: Dict[str, str | None] = {}
    for seg in identified_segs:
        raw_id = seg.get("speaker", "")
        if raw_id and raw_id not in id_to_label:
            id_to_label[raw_id] = seg.get("speaker_label", "Unknown")
            id_to_profile[raw_id] = seg.get("speaker_profile_id")

    out = []
    for seg in wx_result.get("segments", []):
        raw_id = seg.get("speaker", "")
        label = id_to_label.get(raw_id, seg.get("speaker", "Unknown"))
        profile_id = id_to_profile.get(raw_id)

        enriched_words = []
        for w in seg.get("words", []):
            w_raw = w.get("speaker", raw_id)
            enriched_words.append({
                "word": w.get("word", "").strip(),
                "start": round(float(w.get("start", seg["start"])), 3),
                "end": round(float(w.get("end", seg["end"])), 3),
                "probability": round(float(w.get("score", 1.0)), 4),
                "speaker_label": id_to_label.get(w_raw, label),
            })

        seg_start = seg["start"]
        seg_end = seg["end"]
        is_overlap = any(
            s.get("is_overlap", False)
            and max(seg_start, s["start"]) < min(seg_end, s["end"])
            for s in identified_segs
        )
        out.append({
            "start": round(float(seg_start), 3),
            "end": round(float(seg_end), 3),
            "text": seg.get("text", "").strip(),
            "words": enriched_words,
            "avg_logprob": round(float(seg.get("avg_logprob", 0.0)), 4),
            "speaker_label": label,
            "speaker_profile_id": profile_id,
            "is_overlap": is_overlap,
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FINALIZE PIPELINE — merge chunks + full-audio diarization
# ═══════════════════════════════════════════════════════════════════════════════

async def run_finalize_pipeline(
    recording_id: str,
    full_wav_path: str,
    chunk_ids: List[str],
    user_id: str,
    meeting_prompt: str = "",
    participant_voice_ids: List[str] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
):
    try:
        await _run_finalize_pipeline_impl(
            recording_id=recording_id,
            full_wav_path=full_wav_path,
            chunk_ids=chunk_ids,
            user_id=user_id,
            meeting_prompt=meeting_prompt,
            participant_voice_ids=participant_voice_ids,
            use_vocabulary=use_vocabulary,
            speaker_summary=speaker_summary,
        )
    except asyncio.CancelledError:
        logger.info(f"[FinalPipeline] {recording_id} — Task was CANCELLED.")
        try:
            async with get_db() as db:
                await db.execute(
                    text("UPDATE recordings SET status='cancelled', progress=NULL, error_message='Cancelled by user' WHERE id=:rid"),
                    {"rid": recording_id}
                )
                await db.commit()
        except Exception as e:
            logger.error(f"[FinalPipeline] {recording_id} — Failed to update cancelled status in DB: {e}")
        raise
    finally:
        unregister_task(recording_id)
        unload_all_models()


async def _run_finalize_pipeline_impl(
    recording_id: str,
    full_wav_path: str,
    chunk_ids: List[str],
    user_id: str,
    meeting_prompt: str = "",
    participant_voice_ids: List[str] = None,
    use_vocabulary: bool = False,
    speaker_summary: bool = False,
):
    """
    Final merge pipeline for chunked recordings.

    1. Wait for all background chunk jobs to reach 'done' or 'error'.
    2. Merge transcripts from all completed chunks in order.
    3. Run diarization ONCE on the full audio.
    4. Assign speakers from diarization to merged words using timestamps.
    5. Run AI insights on the merged transcript.
    6. Save the unified result.
    """
    participant_voice_ids = participant_voice_ids or []
    logger.info(
        f"[FinalPipeline] ===== START recording={recording_id} "
        f"chunks={chunk_ids} file={full_wav_path} ====="
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error(f"[FinalPipeline] {recording_id} — No running event loop!")
        return

    # ── Analytics accumulators ─────────────────────────────────────────
    _pipeline_start = time.monotonic()
    _chunk_wait_start = time.monotonic()
    _analytics_fin: Dict[str, Any] = {
        "recording_id": recording_id,
        "user_id": user_id,
        "pipeline_type": "chunked",
        "source_type": "chunked",
        "chunk_count": len(chunk_ids),
        "file_size_bytes": _file_size_bytes(full_wav_path),
        "whisper_device": settings.WHISPER_DEVICE,
        "whisper_compute_type": settings.WHISPER_COMPUTE_TYPE,
        "whisper_model_size": settings.WHISPER_MODEL_SIZE,
        "similarity_threshold": settings.SPEAKER_SIMILARITY_THRESHOLD,
        "pyannote_available": is_pyannote_available(),
        "use_vocabulary": use_vocabulary,
        "meeting_prompt_chars": len(meeting_prompt) if meeting_prompt else 0,
        "job_created_at": dt_to_str(datetime.now(timezone.utc)),
        "final_status": "error",
    }

    await _update_status_safe(recording_id, "processing", {"progress": "transcribing"})

    # ── Step 1: Wait for all background chunks to finish ─────────────────
    MAX_WAIT_SEC = 1800  # 30-minute absolute timeout
    POLL_INTERVAL = 3    # poll DB every 3 seconds
    waited = 0

    if chunk_ids:
        logger.info(f"[FinalPipeline] {recording_id} — Waiting for {len(chunk_ids)} background chunks...")
        while waited < MAX_WAIT_SEC:
            async with get_db() as db:
                placeholders = ", ".join([f":id{i}" for i in range(len(chunk_ids))])
                params = {f"id{i}": cid for i, cid in enumerate(chunk_ids)}
                r = await db.execute(
                    text(f"SELECT id, status FROM recording_chunks WHERE id IN ({placeholders})"),
                    params,
                )
                rows = r.mappings().fetchall()

            pending = [row for row in rows if row["status"] == "pending"]
            if not pending:
                break

            logger.info(
                f"[FinalPipeline] {recording_id} — {len(pending)} chunks still pending, "
                f"waiting {POLL_INTERVAL}s (elapsed={waited}s)..."
            )
            await asyncio.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL

        if waited >= MAX_WAIT_SEC:
            logger.warning(f"[FinalPipeline] {recording_id} — Chunk wait timeout after {waited}s; continuing with available results.")
    _analytics_fin["chunk_wait_sec"] = round(time.monotonic() - _chunk_wait_start, 3)

    # ── Step 2: Load all chunk transcripts from DB ───────────────────────
    logger.info(f"[FinalPipeline] {recording_id} — Loading chunk results from DB")
    merged_segments: List[Dict[str, Any]] = []
    merged_raw_parts: List[str] = []
    merged_aligned_segments: List[Dict[str, Any]] = []

    if chunk_ids:
        async with get_db() as db:
            placeholders = ", ".join([f":id{i}" for i in range(len(chunk_ids))])
            params = {f"id{i}": cid for i, cid in enumerate(chunk_ids)}
            r = await db.execute(
                text(
                    f"SELECT * FROM recording_chunks WHERE id IN ({placeholders}) "
                    "ORDER BY chunk_index ASC"
                ),
                params,
            )
            chunk_rows = r.mappings().fetchall()

        for row in chunk_rows:
            if row["status"] != "done":
                logger.warning(f"[FinalPipeline] Chunk {row['id']} has status={row['status']}, skipping.")
                continue
            segs = from_json(row["transcript"], [])
            aligned = from_json(row["aligned_result"], {"segments": segs})
            merged_segments.extend(segs)
            if row["raw_text"]:
                merged_raw_parts.append(row["raw_text"])
            merged_aligned_segments.extend(aligned.get("segments", segs))

        logger.info(
            f"[FinalPipeline] {recording_id} — Merged {len(merged_segments)} segments "
            f"from {len(chunk_rows)} chunks"
        )

    # ── Step 3: Transcribe the final partial chunk (full_wav_path) ───────
    # full_wav_path is the complete recording WAV. For chunked recordings this
    # also covers the tail after the last background chunk was submitted.
    # For simplicity we transcribe the full audio and use it to supplement
    # the merged chunk data; for very long recordings only the tail portion
    # is new — but re-transcribing the full audio ensures we don't miss anything.
    # An optimisation would be to submit only the tail, but full-audio
    # transcription is safer and Whisper caches model state.

    logger.info(f"[FinalPipeline] {recording_id} — Transcribing full audio for final merge")
    await _update_status_safe(recording_id, "processing", {"progress": "transcribing"})

    # Build prompt
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
        _analytics_fin["vocab_term_count"] = len(vocab_words)
        _analytics_fin["initial_prompt_chars"] = len(initial_prompt)
    except Exception as e:
        logger.warning(f"[FinalPipeline] {recording_id} — Prompt build failed (non-fatal): {e}")

    # Fetch pre-detected language from DB (reused from first chunk)
    detected_language = None
    try:
        async with get_db() as db:
            r = await db.execute(
                text("SELECT language FROM recordings WHERE id = :rid"),
                {"rid": recording_id}
            )
            rec_row = r.fetchone()
            if rec_row and rec_row[0]:
                detected_language = rec_row[0]
                logger.info(f"[FinalPipeline] {recording_id} — Reusing pre-detected language: {detected_language}")
    except Exception as e:
        logger.warning(f"[FinalPipeline] {recording_id} — Failed to fetch language from DB: {e}")

    _t0_transcription = time.monotonic()
    try:
        t_result = await loop.run_in_executor(
            None,
            lambda: transcribe(full_wav_path, initial_prompt=initial_prompt, language=detected_language),
        )
        full_raw_text = t_result.get("raw_text", "")
        language = t_result.get("language", "en")
        # Use full-audio aligned result for speaker assignment (most accurate timestamps)
        aligned_result = t_result.get("aligned_result", {"segments": t_result.get("segments", [])})
        _t_segs = t_result.get("segments", [])
        _avg_conf_f, _min_conf_f, _wc_f = _word_confidence_stats(_t_segs)
        _analytics_fin.update({
            "language_detected": language,
            "transcript_segment_count": len(_t_segs),
            "transcript_word_count": _wc_f or _count_total_words(_t_segs),
            "avg_word_confidence": _avg_conf_f,
            "min_word_confidence": _min_conf_f,
            "alignment_used": aligned_result is not t_result,
        })
        logger.info(
            f"[FinalPipeline] {recording_id} — Full audio transcription OK: "
            f"{len(t_result.get('segments', []))} segments, lang={language}"
        )
    except Exception as e:
        logger.error(f"[FinalPipeline] {recording_id} — Full audio transcription FAILED: {e}", exc_info=True)
        if not merged_segments:
            _analytics_fin["error_stage"] = "transcription"
            _analytics_fin["error_message"] = str(e)
            _analytics_fin["total_pipeline_sec"] = round(time.monotonic() - _pipeline_start, 3)
            await _emit_analytics(_analytics_fin)
            await _update_status_safe(recording_id, "error", {"error_message": f"Transcription failed: {str(e)}"})
            unload_all_models()
            return
        # Fall back to merged chunks only
        full_raw_text = " ".join(merged_raw_parts)
        language = "en"
        aligned_result = {"segments": merged_aligned_segments}
    _analytics_fin["transcription_sec"] = round(time.monotonic() - _t0_transcription, 3)

    raw_text = full_raw_text or " ".join(merged_raw_parts)

    # ── Step 4: Diarization (once on full audio) ─────────────────────────
    logger.info(f"[FinalPipeline] {recording_id} — Running diarization on full audio")
    await _update_status_safe(recording_id, "processing", {"progress": "diarizing"})
    _t0_diar = time.monotonic()
    try:
        diar_segs = await loop.run_in_executor(None, diarize, full_wav_path)
        logger.info(f"[FinalPipeline] {recording_id} — Diarization OK: {len(diar_segs)} segments")
    except Exception as e:
        logger.error(f"[FinalPipeline] {recording_id} — Diarization FAILED: {e}", exc_info=True)
        diar_segs = []
    _analytics_fin["diarization_sec"] = round(time.monotonic() - _t0_diar, 3)
    _analytics_fin["diarization_engine"] = "pyannote" if is_pyannote_available() else "energy"
    _analytics_fin["diar_raw_segment_count"] = len(diar_segs)
    _analytics_fin["diar_overlap_segment_count"] = sum(1 for s in diar_segs if s.get("is_overlap"))
    _analytics_fin["diar_unique_speakers"] = len({s["speaker"] for s in diar_segs})

    # ── Step 5: Load voice profiles ──────────────────────────────────────
    logger.info(f"[FinalPipeline] {recording_id} — Loading voice profiles")
    try:
        async with get_db() as db:
            r = await db.execute(
                text("SELECT * FROM voice_profiles WHERE user_id = :uid LIMIT 100"),
                {"uid": user_id},
            )
            raw_profiles = r.mappings().fetchall()
    except Exception as e:
        logger.error(f"[FinalPipeline] {recording_id} — Voice profile load FAILED: {e}", exc_info=True)
        raw_profiles = []

    voice_profiles = []
    for p in raw_profiles:
        profile = dict(p)
        profile["embeddings"] = from_json(p["embeddings"], [])
        voice_profiles.append(profile)
    if participant_voice_ids:
        voice_profiles = [vp for vp in voice_profiles if vp.get("id") in participant_voice_ids]

    # User similarity threshold
    threshold = settings.SPEAKER_SIMILARITY_THRESHOLD
    try:
        async with get_db() as db:
            r = await db.execute(
                text("SELECT * FROM user_settings WHERE user_id = :uid"),
                {"uid": user_id},
            )
            user_settings_row = r.mappings().fetchone()
        if user_settings_row and user_settings_row.get("speaker_similarity_threshold") is not None:
            threshold = float(user_settings_row["speaker_similarity_threshold"])
    except Exception:
        pass

    # ── Step 6: Speaker identification ──────────────────────────────────
    logger.info(f"[FinalPipeline] {recording_id} — Identifying speakers")
    await _update_status_safe(recording_id, "processing", {"progress": "identifying_speakers"})
    _analytics_fin["voice_profiles_loaded"] = len(voice_profiles)
    _analytics_fin["similarity_threshold"] = threshold
    _t0_sid = time.monotonic()
    if diar_segs:
        try:
            identified_segs = await loop.run_in_executor(
                None,
                lambda: identify_speakers(
                    file_path=full_wav_path,
                    diarization_segments=diar_segs,
                    voice_profiles=voice_profiles,
                    similarity_threshold=threshold,
                ),
            )
        except Exception as e:
            logger.error(f"[FinalPipeline] {recording_id} — Speaker ID FAILED: {e}", exc_info=True)
            identified_segs = [
                {**seg, "speaker_label": f"Speaker {i+1}", "speaker_profile_id": None, "similarity": 0.0}
                for i, seg in enumerate(diar_segs)
            ]
    else:
        identified_segs = []
    _analytics_fin["speaker_id_sec"] = round(time.monotonic() - _t0_sid, 3)
    _sims_f = [s.get("similarity", 0.0) for s in identified_segs if not s.get("is_overlap")]
    _ident_f = sum(1 for s in identified_segs if s.get("speaker_profile_id") is not None)
    _analytics_fin["speaker_identified_count"] = _ident_f
    _analytics_fin["speaker_unidentified_count"] = len(identified_segs) - _ident_f
    _analytics_fin["avg_speaker_similarity"] = round(sum(_sims_f) / len(_sims_f), 4) if _sims_f else None

    # ── Step 7: Assign speakers to words ────────────────────────────────
    logger.info(f"[FinalPipeline] {recording_id} — Assigning speakers to words")
    _analytics_fin.setdefault("speaker_assignment_method", "whisperx")
    if identified_segs:
        try:
            import whisperx
            diar_df = _convert_diar_to_whisperx_format(identified_segs)
            if diar_df is not None and not diar_df.empty:
                wx_assigned = whisperx.assign_word_speakers(diar_df, aligned_result)
                speaker_segments = _post_process_whisperx_segments(wx_assigned, identified_segs)
            else:
                raise ValueError("Empty diarization dataframe")
        except Exception as e:
            logger.warning(f"[FinalPipeline] {recording_id} — assign_word_speakers failed ({e}), using manual assignment")
            _analytics_fin["speaker_assignment_method"] = "manual"
            speaker_segments = _assign_speakers_to_words_manual(aligned_result, identified_segs)
    else:
        # No diarization — label all segments as Speaker 1
        from services.identification import merge_transcript_with_speakers
        _analytics_fin["speaker_assignment_method"] = "manual"
        speaker_segments = _assign_speakers_to_words_manual(
            aligned_result,
            [{"start": 0, "end": 999999, "speaker": "SPEAKER_00",
              "speaker_label": "Speaker 1", "speaker_profile_id": None, "is_overlap": False}],
        )

    # ── Step 8: Build final segments ─────────────────────────────────────
    final_segments = []
    for seg in speaker_segments:
        final_segments.append({
            "speaker_label": seg.get("speaker_label", "Unknown"),
            "speaker_profile_id": seg.get("speaker_profile_id"),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "words": seg.get("words", []),
            "is_overlap": seg.get("is_overlap", False),
        })

    speakers_detected = list({
        s["speaker_label"]
        for s in final_segments
        if s["speaker_label"] not in ("Unknown", "[Multiple Speakers]")
    })
    _analytics_fin["final_segment_count"] = len(final_segments)
    _analytics_fin["speakers_detected_count"] = len(speakers_detected)
    _analytics_fin["overlap_segments_in_final"] = sum(1 for s in final_segments if s.get("is_overlap"))
    logger.info(
        f"[FinalPipeline] {recording_id} — Final: {len(final_segments)} segments, "
        f"speakers={speakers_detected}"
    )

    # ── Save transcript immediately (Phase 1 complete) ──────────────────
    try:
        async with get_db() as db:
            await db.execute(
                text("""
                    UPDATE recordings SET
                        status = 'transcript_ready', progress = 'generating_mom',
                        transcript = :transcript, raw_text = :raw_text, language = :language,
                        speakers_detected = :speakers_detected
                    WHERE id = :recording_id
                """),
                {
                    "transcript": to_json(final_segments),
                    "raw_text": raw_text,
                    "language": language,
                    "speakers_detected": to_json(speakers_detected),
                    "recording_id": recording_id,
                },
            )
            await db.commit()
        logger.info(f"[FinalPipeline] {recording_id} — Transcript saved (status=transcript_ready) ✓")
    except Exception as e:
        logger.error(f"[FinalPipeline] {recording_id} — Transcript DB save FAILED: {e}", exc_info=True)
        await _update_status_safe(recording_id, "error", {"error_message": f"Transcript save failed: {str(e)}"})
        unload_all_models()
        return

    # ── Phase 2: Build context_summary from chunk summaries (incremental) ───
    # For chunked recordings, each chunk already has a chunk_summary stored in DB.
    # Merge them here with one final summarization pass — no need to re-read the
    # full transcript.  Falls back to building from final_segments if unavailable.
    logger.info(f"[FinalPipeline] {recording_id} — PHASE 2: Building context_summary")
    # Note: status is already 'transcript_ready' + progress 'generating_mom'.

    _t0_mom_f = time.monotonic()
    context_summary = ""
    try:
        conf_threshold = settings.MIN_AVG_SEGMENT_CONFIDENCE
        filtered_for_mom = _filter_high_confidence_segments(final_segments, conf_threshold)
        logger.info(
            f"[FinalPipeline] {recording_id} — MoM input: {len(filtered_for_mom)}/{len(final_segments)} segments "
            f"(confidence >= {conf_threshold})"
        )

        # Try to merge chunk summaries first (incremental path)
        chunk_summaries: List[str] = []
        try:
            async with get_db() as db:
                cs_rows = await db.execute(
                    text(
                        "SELECT chunk_summary FROM recording_chunks "
                        "WHERE recording_id = :rid AND status = 'done' "
                        "AND chunk_summary IS NOT NULL "
                        "ORDER BY chunk_index ASC"
                    ),
                    {"rid": recording_id},
                )
                chunk_summaries = [
                    r[0] for r in cs_rows.fetchall() if r[0] and r[0].strip()
                ]
        except Exception as e:
            logger.warning(f"[FinalPipeline] {recording_id} — Could not fetch chunk summaries (non-fatal): {e}")

        if chunk_summaries:
            logger.info(
                f"[FinalPipeline] {recording_id} — Merging {len(chunk_summaries)} chunk summaries "
                "(incremental path — skipping full-transcript re-read)"
            )
            merged_chunks = "\n\n".join(chunk_summaries)
            # Single final-pass compression (already short-ish text)
            try:
                from services.ai_provider import get_provider, CHUNK_SUMMARY_PROMPT
                if len(merged_chunks.split()) > 1500:
                    context_summary = await loop.run_in_executor(
                        None,
                        lambda: get_provider()._infer(
                            CHUNK_SUMMARY_PROMPT.format(chunk=merged_chunks[:3000]),
                            max_new_tokens=350,
                        ),
                    )
                    logger.info(
                        f"[FinalPipeline] {recording_id} — Final-pass merge complete "
                        f"({len((context_summary or '').split())} words)"
                    )
                else:
                    context_summary = merged_chunks
            except Exception as e:
                logger.warning(f"[FinalPipeline] {recording_id} — Final-pass merge failed, using concatenated chunks: {e}")
                context_summary = merged_chunks
        else:
            logger.info(
                f"[FinalPipeline] {recording_id} — No chunk summaries found, building context from final segments"
            )
            context_summary = await _build_and_store_context_summary(
                recording_id, filtered_for_mom, raw_text, loop
            )

        # Store context_summary if built via incremental path
        if chunk_summaries and context_summary:
            current_hash = _raw_text_hash(raw_text)
            try:
                async with get_db() as db:
                    await db.execute(
                        text(
                            "UPDATE recordings SET "
                            "context_summary = :ctx, context_summary_hash = :h "
                            "WHERE id = :rid"
                        ),
                        {"ctx": context_summary, "h": current_hash, "rid": recording_id},
                    )
                    await db.commit()
                logger.info(
                    f"[FinalPipeline] {recording_id} — context_summary stored "
                    f"({len(context_summary.split())} words) ✓"
                )
            except Exception as e:
                logger.warning(f"[FinalPipeline] {recording_id} — Failed to store context_summary: {e}")

        async with get_db() as db:
            _rec_row = await db.execute(
                text("SELECT filename, created_at, duration FROM recordings WHERE id = :rid"),
                {"rid": recording_id},
            )
            _rec_meta = _rec_row.mappings().fetchone()

        recording_meta = {
            "filename": _rec_meta["filename"] if _rec_meta else "Meeting Notes",
            "created_at": _rec_meta["created_at"] if _rec_meta else "",
            "duration": _rec_meta["duration"] if _rec_meta else 0,
            "speakers_detected": speakers_detected,
        }

        mom_data = await loop.run_in_executor(
            None,
            lambda: generate_mom(
                filtered_for_mom, recording_meta,
                context=context_summary or None,
            )
        )
        logger.info(f"[FinalPipeline] {recording_id} — MoM generated: {mom_data.get('title', '')}")

        now_mom = datetime.now(timezone.utc)
        mom_id = str(uuid.uuid4())
        initial_version = [{"version": 1, "data": mom_data, "saved_at": dt_to_str(now_mom)}]

        async with get_db() as db:
            _existing = await db.execute(
                text("SELECT id FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
                {"rid": recording_id, "uid": user_id},
            )
            existing_mom = _existing.fetchone()

            if existing_mom:
                await db.execute(
                    text("""
                        UPDATE minutes_of_meeting SET
                            title = :title, date = :date, duration = :duration,
                            planned_start_time = :planned_start_time,
                            actual_start_time = :actual_start_time,
                            participants = :participants,
                            introduction = :introduction,
                            points_discussed = :points_discussed,
                            action_items = :action_items,
                            conclusion = :conclusion,
                            versions = :versions, is_draft = 0, updated_at = :updated_at
                        WHERE recording_id = :rid AND user_id = :uid
                    """),
                    {
                        "title": mom_data.get("title", ""),
                        "date": mom_data.get("date", ""),
                        "duration": mom_data.get("duration", 0),
                        "planned_start_time": mom_data.get("planned_start_time", ""),
                        "actual_start_time": mom_data.get("actual_start_time", ""),
                        "participants": to_json(mom_data.get("participants", [])),
                        "introduction": mom_data.get("introduction", ""),
                        "points_discussed": to_json(mom_data.get("points_discussed", [])),
                        "action_items": to_json(mom_data.get("action_items", [])),
                        "conclusion": mom_data.get("conclusion", ""),
                        "versions": to_json(initial_version),
                        "updated_at": dt_to_str(now_mom),
                        "rid": recording_id,
                        "uid": user_id,
                    },
                )
            else:
                await db.execute(
                    text("""
                        INSERT INTO minutes_of_meeting (
                            id, recording_id, user_id, title, date, duration,
                            planned_start_time, actual_start_time,
                            participants, introduction, points_discussed,
                            action_items, conclusion,
                            versions, is_draft, created_at, updated_at
                        )
                        VALUES (
                            :id, :rid, :uid, :title, :date, :duration,
                            :planned_start_time, :actual_start_time,
                            :participants, :introduction, :points_discussed,
                            :action_items, :conclusion,
                            :versions, 0, :created_at, :updated_at
                        )
                    """),
                    {
                        "id": mom_id,
                        "rid": recording_id,
                        "uid": user_id,
                        "title": mom_data.get("title", ""),
                        "date": mom_data.get("date", ""),
                        "duration": mom_data.get("duration", 0),
                        "planned_start_time": mom_data.get("planned_start_time", ""),
                        "actual_start_time": mom_data.get("actual_start_time", ""),
                        "participants": to_json(mom_data.get("participants", [])),
                        "introduction": mom_data.get("introduction", ""),
                        "points_discussed": to_json(mom_data.get("points_discussed", [])),
                        "action_items": to_json(mom_data.get("action_items", [])),
                        "conclusion": mom_data.get("conclusion", ""),
                        "versions": to_json(initial_version),
                        "created_at": dt_to_str(now_mom),
                        "updated_at": dt_to_str(now_mom),
                    },
                )
            await db.commit()
        logger.info(f"[FinalPipeline] {recording_id} — MoM saved to DB ✓")

    except Exception as e:
        logger.warning(f"[FinalPipeline] {recording_id} — MoM generation failed (non-fatal): {e}", exc_info=True)
    _analytics_fin["mom_generation_sec"] = round(time.monotonic() - _t0_mom_f, 3)

    # ── Save final result (no AI insight fields) ─────────────────────────
    now = datetime.now(timezone.utc)
    try:
        async with get_db() as db:
            await db.execute(
                text("""
                    UPDATE recordings SET
                        status = 'done', progress = 'done',
                        processed_at = :processed_at
                    WHERE id = :recording_id
                """),
                {
                    "processed_at": dt_to_str(now),
                    "recording_id": recording_id,
                },
            )
            await db.commit()
        logger.info(f"[FinalPipeline] {recording_id} — ===== FINALIZE PIPELINE COMPLETE ✓ =====")
    except Exception as e:
        logger.error(f"[FinalPipeline] {recording_id} — Final DB update FAILED: {e}", exc_info=True)
        try:
            await _update_status_safe(recording_id, "done", {"processed_at": dt_to_str(now)})
        except Exception:
            pass

    # ── Analytics: emit record (always runs, never raises) ─────────────
    _analytics_fin["final_status"] = "done"
    _analytics_fin["total_pipeline_sec"] = round(time.monotonic() - _pipeline_start, 3)
    try:
        async with get_db() as db:
            _dr = await db.execute(
                text("SELECT duration FROM recordings WHERE id = :rid"),
                {"rid": recording_id},
            )
            _dur_row = _dr.mappings().fetchone()
            if _dur_row:
                _analytics_fin["audio_duration_sec"] = _dur_row["duration"]
    except Exception:
        pass
    await _emit_analytics(_analytics_fin)
    unload_all_models()

