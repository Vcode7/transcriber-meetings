"""
Background pipeline: transcribe → diarize → identify speakers → generate AI insights → save.

WhisperX integration:
  - transcribe() returns an `aligned_result` dict consumed by whisperx.assign_word_speakers()
  - assign_word_speakers() annotates every word with the speaker from pyannote diarization
  - We then run our voice-profile identification on top to map pyannote IDs → human names
"""
import json
import logging
import asyncio
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Any
from sqlalchemy import text

from database import get_db, dt_to_str, to_json, from_json
from services.transcription import transcribe
from services.diarization import diarize
from services.identification import identify_speakers
from services.llm import generate_summary, generate_key_points, generate_action_items, generate_short_summary, generate_detailed_summary, generate_speaker_summaries
from services.prompt_builder import build_whisper_prompt
from services.dictionary_service import get_global_prompt, list_vocabulary
from config import settings

logger = logging.getLogger(__name__)


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
    """
    Full async pipeline for a single recording.

    Phase 1 (fast): Transcription → Diarization → Speaker ID → save transcript
                    Sets status = 'transcript_ready' so frontend can show transcript immediately.
    Phase 2 (slow): Qwen3 AI insights → set status = 'done'
    """
    participant_voice_ids = participant_voice_ids or []
    logger.info(f"[Pipeline] ===== START recording_id={recording_id} file={file_path} =====")

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
    except Exception as e:
        logger.warning(f"[Pipeline] {recording_id} — Prompt build failed (non-fatal): {e}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1: Transcription → Diarization → Speaker ID → Save transcript
    # ─────────────────────────────────────────────────────────────────────

    try:
        # ── Stage 1: Transcription (WhisperX + alignment) ─────────────
        logger.info(f"[Pipeline] {recording_id} — STAGE 1: Transcribing {file_path}")
        await _update_status_safe(recording_id, "processing", {"progress": "transcribing"})

        try:
            t_result = await loop.run_in_executor(
                None,
                lambda: transcribe(file_path, initial_prompt=initial_prompt),
            )
        except Exception as e:
            logger.error(f"[Pipeline] {recording_id} — Transcription FAILED: {e}", exc_info=True)
            await _update_status_safe(recording_id, "error", {"error_message": f"Transcription failed: {str(e)}"})
            return

        transcript_segs = t_result["segments"]
        raw_text = t_result["raw_text"]
        language = t_result.get("language", "en")
        aligned_result = t_result.get("aligned_result", {"segments": transcript_segs})
        logger.info(f"[Pipeline] {recording_id} — Transcription OK: {len(transcript_segs)} segments, lang={language}")

        # ── Stage 2: Diarization ──────────────────────────────────────
        logger.info(f"[Pipeline] {recording_id} — STAGE 2: Diarizing")
        await _update_status_safe(recording_id, "processing", {"progress": "diarizing"})
        diar_segs = await loop.run_in_executor(None, diarize, file_path)
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

        logger.info(f"[Pipeline] {recording_id} — Speaker ID OK: {len(identified_segs)} identified segments")

        # ── Stage 5: Word-level speaker assignment (WhisperX-native) ──
        logger.info(f"[Pipeline] {recording_id} — STAGE 5: Word→speaker assignment")
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
            speaker_segments = _assign_speakers_to_words_manual(aligned_result, identified_segs)

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
                            status = 'transcript_ready', progress = 'generating_insights',
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
            await _update_status_safe(recording_id, "error", {"error_message": f"Transcript save failed: {str(e)}"})
            return

    except Exception as e:
        logger.error(f"[Pipeline] {recording_id} — PHASE 1 FAILED: {e}", exc_info=True)
        await _update_status_safe(recording_id, "error", {"error_message": f"Pipeline Phase 1 failed: {str(e)}"})
        return

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2: AI insights (slow Qwen3 — transcript already saved/visible)
    # A failure here does NOT hide the transcript.
    # ─────────────────────────────────────────────────────────────────────

    logger.info(f"[Pipeline] {recording_id} — PHASE 2 START: Generating AI insights (Qwen3 4B)")

    summary = ""
    short_summary = ""
    detailed_summary = ""
    key_points: List[str] = []
    action_items: List[str] = []

    try:
        logger.info(f"[Pipeline] {recording_id} — Generating short summary")
        short_summary = await loop.run_in_executor(None, generate_short_summary, final_segments)
        logger.info(f"[Pipeline] {recording_id} — Short summary done ({len(short_summary)} chars)")

        logger.info(f"[Pipeline] {recording_id} — Generating detailed summary")
        detailed_summary = await loop.run_in_executor(None, generate_detailed_summary, final_segments)
        logger.info(f"[Pipeline] {recording_id} — Detailed summary done ({len(detailed_summary)} chars)")

        summary = short_summary  # backward compat

        logger.info(f"[Pipeline] {recording_id} — Generating key points")
        key_points = await loop.run_in_executor(None, generate_key_points, final_segments)
        logger.info(f"[Pipeline] {recording_id} — Key points done ({len(key_points)} items)")

        logger.info(f"[Pipeline] {recording_id} — Generating action items")
        action_items = await loop.run_in_executor(None, generate_action_items, final_segments)
        logger.info(f"[Pipeline] {recording_id} — Action items done ({len(action_items)} items)")

    except Exception as e:
        logger.warning(
            f"[Pipeline] {recording_id} — AI insights failed (non-fatal, transcript is already saved): {e}",
            exc_info=True
        )
        # Don't return here — still save done status with whatever we have

    # ── Phase 2b (optional): Per-speaker summaries ──────────────
    speaker_summary_data: dict = {}
    if speaker_summary:
        logger.info(f"[Pipeline] {recording_id} — PHASE 2b: Generating per-speaker summaries")
        try:
            speaker_summary_data = await loop.run_in_executor(
                None, generate_speaker_summaries, final_segments
            )
            logger.info(
                f"[Pipeline] {recording_id} — Speaker summaries done for {len(speaker_summary_data)} speakers"
            )
        except Exception as e:
            logger.warning(
                f"[Pipeline] {recording_id} — Speaker summaries failed (non-fatal): {e}",
                exc_info=True
            )

    # ── Stage 8: Persist AI results → mark done ───────────────────────
    logger.info(f"[Pipeline] {recording_id} — STAGE 8: Persisting AI results and marking done")
    try:
        now = datetime.now(timezone.utc)
        async with get_db() as db:
            await db.execute(
                text("""
                    UPDATE recordings SET
                        status = 'done', progress = 'done',
                        summary = :summary, short_summary = :short_summary,
                        detailed_summary = :detailed_summary,
                        key_points = :key_points, action_items = :action_items,
                        speaker_summary = :speaker_summary,
                        processed_at = :processed_at
                    WHERE id = :recording_id
                """),
                {
                    "summary": summary,
                    "short_summary": short_summary,
                    "detailed_summary": detailed_summary,
                    "key_points": to_json(key_points),
                    "action_items": to_json(action_items),
                    "speaker_summary": to_json(speaker_summary_data) if speaker_summary_data else None,
                    "processed_at": dt_to_str(now),
                    "recording_id": recording_id,
                },
            )
            await db.commit()
        logger.info(f"[Pipeline] {recording_id} — ===== PIPELINE COMPLETE ✓ =====")
    except Exception as e:
        logger.error(f"[Pipeline] {recording_id} — Final DB update FAILED: {e}", exc_info=True)
        # Try at minimum to set status=done so frontend stops waiting
        try:
            await _update_status_safe(recording_id, "done", {"processed_at": dt_to_str(datetime.now(timezone.utc))})
        except Exception as e2:
            logger.error(f"[Pipeline] {recording_id} — Even fallback done-update FAILED: {e2}")


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
