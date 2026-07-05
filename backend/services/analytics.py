"""
Processing Analytics Service — one record per completed job.

Design goals:
  - Fully independent of the main pipeline: any exception here is swallowed.
  - Collected at the end of run_pipeline / run_finalize_pipeline.
  - Exposed via the public GET /analysis endpoint (no auth required).
  - One SQLite row per job with all metrics stored as flat columns + JSON blobs.
"""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from database import get_db, dt_to_str

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# DB table definition (called from database.py connect_db)
# ─────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS processing_analytics (
    id                          TEXT PRIMARY KEY,
    recording_id                TEXT NOT NULL,
    user_id                     TEXT NOT NULL,

    -- Pipeline variant & source
    pipeline_type               TEXT NOT NULL DEFAULT 'single',
    source_type                 TEXT NOT NULL DEFAULT 'upload',

    -- Audio characteristics
    audio_duration_sec          REAL,
    file_size_bytes             INTEGER,
    language_detected           TEXT,

    -- Final outcome
    final_status                TEXT NOT NULL,
    error_stage                 TEXT,
    error_message               TEXT,

    -- End-to-end timing (seconds, float)
    total_pipeline_sec          REAL,
    transcription_sec           REAL,
    alignment_sec               REAL,
    diarization_sec             REAL,
    speaker_id_sec              REAL,
    ai_insights_sec             REAL,
    speaker_summary_sec         REAL,

    -- Transcription quality
    transcript_segment_count    INTEGER,
    transcript_word_count       INTEGER,
    avg_word_confidence         REAL,
    min_word_confidence         REAL,
    alignment_used              INTEGER NOT NULL DEFAULT 0,

    -- Diarization
    diarization_engine          TEXT,
    diar_raw_segment_count      INTEGER,
    diar_overlap_segment_count  INTEGER,
    diar_unique_speakers        INTEGER,

    -- Speaker identification
    voice_profiles_loaded       INTEGER,
    speaker_identified_count    INTEGER,
    speaker_unidentified_count  INTEGER,
    avg_speaker_similarity      REAL,
    speaker_assignment_method   TEXT,

    -- AI insights output
    short_summary_chars         INTEGER,
    detailed_summary_chars      INTEGER,
    key_points_count            INTEGER,
    action_items_count          INTEGER,
    speaker_summaries_generated INTEGER NOT NULL DEFAULT 0,

    -- Final content
    final_segment_count         INTEGER,
    speakers_detected_count     INTEGER,
    overlap_segments_in_final   INTEGER,

    -- Chunked recording extras
    chunk_count                 INTEGER,
    chunk_wait_sec              REAL,

    -- Prompt / vocabulary
    use_vocabulary              INTEGER NOT NULL DEFAULT 0,
    vocab_term_count            INTEGER,
    initial_prompt_chars        INTEGER,
    meeting_prompt_chars        INTEGER,

    -- Runtime configuration (snapshot at job time)
    whisper_device              TEXT,
    whisper_compute_type        TEXT,
    whisper_model_size          TEXT,
    similarity_threshold        REAL,
    pyannote_available          INTEGER NOT NULL DEFAULT 0,

    -- Timestamps
    created_at                  TEXT NOT NULL,
    completed_at                TEXT NOT NULL
)
"""

ADD_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_analytics_recording_id
    ON processing_analytics(recording_id)
"""

ADD_INDEX_COMPLETED_SQL = """
CREATE INDEX IF NOT EXISTS idx_analytics_completed_at
    ON processing_analytics(completed_at)
"""


# ─────────────────────────────────────────────────────────────
# Public write API (called by pipeline.py — never raises)
# ─────────────────────────────────────────────────────────────

async def record_pipeline_analytics(metrics: Dict[str, Any]) -> None:
    """
    Persist one analytics row for a completed pipeline job.

    ``metrics`` is a dict built by pipeline.py containing all known values.
    Any missing key defaults to None / 0 safely.
    Only raises on truly unrecoverable DB errors — but even those are caught
    by the caller (run_pipeline) which wraps this in its own try/except.
    """
    try:
        row_id = str(uuid.uuid4())
        now_str = dt_to_str(datetime.now(timezone.utc))

        def _i(key: str, default=None):
            """Get integer metric safely."""
            val = metrics.get(key, default)
            if val is None:
                return None
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        def _f(key: str, default=None):
            """Get float metric safely."""
            val = metrics.get(key, default)
            if val is None:
                return None
            try:
                return round(float(val), 4)
            except (TypeError, ValueError):
                return default

        def _s(key: str, default=None):
            """Get string metric safely."""
            val = metrics.get(key, default)
            return str(val) if val is not None else default

        async with get_db() as db:
            await db.execute(
                text("""
                    INSERT INTO processing_analytics (
                        id, recording_id, user_id,
                        pipeline_type, source_type,
                        audio_duration_sec, file_size_bytes, language_detected,
                        final_status, error_stage, error_message,
                        total_pipeline_sec, transcription_sec, alignment_sec,
                        diarization_sec, speaker_id_sec, ai_insights_sec, speaker_summary_sec,
                        transcript_segment_count, transcript_word_count,
                        avg_word_confidence, min_word_confidence, alignment_used,
                        diarization_engine, diar_raw_segment_count,
                        diar_overlap_segment_count, diar_unique_speakers,
                        voice_profiles_loaded, speaker_identified_count,
                        speaker_unidentified_count, avg_speaker_similarity,
                        speaker_assignment_method,
                        short_summary_chars, detailed_summary_chars,
                        key_points_count, action_items_count, speaker_summaries_generated,
                        final_segment_count, speakers_detected_count, overlap_segments_in_final,
                        chunk_count, chunk_wait_sec,
                        use_vocabulary, vocab_term_count, initial_prompt_chars, meeting_prompt_chars,
                        whisper_device, whisper_compute_type, whisper_model_size,
                        similarity_threshold, pyannote_available,
                        created_at, completed_at
                    ) VALUES (
                        :id, :recording_id, :user_id,
                        :pipeline_type, :source_type,
                        :audio_duration_sec, :file_size_bytes, :language_detected,
                        :final_status, :error_stage, :error_message,
                        :total_pipeline_sec, :transcription_sec, :alignment_sec,
                        :diarization_sec, :speaker_id_sec, :ai_insights_sec, :speaker_summary_sec,
                        :transcript_segment_count, :transcript_word_count,
                        :avg_word_confidence, :min_word_confidence, :alignment_used,
                        :diarization_engine, :diar_raw_segment_count,
                        :diar_overlap_segment_count, :diar_unique_speakers,
                        :voice_profiles_loaded, :speaker_identified_count,
                        :speaker_unidentified_count, :avg_speaker_similarity,
                        :speaker_assignment_method,
                        :short_summary_chars, :detailed_summary_chars,
                        :key_points_count, :action_items_count, :speaker_summaries_generated,
                        :final_segment_count, :speakers_detected_count, :overlap_segments_in_final,
                        :chunk_count, :chunk_wait_sec,
                        :use_vocabulary, :vocab_term_count, :initial_prompt_chars, :meeting_prompt_chars,
                        :whisper_device, :whisper_compute_type, :whisper_model_size,
                        :similarity_threshold, :pyannote_available,
                        :created_at, :completed_at
                    )
                """),
                {
                    "id": row_id,
                    "recording_id": _s("recording_id"),
                    "user_id": _s("user_id"),
                    "pipeline_type": _s("pipeline_type", "single"),
                    "source_type": _s("source_type", "upload"),
                    "audio_duration_sec": _f("audio_duration_sec"),
                    "file_size_bytes": _i("file_size_bytes"),
                    "language_detected": _s("language_detected"),
                    "final_status": _s("final_status", "unknown"),
                    "error_stage": _s("error_stage"),
                    "error_message": _s("error_message"),
                    "total_pipeline_sec": _f("total_pipeline_sec"),
                    "transcription_sec": _f("transcription_sec"),
                    "alignment_sec": _f("alignment_sec"),
                    "diarization_sec": _f("diarization_sec"),
                    "speaker_id_sec": _f("speaker_id_sec"),
                    "ai_insights_sec": _f("ai_insights_sec"),
                    "speaker_summary_sec": _f("speaker_summary_sec"),
                    "transcript_segment_count": _i("transcript_segment_count"),
                    "transcript_word_count": _i("transcript_word_count"),
                    "avg_word_confidence": _f("avg_word_confidence"),
                    "min_word_confidence": _f("min_word_confidence"),
                    "alignment_used": 1 if metrics.get("alignment_used") else 0,
                    "diarization_engine": _s("diarization_engine"),
                    "diar_raw_segment_count": _i("diar_raw_segment_count"),
                    "diar_overlap_segment_count": _i("diar_overlap_segment_count"),
                    "diar_unique_speakers": _i("diar_unique_speakers"),
                    "voice_profiles_loaded": _i("voice_profiles_loaded"),
                    "speaker_identified_count": _i("speaker_identified_count"),
                    "speaker_unidentified_count": _i("speaker_unidentified_count"),
                    "avg_speaker_similarity": _f("avg_speaker_similarity"),
                    "speaker_assignment_method": _s("speaker_assignment_method"),
                    "short_summary_chars": _i("short_summary_chars"),
                    "detailed_summary_chars": _i("detailed_summary_chars"),
                    "key_points_count": _i("key_points_count"),
                    "action_items_count": _i("action_items_count"),
                    "speaker_summaries_generated": 1 if metrics.get("speaker_summaries_generated") else 0,
                    "final_segment_count": _i("final_segment_count"),
                    "speakers_detected_count": _i("speakers_detected_count"),
                    "overlap_segments_in_final": _i("overlap_segments_in_final"),
                    "chunk_count": _i("chunk_count"),
                    "chunk_wait_sec": _f("chunk_wait_sec"),
                    "use_vocabulary": 1 if metrics.get("use_vocabulary") else 0,
                    "vocab_term_count": _i("vocab_term_count"),
                    "initial_prompt_chars": _i("initial_prompt_chars"),
                    "meeting_prompt_chars": _i("meeting_prompt_chars"),
                    "whisper_device": _s("whisper_device"),
                    "whisper_compute_type": _s("whisper_compute_type"),
                    "whisper_model_size": _s("whisper_model_size"),
                    "similarity_threshold": _f("similarity_threshold"),
                    "pyannote_available": 1 if metrics.get("pyannote_available") else 0,
                    "created_at": _s("job_created_at", now_str),
                    "completed_at": now_str,
                },
            )
            await db.commit()

        logger.info(
            f"[Analytics] Recorded analytics row {row_id} for recording={metrics.get('recording_id')} "
            f"status={metrics.get('final_status')}"
        )

    except Exception as exc:
        # Analytics must never crash the pipeline
        logger.warning(f"[Analytics] Failed to persist analytics (non-fatal): {exc}", exc_info=True)


# ─────────────────────────────────────────────────────────────
# Public read API (used by GET /analysis router)
# ─────────────────────────────────────────────────────────────

async def fetch_all_analytics() -> List[Dict[str, Any]]:
    """
    Return all analytics records ordered by completed_at DESC.
    Each record is returned as a structured dict with nested groups.
    """
    try:
        async with get_db() as db:
            result = await db.execute(
                text("SELECT * FROM processing_analytics ORDER BY completed_at DESC LIMIT 1000")
            )
            rows = result.mappings().fetchall()
    except Exception as exc:
        logger.error(f"[Analytics] Failed to fetch analytics: {exc}", exc_info=True)
        return []

    records = []
    for row in rows:
        r = dict(row)
        record = {
            "id": r.get("id"),
            "recording_id": r.get("recording_id"),
            "user_id": r.get("user_id"),
            "pipeline_type": r.get("pipeline_type"),
            "source_type": r.get("source_type"),
            "created_at": r.get("created_at"),
            "completed_at": r.get("completed_at"),
            # ── Grouped sub-objects for dashboard readability ──
            "audio": {
                "duration_sec": r.get("audio_duration_sec"),
                "file_size_bytes": r.get("file_size_bytes"),
                "language": r.get("language_detected"),
            },
            "outcome": {
                "status": r.get("final_status"),
                "error_stage": r.get("error_stage"),
                "error_message": r.get("error_message"),
            },
            "timing": {
                "total_sec": r.get("total_pipeline_sec"),
                "transcription_sec": r.get("transcription_sec"),
                "alignment_sec": r.get("alignment_sec"),
                "diarization_sec": r.get("diarization_sec"),
                "speaker_id_sec": r.get("speaker_id_sec"),
                "ai_insights_sec": r.get("ai_insights_sec"),
                "speaker_summary_sec": r.get("speaker_summary_sec"),
            },
            "transcription": {
                "segment_count": r.get("transcript_segment_count"),
                "word_count": r.get("transcript_word_count"),
                "avg_word_confidence": r.get("avg_word_confidence"),
                "min_word_confidence": r.get("min_word_confidence"),
                "alignment_used": bool(r.get("alignment_used")),
            },
            "diarization": {
                "engine": r.get("diarization_engine"),
                "raw_segment_count": r.get("diar_raw_segment_count"),
                "overlap_segment_count": r.get("diar_overlap_segment_count"),
                "unique_speakers": r.get("diar_unique_speakers"),
            },
            "speaker_identification": {
                "voice_profiles_loaded": r.get("voice_profiles_loaded"),
                "identified_count": r.get("speaker_identified_count"),
                "unidentified_count": r.get("speaker_unidentified_count"),
                "avg_similarity": r.get("avg_speaker_similarity"),
                "assignment_method": r.get("speaker_assignment_method"),
            },
            "ai_insights": {
                "short_summary_chars": r.get("short_summary_chars"),
                "detailed_summary_chars": r.get("detailed_summary_chars"),
                "key_points_count": r.get("key_points_count"),
                "action_items_count": r.get("action_items_count"),
                "speaker_summaries_generated": bool(r.get("speaker_summaries_generated")),
            },
            "content": {
                "final_segment_count": r.get("final_segment_count"),
                "speakers_detected_count": r.get("speakers_detected_count"),
                "overlap_segments_in_final": r.get("overlap_segments_in_final"),
            },
            "chunking": {
                "chunk_count": r.get("chunk_count"),
                "chunk_wait_sec": r.get("chunk_wait_sec"),
            },
            "vocabulary": {
                "use_vocabulary": bool(r.get("use_vocabulary")),
                "vocab_term_count": r.get("vocab_term_count"),
                "initial_prompt_chars": r.get("initial_prompt_chars"),
                "meeting_prompt_chars": r.get("meeting_prompt_chars"),
            },
            "config": {
                "whisper_device": r.get("whisper_device"),
                "whisper_compute_type": r.get("whisper_compute_type"),
                "whisper_model_size": r.get("whisper_model_size"),
                "similarity_threshold": r.get("similarity_threshold"),
                "pyannote_available": bool(r.get("pyannote_available")),
            },
        }
        records.append(record)

    return records


# ─────────────────────────────────────────────────────────────
# Aggregated summary stats for /analysis?summary=true
# ─────────────────────────────────────────────────────────────

async def fetch_analytics_summary() -> Dict[str, Any]:
    """
    Return aggregate stats across all analytics rows.
    Useful for a quick product health dashboard.
    """
    try:
        async with get_db() as db:
            result = await db.execute(text("""
                SELECT
                    COUNT(*) AS total_jobs,
                    SUM(CASE WHEN final_status = 'done' THEN 1 ELSE 0 END) AS successful_jobs,
                    SUM(CASE WHEN final_status = 'error' THEN 1 ELSE 0 END) AS failed_jobs,
                    AVG(audio_duration_sec)   AS avg_audio_duration_sec,
                    MAX(audio_duration_sec)   AS max_audio_duration_sec,
                    AVG(total_pipeline_sec)   AS avg_total_pipeline_sec,
                    AVG(transcription_sec)    AS avg_transcription_sec,
                    AVG(diarization_sec)      AS avg_diarization_sec,
                    AVG(speaker_id_sec)       AS avg_speaker_id_sec,
                    AVG(ai_insights_sec)      AS avg_ai_insights_sec,
                    AVG(transcript_word_count) AS avg_word_count,
                    AVG(avg_word_confidence)  AS avg_word_confidence,
                    SUM(CASE WHEN alignment_used = 1 THEN 1 ELSE 0 END) AS jobs_with_alignment,
                    SUM(CASE WHEN pyannote_available = 1 THEN 1 ELSE 0 END) AS jobs_with_pyannote,
                    SUM(CASE WHEN use_vocabulary = 1 THEN 1 ELSE 0 END) AS jobs_with_vocabulary,
                    SUM(CASE WHEN pipeline_type = 'chunked' THEN 1 ELSE 0 END) AS chunked_jobs,
                    AVG(speakers_detected_count) AS avg_speakers_per_meeting,
                    AVG(key_points_count)     AS avg_key_points,
                    AVG(action_items_count)   AS avg_action_items,
                    MIN(completed_at)         AS first_job_at,
                    MAX(completed_at)         AS last_job_at
                FROM processing_analytics
            """))
            row = result.mappings().fetchone()

        if not row:
            return {}

        r = dict(row)

        def _round(val, ndigits=2):
            if val is None:
                return None
            try:
                return round(float(val), ndigits)
            except (TypeError, ValueError):
                return None

        total = r.get("total_jobs") or 0
        success = r.get("successful_jobs") or 0

        return {
            "total_jobs": total,
            "successful_jobs": success,
            "failed_jobs": r.get("failed_jobs") or 0,
            "success_rate_pct": _round((success / total * 100) if total > 0 else None),
            "first_job_at": r.get("first_job_at"),
            "last_job_at": r.get("last_job_at"),
            "audio": {
                "avg_duration_sec": _round(r.get("avg_audio_duration_sec")),
                "max_duration_sec": _round(r.get("max_audio_duration_sec")),
            },
            "performance": {
                "avg_total_pipeline_sec": _round(r.get("avg_total_pipeline_sec")),
                "avg_transcription_sec": _round(r.get("avg_transcription_sec")),
                "avg_diarization_sec": _round(r.get("avg_diarization_sec")),
                "avg_speaker_id_sec": _round(r.get("avg_speaker_id_sec")),
                "avg_ai_insights_sec": _round(r.get("avg_ai_insights_sec")),
            },
            "quality": {
                "avg_word_count": _round(r.get("avg_word_count")),
                "avg_word_confidence": _round(r.get("avg_word_confidence"), 4),
                "avg_speakers_per_meeting": _round(r.get("avg_speakers_per_meeting")),
                "avg_key_points": _round(r.get("avg_key_points")),
                "avg_action_items": _round(r.get("avg_action_items")),
            },
            "features": {
                "jobs_with_alignment": r.get("jobs_with_alignment") or 0,
                "jobs_with_pyannote": r.get("jobs_with_pyannote") or 0,
                "jobs_with_vocabulary": r.get("jobs_with_vocabulary") or 0,
                "chunked_jobs": r.get("chunked_jobs") or 0,
            },
        }

    except Exception as exc:
        logger.error(f"[Analytics] Failed to compute summary: {exc}", exc_info=True)
        return {}
