"""Settings router — get/update user threshold settings."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text

from database import get_db, get_db_context, dt_to_str
from routers.auth import get_current_user
from pydantic import BaseModel
from models.settings import UserSettingsUpdate
from services.action_importance import clamp_min_importance

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


def normalize_ollama_url(url: str) -> str:
    """
    Validate and normalize Ollama Server URL:
    - Trim whitespace
    - Ensure protocol (http:// or https://)
    - Remove trailing slash(es)
    - Ensure default port 11434 if no port is specified on localhost or IP address
    """
    import urllib.parse
    import re

    if not url:
        return "http://localhost:11434"
    u = url.strip()
    if not u.startswith("http://") and not u.startswith("https://"):
        u = f"http://{u}"
    u = u.rstrip("/")
    try:
        parsed = urllib.parse.urlparse(u)
        if not parsed.port and (parsed.hostname in ("localhost", "127.0.0.1") or (parsed.hostname and re.match(r"^\d+\.\d+\.\d+\.\d+$", parsed.hostname))):
            u = f"{u}:11434"
    except Exception:
        pass
    return u


class TestOllamaRequest(BaseModel):
    server_url: str | None = None


@router.get("")
async def get_settings(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT * FROM user_settings WHERE user_id = :uid"),
            {"uid": user_id},
        )
        doc = r.mappings().fetchone()

    defaults = {
        "speaker_similarity_threshold": 0.75,
        "word_conf_low": 0.7,
        "word_conf_mid": 0.85,
        "min_segment_duration": 1.5,
        "use_ollama": False,
        "ollama_server_url": "http://localhost:11434",
        "ollama_port": 11434,
        "ollama_model_priority": "gemma,qwen,llama,deepseek,mistral",
        "rag_chunk_size": 400,
        "rag_chunk_overlap": 50,
        "rag_retrieval_k_global": 2,
        "rag_retrieval_k_meeting": 3,
        "rag_retrieval_k_transcript": 10,
        "rag_max_collection_context": 10,
        "rag_relative_score_cutoff": 0.01,
        "generate_mom_auto": True,
        "embedding_model": "Qwen3-Embedding-0.6B",
        "speaker_embedding_model": "ecapa",
        "restrict_reassignment_to_meeting_speakers": False,
        "ollama_num_ctx": 32768,
        "ollama_dynamic_ctx": True,
        "ollama_think": False,
        "ollama_temperature": 0.0,
        "ollama_top_p": 0.9,
        "ollama_top_k": 40,
        "ollama_repeat_penalty": 1.15,
        "ollama_seed": -1,
        "ollama_stop": "",
        "ollama_keep_alive": "5m",
        "ollama_num_thread": 0,
        "ollama_num_gpu": -1,
        "max_tokens_mom": 1500,
        "max_tokens_mom_merge": 3072,
        "max_tokens_agenda_compress": 2000,
        "max_tokens_reference_compress": 2000,
        "max_tokens_agenda_from_summary": 1024,
        "max_tokens_executive_summary": 700,
        "max_tokens_short_summary": 120,
        "max_tokens_detailed_summary": 3000,
        "max_tokens_chunk_summary": 256,
        "max_tokens_key_points": 1028,
        "max_tokens_action_items": 1028,
        "max_tokens_key_decisions": 1028,
        "max_tokens_speaker_summary": 200,
        "max_tokens_speaker_key_points": 350,
        "max_tokens_speaker_action_items": 250,
        "max_tokens_collection_chat": 1500,
        "max_tokens_collection_compare": 1500,
        "max_tokens_collection_topic_growth": 1500,
        "max_tokens_vocab_extractor": 512,
        "enable_vad": True,
        "enable_transcription_vad": True,
        "enable_alignment_vad": True,
        "enable_audio_normalization": True,
        "norm_target_dbfs": -3.0,
        "norm_compression_ratio": 2.0,
        "enable_adaptive_vad": True,
        "vad_speech_threshold": 0.15,
        "vad_silence_threshold": 0.10,
        "vad_min_speech_ms": 250,
        "vad_min_silence_ms": 400,
        "enable_speech_padding": True,
        "speech_pad_ms": 400,
        "enable_speech_segment_merging": True,
        "max_merge_silence_ms": 500,
        "enable_low_volume_recovery": True,
        "recovery_energy_threshold": -45.0,
        "recovery_min_duration_ms": 300,
        "enable_audio_validation": True,
        "min_audio_duration_seconds": 2.0,
        "min_audio_rms_threshold": 0.003,
        "whisper_batch_size": 8,
        "whisper_parallel_processing": 1,
        "whisper_parallel_chunk_minutes": 10,
        "parallel_transcription_diarization": False,
        "rom_parallel_window_processing": 2,

        "rom_separate_action_extraction": False,
        "rom_action_owner_verification": False,
        "rom_action_generation_chunk_size": 10,
        "action_point_min_importance": 0,
        "rom_stage2_process_all_together": False,
        "rom_pipeline_mode": "base",
        "rom_short_model_mode": "base",
        "rom_short_model_variant_id": None,
        "missing_transcript_recovery_enabled": False,
        "missing_segment_min_duration_sec": 2.0,
        "max_tokens_rom_discussion": 4096,
        "max_tokens_rom_discussion_no_actions": 4096,
        "max_tokens_rom_action_extraction": 2048,
        "max_tokens_mom_extract_actions": 4096,
        "max_tokens_stage1_json_repair": 4548,
        "max_tokens_mom_action_regen": 4048,
        "max_tokens_rom_polish": 4096,
        "max_tokens_rom_enhance_window": 4096,
        "max_tokens_rom_deduplicate": 2048,
        "max_tokens_rom_agenda": 2048,
        "max_tokens_rom_mom_expansion": 3000,
        "max_tokens_rom_agenda_assign_batch": 4096,
        "max_tokens_rom_agenda_doc_points": 1024,
        "max_tokens_rom_version_short": 4096,
        "max_tokens_rom_version_medium": 4096,
        "reasoning_mode_short_rom": False,
        "reasoning_mode_action_points": False,
    }

    if not doc:
        return defaults

    res = dict(doc)
    for k, v in defaults.items():
        if res.get(k) is None:
            res[k] = v

    res["use_ollama"] = bool(res["use_ollama"])
    res["generate_mom_auto"] = bool(res["generate_mom_auto"])
    res["ollama_dynamic_ctx"] = bool(res["ollama_dynamic_ctx"])
    res["enable_vad"] = bool(res["enable_vad"])
    res["enable_transcription_vad"] = bool(res["enable_transcription_vad"])
    res["enable_alignment_vad"] = bool(res["enable_alignment_vad"])
    res["enable_audio_normalization"] = bool(res["enable_audio_normalization"])
    res["enable_adaptive_vad"] = bool(res["enable_adaptive_vad"])
    res["enable_speech_padding"] = bool(res["enable_speech_padding"])
    res["enable_speech_segment_merging"] = bool(res["enable_speech_segment_merging"])
    res["enable_low_volume_recovery"] = bool(res["enable_low_volume_recovery"])
    res["enable_audio_validation"] = bool(res["enable_audio_validation"])
    res["rom_separate_action_extraction"] = bool(res.get("rom_separate_action_extraction", 0))
    res["rom_action_owner_verification"] = bool(res.get("rom_action_owner_verification", 0))
    res["rom_action_generation_chunk_size"] = int(res.get("rom_action_generation_chunk_size") or 10)
    res["action_point_min_importance"] = clamp_min_importance(res.get("action_point_min_importance"))
    res["rom_stage2_process_all_together"] = bool(res.get("rom_stage2_process_all_together", 0))
    res["rom_pipeline_mode"] = str(res.get("rom_pipeline_mode") or "base")
    res["missing_transcript_recovery_enabled"] = bool(res.get("missing_transcript_recovery_enabled", 0))
    res["missing_segment_min_duration_sec"] = float(res.get("missing_segment_min_duration_sec") or 2.0)
    res["restrict_reassignment_to_meeting_speakers"] = bool(res.get("restrict_reassignment_to_meeting_speakers", 0))
    res["reasoning_mode_short_rom"] = bool(res.get("reasoning_mode_short_rom", 0))
    res["reasoning_mode_action_points"] = bool(res.get("reasoning_mode_action_points", 0))
    if res.get("overlap_separation_enabled") is None:
        from config import settings as _cfg
        res["overlap_separation_enabled"] = bool(getattr(_cfg, "OVERLAP_SEPARATION_ENABLED", False))
    else:
        res["overlap_separation_enabled"] = bool(res["overlap_separation_enabled"])
    from services.overlap_separation import is_separation_available
    res["overlap_separation_available"] = is_separation_available()

    if res.get("embedding_model"):
        # Keep the runtime model in sync with this user's selection — through the
        # validated setter, which also unloads the cached embedder on change.
        from services.text_embedding_service import (
            EmbeddingModelLockedError,
            get_active_embedding_model,
            set_active_embedding_model,
        )
        try:
            set_active_embedding_model(res["embedding_model"], source="settings load")
        except EmbeddingModelLockedError:
            pass  # a reindex pins the model; it will be applied on the next load
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"[Settings] Persisted embedding model '{res['embedding_model']}' unavailable: {e}")
        res["active_embedding_model"] = get_active_embedding_model()

    # Sync speaker embedding model with embedding router
    if res.get("speaker_embedding_model"):
        try:
            from services.embedding_router import set_active_model
            set_active_model(res["speaker_embedding_model"])
        except Exception:
            pass

    return res


@router.put("")
async def update_settings(
    body: UserSettingsUpdate,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]
    now = datetime.now(timezone.utc)
    patch = {k: v for k, v in body.model_dump().items() if v is not None}

    if not patch:
        return {"message": "Nothing to update."}

    # Normalize ollama_server_url if present
    if "ollama_server_url" in patch:
        patch["ollama_server_url"] = normalize_ollama_url(patch["ollama_server_url"])

    # Convert bool to int for sqlite
    if "use_ollama" in patch:
        patch["use_ollama"] = 1 if patch["use_ollama"] else 0
    if "generate_mom_auto" in patch:
        patch["generate_mom_auto"] = 1 if patch["generate_mom_auto"] else 0
    if "ollama_dynamic_ctx" in patch:
        patch["ollama_dynamic_ctx"] = 1 if patch["ollama_dynamic_ctx"] else 0
    if "ollama_think" in patch:
        patch["ollama_think"] = 1 if patch["ollama_think"] else 0
    if "enable_vad" in patch:
        patch["enable_vad"] = 1 if patch["enable_vad"] else 0
    if "enable_transcription_vad" in patch:
        patch["enable_transcription_vad"] = 1 if patch["enable_transcription_vad"] else 0
    if "enable_alignment_vad" in patch:
        patch["enable_alignment_vad"] = 1 if patch["enable_alignment_vad"] else 0
    if "enable_audio_normalization" in patch:
        patch["enable_audio_normalization"] = 1 if patch["enable_audio_normalization"] else 0
    if "enable_adaptive_vad" in patch:
        patch["enable_adaptive_vad"] = 1 if patch["enable_adaptive_vad"] else 0
    if "enable_speech_padding" in patch:
        patch["enable_speech_padding"] = 1 if patch["enable_speech_padding"] else 0
    if "enable_speech_segment_merging" in patch:
        patch["enable_speech_segment_merging"] = 1 if patch["enable_speech_segment_merging"] else 0
    if "enable_low_volume_recovery" in patch:
        patch["enable_low_volume_recovery"] = 1 if patch["enable_low_volume_recovery"] else 0
    if "enable_audio_validation" in patch:
        patch["enable_audio_validation"] = 1 if patch["enable_audio_validation"] else 0
    if "rom_separate_action_extraction" in patch:
        patch["rom_separate_action_extraction"] = 1 if patch["rom_separate_action_extraction"] else 0
    if "rom_action_owner_verification" in patch:
        patch["rom_action_owner_verification"] = 1 if patch["rom_action_owner_verification"] else 0
    if "reasoning_mode_short_rom" in patch:
        patch["reasoning_mode_short_rom"] = 1 if patch["reasoning_mode_short_rom"] else 0
    if "reasoning_mode_action_points" in patch:
        patch["reasoning_mode_action_points"] = 1 if patch["reasoning_mode_action_points"] else 0
    if "rom_action_generation_chunk_size" in patch:
        try:
            val = int(patch["rom_action_generation_chunk_size"])
            patch["rom_action_generation_chunk_size"] = max(1, min(val, 100))
        except (ValueError, TypeError):
            patch["rom_action_generation_chunk_size"] = 10
    if "action_point_min_importance" in patch:
        patch["action_point_min_importance"] = clamp_min_importance(patch["action_point_min_importance"])
    if "rom_stage2_process_all_together" in patch:
        patch["rom_stage2_process_all_together"] = 1 if patch["rom_stage2_process_all_together"] else 0
    if "missing_transcript_recovery_enabled" in patch:
        patch["missing_transcript_recovery_enabled"] = 1 if patch["missing_transcript_recovery_enabled"] else 0
    if "overlap_separation_enabled" in patch:
        patch["overlap_separation_enabled"] = 1 if patch["overlap_separation_enabled"] else 0
    if "restrict_reassignment_to_meeting_speakers" in patch:
        patch["restrict_reassignment_to_meeting_speakers"] = 1 if patch["restrict_reassignment_to_meeting_speakers"] else 0
    if "missing_segment_min_duration_sec" in patch:
        try:
            val = float(patch["missing_segment_min_duration_sec"])
            if val <= 0:
                raise ValueError("Must be positive")
            patch["missing_segment_min_duration_sec"] = round(val, 2)
        except (ValueError, TypeError):
            patch["missing_segment_min_duration_sec"] = 2.0

    # Sync embedding model setting with runtime config & unload existing text embedder if changed
    embedding_model_changed = False
    if "embedding_model" in patch and patch["embedding_model"]:
        new_model = str(patch["embedding_model"]).strip()
        patch["embedding_model"] = new_model
        from services.text_embedding_service import EmbeddingModelLockedError, set_active_embedding_model
        try:
            embedding_model_changed = set_active_embedding_model(new_model, source=f"settings update by {user_id}")
        except EmbeddingModelLockedError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except (FileNotFoundError, ValueError) as e:
            async with get_db_context() as db:
                r = await db.execute(
                    text("SELECT embedding_model FROM user_settings WHERE user_id = :uid"),
                    {"uid": user_id},
                )
                row = r.fetchone()
            if row and row[0] == new_model:
                # Unchanged value whose files are currently missing: don't block saving other settings.
                logger.warning(f"[Settings] Embedding model '{new_model}' unavailable, left unchanged: {e}")
            else:
                raise HTTPException(status_code=400, detail=f"Embedding model '{new_model}' is not available: {e}")

    # Validate and sync speaker_embedding_model setting
    if "speaker_embedding_model" in patch and patch["speaker_embedding_model"]:
        val = str(patch["speaker_embedding_model"]).strip().lower()
        if val not in ("ecapa", "eres2net_large"):
            val = "ecapa"  # fallback to safe default
        patch["speaker_embedding_model"] = val
        try:
            from services.embedding_router import set_active_model
            set_active_model(val)
        except Exception:
            pass

    # Build SET clause dynamically from provided fields
    set_parts = [f"{k} = :{k}" for k in patch]
    set_parts.append("updated_at = :updated_at")
    set_clause = ", ".join(set_parts)

    params = {**patch, "user_id": user_id, "updated_at": dt_to_str(now)}

    async with get_db_context() as db:
        # Try update first
        r = await db.execute(
            text(f"UPDATE user_settings SET {set_clause} WHERE user_id = :user_id"),
            params,
        )
        if r.rowcount == 0:
            # Row doesn't exist yet — insert with defaults
            await db.execute(
                text("""
                    INSERT INTO user_settings (user_id, speaker_similarity_threshold, word_conf_low,
                        word_conf_mid, min_segment_duration, use_ollama, ollama_server_url, ollama_port, ollama_model_priority,
                        rag_chunk_size, rag_chunk_overlap, rag_retrieval_k_global, rag_retrieval_k_meeting,
                        rag_retrieval_k_transcript, rag_max_collection_context, rag_relative_score_cutoff, generate_mom_auto, embedding_model,
                        ollama_num_ctx, ollama_dynamic_ctx, ollama_think, ollama_temperature, ollama_top_p, ollama_top_k, ollama_repeat_penalty,
                        ollama_seed, ollama_stop, ollama_keep_alive, ollama_num_thread, ollama_num_gpu,
                        max_tokens_mom, max_tokens_mom_merge,
                        max_tokens_agenda_compress, max_tokens_reference_compress, max_tokens_agenda_from_summary,
                        max_tokens_executive_summary, max_tokens_short_summary, max_tokens_detailed_summary, max_tokens_chunk_summary,
                        max_tokens_key_points, max_tokens_action_items, max_tokens_key_decisions,
                        max_tokens_speaker_summary, max_tokens_speaker_key_points, max_tokens_speaker_action_items,
                        max_tokens_collection_chat, max_tokens_collection_compare, max_tokens_collection_topic_growth, max_tokens_vocab_extractor,
                        restrict_reassignment_to_meeting_speakers, rom_short_model_mode, rom_short_model_variant_id,
                        max_tokens_rom_version_short, max_tokens_rom_version_medium,
                        reasoning_mode_short_rom, reasoning_mode_action_points, action_point_min_importance,
                        updated_at)
                    VALUES (:user_id, :threshold, :low, :mid, :min_dur, :use_ollama, :ollama_server_url, :ollama_port, :ollama_model_priority,
                        :rag_chunk_size, :rag_chunk_overlap, :rag_retrieval_k_global, :rag_retrieval_k_meeting,
                        :rag_retrieval_k_transcript, :rag_max_collection_context, :rag_relative_score_cutoff, :generate_mom_auto, :embedding_model,
                        :ollama_num_ctx, :ollama_dynamic_ctx, :ollama_think, :ollama_temperature, :ollama_top_p, :ollama_top_k, :ollama_repeat_penalty,
                        :ollama_seed, :ollama_stop, :ollama_keep_alive, :ollama_num_thread, :ollama_num_gpu,
                        :max_tokens_mom, :max_tokens_mom_merge,
                        :max_tokens_agenda_compress, :max_tokens_reference_compress, :max_tokens_agenda_from_summary,
                        :max_tokens_executive_summary, :max_tokens_short_summary, :max_tokens_detailed_summary, :max_tokens_chunk_summary,
                        :max_tokens_key_points, :max_tokens_action_items, :max_tokens_key_decisions,
                        :max_tokens_speaker_summary, :max_tokens_speaker_key_points, :max_tokens_speaker_action_items,
                        :max_tokens_collection_chat, :max_tokens_collection_compare, :max_tokens_collection_topic_growth, :max_tokens_vocab_extractor,
                        :restrict_reassignment_to_meeting_speakers, :rom_short_model_mode, :rom_short_model_variant_id,
                        :max_tokens_rom_version_short, :max_tokens_rom_version_medium,
                        :reasoning_mode_short_rom, :reasoning_mode_action_points, :action_point_min_importance,
                        :updated_at)
                """),
                {
                    "user_id": user_id,
                    "threshold": patch.get("speaker_similarity_threshold", 0.75),
                    "low": patch.get("word_conf_low", 0.7),
                    "mid": patch.get("word_conf_mid", 0.85),
                    "min_dur": patch.get("min_segment_duration", 1.5),
                    "use_ollama": patch.get("use_ollama", 0),
                    "ollama_server_url": patch.get("ollama_server_url", "http://localhost:11434"),
                    "ollama_port": patch.get("ollama_port", 11434),
                    "ollama_model_priority": patch.get("ollama_model_priority", "gemma,qwen,llama,deepseek,mistral"),
                    "rag_chunk_size": patch.get("rag_chunk_size", 400),
                    "rag_chunk_overlap": patch.get("rag_chunk_overlap", 50),
                    "rag_retrieval_k_global": patch.get("rag_retrieval_k_global", 2),
                    "rag_retrieval_k_meeting": patch.get("rag_retrieval_k_meeting", 3),
                    "rag_retrieval_k_transcript": patch.get("rag_retrieval_k_transcript", 10),
                    "rag_max_collection_context": patch.get("rag_max_collection_context", 10),
                    "rag_relative_score_cutoff": patch.get("rag_relative_score_cutoff", 0.01),
                    "generate_mom_auto": patch.get("generate_mom_auto", 1),
                    "embedding_model": patch.get("embedding_model", "Qwen3-Embedding-0.6B"),
                    "ollama_num_ctx": patch.get("ollama_num_ctx", 32768),
                    "ollama_dynamic_ctx": patch.get("ollama_dynamic_ctx", 1),
                    "ollama_think": patch.get("ollama_think", 0),
                    "ollama_temperature": patch.get("ollama_temperature", 0.0),
                    "ollama_top_p": patch.get("ollama_top_p", 0.9),
                    "ollama_top_k": patch.get("ollama_top_k", 40),
                    "ollama_repeat_penalty": patch.get("ollama_repeat_penalty", 1.15),
                    "ollama_seed": patch.get("ollama_seed", -1),
                    "ollama_stop": patch.get("ollama_stop", ""),
                    "ollama_keep_alive": patch.get("ollama_keep_alive", "5m"),
                    "ollama_num_thread": patch.get("ollama_num_thread", 0),
                    "ollama_num_gpu": patch.get("ollama_num_gpu", -1),
                    "max_tokens_mom": patch.get("max_tokens_mom", 1500),
                    "max_tokens_mom_merge": patch.get("max_tokens_mom_merge", 3072),
                    "max_tokens_agenda_compress": patch.get("max_tokens_agenda_compress", 2000),
                    "max_tokens_reference_compress": patch.get("max_tokens_reference_compress", 2000),
                    "max_tokens_agenda_from_summary": patch.get("max_tokens_agenda_from_summary", 1024),
                    "max_tokens_executive_summary": patch.get("max_tokens_executive_summary", 700),
                    "max_tokens_short_summary": patch.get("max_tokens_short_summary", 120),
                    "max_tokens_detailed_summary": patch.get("max_tokens_detailed_summary", 3000),
                    "max_tokens_chunk_summary": patch.get("max_tokens_chunk_summary", 256),
                    "max_tokens_key_points": patch.get("max_tokens_key_points", 1028),
                    "max_tokens_action_items": patch.get("max_tokens_action_items", 1028),
                    "max_tokens_key_decisions": patch.get("max_tokens_key_decisions", 1028),
                    "max_tokens_speaker_summary": patch.get("max_tokens_speaker_summary", 200),
                    "max_tokens_speaker_key_points": patch.get("max_tokens_speaker_key_points", 350),
                    "max_tokens_speaker_action_items": patch.get("max_tokens_speaker_action_items", 250),
                    "max_tokens_collection_chat": patch.get("max_tokens_collection_chat", 1500),
                    "max_tokens_collection_compare": patch.get("max_tokens_collection_compare", 1500),
                    "max_tokens_collection_topic_growth": patch.get("max_tokens_collection_topic_growth", 1500),
                    "max_tokens_vocab_extractor": patch.get("max_tokens_vocab_extractor", 512),
                    "restrict_reassignment_to_meeting_speakers": patch.get("restrict_reassignment_to_meeting_speakers", 0),
                    "rom_short_model_mode": patch.get("rom_short_model_mode", "base"),
                    "rom_short_model_variant_id": patch.get("rom_short_model_variant_id", None),
                    "max_tokens_rom_version_short": patch.get("max_tokens_rom_version_short", 4096),
                    "max_tokens_rom_version_medium": patch.get("max_tokens_rom_version_medium", 4096),
                    "reasoning_mode_short_rom": patch.get("reasoning_mode_short_rom", 0),
                    "reasoning_mode_action_points": patch.get("reasoning_mode_action_points", 0),
                    "action_point_min_importance": patch.get("action_point_min_importance", 0),
                    "updated_at": dt_to_str(now),
                },
            )
        await db.commit()

    if embedding_model_changed:
        from services.embedding_training.variant_manager import sync_variant_active_flags
        try:
            await sync_variant_active_flags(patch["embedding_model"])
        except Exception as e:
            logger.warning(f"[Settings] Could not sync embedding variant flags: {e}")

    # Convert back to bool for response representation
    if "use_ollama" in patch:
        patch["use_ollama"] = bool(patch["use_ollama"])
    if "generate_mom_auto" in patch:
        patch["generate_mom_auto"] = bool(patch["generate_mom_auto"])
    if "restrict_reassignment_to_meeting_speakers" in patch:
        patch["restrict_reassignment_to_meeting_speakers"] = bool(patch["restrict_reassignment_to_meeting_speakers"])
    if "reasoning_mode_short_rom" in patch:
        patch["reasoning_mode_short_rom"] = bool(patch["reasoning_mode_short_rom"])
    if "reasoning_mode_action_points" in patch:
        patch["reasoning_mode_action_points"] = bool(patch["reasoning_mode_action_points"])
    if "overlap_separation_enabled" in patch:
        patch["overlap_separation_enabled"] = bool(patch["overlap_separation_enabled"])

    response = {"message": "Settings updated.", **patch}
    if embedding_model_changed:
        response["embedding_model_changed"] = True
    return response



@router.post("/test-ollama")
async def test_ollama_connection(
    body: TestOllamaRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Test connectivity to the configured Ollama server.
    Returns connectivity status, error message (if unreachable), and available/running models.
    """
    url_to_test = body.server_url
    if not url_to_test:
        user_id = current_user["id"]
        async with get_db_context() as db:
            r = await db.execute(
                text("SELECT ollama_server_url FROM user_settings WHERE user_id = :uid"),
                {"uid": user_id},
            )
            row = r.mappings().fetchone()
            if row and row.get("ollama_server_url"):
                url_to_test = row["ollama_server_url"]
            else:
                url_to_test = "http://localhost:11434"

    server_url = normalize_ollama_url(url_to_test)

    import urllib.request
    import json

    running_models = []
    available_models = []

    # 1. Fetch available models from /api/tags
    try:
        req_tags = urllib.request.Request(f"{server_url}/api/tags")
        with urllib.request.urlopen(req_tags, timeout=5.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                available_models = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
    except Exception as e:
        return {
            "success": False,
            "server_url": server_url,
            "error": f"Could not connect to Ollama server at {server_url}: {str(e)}",
            "message": "Connection failed.",
        }

    # 2. Fetch running models from /api/ps (optional)
    try:
        req_ps = urllib.request.Request(f"{server_url}/api/ps")
        with urllib.request.urlopen(req_ps, timeout=3.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                running_models = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
    except Exception:
        pass

    return {
        "success": True,
        "server_url": server_url,
        "available_models": available_models,
        "running_models": running_models,
        "message": f"Successfully connected to Ollama server at {server_url}.",
    }


@router.get("/embedding-models")
async def get_embedding_models(current_user: dict = Depends(get_current_user)):
    """
    List supported text embedding models along with their local installation status.
    """
    from pathlib import Path
    from config import settings, BASE_DIR, RUNTIME_DIR

    known_models = [
        {
            "id": "Qwen3-Embedding-0.6B",
            "name": "Qwen3-Embedding-0.6B",
            "description": "Fast & lightweight 0.6B model (Recommended for CPU & low VRAM)",
            "dim": 1024,
            "quantization": "float16 / float32",
        },
        {
            "id": "Qwen3-Embedding-4B-Instruct-INT8",
            "name": "Qwen3-Embedding-4B-Instruct-INT8",
            "description": "High-performance 4B Instruct model (INT8 Quantized for GPU)",
            "dim": 2560,
            "quantization": "int8",
        },
        {
            "id": "Qwen3-Embedding-4B-Instruct",
            "name": "Qwen3-Embedding-4B-Instruct",
            "description": "Full FP16 4B Instruct model (Maximum retrieval accuracy)",
            "dim": 2560,
            "quantization": "float16",
        },
    ]

    search_dirs = [
        BASE_DIR.parent / "Application" / "runtime" / "embeddings",
        RUNTIME_DIR / "embeddings",
    ]

    from services.text_embedding_service import validate_embedding_model_dir

    # Only directories that actually contain a loadable model are "installed".
    installed_map = {}
    for d in search_dirs:
        if d.exists() and d.is_dir():
            for child in d.iterdir():
                if child.name.startswith((".", "_")):
                    continue  # hidden/in-progress directories (e.g. a variant being saved)
                if child.name not in installed_map and child.is_dir() and not validate_embedding_model_dir(child):
                    installed_map[child.name] = child

    def _model_details(path: Path) -> dict:
        details = {"dim": None, "is_trained_variant": False, "base_model": None}
        try:
            with open(path / "config.json", "r", encoding="utf-8") as f:
                details["dim"] = json.load(f).get("hidden_size")
        except Exception:
            pass
        meta_path = path / "variant_meta.json"
        if meta_path.is_file():
            details["is_trained_variant"] = True
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    details["base_model"] = json.load(f).get("base_model")
            except Exception:
                pass
        return details

    models_res = []
    seen_ids = set()

    for km in known_models:
        mid = km["id"]
        seen_ids.add(mid)
        installed = mid in installed_map
        models_res.append({
            **km,
            "is_trained_variant": False,
            "base_model": None,
            "installed": installed,
            "path": str(installed_map[mid]) if installed else None,
        })

    for name, path in installed_map.items():
        if name not in seen_ids:
            details = _model_details(path)
            if details["is_trained_variant"]:
                description = f"Trained variant of {details['base_model'] or 'a base model'}"
            else:
                description = "Custom local embedding model"
            models_res.append({
                "id": name,
                "name": name,
                "description": description,
                "quantization": "unknown",
                "installed": True,
                "path": str(path),
                **details,
            })

    active = settings.EMBEDDING_MODEL
    active_entry = next((m for m in models_res if m["id"] == active), None)
    return {
        "active_model": active,
        "active_model_is_trained_variant": bool(active_entry and active_entry.get("is_trained_variant")),
        "models": models_res,
    }


# ── ChromaDB reindex with the active (trained) embedding model ────────────────

class ReindexRequest(BaseModel):
    force: bool = False


def _active_model_info() -> dict:
    from config import settings
    from services.text_embedding_service import get_active_embedding_model, validate_embedding_model_dir

    name = get_active_embedding_model()
    path = Path(settings.embedding_model_dir(name))
    info = {
        "name": name,
        "available": validate_embedding_model_dir(path) is None,
        "is_trained_variant": False,
        "base_model": None,
    }
    meta_path = path / "variant_meta.json"
    if meta_path.is_file():
        info["is_trained_variant"] = True
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                info["base_model"] = json.load(f).get("base_model")
        except Exception:
            pass
    return info


@router.get("/embedding-index/status")
async def get_embedding_index_status(current_user: dict = Depends(get_current_user)):
    """Which embedding model every ChromaDB collection was embedded with vs. the active model."""
    from services.embedding_reindex import get_index_status, get_reindex_status
    from services.text_embedding_service import get_embedding_model_switch_lock

    try:
        status = await run_in_threadpool(get_index_status)
    except Exception as e:
        logger.exception("[Settings] Failed to read ChromaDB index status")
        raise HTTPException(status_code=500, detail=f"Could not read ChromaDB index status: {e}")
    return {
        **status,
        "active_model_info": _active_model_info(),
        "model_switch_locked": get_embedding_model_switch_lock(),
        "reindex_job": get_reindex_status(),
    }


@router.post("/embedding-index/reindex")
async def start_embedding_reindex(body: ReindexRequest, current_user: dict = Depends(get_current_user)):
    """
    Re-embed all existing ChromaDB data with the active embedding model (normally the
    selected trained variant). Runs in the background; poll /embedding-index/reindex/status.
    """
    from services.embedding_reindex import ReindexAlreadyRunningError, start_reindex_job

    try:
        return start_reindex_job(force=body.force, requested_by=current_user["id"])
    except ReindexAlreadyRunningError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"Active embedding model is not available: {e}")


@router.get("/embedding-index/reindex/status")
async def get_embedding_reindex_status(current_user: dict = Depends(get_current_user)):
    from services.embedding_reindex import get_reindex_status

    return get_reindex_status() or {"status": "idle"}


@router.post("/embedding-index/reindex/cancel")
async def cancel_embedding_reindex(current_user: dict = Depends(get_current_user)):
    from services.embedding_reindex import cancel_reindex

    if not cancel_reindex():
        raise HTTPException(status_code=409, detail="No ChromaDB reindex is running.")
    return {"status": "cancelling"}



