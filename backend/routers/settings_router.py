"""Settings router — get/update user threshold settings."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text

from database import get_db, dt_to_str
from routers.auth import get_current_user
from pydantic import BaseModel

from database import get_db, dt_to_str
from routers.auth import get_current_user
from models.settings import UserSettingsUpdate

router = APIRouter(prefix="/settings", tags=["settings"])


def normalize_ollama_url(url: str) -> str:
    """
    Validate and normalize Ollama Server URL:
    - Trim whitespace
    - Ensure protocol (http:// or https://)
    - Remove trailing slash(es)
    """
    if not url:
        return "http://localhost:11434"
    u = url.strip()
    if not u.startswith("http://") and not u.startswith("https://"):
        u = f"http://{u}"
    return u.rstrip("/")


class TestOllamaRequest(BaseModel):
    server_url: str | None = None


@router.get("")
async def get_settings(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db() as db:
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
        "ollama_num_ctx": 32768,
        "ollama_dynamic_ctx": True,
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
        "max_tokens_raw_mom_to_mom": 3000,
        "max_tokens_raw_mom_extraction": 1024,
        "max_tokens_raw_mom_repair": 1024,
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

    if res.get("embedding_model"):
        from config import settings
        if settings.EMBEDDING_MODEL != res["embedding_model"]:
            settings.EMBEDDING_MODEL = res["embedding_model"]
            settings.QWEN_EMBEDDING_MODEL_NAME = res["embedding_model"]

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

    # Sync embedding model setting with runtime config & unload existing text embedder if changed
    if "embedding_model" in patch and patch["embedding_model"]:
        new_model = str(patch["embedding_model"]).strip()
        if new_model:
            from config import settings
            if settings.EMBEDDING_MODEL != new_model:
                settings.EMBEDDING_MODEL = new_model
                settings.QWEN_EMBEDDING_MODEL_NAME = new_model
                try:
                    from services.text_embedding_service import unload_text_embedder
                    unload_text_embedder()
                except Exception:
                    pass

    # Build SET clause dynamically from provided fields
    set_parts = [f"{k} = :{k}" for k in patch]
    set_parts.append("updated_at = :updated_at")
    set_clause = ", ".join(set_parts)

    params = {**patch, "user_id": user_id, "updated_at": dt_to_str(now)}

    async with get_db() as db:
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
                        ollama_num_ctx, ollama_dynamic_ctx, ollama_temperature, ollama_top_p, ollama_top_k, ollama_repeat_penalty,
                        ollama_seed, ollama_stop, ollama_keep_alive, ollama_num_thread, ollama_num_gpu,
                        max_tokens_mom, max_tokens_mom_merge, max_tokens_raw_mom_to_mom, max_tokens_raw_mom_extraction, max_tokens_raw_mom_repair,
                        max_tokens_agenda_compress, max_tokens_reference_compress, max_tokens_agenda_from_summary,
                        max_tokens_executive_summary, max_tokens_short_summary, max_tokens_detailed_summary, max_tokens_chunk_summary,
                        max_tokens_key_points, max_tokens_action_items, max_tokens_key_decisions,
                        max_tokens_speaker_summary, max_tokens_speaker_key_points, max_tokens_speaker_action_items,
                        max_tokens_collection_chat, max_tokens_collection_compare, max_tokens_collection_topic_growth, max_tokens_vocab_extractor,
                        updated_at)
                    VALUES (:user_id, :threshold, :low, :mid, :min_dur, :use_ollama, :ollama_server_url, :ollama_port, :ollama_model_priority,
                        :rag_chunk_size, :rag_chunk_overlap, :rag_retrieval_k_global, :rag_retrieval_k_meeting,
                        :rag_retrieval_k_transcript, :rag_max_collection_context, :rag_relative_score_cutoff, :generate_mom_auto, :embedding_model,
                        :ollama_num_ctx, :ollama_dynamic_ctx, :ollama_temperature, :ollama_top_p, :ollama_top_k, :ollama_repeat_penalty,
                        :ollama_seed, :ollama_stop, :ollama_keep_alive, :ollama_num_thread, :ollama_num_gpu,
                        :max_tokens_mom, :max_tokens_mom_merge, :max_tokens_raw_mom_to_mom, :max_tokens_raw_mom_extraction, :max_tokens_raw_mom_repair,
                        :max_tokens_agenda_compress, :max_tokens_reference_compress, :max_tokens_agenda_from_summary,
                        :max_tokens_executive_summary, :max_tokens_short_summary, :max_tokens_detailed_summary, :max_tokens_chunk_summary,
                        :max_tokens_key_points, :max_tokens_action_items, :max_tokens_key_decisions,
                        :max_tokens_speaker_summary, :max_tokens_speaker_key_points, :max_tokens_speaker_action_items,
                        :max_tokens_collection_chat, :max_tokens_collection_compare, :max_tokens_collection_topic_growth, :max_tokens_vocab_extractor,
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
                    "max_tokens_raw_mom_to_mom": patch.get("max_tokens_raw_mom_to_mom", 3000),
                    "max_tokens_raw_mom_extraction": patch.get("max_tokens_raw_mom_extraction", 1024),
                    "max_tokens_raw_mom_repair": patch.get("max_tokens_raw_mom_repair", 1024),
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
                    "updated_at": dt_to_str(now),
                },
            )
        await db.commit()

    # Convert back to bool for response representation
    if "use_ollama" in patch:
        patch["use_ollama"] = bool(patch["use_ollama"])
    if "generate_mom_auto" in patch:
        patch["generate_mom_auto"] = bool(patch["generate_mom_auto"])

    return {"message": "Settings updated.", **patch}



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
        async with get_db() as db:
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

    installed_map = {}
    for d in search_dirs:
        if d.exists() and d.is_dir():
            for child in d.iterdir():
                if child.is_dir() and any(child.iterdir()):
                    installed_map[child.name] = str(child)

    models_res = []
    seen_ids = set()

    for km in known_models:
        mid = km["id"]
        seen_ids.add(mid)
        installed = mid in installed_map
        models_res.append({
            **km,
            "installed": installed,
            "path": installed_map.get(mid),
        })

    for name, path in installed_map.items():
        if name not in seen_ids:
            models_res.append({
                "id": name,
                "name": name,
                "description": "Custom local embedding model",
                "dim": None,
                "quantization": "unknown",
                "installed": True,
                "path": path,
            })

    return {
        "active_model": settings.EMBEDDING_MODEL,
        "models": models_res,
    }



