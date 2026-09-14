"""
ai_chat_service.py — Standalone AI Chat orchestration service.

Handles query triage, Stage 2 ChromaDB retrieval, prompt construction,
and LLM inference for the standalone AI Chat tab.

Retrieval source: ONLY Stage 2 Long Changes points from ChromaDB.
This module does NOT import from collection_ai_service.py.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


# ── Prompt Templates ─────────────────────────────────────────────────────────

AI_CHAT_TRIAGE_PROMPT = """\
You are an AI assistant for a meeting analysis platform.
Analyze the user's question and recent conversation history, then decide if meeting context is needed.

You MUST respond with valid JSON strictly following this schema:
```json
{{
  "context_required": true | false,
  "detail": "string"
}}
```

Rules:
1. If the question requires information from meetings, transcripts, discussions, decisions, action items, speakers, or project context:
   - Set "context_required": true
   - Set "detail": A concise, targeted search query string with key terms, entities, topics, or speaker names for vector search. Do NOT answer the question at this stage.

2. If the question is general conversation, greeting (e.g., "Hello"), meta-question (e.g., "What can you do?"), or basic knowledge that does NOT require meeting context:
   - Set "context_required": false
   - Set "detail": The complete, polite, and helpful final answer to the user's question.

{conversation_history}

USER QUESTION: {question}

JSON RESPONSE:"""


AI_CHAT_ANSWER_PROMPT = """\
You are an expert meeting analyst assistant. Answer the user's question using ONLY the meeting discussion points provided below.

Rules:
- Base your answer ONLY on the provided context. Do NOT make up information.
- When referencing information from a specific meeting, cite it: [Meeting: <meeting_name>]
- If the context does not contain enough information to answer, say so clearly.
- Preserve all technical terms, project names, and acronyms exactly.
- Format your response in clear Markdown with headers, bullets, and bold text where appropriate.

{conversation_history}

MEETING DISCUSSION POINTS:
{context}

USER QUESTION: {question}

Answer:"""


# ── Triage ───────────────────────────────────────────────────────────────────

def triage_query(
    question: str,
    chat_history: List[Dict],
) -> Tuple[bool, str]:
    """
    Send the user's question to the LLM to determine if meeting context is needed.
    
    Returns (context_required, detail):
    - If context_required=True: detail is the optimized search query
    - If context_required=False: detail is the complete direct answer
    """
    from services.ai_provider import get_provider

    history_text = ""
    if chat_history:
        recent = chat_history[-4:]
        lines = ["PREVIOUS CONVERSATION:"]
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)

    prompt = AI_CHAT_TRIAGE_PROMPT.format(
        conversation_history=history_text,
        question=question,
    )

    try:
        provider = get_provider()
        raw_resp = provider._infer(prompt, max_new_tokens=300)
        if raw_resp:
            clean = raw_resp.strip()
            # Extract JSON from markdown codeblock if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                first = clean.find('{')
                last = clean.rfind('}')
                if first != -1 and last > first:
                    json_str = clean[first:last + 1]
                else:
                    json_str = clean

            try:
                data = json.loads(json_str)
                ctx_req = bool(data.get("context_required", True))
                detail = str(data.get("detail", "")).strip()
                if detail:
                    logger.info(f"[AIChat Triage] context_required={ctx_req}, detail='{detail[:80]}...'")
                    return ctx_req, detail
            except Exception:
                logger.warning(f"[AIChat Triage] JSON parse failed: '{clean[:100]}'")
            return True, clean
    except Exception as e:
        logger.warning(f"[AIChat Triage] Failed: {e}. Using raw query.")

    return True, question.strip()


# ── ChromaDB Search ──────────────────────────────────────────────────────────

def _get_embedder():
    """Lazy-import the text embedder."""
    from services.text_embedding_service import get_text_embedder
    embedder = get_text_embedder()
    embedder.load()
    return embedder


def search_stage2_points(
    query: str,
    user_id: str,
    meeting_id: Optional[str] = None,
    max_results: int = 8,
    max_total_chars: int = 15000,
) -> List[Dict[str, Any]]:
    """
    Search Stage 2 Long Changes points in ChromaDB.
    
    If meeting_id is provided, filters to that meeting only.
    Returns list of result dicts with metadata.
    """
    from services.vector_store import get_stage2_points_store

    embedder = _get_embedder()
    dim = embedder.embedding_dim()
    query_embedding = embedder.encode(query)

    store = get_stage2_points_store(user_id, dim)
    if not store.exists():
        logger.info("[AIChat] No Stage 2 points store found.")
        return []

    where_filter = None
    if meeting_id:
        where_filter = {"meeting_id": meeting_id}

    results = store.search(
        query_embedding,
        k=max_results * 2,  # Fetch extra for char budget filtering
        score_threshold=0.0,
        where=where_filter,
    )

    # Apply character budget
    filtered = []
    total_chars = 0
    for r in results:
        text = r.get("_text", "").strip()
        if not text:
            continue
        if total_chars + len(text) > max_total_chars:
            continue
        if len(filtered) >= max_results:
            break
        filtered.append(r)
        total_chars += len(text)

    logger.info(
        f"[AIChat] Retrieved {len(filtered)} Stage 2 points "
        f"({total_chars} chars) for query: '{query[:60]}...'"
        f"{f' [meeting={meeting_id}]' if meeting_id else ''}"
    )
    return filtered


def extract_meeting_sources(results: List[Dict]) -> List[Dict[str, str]]:
    """
    Extract unique meeting-level sources from search results.
    Returns deduplicated list of {id, name, date}.
    """
    seen = {}
    for r in results:
        mid = r.get("meeting_id") or r.get("recording_id", "")
        if not mid or mid in seen:
            continue
        seen[mid] = {
            "id": mid,
            "name": r.get("meeting_name", "Unknown Meeting"),
            "date": r.get("date", ""),
        }
    return list(seen.values())


def get_available_meetings(user_id: str) -> List[Dict[str, str]]:
    """
    Get all meetings that have Stage 2 points indexed in ChromaDB.
    Returns list of {id, name, date} sorted by date descending.
    """
    from services.vector_store import get_stage2_points_store

    embedder = _get_embedder()
    dim = embedder.embedding_dim()
    store = get_stage2_points_store(user_id, dim)

    if not store.exists():
        return []

    try:
        store.load_or_create()
        if store._collection is None:
            return []

        # Get all metadata (just need meeting_id, meeting_name, date)
        results = store._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        seen = {}
        for meta in metadatas:
            if not meta:
                continue
            mid = meta.get("meeting_id") or meta.get("recording_id", "")
            if not mid or mid in seen:
                continue
            seen[mid] = {
                "id": mid,
                "name": meta.get("meeting_name", "Unknown Meeting"),
                "date": meta.get("date", ""),
            }

        meetings = list(seen.values())
        # Sort by date descending (newest first)
        meetings.sort(key=lambda m: m.get("date", ""), reverse=True)
        logger.info(f"[AIChat] Found {len(meetings)} meetings with Stage 2 data.")
        return meetings

    except Exception as e:
        logger.error(f"[AIChat] Failed to get available meetings: {e}")
        return []


def get_stage2_meeting_ids(user_id: str) -> Set[str]:
    """
    Return the set of all meeting_ids that have Stage 2 points indexed in ChromaDB.
    """
    from services.vector_store import get_stage2_points_store

    embedder = _get_embedder()
    dim = embedder.embedding_dim()
    store = get_stage2_points_store(user_id, dim)

    if not store.exists():
        return set()

    try:
        store.load_or_create()
        if store._collection is None:
            return set()

        results = store._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        mids: Set[str] = set()
        for meta in metadatas:
            if not meta:
                continue
            mid = meta.get("meeting_id") or meta.get("recording_id", "")
            if mid:
                mids.add(str(mid))

        return mids
    except Exception as e:
        logger.error(f"[AIChat] Failed to get Stage 2 meeting IDs: {e}")
        return set()


# ── Context Formatting ───────────────────────────────────────────────────────

def format_context(results: List[Dict]) -> str:
    """Format retrieved Stage 2 points into a context string grouped by meeting."""
    if not results:
        return "(No relevant meeting discussion points found.)"

    sections: Dict[str, List[str]] = {}
    for r in results:
        meeting = r.get("meeting_name", "Unknown Meeting")
        if meeting not in sections:
            sections[meeting] = []
        text = r.get("_text", "").strip()
        speakers = r.get("speakers", "")
        if isinstance(speakers, list):
            speakers = ", ".join(speakers)
        speaker_info = f" [Speakers: {speakers}]" if speakers else ""
        sections[meeting].append(f"{text}{speaker_info}")

    parts = []
    for name, texts in sections.items():
        parts.append(f"--- {name} ---")
        for t in texts:
            parts.append(f"• {t}")
        parts.append("")

    return "\n".join(parts)


# ── Prompt Building ──────────────────────────────────────────────────────────

def build_answer_prompt(
    question: str,
    context_results: List[Dict],
    chat_history: List[Dict],
) -> str:
    """Build the final LLM prompt with retrieved Stage 2 context."""
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        lines = ["PREVIOUS CONVERSATION:"]
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)

    context_text = format_context(context_results)

    return AI_CHAT_ANSWER_PROMPT.format(
        conversation_history=history_text,
        context=context_text,
        question=question,
    )


# ── LLM Streaming Inference ──────────────────────────────────────────────────

def stream_inference(
    prompt: str,
    max_new_tokens: int = 1500,
    chunk_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Run LLM inference with streaming support.
    Uses the same provider pattern as collection_ai_service.
    Returns the complete generated text.
    """
    from services.ai_provider import get_provider, QwenProvider
    from config import settings
    import urllib.request

    provider = get_provider()

    # Load active settings
    cfg = QwenProvider._get_active_settings()
    use_ollama = cfg["use_ollama"]
    server_url = cfg["ollama_server_url"]
    priority_list = cfg["ollama_model_priority"]

    # Try Ollama streaming if enabled
    if use_ollama and chunk_callback:
        model_name = QwenProvider._detect_ollama_model(server_url, priority_list)
        if model_name:
            try:
                return _stream_ollama_chat(
                    server_url, model_name, prompt,
                    max_new_tokens, chunk_callback, cfg,
                )
            except Exception as e:
                logger.warning(f"[AIChat] Ollama streaming failed: {e}. Falling back to local.")

    # Fallback: local inference
    try:
        full_response = provider._infer(prompt, max_new_tokens=max_new_tokens)
        full_response = full_response.strip()

        if chunk_callback and full_response:
            words = full_response.split(" ")
            buffer = []
            for i, word in enumerate(words):
                buffer.append(word)
                if len(buffer) >= 3 or i == len(words) - 1:
                    chunk_callback(" ".join(buffer) + (" " if i < len(words) - 1 else ""))
                    buffer = []

        return full_response
    except Exception as e:
        logger.error(f"[AIChat] LLM inference failed: {e}", exc_info=True)
        error_msg = f"⚠️ AI generation failed: {str(e)}"
        if chunk_callback:
            chunk_callback(error_msg)
        return error_msg
    finally:
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _stream_ollama_chat(
    server_url: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    chunk_callback: Callable[[str], None],
    cfg: dict,
) -> str:
    """Stream from Ollama's /api/chat endpoint."""
    import urllib.request

    from services.ai_provider import QwenProvider, calculate_dynamic_num_ctx, strip_thinking

    system_content = (
        "You are an expert meeting analyst assistant. "
        "Answer questions accurately using provided meeting context. "
        "Preserve all technical terminology."
    )

    dynamic_enabled = bool(cfg.get("ollama_dynamic_ctx", True))
    manual_num_ctx = int(cfg.get("ollama_num_ctx", 32768))

    est_input_tokens, calculated_num_ctx = calculate_dynamic_num_ctx(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        safety_buffer=512,
        system_content=system_content,
        tokenizer=QwenProvider._tokenizer,
    )

    selected_num_ctx = calculated_num_ctx if dynamic_enabled else manual_num_ctx
    think_enabled = bool(cfg.get("ollama_think", False))

    options = {
        "num_predict": max_new_tokens,
        "temperature": float(cfg["ollama_temperature"]),
        "num_ctx": selected_num_ctx,
        "repeat_penalty": float(cfg["ollama_repeat_penalty"]),
        "top_p": float(cfg["ollama_top_p"]),
        "top_k": int(cfg["ollama_top_k"]),
        "think": think_enabled,
    }
    if cfg["ollama_seed"] is not None and cfg["ollama_seed"] >= 0:
        options["seed"] = int(cfg["ollama_seed"])
    if cfg["ollama_stop"]:
        stop_seqs = [s.strip() for s in cfg["ollama_stop"].split(",") if s.strip()]
        if stop_seqs:
            options["stop"] = stop_seqs
    if cfg["ollama_num_thread"] is not None and cfg["ollama_num_thread"] > 0:
        options["num_thread"] = int(cfg["ollama_num_thread"])
    if cfg["ollama_num_gpu"] is not None and cfg["ollama_num_gpu"] >= 0:
        options["num_gpu"] = int(cfg["ollama_num_gpu"])

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        "options": options,
        "think": think_enabled,
        "stream": True,
    }
    if cfg["ollama_keep_alive"] is not None:
        try:
            payload["keep_alive"] = int(cfg["ollama_keep_alive"])
        except ValueError:
            payload["keep_alive"] = str(cfg["ollama_keep_alive"])

    base_url = server_url.rstrip("/")
    url = f"{base_url}/api/chat"

    logger.info(
        f"[AIChat] Ollama streaming: model={model}, num_ctx={selected_num_ctx}, "
        f"prompt_chars={len(prompt)}"
    )

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    full_response = []
    with urllib.request.urlopen(req, timeout=120.0) as response:
        for line in response:
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                if content:
                    full_response.append(content)
                    chunk_callback(content)
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                continue

    return strip_thinking("".join(full_response))
