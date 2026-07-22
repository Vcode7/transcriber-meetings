"""
collection_ai_service.py — Business logic for Collection AI Chat.

Handles multi-meeting RAG retrieval, prompt construction, and LLM inference
for three AI capabilities within a meeting collection:
  1. Free-form Q&A (chat)
  2. Meeting comparison reports
  3. Topic growth tracking

This module does NOT handle HTTP concerns (those live in collection_ai_router.py).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

class MeetingChunk:
    """A single retrieved text chunk with meeting-level metadata."""
    __slots__ = ("meeting_id", "meeting_name", "meeting_date", "text",
                 "score", "start", "end", "source", "speakers")

    def __init__(self, meeting_id: str, meeting_name: str, meeting_date: str,
                 text: str, score: float, start: float = 0.0, end: float = 0.0,
                 source: str = "transcript", speakers: list | None = None):
        self.meeting_id = meeting_id
        self.meeting_name = meeting_name
        self.meeting_date = meeting_date
        self.text = text
        self.score = score
        self.start = start
        self.end = end
        self.source = source
        self.speakers = speakers or []


class CollectionContext:
    """Aggregated retrieval results across multiple meetings."""
    def __init__(self):
        self.chunks: List[MeetingChunk] = []
        self.meeting_ids: List[str] = []
        self.total_chars: int = 0


# ── Embedding helper ─────────────────────────────────────────────────────────

def _get_embedder():
    """Lazy-import the text embedder."""
    from services.text_embedding_service import get_text_embedder
    embedder = get_text_embedder()
    embedder.load()
    return embedder


# ── Multi-meeting RAG retrieval ──────────────────────────────────────────────

def retrieve_collection_context(
    meeting_ids: List[str],
    meeting_meta: Dict[str, Dict],
    query: str,
    user_id: str,
    max_chunks: int = 10,
    k_per_meeting: int = 5,
    max_total_chars: int = 20000,
    include_global: bool = True,
) -> CollectionContext:
    """
    Query FAISS transcript stores for each meeting in the collection and
    merge results into a ranked, deduplicated, score-sorted context capped at max_chunks.

    Parameters
    ----------
    meeting_ids    : List of recording IDs in the collection.
    meeting_meta   : Dict mapping recording_id -> {filename, created_at, ...}
    query          : Search query string (or LLM retrieval plan).
    user_id        : Current user ID (for global context lookup).
    max_chunks     : Maximum number of retrieved chunks to include in final context.
    k_per_meeting  : Max chunks to retrieve per meeting store.
    max_total_chars: Hard character budget for assembled context.
    include_global : Whether to include global context docs.

    Returns
    -------
    CollectionContext with ranked, deduplicated chunks.
    """
    from services.vector_store import get_transcript_store, get_global_context_store

    embedder = _get_embedder()
    dim = embedder.embedding_dim()
    query_embedding = embedder.encode(query)

    ctx = CollectionContext()
    ctx.meeting_ids = meeting_ids

    all_chunks: List[MeetingChunk] = []

    # ── Search each meeting's transcript store ────────────────────────────────
    for mid in meeting_ids:
        meta = meeting_meta.get(mid, {})
        name = meta.get("filename", "Unknown Meeting")
        date = meta.get("created_at", "")

        try:
            store = get_transcript_store(mid, dim)
            if not store.exists():
                continue

            results = store.search(query_embedding, k=k_per_meeting)
            for res in results:
                text = res.get("_text", "").strip()
                if not text:
                    continue
                all_chunks.append(MeetingChunk(
                    meeting_id=mid,
                    meeting_name=name,
                    meeting_date=date,
                    text=text,
                    score=float(res.get("score", 0.0)),
                    start=float(res.get("start", 0.0)),
                    end=float(res.get("end", 0.0)),
                    source="transcript",
                    speakers=res.get("speakers", []),
                ))
        except Exception as e:
            logger.warning(f"[CollectionAI] Transcript search failed for {mid}: {e}")

    # ── Optionally include global context ──────────────────────────────────────
    if include_global:
        try:
            g_store = get_global_context_store(user_id, dim)
            if g_store.exists():
                g_results = g_store.search(query_embedding, k=3)
                for res in g_results:
                    text = res.get("_text", "").strip()
                    if text:
                        all_chunks.append(MeetingChunk(
                            meeting_id="global",
                            meeting_name="Global Knowledge Base",
                            meeting_date="",
                            text=text,
                            score=float(res.get("score", 0.0)),
                            source="global_context",
                        ))
        except Exception as e:
            logger.warning(f"[CollectionAI] Global context search failed: {e}")

    # ── Rank by similarity score descending ──────────────────────────────────
    all_chunks.sort(key=lambda c: c.score, reverse=True)

    # ── Deduplicate identical chunks ──────────────────────────────────────────
    deduped_chunks: List[MeetingChunk] = []
    seen_signatures = set()

    for chunk in all_chunks:
        # Signature based on meeting ID, start/end timestamps, and text prefix
        sig = (chunk.meeting_id, round(chunk.start, 1), round(chunk.end, 1), chunk.text[:80])
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        deduped_chunks.append(chunk)

    # ── Enforce max_chunks cap and character budget ────────────────────────────
    total_chars = 0
    for chunk in deduped_chunks:
        if len(ctx.chunks) >= max_chunks:
            break
        if total_chars + len(chunk.text) > max_total_chars:
            continue
        ctx.chunks.append(chunk)
        total_chars += len(chunk.text)

    ctx.total_chars = total_chars
    logger.info(
        f"[CollectionAI] Retrieved {len(ctx.chunks)} chunks "
        f"(max_chunks={max_chunks}, {total_chars} chars) from {len(meeting_ids)} meetings"
    )
    return ctx


# ── Two-Stage Retrieval Orchestrator ─────────────────────────────────────────

def generate_retrieval_plan(
    question: str,
    chat_history: List[Dict],
) -> tuple[bool, str]:
    """
    Stage 1 — Context Planning & Triage:
    Pass user question (without context) to LLM.
    Returns (context_required: bool, detail: str).

    If context_required is True: detail is the retrieval plan for vector search.
    If context_required is False: detail is the final complete answer to the user question.
    """
    import json
    import re
    from services.ai_provider import _get_prompt, get_provider

    template = _get_prompt("collection_planning")
    if not template:
        return True, question.strip()

    history_text = ""
    if chat_history:
        recent = chat_history[-4:]
        history_lines = ["PREVIOUS CONVERSATION:"]
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

    prompt = template.format(
        conversation_history=history_text,
        question=question,
    )

    try:
        provider = get_provider()
        raw_resp = provider._infer(prompt, max_new_tokens=300)
        if raw_resp:
            clean_resp = raw_resp.strip()
            # Extract JSON block if surrounded by markdown codeblocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean_resp, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Find outer braces
                first_brace = clean_resp.find('{')
                last_brace = clean_resp.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = clean_resp[first_brace:last_brace + 1]
                else:
                    json_str = clean_resp

            try:
                data = json.loads(json_str)
                context_req = bool(data.get("context_required", True))
                detail = str(data.get("detail", "")).strip()

                if detail:
                    logger.info(
                        f"[CollectionAI Stage 1] Triage parsed — context_required={context_req}, "
                        f"detail='{detail[:80]}...'"
                    )
                    return context_req, detail
            except Exception:
                logger.warning(f"[CollectionAI Stage 1] JSON parse failed for response: '{clean_resp[:100]}'")

            # Fallback if text response wasn't valid JSON
            return True, clean_resp
    except Exception as e:
        logger.warning(f"[CollectionAI Stage 1] Retrieval plan generation failed: {e}. Using raw query.")

    return True, question.strip()


def retrieve_collection_context_two_stage(
    meeting_ids: List[str],
    meeting_meta: Dict[str, Dict],
    query: str,
    user_id: str,
    chat_history: List[Dict] | None = None,
    max_context: int = 10,
    k_per_meeting: int = 5,
    max_total_chars: int = 20000,
    include_global: bool = True,
) -> tuple[CollectionContext, str, bool]:
    """
    Two-Stage Retrieval Pipeline:
      Stage 1: Generate retrieval plan / triage decision via LLM.
      Stage 2: Vector search with retrieval plan (if context_required==True).
      Fallback: Retries search using raw user query if retrieval plan yields 0 chunks.

    Returns (CollectionContext, detail_or_plan_used, context_required)
    """
    # Stage 1: Context Planning & Triage
    context_required, detail = generate_retrieval_plan(query, chat_history or [])

    if not context_required:
        # Direct answer generated in Stage 1 — skip RAG vector search entirely
        logger.info("[CollectionAI Stage 1] context_required=False. Skipping RAG retrieval.")
        ctx = CollectionContext()
        return ctx, detail, False

    # Stage 2: Vector Search with Retrieval Plan
    ctx = retrieve_collection_context(
        meeting_ids=meeting_ids,
        meeting_meta=meeting_meta,
        query=detail,
        user_id=user_id,
        max_chunks=max_context,
        k_per_meeting=k_per_meeting,
        max_total_chars=max_total_chars,
        include_global=include_global,
    )

    # Fallback to raw user query if retrieval plan yields 0 chunks
    if not ctx.chunks and detail != query.strip():
        logger.info("[CollectionAI Stage 2 Fallback] Retrieval plan produced 0 chunks. Retrying search with raw query...")
        ctx = retrieve_collection_context(
            meeting_ids=meeting_ids,
            meeting_meta=meeting_meta,
            query=query,
            user_id=user_id,
            max_chunks=max_context,
            k_per_meeting=k_per_meeting,
            max_total_chars=max_total_chars,
            include_global=include_global,
        )
        return ctx, query.strip(), True

    return ctx, detail, True


def retrieve_meeting_context(
    meeting_id: str,
    meeting_meta: Dict,
    query: str,
    user_id: str,
    k: int = 10,
    max_chars: int = 12000,
) -> List[MeetingChunk]:
    """
    Retrieve context from a single meeting's transcript store.
    Used by the comparison feature to get focused context per meeting.
    """
    from services.vector_store import get_transcript_store

    embedder = _get_embedder()
    dim = embedder.embedding_dim()

    # For comparison, we use a broad query combining common meeting topics
    query_embedding = embedder.encode(query)

    name = meeting_meta.get("filename", "Unknown Meeting")
    date = meeting_meta.get("created_at", "")

    chunks: List[MeetingChunk] = []
    total_chars = 0

    try:
        store = get_transcript_store(meeting_id, dim)
        if not store.exists():
            return chunks

        results = store.search(query_embedding, k=k)
        for res in results:
            text = res.get("_text", "").strip()
            if not text:
                continue
            if total_chars + len(text) > max_chars:
                continue
            chunks.append(MeetingChunk(
                meeting_id=meeting_id,
                meeting_name=name,
                meeting_date=date,
                text=text,
                score=float(res.get("score", 0.0)),
                start=float(res.get("start", 0.0)),
                end=float(res.get("end", 0.0)),
                source="transcript",
                speakers=res.get("speakers", []),
            ))
            total_chars += len(text)
    except Exception as e:
        logger.warning(f"[CollectionAI] Single meeting retrieval failed for {meeting_id}: {e}")

    return chunks


def get_meeting_full_context(
    meeting_id: str,
    meeting_meta: Dict,
    max_chars: int = 12000,
) -> List[MeetingChunk]:
    """
    Retrieve ALL transcript chunks from a meeting (not query-based).
    Used for comparison to get a broad view of the meeting.
    Reads chunks chronologically from the FAISS store metadata.
    """
    from services.vector_store import get_transcript_store

    embedder = _get_embedder()
    dim = embedder.embedding_dim()

    name = meeting_meta.get("filename", "Unknown Meeting")
    date = meeting_meta.get("created_at", "")

    chunks: List[MeetingChunk] = []
    total_chars = 0

    try:
        store = get_transcript_store(meeting_id, dim)
        if not store.exists():
            return chunks

        store.load_or_create()
        # Read all metadata entries chronologically
        sorted_meta = sorted(store._meta, key=lambda m: m.get("start", 0.0))

        for meta_entry in sorted_meta:
            text = meta_entry.get("_text", "").strip()
            if not text:
                continue
            if total_chars + len(text) > max_chars:
                break
            chunks.append(MeetingChunk(
                meeting_id=meeting_id,
                meeting_name=name,
                meeting_date=date,
                text=text,
                score=1.0,
                start=float(meta_entry.get("start", 0.0)),
                end=float(meta_entry.get("end", 0.0)),
                source="transcript",
                speakers=meta_entry.get("speakers", []),
            ))
            total_chars += len(text)
    except Exception as e:
        logger.warning(f"[CollectionAI] Full context retrieval failed for {meeting_id}: {e}")

    return chunks


# ── Context formatting ───────────────────────────────────────────────────────

def _format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def format_collection_context(ctx: CollectionContext) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    if not ctx.chunks:
        return "(No relevant meeting content found.)"

    sections: Dict[str, List[str]] = {}
    for chunk in ctx.chunks:
        key = chunk.meeting_name
        if key not in sections:
            sections[key] = []

        ts_info = ""
        if chunk.start > 0:
            ts_info = f" [{_format_timestamp(chunk.start)} - {_format_timestamp(chunk.end)}]"

        sections[key].append(f"{chunk.text}{ts_info}")

    parts = []
    for meeting_name, texts in sections.items():
        parts.append(f"--- {meeting_name} ---")
        parts.extend(texts)
        parts.append("")

    return "\n".join(parts)


def format_meeting_chunks(chunks: List[MeetingChunk]) -> str:
    """Format chunks from a single meeting into context text."""
    if not chunks:
        return "(No content available.)"

    lines = []
    for chunk in chunks:
        ts_info = ""
        if chunk.start > 0:
            ts_info = f" [{_format_timestamp(chunk.start)} - {_format_timestamp(chunk.end)}]"
        lines.append(f"{chunk.text}{ts_info}")

    return "\n".join(lines)


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_chat_prompt(
    question: str,
    context: CollectionContext,
    chat_history: List[Dict],
) -> str:
    """Build the prompt for free-form collection Q&A."""
    from services.ai_provider import _get_prompt

    template = _get_prompt("collection_chat")

    # Format conversation history (last 6 messages for context continuity)
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        history_lines = ["PREVIOUS CONVERSATION:"]
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            # Truncate long previous messages
            if len(content) > 500:
                content = content[:500] + "..."
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

    context_text = format_collection_context(context)

    return template.format(
        conversation_history=history_text,
        context=context_text,
        question=question,
    )


def build_comparison_prompt(
    meeting_a_name: str,
    meeting_a_date: str,
    meeting_a_chunks: List[MeetingChunk],
    meeting_b_name: str,
    meeting_b_date: str,
    meeting_b_chunks: List[MeetingChunk],
) -> str:
    """Build the prompt for meeting comparison."""
    from services.ai_provider import _get_prompt

    template = _get_prompt("collection_compare")

    return template.format(
        meeting_a_name=meeting_a_name,
        meeting_a_date=meeting_a_date,
        meeting_a_context=format_meeting_chunks(meeting_a_chunks),
        meeting_b_name=meeting_b_name,
        meeting_b_date=meeting_b_date,
        meeting_b_context=format_meeting_chunks(meeting_b_chunks),
    )


def build_topic_growth_prompt(
    topic: str,
    meetings_data: List[Dict],
) -> str:
    """
    Build the prompt for topic growth tracking.

    meetings_data: sorted chronologically, each with:
      {"name": str, "date": str, "chunks": List[MeetingChunk]}
    """
    from services.ai_provider import _get_prompt

    template = _get_prompt("collection_topic_growth")

    meeting_sections = []
    for md in meetings_data:
        section = f"--- {md['name']} ({md['date']}) ---\n"
        section += format_meeting_chunks(md["chunks"])
        meeting_sections.append(section)

    return template.format(
        topic=topic,
        meetings_context="\n\n".join(meeting_sections),
    )


# ── LLM inference ────────────────────────────────────────────────────────────

def run_collection_ai_inference(
    prompt: str,
    max_new_tokens: int = 1500,
) -> str:
    """
    Run LLM inference for a collection AI task.

    Uses the existing QwenProvider._infer() method which handles:
    - Ollama fallback (if enabled in settings)
    - Local Qwen3 4B model (default)

    Returns the generated text.
    """
    from services.ai_provider import get_provider

    provider = get_provider()
    try:
        result = provider._infer(prompt, max_new_tokens=max_new_tokens)
        return result.strip()
    except Exception as e:
        logger.error(f"[CollectionAI] LLM inference failed: {e}", exc_info=True)
        return f"⚠️ AI generation failed: {str(e)}"
    finally:
        # Clean up GPU memory after inference
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def run_collection_ai_inference_streaming(
    prompt: str,
    max_new_tokens: int = 1500,
    chunk_callback: Optional[Callable[[str], None]] = None,
    task_key: Optional[str] = None,
) -> str:
    """
    Run LLM inference with streaming support.

    For Ollama: uses actual streaming via /api/chat endpoint.
    For local Qwen: generates full response, then delivers via callback in chunks.

    Returns the complete generated text.
    """
    from services.ai_provider import get_provider, QwenProvider
    from config import settings
    import sqlite3
    import json
    import urllib.request

    provider = get_provider()

    # Load active settings
    cfg = QwenProvider._get_active_settings()
    use_ollama = cfg["use_ollama"]
    server_url = cfg["ollama_server_url"]
    port = cfg["ollama_port"]
    priority_list = cfg["ollama_model_priority"]

    # Override max_new_tokens dynamically from config if task_key matches
    if task_key:
        db_max_tokens = cfg.get(f"max_tokens_{task_key}")
        if db_max_tokens is not None and db_max_tokens > 0:
            max_new_tokens = db_max_tokens

    # Try Ollama streaming if enabled
    if use_ollama and chunk_callback:
        model_name = QwenProvider._detect_ollama_model(server_url, priority_list)
        if model_name:
            try:
                return _stream_ollama(server_url, model_name, prompt, max_new_tokens, chunk_callback, cfg)
            except Exception as e:
                logger.warning(f"[CollectionAI] Ollama streaming failed: {e}. Falling back to local.")

    # Fallback: local inference (non-streaming, then deliver in chunks)
    try:
        full_response = provider._infer(prompt, max_new_tokens=max_new_tokens, task_key=task_key)
        full_response = full_response.strip()

        if chunk_callback and full_response:
            # Deliver in word-sized chunks to simulate streaming
            words = full_response.split(" ")
            buffer = []
            for i, word in enumerate(words):
                buffer.append(word)
                if len(buffer) >= 3 or i == len(words) - 1:
                    chunk_callback(" ".join(buffer) + (" " if i < len(words) - 1 else ""))
                    buffer = []

        return full_response
    except Exception as e:
        logger.error(f"[CollectionAI] LLM inference failed: {e}", exc_info=True)
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


def _stream_ollama(
    server_url: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    chunk_callback: Callable[[str], None],
    cfg: dict,
) -> str:
    """Stream from Ollama's /api/chat endpoint with stream=True."""
    import urllib.request
    import json

    from services.ai_provider import QwenProvider, calculate_dynamic_num_ctx

    system_content = (
        "You are an expert enterprise meeting analyst. "
        "Follow instructions exactly. Preserve all technical terminology."
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

    # Prepare options payload with validation defaults
    options = {
        "num_predict": max_new_tokens,
        "temperature": float(cfg["ollama_temperature"]),
        "num_ctx": selected_num_ctx,
        "repeat_penalty": float(cfg["ollama_repeat_penalty"]),
        "top_p": float(cfg["ollama_top_p"]),
        "top_k": int(cfg["ollama_top_k"]),
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
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": prompt},
        ],
        "options": options,
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
        f"[CollectionAI] Sending Streaming Ollama Request:\n"
        f"  - Model: {model}\n"
        f"  - Server URL: {server_url}\n"
        f"  - Dynamic Context Window: {'ENABLED (ON)' if dynamic_enabled else 'DISABLED (OFF)'}\n"
        f"  - Estimated Input Tokens: {est_input_tokens}\n"
        f"  - Max Output Tokens (num_predict): {max_new_tokens}\n"
        f"  - Selected num_ctx: {selected_num_ctx} (calculated: {calculated_num_ctx}, manual setting: {manual_num_ctx})\n"
        f"  - Prompt Chars: {len(prompt)}\n"
        f"  - Options: {options}\n"
        f"  - Keep Alive: {payload.get('keep_alive', 'N/A')}"
    )
    logger.debug(f"[CollectionAI] Complete Streaming Ollama Request JSON Payload: {json.dumps(payload)}")

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    full_response = []
    response_metadata = {}
    
    with urllib.request.urlopen(req, timeout=120.0) as response:
        status_code = response.status
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
                    response_metadata = data
                    break
            except json.JSONDecodeError:
                continue

    # Post-logging response details
    content_str = "".join(full_response)
    total_dur = response_metadata.get("total_duration")
    load_dur = response_metadata.get("load_duration")
    prompt_eval_dur = response_metadata.get("prompt_eval_duration")
    eval_dur = response_metadata.get("eval_duration")
    
    total_dur_str = f"{total_dur / 1e9:.2f}s" if total_dur is not None else "N/A"
    load_dur_str = f"{load_dur / 1e9:.2f}s" if load_dur is not None else "N/A"
    prompt_eval_dur_str = f"{prompt_eval_dur / 1e9:.2f}s" if prompt_eval_dur is not None else "N/A"
    eval_dur_str = f"{eval_dur / 1e9:.2f}s" if eval_dur is not None else "N/A"

    done_reason = response_metadata.get("done_reason", "")
    end_reason = "Unknown"
    if done_reason == "stop":
        end_reason = "EOS token or stop sequence reached"
    elif done_reason == "length":
        end_reason = "Reached maximum tokens (num_predict limit)"

    resp_tokens = "N/A"
    try:
        from services.ai_provider import QwenProvider
        if QwenProvider._tokenizer:
            resp_tokens = len(QwenProvider._tokenizer.encode(content_str))
    except Exception:
        pass

    logger.info(
        f"[CollectionAI] Received Streaming Ollama Response:\n"
        f"  - HTTP Status: {status_code}\n"
        f"  - Prompt Eval Tokens: {response_metadata.get('prompt_eval_count', 'N/A')}\n"
        f"  - Output Tokens: {response_metadata.get('eval_count', 'N/A')}\n"
        f"  - Total Duration: {total_dur_str}\n"
        f"  - Load Duration: {load_dur_str}\n"
        f"  - Prompt Eval Duration: {prompt_eval_dur_str}\n"
        f"  - Output Eval Duration: {eval_dur_str}\n"
        f"  - Done: {response_metadata.get('done', 'N/A')}\n"
        f"  - Done Reason: {done_reason} ({end_reason})\n"
        f"  - Response Chars: {len(content_str)}\n"
        f"  - Response Tokens (est): {resp_tokens}"
    )

    from services.ai_provider import strip_thinking
    return strip_thinking(content_str)


# ── Unload helper ────────────────────────────────────────────────────────────

def unload_ai_models():
    """Unload LLM and embedding models to free GPU memory."""
    try:
        from services.ai_provider import QwenProvider
        QwenProvider.unload_model()
    except Exception as e:
        logger.warning(f"[CollectionAI] Failed to unload LLM: {e}")

    try:
        from services.text_embedding_service import unload_text_embedder
        unload_text_embedder()
    except Exception as e:
        logger.warning(f"[CollectionAI] Failed to unload text embedder: {e}")

    logger.info("[CollectionAI] GPU cleanup finished.")


def ensure_meetings_are_embedded(meeting_ids: List[str], user_id: str) -> None:
    """
    Check if the FAISS store exists for each meeting_id.
    If not, load the transcript from SQLite, chunk it, embed it, and save the store.
    """
    from services.vector_store import get_transcript_store
    from services.rag_pipeline import embed_transcript
    from database import from_json
    from config import settings
    import sqlite3

    db_url = getattr(settings, "DATABASE_URL", "")
    db_path = None
    if db_url.startswith("sqlite+aiosqlite:///"):
        db_path = db_url[len("sqlite+aiosqlite:///"):]
    elif db_url.startswith("sqlite:///"):
        db_path = db_url[len("sqlite:///"):]
    elif db_url:
        db_path = db_url

    if not db_path:
        logger.error("[CollectionAI] Database path not configured, cannot ensure embeddings.")
        return

    embedder = _get_embedder()
    dim = embedder.embedding_dim()

    for mid in meeting_ids:
        store = get_transcript_store(mid, dim)
        if not store.exists():
            logger.info(f"[CollectionAI] FAISS transcript store not found for meeting {mid}. Auto-embedding transcript...")
            try:
                conn = sqlite3.connect(db_path, timeout=10.0)
                cursor = conn.cursor()
                cursor.execute("SELECT transcript FROM recordings WHERE id = ?", (mid,))
                row = cursor.fetchone()
                
                if row and row[0]:
                    transcript_data = from_json(row[0], [])
                    if transcript_data:
                        # Call embed_transcript (this creates and saves the FAISS index)
                        added = embed_transcript(mid, transcript_data, user_id)
                        if added > 0:
                            # Update transcript_embedded column synchronously
                            cursor.execute("UPDATE recordings SET transcript_embedded = 1 WHERE id = ?", (mid,))
                            conn.commit()
                            logger.info(f"[CollectionAI] Successfully auto-embedded {added} chunks for meeting {mid} and updated DB.")
                conn.close()
            except Exception as e:
                logger.error(f"[CollectionAI] Failed to auto-embed meeting {mid}: {e}", exc_info=True)

