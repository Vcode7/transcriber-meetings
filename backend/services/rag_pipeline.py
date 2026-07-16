"""
rag_pipeline.py — Core RAG orchestration for the Raw MoM extraction pipeline.

This module handles all embedding and retrieval operations:
  1. embed_global_context_doc  — embed an org-wide document into per-user FAISS
  2. embed_meeting_context     — embed meeting context attachments for a recording
  3. embed_transcript          — embed transcript chunks for a recording
  4. parse_agenda_items        — parse raw agenda text into [{topic, speaker}]
  5. retrieve_evidence_for_agenda — query all three FAISS stores and merge results
  6. generate_raw_mom          — orchestrate the full pipeline for a recording

The existing MoM pipeline is NOT involved here at all.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Embedding helper ──────────────────────────────────────────────────────────

def _get_embedder():
    """Lazy-import the text embedder to avoid loading at module import time."""
    from services.text_embedding_service import get_text_embedder
    embedder = get_text_embedder()
    embedder.load()
    return embedder


def _get_dim() -> int:
    """Return the embedding dimension from the loaded model."""
    return _get_embedder().embedding_dim()


def _filter_by_relative_similarity(
    results: List[Dict],
    cutoff_value: float,
    source_name: str
) -> List[Dict]:
    """
    Filter retrieved vector chunks based on relative similarity to the best score.
    
    Checks that: score >= best_score - cutoff_value
    Falls back to returning the single best chunk if no chunks pass the relative threshold.
    """
    if not results:
        logger.info(f"[RAG] Retrieved 0 {source_name} chunks.")
        return []

    # Sort results by similarity score descending
    results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)

    best_score = results[0].get("score", 0.0)
    cutoff = best_score - cutoff_value

    logger.info(f"[RAG] Retrieved {len(results)} {source_name} chunks.")
    logger.info(f"[RAG] Best similarity: {best_score:.3f}")
    logger.info(f"[RAG] Relative cutoff: {cutoff:.3f}")

    retained = []
    for res in results:
        score = res.get("score", 0.0)
        chunk_idx = res.get("chunk_index", 0)
        filename = res.get("filename", "transcript" if source_name == "transcript" else "unknown")
        text_len = len(res.get("_text", ""))

        if score >= cutoff:
            retained.append(res)
            logger.info(f"[RAG] Keep chunk={chunk_idx} score={score:.3f} source={filename} chars={text_len}")
        else:
            logger.info(f"[RAG] Discard chunk={chunk_idx} score={score:.3f} (below relative cutoff)")

    if not retained and results:
        best_res = results[0]
        retained.append(best_res)
        best_idx = best_res.get("chunk_index", 0)
        best_fn = best_res.get("filename", "transcript" if source_name == "transcript" else "unknown")
        best_len = len(best_res.get("_text", ""))
        logger.info(f"[RAG] Fallback to single best: Keep chunk={best_idx} score={best_score:.3f} source={best_fn} chars={best_len}")

    logger.info(f"[RAG] Filtered {len(results)} -> {len(retained)} chunks.")
    return retained



# ── 1. Global Context Document Embedding ─────────────────────────────────────

def embed_global_context_doc(
    doc_id: str,
    file_path: str,
    filename: str,
    user_id: str,
) -> int:
    """
    Extract text from a document, chunk it, embed it, and store in the
    per-user global context FAISS index.

    Parameters
    ----------
    doc_id    : UUID of the global_context_documents record.
    file_path : Absolute path to the uploaded file.
    filename  : Original filename (used for format detection).
    user_id   : Owner user ID (indexes are per-user).

    Returns
    -------
    Number of chunks added to the index.
    """
    from services.doc_extractor import extract_text_from_file
    from services.text_chunker import chunk_text
    from services.vector_store import get_global_context_store
    from config import settings

    logger.info(f"[RAG] Embedding global context doc: {filename} (doc_id={doc_id})")

    text = extract_text_from_file(file_path, filename)
    if not text or not text.strip():
        logger.warning(f"[RAG] No text extracted from {filename} — skipping embed")
        return 0

    chunks = chunk_text(
        text,
        chunk_size=settings.RAG_CHUNK_SIZE,
        overlap=settings.RAG_CHUNK_OVERLAP,
    )
    if not chunks:
        logger.warning(f"[RAG] No chunks produced for {filename}")
        return 0

    embedder = _get_embedder()
    dim = embedder.embedding_dim()
    store = get_global_context_store(user_id, dim)

    # Remove any previous chunks for this doc (re-embedding on file change)
    store.delete_by_filter("doc_id", doc_id)

    texts = [c["text"] for c in chunks]
    import numpy as np
    embeddings = embedder.encode_batch(texts)

    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": c["chunk_index"],
            "source": "global_context",
            "user_id": user_id,
        }
        for c in chunks
    ]

    added = store.add(texts, metadatas, embeddings=embeddings)
    logger.info(f"[RAG] Added {added} chunks for global context doc {doc_id}")
    return added


def remove_global_context_doc(doc_id: str, user_id: str) -> int:
    """Remove all chunks for a global context document from the FAISS index."""
    from services.vector_store import get_global_context_store
    dim = _get_dim()
    store = get_global_context_store(user_id, dim)
    removed = store.delete_by_filter("doc_id", doc_id)
    logger.info(f"[RAG] Removed {removed} chunks for global context doc {doc_id}")
    return removed


# ── 2. Meeting Context Embedding ──────────────────────────────────────────────

def embed_meeting_context(recording_id: str, user_id: str) -> int:
    """
    Embed all 'context' type attachments for a recording.

    Reads attachments from the DB, extracts text, chunks, embeds, and stores
    in a per-recording FAISS index. Clears any previous meeting context index
    first to ensure freshness.

    Returns
    -------
    Total number of chunks added.
    """
    import asyncio
    from services.doc_extractor import extract_text_from_file
    from services.text_chunker import chunk_text
    from services.vector_store import get_meeting_context_store
    from config import settings

    logger.info(f"[RAG] Embedding meeting context for recording {recording_id}")

    # Sync helper to fetch attachments via async DB
    async def _fetch_attachments():
        from database import get_db
        from sqlalchemy import text
        async with get_db() as db:
            r = await db.execute(
                text(
                    "SELECT filename, file_path, file_hash FROM recording_attachments "
                    "WHERE recording_id = :rid AND user_id = :uid AND type = 'context' "
                    "ORDER BY created_at ASC"
                ),
                {"rid": recording_id, "uid": user_id},
            )
            return r.mappings().fetchall()

    try:
        loop = asyncio.get_event_loop()
        attachments = loop.run_until_complete(_fetch_attachments())
    except RuntimeError:
        # No running loop — create one
        attachments = asyncio.run(_fetch_attachments())

    if not attachments:
        logger.info(f"[RAG] No context attachments for {recording_id}")
        return 0

    embedder = _get_embedder()
    dim = embedder.embedding_dim()
    store = get_meeting_context_store(recording_id, dim)
    store.clear()  # start fresh on re-embed

    total_added = 0
    for att in attachments:
        text_content = extract_text_from_file(att["file_path"], att["filename"])
        if not text_content.strip():
            continue

        chunks = chunk_text(
            text_content,
            chunk_size=settings.RAG_CHUNK_SIZE,
            overlap=settings.RAG_CHUNK_OVERLAP,
        )
        if not chunks:
            continue

        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode_batch(texts)

        metadatas = [
            {
                "recording_id": recording_id,
                "filename": att["filename"],
                "file_hash": att["file_hash"],
                "chunk_index": c["chunk_index"],
                "source": "meeting_context",
            }
            for c in chunks
        ]

        added = store.add(texts, metadatas, embeddings=embeddings)
        total_added += added
        logger.info(f"[RAG] Meeting context: added {added} chunks from {att['filename']}")

    logger.info(f"[RAG] Total meeting context chunks: {total_added}")
    return total_added


# ── 3. Transcript Embedding ───────────────────────────────────────────────────

def embed_transcript(recording_id: str, transcript: List[Dict]) -> int:
    """
    Chunk and embed a meeting transcript into the per-recording transcript FAISS index.

    If embeddings already exist (transcript_embedded=1), this is a no-op unless
    called with force=True. The caller (raw_mom_router) should check the DB flag
    and skip this if already done.

    Returns
    -------
    Number of chunks added.
    """
    from services.text_chunker import chunk_transcript
    from services.vector_store import get_transcript_store
    from config import settings

    logger.info(f"[RAG] Embedding transcript for recording {recording_id}")

    chunks = chunk_transcript(
        transcript,
        chunk_size=settings.RAG_CHUNK_SIZE,
        overlap=settings.RAG_CHUNK_OVERLAP,
    )
    if not chunks:
        logger.warning(f"[RAG] No transcript chunks for recording {recording_id}")
        return 0

    embedder = _get_embedder()
    dim = embedder.embedding_dim()
    store = get_transcript_store(recording_id, dim)
    store.clear()  # start fresh

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_batch(texts)

    metadatas = [
        {
            "recording_id": recording_id,
            "chunk_index": c["chunk_index"],
            "start": c.get("start", 0.0),
            "end": c.get("end", 0.0),
            "speakers": c.get("speakers", []),
            "source": "transcript",
        }
        for c in chunks
    ]

    added = store.add(texts, metadatas, embeddings=embeddings)
    logger.info(f"[RAG] Added {added} transcript chunks for recording {recording_id}")
    return added


# ── 4. Agenda Parsing ─────────────────────────────────────────────────────────

def parse_agenda_items(agenda_text: str) -> List[Dict]:
    """
    Parse raw agenda text into a list of {topic, speaker} dicts.

    Topic is copied VERBATIM. Speaker is the presenter if mentioned, else None.

    Uses the QwenProvider (main LLM) for structured parsing.

    Returns
    -------
    List of dicts: [{"topic": str, "speaker": str|None}]
    """
    from services.ai_provider import get_provider
    provider = get_provider()
    return provider.parse_agenda_items(agenda_text)


# ── 5. Evidence Retrieval ─────────────────────────────────────────────────────

def retrieve_evidence_for_agenda(
    agenda_topic: str,
    recording_id: str,
    user_id: str,
    k_global: Optional[int] = None,
    k_meeting: Optional[int] = None,
    k_transcript: Optional[int] = None,
) -> str:
    """
    Retrieve relevant evidence for one agenda topic from all three FAISS stores.

    Priority order (reflected in output structure):
      1. Transcript — actual discussion (highest priority)
      2. Meeting Context — slides / specs
      3. Global Context — org knowledge

    Parameters
    ----------
    agenda_topic : The agenda item topic (used as retrieval query).
    recording_id : Meeting recording ID.
    user_id      : User ID (for global context lookup).
    k_global     : Override default retrieval K for global context.
    k_meeting    : Override default retrieval K for meeting context.
    k_transcript : Override default retrieval K for transcript.

    Returns
    -------
    Formatted evidence string ready for the LLM extraction prompt.
    Empty string if no evidence found.
    """
    from services.vector_store import (
        get_global_context_store,
        get_meeting_context_store,
        get_transcript_store,
    )
    from config import settings
    import numpy as np

    k_g = k_global or settings.RAG_RETRIEVAL_K_GLOBAL
    k_m = k_meeting or settings.RAG_RETRIEVAL_K_MEETING
    k_t = k_transcript or settings.RAG_RETRIEVAL_K_TRANSCRIPT

    # Detect procedural agenda items to optimize retrieval and prevent OOM/truncation
    topic_lower = agenda_topic.lower()
    procedural_keywords = [
        "call to order", "roll call", "apologies", "adjournment", "adjourn",
        "welcome", "introductions", "approval of minutes", "opening remarks",
        "procedural"
    ]
    is_procedural = any(kw in topic_lower for kw in procedural_keywords)
    if is_procedural:
        k_g = 0
        k_m = 0
        k_t = min(k_t, 3)
        logger.info(
            f"[RAG] Procedural topic detected '{agenda_topic}': "
            f"disabled global/meeting RAG, capped transcript RAG at {k_t}"
        )

    embedder = _get_embedder()
    dim = embedder.embedding_dim()

    query_embedding = embedder.encode(agenda_topic)

    sections = []
    total_chars = 0
    char_limit = 15000  # Strict cap to prevent CUDA OOM on client RTX GPUs

    # ── Transcript (highest priority) ─────────────────────────────────────────
    try:
        t_store = get_transcript_store(recording_id, dim)
        if t_store.exists() and k_t > 0:
            t_results = t_store.search(query_embedding, k=k_t)
            if t_results:
                t_results = _filter_by_relative_similarity(t_results, settings.RAG_RELATIVE_SCORE_CUTOFF, "transcript")
                section_lines = []
                for res in t_results:
                    text = res.get("_text", "").strip()
                    if text:
                        if total_chars + len(text) > char_limit:
                            break
                        section_lines.append(text)
                        total_chars += len(text)
                if section_lines:
                    sections.append("=== TRANSCRIPT (Actual Discussion) ===\n" + "\n".join(section_lines))
                logger.debug(f"[RAG] Transcript: {len(section_lines)} chunks retrieved")
    except Exception as e:
        logger.warning(f"[RAG] Transcript retrieval failed: {e}")

    # ── Meeting Context (medium priority) ─────────────────────────────────────
    try:
        m_store = get_meeting_context_store(recording_id, dim)
        if m_store.exists() and total_chars < char_limit and k_m > 0:
            m_results = m_store.search(query_embedding, k=k_m)
            if m_results:
                m_results = _filter_by_relative_similarity(m_results, settings.RAG_RELATIVE_SCORE_CUTOFF, "meeting")
                section_lines = []
                for res in m_results:
                    text = res.get("_text", "").strip()
                    fname = res.get("filename", "")
                    if text:
                        prefix = f"[{fname}] " if fname else ""
                        formatted_line = f"{prefix}{text}"
                        if total_chars + len(formatted_line) > char_limit:
                            break
                        section_lines.append(formatted_line)
                        total_chars += len(formatted_line)
                if section_lines:
                    sections.append("=== MEETING CONTEXT (Slides / Documents) ===\n" + "\n".join(section_lines))
                logger.debug(f"[RAG] Meeting context: {len(section_lines)} chunks retrieved")
    except Exception as e:
        logger.warning(f"[RAG] Meeting context retrieval failed: {e}")

    # ── Global Context (organizational knowledge) ─────────────────────────────
    try:
        g_store = get_global_context_store(user_id, dim)
        if g_store.exists() and total_chars < char_limit and k_g > 0:
            g_results = g_store.search(query_embedding, k=k_g)
            if g_results:
                g_results = _filter_by_relative_similarity(g_results, settings.RAG_RELATIVE_SCORE_CUTOFF, "global_context")
                section_lines = []
                for res in g_results:
                    text = res.get("_text", "").strip()
                    fname = res.get("filename", "")
                    if text:
                        prefix = f"[{fname}] " if fname else ""
                        formatted_line = f"{prefix}{text}"
                        if total_chars + len(formatted_line) > char_limit:
                            break
                        section_lines.append(formatted_line)
                        total_chars += len(formatted_line)
                if section_lines:
                    sections.append("=== GLOBAL CONTEXT (Organizational Knowledge) ===\n" + "\n".join(section_lines))
                logger.debug(f"[RAG] Global context: {len(section_lines)} chunks retrieved")
    except Exception as e:
        logger.warning(f"[RAG] Global context retrieval failed: {e}")

    if not sections:
        logger.warning(f"[RAG] No evidence retrieved for agenda: {agenda_topic[:60]}")
        return ""

    evidence = "\n\n".join(sections)
    logger.info(
        f"[RAG] Evidence assembled for '{agenda_topic[:50]}': "
        f"{len(evidence)} chars from {len(sections)} source(s)"
    )
    return evidence


def retrieve_evidence_raw(
    agenda_topic: str,
    recording_id: str,
    user_id: str,
    k_global: int,
    k_meeting: int,
    k_transcript: int,
    relative_cutoff: float,
    char_limit: int = 15000,
) -> Dict:
    """
    Retrieve evidence for one agenda topic and return raw chunk dicts with
    full metadata. Unlike retrieve_evidence_for_agenda(), this does NOT
    assemble chunks into a formatted string — callers receive the raw results
    so they can inspect, filter, or reorder them before assembly.

    Returns
    -------
    {
        "transcript": [{"chunk_id": str, "source": "transcript", "score": float,
                         "text": str, "speakers": list, "start": float,
                         "end": float, "word_count": int, "char_count": int}, ...],
        "meeting":    [{"chunk_id": str, "source": "meeting", "score": float,
                         "text": str, "filename": str, "page": int|None,
                         "char_count": int}, ...],
        "global":     [{"chunk_id": str, "source": "global", "score": float,
                         "text": str, "filename": str, "char_count": int}, ...],
        "is_procedural": bool,
    }
    """
    from services.vector_store import (
        get_global_context_store,
        get_meeting_context_store,
        get_transcript_store,
    )
    import numpy as np

    # Detect procedural topics (same logic as retrieve_evidence_for_agenda)
    topic_lower = agenda_topic.lower()
    procedural_keywords = [
        "call to order", "roll call", "apologies", "adjournment", "adjourn",
        "welcome", "introductions", "approval of minutes", "opening remarks",
        "procedural"
    ]
    is_procedural = any(kw in topic_lower for kw in procedural_keywords)
    if is_procedural:
        k_global = 0
        k_meeting = 0
        k_transcript = min(k_transcript, 3)
        logger.info(
            f"[RAG] Procedural topic detected '{agenda_topic}': "
            f"disabled global/meeting RAG, capped transcript RAG at {k_transcript}"
        )

    embedder = _get_embedder()
    dim = embedder.embedding_dim()
    query_embedding = embedder.encode(agenda_topic)

    def _make_chunk_id(source: str, idx: int) -> str:
        return f"{source[0]}-{idx}"

    transcript_chunks = []
    meeting_chunks = []
    global_chunks = []

    # ── Transcript ────────────────────────────────────────────────────────────
    if k_transcript > 0:
        try:
            t_store = get_transcript_store(recording_id, dim)
            if t_store.exists():
                t_results = t_store.search(query_embedding, k=k_transcript)
                if t_results:
                    t_results = _filter_by_relative_similarity(t_results, relative_cutoff, "transcript")
                    for i, res in enumerate(t_results):
                        text = res.get("_text", "").strip()
                        if not text:
                            continue
                        transcript_chunks.append({
                            "chunk_id": _make_chunk_id("transcript", i),
                            "source": "transcript",
                            "score": round(res.get("score", 0.0), 4),
                            "text": text,
                            "speakers": res.get("speakers", []),
                            "start": res.get("start", 0.0),
                            "end": res.get("end", 0.0),
                            "chunk_index": res.get("chunk_index", i),
                            "word_count": len(text.split()),
                            "char_count": len(text),
                        })
        except Exception as e:
            logger.warning(f"[RAG] Transcript raw retrieval failed: {e}")

    # ── Meeting Context ───────────────────────────────────────────────────────
    if k_meeting > 0:
        try:
            m_store = get_meeting_context_store(recording_id, dim)
            if m_store.exists():
                m_results = m_store.search(query_embedding, k=k_meeting)
                if m_results:
                    m_results = _filter_by_relative_similarity(m_results, relative_cutoff, "meeting")
                    for i, res in enumerate(m_results):
                        text = res.get("_text", "").strip()
                        if not text:
                            continue
                        meeting_chunks.append({
                            "chunk_id": _make_chunk_id("meeting", i),
                            "source": "meeting",
                            "score": round(res.get("score", 0.0), 4),
                            "text": text,
                            "filename": res.get("filename", ""),
                            "page": res.get("page"),
                            "chunk_index": res.get("chunk_index", i),
                            "char_count": len(text),
                        })
        except Exception as e:
            logger.warning(f"[RAG] Meeting raw retrieval failed: {e}")

    # ── Global Context ────────────────────────────────────────────────────────
    if k_global > 0:
        try:
            g_store = get_global_context_store(user_id, dim)
            if g_store.exists():
                g_results = g_store.search(query_embedding, k=k_global)
                if g_results:
                    g_results = _filter_by_relative_similarity(g_results, relative_cutoff, "global_context")
                    for i, res in enumerate(g_results):
                        text = res.get("_text", "").strip()
                        if not text:
                            continue
                        global_chunks.append({
                            "chunk_id": _make_chunk_id("global", i),
                            "source": "global",
                            "score": round(res.get("score", 0.0), 4),
                            "text": text,
                            "filename": res.get("filename", ""),
                            "chunk_index": res.get("chunk_index", i),
                            "char_count": len(text),
                        })
        except Exception as e:
            logger.warning(f"[RAG] Global raw retrieval failed: {e}")

    logger.info(
        f"[RAG] Raw retrieval for '{agenda_topic[:50]}': "
        f"{len(transcript_chunks)}T / {len(meeting_chunks)}M / {len(global_chunks)}G chunks"
    )

    return {
        "transcript": transcript_chunks,
        "meeting": meeting_chunks,
        "global": global_chunks,
        "is_procedural": is_procedural,
    }


# ── 6. Full Raw MoM Generation ────────────────────────────────────────────────

def generate_raw_mom(
    recording_id: str,
    user_id: str,
    transcript: List[Dict],
    agenda_text: Optional[str],
    force_reembed_transcript: bool = False,
    force_reembed_meeting: bool = False,
) -> Dict:
    """
    Full Raw MoM generation pipeline for one meeting.

    Steps:
    1. Parse agenda → [{topic, speaker}]
    2. Embed transcript (if not already done)
    3. Embed meeting context attachments (if not already done)
    4. For each agenda item:
       a. Retrieve evidence from all FAISS stores
       b. LLM extraction → structured JSON for this agenda
    5. Assemble final raw_mom JSON

    Parameters
    ----------
    recording_id           : The recording to process.
    user_id                : Owner user ID.
    transcript             : Transcript segment list.
    agenda_text            : Raw text of the agenda document (may be None).
    force_reembed_transcript : Re-embed even if transcript_embedded=1.
    force_reembed_meeting    : Re-embed even if meeting_context_embedded=1.

    Returns
    -------
    {
        "meeting": {
            "agendas": [
                {
                    "agenda_topic": str,
                    "agenda_speaker": str|None,
                    "discussion": [...]
                },
                ...
            ]
        }
    }
    """
    from services.ai_provider import get_provider
    import gc, torch

    logger.info(f"[RAG] Starting Raw MoM generation for recording {recording_id}")

    try:
        # ── Step 1: Parse agenda ───────────────────────────────────────────────────
        agenda_items: List[Dict] = []
        
        # Try loading cached parsed agenda list first
        cached_agenda = _load_parsed_agenda(recording_id)
        if cached_agenda:
            agenda_items = cached_agenda
            logger.info(f"[RAG] Reusing parsed agenda from database ({len(agenda_items)} items)")
        else:
            if agenda_text and agenda_text.strip():
                logger.info(f"[RAG] Parsing agenda ({len(agenda_text)} chars)")
                # Use the LLM to parse the agenda
                provider = get_provider()
                try:
                    agenda_items = provider.parse_agenda_items(agenda_text)
                    if agenda_items:
                        _save_parsed_agenda(recording_id, user_id, agenda_items)
                except Exception as e:
                    logger.warning(f"[RAG] Agenda parsing failed: {e}")
                # Note: Do NOT unload LLM here because we will use it in Step 4 for extraction.

        if not agenda_items:
            # Fallback: create a single "General Meeting" agenda item
            logger.warning("[RAG] No agenda items found; using generic fallback topic")
            agenda_items = [{"topic": "General Meeting Discussion", "speaker": None}]

        logger.info(f"[RAG] Processing {len(agenda_items)} agenda items")

        # ── Step 2: Embed transcript (if needed) ───────────────────────────────────
        if transcript and (force_reembed_transcript or not _transcript_embedded(recording_id)):
            logger.info(f"[RAG] Embedding transcript for {recording_id}")
            try:
                count = embed_transcript(recording_id, transcript)
                if count > 0:
                    _mark_transcript_embedded(recording_id, user_id)
            except Exception as e:
                logger.error(f"[RAG] Transcript embedding failed: {e}", exc_info=True)

        # ── Step 3: Embed meeting context (if needed) ─────────────────────────────
        if force_reembed_meeting or not _meeting_context_embedded(recording_id):
            logger.info(f"[RAG] Embedding meeting context for {recording_id}")
            try:
                count = embed_meeting_context(recording_id, user_id)
                if count > 0:
                    _mark_meeting_context_embedded(recording_id, user_id)
            except Exception as e:
                logger.error(f"[RAG] Meeting context embedding failed: {e}", exc_info=True)

        # ── Step 4: Per-agenda retrieval + extraction ──────────────────────────────
        processed_agendas = []

        for idx, agenda_item in enumerate(agenda_items):
            topic = agenda_item.get("topic", "")
            speaker = agenda_item.get("speaker")

            if not topic:
                continue

            logger.info(f"[RAG] Agenda {idx+1}/{len(agenda_items)}: {topic[:60]}")

            # Retrieve evidence
            try:
                evidence = retrieve_evidence_for_agenda(
                    agenda_topic=topic,
                    recording_id=recording_id,
                    user_id=user_id,
                )
            except Exception as e:
                logger.error(f"[RAG] Evidence retrieval failed for '{topic}': {e}")
                evidence = ""

            # LLM extraction for this agenda item
            provider = get_provider()
            try:
                agenda_result = provider.extract_raw_mom_for_agenda(
                    agenda_topic=topic,
                    agenda_speaker=speaker,
                    evidence=evidence,
                )
                processed_agendas.append(agenda_result)
                logger.info(
                    f"[RAG] Agenda '{topic[:40]}' extracted: "
                    f"{len(agenda_result.get('discussion', []))} discussion entries"
                )
            except Exception as e:
                logger.error(f"[RAG] Extraction failed for '{topic}': {e}")
                processed_agendas.append({
                    "agenda_topic": topic,
                    "agenda_speaker": speaker,
                    "discussion": [],
                })
            finally:
                # We no longer unload the LLM after each agenda item to avoid
                # slow load/unload cycles. The model will be kept in memory
                # and unloaded exactly once at the end of the pipeline.
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        raw_mom = {
            "meeting": {
                "agendas": processed_agendas,
            }
        }
        logger.info(
            f"[RAG] Raw MoM extraction finished for recording={recording_id} "
            f"({len(processed_agendas)} agendas)"
        )
        return raw_mom

    finally:
        # ── Step 5: Unload all models to release GPU memory ──────────────
        try:
            from services.ai_provider import QwenProvider
            QwenProvider.unload_model()
        except Exception as e:
            logger.warning(f"[RAG] Failed to unload LLM: {e}")

        try:
            from services.text_embedding_service import unload_text_embedder
            unload_text_embedder()
        except Exception as e:
            logger.warning(f"[RAG] Failed to unload text embedder: {e}")

        logger.info(f"[RAG] GPU cleanup finished for recording={recording_id}")


# ── DB flag helpers (async-to-sync wrappers) ──────────────────────────────────

def _transcript_embedded(recording_id: str) -> bool:
    """Check if transcript has been embedded (sync wrapper)."""
    import asyncio
    async def _check():
        from database import get_db
        from sqlalchemy import text
        async with get_db() as db:
            r = await db.execute(
                text("SELECT transcript_embedded FROM recordings WHERE id = :id"),
                {"id": recording_id},
            )
            row = r.fetchone()
            return row and row[0] == 1
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context — cannot block; assume not embedded
            return False
        return loop.run_until_complete(_check())
    except Exception:
        return False


def _meeting_context_embedded(recording_id: str) -> bool:
    """Check if meeting context has been embedded (sync wrapper)."""
    import asyncio
    async def _check():
        from database import get_db
        from sqlalchemy import text
        async with get_db() as db:
            r = await db.execute(
                text("SELECT meeting_context_embedded FROM recordings WHERE id = :id"),
                {"id": recording_id},
            )
            row = r.fetchone()
            return row and row[0] == 1
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return False
        return loop.run_until_complete(_check())
    except Exception:
        return False


def _mark_transcript_embedded(recording_id: str, user_id: str) -> None:
    """Mark transcript as embedded in DB (sync wrapper)."""
    import asyncio
    async def _mark():
        from database import get_db
        from sqlalchemy import text
        async with get_db() as db:
            await db.execute(
                text("UPDATE recordings SET transcript_embedded = 1 WHERE id = :id AND user_id = :uid"),
                {"id": recording_id, "uid": user_id},
            )
            await db.commit()
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            loop.run_until_complete(_mark())
    except Exception as e:
        logger.warning(f"[RAG] Could not mark transcript_embedded: {e}")


def _mark_meeting_context_embedded(recording_id: str, user_id: str) -> None:
    """Mark meeting context as embedded in DB (sync wrapper)."""
    import asyncio
    async def _mark():
        from database import get_db
        from sqlalchemy import text
        async with get_db() as db:
            await db.execute(
                text("UPDATE recordings SET meeting_context_embedded = 1 WHERE id = :id AND user_id = :uid"),
                {"id": recording_id, "uid": user_id},
            )
            await db.commit()
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            loop.run_until_complete(_mark())
    except Exception as e:
        logger.warning(f"[RAG] Could not mark meeting_context_embedded: {e}")


def _load_parsed_agenda(recording_id: str) -> Optional[List[Dict]]:
    """Load cached parsed agenda from the DB (sync wrapper)."""
    import asyncio
    import json
    async def _load():
        from database import get_db
        from sqlalchemy import text
        async with get_db() as db:
            r = await db.execute(
                text("SELECT parsed_agenda_json FROM recordings WHERE id = :id"),
                {"id": recording_id},
            )
            row = r.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except Exception:
                    pass
            return None
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return None
            return loop.run_until_complete(_load())
        except RuntimeError:
            return asyncio.run(_load())
    except Exception as e:
        logger.warning(f"[RAG] Failed to load parsed agenda: {e}")
        return None


def _save_parsed_agenda(recording_id: str, user_id: str, agenda_items: List[Dict]) -> None:
    """Save parsed agenda to the DB (sync wrapper)."""
    import asyncio
    import json
    async def _save():
        from database import get_db
        from sqlalchemy import text
        async with get_db() as db:
            await db.execute(
                text("UPDATE recordings SET parsed_agenda_json = :json WHERE id = :id AND user_id = :uid"),
                {"json": json.dumps(agenda_items, ensure_ascii=False), "id": recording_id, "uid": user_id},
            )
            await db.commit()
    try:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(_save())
        except RuntimeError:
            asyncio.run(_save())
    except Exception as e:
        logger.warning(f"[RAG] Failed to save parsed agenda: {e}")
