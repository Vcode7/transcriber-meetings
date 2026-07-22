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


def _get_rag_settings(user_id: Optional[str] = None):
    """Retrieve RAG settings synchronously from SQLite or fall back to default settings."""
    from config import settings
    chunk_size = settings.RAG_CHUNK_SIZE
    overlap = settings.RAG_CHUNK_OVERLAP
    k_global = settings.RAG_RETRIEVAL_K_GLOBAL
    k_meeting = settings.RAG_RETRIEVAL_K_MEETING
    k_transcript = settings.RAG_RETRIEVAL_K_TRANSCRIPT
    score_cutoff = settings.RAG_RELATIVE_SCORE_CUTOFF

    import sqlite3
    db_url = getattr(settings, "DATABASE_URL", "")
    db_path = None
    if db_url.startswith("sqlite+aiosqlite:///"):
        db_path = db_url[len("sqlite+aiosqlite:///"):]
    elif db_url.startswith("sqlite:///"):
        db_path = db_url[len("sqlite:///"):]
    elif db_url:
        db_path = db_url

    if db_path:
        try:
            conn = sqlite3.connect(db_path, timeout=5.0)
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    "SELECT rag_chunk_size, rag_chunk_overlap, rag_retrieval_k_global, "
                    "rag_retrieval_k_meeting, rag_retrieval_k_transcript, rag_relative_score_cutoff "
                    "FROM user_settings WHERE user_id = ? LIMIT 1",
                    (user_id,)
                )
            else:
                cursor.execute(
                    "SELECT rag_chunk_size, rag_chunk_overlap, rag_retrieval_k_global, "
                    "rag_retrieval_k_meeting, rag_retrieval_k_transcript, rag_relative_score_cutoff "
                    "FROM user_settings ORDER BY id DESC LIMIT 1"
                )
            row = cursor.fetchone()
            conn.close()
            if row:
                if row[0] is not None: chunk_size = int(row[0])
                if row[1] is not None: overlap = int(row[1])
                if row[2] is not None: k_global = int(row[2])
                if row[3] is not None: k_meeting = int(row[3])
                if row[4] is not None: k_transcript = int(row[4])
                if row[5] is not None: score_cutoff = float(row[5])
        except Exception as e:
            logger.debug(f"[RAG] Failed to read RAG settings from SQLite DB synchronously: {e}")

    return chunk_size, overlap, k_global, k_meeting, k_transcript, score_cutoff


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
    relative_path: Optional[str] = None,
) -> int:
    """
    Extract text from a document, chunk it, embed it, and store in the
    per-user global context FAISS index.

    Parameters
    ----------
    doc_id        : UUID of the global_context_documents record.
    file_path     : Absolute path to the uploaded file.
    filename      : Original filename (used for format detection).
    user_id       : Owner user ID (indexes are per-user).
    relative_path : Preserved relative directory path if imported from folder.

    Returns
    -------
    Number of chunks added to the index.
    """
    from services.doc_extractor import extract_text_from_file
    from services.text_chunker import chunk_text
    from services.vector_store import get_global_context_store
    from config import settings

    logger.info(f"[RAG] Embedding global context doc: {filename} (doc_id={doc_id}, rel_path={relative_path})")

    text = extract_text_from_file(file_path, filename)
    if not text or not text.strip():
        logger.warning(f"[RAG] No text extracted from {filename} — skipping embed")
        return 0

    chunk_size, overlap, _, _, _, _ = _get_rag_settings(user_id)
    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        logger.warning(f"[RAG] No chunks produced for {filename}")
        return 0

    embedder = _get_embedder()
    dim = embedder.embedding_dim()
    store = get_global_context_store(user_id, dim)

    # Remove any previous chunks for this doc (re-embedding on file change)
    store.delete_by_filter("doc_id", doc_id)

    # Term expansion preprocessing before embedding
    async def _fetch_shortcuts():
        from services.dictionary_service import list_shortcuts
        from database import get_db
        async with get_db() as db:
            return await list_shortcuts(db, user_id)
    import asyncio
    try:
        # Use asyncio.run() which always creates a fresh event loop — safe to call
        # from executor threads where the main asyncio loop is already running.
        shortcuts = asyncio.run(_fetch_shortcuts())
    except Exception:
        shortcuts = []

    from services.dictionary_service import expand_terms_in_chunks
    expanded_texts = expand_terms_in_chunks(chunks, shortcuts)

    texts = [c["text"] for c in chunks]
    import numpy as np
    embeddings = embedder.encode_batch(expanded_texts)

    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "relative_path": relative_path or filename,
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

        chunk_size, overlap, _, _, _, _ = _get_rag_settings(user_id)
        chunks = chunk_text(
            text_content,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not chunks:
            continue

        # Term expansion preprocessing before embedding
        async def _fetch_shortcuts():
            from services.dictionary_service import list_shortcuts
            from database import get_db
            async with get_db() as db:
                return await list_shortcuts(db, user_id)
        import asyncio
        try:
            # Use asyncio.run() — safe from executor threads (creates a fresh loop).
            shortcuts = asyncio.run(_fetch_shortcuts())
        except Exception:
            shortcuts = []

        from services.dictionary_service import expand_terms_in_chunks
        expanded_texts = expand_terms_in_chunks(chunks, shortcuts)

        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode_batch(expanded_texts)

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

def embed_transcript(recording_id: str, transcript: List[Dict], user_id: str) -> int:
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

    chunk_size, overlap, _, _, _, _ = _get_rag_settings(user_id)
    chunks = chunk_transcript(
        transcript,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        logger.warning(f"[RAG] No transcript chunks for recording {recording_id}")
        return 0

    embedder = _get_embedder()
    dim = embedder.embedding_dim()
    store = get_transcript_store(recording_id, dim)
    store.clear()  # start fresh

    # Term expansion preprocessing before embedding
    async def _fetch_shortcuts():
        from services.dictionary_service import list_shortcuts
        from database import get_db
        async with get_db() as db:
            return await list_shortcuts(db, user_id)
    import asyncio
    try:
        # Use asyncio.run() — safe from executor threads (creates a fresh loop).
        shortcuts = asyncio.run(_fetch_shortcuts())
    except Exception:
        shortcuts = []

    from services.dictionary_service import expand_terms_in_chunks
    expanded_texts = expand_terms_in_chunks(chunks, shortcuts)

    # Strip timeline headers and speaker labels from the text used to generate embeddings.
    # Chunk text is now formatted as:
    #   [HH:MM:SS - HH:MM:SS] Speaker Name
    #   Transcript text...
    #
    # We keep only the transcript text lines (not the header lines) so that
    # embedding vectors capture semantic content without positional noise.
    import re as _re
    _timeline_header_re = _re.compile(
        r'^\[\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}\]\s*.+$'
    )
    cleaned_expanded_texts = []
    for text in expanded_texts:
        cleaned_lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Skip timeline header lines
            if _timeline_header_re.match(stripped):
                continue
            # Skip legacy "Speaker: text" header lines (backward compat)
            parts = stripped.split(": ", 1)
            if len(parts) > 1 and not stripped.startswith("["):
                cleaned_lines.append(parts[1])
            else:
                cleaned_lines.append(line)
        cleaned_expanded_texts.append("\n".join(cleaned_lines))

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_batch(cleaned_expanded_texts)

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

    _, _, db_k_g, db_k_m, db_k_t, db_score_cutoff = _get_rag_settings(user_id)
    k_g = k_global or db_k_g
    k_m = k_meeting or db_k_m
    k_t = k_transcript or db_k_t

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
                t_results = _filter_by_relative_similarity(t_results, db_score_cutoff, "transcript")
                section_lines = []
                for res in t_results:
                    # Bug fix: VectorStore stores raw text under "_text"; timeline chunks
                    # (built from store metadata in retrieve_transcript_hybrid) use "text".
                    # Support both keys so no chunk is silently dropped.
                    text = (res.get("_text") or res.get("text") or "").strip()
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
                m_results = _filter_by_relative_similarity(m_results, db_score_cutoff, "meeting")
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
                g_results = _filter_by_relative_similarity(g_results, db_score_cutoff, "global_context")
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


def retrieve_transcript_hybrid(
    recording_id: str,
    query_embedding,
    agenda_index: int,
    total_agendas: int,
    recording_duration: float,
    k_transcript: int,
    high_confidence_threshold: float = 0.70,
    timeline_stride: float = 60.0,
    relative_cutoff: float = 0.01,
    retrieve_by_timeline: bool = True,
) -> List[Dict]:
    """
    Hybrid transcript retrieval combining timeline-window and full semantic search.

    Strategy
    --------
    1. Timeline window: compute expected time slice for this agenda item by
       dividing recording duration equally among all agenda items, then expand
       the window by ``timeline_stride`` seconds on each side.  Every stored
       transcript chunk whose [start, end] overlaps the window is included.
    2. Full semantic search: run a FAISS search across all transcript chunks
       with the agenda topic as the query.
    3. Merge: deduplicate by chunk_index; unconditionally include any semantic
       result with score >= high_confidence_threshold regardless of timeline;
       sort final set by start time.

    Parameters
    ----------
    recording_id             : Recording whose transcript store to query.
    query_embedding          : Pre-computed embedding for the agenda topic.
    agenda_index             : 0-based position of this agenda item.
    total_agendas            : Total number of agenda items.
    recording_duration       : Total duration in seconds (0 → skip timeline).
    k_transcript             : Max number of semantic FAISS results to fetch.
    high_confidence_threshold: Score >= this always included even outside window.
    timeline_stride          : Seconds to pad on each side of the timeline window.
    relative_cutoff          : Relative similarity cutoff for semantic results.

    Returns
    -------
    List of chunk dicts with full metadata, sorted by start time.
    """
    from services.vector_store import get_transcript_store

    embedder = _get_embedder()
    dim = embedder.embedding_dim()

    try:
        t_store = get_transcript_store(recording_id, dim)
        if not t_store.exists():
            return []
    except Exception as e:
        logger.warning(f"[RAG] Could not open transcript store: {e}")
        return []

    # ── 1. Timeline window chunks (metadata scan) ──────────────────────────────
    timeline_chunks_by_index: dict = {}
    if retrieve_by_timeline and recording_duration > 0 and total_agendas > 0:
        slot_size = recording_duration / total_agendas
        win_start = max(0.0, agenda_index * slot_size - timeline_stride)
        win_end = min(recording_duration, (agenda_index + 1) * slot_size + timeline_stride)
        logger.debug(
            f"[RAG] Timeline window for agenda {agenda_index}: "
            f"{win_start:.1f}s – {win_end:.1f}s"
        )
        # Walk all metadata to find overlapping chunks (O(n), fast for typical transcripts)
        try:
            t_store.load_or_create()
            for meta_entry in t_store._meta:
                chunk_start = meta_entry.get("start", 0.0) or 0.0
                chunk_end = meta_entry.get("end", 0.0) or 0.0
                # Overlap condition: chunk starts before window ends AND ends after window starts
                if chunk_start <= win_end and chunk_end >= win_start:
                    cidx = meta_entry.get("chunk_index", -1)
                    if cidx >= 0:
                        timeline_chunks_by_index[cidx] = {
                            "chunk_index": cidx,
                            "start": chunk_start,
                            "end": chunk_end,
                            "text": meta_entry.get("_text", ""),
                            "speakers": meta_entry.get("speakers", []),
                            "score": 0.0,  # will be updated if also in semantic results
                            "_from_timeline": True,
                        }
        except Exception as e:
            logger.warning(f"[RAG] Timeline scan failed: {e}")

    # ── 2. Semantic FAISS search ──────────────────────────────────────────────
    semantic_by_index: dict = {}
    if k_transcript > 0:
        try:
            # Fetch more than k to give the relative-cutoff filter room
            raw_results = t_store.search(query_embedding, k=min(k_transcript * 3, 50))
            if raw_results:
                raw_results = _filter_by_relative_similarity(raw_results, relative_cutoff, "transcript")
                for res in raw_results:
                    cidx = res.get("chunk_index", -1)
                    if cidx < 0:
                        continue
                    semantic_by_index[cidx] = {
                        "chunk_index": cidx,
                        "start": res.get("start", 0.0),
                        "end": res.get("end", 0.0),
                        "text": res.get("_text", ""),
                        "speakers": res.get("speakers", []),
                        "score": float(res.get("score", 0.0)),
                        "_from_semantic": True,
                    }
        except Exception as e:
            logger.warning(f"[RAG] Semantic transcript search failed: {e}")

    # ── 3. Filter out timeline duplicates and construct additional transcript evidence ──
    # Sort semantic candidates by score descending to pick top-K highest similarity chunks
    candidates = sorted(
        semantic_by_index.values(),
        key=lambda c: c.get("score", 0.0),
        reverse=True,
    )

    final_semantic: dict = {}
    for chunk in candidates:
        cidx = chunk.get("chunk_index")
        # Exclude chunks already present in the timeline transcript
        if cidx in timeline_chunks_by_index:
            continue

        if k_transcript > 0 and len(final_semantic) >= k_transcript:
            break

        final_semantic[cidx] = chunk

    # Sort chronologically by start time
    result = sorted(final_semantic.values(), key=lambda c: c.get("start", 0.0))

    logger.info(
        f"[RAG] Hybrid transcript retrieval for agenda {agenda_index}: "
        f"Timeline count: {len(timeline_chunks_by_index)}, Semantic count: {len(semantic_by_index)} "
        f"→ {len(result)} additional non-duplicate chunks"
    )
    return result



def retrieve_evidence_raw(
    agenda_topic: str,
    recording_id: str,
    user_id: str,
    k_global: int,
    k_meeting: int,
    k_transcript: int,
    relative_cutoff: float,
    char_limit: int = 15000,
    # ── Hybrid transcript retrieval params ────────────────────────────────────
    agenda_index: int = 0,
    total_agendas: int = 1,
    recording_duration: float = 0.0,
    timeline_stride: float = 60.0,
    high_confidence_threshold: float = 0.70,
    retrieve_by_timeline: bool = True,
) -> Dict:
    """
    Retrieve evidence for one agenda topic and return raw chunk dicts with
    full metadata. Unlike retrieve_evidence_for_agenda(), this does NOT
    assemble chunks into a formatted string — callers receive the raw results
    so they can inspect, filter, or reorder them before assembly.

    Transcript retrieval uses a **hybrid strategy** (timeline window + semantic).
    Meeting and global context retrieval remain purely semantic.

    Returns
    -------
    {
        "transcript": [{chunk_id, source, score, text, speakers, start, end,
                         word_count, char_count, from_timeline, from_semantic}, ...],
        "meeting":    [{chunk_id, source, score, text, filename, page, char_count}, ...],
        "global":     [{chunk_id, source, score, text, filename, char_count}, ...],
        "is_procedural": bool,
    }
    """
    from services.vector_store import (
        get_global_context_store,
        get_meeting_context_store,
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
        # Disable timeline for procedural items — limit to pure semantic
        recording_duration = 0.0
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

    # ── Transcript (hybrid: timeline + semantic) ──────────────────────────────
    if k_transcript > 0:
        hybrid_results = retrieve_transcript_hybrid(
            recording_id=recording_id,
            query_embedding=query_embedding,
            agenda_index=agenda_index,
            total_agendas=total_agendas,
            recording_duration=recording_duration,
            k_transcript=k_transcript,
            high_confidence_threshold=high_confidence_threshold,
            timeline_stride=timeline_stride,
            relative_cutoff=relative_cutoff,
            retrieve_by_timeline=retrieve_by_timeline,
        )
        for i, res in enumerate(hybrid_results):
            text = res.get("text", "").strip()
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
                "from_timeline": res.get("_from_timeline", False),
                "from_semantic": res.get("_from_semantic", False),
            })

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


# ── Chunk-wise Retrieval ──────────────────────────────────────────────────────

def retrieve_evidence_chunkwise(
    recording_id: str,
    user_id: str,
    agenda_items: List[Dict],
    k_global: int,
    k_meeting: int,
    k_transcript: int = 10,
    relative_cutoff: float = 0.05,
    char_limit: int = 15000,
    recording_duration: float = 0.0,
    timeline_stride: float = 60.0,
    high_confidence_threshold: float = 0.70,
    retrieve_by_timeline: bool = False,
    max_overlap_chunks: int = 2,
) -> List[Dict]:
    """
    Chunk-wise retrieval strategy.

    Instead of agendas searching for transcript chunks, each transcript chunk
    is compared against ALL agenda embeddings. The chunk is assigned to the
    best-matching agenda and to every other agenda whose similarity falls within
    the configured Relative Similarity Threshold of the best score.

    This allows overlapping discussions to be preserved naturally.

    Returns
    -------
    A list of agenda result dicts — same structure as agenda-wise retrieval:
    [
        {
            "topic": str,
            "speaker": str|None,
            "is_procedural": bool,
            "transcript_chunks": [...],
            "meeting_chunks": [...],
            "global_chunks": [...],
        },
        ...
    ]
    """
    from services.vector_store import (
        get_global_context_store,
        get_meeting_context_store,
        get_transcript_store,
    )
    import numpy as np

    if not agenda_items:
        return []

    embedder = _get_embedder()
    dim = embedder.embedding_dim()

    # ── 1. Embed all agenda topics ────────────────────────────────────────────
    procedural_keywords = [
        "call to order", "roll call", "apologies", "adjournment", "adjourn",
        "welcome", "introductions", "approval of minutes", "opening remarks",
        "procedural"
    ]

    agenda_embeddings = []
    is_procedural_flags = []
    for item in agenda_items:
        topic = item.get("topic", "")
        emb = embedder.encode(topic)
        agenda_embeddings.append(emb)
        is_proc = any(kw in topic.lower() for kw in procedural_keywords)
        is_procedural_flags.append(is_proc)

    total_agendas = len(agenda_items)

    # ── 2. Initialise per-agenda result buckets ───────────────────────────────
    # transcript_chunks_by_agenda[i] = list of chunk dicts assigned to agenda i
    transcript_chunks_by_agenda: List[List[Dict]] = [[] for _ in range(total_agendas)]
    meeting_chunks_by_agenda: List[List[Dict]] = [[] for _ in range(total_agendas)]
    global_chunks_by_agenda: List[List[Dict]] = [[] for _ in range(total_agendas)]

    def _make_chunk_id(source: str, idx: int) -> str:
        return f"{source[0]}-{idx}"

    # ── 3. Chunk-wise transcript assignment ───────────────────────────────────
    try:
        t_store = get_transcript_store(recording_id, dim)
        if t_store.exists():
            t_store.load_or_create()
            # Iterate every stored transcript chunk.
            # Use enumerate() so faiss_pos tracks the physical FAISS vector position,
            # which always matches the index into _meta but may differ from the logical
            # chunk_index (e.g. after an index rebuild via delete_by_filter).
            for faiss_pos, meta_entry in enumerate(t_store._meta):
                chunk_idx = meta_entry.get("chunk_index", -1)
                text = meta_entry.get("_text", "").strip()
                if not text or chunk_idx < 0:
                    continue

                # Retrieve chunk vector from FAISS by its physical position in the index.
                # faiss_pos (not chunk_idx) is the correct argument to reconstruct().
                try:
                    chunk_vector = t_store._index.reconstruct(faiss_pos)
                except Exception:
                    # Fallback: re-embed the text on any FAISS error
                    chunk_vector = embedder.encode(text)

                # Compute similarity against every agenda embedding
                scores = []
                for agenda_emb in agenda_embeddings:
                    # Cosine similarity via dot product (vectors are normalised)
                    sim = float(np.dot(chunk_vector, agenda_emb) /
                                (np.linalg.norm(chunk_vector) * np.linalg.norm(agenda_emb) + 1e-10))
                    scores.append(sim)

                best_score = max(scores)
                threshold = best_score - relative_cutoff

                chunk_start = meta_entry.get("start", 0.0) or 0.0
                chunk_end = meta_entry.get("end", 0.0) or 0.0
                chunk_speakers = meta_entry.get("speakers", [])

                qualifying_agendas = []
                for agenda_idx, score in enumerate(scores):
                    # Skip procedural agendas entirely
                    if is_procedural_flags[agenda_idx]:
                        continue
                    if score >= threshold:
                        qualifying_agendas.append((agenda_idx, score))

                # Sort qualifying agendas by score descending
                qualifying_agendas.sort(key=lambda x: x[1], reverse=True)

                # Cap to max_overlap_chunks if configured (> 0)
                if max_overlap_chunks > 0:
                    qualifying_agendas = qualifying_agendas[:max_overlap_chunks]

                for agenda_idx, score in qualifying_agendas:
                    transcript_chunks_by_agenda[agenda_idx].append({
                        "chunk_id": _make_chunk_id("transcript", chunk_idx),
                        "source": "transcript",
                        "score": round(score, 4),
                        "text": text,
                        "speakers": chunk_speakers,
                        "start": chunk_start,
                        "end": chunk_end,
                        "chunk_index": chunk_idx,
                        "word_count": len(text.split()),
                        "char_count": len(text),
                        "from_timeline": False,
                        "from_semantic": True,
                    })
    except Exception as e:
        logger.warning(f"[RAG] Chunk-wise transcript retrieval failed: {e}")

    # Sort each agenda's transcript chunks by score descending, then cap to k_transcript
    for idx in range(total_agendas):
        chunks = transcript_chunks_by_agenda[idx]
        # Sort by score desc, then by start time asc
        chunks.sort(key=lambda c: (-c["score"], c["start"]))
        if k_transcript > 0:
            chunks = chunks[:k_transcript]
        transcript_chunks_by_agenda[idx] = chunks

    # ── 4. Meeting context — per-agenda semantic search (unchanged) ───────────
    if k_meeting > 0:
        for agenda_idx, item in enumerate(agenda_items):
            if is_procedural_flags[agenda_idx]:
                continue
            topic = item.get("topic", "")
            query_emb = agenda_embeddings[agenda_idx]
            try:
                m_store = get_meeting_context_store(recording_id, dim)
                if m_store.exists():
                    m_results = m_store.search(query_emb, k=k_meeting)
                    if m_results:
                        m_results = _filter_by_relative_similarity(m_results, relative_cutoff, "meeting")
                        for i, res in enumerate(m_results):
                            text = res.get("_text", "").strip()
                            if not text:
                                continue
                            meeting_chunks_by_agenda[agenda_idx].append({
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
                logger.warning(f"[RAG] Chunk-wise meeting retrieval failed for agenda {agenda_idx}: {e}")

    # ── 5. Global context — per-agenda semantic search (unchanged) ────────────
    if k_global > 0:
        for agenda_idx, item in enumerate(agenda_items):
            if is_procedural_flags[agenda_idx]:
                continue
            query_emb = agenda_embeddings[agenda_idx]
            try:
                g_store = get_global_context_store(user_id, dim)
                if g_store.exists():
                    g_results = g_store.search(query_emb, k=k_global)
                    if g_results:
                        g_results = _filter_by_relative_similarity(g_results, relative_cutoff, "global_context")
                        for i, res in enumerate(g_results):
                            text = res.get("_text", "").strip()
                            if not text:
                                continue
                            global_chunks_by_agenda[agenda_idx].append({
                                "chunk_id": _make_chunk_id("global", i),
                                "source": "global",
                                "score": round(res.get("score", 0.0), 4),
                                "text": text,
                                "filename": res.get("filename", ""),
                                "chunk_index": res.get("chunk_index", i),
                                "char_count": len(text),
                            })
            except Exception as e:
                logger.warning(f"[RAG] Chunk-wise global retrieval failed for agenda {agenda_idx}: {e}")

    # ── 6. Assemble final result list ─────────────────────────────────────────
    results = []
    for agenda_idx, item in enumerate(agenda_items):
        topic = item.get("topic", "")
        t_chunks = transcript_chunks_by_agenda[agenda_idx]
        m_chunks = meeting_chunks_by_agenda[agenda_idx]
        g_chunks = global_chunks_by_agenda[agenda_idx]
        logger.info(
            f"[RAG] Chunk-wise result for '{topic[:50]}': "
            f"{len(t_chunks)}T / {len(m_chunks)}M / {len(g_chunks)}G chunks"
        )
        results.append({
            "topic": topic,
            "speaker": item.get("speaker"),
            "is_procedural": is_procedural_flags[agenda_idx],
            "transcript_chunks": t_chunks,
            "meeting_chunks": m_chunks,
            "global_chunks": g_chunks,
        })

    return results




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
        # ── Step 1: Get/create agenda items ────────────────────────────────────────
        agenda_items = get_or_create_agenda_items(
            recording_id=recording_id,
            user_id=user_id,
            transcript=transcript,
            agenda_text=agenda_text,
        )

        logger.info(f"[RAG] Processing {len(agenda_items)} agenda items")

        # ── Step 2: Embed transcript (if needed) ───────────────────────────────────
        if transcript and (force_reembed_transcript or not _transcript_embedded(recording_id)):
            logger.info(f"[RAG] Embedding transcript for {recording_id}")
            try:
                count = embed_transcript(recording_id, transcript, user_id)
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
    """Check if transcript has been embedded (sync wrapper).

    Uses asyncio.run() which creates a brand-new event loop — safe to call from
    FastAPI's thread executor where the main loop is already running.
    """
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
            return bool(row and row[0] == 1)
    try:
        return asyncio.run(_check())
    except Exception:
        return False


def _meeting_context_embedded(recording_id: str) -> bool:
    """Check if meeting context has been embedded (sync wrapper).

    Uses asyncio.run() which creates a brand-new event loop — safe to call from
    FastAPI's thread executor where the main loop is already running.
    """
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
            return bool(row and row[0] == 1)
    try:
        return asyncio.run(_check())
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
        asyncio.run(_mark())
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
        asyncio.run(_mark())
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


def _load_context_summary(recording_id: str) -> Optional[str]:
    """Load cached context_summary from the DB (sync wrapper)."""
    import asyncio
    async def _load():
        from database import get_db
        from sqlalchemy import text
        async with get_db() as db:
            r = await db.execute(
                text("SELECT context_summary FROM recordings WHERE id = :id"),
                {"id": recording_id},
            )
            row = r.fetchone()
            if row and row[0]:
                return row[0].strip()
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
        logger.warning(f"[RAG] Failed to load context_summary: {e}")
        return None


def _retrieve_global_context_for_agenda(agenda_text: str, user_id: str, k: int = 4) -> str:
    """
    Retrieve the top-k most relevant Global Context chunks for the uploaded agenda text.

    This is used during agenda creation to give the LLM background knowledge
    (terminology, abbreviations, project names, org context) WITHOUT letting
    it alter the agenda structure.

    Returns
    -------
    Formatted string of global context chunks, or empty string if none found.
    """
    if not agenda_text or not agenda_text.strip():
        return ""
    if not user_id:
        return ""

    try:
        from services.vector_store import get_global_context_store

        embedder = _get_embedder()
        dim = embedder.embedding_dim()

        g_store = get_global_context_store(user_id, dim)
        if not g_store.exists():
            logger.info("[RAG] No global context store found for user — skipping agenda context retrieval")
            return ""

        # Embed the agenda text to use as the retrieval query
        query_emb = embedder.encode(agenda_text.strip()[:2000])  # cap to avoid excessive token usage

        results = g_store.search(query_emb, k=k)
        if not results:
            return ""

        lines = []
        for res in results:
            text = res.get("_text", "").strip()
            fname = res.get("filename", "")
            if text:
                prefix = f"[{fname}] " if fname else ""
                lines.append(f"{prefix}{text}")

        global_ctx = "\n\n".join(lines)
        logger.info(
            f"[RAG] Retrieved {len(lines)} global context chunks for agenda creation "
            f"({len(global_ctx)} chars)"
        )
        return global_ctx

    except Exception as e:
        logger.warning(f"[RAG] Global context retrieval for agenda creation failed: {e}")
        return ""


def get_or_create_agenda_items(
    recording_id: str,
    user_id: str,
    transcript: List[Dict],
    agenda_text: Optional[str],
) -> List[Dict]:
    """
    Acquire agenda items for a recording:
    1. Load cached parsed agenda from DB.
    2. If not found and raw agenda text exists, parse it using LLM.
       - Also retrieves 3-4 Global Context chunks to improve LLM understanding
         of terminology/abbreviations WITHOUT allowing agenda structure changes.
    3. If still not found, generate agenda from the recording's context_summary (if it exists).
    4. Fallback to generic general meeting topic if all else fails.
    """
    from services.ai_provider import get_provider

    # 1. Try loading cached parsed agenda list first
    agenda_items = _load_parsed_agenda(recording_id)
    if agenda_items:
        logger.info(f"[RAG] Reusing parsed agenda from database ({len(agenda_items)} items)")
        return agenda_items

    # 2. Parse raw agenda text if provided
    if agenda_text and agenda_text.strip():
        logger.info(f"[RAG] Parsing agenda document ({len(agenda_text)} chars)")
        provider = get_provider()
        try:
            # Retrieve global context to improve LLM understanding of the agenda
            global_context = _retrieve_global_context_for_agenda(agenda_text, user_id, k=4)

            if global_context:
                logger.info("[RAG] Using global context to enhance agenda parsing")
                agenda_items = provider.parse_agenda_items_with_context(agenda_text, global_context)
            else:
                agenda_items = provider.parse_agenda_items(agenda_text)

            if agenda_items:
                _save_parsed_agenda(recording_id, user_id, agenda_items)
                return agenda_items
        except Exception as e:
            logger.warning(f"[RAG] Agenda parsing failed: {e}")

    # 3. No agenda document. Try generating agenda from context_summary
    logger.info("[RAG] No agenda document found; trying to generate agenda from transcription summary")
    provider = get_provider()
    context_summary = _load_context_summary(recording_id)

    # 4. Generate agenda from summary
    if context_summary:
        try:
            agenda_items = provider.generate_agenda_from_summary(context_summary)
            if agenda_items:
                _save_parsed_agenda(recording_id, user_id, agenda_items)
                logger.info(f"[RAG] Reconstructed {len(agenda_items)} agenda items from transcription summary")
                return agenda_items
        except Exception as e:
            logger.warning(f"[RAG] Failed to generate agenda from summary: {e}")

    # 5. Fallback
    logger.warning("[RAG] No agenda could be parsed or generated; using generic fallback topic")
    return [{"topic": "General Meeting Discussion", "speaker": None}]

