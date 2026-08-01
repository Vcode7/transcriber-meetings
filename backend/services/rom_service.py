import logging
import json
import uuid
import re
import gc
from typing import List, Dict, Optional, Any
import numpy as np

import math

logger = logging.getLogger(__name__)

def clean_calendar_dates(dates_list) -> list[str]:
    """
    Filter and sanitize dates array to ensure ONLY actual calendar dates remain.
    Strips out transcript timestamps, float offsets, audio timelines, and window ranges.
    """
    if not dates_list:
        return []

    if isinstance(dates_list, str):
        dates_list = [dates_list]

    clean_dates = []
    for d in dates_list:
        if isinstance(d, dict):
            val = str(d.get("value") or d.get("date") or "").strip()
        else:
            val = str(d).strip()

        if not val:
            continue

        if re.search(r'^\d+(\.\d+)?\s*[\-–—]\s*\d+(\.\d+)?$', val):
            continue
        if re.search(r'^\d{1,2}:\d{2}(:\d{2})?\s*[\-–—]\s*\d{1,2}:\d{2}(:\d{2})?$', val):
            continue
        if re.search(r'^\d{4,}\.\d+$', val) or re.search(r'^\d+\.\d{2,}$', val):
            continue
        if re.search(r'\b(timeline|window|timestamp|seconds?|offset)\b', val, re.IGNORECASE):
            continue

        has_month = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b', val, re.IGNORECASE)
        has_year = re.search(r'\b(19|20)\d{2}\b', val)
        has_date_fmt = re.search(r'\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}\b', val)
        has_day_spec = re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|yesterday|next week|end of month|q[1-4])\b', val, re.IGNORECASE)

        if has_month or has_year or has_date_fmt or has_day_spec or len(val) >= 4:
            if not re.match(r'^[\d\s.:\-–—]+$', val) or has_date_fmt or has_year:
                clean_dates.append(val)

    return clean_dates

def _extract_chunk_text(res: Any) -> str:
    if isinstance(res, dict):
        txt = (
            res.get("text") or res.get("content") or res.get("chunk") or
            res.get("page_content") or res.get("snippet") or res.get("raw_text") or ""
        )
    else:
        txt = (
            getattr(res, "text", None) or getattr(res, "content", None) or
            getattr(res, "chunk", None) or getattr(res, "page_content", None) or ""
        )
    return str(txt).strip()

def _format_time_hhmm(seconds: float) -> str:
    """Format seconds into HH:MM (or MM:SS if under 1 hour)."""
    total_sec = int(round(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def _validate_user_id(user_obj: Any) -> str:
    """
    Validates and extracts a primitive string user_id from any incoming value
    (string, user dict, Pydantic model, or user instance).
    Defends SQLite and FAISS stores against dict parameter binding errors.
    """
    if isinstance(user_obj, str):
        uid = user_obj.strip()
    elif isinstance(user_obj, dict):
        uid = user_obj.get("id") or user_obj.get("user_id") or user_obj.get("sub")
        if isinstance(uid, str):
            uid = uid.strip()
    elif hasattr(user_obj, "id"):
        uid = getattr(user_obj, "id")
        if isinstance(uid, str):
            uid = uid.strip()
    else:
        uid = None

    if not uid or not isinstance(uid, str):
        logger.error(f"[RomService] Invalid user_id parameter: type={type(user_obj).__name__}, value={user_obj}")
        raise ValueError(f"Invalid user_id parameter: expected string UUID, got {type(user_obj).__name__}")
    return uid


class RomService:
    def extract_discussion_points(
        self,
        transcript: List[Dict],
        window_minutes: float = 2.0,
        user_id: str = None,
        video_transcript: Optional[List[Dict]] = None,
        source_type: str = "audio",
    ) -> Dict:
        """Stage 1: Process transcript in sliding windows to extract discussion points.
        
        Args:
            transcript:       List of diarized transcript segments.
            window_minutes:   Transcript window size in minutes.
            user_id:          Authenticated user ID (for logging/validation).
            video_transcript: Optional merged OCR timeline [{start, end, text}].
                              When provided (source_type='video'), each window is
                              augmented with overlapping OCR blocks before the LLM call.
            source_type:      'audio' or 'video'. Controls whether OCR context is injected.
        """
        from services.ai_provider import get_provider
        from services.video_processing_service import get_overlapping_ocr_blocks
        
        if user_id:
            user_id = _validate_user_id(user_id)
            
        if not transcript:
            return {"discussion_points": [], "windows_processed": 0}
            
        provider = get_provider()
        all_points = []
        
        # 1. Parse transcript segments with start/end times
        # 2. Create windows - group segments by time, never splitting a segment
        windows = []
        current_window = []
        current_start_time = transcript[0].get('start', 0.0)
        window_seconds = window_minutes * 60.0
        
        for segment in transcript:
            seg_start = segment.get('start', 0.0)
            if current_window and (seg_start - current_start_time) >= window_seconds:
                windows.append(current_window)
                current_window = [segment]
                current_start_time = seg_start
            else:
                current_window.append(segment)
                
        if current_window:
            windows.append(current_window)
            
        previous_points_json = None
        total_windows = len(windows)
        has_video_ocr = bool(video_transcript)
        logger.info(
            f"[ROM Service] Stage 1 starting: {total_windows} time window(s) to process"
            f" | video_ocr={'enabled' if has_video_ocr else 'disabled'}"
        )
        
        try:
            for i, window in enumerate(windows):
                w_start = round(window[0].get('start', 0.0), 2) if window else 0.0
                w_end = round(window[-1].get('end', 0.0), 2) if window else 0.0
                
                t_range = f"{_format_time_hhmm(w_start)}-{_format_time_hhmm(w_end)}"
                logger.info(f"[ROM Service] Processing Window {i+1}/{total_windows} ({t_range})")

                window_text = ""
                for seg in window:
                    speaker = seg.get('speaker', 'Unknown')
                    start = seg.get('start', 0.0)
                    end = seg.get('end', 0.0)
                    text = seg.get('text', '').strip()
                    window_text += f"[{start:.1f}-{end:.1f}] {speaker}: {text}\n"

                # ── Video OCR context for this window ────────────────────────────
                video_context = ""
                if has_video_ocr:
                    relevant_blocks = get_overlapping_ocr_blocks(video_transcript, w_start, w_end)
                    if relevant_blocks:
                        lines = []
                        for b in relevant_blocks:
                            ts = f"{_format_time_hhmm(b['start'])} \u2192 {_format_time_hhmm(b['end'])}"
                            lines.append(f"[{ts}]\n{b['text']}")
                        video_context = "\n\n".join(lines)
                        logger.debug(
                            f"[ROM Service] Window {i+1}: {len(relevant_blocks)} OCR block(s) injected"
                        )
                
                result = provider.extract_rom_discussion_points(
                    window_text,
                    previous_points_json,
                    video_context=video_context,
                )
                points = result.get("discussion_points", [])
                
                # Assign UUIDs, window indices, fixed window timelines, and raw transcript text
                for p in points:
                    p["id"] = str(uuid.uuid4())
                    p["window_index"] = i
                    p["timeline_start"] = w_start
                    p["timeline_end"] = w_end
                    p["raw_transcript_text"] = window_text.strip()
                    # Stage 1 document: store which OCR blocks were supplied for this window
                    p["video_transcript_context"] = video_context
                    all_points.append(p)
                    
                # Update previous points for context (only keep latest few points to avoid explosion)
                if points:
                    recent_points = all_points[-5:]
                    context_points = [{"discussion_point": p.get("discussion_point", ""), "technical_terms": p.get("technical_terms", [])} for p in recent_points]
                    previous_points_json = json.dumps(context_points)
                    
            logger.info(f"[ROM Service] Stage 1 complete: {len(all_points)} discussion points extracted across {total_windows} window(s)")
            return {
                "discussion_points": all_points,
                "windows_processed": len(windows)
            }
        finally:
            provider.unload_model()
            gc.collect()


    def enhance_discussion_points(
        self,
        discussion_points: List[Dict],
        recording_id: str,
        user_id: str,
        meeting_top_k: int = 5,
        global_top_k: int = 3,
        discussion_window_size: int = 5,
    ) -> List[Dict]:
        """
        Stage 2: Enhance discussion points using hybrid retrieval (Semantic + BM25 + Metadata)
        with Reciprocal Rank Fusion and independent Meeting / Global context sources.

        Pipeline per window
        -------------------
        1. Combine all point texts in the window → single retrieval query (no LLM rewrite)
        2. Independent retrieval from Meeting Context:
             a. Semantic   – FAISS cosine similarity
             b. Keyword    – BM25+ over metadata sidecar JSON
             c. Metadata   – keyword/entity/acronym field overlap scoring
        3. Same three-way retrieval from Global Context
        4. Merge per source with Reciprocal Rank Fusion (RRF, k=60)
        5. Diversity deduplication (cosine sim > 0.92 → drop lower-ranked duplicate)
        6. Single LLM call per window with both context blocks
        7. Post-process: UUID, timeline merge, speaker merge, date sanitisation
        8. Final semantic deduplication across all enhanced points (≥ 0.90 threshold)
        """
        from services.ai_provider import get_provider
        from services.text_embedding_service import get_text_embedder, unload_text_embedder
        from services.vector_store import get_meeting_context_store, get_global_context_store
        from services.bm25 import BM25Index, _tokenize

        user_id = _validate_user_id(user_id)

        if not discussion_points:
            return []

        provider = get_provider()
        embedder = get_text_embedder()
        embedder.load()

        dim = getattr(embedder, "_dim", 1024)
        meeting_store = get_meeting_context_store(recording_id, dim)
        global_store = get_global_context_store(user_id, dim)

        # ── Build BM25 indexes lazily from the metadata sidecars ──────────────
        def _build_bm25(store) -> Optional[BM25Index]:
            """Build a BM25 index over the FAISS store's metadata sidecar."""
            try:
                store.load_or_create()
                metas = store._meta  # list of dicts with _text key
                if not metas:
                    return None
                texts = [m.get("_text") or m.get("text") or "" for m in metas]
                return BM25Index.from_documents(texts, metas)
            except Exception as e:
                logger.warning(f"[ROM Service] BM25 index build failed: {e}")
                return None

        meeting_bm25 = _build_bm25(meeting_store)
        global_bm25 = _build_bm25(global_store)

        # ── RRF merge helper ──────────────────────────────────────────────────

        def _rrf_merge(
            result_lists: List[List[Dict]],
            k: int = 60,
        ) -> List[Dict]:
            """
            Reciprocal Rank Fusion over multiple ranked result lists.
            Deduplicates by '_text' content; returns merged list sorted by descending RRF score.
            """
            rrf_scores: Dict[str, float] = {}
            best_entry: Dict[str, Dict] = {}

            for ranked in result_lists:
                for rank, entry in enumerate(ranked, start=1):
                    key = (entry.get("_text") or entry.get("text") or "").strip()
                    if not key:
                        continue
                    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                    if key not in best_entry or entry.get("score", 0) > best_entry[key].get("score", 0):
                        best_entry[key] = entry

            merged = sorted(
                best_entry.values(),
                key=lambda e: rrf_scores.get((e.get("_text") or e.get("text") or "").strip(), 0.0),
                reverse=True,
            )
            return merged

        # ── Diversity deduplication (cosine sim ≥ 0.92 → drop) ──────────────

        def _diversity_dedup(
            results: List[Dict],
            sim_threshold: float = 0.92,
        ) -> List[Dict]:
            """
            Remove chunks that are nearly identical to a higher-ranked chunk
            by comparing embedded texts (cosine similarity).  If the embedder
            is unavailable, falls back to exact-text deduplication.
            """
            if len(results) <= 1:
                return results
            try:
                texts = [r.get("_text") or r.get("text") or "" for r in results]
                vecs = embedder.encode(texts)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1.0, norms)
                vecs = vecs / norms

                kept = []
                for i, (res, vec) in enumerate(zip(results, vecs)):
                    is_dup = False
                    for j in kept:
                        if np.dot(vec, vecs[j]) >= sim_threshold:
                            is_dup = True
                            break
                    if not is_dup:
                        kept.append(i)
                return [results[i] for i in kept]
            except Exception:
                # Fallback: deduplicate by exact text
                seen: set = set()
                deduped = []
                for r in results:
                    txt = (r.get("_text") or r.get("text") or "").strip()
                    if txt and txt not in seen:
                        seen.add(txt)
                        deduped.append(r)
                return deduped

        # ── Retrieval helper for one store ────────────────────────────────────

        def _hybrid_retrieve(
            query: str,
            query_vec: np.ndarray,
            store,
            bm25_idx: Optional[BM25Index],
            top_k: int,
        ) -> List[Dict]:
            """
            Run semantic + keyword + metadata retrieval for a single store,
            fuse with RRF, apply diversity dedup, and return top_k results.
            """
            fetch_k = max(top_k * 3, 15)  # over-fetch for RRF headroom

            # (a) Semantic
            try:
                semantic_results = store.search(query_vec, k=fetch_k)
            except Exception as e:
                logger.warning(f"[ROM Service] Semantic search failed: {e}")
                semantic_results = []

            # (b) Keyword (BM25)
            keyword_results: List[Dict] = []
            if bm25_idx is not None:
                try:
                    keyword_results = bm25_idx.search(query, k=fetch_k)
                except Exception as e:
                    logger.warning(f"[ROM Service] BM25 search failed: {e}")

            # (c) Metadata field overlap
            metadata_results: List[Dict] = []
            try:
                query_tokens = _tokenize(query)
                all_metas = store._meta if hasattr(store, "_meta") else []
                metadata_results = BM25Index.metadata_search(
                    query_tokens, all_metas, k=fetch_k
                )
            except Exception as e:
                logger.warning(f"[ROM Service] Metadata search failed: {e}")

            # RRF merge
            fused = _rrf_merge([semantic_results, keyword_results, metadata_results])

            # Diversity dedup
            diverse = _diversity_dedup(fused)

            return diverse[:top_k]

        # ── Window loop ───────────────────────────────────────────────────────

        total_points = len(discussion_points)
        total_windows = math.ceil(total_points / discussion_window_size) if total_points > 0 else 0
        logger.info(
            f"[ROM Service] Stage 2 starting: {total_points} point(s) across "
            f"{total_windows} window(s) of size {discussion_window_size}"
        )

        polished_points: List[Dict] = []

        try:
            for win_start in range(0, total_points, discussion_window_size):
                window = discussion_points[win_start: win_start + discussion_window_size]
                win_num = (win_start // discussion_window_size) + 1
                logger.info(
                    f"[ROM Service] Processing Window {win_num}/{total_windows} "
                    f"({len(window)} points)"
                )

                # ── 1. Combine window → single query ─────────────────────────
                combined_query_parts = []
                for p in window:
                    pt_text = p.get("discussion_point", "")
                    if p.get("required_information"):
                        pt_text += " " + " ".join(p["required_information"])
                    if p.get("technical_terms"):
                        pt_text += " " + " ".join(p["technical_terms"])
                    combined_query_parts.append(pt_text)

                combined_query = " ".join(combined_query_parts).strip()

                # Encode query once
                query_vecs = embedder.encode_batch([combined_query])
                norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1.0, norms)
                query_vec = (query_vecs / norms)[0]

                # ── 2 & 3. Independent hybrid retrieval ──────────────────────
                meeting_results: List[Dict] = []
                global_results: List[Dict] = []

                if meeting_top_k > 0:
                    meeting_results = _hybrid_retrieve(
                        combined_query, query_vec, meeting_store, meeting_bm25, meeting_top_k
                    )

                if global_top_k > 0:
                    global_results = _hybrid_retrieve(
                        combined_query, query_vec, global_store, global_bm25, global_top_k
                    )

                # ── 4. Build context strings ──────────────────────────────────
                meeting_context_parts: List[str] = []
                meeting_filenames: List[str] = []
                for res in meeting_results:
                    txt = _extract_chunk_text(res)
                    fn = res.get("filename") or res.get("document_name") or "Meeting Document"
                    if txt:
                        meeting_context_parts.append(f"[{fn}]\n{txt}")
                        if fn not in meeting_filenames:
                            meeting_filenames.append(fn)

                global_context_parts: List[str] = []
                global_filenames: List[str] = []
                for res in global_results:
                    txt = _extract_chunk_text(res)
                    fn = res.get("filename") or res.get("document_name") or "Global Document"
                    if txt:
                        global_context_parts.append(f"[{fn}]\n{txt}")
                        if fn not in global_filenames:
                            global_filenames.append(fn)

                meeting_context_str = "\n\n".join(meeting_context_parts)
                global_context_str = "\n\n".join(global_context_parts)

                # ── 5. Prepare window JSON for LLM ───────────────────────────
                window_for_llm = json.dumps(
                    [
                        {
                            "id": p.get("id"),
                            "discussion_point": p.get("discussion_point"),
                            "timeline_start": p.get("timeline_start", 0.0),
                            "timeline_end": p.get("timeline_end", 0.0),
                            "speakers": p.get("speakers", []),
                            "technical_terms": p.get("technical_terms", []),
                            "decisions": p.get("decisions", []),
                            "action_items": p.get("action_items", []),
                        }
                        for p in window
                    ],
                    ensure_ascii=False,
                )

                # ── 6. Single LLM call per window ─────────────────────────────
                result = provider.enhance_rom_discussion_window(
                    window_json=window_for_llm,
                    meeting_context=meeting_context_str,
                    global_context=global_context_str,
                )
                enhanced_batch = result.get("enhanced_points", [])

                # ── 7. Post-process enhanced batch ────────────────────────────
                batch_by_id = {p.get("id"): p for p in window}

                for ep in enhanced_batch:
                    ep["id"] = str(uuid.uuid4())

                    # Resolve original IDs
                    orig_ids = ep.get("original_point_ids")
                    if isinstance(orig_ids, str):
                        orig_ids = [orig_ids]
                    elif not isinstance(orig_ids, list):
                        orig_ids = []
                    if not orig_ids:
                        orig_ids = [p.get("id") for p in window if p.get("id")]

                    matched_points = [batch_by_id[oid] for oid in orig_ids if oid in batch_by_id]
                    if not matched_points:
                        matched_points = window
                        orig_ids = [p.get("id") for p in window if p.get("id")]

                    # Merge timeline and speakers
                    t_start = min(m.get("timeline_start", 0.0) for m in matched_points)
                    t_end = max(m.get("timeline_end", 0.0) for m in matched_points)
                    speakers = list(
                        dict.fromkeys(
                            spk for m in matched_points for spk in m.get("speakers", []) if spk
                        )
                    )

                    ep["original_point_ids"] = orig_ids
                    ep["original_point_id"] = orig_ids[0] if orig_ids else "N/A"
                    ep["timeline_start"] = t_start
                    ep["timeline_end"] = t_end
                    ep["speakers"] = speakers

                    # Map LLM output field name → internal field name
                    if "enhanced_text" in ep and "polished_text" not in ep:
                        ep["polished_text"] = ep.pop("enhanced_text")

                    # Defaults for optional fields
                    ep.setdefault("action_owner", None)
                    ep.setdefault("decisions", [])
                    ep.setdefault("technical_terms", [])
                    ep["dates"] = clean_calendar_dates(ep.get("dates", []))
                    ep.setdefault("numbers", [])
                    ep.setdefault("references", [])
                    ep.setdefault("action_items", [])

                    # Normalise / default context_usage_report
                    cur = ep.get("context_usage_report")
                    if not isinstance(cur, dict):
                        cur = {}
                    ep["context_usage_report"] = {
                        "verified": bool(cur.get("verified", False)),
                        "technical_details_added": bool(cur.get("technical_details_added", False)),
                        "abbreviations_expanded": bool(cur.get("abbreviations_expanded", False)),
                        "references_added": bool(cur.get("references_added", False)),
                        "terminology_clarified": bool(cur.get("terminology_clarified", False)),
                        "no_useful_context": bool(
                            cur.get("no_useful_context", not meeting_context_str and not global_context_str)
                        ),
                        "meeting_context_docs": (
                            cur.get("meeting_context_docs")
                            if isinstance(cur.get("meeting_context_docs"), list)
                            else meeting_filenames
                        ),
                        "global_context_docs": (
                            cur.get("global_context_docs")
                            if isinstance(cur.get("global_context_docs"), list)
                            else global_filenames
                        ),
                    }

                    ep["retrieved_context"] = {
                        "meeting_chunks": [
                            {
                                "text": _extract_chunk_text(r),
                                "score": float(r.get("score", 0.0)),
                                "filename": r.get("filename") or r.get("document_name") or "Meeting Document",
                            }
                            for r in meeting_results
                        ],
                        "global_chunks": [
                            {
                                "text": _extract_chunk_text(r),
                                "score": float(r.get("score", 0.0)),
                                "filename": r.get("filename") or r.get("document_name") or "Global Document",
                            }
                            for r in global_results
                        ],
                        "context_usage_report": ep["context_usage_report"],
                    }

                    polished_points.append(ep)

                logger.info(
                    f"[ROM Service] Window {win_num}/{total_windows} complete: "
                    f"{len(enhanced_batch)} enhanced point(s)"
                )

            # ── Stage 2 Final Semantic Deduplication Step (≥ 0.90 + LLM) ────
            if len(polished_points) > 1:
                try:
                    logger.info("[ROM Service] Stage 2 starting final semantic deduplication check")
                    texts = [p.get("polished_text", "") for p in polished_points]
                    embeddings = embedder.encode(texts)
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-9, 1.0, norms)
                    embeddings = embeddings / norms

                    sim_matrix = np.dot(embeddings, embeddings.T)
                    num_points = len(polished_points)

                    visited = [False] * num_points
                    clusters = []
                    for i in range(num_points):
                        if visited[i]:
                            continue
                        cluster = [i]
                        visited[i] = True
                        for j in range(i + 1, num_points):
                            if not visited[j] and sim_matrix[i, j] >= 0.90:
                                cluster.append(j)
                                visited[j] = True
                        clusters.append(cluster)

                    deduped_points: List[Dict] = []

                    for cluster in clusters:
                        if len(cluster) == 1:
                            deduped_points.append(polished_points[cluster[0]])
                        else:
                            cluster_points = [polished_points[idx] for idx in cluster]
                            logger.info(
                                f"[ROM Service] Deduplicating candidate cluster of {len(cluster_points)} "
                                f"point(s) with similarity >= 0.90 via LLM"
                            )

                            cand_json = json.dumps(
                                [
                                    {
                                        "id": p.get("id"),
                                        "original_point_ids": p.get("original_point_ids", []),
                                        "polished_text": p.get("polished_text", ""),
                                        "speakers": p.get("speakers", []),
                                        "action_owner": p.get("action_owner"),
                                        "decisions": p.get("decisions", []),
                                        "technical_terms": p.get("technical_terms", []),
                                        "dates": p.get("dates", []),
                                        "numbers": p.get("numbers", []),
                                        "action_items": p.get("action_items", []),
                                    }
                                    for p in cluster_points
                                ],
                                ensure_ascii=False,
                            )

                            eval_result = provider.deduplicate_rom_points(cand_json)
                            eval_type = eval_result.get("group_classification", "different")
                            res_points = eval_result.get("result_points", [])

                            if eval_type in ("duplicate", "complementary") and res_points:
                                all_orig_ids = list(
                                    dict.fromkeys(
                                        oid for p in cluster_points for oid in p.get("original_point_ids", [])
                                    )
                                )
                                all_speakers = list(
                                    dict.fromkeys(
                                        spk for p in cluster_points for spk in p.get("speakers", []) if spk
                                    )
                                )
                                min_start = min(p.get("timeline_start", 0.0) for p in cluster_points)
                                max_end = max(p.get("timeline_end", 0.0) for p in cluster_points)

                                for r_pt in res_points:
                                    r_pt["id"] = str(uuid.uuid4())
                                    r_pt["original_point_ids"] = all_orig_ids
                                    r_pt["original_point_id"] = all_orig_ids[0] if all_orig_ids else "N/A"
                                    r_pt["timeline_start"] = min_start
                                    r_pt["timeline_end"] = max_end
                                    r_pt["speakers"] = all_speakers
                                    r_pt["dates"] = clean_calendar_dates(
                                        r_pt.get("dates", []) or [d for p in cluster_points for d in p.get("dates", [])]
                                    )
                                    r_pt.setdefault("action_owner", None)
                                    r_pt.setdefault("decisions", [])
                                    r_pt.setdefault("technical_terms", [])
                                    r_pt.setdefault("numbers", [])
                                    r_pt.setdefault("references", [])
                                    r_pt.setdefault("action_items", [])
                                    # Inherit context from the first cluster member
                                    r_pt["retrieved_context"] = cluster_points[0].get("retrieved_context", {})
                                    r_pt["context_usage_report"] = cluster_points[0].get("context_usage_report", {})
                                    # Ensure polished_text is present
                                    if "polished_text" not in r_pt and "enhanced_text" in r_pt:
                                        r_pt["polished_text"] = r_pt.pop("enhanced_text")
                                    deduped_points.append(r_pt)
                            else:
                                for p in cluster_points:
                                    deduped_points.append(p)

                    logger.info(
                        f"[ROM Service] Stage 2 final deduplication complete: "
                        f"{len(polished_points)} -> {len(deduped_points)} point(s)"
                    )
                    polished_points = deduped_points

                except Exception as e:
                    logger.warning(f"[ROM Service] Stage 2 final deduplication failed: {e}")

            logger.info(
                f"[ROM Service] Stage 2 complete: {len(polished_points)} enhanced point(s) "
                f"from {total_points} original point(s)"
            )
            return polished_points

        finally:
            provider.unload_model()
            unload_text_embedder()
            gc.collect()




    def generate_agendas_and_map(
        self,
        polished_points: List[Dict],
        agenda_text: Optional[str],
        recording_id: str,
        user_id: str,
        meeting_context_top_k: int = 5,
        global_context_top_k: int = 3,
        force_reextract: bool = False,
        transcript: Optional[List[Dict]] = None,
        previous_mom_texts: Optional[List[str]] = None,
        batch_size: int = 20,
    ) -> Dict:
        """
        Stage 3 – 5-Phase Intelligent Pipeline
        ----------------------------------------
        Phase 1  Previous MoM Expansion (only if files uploaded)
        Phase 2  Per-agenda sequential RAG retrieval → ExpandedAgenda representation
        Phase 3  Top-3 candidate retrieval via cosine similarity
        Phase 4  Batch LLM agenda assignment (batch_size pts/call)
                 Fallback: assign to best-scoring candidate, mark as 'probable'
        Phase 5  Agenda grouping → structured output for final MoM generation
        """
        from services.ai_provider import get_provider
        from services.text_embedding_service import get_text_embedder, unload_text_embedder
        from services.vector_store import get_meeting_context_store, get_global_context_store
        from services.bm25 import BM25Index, _tokenize
        from services.rag_pipeline import (
            _load_parsed_agenda,
            _save_parsed_agenda,
            get_or_create_agenda_items,
        )

        user_id = _validate_user_id(user_id)

        if not polished_points:
            return {
                "agendas": [],
                "expanded_agendas": [],
                "candidate_results": [],
                "batch_assignments": [],
                "point_mappings": {},
                "agenda_groups": {},
                "similarity_matrix": [],
            }

        agendas: List[Dict] = []
        provider = get_provider()
        embedder = get_text_embedder()
        embedder.load()

        dim = getattr(embedder, "_dim", 1024)
        meeting_store = get_meeting_context_store(recording_id, dim)
        global_store  = get_global_context_store(user_id, dim)

        # ── Build BM25 indexes ────────────────────────────────────────────────
        def _build_bm25(store) -> Optional[BM25Index]:
            try:
                store.load_or_create()
                metas = store._meta
                if not metas:
                    return None
                texts = [m.get("_text") or m.get("text") or "" for m in metas]
                return BM25Index.from_documents(texts, metas)
            except Exception as e:
                logger.warning(f"[ROM S3] BM25 build failed: {e}")
                return None

        meeting_bm25 = _build_bm25(meeting_store)
        global_bm25  = _build_bm25(global_store)

        # ── RRF merge helper ──────────────────────────────────────────────────
        def _rrf_merge(result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
            rrf_scores: Dict[str, float] = {}
            best_entry: Dict[str, Dict]  = {}
            for ranked in result_lists:
                for rank, entry in enumerate(ranked, start=1):
                    key = (entry.get("_text") or entry.get("text") or "").strip()
                    if not key:
                        continue
                    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                    if key not in best_entry or entry.get("score", 0) > best_entry[key].get("score", 0):
                        best_entry[key] = entry
            return sorted(best_entry.values(),
                          key=lambda e: rrf_scores.get((e.get("_text") or e.get("text") or "").strip(), 0.0),
                          reverse=True)

        # ── Diversity dedup ───────────────────────────────────────────────────
        def _diversity_dedup(results: List[Dict], threshold: float = 0.92) -> List[Dict]:
            if len(results) <= 1:
                return results
            try:
                texts = [r.get("_text") or r.get("text") or "" for r in results]
                vecs  = embedder.encode(texts)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1.0, norms)
                vecs  = vecs / norms
                kept: List[int] = []
                for i, (_, vec) in enumerate(zip(results, vecs)):
                    dup = any(np.dot(vec, vecs[j]) >= threshold for j in kept)
                    if not dup:
                        kept.append(i)
                return [results[i] for i in kept]
            except Exception:
                seen: set = set()
                deduped = []
                for r in results:
                    txt = (r.get("_text") or r.get("text") or "").strip()
                    if txt and txt not in seen:
                        seen.add(txt)
                        deduped.append(r)
                return deduped

        # ── Hybrid retrieval for one store ────────────────────────────────────
        def _hybrid_retrieve(query: str, query_vec: np.ndarray, store, bm25_idx, top_k: int) -> List[Dict]:
            fetch_k = max(top_k * 3, 15)
            try:
                sem = store.search(query_vec, k=fetch_k)
            except Exception:
                sem = []
            kw: List[Dict] = []
            if bm25_idx is not None:
                try:
                    kw = bm25_idx.search(query, k=fetch_k)
                except Exception:
                    pass
            meta: List[Dict] = []
            try:
                tokens   = _tokenize(query)
                all_meta = store._meta if hasattr(store, "_meta") else []
                meta     = BM25Index.metadata_search(tokens, all_meta, k=fetch_k)
            except Exception:
                pass
            fused   = _rrf_merge([sem, kw, meta])
            diverse = _diversity_dedup(fused)
            return diverse[:top_k]

        # ══════════════════════════════════════════════════════════════════════
        # Load / extract agendas (same logic as before)
        # ══════════════════════════════════════════════════════════════════════
        if not force_reextract:
            cached = _load_parsed_agenda(recording_id)
            if cached and isinstance(cached, list) and len(cached) > 0:
                logger.info(f"[ROM S3] Reusing {len(cached)} cached agenda item(s)")
                for i, item in enumerate(cached):
                    topic = item.get("topic") or item.get("title") or f"Agenda Topic {i+1}"
                    desc  = item.get("details") or item.get("description") or ""
                    agendas.append({
                        "agenda_id": item.get("agenda_id") or f"A{i+1}",
                        "title": topic,
                        "description": desc,
                        "speaker": item.get("speaker") or item.get("presenter"),
                        "keywords": item.get("keywords") if isinstance(item.get("keywords"), list) else [k for k in [topic, item.get("speaker")] if k],
                        "related_concepts": item.get("related_concepts") if isinstance(item.get("related_concepts"), list) else [],
                        "alternative_terminology": item.get("alternative_terminology") if isinstance(item.get("alternative_terminology"), list) else [],
                        "expected_themes": item.get("expected_themes") if isinstance(item.get("expected_themes"), list) else [],
                    })

        if not agendas:
            if agenda_text and agenda_text.strip():
                retrieved_context_str = ""
                try:
                    q_vec = embedder.encode_batch([agenda_text.strip()[:1000]])
                    norms = np.linalg.norm(q_vec, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-9, 1.0, norms)
                    q_vec = (q_vec / norms)[0]
                    parts: List[str] = []
                    if meeting_context_top_k > 0:
                        for res in meeting_store.search(q_vec, k=meeting_context_top_k):
                            txt = _extract_chunk_text(res)
                            if txt:
                                parts.append(txt)
                    if global_context_top_k > 0:
                        for res in global_store.search(q_vec, k=global_context_top_k):
                            txt = _extract_chunk_text(res)
                            if txt:
                                parts.append(txt)
                    if parts:
                        retrieved_context_str = "\n\n".join(parts)
                except Exception as e:
                    logger.warning(f"[ROM S3] Agenda-level RAG failed: {e}")

                logger.info("[ROM S3] Generating/refining agenda points via LLM from uploaded agenda text")
                agenda_result = provider.generate_rom_agendas(agenda_text, context=retrieved_context_str)
                raw_agendas   = agenda_result.get("agendas", [])
                if raw_agendas:
                    agendas = raw_agendas
                    parsed_to_save = []
                    for a in agendas:
                        if a.get("title"):
                            parsed_to_save.append({
                                "topic": a.get("title"),
                                "speaker": a.get("speaker") or a.get("presenter"),
                                "details": a.get("description") or a.get("details") or "",
                                "keywords": a.get("keywords") or [],
                                "related_concepts": a.get("related_concepts") or [],
                                "alternative_terminology": a.get("alternative_terminology") or [],
                                "expected_themes": a.get("expected_themes") or [],
                            })
                    if parsed_to_save:
                        _save_parsed_agenda(recording_id, user_id, parsed_to_save)

        if not agendas:
            logger.info("[ROM S3] No user agenda provided. Using default 'General Discussion' container.")
            agendas = [{
                "agenda_id": "A1",
                "title": "General Discussion",
                "description": "General meeting discussion points",
                "keywords": [],
                "related_concepts": [],
                "alternative_terminology": [],
                "expected_themes": [],
                "is_default_agenda": True
            }]

        logger.info(f"[ROM S3] {len(agendas)} agenda(s) loaded. Starting 5-phase pipeline.")

        try:
            # ══════════════════════════════════════════════════════════════════
            # PHASE 1 – Previous MoM Expansion (only when files uploaded)
            # ══════════════════════════════════════════════════════════════════
            previous_mom_expansion: Dict[str, Dict] = {}
            has_previous_mom = bool(previous_mom_texts and any(t.strip() for t in previous_mom_texts))

            if has_previous_mom:
                logger.info("[ROM S3] Phase 1: Expanding agendas with previous MoM documents")
                combined_mom = "\n\n---\n\n".join(t for t in previous_mom_texts if t and t.strip())
                agenda_list_for_expansion = json.dumps(
                    [{"agenda_id": a["agenda_id"], "title": a["title"], "description": a.get("description", "")} for a in agendas],
                    ensure_ascii=False
                )
                expansion_result = provider.expand_agendas_with_previous_mom(agenda_list_for_expansion, combined_mom)
                for item in expansion_result.get("expansions", []):
                    aid = item.get("agenda_id")
                    if aid:
                        previous_mom_expansion[aid] = item
                logger.info(f"[ROM S3] Phase 1 complete: {len(previous_mom_expansion)} agenda(s) expanded from previous MoMs")
            else:
                logger.info("[ROM S3] Phase 1: Skipped (no previous MoM files uploaded)")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 2 – Per-agenda sequential RAG retrieval → ExpandedAgenda
            # ══════════════════════════════════════════════════════════════════
            logger.info("[ROM S3] Phase 2: Per-agenda context expansion via RAG (sequential)")
            expanded_agendas: List[Dict] = []

            for agenda in agendas:
                aid   = agenda["agenda_id"]
                title = agenda.get("title", "")
                desc  = agenda.get("description", "")
                kws   = agenda.get("keywords", [])
                rc    = agenda.get("related_concepts", [])
                alt   = agenda.get("alternative_terminology", [])
                themes = agenda.get("expected_themes", [])

                # Build retrieval query for this agenda
                agenda_query = " ".join(filter(None, [title, desc] + kws + rc + alt + themes))

                meeting_ctx_str = ""
                global_ctx_str  = ""

                if agenda_query.strip():
                    try:
                        q_vecs = embedder.encode_batch([agenda_query[:1000]])
                        norms  = np.linalg.norm(q_vecs, axis=1, keepdims=True)
                        norms  = np.where(norms < 1e-9, 1.0, norms)
                        q_vec  = (q_vecs / norms)[0]

                        if meeting_context_top_k > 0:
                            m_results = _hybrid_retrieve(agenda_query, q_vec, meeting_store, meeting_bm25, meeting_context_top_k)
                            meeting_parts = [_extract_chunk_text(r) for r in m_results if _extract_chunk_text(r)]
                            meeting_ctx_str = "\n\n".join(meeting_parts)

                        if global_context_top_k > 0:
                            g_results = _hybrid_retrieve(agenda_query, q_vec, global_store, global_bm25, global_context_top_k)
                            global_parts = [_extract_chunk_text(r) for r in g_results if _extract_chunk_text(r)]
                            global_ctx_str = "\n\n".join(global_parts)

                    except Exception as e:
                        logger.warning(f"[ROM S3] Phase 2 RAG failed for agenda {aid}: {e}")

                # Previous MoM expansion for this agenda
                prev_expansion = previous_mom_expansion.get(aid, {})
                prev_discussion = prev_expansion.get("previous_discussion", "")

                # Build the embedding text: rich concatenation of all sources
                embedding_text_parts = [title, desc]
                embedding_text_parts.extend(kws)
                embedding_text_parts.extend(rc)
                embedding_text_parts.extend(alt)
                embedding_text_parts.extend(themes)
                if prev_discussion:
                    embedding_text_parts.append(prev_discussion)
                for field in ("previous_decisions", "previous_action_items", "pending_work", "follow_up_items"):
                    vals = prev_expansion.get(field, [])
                    if isinstance(vals, list):
                        embedding_text_parts.extend(vals)
                # Append first 2000 chars of retrieved context
                if meeting_ctx_str:
                    embedding_text_parts.append(meeting_ctx_str[:2000])
                if global_ctx_str:
                    embedding_text_parts.append(global_ctx_str[:1000])

                expanded = {
                    "agenda_id": aid,
                    "title": title,
                    "description": desc,
                    "keywords": kws,
                    "related_concepts": rc,
                    "alternative_terminology": alt,
                    "expected_themes": themes,
                    "previous_meeting_summary": prev_expansion if prev_expansion.get("found_in_previous_meeting") else {},
                    "meeting_context_snippet": meeting_ctx_str[:1500] if meeting_ctx_str else "",
                    "global_context_snippet": global_ctx_str[:1000] if global_ctx_str else "",
                    "_embedding_text": " ".join(p for p in embedding_text_parts if p),
                }
                expanded_agendas.append(expanded)
                logger.info(f"[ROM S3] Phase 2 expanded agenda {aid}: meeting_ctx={len(meeting_ctx_str)} chars, global_ctx={len(global_ctx_str)} chars, prev_mom={'yes' if prev_expansion.get('found_in_previous_meeting') else 'no'}")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 3 – Top-3 Candidate Retrieval via Cosine Similarity
            # ══════════════════════════════════════════════════════════════════
            logger.info("[ROM S3] Phase 3: Computing Top-3 candidate agendas per discussion point")
            expanded_texts = [ea["_embedding_text"] for ea in expanded_agendas]
            point_texts    = [p.get("polished_text", "") for p in polished_points]
            all_texts      = expanded_texts + point_texts

            embeddings = embedder.encode(all_texts)
            norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms      = np.where(norms < 1e-9, 1.0, norms)
            embeddings = embeddings / norms

            agenda_embs = embeddings[:len(expanded_agendas)]
            point_embs  = embeddings[len(expanded_agendas):]

            sim_matrix      = np.dot(point_embs, agenda_embs.T)  # (n_points, n_agendas)
            sim_matrix_list = sim_matrix.tolist()

            n_candidates    = min(3, len(expanded_agendas))
            candidate_results: List[Dict] = []

            for i, point in enumerate(polished_points):
                row         = sim_matrix[i]
                top_indices = np.argsort(row)[::-1][:n_candidates]
                candidates  = [
                    {"agenda_id": expanded_agendas[idx]["agenda_id"],
                     "agenda_title": expanded_agendas[idx]["title"],
                     "score": float(row[idx])}
                    for idx in top_indices
                ]
                candidate_results.append({"point_id": point["id"], "candidates": candidates})

            logger.info(f"[ROM S3] Phase 3 complete: Top-{n_candidates} candidates computed for {len(polished_points)} point(s)")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 4 – Batch LLM Agenda Assignment
            # ══════════════════════════════════════════════════════════════════
            logger.info(f"[ROM S3] Phase 4: Batch LLM agenda assignment (batch_size={batch_size})")
            valid_agenda_ids = {a["agenda_id"] for a in agendas}
            agenda_ref_json  = json.dumps(
                [{"agenda_id": a["agenda_id"], "title": a.get("title", ""), "description": a.get("description", "")} for a in agendas],
                ensure_ascii=False
            )
            # Build point_id → candidate list lookup
            cand_by_point_id: Dict[str, List[Dict]] = {c["point_id"]: c["candidates"] for c in candidate_results}
            batch_assignments: List[Dict] = []
            total_points      = len(polished_points)
            total_batches     = math.ceil(total_points / batch_size)

            for batch_num, batch_start in enumerate(range(0, total_points, batch_size), start=1):
                batch_points = polished_points[batch_start: batch_start + batch_size]
                logger.info(f"[ROM S3] Phase 4 batch {batch_num}/{total_batches}: {len(batch_points)} point(s)")

                batch_items = []
                for point in batch_points:
                    pid        = point["id"]
                    candidates = cand_by_point_id.get(pid, [])
                    batch_items.append({
                        "point_id": pid,
                        "enhanced_point": point.get("polished_text", ""),
                        "timeline": f"{_format_time_hhmm(point.get('timeline_start', 0))} – {_format_time_hhmm(point.get('timeline_end', 0))}",
                        "speakers": point.get("speakers", []),
                        "candidate_agendas": candidates,
                    })

                batch_json_str = json.dumps(batch_items, ensure_ascii=False)
                llm_result     = provider.assign_agenda_batch(batch_json_str, agenda_ref_json)

                # Build a map from the LLM response
                llm_map: Dict[str, Dict] = {a.get("point_id", ""): a for a in llm_result.get("assignments", []) if a.get("point_id")}

                for point in batch_points:
                    pid        = point["id"]
                    candidates = cand_by_point_id.get(pid, [])
                    llm_entry  = llm_map.get(pid, {})

                    assigned    = llm_entry.get("assigned_agenda_id", "")
                    confidence  = llm_entry.get("confidence", "medium")
                    reason      = llm_entry.get("reason", "")
                    is_probable = False

                    # Fallback: if LLM returned unknown/empty ID, use highest-scoring candidate
                    if not assigned or assigned not in valid_agenda_ids:
                        is_probable = True
                        confidence  = "probable"
                        reason      = f"Automatically assigned to best-matching candidate (LLM did not return a valid agenda ID)."
                        assigned    = candidates[0]["agenda_id"] if candidates else (agendas[0]["agenda_id"] if agendas else "A1")

                    batch_assignments.append({
                        "point_id": pid,
                        "assigned_agenda_id": assigned,
                        "confidence": "probable" if is_probable else confidence,
                        "reason": reason,
                        "is_probable": is_probable,
                    })

                logger.info(f"[ROM S3] Phase 4 batch {batch_num}/{total_batches} complete")

            logger.info(f"[ROM S3] Phase 4 complete: {len(batch_assignments)} assignment(s)")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 5 – Agenda Grouping
            # ══════════════════════════════════════════════════════════════════
            logger.info("[ROM S3] Phase 5: Grouping discussion points by assigned agenda")
            agenda_groups: Dict[str, List[str]] = {a["agenda_id"]: [] for a in agendas}
            point_mappings: Dict[str, str]       = {}

            for assignment in batch_assignments:
                pid = assignment["point_id"]
                aid = assignment["assigned_agenda_id"]
                point_mappings[pid] = aid
                if aid in agenda_groups:
                    agenda_groups[aid].append(pid)
                else:
                    # Shouldn't happen after fallback, but guard anyway
                    fallback_id = agendas[0]["agenda_id"] if agendas else "A1"
                    point_mappings[pid] = fallback_id
                    agenda_groups.setdefault(fallback_id, []).append(pid)

            logger.info(f"[ROM S3] Phase 5 complete. Groups: { {k: len(v) for k, v in agenda_groups.items()} }")

            # Strip internal embedding key before returning
            for ea in expanded_agendas:
                ea.pop("_embedding_text", None)

            return {
                "agendas": agendas,
                "expanded_agendas": expanded_agendas,
                "candidate_results": candidate_results,
                "batch_assignments": batch_assignments,
                "point_mappings": point_mappings,
                "agenda_groups": agenda_groups,
                "similarity_matrix": sim_matrix_list,  # kept for backward compat / DOCX download
            }

        finally:
            provider.unload_model()
            unload_text_embedder()
            gc.collect()


    def generate_agendas_only(
        self,
        agenda_text: Optional[str],
        recording_id: str,
        user_id: str,
        meeting_context_top_k: int = 5,
        global_context_top_k: int = 3,
        force_reextract: bool = False,
        transcript: Optional[List[Dict]] = None,
        previous_mom_texts: Optional[List[str]] = None,
    ) -> Dict:
        """
        Stage 3 – Step 1: Generate agendas only (no point mapping).
        Returns agendas + expanded agenda context chunks.
        Phases 1 & 2 from the original pipeline.
        """
        from services.ai_provider import get_provider
        from services.text_embedding_service import get_text_embedder, unload_text_embedder
        from services.vector_store import get_meeting_context_store, get_global_context_store
        from services.bm25 import BM25Index, _tokenize
        from services.rag_pipeline import (
            _load_parsed_agenda,
            _save_parsed_agenda,
            get_or_create_agenda_items,
        )

        user_id = _validate_user_id(user_id)

        agendas: List[Dict] = []
        provider = get_provider()
        embedder = get_text_embedder()
        embedder.load()

        dim = getattr(embedder, "_dim", 1024)
        meeting_store = get_meeting_context_store(recording_id, dim)
        global_store  = get_global_context_store(user_id, dim)

        def _build_bm25(store):
            try:
                store.load_or_create()
                metas = store._meta
                if not metas:
                    return None
                texts = [m.get("_text") or m.get("text") or "" for m in metas]
                return BM25Index.from_documents(texts, metas)
            except Exception as e:
                logger.warning(f"[ROM S3-Agenda] BM25 build failed: {e}")
                return None

        meeting_bm25 = _build_bm25(meeting_store)
        global_bm25  = _build_bm25(global_store)

        def _rrf_merge(result_lists, k=60):
            rrf_scores: Dict[str, float] = {}
            best_entry: Dict[str, Dict] = {}
            for ranked in result_lists:
                for rank, entry in enumerate(ranked, start=1):
                    key = (entry.get("_text") or entry.get("text") or "").strip()
                    if not key:
                        continue
                    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                    if key not in best_entry or entry.get("score", 0) > best_entry[key].get("score", 0):
                        best_entry[key] = entry
            return sorted(best_entry.values(),
                          key=lambda e: rrf_scores.get((e.get("_text") or e.get("text") or "").strip(), 0.0),
                          reverse=True)

        def _diversity_dedup(results, threshold=0.92):
            if len(results) <= 1:
                return results
            try:
                texts = [r.get("_text") or r.get("text") or "" for r in results]
                vecs  = embedder.encode(texts)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1.0, norms)
                vecs  = vecs / norms
                kept: List[int] = []
                for i, (_, vec) in enumerate(zip(results, vecs)):
                    dup = any(np.dot(vec, vecs[j]) >= threshold for j in kept)
                    if not dup:
                        kept.append(i)
                return [results[i] for i in kept]
            except Exception:
                seen: set = set()
                deduped = []
                for r in results:
                    txt = (r.get("_text") or r.get("text") or "").strip()
                    if txt and txt not in seen:
                        seen.add(txt)
                        deduped.append(r)
                return deduped

        def _hybrid_retrieve(query, query_vec, store, bm25_idx, top_k):
            fetch_k = max(top_k * 3, 15)
            try:
                sem = store.search(query_vec, k=fetch_k)
            except Exception:
                sem = []
            kw: List[Dict] = []
            if bm25_idx is not None:
                try:
                    kw = bm25_idx.search(query, k=fetch_k)
                except Exception:
                    pass
            meta: List[Dict] = []
            try:
                tokens   = _tokenize(query)
                all_meta = store._meta if hasattr(store, "_meta") else []
                meta     = BM25Index.metadata_search(tokens, all_meta, k=fetch_k)
            except Exception:
                pass
            fused   = _rrf_merge([sem, kw, meta])
            diverse = _diversity_dedup(fused)
            return diverse[:top_k]

        try:
            # ── Load / extract agendas (same logic as generate_agendas_and_map) ──
            if not force_reextract:
                cached = _load_parsed_agenda(recording_id)
                if cached and isinstance(cached, list) and len(cached) > 0:
                    logger.info(f"[ROM S3-Agenda] Reusing {len(cached)} cached agenda item(s)")
                    for i, item in enumerate(cached):
                        topic = item.get("topic") or item.get("title") or f"Agenda Topic {i+1}"
                        desc  = item.get("details") or item.get("description") or ""
                        agendas.append({
                            "agenda_id": item.get("agenda_id") or f"A{i+1}",
                            "title": topic,
                            "description": desc,
                            "speaker": item.get("speaker") or item.get("presenter"),
                            "keywords": item.get("keywords") if isinstance(item.get("keywords"), list) else [k for k in [topic, item.get("speaker")] if k],
                            "related_concepts": item.get("related_concepts") if isinstance(item.get("related_concepts"), list) else [],
                            "alternative_terminology": item.get("alternative_terminology") if isinstance(item.get("alternative_terminology"), list) else [],
                            "expected_themes": item.get("expected_themes") if isinstance(item.get("expected_themes"), list) else [],
                        })

            if not agendas:
                if agenda_text and agenda_text.strip():
                    retrieved_context_str = ""
                    try:
                        q_vec = embedder.encode_batch([agenda_text.strip()[:1000]])
                        norms = np.linalg.norm(q_vec, axis=1, keepdims=True)
                        norms = np.where(norms < 1e-9, 1.0, norms)
                        q_vec = (q_vec / norms)[0]
                        parts: List[str] = []
                        if meeting_context_top_k > 0:
                            for res in meeting_store.search(q_vec, k=meeting_context_top_k):
                                txt = _extract_chunk_text(res)
                                if txt:
                                    parts.append(txt)
                        if global_context_top_k > 0:
                            for res in global_store.search(q_vec, k=global_context_top_k):
                                txt = _extract_chunk_text(res)
                                if txt:
                                    parts.append(txt)
                        if parts:
                            retrieved_context_str = "\n\n".join(parts)
                    except Exception as e:
                        logger.warning(f"[ROM S3-Agenda] RAG failed: {e}")

                    logger.info("[ROM S3-Agenda] Generating agenda points via LLM from uploaded agenda text")
                    agenda_result = provider.generate_rom_agendas(agenda_text, context=retrieved_context_str)
                    raw_agendas   = agenda_result.get("agendas", [])
                    if raw_agendas:
                        agendas = raw_agendas
                        parsed_to_save = []
                        for a in agendas:
                            if a.get("title"):
                                parsed_to_save.append({
                                    "topic": a.get("title"),
                                    "speaker": a.get("speaker") or a.get("presenter"),
                                    "details": a.get("description") or a.get("details") or "",
                                    "keywords": a.get("keywords") or [],
                                    "related_concepts": a.get("related_concepts") or [],
                                    "alternative_terminology": a.get("alternative_terminology") or [],
                                    "expected_themes": a.get("expected_themes") or [],
                                })
                        if parsed_to_save:
                            _save_parsed_agenda(recording_id, user_id, parsed_to_save)

            if not agendas:
                logger.info("[ROM S3-Agenda] No user agenda provided. Using default 'General Discussion' container.")
                agendas = [{
                    "agenda_id": "A1",
                    "title": "General Discussion",
                    "description": "General meeting discussion points",
                    "keywords": [],
                    "related_concepts": [],
                    "alternative_terminology": [],
                    "expected_themes": [],
                    "is_default_agenda": True
                }]

            logger.info(f"[ROM S3-Agenda] {len(agendas)} agenda(s) loaded. Running Phase 1 & 2.")

            # Phase 1 – Previous MoM Expansion
            previous_mom_expansion: Dict[str, Dict] = {}
            has_previous_mom = bool(previous_mom_texts and any(t.strip() for t in previous_mom_texts))
            if has_previous_mom:
                logger.info("[ROM S3-Agenda] Phase 1: Expanding agendas with previous MoM documents")
                combined_mom = "\n\n---\n\n".join(t for t in previous_mom_texts if t and t.strip())
                agenda_list_for_expansion = json.dumps(
                    [{"agenda_id": a["agenda_id"], "title": a["title"], "description": a.get("description", "")} for a in agendas],
                    ensure_ascii=False
                )
                expansion_result = provider.expand_agendas_with_previous_mom(agenda_list_for_expansion, combined_mom)
                for item in expansion_result.get("expansions", []):
                    aid = item.get("agenda_id")
                    if aid:
                        previous_mom_expansion[aid] = item
                logger.info(f"[ROM S3-Agenda] Phase 1 complete: {len(previous_mom_expansion)} agenda(s) expanded")
            else:
                logger.info("[ROM S3-Agenda] Phase 1: Skipped (no previous MoM files uploaded)")

            # Phase 2 – Per-agenda RAG retrieval
            logger.info("[ROM S3-Agenda] Phase 2: Per-agenda context expansion via RAG")
            expanded_agendas: List[Dict] = []
            retrieved_context_chunks: Dict[str, str] = {}

            for agenda in agendas:
                aid   = agenda["agenda_id"]
                title = agenda.get("title", "")
                desc  = agenda.get("description", "")
                kws   = agenda.get("keywords", [])
                rc    = agenda.get("related_concepts", [])
                alt   = agenda.get("alternative_terminology", [])
                themes = agenda.get("expected_themes", [])

                agenda_query = " ".join(filter(None, [title, desc] + kws + rc + alt + themes))

                meeting_ctx_str = ""
                global_ctx_str  = ""

                if agenda_query.strip():
                    try:
                        q_vecs = embedder.encode_batch([agenda_query[:1000]])
                        norms  = np.linalg.norm(q_vecs, axis=1, keepdims=True)
                        norms  = np.where(norms < 1e-9, 1.0, norms)
                        q_vec  = (q_vecs / norms)[0]

                        if meeting_context_top_k > 0:
                            m_results = _hybrid_retrieve(agenda_query, q_vec, meeting_store, meeting_bm25, meeting_context_top_k)
                            meeting_parts = [_extract_chunk_text(r) for r in m_results if _extract_chunk_text(r)]
                            meeting_ctx_str = "\n\n".join(meeting_parts)

                        if global_context_top_k > 0:
                            g_results = _hybrid_retrieve(agenda_query, q_vec, global_store, global_bm25, global_context_top_k)
                            global_parts = [_extract_chunk_text(r) for r in g_results if _extract_chunk_text(r)]
                            global_ctx_str = "\n\n".join(global_parts)

                    except Exception as e:
                        logger.warning(f"[ROM S3-Agenda] Phase 2 RAG failed for agenda {aid}: {e}")

                prev_expansion = previous_mom_expansion.get(aid, {})
                prev_discussion = prev_expansion.get("previous_discussion", "")

                embedding_text_parts = [title, desc]
                embedding_text_parts.extend(kws)
                embedding_text_parts.extend(rc)
                embedding_text_parts.extend(alt)
                embedding_text_parts.extend(themes)
                if prev_discussion:
                    embedding_text_parts.append(prev_discussion)
                for field in ("previous_decisions", "previous_action_items", "pending_work", "follow_up_items"):
                    vals = prev_expansion.get(field, [])
                    if isinstance(vals, list):
                        embedding_text_parts.extend(vals)
                if meeting_ctx_str:
                    embedding_text_parts.append(meeting_ctx_str[:2000])
                if global_ctx_str:
                    embedding_text_parts.append(global_ctx_str[:1000])

                expanded = {
                    "agenda_id": aid,
                    "title": title,
                    "description": desc,
                    "keywords": kws,
                    "related_concepts": rc,
                    "alternative_terminology": alt,
                    "expected_themes": themes,
                    "previous_meeting_summary": prev_expansion if prev_expansion.get("found_in_previous_meeting") else {},
                    "meeting_context_snippet": meeting_ctx_str[:1500] if meeting_ctx_str else "",
                    "global_context_snippet": global_ctx_str[:1000] if global_ctx_str else "",
                    "_embedding_text": " ".join(p for p in embedding_text_parts if p),
                }
                expanded_agendas.append(expanded)
                retrieved_context_chunks[aid] = meeting_ctx_str[:1500] if meeting_ctx_str else ""

            # Strip internal embedding key before returning
            for ea in expanded_agendas:
                ea.pop("_embedding_text", None)

            logger.info(f"[ROM S3-Agenda] Complete: {len(agendas)} agendas ready for use.")
            return {
                "agendas": agendas,
                "expanded_agendas": expanded_agendas,
                "retrieved_context_chunks": retrieved_context_chunks,
            }

        finally:
            provider.unload_model()
            unload_text_embedder()
            gc.collect()


    def extract_agenda_document_points(
        self,
        agenda_id: str,
        agenda_title: str,
        agenda_description: str,
        document_text: str,
    ) -> Dict:
        """
        Extract 2-5 factual, agenda-specific points from a supporting document
        uploaded for an agenda item. These points are not used for retrieval/context.
        """
        from services.ai_provider import get_provider

        if not document_text or not document_text.strip():
            return {"agenda_id": agenda_id, "points": []}

        provider = get_provider()
        try:
            result = provider.generate_agenda_document_points(
                agenda_title=agenda_title,
                agenda_description=agenda_description,
                document_text=document_text,
            )
            points = result.get("points", [])
            presenter = result.get("presenter")
            logger.info(f"[ROM S3-DocPoints] Agenda {agenda_id}: {len(points)} point(s) extracted from document, presenter={presenter}")
            return {"agenda_id": agenda_id, "points": points, "presenter": presenter}
        except Exception as e:
            logger.error(f"[ROM S3-DocPoints] Failed to extract document points for agenda {agenda_id}: {e}")
            return {"agenda_id": agenda_id, "points": [], "presenter": None}
        finally:
            provider.unload_model()
            gc.collect()


    def map_points_to_agendas(
        self,
        polished_points: List[Dict],
        agendas: List[Dict],
        expanded_agendas: List[Dict],
        recording_id: str,
        user_id: str,
        meeting_context_top_k: int = 5,
        global_context_top_k: int = 3,
        batch_size: int = 20,
        include_agenda_doc_points: bool = False,
        agenda_doc_points: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Stage 3 – Step 2: Map discussion points to agendas (Phases 3-5).
        Requires pre-generated agendas and expanded_agendas from generate_agendas_only().
        Optionally merges agenda document points into the final ROM.
        """
        from services.ai_provider import get_provider
        from services.text_embedding_service import get_text_embedder, unload_text_embedder

        user_id = _validate_user_id(user_id)

        if not polished_points:
            return {
                "candidate_results": [],
                "batch_assignments": [],
                "point_mappings": {},
                "agenda_groups": {},
                "similarity_matrix": [],
                "final_rom_agendas": [],
            }

        if not agendas:
            return {
                "candidate_results": [],
                "batch_assignments": [],
                "point_mappings": {},
                "agenda_groups": {},
                "similarity_matrix": [],
                "final_rom_agendas": [],
            }

        # If no expanded_agendas passed in, use agendas directly (build minimal expanded)
        if not expanded_agendas:
            expanded_agendas = [
                {
                    "agenda_id": a["agenda_id"],
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "keywords": a.get("keywords", []),
                    "related_concepts": a.get("related_concepts", []),
                    "alternative_terminology": a.get("alternative_terminology", []),
                    "expected_themes": a.get("expected_themes", []),
                    "_embedding_text": " ".join(filter(None, [
                        a.get("title", ""), a.get("description", "")
                    ] + a.get("keywords", []) + a.get("related_concepts", []))),
                }
                for a in agendas
            ]
        else:
            # Re-add embedding text from title+description if missing
            for ea in expanded_agendas:
                if "_embedding_text" not in ea or not ea.get("_embedding_text"):
                    ea["_embedding_text"] = " ".join(filter(None, [
                        ea.get("title", ""), ea.get("description", "")
                    ] + ea.get("keywords", []) + ea.get("related_concepts", [])))

        provider = get_provider()
        embedder = get_text_embedder()
        embedder.load()

        try:
            # ── Phase 3: Cosine similarity candidate retrieval ──
            logger.info("[ROM S3-Map] Phase 3: Computing Top-3 candidate agendas per discussion point")
            expanded_texts = [ea.get("_embedding_text", ea.get("title", "")) for ea in expanded_agendas]
            point_texts    = [p.get("polished_text", "") for p in polished_points]
            all_texts      = expanded_texts + point_texts

            embeddings = embedder.encode(all_texts)
            norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms      = np.where(norms < 1e-9, 1.0, norms)
            embeddings = embeddings / norms

            agenda_embs = embeddings[:len(expanded_agendas)]
            point_embs  = embeddings[len(expanded_agendas):]

            sim_matrix      = np.dot(point_embs, agenda_embs.T)
            sim_matrix_list = sim_matrix.tolist()

            n_candidates    = min(3, len(expanded_agendas))
            candidate_results: List[Dict] = []

            for i, point in enumerate(polished_points):
                row         = sim_matrix[i]
                top_indices = np.argsort(row)[::-1][:n_candidates]
                candidates  = [
                    {"agenda_id": expanded_agendas[idx]["agenda_id"],
                     "agenda_title": expanded_agendas[idx]["title"],
                     "score": float(row[idx])}
                    for idx in top_indices
                ]
                candidate_results.append({"point_id": point["id"], "candidates": candidates})

            logger.info(f"[ROM S3-Map] Phase 3 complete: Top-{n_candidates} candidates computed for {len(polished_points)} point(s)")

            # ── Phase 4: Batch LLM Agenda Assignment ──
            logger.info(f"[ROM S3-Map] Phase 4: Batch LLM agenda assignment (batch_size={batch_size})")
            valid_agenda_ids = {a["agenda_id"] for a in agendas}
            agenda_ref_json  = json.dumps(
                [{"agenda_id": a["agenda_id"], "title": a.get("title", ""), "description": a.get("description", "")} for a in agendas],
                ensure_ascii=False
            )
            cand_by_point_id: Dict[str, List[Dict]] = {c["point_id"]: c["candidates"] for c in candidate_results}
            batch_assignments: List[Dict] = []
            total_points      = len(polished_points)
            total_batches     = math.ceil(total_points / batch_size)

            for batch_num, batch_start in enumerate(range(0, total_points, batch_size), start=1):
                batch_points = polished_points[batch_start: batch_start + batch_size]
                logger.info(f"[ROM S3-Map] Phase 4 batch {batch_num}/{total_batches}: {len(batch_points)} point(s)")

                batch_items = []
                for point in batch_points:
                    pid        = point["id"]
                    candidates = cand_by_point_id.get(pid, [])
                    batch_items.append({
                        "point_id": pid,
                        "enhanced_point": point.get("polished_text", ""),
                        "timeline": f"{_format_time_hhmm(point.get('timeline_start', 0))} – {_format_time_hhmm(point.get('timeline_end', 0))}",
                        "speakers": point.get("speakers", []),
                        "candidate_agendas": candidates,
                    })

                batch_json_str = json.dumps(batch_items, ensure_ascii=False)
                llm_result     = provider.assign_agenda_batch(batch_json_str, agenda_ref_json)

                llm_map: Dict[str, Dict] = {a.get("point_id", ""): a for a in llm_result.get("assignments", []) if a.get("point_id")}

                for point in batch_points:
                    pid        = point["id"]
                    candidates = cand_by_point_id.get(pid, [])
                    llm_entry  = llm_map.get(pid, {})

                    assigned    = llm_entry.get("assigned_agenda_id", "")
                    confidence  = llm_entry.get("confidence", "medium")
                    reason      = llm_entry.get("reason", "")
                    is_probable = False

                    if not assigned or assigned not in valid_agenda_ids:
                        is_probable = True
                        confidence  = "probable"
                        reason      = "Automatically assigned to best-matching candidate (LLM did not return a valid agenda ID)."
                        assigned    = candidates[0]["agenda_id"] if candidates else (agendas[0]["agenda_id"] if agendas else "A1")

                    batch_assignments.append({
                        "point_id": pid,
                        "assigned_agenda_id": assigned,
                        "confidence": "probable" if is_probable else confidence,
                        "reason": reason,
                        "is_probable": is_probable,
                    })

                logger.info(f"[ROM S3-Map] Phase 4 batch {batch_num}/{total_batches} complete")

            logger.info(f"[ROM S3-Map] Phase 4 complete: {len(batch_assignments)} assignment(s)")

            # ── Phase 5: Agenda Grouping ──
            logger.info("[ROM S3-Map] Phase 5: Grouping discussion points by assigned agenda")
            agenda_groups: Dict[str, List[str]] = {a["agenda_id"]: [] for a in agendas}
            point_mappings: Dict[str, str]       = {}

            for assignment in batch_assignments:
                pid = assignment["point_id"]
                aid = assignment["assigned_agenda_id"]
                point_mappings[pid] = aid
                if aid in agenda_groups:
                    agenda_groups[aid].append(pid)
                else:
                    fallback_id = agendas[0]["agenda_id"] if agendas else "A1"
                    point_mappings[pid] = fallback_id
                    agenda_groups.setdefault(fallback_id, []).append(pid)

            logger.info(f"[ROM S3-Map] Phase 5 complete. Groups: { {k: len(v) for k, v in agenda_groups.items()} }")

            # ── Build final_rom agendas ──
            assign_map: dict = {a["point_id"]: a for a in batch_assignments if a.get("point_id")}
            final_agendas = []

            for a in agendas:
                agenda_points = []
                for p in polished_points:
                    if point_mappings.get(p.get("id")) == a.get("agenda_id"):
                        p_copy = dict(p)
                        p_copy["text"] = p.get("polished_text") or p.get("text", "")
                        assignment = assign_map.get(p.get("id"), {})
                        p_copy["assignment_confidence"] = assignment.get("confidence", "medium")
                        p_copy["assignment_reason"]     = assignment.get("reason", "")
                        p_copy["is_probable"]           = assignment.get("is_probable", False)
                        agenda_points.append(p_copy)

                # Merge in agenda document points if requested
                agenda_id = a.get("agenda_id", "")
                if include_agenda_doc_points and agenda_doc_points and agenda_id in agenda_doc_points:
                    doc_entry = agenda_doc_points.get(agenda_id, {})
                    doc_pts   = doc_entry.get("points", []) if isinstance(doc_entry, dict) else []
                    doc_name  = doc_entry.get("doc_name", "Supporting Document") if isinstance(doc_entry, dict) else "Supporting Document"
                    for dp_text in doc_pts:
                        if dp_text and str(dp_text).strip():
                            agenda_points.append({
                                "id": str(uuid.uuid4()),
                                "text": str(dp_text).strip(),
                                "polished_text": str(dp_text).strip(),
                                "speaker": "From Document",
                                "speakers": ["From Document"],
                                "action_owner": None,
                                "decisions": [],
                                "action_items": [],
                                "technical_terms": [],
                                "dates": [],
                                "numbers": [],
                                "references": [doc_name],
                                "timeline_start": 0.0,
                                "timeline_end": 0.0,
                                "is_doc_point": True,
                                "assignment_confidence": "high",
                                "assignment_reason": f"Extracted from uploaded document: {doc_name}",
                                "is_probable": False,
                            })

                if agenda_points:
                    agenda_copy = dict(a)
                    agenda_copy["discussion_points"] = agenda_points
                    final_agendas.append(agenda_copy)

            return {
                "candidate_results": candidate_results,
                "batch_assignments": batch_assignments,
                "point_mappings": point_mappings,
                "agenda_groups": agenda_groups,
                "similarity_matrix": sim_matrix_list,
                "final_rom_agendas": final_agendas,
            }

        finally:
            provider.unload_model()
            unload_text_embedder()
            gc.collect()

    def generate_mom_from_enhanced_rom(
        self,
        polished_points: List[Dict],
        recording_meta: Dict,
        recording_id: str = "",
        user_id: str = "",
    ) -> Dict:
        """
        Generate Minutes of Meeting (MOM) using Stage 2/Stage 3 Enhanced Discussion Points.
        1. LLM is passed the points ONLY to generate Introduction and Conclusion.
        2. Key Discussion Points are constructed directly from each enhanced discussion point.
        3. Action Items are extracted directly from each point's action_items/action_owner/dates.
        """
        from services.ai_provider import get_provider

        if user_id:
            user_id = _validate_user_id(user_id)

        filename = recording_meta.get("filename", "Meeting MoM")
        date_str = str(recording_meta.get("created_at", "") or "")
        dur_str = str(recording_meta.get("duration", "") or "")
        participants = recording_meta.get("speakers_detected", [])
        if isinstance(participants, str):
            try:
                import json as _j
                participants = _j.loads(participants)
            except Exception:
                participants = [p.strip() for p in participants.split(",") if p.strip()]

        if not polished_points:
            return {
                "title": filename,
                "date": date_str,
                "duration": dur_str,
                "participants": participants if isinstance(participants, list) else [],
                "introduction": "No enhanced discussion points available.",
                "points_discussed": [],
                "action_items": [],
                "conclusion": "",
            }

        # 1. Build points_discussed as clean strings: "Topic: Summary"
        points_discussed: list[str] = []
        points_text_lines = []

        for idx, p in enumerate(polished_points, start=1):
            pt_text = (p.get("polished_text") or p.get("text") or p.get("discussion_point") or "").strip()
            if not pt_text:
                continue

            # Derive topic title from technical_terms, project_names, or first sentence
            topic = ""
            if p.get("technical_terms"):
                tt = p["technical_terms"]
                topic = ", ".join(tt) if isinstance(tt, list) else str(tt)
            if not topic and p.get("project_names"):
                pn = p["project_names"]
                topic = ", ".join(pn) if isinstance(pn, list) else str(pn)
            if not topic:
                first_sent = pt_text.split(".")[0].strip()
                topic = first_sent[:70] if len(first_sent) > 5 else pt_text[:70]

            # Store as a clean "Topic: Summary" string — no objects, no JSON
            topic_clean = topic.strip(": ")
            point_str = f"{topic_clean}: {pt_text}" if topic_clean and not pt_text.lower().startswith(topic_clean.lower()) else pt_text
            points_discussed.append(point_str)
            points_text_lines.append(f"Point {idx}: {pt_text}")

        # 2. Directly extract action_items from each enhanced point
        action_items = []
        for p in polished_points:
            raw_acts = p.get("action_items")
            if not raw_acts:
                continue

            if isinstance(raw_acts, list):
                act_list = [str(a).strip() for a in raw_acts if str(a).strip()]
            else:
                act_list = [str(raw_acts).strip()]

            owner = p.get("action_owner")
            if not owner:
                spk_list = p.get("speakers")
                if isinstance(spk_list, list) and spk_list:
                    owner = ", ".join(spk_list)
                elif p.get("speaker"):
                    owner = str(p.get("speaker"))
                else:
                    owner = "Unassigned"
            elif isinstance(owner, list):
                owner = ", ".join(str(o) for o in owner)

            dates = p.get("dates")
            deadline = "ASAP"
            if dates:
                d_str = ", ".join(dates) if isinstance(dates, list) else str(dates)
                if d_str.strip() and d_str.strip().lower() not in ("none", "n/a", "null"):
                    deadline = d_str.strip()

            for act_text in act_list:
                if act_text and act_text.lower() not in ("none", "n/a", "null"):
                    action_items.append({
                        "task": act_text,
                        "item": act_text,
                        "description": act_text,
                        "owner": str(owner).strip(),
                        "deadline": deadline,
                        "status": "open",
                    })

        # 3. Generate Title, Introduction, and Conclusion using standard app generator
        overview = self.generate_mom_overview(points_text_lines, recording_meta)

        return {
            "title": overview["title"],
            "date": date_str,
            "duration": dur_str,
            "participants": participants if isinstance(participants, list) else [],
            "introduction": overview["introduction"],
            "points_discussed": points_discussed,
            "action_items": action_items,
            "conclusion": overview["conclusion"],
        }

    def generate_mom_overview(
        self,
        points_text_lines: List[str],
        recording_meta: Dict,
    ) -> Dict[str, str]:
        """
        Generate professional Title, Introduction, and Conclusion using standard application LLM prompt.
        Reusable helper used by generate_mom_from_enhanced_rom and generate_advanced_mom.
        """
        from services.ai_provider import get_provider

        filename = recording_meta.get("filename", "Meeting MoM")
        date_str = str(recording_meta.get("created_at", "") or "")
        participants = recording_meta.get("speakers_detected", [])
        if isinstance(participants, str):
            try:
                import json as _j
                participants = _j.loads(participants)
            except Exception:
                participants = [p.strip() for p in participants.split(",") if p.strip()]

        points_combined_str = "\n".join(points_text_lines)
        first_topic = points_text_lines[0][:80] if points_text_lines else filename
        intro = f"This meeting covered {len(points_text_lines)} key discussion point(s) including topics related to {first_topic}."
        conclusion = "The meeting concluded following thorough discussion of all agenda items, with clear action items assigned and next steps agreed upon."
        title = filename

        prompt = (
            f"You are an executive assistant generating a formal Minutes of Meeting (MoM) document.\n\n"
            f"Recording name: '{filename}'\n"
            f"Date: {date_str}\n"
            f"Participants: {', '.join(str(p) for p in participants) if participants else 'Unknown'}\n\n"
            f"Discussion points from the meeting:\n"
            f"{points_combined_str}\n\n"
            f"Tasks:\n"
            f"1. Generate a professional, concise meeting TITLE that reflects the actual topics discussed (do NOT just use the filename; infer from content). Example: 'Q3 Engineering Review Meeting', 'Budget & Procurement Planning Session'.\n"
            f"2. Write a high-level 2-3 sentence executive Introduction describing the meeting's objective, scope, and key participants.\n"
            f"3. Write a 2-3 sentence professional Conclusion summarizing overall outcomes, key decisions made, and agreed next steps.\n\n"
            f"Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):\n"
            f"{{\n"
            f'  "title": "...",\n'
            f'  "introduction": "...",\n'
            f'  "conclusion": "..."\n'
            f"}}\n"
        )

        provider = get_provider()
        try:
            if hasattr(provider, "query"):
                raw_resp = provider.query(prompt, max_tokens=1024, temperature=0.2)
            else:
                raw_resp = provider._infer(prompt, max_new_tokens=1024)
            cleaned = str(raw_resp or "").strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            import json as _json
            data_parsed = _json.loads(cleaned)
            if isinstance(data_parsed, dict):
                if data_parsed.get("title") and str(data_parsed["title"]).strip():
                    title = str(data_parsed["title"]).strip()
                if data_parsed.get("introduction"):
                    intro = str(data_parsed["introduction"]).strip()
                if data_parsed.get("conclusion"):
                    conclusion = str(data_parsed["conclusion"]).strip()
        except Exception as e:
            logger.warning(f"[RomService] Title/Intro/Conclusion generation fallback: {e}")
        finally:
            provider.unload_model()
            gc.collect()

        return {
            "title": title,
            "introduction": intro,
            "conclusion": conclusion,
        }

    def generate_advanced_mom(
        self,
        final_rom: Dict,
        existing_mom: Dict,
        custom_prompt: str,
        regenerate_title: bool = False,
        regenerate_intro: bool = False,
        regenerate_conclusion: bool = False,
        recording_meta: Dict = None,
        recording_id: str = "",
        user_id: str = "",
    ) -> Dict:
        """
        Single Responsibility: Refine, enhance, organize, and enrich extracted action points using custom user prompt instructions.
        Adds clarity, ownership, deadlines, dependencies, and context while strictly preserving factual accuracy.
        If regenerate_title, regenerate_intro, or regenerate_conclusion are True, calls the existing
        generate_mom_overview generator after action item refinement is complete.
        """
        import json as _json
        from services.ai_provider import get_provider

        recording_meta = recording_meta or {}
        filename = recording_meta.get("filename", "Meeting MoM")

        agendas = final_rom.get("agendas", []) if isinstance(final_rom, dict) else []

        # Extract lean action items context containing ONLY action_point, action_owner, speaker, date
        action_points = []
        for ag in agendas:
            if isinstance(ag, dict):
                pts = ag.get("discussion_points", [])
                if isinstance(pts, list):
                    for p in pts:
                        if isinstance(p, dict):
                            pt_text = (p.get("polished_text") or p.get("text") or "").strip()
                            owner = p.get("action_owner") or "Unassigned"
                            speakers = p.get("speakers") or ([p.get("speaker")] if p.get("speaker") else [])
                            speakers_str = ", ".join([str(s) for s in speakers if s]) if speakers else (str(p.get("speaker")) if p.get("speaker") else "Unknown")
                            dates = p.get("dates") or []
                            dates_str = ", ".join([str(d) for d in dates if d]) if isinstance(dates, list) else str(dates or "ASAP")

                            if pt_text or owner != "Unassigned" or p.get("action_items"):
                                action_points.append({
                                    "action_point": pt_text,
                                    "action_owner": owner,
                                    "speaker": speakers_str,
                                    "date": dates_str if dates_str else "ASAP"
                                })

        if isinstance(existing_mom, dict):
            ex_actions = existing_mom.get("action_items", [])
            if isinstance(ex_actions, list):
                for act in ex_actions:
                    if isinstance(act, dict):
                        task = str(act.get("task") or act.get("action_point") or "").strip()
                        if task:
                            action_points.append({
                                "action_point": task,
                                "action_owner": str(act.get("owner") or act.get("action_owner") or "Unassigned"),
                                "speaker": str(act.get("speaker") or "Unknown"),
                                "date": str(act.get("deadline") or act.get("date") or "ASAP")
                            })

        action_context_json_str = _json.dumps(action_points, indent=2)

        schema_str = (
            "{\n"
            '  "action_items": [\n'
            '    {\n'
            '      "task": "Refined and enriched action item / task description (with clarity, dependencies, and context)",\n'
            '      "owner": "Speaker name or Unassigned",\n'
            '      "deadline": "Deadline date if mentioned or derived, otherwise ASAP or null"\n'
            '    }\n'
            '  ]\n'
            "}"
        )

        prompt = (
            f"You are an expert AI meeting editor specializing in refining, enhancing, organizing, and enriching Action Points for Minutes of Meeting (MoM).\n\n"
            f"Meeting Title / Context: '{filename}'\n\n"
            f"USER CUSTOM INSTRUCTIONS:\n"
            f"\"\"\"\n{custom_prompt.strip()}\n\"\"\"\n\n"
            f"STRICT RULES:\n"
            f"1. You MUST NOT modify or hallucinate core meeting facts, speaker names, action owners, or deadlines.\n"
            f"2. Focus exclusively on refining, enhancing, organizing, and enriching the extracted action points (adding clarity, ownership, deadlines if available, dependencies, and context) based on the user's instructions.\n"
            f"3. Do NOT include any agenda details or extraneous discussion point hierarchies in the output.\n\n"
            f"CURRENT ACTION POINTS & METADATA:\n"
            f"```json\n{action_context_json_str}\n```\n\n"
            f"REQUIRED TASKS:\n"
            f"Extract and refine action_items in format: [{{\"task\": \"...\", \"owner\": \"...\", \"deadline\": \"...\"}}]\n\n"
            f"Respond ONLY with a single valid JSON object in this exact schema (no markdown, no code block backticks):\n"
            f"{schema_str}\n"
        )

        provider = get_provider()
        result_data = {}
        try:
            if hasattr(provider, "query"):
                raw_resp = provider.query(prompt, max_tokens=4026, temperature=0.2)
            else:
                raw_resp = provider._infer(prompt, max_new_tokens=4026)

            cleaned = str(raw_resp or "").strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

            result_data = _json.loads(cleaned)
        except Exception as e:
            logger.warning(f"[RomService] Advanced MoM LLM enhancement error: {e}")
            result_data = {}
        finally:
            provider.unload_model()
            gc.collect()

        # Agendas remain unchanged (Advanced MoM focuses strictly on Action Point refinement)
        updated_agendas = agendas

        # Extract action items
        raw_action_items = result_data.get("action_items") if isinstance(result_data, dict) else None
        action_items = []
        if isinstance(raw_action_items, list):
            for item in raw_action_items:
                if isinstance(item, dict):
                    task_str = str(item.get("task") or item.get("item") or item.get("description") or "").strip()
                    owner_str = str(item.get("owner", "Unassigned") or "Unassigned").strip()
                    deadline_str = str(item.get("deadline", "ASAP") or "ASAP").strip()
                    if task_str:
                        action_items.append({
                            "task": task_str,
                            "owner": owner_str,
                            "deadline": deadline_str,
                            "status": item.get("status", "open")
                        })
                elif isinstance(item, str) and item.strip():
                    action_items.append({"task": item.strip(), "owner": "Unassigned", "deadline": "ASAP", "status": "open"})
        
        if not action_items and isinstance(existing_mom, dict):
            action_items = existing_mom.get("action_items", [])

        # Trigger existing overview generator ONLY if any checkbox was selected
        title = existing_mom.get("title") if isinstance(existing_mom, dict) else filename
        intro = existing_mom.get("introduction") if isinstance(existing_mom, dict) else ""
        conclusion = existing_mom.get("conclusion") if isinstance(existing_mom, dict) else ""

        if regenerate_title or regenerate_intro or regenerate_conclusion:
            # Extract discussion point lines from enhanced agendas for the generator
            point_lines = []
            for ag in updated_agendas:
                pts = ag.get("discussion_points") if isinstance(ag, dict) else []
                if isinstance(pts, list):
                    for p in pts:
                        if isinstance(p, dict):
                            t = (p.get("polished_text") or p.get("text") or p.get("discussion_point") or "").strip()
                            if t:
                                point_lines.append(t)
            if not point_lines:
                point_lines = [filename]

            overview = self.generate_mom_overview(point_lines, recording_meta)
            if regenerate_title and overview.get("title"):
                title = overview["title"]
            if regenerate_intro and overview.get("introduction"):
                intro = overview["introduction"]
            if regenerate_conclusion and overview.get("conclusion"):
                conclusion = overview["conclusion"]

        return {
            "title": title,
            "introduction": intro,
            "conclusion": conclusion,
            "agendas": updated_agendas,
            "action_items": action_items,
        }


def apply_speaker_mappings_to_final_rom(final_rom: Dict) -> Dict:
    """
    Replaces every occurrence of mapped Speaker_IDs across all discussion points,
    action items, presenters, participants, and text fields throughout the Final ROM structure.
    Unmapped Speaker_IDs remain unchanged.
    """
    if not final_rom or not isinstance(final_rom, dict):
        return final_rom

    speaker_mappings = final_rom.get("speaker_mappings", {})
    if not speaker_mappings or not isinstance(speaker_mappings, dict):
        return final_rom

    from services.speaker_sync import build_clean_speaker_mappings, replace_speaker_in_text, replace_deep_speaker_names
    clean_mappings = build_clean_speaker_mappings(speaker_mappings)
    if not clean_mappings:
        return final_rom

    updated = replace_deep_speaker_names(final_rom, clean_mappings)

    # Specific field list updates
    parts = updated.get("participants")
    if isinstance(parts, list):
        updated["participants"] = [clean_mappings.get(p, replace_speaker_in_text(p, clean_mappings)) for p in parts]

    updated["speaker_mappings"] = clean_mappings
    return updated


rom_service = RomService()

