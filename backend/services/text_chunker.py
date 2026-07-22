"""
text_chunker.py — Sentence-aware text chunking for the RAG pipeline.

All three RAG sources (global context, meeting context, transcript) use
the same chunking utilities so embedding vectors are consistently sized.

Functions
---------
chunk_text(text, chunk_size, overlap)
    Chunk raw text into overlapping windows (word-count based, sentence-aware).

chunk_transcript(transcript, chunk_size)
    Chunk a transcript segment list into text windows, preserving speaker labels.

chunk_pdf_pages(pages, chunk_size, overlap)
    Chunk a list of (page_num, text) tuples, preserving page metadata.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Tuple


# ── Timestamp formatter ───────────────────────────────────────────────────────

def _fmt_time(secs: float) -> str:
    """
    Format seconds as HH:MM:SS.

    Examples
    --------
    _fmt_time(0)      → '00:00:00'
    _fmt_time(75.3)   → '00:01:15'
    _fmt_time(3662.0) → '01:01:02'
    """
    total = int(secs)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Sentence splitter ─────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple punctuation rules.
    Returns a list of non-empty stripped sentences.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# ── Core text chunker ─────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Dict[str, Any]]:
    """
    Chunk raw text into overlapping windows.

    Parameters
    ----------
    text       : Input text (any length).
    chunk_size : Target words per chunk (approximate — always breaks on sentence boundaries).
    overlap    : Words of overlap between consecutive chunks.

    Returns
    -------
    List of dicts: [{"text": str, "chunk_index": int}]
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[Dict[str, Any]] = []
    current_sentences: List[str] = []
    current_words = 0
    chunk_index = 0

    # Overlap buffer: remember last N words to prepend to the next chunk
    overlap_buf: List[str] = []

    for sentence in sentences:
        word_count = len(sentence.split())
        current_sentences.append(sentence)
        current_words += word_count

        if current_words >= chunk_size:
            chunk_text_str = " ".join(current_sentences)
            if overlap_buf:
                chunk_text_str = " ".join(overlap_buf) + " " + chunk_text_str

            chunks.append({
                "text": chunk_text_str.strip(),
                "chunk_index": chunk_index,
            })
            chunk_index += 1

            # Compute overlap: take last N words from current chunk
            all_words = chunk_text_str.split()
            overlap_buf = all_words[-overlap:] if overlap > 0 else []

            current_sentences = []
            current_words = 0

    # Flush remaining
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        if overlap_buf:
            chunk_text_str = " ".join(overlap_buf) + " " + chunk_text_str
        chunks.append({
            "text": chunk_text_str.strip(),
            "chunk_index": chunk_index,
        })

    return chunks


# ── Transcript chunker ────────────────────────────────────────────────────────

def _merge_consecutive_speaker_lines(
    lines: List[Tuple[str, str, float, float]]
) -> List[Tuple[str, str, float, float]]:
    """
    Merge consecutive segments from the same speaker into a single block,
    combining their text and extending the start/end timestamps.

    Example
    -------
    [("A", "Hello", 0.0, 1.0), ("A", "world.", 1.1, 2.0), ("B", "Hi.", 2.5, 3.0)]
    → [("A", "Hello world.", 0.0, 2.0), ("B", "Hi.", 2.5, 3.0)]
    """
    if not lines:
        return []
    merged: List[Tuple[str, str, float, float]] = []
    cur_speaker, cur_text, cur_start, cur_end = lines[0]
    for speaker, text, start, end in lines[1:]:
        if speaker == cur_speaker:
            cur_text = cur_text.rstrip() + " " + text
            cur_end = end
        else:
            merged.append((cur_speaker, cur_text, cur_start, cur_end))
            cur_speaker, cur_text, cur_start, cur_end = speaker, text, start, end
    merged.append((cur_speaker, cur_text, cur_start, cur_end))
    return merged


def chunk_transcript(
    transcript: List[Dict[str, Any]],
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Dict[str, Any]]:
    """
    Chunk a list of transcript segment dicts into overlapping text windows.

    Each segment has: {"speaker_label": str, "text": str, "start": float, "end": float, ...}

    Each speaker block in the chunk text is formatted as:

        [HH:MM:SS - HH:MM:SS] Speaker Name
        Transcript text for that block...

    Consecutive segments from the same speaker are merged before formatting
    so each block has a single consolidated timeline header.

    Stores start/end timestamps of the first and last segment in each chunk.

    Parameters
    ----------
    transcript : List of transcript segment dicts.
    chunk_size : Target words per chunk.
    overlap    : Words of overlap between chunks.

    Returns
    -------
    List of dicts:
    [{"text": str, "chunk_index": int, "start": float, "end": float, "speakers": List[str]}]
    """
    if not transcript:
        return []

    # Build (speaker, text, start, end) tuples
    lines: List[Tuple[str, str, float, float]] = []
    for seg in transcript:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker_label") or "Unknown"
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        lines.append((speaker, text, start, end))

    if not lines:
        return []

    # Merge consecutive same-speaker segments (preserves timestamps)
    lines = _merge_consecutive_speaker_lines(lines)

    chunks: List[Dict[str, Any]] = []
    current_lines: List[Tuple[str, str, float, float]] = []
    current_words = 0
    chunk_index = 0
    overlap_lines: List[Tuple[str, str, float, float]] = []

    def _build_chunk_text(all_lns: List[Tuple[str, str, float, float]]) -> str:
        """Format merged-speaker lines with timeline headers."""
        parts = []
        for spk, txt, st, en in all_lns:
            header = f"[{_fmt_time(st)} - {_fmt_time(en)}] {spk}"
            parts.append(f"{header}\n{txt}")
        return "\n\n".join(parts)

    for line in lines:
        speaker, text, start, end = line
        word_count = len(text.split())
        current_lines.append(line)
        current_words += word_count

        if current_words >= chunk_size:
            all_lines = overlap_lines + current_lines
            chunk_text_str = _build_chunk_text(all_lines)
            all_speakers = list(dict.fromkeys(s for s, _, _, _ in all_lines))
            chunk_start = all_lines[0][2]
            chunk_end = all_lines[-1][3]

            chunks.append({
                "text": chunk_text_str.strip(),
                "chunk_index": chunk_index,
                "start": chunk_start,
                "end": chunk_end,
                "speakers": all_speakers,
            })
            chunk_index += 1

            # Overlap: keep last few lines whose word count <= overlap
            overlap_buf_words = 0
            overlap_lines = []
            for prev_line in reversed(current_lines):
                pw = len(prev_line[1].split())
                if overlap_buf_words + pw <= overlap:
                    overlap_lines.insert(0, prev_line)
                    overlap_buf_words += pw
                else:
                    break

            current_lines = []
            current_words = 0

    # Flush remaining
    if current_lines:
        all_lines = overlap_lines + current_lines
        chunk_text_str = _build_chunk_text(all_lines)
        all_speakers = list(dict.fromkeys(s for s, _, _, _ in all_lines))
        chunk_start = all_lines[0][2]
        chunk_end = all_lines[-1][3]
        chunks.append({
            "text": chunk_text_str.strip(),
            "chunk_index": chunk_index,
            "start": chunk_start,
            "end": chunk_end,
            "speakers": all_speakers,
        })

    return chunks


# ── Page-aware chunker ────────────────────────────────────────────────────────

def chunk_pages(
    pages: List[Tuple[int, str]],
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Dict[str, Any]]:
    """
    Chunk a list of (page_number, page_text) tuples, preserving page metadata.

    Returns
    -------
    List of dicts: [{"text": str, "chunk_index": int, "page": int}]
    """
    result: List[Dict[str, Any]] = []
    global_chunk_idx = 0

    for page_num, page_text in pages:
        if not page_text or not page_text.strip():
            continue
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for ch in page_chunks:
            result.append({
                "text": ch["text"],
                "chunk_index": global_chunk_idx,
                "page": page_num,
            })
            global_chunk_idx += 1

    return result
