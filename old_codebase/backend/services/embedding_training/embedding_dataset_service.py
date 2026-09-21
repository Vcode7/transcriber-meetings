"""
embedding_dataset_service.py — Document text extraction, normalization, and chunking for Embedding Domain Adaptation.

Supported formats:
  - PDF (.pdf)
  - Word (.docx, .doc)
  - PowerPoint (.pptx, .ppt)
  - Plain text (.txt)
  - Markdown (.md)

Cleans, normalizes, removes noise/duplicates, and prepares domain text chunks.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from config import settings

logger = logging.getLogger(__name__)

# Temporary directory for training corpora sessions
CORPUS_CACHE_DIR = Path(settings.RUNTIME_DIR) / "embedding_corpus_cache"
CORPUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _clean_text(raw_text: str) -> str:
    """Normalize unicode, strip control characters, and collapse irregular whitespace."""
    if not raw_text:
        return ""
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", raw_text)
    # Replace carriage returns and non-breaking spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    # Replace non-printable ASCII control chars except newline and tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse multiple consecutive newlines (more than 2 -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_domain_terms(text: str, top_k: int = 25) -> List[Dict[str, Any]]:
    """
    Extract prominent domain terms, technical acronyms, and abbreviations from the text.
    Focuses on uppercase acronyms (e.g. ADA, LCA, FADEC, TCAS, GPS, IMU) and technical terms.
    """
    # Find all 2-7 letter uppercase acronyms (e.g., ADA, INS, FADEC)
    acronyms = re.findall(r"\b[A-Z0-9]{2,7}\b", text)
    # Filter common English non-domain words
    stop_acronyms = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN", "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE"}
    filtered_acronyms = [a for a in acronyms if a not in stop_acronyms and not a.isdigit()]

    # Technical title case / domain phrases (e.g., Air Data, Flight Control)
    phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)

    counts: Dict[str, int] = {}
    for term in filtered_acronyms:
        counts[term] = counts.get(term, 0) + 1
    for p in phrases:
        if len(p.split()) <= 4:
            counts[p] = counts.get(p, 0) + 1

    sorted_terms = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [{"term": term, "count": count} for term, count in sorted_terms[:top_k]]


def _chunk_text(text: str, min_words: int = 20, max_words: int = 150) -> List[str]:
    """
    Split cleaned text into cohesive semantic chunks suitable for embedding model training.
    Splits by paragraph first, then sentences if paragraphs are too long.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    for para in paragraphs:
        # Ignore extremely short lines (e.g. isolated page numbers or headers)
        words = para.split()
        if len(words) < min_words:
            # Check if it has valuable technical acronyms or content
            if len(words) >= 8 and any(re.match(r"^[A-Z0-9]{2,6}$", w) for w in words):
                chunks.append(para)
            continue

        if len(words) <= max_words:
            chunks.append(para)
        else:
            # Split paragraph into sentence clusters
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current_cluster: List[str] = []
            current_word_count = 0

            for s in sentences:
                s_words = len(s.split())
                if current_word_count + s_words > max_words and current_cluster:
                    chunks.append(" ".join(current_cluster))
                    current_cluster = [s]
                    current_word_count = s_words
                else:
                    current_cluster.append(s)
                    current_word_count += s_words

            if current_cluster and current_word_count >= min_words:
                chunks.append(" ".join(current_cluster))

    return chunks


def process_uploaded_document(
    file_bytes: bytes,
    filename: str,
    content_type: str = "",
) -> Dict[str, Any]:
    """
    Extract and process a single document file.
    Writes bytes to a temporary file so doc_extractor can parse it via PyMuPDF, python-docx, etc.
    Returns extracted text, cleaned text, statistics, and status.
    """
    import tempfile
    raw_text = ""
    error = None
    ext = Path(filename).suffix.lower() or ".tmp"

    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            temp_file = tf.name
            tf.write(file_bytes)

        from services.doc_extractor import extract_text_from_file
        raw_text = extract_text_from_file(temp_file, filename)
    except Exception as exc:
        logger.warning(f"[DocProcessing] doc_extractor failed for {filename}: {exc}")
        # Plain-text fallback for TXT and MD
        if ext in (".txt", ".md", ".csv", ".json", ".log"):
            try:
                raw_text = file_bytes.decode("utf-8", errors="replace")
            except Exception as e:
                error = f"Encoding error: {e}"
        else:
            error = str(exc)
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

    if not error and not raw_text.strip():
        error = "No readable text could be extracted from document."

    cleaned = _clean_text(raw_text) if raw_text else ""
    word_count = len(cleaned.split()) if cleaned else 0
    char_count = len(cleaned) if cleaned else 0

    return {
        "filename": filename,
        "size_bytes": len(file_bytes),
        "raw_char_count": len(raw_text),
        "cleaned_char_count": char_count,
        "cleaned_word_count": word_count,
        "text": cleaned,
        "status": "extracted" if not error else "error",
        "error": error,
    }



def process_and_prepare_corpus(
    uploaded_files: List[Tuple[str, bytes, str]],
    session_id: str,
) -> Dict[str, Any]:
    """
    Process multiple documents, clean text, deduplicate chunks, extract statistics,
    and save the prepared corpus chunks to disk for training.
    """
    doc_results: List[Dict[str, Any]] = []
    all_chunks: List[str] = []
    seen_chunk_hashes: set[str] = set()
    dedup_count = 0
    full_corpus_text = []

    for filename, file_bytes, content_type in uploaded_files:
        res = process_uploaded_document(file_bytes, filename, content_type)
        doc_results.append({
            "filename": res["filename"],
            "size_bytes": res["size_bytes"],
            "word_count": res["cleaned_word_count"],
            "char_count": res["cleaned_char_count"],
            "status": res["status"],
            "error": res["error"],
        })

        if res["status"] == "extracted" and res["text"]:
            full_corpus_text.append(res["text"])
            doc_chunks = _chunk_text(res["text"])
            for ch in doc_chunks:
                # Normalize chunk hash for deduplication
                norm_str = re.sub(r"\s+", " ", ch.lower().strip())
                h = hashlib.md5(norm_str.encode("utf-8")).hexdigest()
                if h in seen_chunk_hashes:
                    dedup_count += 1
                else:
                    seen_chunk_hashes.add(h)
                    all_chunks.append(ch)

    combined_text = "\n\n".join(full_corpus_text)
    domain_terms = _extract_domain_terms(combined_text)

    # Save prepared chunks to session corpus file
    corpus_file = CORPUS_CACHE_DIR / f"corpus_{session_id}.json"
    import json
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump({
            "session_id": session_id,
            "total_chunks": len(all_chunks),
            "chunks": all_chunks,
            "document_names": [d["filename"] for d in doc_results if d["status"] == "extracted"],
        }, f, ensure_ascii=False, indent=2)

    total_words = sum(d["word_count"] for d in doc_results if d["status"] == "extracted")
    total_chars = sum(d["char_count"] for d in doc_results if d["status"] == "extracted")

    return {
        "session_id": session_id,
        "total_documents": len(uploaded_files),
        "successful_documents": sum(1 for d in doc_results if d["status"] == "extracted"),
        "failed_documents": sum(1 for d in doc_results if d["status"] == "error"),
        "total_cleaned_words": total_words,
        "total_cleaned_chars": total_chars,
        "total_chunks": len(all_chunks),
        "deduplicated_chunks": dedup_count,
        "domain_terms": domain_terms,
        "document_stats": doc_results,
        "sample_chunks": all_chunks[:5] if all_chunks else [],
    }


def load_corpus_chunks(session_id: str) -> List[str]:
    """Load cached chunks for a given session ID."""
    corpus_file = CORPUS_CACHE_DIR / f"corpus_{session_id}.json"
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus session {session_id} not found or expired.")
    import json
    with open(corpus_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chunks", [])
