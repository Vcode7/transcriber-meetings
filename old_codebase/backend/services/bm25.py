"""
bm25.py — Lightweight in-memory BM25+ keyword retrieval.

Used by Stage 2 of the ROM pipeline as the keyword-retrieval leg of
the hybrid retrieval strategy.  No external dependencies beyond stdlib.

Design
------
- BM25+ variant (Lv & Zhai, 2011) — avoids zero term-frequency problem
- Standard parameters: k1=1.5, b=0.75, delta=1.0
- Documents are tokenised by splitting on whitespace and punctuation;
  stopwords are stripped with a compact built-in list
- `search()` returns dicts in the same shape as VectorStore.search()
  so both can be merged uniformly downstream

Usage
-----
    from services.bm25 import BM25Index
    idx = BM25Index.from_documents(texts, metadatas)
    results = idx.search("hardware interface requirement", k=5)
    # results: [{"score": float, "text": str, <metadata fields>}, ...]
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

# ── Compact stopword list (English) ───────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "can", "could", "not",
    "no", "nor", "so", "yet", "both", "either", "as", "that", "this",
    "these", "those", "it", "its", "we", "they", "he", "she", "i", "you",
    "all", "any", "each", "more", "most", "also", "than", "then", "there",
    "when", "where", "which", "who", "how", "what", "such", "into", "up",
    "out", "about", "over", "after", "before", "between", "through",
})

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:['\-][a-zA-Z0-9]+)*")


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on punctuation/whitespace, remove stopwords."""
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOPWORDS and len(tok) > 1
    ]


class BM25Index:
    """
    In-memory BM25+ index over a list of documents.

    Parameters
    ----------
    k1    : Term saturation parameter (default 1.5)
    b     : Length normalisation parameter (default 0.75)
    delta : BM25+ lower-bound parameter (default 1.0)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        self.k1 = k1
        self.b = b
        self.delta = delta

        # Populated by _build()
        self._doc_tokens: List[List[str]] = []
        self._doc_meta: List[Dict[str, Any]] = []  # parallel metadata list
        self._doc_len: List[int] = []
        self._avg_len: float = 0.0
        self._idf: Dict[str, float] = {}
        self._tf: List[Dict[str, int]] = []  # per-doc term frequency
        self._n_docs: int = 0

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_documents(
        cls,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
    ) -> "BM25Index":
        """Build a BM25 index from a parallel list of texts and metadata dicts."""
        idx = cls(k1=k1, b=b, delta=delta)
        idx._build(texts, metadatas)
        return idx

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self._n_docs = len(texts)
        self._doc_meta = list(metadatas)

        # Tokenise each document and compute TF
        df: Dict[str, int] = {}  # document frequency per term
        for text in texts:
            toks = _tokenize(text)
            self._doc_tokens.append(toks)
            self._doc_len.append(len(toks))

            tf: Dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            self._tf.append(tf)

            for term in tf:
                df[term] = df.get(term, 0) + 1

        self._avg_len = (sum(self._doc_len) / self._n_docs) if self._n_docs else 0.0

        # IDF with BM25+ formulation (add 1 in numerator to avoid zero)
        for term, freq in df.items():
            self._idf[term] = math.log(
                (self._n_docs - freq + 0.5) / (freq + 0.5) + 1.0
            )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k documents by BM25+ score.

        Parameters
        ----------
        query           : Free-text query string.
        k               : Maximum number of results.
        score_threshold : Minimum score to include a result (0.0 = all).

        Returns
        -------
        List of dicts sorted by descending score:
        [{"score": float, "text": str, <metadata fields>}, ...]
        """
        if self._n_docs == 0:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scores: List[float] = [0.0] * self._n_docs

        for term in query_terms:
            if term not in self._idf:
                continue
            idf = self._idf[term]
            for doc_idx in range(self._n_docs):
                tf = self._tf[doc_idx].get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_len[doc_idx]
                norm = 1.0 - self.b + self.b * (dl / self._avg_len) if self._avg_len else 1.0
                # BM25+ score contribution
                scores[doc_idx] += idf * (
                    self.delta + (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
                )

        # Collect results above threshold, sorted descending
        ranked = sorted(
            ((scores[i], i) for i in range(self._n_docs) if scores[i] > score_threshold),
            reverse=True,
        )

        results: List[Dict[str, Any]] = []
        for score, idx in ranked[:k]:
            entry = dict(self._doc_meta[idx])
            entry["score"] = score
            # Ensure the text is accessible (VectorStore stores it under _text)
            if "text" not in entry:
                entry["text"] = entry.get("_text", "")
            results.append(entry)

        return results

    # ── Metadata-overlap retrieval (static helper) ────────────────────────────

    @staticmethod
    def metadata_search(
        query_tokens: List[str],
        metadatas: List[Dict[str, Any]],
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Score documents by overlap between query tokens and their metadata fields.

        Checks: keywords, technical_entities, acronyms, heading, section fields.
        Each matching metadata token scores +1.  Results are sorted descending.

        Parameters
        ----------
        query_tokens : Pre-tokenised query (use _tokenize() from this module).
        metadatas    : List of metadata dicts from the vector store.
        k            : Maximum results to return.

        Returns
        -------
        List of {"score": float, "text": str, <metadata fields>} dicts.
        """
        query_set = set(tok.lower() for tok in query_tokens)
        _META_FIELDS = (
            "keywords", "technical_entities", "acronyms",
            "heading", "section", "subsection", "document_name",
        )

        scored: List[tuple[float, int]] = []
        for idx, meta in enumerate(metadatas):
            score = 0.0
            for field in _META_FIELDS:
                val = meta.get(field)
                if not val:
                    continue
                if isinstance(val, list):
                    tokens = [str(v).lower() for v in val]
                else:
                    tokens = _tokenize(str(val))
                for tok in tokens:
                    if tok in query_set:
                        score += 1.0
            if score > 0.0:
                scored.append((score, idx))

        scored.sort(reverse=True)
        results: List[Dict[str, Any]] = []
        for score, idx in scored[:k]:
            entry = dict(metadatas[idx])
            entry["score"] = score
            if "text" not in entry:
                entry["text"] = entry.get("_text", "")
            results.append(entry)
        return results
