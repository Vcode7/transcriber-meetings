"""
tests/test_stage2_enhance.py

Unit tests for Stage 2 redesign components:
  - BM25 index: scoring, tokenisation, document frequency
  - RRF merge: rank fusion ordering, deduplication
  - BM25 metadata_search: entity/acronym/keyword field overlap
  - Window boundary alignment: correct window slices
  - Context usage report schema validation
"""
from __future__ import annotations

import json
import math
import sys
import os
import pytest

# Ensure backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.bm25 import BM25Index, _tokenize, _STOPWORDS


# ══════════════════════════════════════════════════════════════════
# Tokeniser tests
# ══════════════════════════════════════════════════════════════════

class TestTokenize:
    def test_lowercase_and_split(self):
        tokens = _tokenize("Software Requirements Specification")
        assert "software" in tokens
        assert "requirements" in tokens
        assert "specification" in tokens

    def test_stopwords_removed(self):
        tokens = _tokenize("the interface is defined in the document")
        for stop in ["the", "in", "is"]:
            assert stop not in tokens

    def test_single_char_removed(self):
        tokens = _tokenize("a b c delta epsilon")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "delta" in tokens

    def test_hyphenated_terms_preserved(self):
        tokens = _tokenize("signal-to-noise ratio")
        assert any("signal" in t or "noise" in t for t in tokens)

    def test_empty_string(self):
        assert _tokenize("") == []


# ══════════════════════════════════════════════════════════════════
# BM25Index tests
# ══════════════════════════════════════════════════════════════════

_DOCS = [
    "The hardware interface specification defines all signal types and voltage levels.",
    "Software requirements shall be verified through unit testing and integration testing.",
    "The algorithm processes input data using a sliding window of fixed size.",
    "Requirements traceability matrix links each requirement to its verification method.",
    "Signal processing involves Fourier transforms and digital filtering techniques.",
]
_METAS = [
    {"_text": _DOCS[0], "filename": "HW_IF_SPEC.pdf",  "section": "Interface Definition"},
    {"_text": _DOCS[1], "filename": "SW_REQS.pdf",      "section": "Verification"},
    {"_text": _DOCS[2], "filename": "ALGO_DESIGN.pdf",  "section": "Algorithm"},
    {"_text": _DOCS[3], "filename": "RTM.pdf",          "section": "Traceability"},
    {"_text": _DOCS[4], "filename": "DSP.pdf",          "section": "Signal Processing"},
]


class TestBM25IndexBasic:
    @pytest.fixture(autouse=True)
    def build_index(self):
        self.idx = BM25Index.from_documents(_DOCS, _METAS)

    def test_n_docs(self):
        assert self.idx._n_docs == 5

    def test_avg_len_positive(self):
        assert self.idx._avg_len > 0

    def test_idf_computed(self):
        # "requirements" appears in docs 1 and 3 → IDF should be positive
        assert self.idx._idf.get("requirements", 0) > 0

    def test_search_returns_list(self):
        results = self.idx.search("hardware interface", k=3)
        assert isinstance(results, list)

    def test_hardware_interface_top_result(self):
        results = self.idx.search("hardware interface signal", k=3)
        assert len(results) > 0
        # Document 0 should score highest (most overlap)
        assert results[0].get("filename") == "HW_IF_SPEC.pdf"

    def test_requirements_query(self):
        results = self.idx.search("software requirements verification", k=3)
        filenames = [r.get("filename") for r in results]
        assert "SW_REQS.pdf" in filenames

    def test_score_threshold_filters(self):
        results = self.idx.search("hardware", k=5, score_threshold=999.0)
        assert results == []

    def test_k_limits_results(self):
        results = self.idx.search("requirements", k=2)
        assert len(results) <= 2

    def test_empty_query(self):
        results = self.idx.search("")
        assert results == []

    def test_unknown_term(self):
        results = self.idx.search("xyzzy_nonexistent_term_12345")
        assert results == []

    def test_text_in_results(self):
        results = self.idx.search("Fourier transform digital filter", k=1)
        assert len(results) == 1
        assert "text" in results[0]

    def test_scores_descending(self):
        results = self.idx.search("requirements traceability", k=5)
        scores = [r.get("score", 0) for r in results]
        assert scores == sorted(scores, reverse=True)


class TestBM25IndexEmpty:
    def test_empty_corpus(self):
        idx = BM25Index.from_documents([], [])
        assert idx.search("anything") == []

    def test_single_document(self):
        idx = BM25Index.from_documents(["Only document"], [{"_text": "Only document"}])
        results = idx.search("document")
        assert len(results) == 1


# ══════════════════════════════════════════════════════════════════
# Metadata search tests
# ══════════════════════════════════════════════════════════════════

class TestMetadataSearch:
    def test_keyword_field_match(self):
        metas = [
            {"_text": "doc1", "keywords": ["SRS", "interface", "signal"]},
            {"_text": "doc2", "keywords": ["algorithm", "window", "processing"]},
            {"_text": "doc3", "keywords": ["traceability", "matrix", "requirement"]},
        ]
        query_tokens = _tokenize("SRS interface")
        results = BM25Index.metadata_search(query_tokens, metas, k=3)
        assert len(results) > 0
        # doc1 has both "srs" (as lowercased) and "interface" → should be top
        assert results[0].get("_text") == "doc1"

    def test_acronym_field_match(self):
        metas = [
            {"_text": "doc_a", "acronyms": ["SRS", "ICD", "ICF"]},
            {"_text": "doc_b", "acronyms": ["TRM", "FDA"]},
        ]
        query_tokens = _tokenize("SRS ICD specification")
        results = BM25Index.metadata_search(query_tokens, metas, k=2)
        assert results[0].get("_text") == "doc_a"

    def test_no_match_returns_empty(self):
        metas = [{"_text": "doc", "keywords": ["foo", "bar"]}]
        query_tokens = _tokenize("xyzzy nonexistent")
        results = BM25Index.metadata_search(query_tokens, metas, k=5)
        assert results == []

    def test_empty_metadata_list(self):
        results = BM25Index.metadata_search(["test"], [], k=5)
        assert results == []

    def test_text_field_populated(self):
        metas = [{"_text": "chunk text here", "keywords": ["chunk", "text"]}]
        results = BM25Index.metadata_search(["chunk"], metas, k=1)
        assert results[0]["text"] == "chunk text here"


# ══════════════════════════════════════════════════════════════════
# RRF merge tests (inline implementation matching rom_service logic)
# ══════════════════════════════════════════════════════════════════

def _rrf_merge(result_lists, k=60):
    """Extracted RRF logic for standalone testing."""
    rrf_scores = {}
    best_entry = {}
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


class TestRRFMerge:
    def test_single_list_passthrough(self):
        results = [
            [{"_text": "alpha", "score": 0.9}, {"_text": "beta", "score": 0.7}]
        ]
        merged = _rrf_merge(results)
        assert [m["_text"] for m in merged] == ["alpha", "beta"]

    def test_two_lists_deduplication(self):
        """Same document ranked in both lists → appears only once."""
        list_a = [{"_text": "shared", "score": 0.8}, {"_text": "only_a", "score": 0.6}]
        list_b = [{"_text": "shared", "score": 0.7}, {"_text": "only_b", "score": 0.5}]
        merged = _rrf_merge([list_a, list_b])
        texts = [m["_text"] for m in merged]
        assert texts.count("shared") == 1

    def test_top_ranked_in_all_lists_comes_first(self):
        """A document top-ranked in both lists should score highest via RRF."""
        list_a = [{"_text": "winner", "score": 0.9}, {"_text": "loser", "score": 0.5}]
        list_b = [{"_text": "winner", "score": 0.8}, {"_text": "middle", "score": 0.4}]
        merged = _rrf_merge([list_a, list_b])
        assert merged[0]["_text"] == "winner"

    def test_empty_lists(self):
        assert _rrf_merge([[], []]) == []

    def test_empty_text_entries_ignored(self):
        merged = _rrf_merge([[{"_text": "", "score": 0.9}, {"_text": "real", "score": 0.7}]])
        texts = [m.get("_text") for m in merged]
        assert "" not in texts
        assert "real" in texts

    def test_three_way_merge_ordering(self):
        """Three-way RRF: a document top-ranked in all three should beat one ranked only in one."""
        # "top" is first in all three lists: RRF score ≈ 3 * 1/(60+1)
        # "only_c" is first in only one list but in last position in others
        list_a = [{"_text": "top"}, {"_text": "only_c"}]
        list_b = [{"_text": "top"}, {"_text": "other"}]
        list_c = [{"_text": "only_c"}, {"_text": "top"}]
        merged = _rrf_merge([list_a, list_b, list_c])
        # "top" should beat "only_c" due to three-list ranking advantage
        texts = [m["_text"] for m in merged]
        assert texts.index("top") < texts.index("only_c")


# ══════════════════════════════════════════════════════════════════
# Window boundary alignment tests
# ══════════════════════════════════════════════════════════════════

class TestWindowBoundary:
    """Verify that window slicing produces the correct sub-lists."""

    @staticmethod
    def _slice_windows(points, window_size):
        """Replicate the window slicing from enhance_discussion_points."""
        windows = []
        for i in range(0, len(points), window_size):
            windows.append(points[i: i + window_size])
        return windows

    def test_exact_multiple(self):
        points = list(range(10))
        windows = self._slice_windows(points, 5)
        assert len(windows) == 2
        assert windows[0] == [0, 1, 2, 3, 4]
        assert windows[1] == [5, 6, 7, 8, 9]

    def test_remainder_window(self):
        points = list(range(8))
        windows = self._slice_windows(points, 5)
        assert len(windows) == 2
        assert windows[-1] == [5, 6, 7]  # partial last window

    def test_single_window(self):
        points = list(range(3))
        windows = self._slice_windows(points, 5)
        assert len(windows) == 1
        assert windows[0] == [0, 1, 2]

    def test_empty_points(self):
        assert self._slice_windows([], 5) == []

    def test_window_larger_than_total(self):
        points = list(range(4))
        windows = self._slice_windows(points, 10)
        assert len(windows) == 1
        assert windows[0] == points

    def test_window_size_three(self):
        points = list(range(7))
        windows = self._slice_windows(points, 3)
        assert len(windows) == 3  # [0-2], [3-5], [6]
        assert windows[2] == [6]


# ══════════════════════════════════════════════════════════════════
# Context usage report schema tests
# ══════════════════════════════════════════════════════════════════

class TestContextUsageReport:
    """Validate that the context_usage_report normalisation logic is correct."""

    @staticmethod
    def _normalise_report(raw: dict, has_meeting: bool, has_global: bool) -> dict:
        """Replicate the normalisation logic from enhance_discussion_points."""
        return {
            "verified": bool(raw.get("verified", False)),
            "technical_details_added": bool(raw.get("technical_details_added", False)),
            "abbreviations_expanded": bool(raw.get("abbreviations_expanded", False)),
            "references_added": bool(raw.get("references_added", False)),
            "terminology_clarified": bool(raw.get("terminology_clarified", False)),
            "no_useful_context": bool(
                raw.get("no_useful_context", not has_meeting and not has_global)
            ),
            "meeting_context_docs": (
                raw.get("meeting_context_docs")
                if isinstance(raw.get("meeting_context_docs"), list)
                else []
            ),
            "global_context_docs": (
                raw.get("global_context_docs")
                if isinstance(raw.get("global_context_docs"), list)
                else []
            ),
        }

    def test_all_fields_present(self):
        raw = {
            "verified": True,
            "technical_details_added": False,
            "abbreviations_expanded": True,
            "references_added": False,
            "terminology_clarified": True,
            "no_useful_context": False,
            "meeting_context_docs": ["spec.pdf"],
            "global_context_docs": ["iso.pdf"],
        }
        result = self._normalise_report(raw, True, True)
        assert result["verified"] is True
        assert result["abbreviations_expanded"] is True
        assert result["technical_details_added"] is False
        assert result["meeting_context_docs"] == ["spec.pdf"]

    def test_no_context_defaults(self):
        """When no meeting/global context is available, no_useful_context should default True."""
        result = self._normalise_report({}, has_meeting=False, has_global=False)
        assert result["no_useful_context"] is True

    def test_context_present_default_false(self):
        """When context is available, no_useful_context defaults False."""
        result = self._normalise_report({}, has_meeting=True, has_global=False)
        assert result["no_useful_context"] is False

    def test_non_list_docs_replaced_with_empty(self):
        raw = {"meeting_context_docs": "spec.pdf", "global_context_docs": None}
        result = self._normalise_report(raw, True, True)
        assert result["meeting_context_docs"] == []
        assert result["global_context_docs"] == []

    def test_all_bool_fields_coerced(self):
        raw = {
            "verified": 1,                # truthy int
            "technical_details_added": 0, # falsy int
            "abbreviations_expanded": "yes",  # truthy string
        }
        result = self._normalise_report(raw, True, True)
        assert result["verified"] is True
        assert result["technical_details_added"] is False
        assert result["abbreviations_expanded"] is True

    def test_required_keys_always_present(self):
        result = self._normalise_report({}, True, True)
        required = [
            "verified", "technical_details_added", "abbreviations_expanded",
            "references_added", "terminology_clarified", "no_useful_context",
            "meeting_context_docs", "global_context_docs",
        ]
        for key in required:
            assert key in result


# ══════════════════════════════════════════════════════════════════
# BM25 IDF formula test
# ══════════════════════════════════════════════════════════════════

class TestBM25IDFFormula:
    def test_rare_term_higher_idf(self):
        """A term appearing in fewer documents should have a higher IDF score."""
        docs = [
            "common term appears here",
            "common term appears there",
            "rare term only once",
        ]
        metas = [{"_text": d} for d in docs]
        idx = BM25Index.from_documents(docs, metas)

        idf_common = idx._idf.get("common", 0)
        idf_rare   = idx._idf.get("rare", 0)
        assert idf_rare > idf_common

    def test_universal_term_lowest_idf(self):
        """A term that appears in every document should have the lowest IDF."""
        docs = [
            "universal alpha",
            "universal beta",
            "universal gamma",
        ]
        metas = [{"_text": d} for d in docs]
        idx = BM25Index.from_documents(docs, metas)

        idf_universal = idx._idf.get("universal", 0)
        idf_alpha     = idx._idf.get("alpha", 0)
        assert idf_alpha >= idf_universal


# ══════════════════════════════════════════════════════════════════
# Stage 2 Min Similarity Threshold Tests
# ══════════════════════════════════════════════════════════════════

class TestMinSimilarityThreshold:
    def test_threshold_filtering(self):
        """Chunks with similarity < threshold should be filtered out."""
        items = [
            {"_text": "High relevance match", "_similarity_score": 0.85},
            {"_text": "Low relevance match", "_similarity_score": 0.65},
            {"_text": "Medium relevance match", "_similarity_score": 0.79},
        ]
        thresh = 0.80
        filtered = [i for i in items if i["_similarity_score"] >= thresh]
        assert len(filtered) == 1
        assert filtered[0]["_text"] == "High relevance match"

    def test_disabled_threshold_preserves_all(self):
        """When threshold is None or 0.0, all chunks are preserved."""
        items = [
            {"_text": "Match 1", "_similarity_score": 0.40},
            {"_text": "Match 2", "_similarity_score": 0.75},
        ]
        thresh = None
        filtered = items if thresh is None else [i for i in items if i["_similarity_score"] >= thresh]
        assert len(filtered) == 2
