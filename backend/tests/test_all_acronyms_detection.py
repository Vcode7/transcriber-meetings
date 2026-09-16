import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.correction_service import (
    detect_acronyms,
    apply_acronym_expansions_to_transcript,
    _CONTIGUOUS_ACRONYM_PATTERN,
    _SPACED_ACRONYM_PATTERN,
    _DOTTED_ACRONYM_PATTERN,
    _build_acronym_pattern,
)
from services.dictionary_service import make_pattern, expand_terms_in_chunks
from services.text_chunker import _TechnicalTerminologyExtractor, _MetadataEnricher, StructureChunk


class TestAllAcronymsDetection(unittest.TestCase):
    def test_regex_patterns_direct(self):
        """Test regex pattern matching directly across contiguous, spaced, and dotted forms."""
        sample_text = (
            "We discussed LLM models and L L M benchmarks. "
            "Our IB team and I B division compared API specs. "
            "We visited U.S.A. and evaluated A.I. tools. "
            "I am sure that he is happy, and I Saw it on time."
        )

        contiguous = _CONTIGUOUS_ACRONYM_PATTERN.findall(sample_text)
        self.assertIn("LLM", contiguous)
        self.assertIn("IB", contiguous)
        self.assertIn("API", contiguous)

        spaced = _SPACED_ACRONYM_PATTERN.findall(sample_text)
        self.assertIn("L L M", spaced)
        self.assertIn("I B", spaced)
        self.assertNotIn("I Saw", spaced)
        self.assertNotIn("he is", spaced)

        dotted = _DOTTED_ACRONYM_PATTERN.findall(sample_text)
        self.assertTrue(any("U.S.A" in d for d in dotted))
        self.assertTrue(any("A.I" in d for d in dotted))

    def test_detect_acronyms_async(self):
        """Test detect_acronyms with contiguous, spaced, and dotted tokens plus DB matching."""
        async def run_test():
            plain_text = (
                "Today the I B department reviewed L L M architectures and standard API interfaces. "
                "Also checked U.S.A. operations. "
                "I am writing this report so that we can do it on time."
            )

            # Mock DB execute returning known acronyms
            mock_db = AsyncMock()
            mock_cursor = MagicMock()
            # Let's say "LLM" and "API" are known in user's dictionary
            mock_cursor.fetchall.return_value = [
                ("LLM", "Large Language Model"),
                ("API", "Application Programming Interface"),
            ]
            mock_db.execute.return_value = mock_cursor

            results = await detect_acronyms(plain_text, "user-123", mock_db)
            results_by_acr = {r["acronym"]: r for r in results}

            # Contiguous
            self.assertIn("API", results_by_acr)
            self.assertTrue(results_by_acr["API"]["is_known"])
            self.assertEqual(results_by_acr["API"]["full_form"], "Application Programming Interface")

            # Spaced "L L M" canonicalized to "LLM"
            self.assertIn("LLM", results_by_acr)
            self.assertTrue(results_by_acr["LLM"]["is_known"])
            self.assertEqual(results_by_acr["LLM"]["full_form"], "Large Language Model")

            # Spaced "I B" canonicalized to "IB" (unknown)
            self.assertIn("IB", results_by_acr)
            self.assertFalse(results_by_acr["IB"]["is_known"])
            self.assertIsNone(results_by_acr["IB"]["full_form"])

            # Dotted "U.S.A." canonicalized to "USA" (unknown)
            self.assertIn("USA", results_by_acr)
            self.assertFalse(results_by_acr["USA"]["is_known"])

            # Common words should NOT be in results
            for skipped in ["I", "AM", "SO", "WE", "DO", "IT", "ON"]:
                self.assertNotIn(skipped, results_by_acr)

        asyncio.run(run_test())

    def test_apply_acronym_expansions_with_spaced_and_dotted(self):
        """Test that apply_acronym_expansions_to_transcript replaces spaced and dotted tokens."""
        transcript = [
            {"text": "We are evaluating L L M performance in this benchmark."},
            {"text": "The second segment mentions LLM again."},
            {"text": "John works in I B trading."},
            {"text": "The headquarters are in U.S.A. today."},
        ]

        decisions = [
            {
                "acronym": "LLM",
                "full_form": "Large Language Model",
                "enabled": True,
                "not_an_acronym": False,
            },
            {
                "acronym": "IB",
                "full_form": "Investment Banking",
                "enabled": True,
                "not_an_acronym": False,
            },
            {
                "acronym": "USA",
                "full_form": "United States of America",
                "enabled": True,
                "not_an_acronym": False,
            },
        ]

        modified = apply_acronym_expansions_to_transcript(transcript, decisions)

        # First occurrence of LLM was "L L M" -> should be expanded to LLM (Large Language Model)
        self.assertIn("LLM (Large Language Model)", modified[0]["text"])
        self.assertNotIn("L L M", modified[0]["text"])

        # Second occurrence of LLM -> should NOT be expanded (first-occurrence only rule)
        self.assertEqual(modified[1]["text"], "The second segment mentions LLM again.")

        # "I B" -> should be expanded to IB (Investment Banking)
        self.assertIn("IB (Investment Banking)", modified[2]["text"])
        self.assertNotIn("I B", modified[2]["text"])

        # "U.S.A." -> should be expanded to USA (United States of America)
        self.assertIn("USA (United States of America)", modified[3]["text"])

    def test_text_chunker_spaced_and_dotted_extraction(self):
        """Test that _TechnicalTerminologyExtractor and _MetadataEnricher handle spaced & dotted acronyms."""
        extractor = _TechnicalTerminologyExtractor()
        extracted = extractor.extract("We implement Retrieval-Augmented Generation (R A G) and Large Language Model (L.L.M.).")
        self.assertIn("RAG", extracted["acronym_mappings"])
        self.assertEqual(extracted["acronym_mappings"]["RAG"], "Retrieval-Augmented Generation")
        self.assertIn("LLM", extracted["acronym_mappings"])
        self.assertEqual(extracted["acronym_mappings"]["LLM"], "Large Language Model")

        from services.text_chunker import BlockType
        enricher = _MetadataEnricher()
        chunk = StructureChunk(
            text="The I B analysts use L L M tools across U.S.A. branches.",
            block_type=BlockType.PARAGRAPH,
            chunk_index=0,
        )
        enriched = enricher.enrich(chunk)
        self.assertIn("IB", enriched["acronyms"])
        self.assertIn("LLM", enriched["acronyms"])
        self.assertIn("USA", enriched["acronyms"])

    def test_dictionary_service_make_pattern_variants(self):
        """Test dictionary_service make_pattern matching spaced and dotted variants."""
        p_ib = make_pattern("IB")
        self.assertTrue(p_ib.search("Working in IB."))
        self.assertTrue(p_ib.search("Working in I B."))
        self.assertTrue(p_ib.search("Working in I.B."))
        self.assertFalse(p_ib.search("RIBBON has a knot"))

        chunks = [{"text": "We test with I B and L L M."}]
        shortcuts = [
            {"shortcut": "IB", "full_form": "Investment Banking"},
            {"shortcut": "LLM", "full_form": "Large Language Model"},
        ]
        expanded = expand_terms_in_chunks(chunks, shortcuts)
        self.assertIn("Investment Banking", expanded[0])
        self.assertIn("Large Language Model", expanded[0])


if __name__ == "__main__":
    unittest.main()
