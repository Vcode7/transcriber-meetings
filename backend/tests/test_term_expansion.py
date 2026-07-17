import sys
import unittest
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.dictionary_service import make_pattern, expand_terms_in_chunks

class TestTermExpansion(unittest.TestCase):
    def test_make_pattern_word_boundaries(self):
        """Test that make_pattern enforces word boundaries correctly for both alphanumeric and special symbols."""
        # Alphanumeric shortcut
        p_llm = make_pattern("LLM")
        self.assertTrue(p_llm.search("This is an LLM."))
        self.assertTrue(p_llm.search("llm is great"))
        self.assertFalse(p_llm.search("GLLM is not matched"))
        self.assertFalse(p_llm.search("LLMs is not matched"))

        # Special symbol shortcuts
        p_net = make_pattern(".NET")
        self.assertTrue(p_net.search("We use .NET Core."))
        self.assertTrue(p_net.search("dotnet is different but .net works"))
        self.assertFalse(p_net.search("vb.netcore is not matched"))

        p_cpp = make_pattern("C++")
        self.assertTrue(p_cpp.search("C++ developer needed"))
        self.assertTrue(p_cpp.search("c++ is fast"))
        self.assertFalse(p_cpp.search("VC++ is not matched"))

    def test_expand_terms_basic(self):
        """Test basic term expansion, case-insensitivity, and original casing preservation."""
        shortcuts = [
            {"shortcut": "LLM", "full_form": "Large Language Model"},
            {"shortcut": "RAG", "full_form": "Retrieval-Augmented Generation"}
        ]
        chunks = [
            {"text": "We are using an llm for this RAG application."}
        ]
        
        expanded = expand_terms_in_chunks(chunks, shortcuts)
        self.assertEqual(len(expanded), 1)
        # Verify that 'llm' is expanded to 'llm (Large Language Model)' (casing preserved for shortcut)
        # Verify that 'RAG' is expanded to 'RAG (Retrieval-Augmented Generation)'
        self.assertIn("llm (Large Language Model)", expanded[0])
        self.assertIn("RAG (Retrieval-Augmented Generation)", expanded[0])

    def test_expand_terms_once_only(self):
        """Test that each term is expanded at most once across all chunks."""
        shortcuts = [
            {"shortcut": "LLM", "full_form": "Large Language Model"}
        ]
        chunks = [
            {"text": "LLM is the first term."},
            {"text": "The second chunk also mentions LLM."}
        ]

        expanded = expand_terms_in_chunks(chunks, shortcuts)
        self.assertEqual(len(expanded), 2)
        # First chunk should be expanded
        self.assertIn("LLM (Large Language Model)", expanded[0])
        # Second chunk should NOT be expanded (only once per document)
        self.assertNotIn("LLM (Large Language Model)", expanded[1])
        self.assertIn("LLM", expanded[1])

    def test_expand_terms_nearby_check(self):
        """Test that expansion is skipped if the full form is already present nearby."""
        shortcuts = [
            {"shortcut": "LLM", "full_form": "Large Language Model"}
        ]
        
        # 1. Full form exists inside the same chunk (within 120 chars)
        chunks1 = [{"text": "We use an LLM (Large Language Model) here."}]
        expanded1 = expand_terms_in_chunks(chunks1, shortcuts)
        self.assertEqual(expanded1[0], "We use an LLM (Large Language Model) here.") # Unchanged

        chunks2 = [{"text": "Large Language Model is what we mean by LLM."}]
        expanded2 = expand_terms_in_chunks(chunks2, shortcuts)
        self.assertEqual(expanded2[0], "Large Language Model is what we mean by LLM.") # Unchanged

        # 2. Full form is far away (more than 120 chars) - should be expanded
        padding = "a" * 130
        chunks3 = [{"text": f"Large Language Model {padding} LLM"}]
        expanded3 = expand_terms_in_chunks(chunks3, shortcuts)
        self.assertIn("LLM (Large Language Model)", expanded3[0])

        # 3. Full form is nearby in an adjacent chunk
        chunks4 = [
            {"text": "This document introduces the Large Language Model concept."},
            {"text": "LLM is very useful."}
        ]
        # In this case:
        # local_context for chunk 1: "\nThis document introduces the Large Language Model concept.\nLLM is very useful."
        # The match for LLM is at the start of chunk 1 (since it searches chunk-by-chunk).
        # Wait, the match is in chunk 1? No, chunk 1 doesn't contain "LLM", chunk 2 does.
        # So we process chunk 2:
        # local_context for chunk 2: "This document introduces the Large Language Model concept.\nLLM is very useful.\n"
        # The distance between "Large Language Model" and "LLM" is around 20-30 characters (within 120).
        # Thus, it should NOT expand LLM in chunk 2.
        expanded4 = expand_terms_in_chunks(chunks4, shortcuts)
        self.assertNotIn("LLM (Large Language Model)", expanded4[1])
        self.assertIn("LLM", expanded4[1])
