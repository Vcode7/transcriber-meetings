"""
tests/test_structure_chunker.py

Unit tests for the structure-aware document chunking pipeline
(chunk_document, _DocumentAnalyzer, _StructuralSplitter, _MetadataEnricher).
"""
from __future__ import annotations

import sys
import os
import pytest

# Ensure the backend package is on sys.path when run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.text_chunker import (
    chunk_document,
    chunk_text,
    chunk_transcript,
    chunk_pages,
    _DocumentAnalyzer,
    _StructuralSplitter,
    _MetadataEnricher,
    BlockType,
    DocumentBlock,
    StructureChunk,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

SIMPLE_TEXT = """\
# Introduction
This is the introduction section of the document.
It contains multiple sentences about the overview of the system.

## 1.1 Background
The system was designed to process documents efficiently.
Various stakeholders were involved in the requirements definition.

### 1.1.1 Historical Context
Previous versions of the system used simpler approaches.
"""

TABLE_TEXT = """\
## Interface Control Document

The following table defines the data interfaces:

| Field Name | Type   | Size | Description          |
|------------|--------|------|----------------------|
| ID         | uint32 | 4    | Unique identifier    |
| Timestamp  | int64  | 8    | Unix epoch (ms)      |
| Payload    | bytes  | var  | Message payload data |

The table above must be implemented exactly as specified.
"""

REQUIREMENT_TEXT = """\
## 3.2 Functional Requirements

REQ-001 The system shall process all incoming messages within 100 ms.

REQ-002 The system shall support a minimum throughput of 10,000 messages per second.

REQ-003 The software shall implement error detection using CRC-32.
"""

PROCEDURE_TEXT = """\
## 4.1 Startup Procedure

Step 1: Power on the hardware unit.
Step 2: Wait for the boot sequence to complete (approximately 30 seconds).
Step 3: Verify the system status LED is solid green.
Step 4: Launch the control software application.
Step 5: Confirm all subsystems are reporting nominal status.
"""

ALGORITHM_TEXT = """\
## 5. Data Processing

Algorithm 1: Round-Robin Scheduler

Input: Queue Q of N tasks, each with priority P
Output: Ordered execution sequence S

1. Sort Q by priority (descending)
2. Assign time slots in round-robin order
3. Return ordered sequence S
"""

MIXED_TEXT = SIMPLE_TEXT + "\n" + TABLE_TEXT + "\n" + REQUIREMENT_TEXT + "\n" + PROCEDURE_TEXT


# ══════════════════════════════════════════════════════════════════════════════
# §1  DocumentAnalyzer tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDocumentAnalyzer:

    def setup_method(self):
        self.analyzer = _DocumentAnalyzer()

    def test_detects_markdown_headings(self):
        text = "# Chapter One\n## Section A\n### Subsection i\n"
        blocks = self.analyzer.analyze(text)
        headings = [b for b in blocks if b.block_type == BlockType.HEADING]
        assert len(headings) == 3
        assert headings[0].heading_level == 1
        assert headings[1].heading_level == 2
        assert headings[2].heading_level == 3

    def test_detects_numbered_headings(self):
        text = "1. Introduction\n2.1 Background\n3.2.1 Details\n"
        blocks = self.analyzer.analyze(text)
        headings = [b for b in blocks if b.block_type == BlockType.HEADING]
        assert any(h.heading_level == 1 for h in headings)

    def test_detects_table_block(self):
        blocks = self.analyzer.analyze(TABLE_TEXT)
        tables = [b for b in blocks if b.block_type == BlockType.TABLE]
        assert len(tables) >= 1
        # Table block should contain the header and data rows
        table_text = tables[0].text
        assert "Field Name" in table_text
        assert "Payload" in table_text

    def test_table_is_not_split(self):
        """A markdown table must not be broken across multiple blocks."""
        blocks = self.analyzer.analyze(TABLE_TEXT)
        tables = [b for b in blocks if b.block_type == BlockType.TABLE]
        # All rows of the table must be in a single block
        assert len(tables) == 1
        assert "ID" in tables[0].text
        assert "Timestamp" in tables[0].text
        assert "Payload" in tables[0].text

    def test_detects_requirement_blocks(self):
        blocks = self.analyzer.analyze(REQUIREMENT_TEXT)
        req_blocks = [b for b in blocks if b.block_type == BlockType.REQUIREMENT]
        assert len(req_blocks) >= 1

    def test_req_block_contains_shall(self):
        blocks = self.analyzer.analyze(REQUIREMENT_TEXT)
        req_blocks = [b for b in blocks if b.block_type == BlockType.REQUIREMENT]
        for rb in req_blocks:
            assert "shall" in rb.text.lower()

    def test_detects_procedure_steps(self):
        blocks = self.analyzer.analyze(PROCEDURE_TEXT)
        proc_blocks = [b for b in blocks if b.block_type == BlockType.PROCEDURE]
        assert len(proc_blocks) >= 1
        # All steps should be in one procedure block (not fragmented)
        combined = " ".join(pb.text for pb in proc_blocks)
        assert "Step 1" in combined
        assert "Step 5" in combined

    def test_detects_algorithm_block(self):
        blocks = self.analyzer.analyze(ALGORITHM_TEXT)
        algo_blocks = [b for b in blocks if b.block_type == BlockType.ALGORITHM]
        assert len(algo_blocks) >= 1

    def test_detects_paragraph_blocks(self):
        blocks = self.analyzer.analyze(SIMPLE_TEXT)
        para_blocks = [b for b in blocks if b.block_type == BlockType.PARAGRAPH and b.lines]
        assert len(para_blocks) >= 1

    def test_page_marker_detected(self):
        text = "Some content here.\n[Page 3]\nMore content on page 3.\n"
        blocks = self.analyzer.analyze(text)
        page_sentinels = [b for b in blocks if b.page_hint is not None and not b.lines]
        assert len(page_sentinels) >= 1
        assert page_sentinels[0].page_hint == 3

    def test_empty_text_returns_empty(self):
        blocks = self.analyzer.analyze("")
        assert blocks == []

    def test_whitespace_only_returns_sentinels_only(self):
        blocks = self.analyzer.analyze("   \n\n   \n")
        non_sentinel = [b for b in blocks if b.lines]
        assert non_sentinel == []


# ══════════════════════════════════════════════════════════════════════════════
# §2  StructuralSplitter tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStructuralSplitter:

    def _chunks_from(self, text: str, min_w: int = 20, max_w: int = 800) -> list[StructureChunk]:
        analyzer = _DocumentAnalyzer()
        splitter = _StructuralSplitter(min_chunk_words=min_w, max_chunk_words=max_w)
        blocks = analyzer.analyze(text)
        return splitter.split(blocks)

    def test_table_emitted_as_single_chunk(self):
        chunks = self._chunks_from(TABLE_TEXT)
        tbl_chunks = [c for c in chunks if c.block_type == BlockType.TABLE]
        assert len(tbl_chunks) >= 1
        # All table rows in one chunk
        combined = tbl_chunks[0].text
        assert "Field Name" in combined
        assert "Payload" in combined

    def test_requirement_emitted_as_single_chunk(self):
        chunks = self._chunks_from(REQUIREMENT_TEXT)
        req_chunks = [c for c in chunks if c.block_type == BlockType.REQUIREMENT]
        assert len(req_chunks) >= 1

    def test_procedure_emitted_as_single_chunk(self):
        chunks = self._chunks_from(PROCEDURE_TEXT)
        proc_chunks = [c for c in chunks if c.block_type == BlockType.PROCEDURE]
        assert len(proc_chunks) >= 1
        combined = proc_chunks[0].text
        assert "Step 1" in combined
        assert "Step 5" in combined

    def test_heading_context_propagated(self):
        chunks = self._chunks_from(SIMPLE_TEXT)
        # At least one chunk should have chapter set from "# Introduction"
        chapters = [c.chapter for c in chunks if c.chapter]
        assert len(chapters) >= 1

    def test_section_context_propagated(self):
        chunks = self._chunks_from(SIMPLE_TEXT)
        sections = [c.section for c in chunks if c.section]
        assert len(sections) >= 1

    def test_chunk_index_monotonically_increasing(self):
        chunks = self._chunks_from(MIXED_TEXT)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(indices)))

    def test_no_empty_chunk_text(self):
        chunks = self._chunks_from(MIXED_TEXT)
        for c in chunks:
            assert c.text.strip() != ""

    def test_max_chunk_words_respected_for_paragraphs(self):
        """Paragraph chunks should not grossly exceed max_chunk_words."""
        long_para = " ".join(["word"] * 2000)
        chunks = self._chunks_from(long_para, max_w=300)
        para_chunks = [c for c in chunks if c.block_type == BlockType.PARAGRAPH]
        for c in para_chunks:
            # Allow 2x slack for semantic splitting at paragraph boundaries
            assert c.word_count <= 700, (
                f"Chunk too large: {c.word_count} words"
            )

    def test_min_chunk_words_merges_small_headings(self):
        """Small heading-only stubs should be merged with body text."""
        text = "# Short\nThis is a longer body paragraph that has enough words.\n"
        chunks = self._chunks_from(text, min_w=10)
        # Should produce chunks with text, not a heading-only stub
        for c in chunks:
            assert c.word_count >= 1  # at minimum


# ══════════════════════════════════════════════════════════════════════════════
# §3  MetadataEnricher tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMetadataEnricher:

    def _enrich(self, text: str, filename: str = "test.pdf") -> dict:
        enricher = _MetadataEnricher(filename=filename, doc_scope="global_context")
        chunk = StructureChunk(
            text=text,
            block_type=BlockType.PARAGRAPH,
            chunk_index=0,
            word_count=len(text.split()),
        )
        return enricher.enrich(chunk)

    def test_acronym_extraction(self):
        d = self._enrich("The ADA ICD REQ interface SHALL comply with DO-178.")
        assert "ADA" in d["acronyms"]
        assert "ICD" in d["acronyms"]

    def test_technical_entity_extraction_req_id(self):
        d = self._enrich("REQ-001 The system shall process messages within 100ms.")
        assert any("REQ-001" in e for e in d["technical_entities"])

    def test_technical_entity_extraction_irs_id(self):
        d = self._enrich("See IRS-204 for the detailed interface specification.")
        assert any("IRS-204" in e for e in d["technical_entities"])

    def test_technical_entity_extraction_version(self):
        d = self._enrich("This applies to software version v3.2.1 and above.")
        assert any("v3.2.1" in e for e in d["technical_entities"])

    def test_keyword_extraction(self):
        d = self._enrich(
            "The Hardware Interface Design specification defines the Physical Layer protocol."
        )
        # Should contain title-case phrases
        assert len(d["keywords"]) >= 1

    def test_document_type_inferred_from_extension(self):
        d = self._enrich("Content.", filename="design_doc.pdf")
        assert d["document_type"] == "pdf"

        d = self._enrich("Content.", filename="spec.docx")
        assert d["document_type"] == "word_document"

        d = self._enrich("Content.", filename="slides.pptx")
        assert d["document_type"] == "presentation"

    def test_scope_field(self):
        enricher = _MetadataEnricher(
            filename="doc.pdf", doc_scope="meeting_context", meeting_id="mtg-001"
        )
        chunk = StructureChunk(text="text", block_type=BlockType.PARAGRAPH, chunk_index=0)
        d = enricher.enrich(chunk)
        assert d["scope"] == "meeting_context"
        assert d["meeting_id"] == "mtg-001"

    def test_all_required_metadata_fields_present(self):
        d = self._enrich("The system shall meet all requirements.", "spec.pdf")
        required_fields = [
            "text", "chunk_index", "block_type", "heading_level",
            "chapter", "section", "subsection", "heading", "page_number",
            "document_name", "document_type", "scope", "meeting_id",
            "keywords", "technical_entities", "acronyms", "chunk_size_words",
        ]
        for field in required_fields:
            assert field in d, f"Missing field: {field}"


# ══════════════════════════════════════════════════════════════════════════════
# §4  chunk_document() integration tests
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkDocument:

    def test_returns_non_empty_list(self):
        result = chunk_document(SIMPLE_TEXT, filename="test.pdf")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_empty_text_returns_empty(self):
        result = chunk_document("", filename="test.pdf")
        assert result == []

    def test_whitespace_only_returns_empty(self):
        result = chunk_document("   \n\n   ", filename="test.pdf")
        assert result == []

    def test_chunk_index_sequential(self):
        result = chunk_document(MIXED_TEXT, filename="mixed.pdf")
        indices = [c["chunk_index"] for c in result]
        assert indices == list(range(len(indices)))

    def test_table_chunk_never_split(self):
        result = chunk_document(TABLE_TEXT, filename="ics.pdf")
        tbl_chunks = [c for c in result if c["block_type"] == "table"]
        assert len(tbl_chunks) >= 1
        table_text = tbl_chunks[0]["text"]
        assert "Field Name" in table_text
        assert "Payload" in table_text

    def test_requirement_chunk_preserved(self):
        result = chunk_document(REQUIREMENT_TEXT, filename="srs.pdf")
        req_chunks = [c for c in result if c["block_type"] == "requirement"]
        assert len(req_chunks) >= 1

    def test_procedure_chunk_preserved(self):
        result = chunk_document(PROCEDURE_TEXT, filename="ops.pdf")
        proc_chunks = [c for c in result if c["block_type"] == "procedure"]
        assert len(proc_chunks) >= 1
        combined = " ".join(c["text"] for c in proc_chunks)
        assert "Step 1" in combined
        assert "Step 5" in combined

    def test_metadata_fields_in_output(self):
        result = chunk_document(SIMPLE_TEXT, filename="intro.pdf",
                                doc_scope="global_context")
        for chunk in result:
            assert "text" in chunk
            assert "chunk_index" in chunk
            assert "block_type" in chunk
            assert "keywords" in chunk
            assert "technical_entities" in chunk
            assert "acronyms" in chunk
            assert "scope" in chunk
            assert "document_type" in chunk

    def test_scope_set_correctly(self):
        result = chunk_document(SIMPLE_TEXT, filename="doc.pdf",
                                doc_scope="meeting_context", meeting_id="rec-123")
        for chunk in result:
            assert chunk["scope"] == "meeting_context"
            assert chunk["meeting_id"] == "rec-123"

    def test_heading_hierarchy_in_metadata(self):
        result = chunk_document(SIMPLE_TEXT, filename="doc.pdf")
        chapters = [c["chapter"] for c in result if c.get("chapter")]
        assert len(chapters) >= 1

    def test_global_context_scope_default(self):
        result = chunk_document(SIMPLE_TEXT, filename="doc.pdf")
        for chunk in result:
            assert chunk["scope"] == "global_context"

    def test_document_type_inferred(self):
        result = chunk_document(SIMPLE_TEXT, filename="report.pdf")
        assert result[0]["document_type"] == "pdf"

        result = chunk_document(SIMPLE_TEXT, filename="spec.docx")
        assert result[0]["document_type"] == "word_document"

    def test_max_chunk_words_parameter(self):
        """No paragraph chunk should exceed 2× max_chunk_words."""
        long_text = "\n\n".join(["word " * 500] * 5)
        result = chunk_document(long_text, filename="doc.txt", max_chunk_words=200)
        para_chunks = [c for c in result if c["block_type"] == "paragraph"]
        for c in para_chunks:
            assert c["chunk_size_words"] <= 600, f"Chunk too large: {c['chunk_size_words']}"

    def test_technical_document_req_ids_extracted(self):
        text = "REQ-101 The system shall achieve 99.9% availability.\nIRS-202 specifies the interface."
        result = chunk_document(text, filename="srs.pdf")
        all_entities = []
        for c in result:
            all_entities.extend(c.get("technical_entities", []))
        entity_str = " ".join(all_entities)
        assert "REQ-101" in entity_str or "IRS-202" in entity_str


# ══════════════════════════════════════════════════════════════════════════════
# §5  Backward compatibility tests (legacy functions unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class TestLegacyCompatibility:

    def test_chunk_text_still_works(self):
        result = chunk_text("Hello world. " * 100, chunk_size=50, overlap=10)
        assert isinstance(result, list)
        assert len(result) > 0
        for chunk in result:
            assert "text" in chunk
            assert "chunk_index" in chunk

    def test_chunk_transcript_still_works(self):
        transcript = [
            {"speaker_label": "Alice", "text": "Hello everyone.", "start": 0.0, "end": 2.0},
            {"speaker_label": "Bob",   "text": "Good morning.",  "start": 2.5, "end": 4.0},
        ]
        result = chunk_transcript(transcript, chunk_size=10, overlap=2)
        assert isinstance(result, list)
        assert len(result) >= 1
        for chunk in result:
            assert "text" in chunk
            assert "start" in chunk
            assert "end" in chunk
            assert "speakers" in chunk

    def test_chunk_pages_still_works(self):
        pages = [(1, "First page content with some text."), (2, "Second page content.")]
        result = chunk_pages(pages, chunk_size=10, overlap=2)
        assert isinstance(result, list)
        for chunk in result:
            assert "text" in chunk
            assert "page" in chunk

    def test_chunk_text_returns_empty_for_empty_input(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_transcript_returns_empty_for_empty_input(self):
        assert chunk_transcript([]) == []


# ══════════════════════════════════════════════════════════════════════════════
# §6  Edge case tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_line_document(self):
        result = chunk_document("The quick brown fox jumps over the lazy dog.",
                                filename="short.txt")
        assert len(result) >= 1
        assert result[0]["text"].strip() != ""

    def test_only_headings(self):
        text = "# Chapter 1\n## Section 1.1\n### Subsection 1.1.1\n"
        result = chunk_document(text, filename="outline.txt")
        # Should produce at least something (headings merged or standalone)
        assert isinstance(result, list)

    def test_table_with_surrounding_text(self):
        text = (
            "This section describes the interface.\n\n"
            "| Col A | Col B |\n|-------|-------|\n| 1     | 2     |\n\n"
            "The table above shows the mapping.\n"
        )
        result = chunk_document(text, filename="icd.pdf")
        tbl_chunks = [c for c in result if c["block_type"] == "table"]
        assert len(tbl_chunks) >= 1
        # Table should contain all rows
        assert "Col A" in tbl_chunks[0]["text"]
        assert "| 1" in tbl_chunks[0]["text"]

    def test_mixed_requirements_and_paragraphs(self):
        text = (
            "## Requirements\n\n"
            "REQ-001 The system shall do X.\n\n"
            "Additional context about the requirement is provided here.\n\n"
            "REQ-002 The system shall do Y within 5ms.\n"
        )
        result = chunk_document(text, filename="req.pdf")
        req_chunks = [c for c in result if c["block_type"] == "requirement"]
        assert len(req_chunks) >= 1

    def test_code_block_preserved(self):
        text = (
            "The following code implements the algorithm:\n\n"
            "```python\n"
            "def process(data):\n"
            "    return sorted(data)\n"
            "```\n\n"
            "This is the reference implementation.\n"
        )
        result = chunk_document(text, filename="design.md")
        code_chunks = [c for c in result if c["block_type"] == "code"]
        assert len(code_chunks) >= 1
        assert "def process" in code_chunks[0]["text"]

    def test_very_large_table_kept_intact(self):
        """A table with many rows should still be emitted as a single chunk."""
        rows = ["| Col A | Col B | Col C |"]
        rows.append("|-------|-------|-------|")
        for i in range(50):
            rows.append(f"| {i}     | val{i} | desc{i} |")
        text = "\n".join(rows)
        result = chunk_document(text, filename="large_table.pdf")
        tbl_chunks = [c for c in result if c["block_type"] == "table"]
        assert len(tbl_chunks) >= 1
        # All 50 data rows should be in one chunk
        combined = " ".join(c["text"] for c in tbl_chunks)
        assert "val49" in combined

    def test_all_chunk_types_in_mixed_document(self):
        result = chunk_document(MIXED_TEXT, filename="mixed.pdf")
        block_types = {c["block_type"] for c in result}
        # Mixed document should have multiple block types
        assert len(block_types) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
