"""
text_chunker.py — Structure-aware document chunking for the RAG pipeline.

Public API
----------
chunk_document(text, filename, ...)
    NEW: Structure-aware entry point for technical/ADA documents.
    Detects and preserves logical document structure (headings, tables,
    requirements, procedures, algorithms, equations, figures) and returns
    chunks with rich metadata.

chunk_text(text, chunk_size, overlap)
    LEGACY: Word-count based chunking (sentence-aware).  Still used as a
    fallback inside chunk_document() for plain-text bodies and as a direct
    entry point for callers that have not yet migrated.

chunk_transcript(transcript, chunk_size, overlap)
    Chunk a transcript segment list into text windows, preserving speaker
    labels.  Unchanged from previous version.

chunk_pages(pages, chunk_size, overlap)
    Chunk a list of (page_num, text) tuples, preserving page metadata.
    Unchanged from previous version.

Design
------
chunk_document() is implemented as a three-layer pipeline:

  Layer 1 — _DocumentAnalyzer
      Scans raw text line-by-line and emits a flat sequence of
      DocumentBlock objects, each tagged with a BlockType and optional
      structural attributes (heading level, heading text, page hint).

  Layer 2 — _StructuralSplitter
      Receives the block sequence and groups blocks into StructureChunk
      objects following these rules:
        • Atomic blocks (TABLE, REQUIREMENT, PROCEDURE, ALGORITHM,
          EQUATION, FIGURE) are NEVER split — emitted as single chunks.
        • Paragraph blocks are accumulated under the current heading
          context and flushed when the word count reaches max_chunk_words.
        • Overflow paragraphs are split at blank-line / paragraph
          boundaries (semantic split), never mid-paragraph.
        • Heading-only stubs smaller than min_chunk_words are merged with
          the following block.

  Layer 3 — _MetadataEnricher
      Adds per-chunk metadata:
        keywords         — top significant noun phrases (heuristic regex)
        technical_entities — requirement IDs, version strings, part numbers
        acronyms         — all-caps tokens 2-8 chars
        page_number      — estimated from embedded [Page N] markers
        document_type    — inferred from filename extension
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# §1  Timestamp formatter (shared with transcript chunker)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# §2  Block type taxonomy
# ══════════════════════════════════════════════════════════════════════════════

class BlockType(str, Enum):
    """Semantic type assigned to each detected document block."""
    PARAGRAPH   = "paragraph"
    HEADING     = "heading"
    TABLE       = "table"
    REQUIREMENT = "requirement"
    PROCEDURE   = "procedure"
    ALGORITHM   = "algorithm"
    EQUATION    = "equation"
    FIGURE      = "figure"
    LIST        = "list"
    CODE        = "code"


# Atomic block types are never split across chunks
_ATOMIC_BLOCK_TYPES: frozenset[BlockType] = frozenset({
    BlockType.TABLE,
    BlockType.REQUIREMENT,
    BlockType.PROCEDURE,
    BlockType.ALGORITHM,
    BlockType.EQUATION,
    BlockType.FIGURE,
    BlockType.CODE,
})


# ══════════════════════════════════════════════════════════════════════════════
# §3  Internal data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DocumentBlock:
    """A contiguous region of text with a single semantic type."""
    block_type:    BlockType
    lines:         List[str]          # raw text lines belonging to this block
    heading_level: int = 0            # 0=body, 1=chapter, 2=section, 3=subsection …
    heading_text:  Optional[str] = None
    page_hint:     Optional[int] = None   # page number if detectable

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class StructureChunk:
    """A finalised chunk ready for embedding and metadata enrichment."""
    text:          str
    block_type:    BlockType
    chunk_index:   int
    heading_level: int = 0
    chapter:       Optional[str] = None
    section:       Optional[str] = None
    subsection:    Optional[str] = None
    heading:       Optional[str] = None
    page_number:   Optional[int] = None
    word_count:    int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text":          self.text,
            "chunk_index":   self.chunk_index,
            "block_type":    self.block_type.value,
            "heading_level": self.heading_level,
            "chapter":       self.chapter,
            "section":       self.section,
            "subsection":    self.subsection,
            "heading":       self.heading,
            "page_number":   self.page_number,
            "chunk_size_words": self.word_count,
        }


# ══════════════════════════════════════════════════════════════════════════════
# §4  Layer 1 — Document Analyzer
# ══════════════════════════════════════════════════════════════════════════════

# ── Heading patterns ──────────────────────────────────────────────────────────
# Markdown-style
_RE_MD_H1 = re.compile(r"^#\s+(.+)$")
_RE_MD_H2 = re.compile(r"^##\s+(.+)$")
_RE_MD_H3 = re.compile(r"^###\s+(.+)$")
_RE_MD_H4 = re.compile(r"^####\s+(.+)$")

# Numbered section headers  (1. / 1.2 / 1.2.3 / 1.2.3.4  Title Text)
_RE_NUM_HEAD = re.compile(
    r"^(\d+(?:\.\d+){0,3})\.?\s{1,4}([A-Z].{2,80})$"
)

# ALL CAPS headings (3-12 words, not a requirement line)
_RE_ALL_CAPS = re.compile(r"^([A-Z][A-Z\s\-/&,:]{5,80}[A-Z])$")

# RST-style underline headings  (previous line + underline of = or -)
# Detected in a 2-line look-back in the analyzer loop.

# Section/Chapter/Appendix keywords
_RE_SECTION_KW = re.compile(
    r"^(SECTION|CHAPTER|APPENDIX|ANNEX|PART)\s+[\dA-Z][\d\w]*[:\.]?\s*(.*)$",
    re.IGNORECASE,
)

# ── Atomic block patterns ─────────────────────────────────────────────────────
# Table: markdown table row  |...|  or ASCII grid  +---+---+
_RE_TABLE_ROW    = re.compile(r"^\|.+\|$")
_RE_TABLE_GRID   = re.compile(r"^\+[-=+|]+\+$")
_RE_TABLE_HEADER = re.compile(r"^Table\s+\d+", re.IGNORECASE)

# Requirement line: REQ-xxx, [Rxx], IRS-xxx, or "shall" patterns
_RE_REQ_ID    = re.compile(r"\b(REQ|IRS|SRS|SYS|SW|HW|DRS)-?\d+[\w.-]*\b", re.IGNORECASE)
_RE_REQ_TAG   = re.compile(r"\[R\d+[\w.-]*\]")
_RE_SHALL     = re.compile(r"\b(shall|must|is required to|are required to)\b", re.IGNORECASE)
_RE_REQ_LABEL = re.compile(r"^(REQUIREMENT|REQUIRE)[S]?\s*[:\-]", re.IGNORECASE)

# Procedure: "Step N" or sequential numbered list opening with action verb
_RE_STEP     = re.compile(r"^Step\s+\d+[\s:.]", re.IGNORECASE)
_RE_PROC_HDR = re.compile(r"^(PROCEDURE|PROCESS|WORKFLOW|METHOD)[S]?\s*[:\-]", re.IGNORECASE)

# Algorithm
_RE_ALGO_HDR = re.compile(
    r"^(Algorithm|Pseudocode|Pseudo-code|ALGORITHM)\s*(\d+)?[:\-]?", re.IGNORECASE
)

# Equation / math
_RE_EQUATION = re.compile(r"(\$\$.+?\$\$|\\\[.+?\\\]|Equation\s+\d+)", re.IGNORECASE | re.DOTALL)
_RE_MATH_LINE = re.compile(r"^[\w\s]*[=≈≤≥∑∫∂∞±×÷√][^\w\s]{0,3}[\w\s()[\]]+$")

# Figure
_RE_FIGURE = re.compile(r"^(Figure|Fig\.?|FIGURE)\s+\d+[\w.-]*", re.IGNORECASE)

# Code block (triple backtick or 4-space indent)
_RE_CODE_FENCE = re.compile(r"^```")
_RE_CODE_INDENT = re.compile(r"^(    |\t)\S")

# Page marker injected by doc_extractor or recognisable in source text
_RE_PAGE_MARKER = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)
_RE_PAGE_BREAK_MARKER = re.compile(r"^-{3,}\s*Page\s*(\d+)\s*-{3,}$", re.IGNORECASE)


def _detect_heading(line: str, prev_line: str, next_line: str) -> Tuple[bool, int, str]:
    """
    Try to identify *line* as a heading.

    Returns (is_heading, level, heading_text).
    Level 1 = chapter, 2 = section, 3 = subsection, 4+ = deeper.
    """
    stripped = line.strip()
    if not stripped:
        return False, 0, ""

    # Markdown headings
    for pattern, level in [(_RE_MD_H1, 1), (_RE_MD_H2, 2), (_RE_MD_H3, 3), (_RE_MD_H4, 4)]:
        m = pattern.match(stripped)
        if m:
            return True, level, m.group(1).strip()

    # Section/Chapter/Appendix keywords
    m = _RE_SECTION_KW.match(stripped)
    if m:
        title = (m.group(2) or "").strip() or m.group(1)
        return True, 1, f"{m.group(1)} {title}".strip()

    # Numbered section headings  e.g. "3.2.1 Hardware Interface"
    m = _RE_NUM_HEAD.match(stripped)
    if m:
        numbering = m.group(1)
        title = m.group(2).strip()
        # Infer level from depth of numbering (1=1, 1.2=2, 1.2.3=3 …)
        depth = numbering.count(".") + 1
        return True, min(depth, 4), title

    # ALL CAPS heading (min 3 words, max 12 words, not a requirement pattern)
    m = _RE_ALL_CAPS.match(stripped)
    if m and not _RE_SHALL.search(stripped) and not _RE_REQ_LABEL.match(stripped):
        words = stripped.split()
        if 3 <= len(words) <= 12:
            return True, 1, stripped.title()

    # RST-style underline (next line is all = or all -)
    if next_line and re.match(r"^[=\-]{3,}$", next_line.strip()):
        char = next_line.strip()[0]
        level = 1 if char == "=" else 2
        return True, level, stripped

    return False, 0, ""


class _DocumentAnalyzer:
    """
    Layer 1: Scans text line-by-line and emits a flat sequence of
    DocumentBlock objects tagged with semantic BlockType.
    """

    def analyze(self, text: str) -> List[DocumentBlock]:
        lines = text.splitlines()
        blocks: List[DocumentBlock] = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            stripped = line.strip()

            # ── Page marker ───────────────────────────────────────────────────
            pm = _RE_PAGE_MARKER.search(stripped) or _RE_PAGE_BREAK_MARKER.match(stripped)
            if pm:
                # Inject a sentinel that the splitter can read for page tracking
                page_num = int(pm.group(1))
                blocks.append(DocumentBlock(
                    block_type=BlockType.PARAGRAPH,
                    lines=[],
                    page_hint=page_num,
                ))
                i += 1
                continue

            # ── Code fence block  ```...``` ───────────────────────────────────
            if _RE_CODE_FENCE.match(stripped):
                code_lines = [line]
                i += 1
                while i < n:
                    code_lines.append(lines[i])
                    if _RE_CODE_FENCE.match(lines[i].strip()) and len(code_lines) > 1:
                        i += 1
                        break
                    i += 1
                blocks.append(DocumentBlock(BlockType.CODE, code_lines))
                continue

            # ── Table block ───────────────────────────────────────────────────
            if _RE_TABLE_ROW.match(stripped) or _RE_TABLE_GRID.match(stripped):
                tbl_lines = []
                while i < n and (
                    _RE_TABLE_ROW.match(lines[i].strip())
                    or _RE_TABLE_GRID.match(lines[i].strip())
                    or lines[i].strip() == ""  # allow a blank between header/separator
                ):
                    tbl_lines.append(lines[i])
                    i += 1
                # Trim trailing blank lines
                while tbl_lines and not tbl_lines[-1].strip():
                    tbl_lines.pop()
                if tbl_lines:
                    blocks.append(DocumentBlock(BlockType.TABLE, tbl_lines))
                continue

            # ── Table caption / label  "Table N …" ───────────────────────────
            if _RE_TABLE_HEADER.match(stripped):
                # Slurp the caption line plus any following table rows
                tbl_lines = [line]
                i += 1
                while i < n and (
                    _RE_TABLE_ROW.match(lines[i].strip())
                    or _RE_TABLE_GRID.match(lines[i].strip())
                    or (tbl_lines and lines[i].strip() == "")
                ):
                    tbl_lines.append(lines[i])
                    i += 1
                while tbl_lines and not tbl_lines[-1].strip():
                    tbl_lines.pop()
                blocks.append(DocumentBlock(BlockType.TABLE, tbl_lines))
                continue

            # ── Figure ────────────────────────────────────────────────────────
            if _RE_FIGURE.match(stripped):
                fig_lines = [line]
                i += 1
                # Slurp caption continuation (non-empty lines following figure label)
                while i < n and lines[i].strip() and not self._is_structure_boundary(lines[i].strip()):
                    fig_lines.append(lines[i])
                    i += 1
                blocks.append(DocumentBlock(BlockType.FIGURE, fig_lines))
                continue

            # ── Algorithm ─────────────────────────────────────────────────────
            if _RE_ALGO_HDR.match(stripped):
                algo_lines = [line]
                i += 1
                # Slurp until blank line or next heading
                while i < n and lines[i].strip():
                    if i + 1 < n:
                        _, lv, _ = _detect_heading(lines[i], lines[i - 1], lines[i + 1])
                    else:
                        _, lv, _ = _detect_heading(lines[i], lines[i - 1], "")
                    if lv > 0:
                        break
                    algo_lines.append(lines[i])
                    i += 1
                blocks.append(DocumentBlock(BlockType.ALGORITHM, algo_lines))
                continue

            # ── Requirement ───────────────────────────────────────────────────
            if (
                _RE_REQ_LABEL.match(stripped)
                or _RE_REQ_TAG.search(stripped)
                or (_RE_REQ_ID.search(stripped) and _RE_SHALL.search(stripped))
            ):
                req_lines = [line]
                i += 1
                # A requirement block may span several lines (rationale, context)
                # but stops at a blank line or next heading
                while i < n and lines[i].strip():
                    next_l = lines[i + 1].strip() if i + 1 < n else ""
                    is_h, lv, _ = _detect_heading(lines[i], lines[i - 1], next_l)
                    if is_h and lv <= 2:
                        break
                    req_lines.append(lines[i])
                    i += 1
                blocks.append(DocumentBlock(BlockType.REQUIREMENT, req_lines))
                continue

            # ── Procedure / Step ──────────────────────────────────────────────
            if _RE_PROC_HDR.match(stripped) or _RE_STEP.match(stripped):
                proc_lines = [line]
                i += 1
                # Slurp consecutive steps (Step N or numbered list)
                while i < n:
                    sl = lines[i].strip()
                    if not sl:
                        # Allow one blank line within a procedure block
                        if i + 1 < n and _RE_STEP.match(lines[i + 1].strip()):
                            proc_lines.append(lines[i])
                            i += 1
                            continue
                        break
                    next_l = lines[i + 1].strip() if i + 1 < n else ""
                    is_h, lv, _ = _detect_heading(lines[i], lines[i - 1], next_l)
                    if is_h and lv <= 2:
                        break
                    proc_lines.append(lines[i])
                    i += 1
                blocks.append(DocumentBlock(BlockType.PROCEDURE, proc_lines))
                continue

            # ── Heading ───────────────────────────────────────────────────────
            prev_l = lines[i - 1] if i > 0 else ""
            next_l = lines[i + 1] if i + 1 < n else ""
            is_h, level, htxt = _detect_heading(stripped, prev_l, next_l)
            if is_h:
                # RST underline: consume the underline line too
                if next_l and re.match(r"^[=\-]{3,}$", next_l.strip()):
                    blocks.append(DocumentBlock(BlockType.HEADING, [line, lines[i + 1]],
                                                heading_level=level, heading_text=htxt))
                    i += 2
                else:
                    blocks.append(DocumentBlock(BlockType.HEADING, [line],
                                                heading_level=level, heading_text=htxt))
                    i += 1
                continue

            # ── List item ─────────────────────────────────────────────────────
            if re.match(r"^(\s*[-•*]\s+|\s*\d+\.\s+)", line) and len(stripped) > 5:
                list_lines = [line]
                i += 1
                while i < n and lines[i].strip():
                    sl = lines[i].strip()
                    next_l2 = lines[i + 1].strip() if i + 1 < n else ""
                    is_h2, lv2, _ = _detect_heading(sl, lines[i - 1], next_l2)
                    if is_h2:
                        break
                    # Keep going if it looks like another list item or continuation
                    if re.match(r"^(\s*[-•*]\s+|\s*\d+\.\s+)", lines[i]) or lines[i].startswith("  "):
                        list_lines.append(lines[i])
                    else:
                        break
                    i += 1
                blocks.append(DocumentBlock(BlockType.LIST, list_lines))
                continue

            # ── Empty line — paragraph boundary ──────────────────────────────
            if not stripped:
                # Inject a paragraph separator sentinel
                blocks.append(DocumentBlock(BlockType.PARAGRAPH, [], page_hint=None))
                i += 1
                continue

            # ── Default: paragraph line ───────────────────────────────────────
            para_lines = [line]
            i += 1
            while i < n and lines[i].strip():
                sl = lines[i].strip()
                # Break on page marker — re-process it in the outer loop
                if _RE_PAGE_MARKER.search(sl) or _RE_PAGE_BREAK_MARKER.match(sl):
                    break
                # Stop if the next line looks like a structure boundary
                if self._is_structure_boundary(sl):
                    break
                next_l2 = lines[i + 1].strip() if i + 1 < n else ""
                is_h2, _, _ = _detect_heading(sl, lines[i - 1], next_l2)
                if is_h2:
                    break
                para_lines.append(lines[i])
                i += 1
            blocks.append(DocumentBlock(BlockType.PARAGRAPH, para_lines))

        return blocks

    @staticmethod
    def _is_structure_boundary(stripped: str) -> bool:
        """Return True if *stripped* line looks like a new structural element."""
        return bool(
            _RE_TABLE_ROW.match(stripped)
            or _RE_TABLE_GRID.match(stripped)
            or _RE_TABLE_HEADER.match(stripped)
            or _RE_FIGURE.match(stripped)
            or _RE_ALGO_HDR.match(stripped)
            or _RE_PROC_HDR.match(stripped)
            or _RE_STEP.match(stripped)
            or _RE_CODE_FENCE.match(stripped)
        )


# ══════════════════════════════════════════════════════════════════════════════
# §5  Layer 2 — Structural Splitter
# ══════════════════════════════════════════════════════════════════════════════

class _StructuralSplitter:
    """
    Layer 2: Groups DocumentBlock objects into StructureChunk objects,
    respecting atomic-block rules and heading hierarchy.
    """

    def __init__(self, min_chunk_words: int = 50, max_chunk_words: int = 800):
        self._min = min_chunk_words
        self._max = max_chunk_words

    def split(self, blocks: List[DocumentBlock]) -> List[StructureChunk]:
        chunks: List[StructureChunk] = []
        idx = 0

        # Heading hierarchy state
        chapter:    Optional[str] = None
        section:    Optional[str] = None
        subsection: Optional[str] = None
        heading:    Optional[str] = None
        current_level: int = 0
        current_page: Optional[int] = None

        # Accumulation buffer for paragraph/list content
        buf_lines:  List[str] = []
        buf_words:  int = 0
        buf_type:   BlockType = BlockType.PARAGRAPH

        def _flush(force: bool = False) -> None:
            nonlocal buf_lines, buf_words, buf_type, idx
            if not buf_lines:
                return
            text = "\n".join(buf_lines).strip()
            if not text:
                buf_lines, buf_words = [], 0
                return
            wc = len(text.split())
            if not force and wc < self._min:
                return  # keep accumulating
            # Semantic overflow: split large paragraph buffers at paragraph
            # boundaries (consecutive non-blank lines separated by blank lines)
            sub_paras = self._semantic_split(buf_lines, self._max)
            for sp in sub_paras:
                sp_text = "\n".join(sp).strip()
                if not sp_text:
                    continue
                chunks.append(StructureChunk(
                    text=sp_text,
                    block_type=buf_type,
                    chunk_index=idx,
                    heading_level=current_level,
                    chapter=chapter,
                    section=section,
                    subsection=subsection,
                    heading=heading,
                    page_number=current_page,
                    word_count=len(sp_text.split()),
                ))
                idx += 1
            buf_lines, buf_words, buf_type = [], 0, BlockType.PARAGRAPH

        for block in blocks:
            # ── Page sentinel ─────────────────────────────────────────────────
            if block.page_hint is not None and not block.lines:
                current_page = block.page_hint
                continue

            # ── Paragraph separator sentinel ──────────────────────────────────
            if not block.lines:
                if buf_words >= self._max:
                    _flush(force=True)
                elif buf_words >= self._min:
                    _flush(force=False)
                # else: keep accumulating through paragraph gap
                continue

            btype = block.block_type

            # ── Heading: flush buffer, update hierarchy ───────────────────────
            if btype == BlockType.HEADING:
                _flush(force=True)
                lvl = block.heading_level
                ht  = block.heading_text or block.text.strip()
                current_level = lvl
                if lvl == 1:
                    chapter, section, subsection, heading = ht, None, None, None
                elif lvl == 2:
                    section, subsection, heading = ht, None, None
                elif lvl == 3:
                    subsection, heading = ht, None
                else:
                    heading = ht
                # Don't emit a chunk for headings alone — they'll merge with body
                buf_lines = [block.text]
                buf_words = block.word_count
                buf_type  = BlockType.HEADING
                continue

            # ── Atomic blocks — emit immediately, never split ─────────────────
            if btype in _ATOMIC_BLOCK_TYPES:
                _flush(force=True)
                atomic_text = block.text
                if not atomic_text:
                    continue
                chunks.append(StructureChunk(
                    text=atomic_text,
                    block_type=btype,
                    chunk_index=idx,
                    heading_level=current_level,
                    chapter=chapter,
                    section=section,
                    subsection=subsection,
                    heading=heading,
                    page_number=current_page,
                    word_count=block.word_count,
                ))
                idx += 1
                continue

            # ── Paragraph / list: accumulate ──────────────────────────────────
            if buf_lines and buf_type == BlockType.HEADING:
                # A heading was the previous accumulation — reset type to PARAGRAPH
                buf_type = BlockType.PARAGRAPH

            buf_lines.extend(block.lines)
            buf_words += block.word_count

            if buf_words >= self._max:
                _flush(force=True)

        # Final flush
        _flush(force=True)
        return chunks

    @staticmethod
    def _semantic_split(lines: List[str], max_words: int) -> List[List[str]]:
        """
        Split a list of lines into sub-paragraphs at blank-line boundaries,
        keeping each sub-paragraph under max_words where possible.

        If a single paragraph group still exceeds max_words (no blank-line
        boundaries exist within it), fall back to sentence-level splitting
        so that very long single-paragraph blobs are also divided.
        """
        # Identify paragraph groups (split at blank lines)
        groups: List[List[str]] = []
        current: List[str] = []
        for ln in lines:
            if not ln.strip():
                if current:
                    groups.append(current)
                    current = []
            else:
                current.append(ln)
        if current:
            groups.append(current)

        result: List[List[str]] = []
        acc: List[str] = []
        acc_words = 0

        for grp in groups:
            grp_text = " ".join(grp)
            grp_words = len(grp_text.split())

            if grp_words > max_words:
                # Paragraph group is already oversized — sentence-level fallback
                if acc:
                    result.append(acc)
                    acc, acc_words = [], 0
                sentences = re.split(r'(?<=[.!?])\s+', grp_text)
                sent_acc: List[str] = []
                sent_words = 0
                for sent in sentences:
                    sw = len(sent.split())
                    if sent_words + sw > max_words and sent_acc:
                        result.append([" ".join(sent_acc)])
                        sent_acc, sent_words = [], 0
                    sent_acc.append(sent)
                    sent_words += sw
                if sent_acc:
                    result.append([" ".join(sent_acc)])
                continue

            if acc_words + grp_words > max_words and acc:
                result.append(acc)
                acc, acc_words = [], 0
            acc.extend(grp)
            acc.append("")  # blank separator
            acc_words += grp_words

        if acc:
            result.append(acc)

        # Hard word-count split as final fallback for any sub-list still > max_words
        # (handles text with no punctuation or blank lines at all)
        final: List[List[str]] = []
        for sub in (result if result else [lines]):
            sub_text = " ".join(sub)
            sub_words = sub_text.split()
            if len(sub_words) <= max_words:
                final.append(sub)
            else:
                # Split by word count
                for start in range(0, len(sub_words), max_words):
                    final.append([" ".join(sub_words[start:start + max_words])])
        return final if final else [lines]


# ══════════════════════════════════════════════════════════════════════════════
# §6  Layer 3 — Metadata Enricher
# ══════════════════════════════════════════════════════════════════════════════

class _TechnicalTerminologyExtractor:
    """
    Multi-signal technical terminology extractor for RAG retrieval & metadata indexing.
    Extracts:
      - Technology, model, framework, library, database, API, protocol, architecture,
        and standard names (e.g. PyTorch, ChromaDB, FastAPI, ONNX, CUDA, DO-178C).
      - Multi-word compound technical terms and noun phrases (e.g. diffusion model,
        image embeddings, vector database, fine-tuning pipeline, reciprocal rank fusion).
      - Acronym ↔ Expanded Terminology relationships (e.g. Retrieval-Augmented Generation (RAG)).
      - Standard/Requirement/Part identifiers (e.g. REQ-101, PN-99482-A, MIL-STD-810G).
      - Heading and frequency contextual signal weighting.
      - Comprehensive noise filtering for UI labels, generic verbs, boilerplate, and fillers.
      - Normalized variants generation for flexible matching (DO-178C → do-178c, do178c, do 178c).
    """

    _KNOWN_TECH_SEEDS = frozenset({
        "pytorch", "onnx", "onnxruntime", "fastapi", "uvicorn", "chromadb", "faiss",
        "python", "react", "typescript", "vite", "cuda", "gpu", "cpu", "vram", "ram",
        "whisper", "pyannote", "qwen", "bge", "bert", "llm", "rag", "ocr", "rapidocr",
        "spacy", "nltk", "docker", "kubernetes", "grpc", "rest", "http", "https", "json",
        "yaml", "sql", "sqlite", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "do-178c", "do-254", "mil-std-810g", "iso-9001", "as9100", "ieee-802.11", "ada",
        "nlp", "cv", "ai", "ml", "dl", "api", "sdk", "cli", "gui", "ui",
    })

    _TECH_NOUN_SUFFIXES = frozenset({
        "model", "models", "embedding", "embeddings", "pipeline", "pipelines",
        "database", "databases", "vector", "vectors", "retrieval", "index",
        "indexing", "architecture", "architectures", "algorithm", "algorithms",
        "framework", "frameworks", "library", "libraries", "transformer",
        "encoder", "decoder", "checkpoint", "checkpoints", "endpoint", "endpoints",
        "runtime", "runtimes", "tensor", "tensors", "neural", "network", "networks",
        "cache", "session", "cluster", "clusters", "workflow", "workflows",
        "token", "tokens", "tokenizer", "tokenizers", "schema", "schemas",
        "payload", "payloads", "repository", "repositories", "protocol", "protocols",
        "interface", "interfaces", "system", "systems", "service", "services",
        "router", "routers", "dataset", "datasets", "matrix", "matrices",
        "standard", "standards", "specification", "specifications", "requirement",
        "requirements", "module", "modules", "version", "parameter", "parameters",
    })

    _TECH_MODIFIERS = frozenset({
        "diffusion", "vector", "image", "text", "fine-tuning", "fine", "tuning",
        "semantic", "hybrid", "reciprocal", "rank", "fusion", "cosine", "similarity",
        "contextual", "dense", "sparse", "latent", "attention", "transformer",
        "neural", "deep", "machine", "learning", "artificial", "intelligence",
        "natural", "language", "speech", "audio", "video", "spatial", "graph",
        "tree", "binary", "linear", "logistic", "convolutional", "recurrent",
        "backpropagation", "gradient", "descent", "optimizer", "loss", "metric",
    })

    _NOISE_BLOCKLIST = frozenset({
        "showing preview", "first 1200 chars", "first 1,200 chars", "click below", "document details",
        "full text extraction", "on demand", "on-demand", "saved at", "created at",
        "updated at", "document inspection", "select option", "table of contents",
        "page number", "page count", "file size", "file path", "relative path",
        "user id", "recording id", "meeting id", "doc id", "total chunks",
        "chunk index", "extracted text", "context summary", "pipeline steps",
        "show preview", "show full text", "extract full text", "cancel", "submit",
        "is used to", "can be found", "according to", "in order to", "as shown in",
        "refer to", "for example", "such as", "more information", "this document",
        "following section", "overview", "introduction", "conclusion", "summary",
        "chapter", "section", "subsection", "appendix", "table", "figure",
    })

    _GENERIC_STANDALONE_WORDS = frozenset({
        "data", "text", "file", "page", "code", "test", "type", "user", "time",
        "name", "list", "item", "value", "view", "info", "row", "col", "date",
        "note", "notes", "path", "key", "keys", "mode", "modes", "rule", "rules",
    })

    _RE_ACRONYM_PAIR = re.compile(
        r"\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+){1,4})\s*\(([A-Z]{2,8})\)"
    )

    _RE_IDENTIFIER = re.compile(
        r"\b(?:"
        r"(?:DOC|TICKET|JIRA|BUG|ISSUE|INCIDENT|TASK|PR|RFC|SPEC|VER|MODEL|PN|SN|CAT|REF|ID|REQ|SRS|ICD)-?\d+[\w.-]*"
        r"|[A-Z]{2,6}-\d{2,}[\w.-]*"
        r"|[A-Z][0-9]{3,8}[A-Z0-9\-]*"
        r"|MIL-STD-\d+[\w.-]*"
        r"|DO-\d+[\w.-]*"
        r"|ISO\s?\d{4,}[\w.-]*"
        r"|IEEE\s?\d{3,}[\w.-]*"
        r"|v\d+\.\d+[\w.-]*"
        r"|Rev\s?[A-Z0-9]+"
        r")\b",
        re.IGNORECASE,
    )

    _RE_WORD_TOKENS = re.compile(r"\b[a-zA-Z0-9_\-.]+\b")

    def extract(self, text: str, heading_context: str = "") -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "technical_terms": [],
                "acronym_mappings": {},
                "identifiers": [],
            }

        text_clean = text.strip()

        tech_terms_set: set[str] = set()
        acronym_mappings: Dict[str, str] = {}
        identifiers_set: set[str] = set()

        # ── 1. Acronym ↔ Full Term Pairs ──────────────────────────────────────
        for m in self._RE_ACRONYM_PAIR.finditer(text_clean):
            full_form = m.group(1).strip()
            acronym = m.group(2).strip()
            if len(acronym) >= 2 and acronym.lower() not in _STOPWORDS:
                acronym_mappings[acronym] = full_form
                acronym_mappings[full_form] = acronym
                tech_terms_set.add(full_form)
                tech_terms_set.add(acronym)

        # ── 2. Standard & Requirement Identifiers ──────────────────────────────
        for m in self._RE_IDENTIFIER.finditer(text_clean):
            ident = m.group(0).strip()
            if len(ident) >= 2:
                identifiers_set.add(ident)
                tech_terms_set.add(ident)

        # ── 3. Known Technology & Standard Names ───────────────────────────────
        tokens = self._RE_WORD_TOKENS.findall(text_clean)
        for tok in tokens:
            tok_lower = tok.lower()
            if tok_lower in self._KNOWN_TECH_SEEDS:
                tech_terms_set.add(tok)

        # ── 4. Compound Technical Noun Phrases (2 to 4 words) ──────────────────
        words = text_clean.split()
        for window_size in (2, 3, 4):
            for i in range(len(words) - window_size + 1):
                phrase_words = words[i : i + window_size]
                phrase_str = " ".join(phrase_words).strip(".,;:()[]{}'\"")
                phrase_lower = phrase_str.lower()

                if any(noise in phrase_lower for noise in self._NOISE_BLOCKLIST):
                    continue

                p_tokens = phrase_lower.split()
                if len(p_tokens) < 2:
                    continue

                has_tech_suffix = p_tokens[-1] in self._TECH_NOUN_SUFFIXES
                has_tech_modifier = any(tok in self._TECH_MODIFIERS for tok in p_tokens[:-1])
                has_known_seed = any(tok in self._KNOWN_TECH_SEEDS for tok in p_tokens)

                if has_tech_suffix or has_tech_modifier or has_known_seed:
                    if p_tokens[0] not in _STOPWORDS and p_tokens[-1] not in _STOPWORDS:
                        tech_terms_set.add(phrase_str)

        # ── 5. Heading Context Boost ───────────────────────────────────────────
        if heading_context:
            h_words = heading_context.split()
            for w_size in (1, 2, 3, 4):
                for i in range(len(h_words) - w_size + 1):
                    h_phrase = " ".join(h_words[i : i + w_size]).strip(".,;:()[]{}'\"")
                    hp_lower = h_phrase.lower()
                    if hp_lower not in self._NOISE_BLOCKLIST and len(hp_lower) >= 3:
                        hp_toks = hp_lower.split()
                        if any(tok in self._TECH_NOUN_SUFFIXES or tok in self._TECH_MODIFIERS or tok in self._KNOWN_TECH_SEEDS for tok in hp_toks):
                            if hp_toks[0] not in _STOPWORDS and hp_toks[-1] not in _STOPWORDS:
                                tech_terms_set.add(h_phrase)

        # ── 6. Filter Out Noise & Standalone Generic Words ────────────────────
        filtered_terms: List[str] = []
        for term in tech_terms_set:
            t_clean = term.strip(".,;:()[]{}'\"")
            t_lower = t_clean.lower()
            if not t_clean or len(t_clean) < 2:
                continue
            if t_lower in _STOPWORDS or t_lower in self._NOISE_BLOCKLIST or t_lower in self._GENERIC_STANDALONE_WORDS:
                continue
            if any(noise == t_lower for noise in self._NOISE_BLOCKLIST):
                continue
            filtered_terms.append(t_clean)

        # ── 7. Generate Normalized Variants ───────────────────────────────────
        all_terms_with_variants: set[str] = set(filtered_terms)
        for term in list(filtered_terms):
            variants = self.generate_normalized_variants(term)
            for v in variants:
                all_terms_with_variants.add(v)

        return {
            "technical_terms": sorted(all_terms_with_variants),
            "acronym_mappings": acronym_mappings,
            "identifiers": sorted(identifiers_set),
        }

    def generate_normalized_variants(self, term: str) -> List[str]:
        if not term:
            return []

        variants: set[str] = set()
        t_lower = term.lower().strip()
        variants.add(t_lower)

        unhyphenated = re.sub(r'[\-_.]+', ' ', t_lower).strip()
        if unhyphenated != t_lower:
            variants.add(unhyphenated)

        collapsed = re.sub(r'[\-_.\s]+', '', t_lower).strip()
        if len(collapsed) >= 3:
            variants.add(collapsed)

        return sorted(variants)


# Heuristic patterns for entity extraction
_RE_ACRONYM        = re.compile(r"\b([A-Z]{2,8})\b")
_RE_TECH_ENTITY    = re.compile(
    r"\b("
    r"REQ-?\d+[\w.-]*"          # requirement IDs
    r"|IRS-?\d+[\w.-]*"
    r"|SRS-?\d+[\w.-]*"
    r"|ICD-?\d+[\w.-]*"
    r"|MIL-STD-\d+[\w.-]*"      # MIL standards
    r"|DO-\d+[\w.-]*"           # DO-178, DO-254
    r"|AS\d{4,}[\w.-]*"         # AS9100, AS9115
    r"|ISO\s?\d{4,}[\w.-]*"     # ISO standards
    r"|IEEE\s?\d{3,}[\w.-]*"    # IEEE standards
    r"|v\d+\.\d+[\w.-]*"        # version strings
    r"|Rev\s?[A-Z0-9]+"         # Rev A, Rev 2
    r"|[A-Z]{2,6}-\d{3,}[\w-]*" # part numbers like SW-001, HW-123
    r")",
    re.IGNORECASE,
)
_RE_KEYWORD_PHRASE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b"  # Title Case multi-word phrases
)
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "into", "when",
    "where", "which", "while", "shall", "should", "will", "would", "could",
    "have", "has", "been", "are", "were", "was", "not", "but", "also",
    "each", "only", "any", "all", "its", "their", "they", "then", "than",
    "can", "may", "must", "both", "such", "some", "more", "other", "these",
    "those", "after", "before", "between", "during", "within", "without",
    "being", "having", "using", "based", "used", "made", "set", "given",
    "provided", "defined", "required", "specified", "refer", "see",
})

_EXTENSION_TO_DOCTYPE: Dict[str, str] = {
    ".pdf":  "pdf",
    ".docx": "word_document",
    ".doc":  "word_document",
    ".pptx": "presentation",
    ".ppt":  "presentation",
    ".txt":  "text",
    ".md":   "markdown",
    ".xlsx": "spreadsheet",
    ".xls":  "spreadsheet",
    ".csv":  "spreadsheet",
    ".png":  "image",
    ".jpg":  "image",
    ".jpeg": "image",
    ".webp": "image",
}


class _MetadataEnricher:
    """
    Layer 3: Adds semantic metadata fields to each StructureChunk.
    All extraction is heuristic/regex — no LLM calls.
    """

    # ── Extra regex patterns for enhanced metadata ────────────────────────────
    _RE_ACRONYM_MAPPING = re.compile(
        r"\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+){1,4})\s*\(([A-Z]{2,8})\)"
    )

    _RE_GENERIC_IDENTIFIERS = re.compile(
        r"\b(?:"
        r"(?:DOC|TICKET|JIRA|BUG|ISSUE|INCIDENT|TASK|PR|RFC|SPEC|VER|MODEL|PN|SN|CAT|REF|ID|REQ|SRS|ICD)-?\d+[\w.-]*"
        r"|[A-Z]{2,6}-\d{2,}[\w.-]*"
        r"|[A-Z][0-9]{3,8}[A-Z0-9\-]*"
        r"|MIL-STD-\d+[\w.-]*"
        r"|DO-\d+[\w.-]*"
        r"|ISO\s?\d{4,}[\w.-]*"
        r"|IEEE\s?\d{3,}[\w.-]*"
        r"|v\d+\.\d+[\w.-]*"
        r"|Rev\s?[A-Z0-9]+"
        r")\b",
        re.IGNORECASE,
    )

    _RE_NOUN_PHRASE = re.compile(
        r"\b([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+){1,3})\b"
    )

    _RE_CALENDAR_DATE = re.compile(
        r"\b(?:"
        r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"  # 01/02/2024, 1-2-24
        r"|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}"    # 2024-01-02
        r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2}(?:,?\s+\d{4})?"              # January 15, 2024
        r"|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"(?:\s+\d{4})?"                            # 15 January 2024
        r"|Q[1-4]\s*\d{4}"                          # Q1 2024
        r")\b",
        re.IGNORECASE,
    )
    _RE_PROJECT_NAME = re.compile(
        r"\b("
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\s+(?:Project|Program|Initiative|System|Platform|Module|Engine)"
        r"|(?:Project|Program)\s+[A-Z][a-zA-Z0-9_\-]+"
        r")\b"
    )
    _RE_ORG_ENTITY = re.compile(
        r"\b("
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+(?:Inc|LLC|Corp(?:oration)?|Ltd|GmbH|AG|SA|PLC|Co\.?|Group|Team|Department|Division|Committee|Board)"
        r"|(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
        r")\b"
    )
    _RE_IMPORTANT_NUMBER = re.compile(
        r"\b("
        r"\d+(?:\.\d+)?%"                           # percentages
        r"|\$[\d,]+(?:\.\d{1,2})?"                   # currency
        r"|[€£¥]\s*[\d,]+(?:\.\d{1,2})?"            # other currency
        r"|\d+(?:\.\d+)?\s*(?:MHz|GHz|kHz|Hz|MB|GB|TB|KB|ms|μs|ns|sec|seconds?|minutes?|hours?|days?|weeks?|months?|years?)"  # units
        r"|\d+(?:\.\d+)?\s*(?:mm|cm|m|km|in|ft|lbs?|kg|mg|°[CF])"  # measurements
        r"|v\d+(?:\.\d+)+"                            # version numbers
        r"|#\d{3,}"                                   # issue/ticket numbers
        r")\b",
        re.IGNORECASE,
    )

    def __init__(self, filename: str = "", doc_scope: str = "global_context",
                 meeting_id: Optional[str] = None, doc_name: Optional[str] = None,
                 doc_type: Optional[str] = None):
        ext = os.path.splitext(filename.lower())[1] if filename else ""
        self._doc_type  = doc_type or _EXTENSION_TO_DOCTYPE.get(ext, "unknown")
        self._doc_name  = doc_name or filename
        self._scope     = doc_scope
        self._meeting_id = meeting_id
        self._tech_extractor = _TechnicalTerminologyExtractor()

    def enrich(self, chunk: StructureChunk) -> Dict[str, Any]:
        text = chunk.text
        d = chunk.to_dict()

        # ── Document-level metadata ───────────────────────────────────────────
        d["document_name"] = self._doc_name
        d["document_type"] = self._doc_type
        d["scope"]         = self._scope
        d["meeting_id"]    = self._meeting_id
        d["context_type"]  = "global" if self._scope == "global_context" else "meeting"

        # Heading context
        heading_ctx = f"{d.get('heading') or ''} {d.get('section') or ''} {d.get('chapter') or ''}".strip()

        # ── 1. Multi-signal Technical Terminology Extraction ───────────────────
        tech_extracted = self._tech_extractor.extract(text, heading_context=heading_ctx)
        d["technical_terms"] = tech_extracted["technical_terms"]
        d["acronym_mappings"] = tech_extracted["acronym_mappings"]
        d["identifiers"] = tech_extracted["identifiers"]

        # ── 2. Acronyms & Expanded Terms (Uncapped) ───────────────────────────
        acronyms_raw = sorted(set(_RE_ACRONYM.findall(text)))
        acronyms = [a for a in acronyms_raw if len(a) >= 2 and a not in _STOPWORDS]
        expanded_acronyms = list(acronyms)
        for a in acronyms:
            if a in d["acronym_mappings"]:
                expanded_acronyms.append(d["acronym_mappings"][a])

        d["acronyms"] = sorted(set(expanded_acronyms))

        # ── 3. Technical Entities ─────────────────────────────────────────────
        tech_entities = sorted(set(m.group(0) for m in _RE_TECH_ENTITY.finditer(text)) | set(d["identifiers"]) | set(d["technical_terms"]))
        d["technical_entities"] = tech_entities

        # ── 4. Keywords (Title Case + Noun Phrases, Uncapped) ─────────────────
        kw_title = _RE_KEYWORD_PHRASE.findall(text)
        kw_noun = [p for p in self._RE_NOUN_PHRASE.findall(text) if len(p.split()) >= 2]

        seen_kw: set[str] = set()
        keywords: List[str] = []
        for kw in kw_title + kw_noun:
            kw_clean = kw.strip()
            kw_lower = kw_clean.lower()
            if (
                kw_lower not in _STOPWORDS
                and kw_lower not in seen_kw
                and len(kw_clean) > 3
                and not any(w in _STOPWORDS for w in kw_lower.split())
            ):
                seen_kw.add(kw_lower)
                keywords.append(kw_clean)

        d["keywords"] = keywords

        # ── 5. Dates, Projects, Entities, Numbers (Uncapped) ───────────────────
        dates = sorted(set(m.group(0).strip() for m in self._RE_CALENDAR_DATE.finditer(text)))
        d["dates"] = dates

        project_names = sorted(set(m.group(0).strip() for m in self._RE_PROJECT_NAME.finditer(text)))
        d["project_names"] = project_names

        org_entities = sorted(set(m.group(0).strip() for m in self._RE_ORG_ENTITY.finditer(text)))
        d["entities"] = org_entities

        numbers = sorted(set(m.group(0).strip() for m in self._RE_IMPORTANT_NUMBER.finditer(text)))
        d["numbers"] = numbers

        # ── 6. Consolidated Search Terms ──────────────────────────────────────
        search_terms_set: set[str] = set()

        def _add_term(t: str):
            if not t:
                return
            cleaned = str(t).strip()
            if cleaned and len(cleaned) >= 2:
                search_terms_set.add(cleaned.lower())
                # Add un-hyphenated / normalized version
                unhyphenated = re.sub(r'[\-_.]+', ' ', cleaned.lower()).strip()
                if unhyphenated != cleaned.lower():
                    search_terms_set.add(unhyphenated)
                # Add collapsed version (no spaces/hyphens)
                collapsed = re.sub(r'[\-_. ]+', '', cleaned.lower()).strip()
                if len(collapsed) >= 3:
                    search_terms_set.add(collapsed)

        for k in d["keywords"]: _add_term(k)
        for t in d["technical_entities"]: _add_term(t)
        for a in d["acronyms"]: _add_term(a)
        for p in d["project_names"]: _add_term(p)
        for e in d["entities"]: _add_term(e)
        for num in d["numbers"]: _add_term(num)
        for ident in d["identifiers"]: _add_term(ident)
        for k_mapping, v_mapping in d["acronym_mappings"].items():
            _add_term(k_mapping)
            _add_term(v_mapping)

        # Headings hierarchy
        for h_field in ("chapter", "section", "subsection", "heading"):
            h_val = d.get(h_field)
            if h_val:
                _add_term(str(h_val))

        d["search_terms"] = sorted(search_terms_set)
        d["important_terms"] = d["search_terms"]  # Alias for backward compatibility

        # ── 7. Contextual Chunk Embedding String ──────────────────────────────
        hierarchy_parts = []
        if self._doc_name:
            hierarchy_parts.append(f"Document: {self._doc_name}")
        for h_f in ("chapter", "section", "subsection", "heading"):
            val = d.get(h_f)
            if val:
                hierarchy_parts.append(str(val))

        hierarchy_str = " > ".join(hierarchy_parts)
        if hierarchy_str:
            d["embedding_context"] = f"[{hierarchy_str}]\n{text}"
        else:
            d["embedding_context"] = text

        return d


# ══════════════════════════════════════════════════════════════════════════════
# §7  Public API — chunk_document()
# ══════════════════════════════════════════════════════════════════════════════

def chunk_document(
    text: str,
    filename: str = "",
    *,
    doc_scope: str = "global_context",
    meeting_id: Optional[str] = None,
    doc_name: Optional[str] = None,
    doc_type: Optional[str] = None,
    min_chunk_words: int = 50,
    max_chunk_words: int = 800,
) -> List[Dict[str, Any]]:
    """
    Structure-aware document chunking for technical/ADA documents.

    Preserves logical document structure (headings, tables, requirements,
    procedures, algorithms, equations, figures) and returns chunks enriched
    with rich metadata.  Atomic blocks (tables, requirements, procedures,
    algorithms, equations, code) are **never** split across chunk boundaries.

    Parameters
    ----------
    text            : Extracted document text (any length).
    filename        : Original filename — used to infer document type.
    doc_scope       : ``"global_context"`` or ``"meeting_context"``.
    meeting_id      : Recording/meeting ID (required when scope is meeting_context).
    doc_name        : Human-readable document name (defaults to filename).
    doc_type        : Override auto-detected document type string.
    min_chunk_words : Minimum words before a paragraph chunk is emitted
                      (heading stubs are merged into next block).
    max_chunk_words : Soft maximum words per chunk (atomic blocks may exceed
                      this limit — they are kept whole regardless).

    Returns
    -------
    List of dicts, each containing:
      text, chunk_index, block_type, heading_level,
      chapter, section, subsection, heading,
      document_name, document_type, scope, meeting_id, context_type,
      page_number, keywords, technical_entities, acronyms,
      dates, project_names, entities, numbers, important_terms,
      search_terms, identifiers, acronym_mappings, embedding_context,
      previous_chunk_index, next_chunk_index,
      chunk_size_words, total_chunks
    """
    if not text or not text.strip():
        return []

    analyzer = _DocumentAnalyzer()
    splitter = _StructuralSplitter(
        min_chunk_words=min_chunk_words,
        max_chunk_words=max_chunk_words,
    )
    enricher = _MetadataEnricher(
        filename=filename,
        doc_scope=doc_scope,
        meeting_id=meeting_id,
        doc_name=doc_name or filename,
        doc_type=doc_type,
    )

    blocks = analyzer.analyze(text)
    chunks = splitter.split(blocks)

    result: List[Dict[str, Any]] = []
    for chunk in chunks:
        enriched = enricher.enrich(chunk)
        result.append(enriched)

    # Post-enrichment: set total_chunks and neighbor indices on every chunk
    total = len(result)
    for idx, d in enumerate(result):
        d["total_chunks"] = total
        d["previous_chunk_index"] = result[idx - 1]["chunk_index"] if idx > 0 else None
        d["next_chunk_index"] = result[idx + 1]["chunk_index"] if idx < total - 1 else None
        d["previous_chunk_id"] = f"{d.get('document_name', filename)}_chunk_{idx-1}" if idx > 0 else None
        d["next_chunk_id"] = f"{d.get('document_name', filename)}_chunk_{idx+1}" if idx < total - 1 else None

    return result


# ══════════════════════════════════════════════════════════════════════════════
# §8  Legacy API — chunk_text()  (unchanged, kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple punctuation rules.
    Returns a list of non-empty stripped sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


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


# ══════════════════════════════════════════════════════════════════════════════
# §9  Transcript chunker  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# §10  Page-aware chunker  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

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
