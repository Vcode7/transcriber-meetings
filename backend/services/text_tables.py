"""
text_tables.py — Tables inside extracted document text.

One Markdown format is shared by every extractor (doc_extractor for PDF / DOCX / PPTX tables,
ocr_layout for tables rebuilt from OCR'd images and scans):

    [Table: 3 rows x 2 cols]
    | Time | Topic |
    | --- | --- |
    | 10:00 | Opening remarks |

find_table_blocks() returns the tables and table-like blocks of extracted text, so a caller
can work on the tabular part of a document alone ("Extract Agenda from Tables").

Standard library only: doc_extractor must keep working when the OCR engine cannot load.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

TABLE_TITLE = re.compile(r"^\[(?:Nested )?Table\b[^\]]*\]$")
SEPARATOR_ROW = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+$")
_MARKDOWN_ROW = re.compile(r"^\|.*\|$")
_SHEET_HEADERS = re.compile(r"^Headers:\s*\S")
_SHEET_ROW = re.compile(r"^\[Row \d+\]")
_IMAGE_OCR_PREFIX = re.compile(r"^\[Embedded Image OCR:\s*")


def format_markdown_table(matrix: Sequence[Sequence[object]], title: str = "Table") -> str:
    r"""
    Format a 2D matrix of cell strings into a compact Markdown table:
      - "[{title}: R rows x C cols]" first, so the table's extent is explicit.
      - All rows padded to the same number of columns; completely empty rows omitted.
      - Multi-line cell text becomes '<br>' to keep one line per row; '|' is escaped as '\|'.
      - The first row is the header, followed by the '| --- | --- |' separator row.
    """
    if not matrix:
        return ""

    cleaned_rows: List[List[str]] = []
    for row in matrix:
        cleaned_row = []
        for cell in row:
            c_str = str(cell).strip() if cell is not None else ""
            c_str = c_str.replace("|", "\\|")
            c_str = re.sub(r'\r?\n', '<br>', c_str)
            cleaned_row.append(c_str)
        if any(c for c in cleaned_row):
            cleaned_rows.append(cleaned_row)

    if not cleaned_rows:
        return ""

    num_cols = max(len(r) for r in cleaned_rows)
    if num_cols == 0:
        return ""
    for row in cleaned_rows:
        while len(row) < num_cols:
            row.append("")

    lines = [
        f"[{title}: {len(cleaned_rows)} rows x {num_cols} cols]",
        "| " + " | ".join(cleaned_rows[0]) + " |",
        "| " + " | ".join(["---"] * num_cols) + " |",
    ]
    for row in cleaned_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def strip_layout_markup(text: str) -> str:
    """The words of structured text without table titles, separator rows or cell pipes."""
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if TABLE_TITLE.match(s) or SEPARATOR_ROW.match(s):
            continue
        s = re.sub(r"(?<!\\)\|", " ", s).replace("\\|", "|")
        s = " ".join(s.split())
        if s:
            out.append(s)
    return "\n".join(out)


# ── Finding tables in extracted text ─────────────────────────────────────────

def _normalise(line: str) -> str:
    """A line without the embedded-image wrapper doc_extractor puts around OCR text."""
    s = _IMAGE_OCR_PREFIX.sub("", line.strip())
    return s[:-1] if s.endswith("|]") else s


def _delimited_cells(line: str) -> Optional[tuple]:
    """(delimiter kind, cell count) of a table-like line: cells split by '|', tabs or 3+ spaces
    (two spaces often just follow a full stop)."""
    s = line.strip()
    if not s or s.startswith("["):
        return None
    for kind, pattern in (("pipe", r"\s*\|\s*"), ("tab", r"\s*\t\s*"), ("spaces", r" {3,}")):
        if kind == "pipe" and "|" not in s or kind == "tab" and "\t" not in s:
            continue
        cells = [c for c in re.split(pattern, s.strip("|").strip()) if c.strip()]
        if len(cells) >= 2:
            return kind, len(cells)
    return None


def find_table_blocks(text: str) -> List[str]:
    """
    Tables and table-like blocks of extracted text, in document order:
      - Markdown tables (with their "[Table …]" title line) with at least two data rows or a
        header and one data row
      - spreadsheet blocks ("Headers: a | b" + "[Row N] …" lines)
      - 2+ consecutive lines split into the same number (±1) of 2+ cells by the same
        delimiter: '|', tabs (OCR field gaps, typed agendas) or runs of 3+ spaces
    """
    lines = [_normalise(ln) for ln in (text or "").splitlines()]
    blocks: List[str] = []
    i = 0
    while i < len(lines):
        s = lines[i]
        titled = bool(TABLE_TITLE.match(s))
        if titled or _MARKDOWN_ROW.match(s):
            j = i + 1 if titled else i
            rows = []
            while j < len(lines) and _MARKDOWN_ROW.match(lines[j]):
                rows.append(lines[j])
                j += 1
            if sum(1 for r in rows if not SEPARATOR_ROW.match(r)) >= 2:
                blocks.append("\n".join(([s] if titled else []) + rows))
            i = max(j, i + 1)  # Markdown rows are never re-read as delimited lines
            continue

        if _SHEET_HEADERS.match(s):
            j = i + 1
            while j < len(lines) and _SHEET_ROW.match(lines[j]):
                j += 1
            if j - i >= 2:
                blocks.append("\n".join(lines[i:j]))
                i = j
                continue

        first = _delimited_cells(s)
        if first:
            j, low, high = i + 1, first[1], first[1]
            while j < len(lines):
                nxt = _delimited_cells(lines[j])
                if not nxt or nxt[0] != first[0] or max(high, nxt[1]) - min(low, nxt[1]) > 1:
                    break
                low, high = min(low, nxt[1]), max(high, nxt[1])
                j += 1
            if j - i >= 2:
                blocks.append("\n".join(lines[i:j]))
                i = j
                continue
        i += 1
    return blocks
