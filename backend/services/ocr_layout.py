"""
ocr_layout.py — Structured text from the positions of recognised OCR regions.

Plain OCR output joins regions line by line, which shreds tables: some cells of a row are
fused with a space, others land on lines of their own, and short cells ("1", "2.") are
dropped as noise. This module rebuilds the layout from the region boxes:

  1. rows      regions sharing a visual line (vertical overlap >= 50% of the smaller height)
  2. segments  within a row, regions further apart than REGION_GAP line heights (cells, fields)
  3. tables    runs of rows whose segments share whitespace columns, i.e. vertical gaps that
               (almost) no row of the run writes into; wrapped cell lines join their row
  4. order     tables and text lines row by row (XY-cut columns when column detection is on)
  5. render    tables as Markdown ("[Table: R rows x C cols]" + one "| ... |" line per row,
               the format of services/text_tables.py), wide gaps inside a text line as a tab,
               paragraph breaks as a blank line

These are the only markers added, so the text stays compact as LLM input.
ocr_engine imports this module lazily, because this module uses ocr_engine's helpers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from services.ocr_engine import (
        SEPARATOR_RUN, OcrLine, _box_extent, _clean_ocr_text, squash_spaces, xy_cut_order,
    )
    from services.ocr_settings import OcrSettings
    from services.text_tables import format_markdown_table
except ImportError:
    from backend.services.ocr_engine import (  # type: ignore[no-redef]
        SEPARATOR_RUN, OcrLine, _box_extent, _clean_ocr_text, squash_spaces, xy_cut_order,
    )
    from backend.services.ocr_settings import OcrSettings  # type: ignore[no-redef]
    from backend.services.text_tables import format_markdown_table  # type: ignore[no-redef]

REGION_GAP = 0.5         # × line height: separate regions further apart are separate segments
                         # (the detector already joins normal word spaces into one region)
TAB_GAP = 1.0            # × line height: a wider gap inside a text line is written as a tab
                         # (widened words of justified text measured <= 0.62, form fields >= 1.5)
CONTINUATION_GAP = 0.6   # × line height: a wrapped cell line sits at most this far below its row
TABLE_ROW_GAP = 3.0      # × line height: rows further apart do not belong to one table
PARAGRAPH_GAP = 1.0      # × line height: a larger vertical gap starts a new paragraph
LABEL_CHARS = 12         # median cell length of a label column (numbers, times, short names)
PROSE_WORDS = 3          # median words per cell of running text (with lowercase continuations)
EDGE_GAP_FACTOR = 1.8    # a first/last table row this much farther away than the rest is not part of it


# ── Layout model ──────────────────────────────────────────────────────────────

@dataclass
class _Segment:
    items: List[OcrLine]
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def text(self) -> str:
        return " ".join(t for t in (it.text.strip() for it in self.items) if t)


@dataclass
class _Row:
    segments: List[_Segment]
    y0: float
    y1: float
    height: float


@dataclass
class _Block:
    kind: str                                   # "text" | "table"
    row: int                                    # index of its (first) row, for row-major order
    x0: float
    y0: float
    x1: float
    y1: float
    lines: List[OcrLine]                        # the text line, or the table cells row by row
    cells: Optional[List[List[str]]] = None     # table only
    last_row: int = -1                          # table only: index of its last row


@dataclass
class OcrLayout:
    text: str                                   # structured text (tables, tabs, paragraph breaks)
    plain_text: str                             # the same words without markup (counts)
    lines: List[OcrLine] = field(default_factory=list)  # text lines and cells in reading order
    table_count: int = 0


def _rect(x0: float, y0: float, x1: float, y1: float) -> List[List[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _bbox(parts: Sequence) -> Tuple[float, float, float, float]:
    return (min(p.x0 for p in parts), min(p.y0 for p in parts),
            max(p.x1 for p in parts), max(p.y1 for p in parts))


def _joined_line(segments: Sequence[_Segment], text: str) -> OcrLine:
    items = [it for s in segments for it in s.items]
    weights = [max(1, len(it.text)) for it in items]
    score = float(sum(it.score * w for it, w in zip(items, weights)) / sum(weights))
    return OcrLine(box=_rect(*_bbox(segments)), text=text, score=score)


# ── 1–2. Rows and segments ────────────────────────────────────────────────────

def _group_rows(ext: Sequence[Tuple[float, float, float, float]]) -> List[List[int]]:
    """Regions sharing a visual line (same rule as ocr_engine.merge_lines), top to bottom."""
    order = sorted(range(len(ext)), key=lambda i: (ext[i][1] + ext[i][3]) / 2.0)
    rows: List[List[int]] = []
    top = bot = 0.0
    for i in order:
        _, y0, _, y1 = ext[i]
        h = max(1.0, y1 - y0)
        if rows and min(y1, bot) - max(y0, top) >= 0.5 * min(h, max(1.0, bot - top)):
            rows[-1].append(i)
            top, bot = min(top, y0), max(bot, y1)
            continue
        rows.append([i])
        top, bot = y0, y1
    return rows


def _make_row(idx: List[int], items: Sequence[OcrLine], ext, gap_limit: float) -> _Row:
    idx = sorted(idx, key=lambda i: ext[i][0])
    height = float(np.median([max(1.0, ext[i][3] - ext[i][1]) for i in idx]))
    groups: List[List[int]] = [[idx[0]]]
    reach = ext[idx[0]][2]
    for i in idx[1:]:
        if ext[i][0] - reach > gap_limit * height:
            groups.append([i])
            reach = ext[i][2]
        else:
            groups[-1].append(i)
            reach = max(reach, ext[i][2])
    segments = [
        _Segment(items=[items[i] for i in g],
                 x0=min(ext[i][0] for i in g), y0=min(ext[i][1] for i in g),
                 x1=max(ext[i][2] for i in g), y1=max(ext[i][3] for i in g))
        for g in groups
    ]
    return _Row(segments=segments, y0=min(s.y0 for s in segments), y1=max(s.y1 for s in segments),
                height=height)


# ── 3. Tables ─────────────────────────────────────────────────────────────────

def _columns(segments: Sequence[_Segment]) -> List[Tuple[float, float]]:
    """Merged x-ranges of the segments; the gaps between them are whitespace columns."""
    spans = sorted((s.x0, s.x1) for s in segments)
    cols = [list(spans[0])]
    for x0, x1 in spans[1:]:
        if x0 <= cols[-1][1]:
            cols[-1][1] = max(cols[-1][1], x1)
        else:
            cols.append([x0, x1])
    return [(a, b) for a, b in cols]


def _column_of(seg: _Segment, cols: Sequence[Tuple[float, float]]) -> int:
    cx = (seg.x0 + seg.x1) / 2.0
    return min(range(len(cols)), key=lambda k: 0.0 if cols[k][0] <= cx <= cols[k][1]
               else min(abs(cols[k][0] - cx), abs(cols[k][1] - cx)))


def _single_column(seg: _Segment, cols: Sequence[Tuple[float, float]]) -> Optional[int]:
    """The one column a segment overlaps, or None (none, or it spans several)."""
    hits = [k for k, (a, b) in enumerate(cols) if seg.x0 < b and seg.x1 > a]
    return hits[0] if len(hits) == 1 else None


def _voted_columns(rows: Sequence[_Row]) -> List[Tuple[float, float]]:
    """
    Columns of a grown table: x-ranges that more than a quarter of its rows write into. Unlike
    the plain union, one spanning cell (a merged header, an overflowing line) in a table of 4+
    rows does not fuse two columns.
    """
    segs = [s for r in rows for s in r.segments]
    left = int(np.floor(min(s.x0 for s in segs)))
    cover = np.zeros(int(np.ceil(max(s.x1 for s in segs))) - left + 1, dtype=np.int32)
    for row in rows:
        mask = np.zeros_like(cover, dtype=bool)
        for s in row.segments:
            mask[int(s.x0) - left:int(np.ceil(s.x1)) - left + 1] = True
        cover += mask
    filled = np.concatenate([[False], cover > len(rows) // 4, [False]])
    edges = np.flatnonzero(np.diff(filled.astype(np.int8)))
    return [(float(a + left), float(b - 1 + left)) for a, b in zip(edges[::2], edges[1::2])]


def _spans_columns(rows: Sequence[_Row], cols: Sequence[Tuple[float, float]]) -> bool:
    """Every multi-segment row still writes into at least two columns."""
    return len(cols) >= 2 and all(len({_column_of(s, cols) for s in r.segments}) >= 2 for r in rows)


# Leader remnants at a cell edge ("Introduction..", "..10:00", "Closing Remarks -"), and table
# borders read as a character.
_CELL_EDGE = re.compile(r"^(?:[.·_\-~|]{2,}|[|])\s*|\s*(?:[.·_\-~|]{2,}|\s[.·_\-~]|[|])$")


def _cell_text(segments: Sequence[_Segment]) -> str:
    text = " ".join(SEPARATOR_RUN.sub(" ", " ".join(s.text for s in segments)).split())
    return _CELL_EDGE.sub("", text).strip()


def _is_prose(column: List[str]) -> bool:
    """Lines of running text: several words each, most continuing a sentence (lowercase start)."""
    words = float(np.median([len(t.split()) for t in column]))
    lower = sum(1 for t in column[1:] if t[:1].islower()) / max(1, len(column) - 1)
    return words >= 2 * PROSE_WORDS or (words >= PROSE_WORDS and lower >= 0.5)


def _is_table(cells: List[List[str]], cfg: OcrSettings) -> bool:
    """At least 2 rows x 2 columns, and not side-by-side columns of running text (page layout)."""
    if len(cells) < 2 or len(cells[0]) < 2:
        return False
    columns = [[row[c] for row in cells if row[c]] for c in range(len(cells[0]))]
    if cfg.column_detection and min(float(np.median([len(t) for t in col])) for col in columns) > LABEL_CHARS:
        return False  # multi-column reading wins unless one column holds labels
    if len(cells) >= 3 and all(_is_prose(col) for col in columns):
        return False
    return True


def _trim_edges(rows: Sequence[_Row], members: List[Tuple[int, bool]]) -> List[Tuple[int, bool]]:
    """Drop a first or last row that sits much farther from the table than its rows from each
    other (e.g. a "Venue: … Date: …" line above a table whose halves happen to line up)."""
    while True:
        starts = [k for k, (_, cont) in enumerate(members) if not cont]
        if len(starts) < 4:
            return members
        gaps = [rows[members[b][0]].y0 - rows[members[b - 1][0]].y1 for b in starts[1:]]
        typical = float(np.median(gaps[1:-1])) if len(gaps) > 2 else float(np.median(gaps))
        limit = EDGE_GAP_FACTOR * max(typical, 0.0) + 0.25 * rows[members[0][0]].height
        if gaps[0] > limit:
            members = members[starts[1]:]
        elif gaps[-1] > limit:
            members = members[:starts[-1]]
        else:
            return members


def _grow_table(rows: Sequence[_Row], start: int, line_h: float) -> Tuple[List[Tuple[int, bool]], List[Tuple[float, float]]]:
    """Rows from `start` that line up in shared columns: [(row index, is continuation)], columns."""
    members: List[Tuple[int, bool]] = [(start, False)]
    multi = [rows[start]]
    segs = list(rows[start].segments)
    cols = _columns(segs)
    j = start + 1
    while j < len(rows):
        row = rows[j]
        gap = row.y0 - rows[members[-1][0]].y1
        if gap > TABLE_ROW_GAP * line_h:
            break
        # Wrapped lines of cells: tight below their row, each inside one column, and none in
        # the first column (a line with its own first-column cell starts a new row).
        homes = [_single_column(s, cols) for s in row.segments]
        if gap <= CONTINUATION_GAP * line_h and all(k is not None and k > 0 for k in homes):
            members.append((j, True))
            segs.extend(row.segments)
            cols = _columns(segs)
        elif len(row.segments) >= 2:
            new_cols = _columns(segs + row.segments)
            if not _spans_columns(multi + [row], new_cols):
                break
            members.append((j, False))
            multi.append(row)
            segs.extend(row.segments)
            cols = new_cols
        else:
            break
        j += 1
    members = _trim_edges(rows, members)
    kept = [rows[idx] for idx, _ in members]
    union = _columns([s for r in kept for s in r.segments])
    voted = _voted_columns(kept)
    return members, (voted if len(voted) >= len(union) else union)


def _table_cells(rows: Sequence[_Row], members, cols) -> List[List[List[_Segment]]]:
    logical: List[List[List[_Segment]]] = []
    for idx, continuation in members:
        if not continuation or not logical:
            logical.append([[] for _ in cols])
        for seg in rows[idx].segments:
            logical[-1][_column_of(seg, cols)].append(seg)
    return logical


def _find_tables(rows: Sequence[_Row], line_h: float, cfg: OcrSettings) -> List[_Block]:
    tables: List[_Block] = []
    i = 0
    while i < len(rows):
        if len(rows[i].segments) < 2:
            i += 1
            continue
        members, cols = _grow_table(rows, i, line_h)
        logical = _table_cells(rows, members, cols)
        cells = [[_cell_text(cell) for cell in row] for row in logical]
        # Columns left empty after cleaning (e.g. only dot leaders) carry no content.
        keep = [c for c in range(len(cols)) if any(row[c] for row in cells)]
        cells = [[row[c] for c in keep] for row in cells]
        logical = [[row[c] for c in keep] for row in logical]
        if not _is_table(cells, cfg):
            i = max(i, members[-1][0]) + 1  # its shorter tails line up the same way
            continue
        cell_lines = [_joined_line(cell, text)
                      for row, texts in zip(logical, cells) for cell, text in zip(row, texts) if text]
        segs = [s for idx, _ in members for s in rows[idx].segments]
        tables.append(_Block("table", members[0][0], *_bbox(segs), lines=cell_lines, cells=cells,
                             last_row=members[-1][0]))
        i = members[-1][0] + 1
    return tables


# ── 4–5. Text lines, order, rendering ─────────────────────────────────────────

def _text_blocks(index: int, row: _Row, cfg: OcrSettings) -> List[_Block]:
    """A row outside tables: segments within the line-merge gap form one line (gaps wider than
    TAB_GAP as a tab); farther segments (e.g. neighbouring columns) stay separate lines."""
    if not cfg.merge_lines:
        groups = [[_Segment([it], *_box_extent(it.box))] for s in row.segments for it in s.items]
    else:
        groups = [[row.segments[0]]]
        for seg in row.segments[1:]:
            if seg.x0 - groups[-1][-1].x1 > cfg.line_merge_max_gap * row.height:
                groups.append([seg])
            else:
                groups[-1].append(seg)
    blocks = []
    for group in groups:
        raw = group[0].text
        for prev, seg in zip(group, group[1:]):
            raw += ("\t" if seg.x0 - prev.x1 > TAB_GAP * row.height else " ") + seg.text
        text = _clean_ocr_text(raw) if cfg.text_cleanup else squash_spaces(raw)
        if text:
            blocks.append(_Block("text", index, *_bbox(group), lines=[_joined_line(group, text)]))
    return blocks


def _reading_order(blocks: List[_Block], cfg: OcrSettings) -> List[_Block]:
    """Row by row, left to right; with column detection, XY-cut over the blocks (a table
    moves as one unit, so its rows are never split across columns)."""
    if not cfg.column_detection or len(blocks) < 2:
        return sorted(blocks, key=lambda b: (b.row, b.x0))
    proxies = [OcrLine(box=_rect(b.x0, b.y0, b.x1, b.y1), text="", score=0.0) for b in blocks]
    by_proxy: Dict[int, _Block] = {id(p): b for p, b in zip(proxies, blocks)}
    return [by_proxy[id(p)] for p in xy_cut_order(proxies)]


def build_layout(items: List[OcrLine], cfg: OcrSettings) -> OcrLayout:
    """Structured text for the recognised regions of one image."""
    if not items:
        return OcrLayout(text="", plain_text="")
    ext = [_box_extent(it.box) for it in items]
    line_h = float(np.median([max(1.0, y1 - y0) for _, y0, _, y1 in ext]))
    region_gap = min(REGION_GAP, cfg.line_merge_max_gap)
    rows = [_make_row(idx, items, ext, region_gap) for idx in _group_rows(ext)]

    blocks = _find_tables(rows, line_h, cfg)
    in_tables = {r for t in blocks for r in range(t.row, t.last_row + 1)}
    for index, row in enumerate(rows):
        if index not in in_tables:
            blocks.extend(_text_blocks(index, row, cfg))

    out: List[str] = []
    plain: List[str] = []
    lines: List[OcrLine] = []
    prev: Optional[_Block] = None
    for block in _reading_order(blocks, cfg):
        if block.kind == "text" and cfg.text_cleanup and prev is not None and prev.kind == "text" \
                and block.lines[0].text == prev.lines[0].text:
            continue  # consecutive duplicate line
        if prev is not None and (block.kind == "table" or prev.kind == "table"
                                 or block.y0 - prev.y1 > PARAGRAPH_GAP * line_h
                                 or block.y0 < prev.y0 - line_h):
            out.append("")
        if block.kind == "table":
            out.append(format_markdown_table(block.cells or [], title="Table"))
            plain.extend(" ".join(c for c in row if c) for row in block.cells or [])
        else:
            out.append(block.lines[0].text)
            plain.append(block.lines[0].text)
        lines.extend(block.lines)
        prev = block

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return OcrLayout(text=text, plain_text="\n".join(plain).strip(), lines=lines,
                     table_count=sum(1 for b in blocks if b.kind == "table"))
