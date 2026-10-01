"""
text_normalizer.py
-------------------
The common output shape ingestion (LLMWhispererExtractor -- see
src/Ingestion/llmwhisperer_extractor.py) is converted into before
chunking. Keeping this shared NormalizedDocument means Chunker
(src/chunking/chunker.py) never has to know which extraction path
produced a given file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# -- two-column form reflow: REMOVED --------------------------------------
#
# A heuristic here used to de-interleave layout_preserving 2-column form
# pages (fixing HMRC SA104F's box-11-shows-box-19's-value bug) by pulling
# left-column and right-column text into separate streams. Removed after
# three iterations (gap-count, exact-segment-count, then column-band
# clustering) each still misclassified real cases: a genuine 2-column
# TABLE (not a form) has the same 2-recurring-column signature as a
# 2-column form, and exact-width padding edge cases kept slipping past
# whichever gate was current. See tests/test_text_normalizer.py's
# test_KNOWN_GAP_* case for the last confirmed failure mode before removal.
#
# reflow_two_column_text() is gone; page text now passes through exactly
# as LLMWhisperer returns it. This means the original box-11/box-19
# interleaving this was built to fix may resurface for HMRC-style forms
# -- the remaining defense is mode auto-detection (mode_detector.py)
# routing form-shaped documents to mode="form" so LLMWhisperer's own
# field-mapping produces cleaner output at the source, not a post-hoc
# text patch. Re-verify against a real SA104F extraction; if box values
# still land on the wrong line, that needs a fix upstream (mode/prompt
# tuning, or LLMWhisperer's own field-association output) rather than
# another local heuristic here.


@dataclass
class NormalizedTable:
    """A table already rendered to Markdown by whichever parser found it."""

    markdown: str
    row_count: int = 0
    col_count: int = 0


@dataclass
class NormalizedPage:
    page_number: int
    text: str = ""
    headings: List[str] = field(default_factory=list)
    tables: List[NormalizedTable] = field(default_factory=list)


@dataclass
class NormalizedDocument:
    source: str
    document_type: str          # "llmwhisperer"
    parser_used: str            # "llmwhisperer"
    pages: List[NormalizedPage] = field(default_factory=list)
    structured_fields: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Plain-text/Markdown view of the whole document (page text +
        table Markdown, in page order). Used for the extraction.txt
        preview artifact (see src/services/extraction_store.py)."""
        parts: List[str] = []
        for page in self.pages:
            if page.text.strip():
                parts.append(page.text.strip())
            for table in page.tables:
                if table.markdown.strip():
                    parts.append(table.markdown.strip())
        return "\n\n".join(parts)

    @property
    def page_count(self) -> int:
        return len(self.pages)
_GAP_RE = re.compile(r" {15,}")
_COLUMN_SPLIT_COL = 45
_MIN_SPLIT_LINES = 3
_MIN_TWO_COLUMN_LINE_FRACTION = 0.25


def _line_segments(line: str) -> List[Tuple[int, str]]:
    """Split a line on wide (15+ space) gaps into (start_column, text)
    segments, dropping pure whitespace segments."""
    segments: List[Tuple[int, str]] = []
    last_end = 0
    for m in _GAP_RE.finditer(line):
        chunk = line[last_end:m.start()]
        if chunk.strip():
            offset = len(chunk) - len(chunk.lstrip(" "))
            segments.append((last_end + offset, chunk.strip()))
        last_end = m.end()
    tail = line[last_end:]
    if tail.strip():
        offset = len(tail) - len(tail.lstrip(" "))
        segments.append((last_end + offset, tail.strip()))
    return segments


def _looks_two_column(lines: List[str]) -> bool:
    """Heuristic gate: only treat a page as two-column form layout when
    there's real, repeated evidence of it -- a handful of lines whose
    content clearly starts in a second column, or that split into two
    separate chunks across a wide gap. Ordinary paragraphs/tables won't
    trigger this."""
    non_blank = [l for l in lines if l.strip()]
    if len(non_blank) < 6:
        return False

    two_column_lines = 0
    split_lines = 0
    for line in non_blank:
        segs = _line_segments(line)
        if len(segs) >= 2:
            split_lines += 1
            two_column_lines += 1
        elif segs and segs[0][0] >= _COLUMN_SPLIT_COL:
            two_column_lines += 1

    if split_lines < _MIN_SPLIT_LINES:
        return False
    return (two_column_lines / len(non_blank)) >= _MIN_TWO_COLUMN_LINE_FRACTION


def reflow_two_column_text(text: str) -> str:
    """If `text` looks like a layout-preserved two-column form page,
    de-interleave it into column-1-then-column-2 order so each column
    reads as its own correctly-ordered label->value sequence. Returns
    the text unchanged otherwise (including on any page that isn't
    confidently two-column, e.g. tables, single-column prose)."""
    lines = text.split("\n")
    if not _looks_two_column(lines):
        return text

    left_stream: List[str] = []
    right_stream: List[str] = []
    for line in lines:
        for start_col, chunk in _line_segments(line):
            (left_stream if start_col < _COLUMN_SPLIT_COL else right_stream).append(chunk)

    if not left_stream or not right_stream:
        return text

    return "\n".join(left_stream) + "\n\n" + "\n".join(right_stream)
