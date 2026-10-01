"""
chunking_store.py
------------------
Writes/clears the chunk.txt debug-preview artifact -- the
Chunker-stage counterpart to extraction.txt (see
src/services/extraction_store.py). Chunker.chunk() calls
save_chunking_text() itself, immediately after splitting, before the
caller goes on to embed/index anything.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

CHUNKS_DIR = Path("data/chunks")
CHUNKS_FILE = CHUNKS_DIR / "chunk.txt"


def save_chunking_text(chunks: List) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for c in chunks:
        lines.append(f"--- chunk_id={c.chunk_id} page={c.page_number} is_table={c.is_table} ---")
        lines.append(c.text)
        lines.append("")
    CHUNKS_FILE.write_text("\n".join(lines), encoding="utf-8")


def clear_chunking_text() -> None:
    if CHUNKS_FILE.exists():
        CHUNKS_FILE.unlink()