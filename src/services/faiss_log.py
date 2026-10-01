"""
faiss_log.py
------------
Writes/clears faiss.txt: a plain-text, human-readable summary of what's
currently indexed in FaissStore (FaissStore itself owns the actual
binary *.faiss / *.chunks.pkl persistence -- see
src/retrieval/faiss_store.py). Regenerated after every successful
FaissStore.save().
"""
from __future__ import annotations

from pathlib import Path

LOG_DIR = Path("data/vectorstore")
LOG_FILE = LOG_DIR / "faiss.txt"


def save_faiss_log(store) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    chunks = getattr(store, "chunks", [])
    lines = [f"Indexed chunks: {len(chunks)}", ""]
    for c in chunks:
        lines.append(
            f"{c.chunk_id} | source={c.source} | page={c.page_number} | "
            f"table={c.is_table} | parser={c.parser_used}"
        )
    LOG_FILE.write_text("\n".join(lines), encoding="utf-8")


def clear_faiss_log() -> None:
    if LOG_FILE.exists():
        LOG_FILE.unlink()