"""
extraction_store.py
--------------------
Writes/clears the extraction.txt debug-preview artifact: a human
-readable dump of a NormalizedDocument's content, saved immediately
after LLMWhispererExtractor.extract() -- before chunking/embedding
even starts -- so the preview exists even if a later ingestion stage
fails.

(Renamed from normalized_store.py / normalized.txt now that
LLMWhisperer is the single ingestion path -- "extraction" reflects
what actually produces this file.)
"""
from __future__ import annotations

from pathlib import Path

EXTRACTION_DIR = Path("data/extraction")
EXTRACTION_FILE = EXTRACTION_DIR / "extraction.txt"


def save_extraction_document(document_type: str, parser_used: str, content: str) -> None:
    EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
    header = f"document_type: {document_type}\nparser_used: {parser_used}\n\n"
    EXTRACTION_FILE.write_text(header + content, encoding="utf-8")


def clear_extraction_text() -> None:
    if EXTRACTION_FILE.exists():
        EXTRACTION_FILE.unlink()