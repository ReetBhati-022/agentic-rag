"""
llmwhisperer_extractor.py
--------------------------
LLMWhispererExtractor.extract(file_path, source_name) -> NormalizedDocument

Replaces the old routing/parsing pipeline (DocumentRouter + bank
statement parser + PaddleOCR + PyMuPDF4LLM + field association, etc.)
with a single call to Unstract's LLMWhisperer API. LLMWhisperer handles
digital PDFs, scanned PDFs, and forms/tables (layout-preserving output)
all through one endpoint, so there is no more digital/scanned/bank
-statement split to make here.

The result is wrapped into the same NormalizedDocument shape the rest
of the app already expects (Chunker, extraction_store.py), so nothing
downstream of ingestion has to change.
"""
from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from unstract.llmwhisperer import LLMWhispererClientV2
from unstract.llmwhisperer.client_v2 import LLMWhispererClientException

from src.Ingestion.processing.text_normalizer import (
    NormalizedDocument,
    NormalizedPage,
)

load_dotenv()


class LLMWhispererExtractionError(Exception):
    """Raised when LLMWhisperer can't extract a document (missing API
    key, upload/API failure, or an empty result)."""


class LLMWhispererExtractor:
    """Thin wrapper around LLMWhispererClientV2.whisper(), normalized
    into a NormalizedDocument."""

    def __init__(
        self,
        mode: str = "form",
        output_mode: str = "layout_preserving",
        wait_timeout: int = 300,
    ):
        self.mode = mode
        self.output_mode = output_mode
        self.wait_timeout = wait_timeout

        if not os.getenv("LLMWHISPERER_API_KEY"):
            raise LLMWhispererExtractionError(
                "LLMWHISPERER_API_KEY is not set. Add it to your .env file."
            )
        self._client = LLMWhispererClientV2()

    def extract(self, file_path: str, source_name: str) -> NormalizedDocument:
        try:
            result = self._client.whisper(
                file_path=file_path,
                mode=self.mode,
                output_mode=self.output_mode,
                wait_for_completion=True,
                wait_timeout=self.wait_timeout,
            )
        except LLMWhispererClientException as exc:
            raise LLMWhispererExtractionError(f"LLMWhisperer error: {exc}") from exc

        text = (result or {}).get("extraction", {}).get("result_text", "")
        if not text.strip():
            raise LLMWhispererExtractionError(
                f"LLMWhisperer returned no extractable text for '{source_name}'."
            )

        return NormalizedDocument(
            source=source_name,
            document_type="llmwhisperer",
            parser_used="llmwhisperer",
            pages=self._split_into_pages(text),
        )

    @staticmethod
    def _split_into_pages(text: str) -> List[NormalizedPage]:
        """LLMWhisperer separates pages with a form-feed character
        (\\x0c). Falls back to a single page if none is present."""
        raw_pages = text.split("\x0c") if "\x0c" in text else [text]
        # reflow_two_column_text is a no-op (returns text unchanged) unless
        # the page confidently looks like a layout-preserved 2-column form
        # -- see its docstring in text_normalizer.py for why this is needed
        # (fixes label/value pairs picking up the WRONG column's value on
        # forms like SA104F).
        pages = [
            NormalizedPage(page_number=i, text=page_text.strip())
            for i, page_text in enumerate(raw_pages, start=1)
            if page_text.strip()
        ]
        return pages or [NormalizedPage(page_number=1, text=text.strip())]

