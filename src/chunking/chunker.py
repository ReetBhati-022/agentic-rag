"""
chunker.py
----------
Chunker.chunk(NormalizedDocument) -> List[Chunk]

Splitting is done directly with LangChain's ready-made
RecursiveCharacterTextSplitter.from_tiktoken_encoder(...).split_text(),
no custom splitting logic. Unlike a flat TextLoader().load() over the
extraction.txt preview, this walks NormalizedDocument.pages (and each
page's tables) directly, so every resulting Chunk keeps real
page_number/is_table provenance -- see below for why that matters.

The Chunker owns persistence of its own output: every call to chunk()
immediately writes data/chunks/chunk.txt for the document just chunked
(the Chunker-stage counterpart to extraction.txt), before the caller goes
on to embed/index anything. This mirrors the immediate-persist contract
used by the extraction stage:

    LLMWhispererExtractor -> NormalizedDocument
           -> save_extraction_document() [persisted immediately, writes
              data/extraction/extraction.txt -- a human-readable preview
              only; chunking itself reads normalized_document.pages, not
              this file]
           -> Chunker.chunk(normalized_document) [this module]
           -> save_chunking_text()            [persisted immediately, here]
           -> Embedding -> FaissStore          [caller's job from here]

page_number/is_table are set from the page/table each chunk actually
came from (both were always None/False before, since chunking ran over
the flattened extraction.txt instead of normalized_document.pages).
retriever.py builds its "{source}, page {page_number}" citations and the
PDF viewer's jump-to-page from page_number, and is_table flags which
chunks came from LLMWhisperer's table Markdown (e.g. bank-statement
transactions) rather than page prose -- both now flow through again.

separators also now come from config.yaml's chunking.separators instead
of being silently ignored (from_tiktoken_encoder was never passed a
separators kwarg before, so the config key had no effect).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from src.Ingestion.processing.text_normalizer import NormalizedDocument
from src.services.chunking_store import save_chunking_text


@dataclass
class Chunk:
    """A single retrieval-ready unit of text plus provenance metadata."""

    text: str
    chunk_id: str
    chunk_index: int
    source: str
    document_type: str
    parser_used: str
    page_number: Optional[int] = None
    is_table: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


DEFAULT_CHUNK_SIZE = 350
DEFAULT_CHUNK_OVERLAP = 60
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Must match the embedding model actually used downstream (see FaissStore /
# the embedding step). bge-small-en-v1.5 is BERT-based (WordPiece), with a
# hard max_seq_length of 512 tokens -- anything beyond that is silently
# truncated at embedding time. chunk_size must be counted in *this*
# tokenizer's tokens, not an OpenAI one, or the token counts used to build
# chunks won't match what the embedding model actually sees.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Chunker:
    """Splits each page (and each page's tables) of a NormalizedDocument
    via LangChain's RecursiveCharacterTextSplitter, counting chunk_size/
    chunk_overlap in the embedding model's own tokens (bge-small-en-v1.5's
    WordPiece tokenizer), not an OpenAI tokenizer -- otherwise the token
    counts used here don't match what actually reaches the embedding
    model, and chunks can silently exceed bge's 512-token limit and get
    truncated at embedding time.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        cfg = (config or {}).get("chunking", {}) if config else {}
        self.chunk_size = chunk_size or cfg.get("chunk_size", DEFAULT_CHUNK_SIZE)
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else cfg.get(
            "chunk_overlap", DEFAULT_CHUNK_OVERLAP
        )
        self.separators = cfg.get("separators", DEFAULT_SEPARATORS)
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def chunk(self, normalized_document: NormalizedDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        index = 0

        def _emit(text: str, page_number: Optional[int], is_table: bool) -> None:
            nonlocal index
            for piece in self.splitter.split_text(text):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        text=piece,
                        chunk_id=f"{normalized_document.source}_{index}",
                        chunk_index=index,
                        source=normalized_document.source,
                        document_type=normalized_document.document_type,
                        parser_used=normalized_document.parser_used,
                        page_number=page_number,
                        is_table=is_table,
                    )
                )
                index += 1

        for page in normalized_document.pages:
            if page.text.strip():
                _emit(page.text, page.page_number, is_table=False)
            for table in page.tables:
                if table.markdown.strip():
                    _emit(table.markdown, page.page_number, is_table=True)

        save_chunking_text(chunks)
        return chunks