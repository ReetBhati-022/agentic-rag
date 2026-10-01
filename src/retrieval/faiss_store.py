"""
faiss_store.py
--------------
FaissStore owns the actual FAISS index: building the embedding model
from config.yaml's `embeddings` section, adding Chunks, and persisting/
restoring both the FAISS binary index and the parallel list of Chunk
objects (needed for BM25/citations in Retriever, and for the faiss.txt
debug log in src/services/faiss_log.py) to disk under
app.vectorstore_dir.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chunking.chunker import Chunk


class FaissStore:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        emb_cfg = config.get("embeddings", {})
        vs_cfg = config.get("vectorstore", {})

        self.index_dir = Path(config.get("app", {}).get("vectorstore_dir", "data/vectorstore"))
        self.index_name = vs_cfg.get("index_name", "faiss_index")

        self.embeddings = self._build_embeddings(emb_cfg)
        self._vectorstore = None
        self._chunks: List[Chunk] = []
        # Bumped on every index change; lets caches (agent tool results) detect a stale corpus.
        self.version = 0

    @staticmethod
    def _build_embeddings(emb_cfg: Dict[str, Any]):
        from langchain_huggingface import HuggingFaceEmbeddings

        encode_kwargs = dict(emb_cfg.get("encode_kwargs", {}))
        encode_kwargs.setdefault("normalize_embeddings", emb_cfg.get("normalize_embeddings", True))
        return HuggingFaceEmbeddings(
            model_name=emb_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
            model_kwargs={"device": emb_cfg.get("device", "cpu")},
            encode_kwargs=encode_kwargs,
        )

    # -- state -----------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._vectorstore is not None and len(self._chunks) > 0

    @property
    def chunks(self) -> List[Chunk]:
        return self._chunks

    @property
    def vectorstore(self):
        return self._vectorstore

    # -- mutation ----------------------------------------------------------
    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        from langchain_core.documents import Document

        docs = [Document(page_content=c.text, metadata=self._chunk_metadata(c)) for c in chunks]

        if self._vectorstore is None:
            from langchain_community.vectorstores import FAISS

            self._vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self._vectorstore.add_documents(docs)
        self._chunks.extend(chunks)
        self.version += 1

    @staticmethod
    def _chunk_metadata(chunk: Chunk) -> Dict[str, Any]:
        return {
            # chunk.metadata carries headings + NormalizedDocument.structured_fields
            # (e.g. bank-statement transactions, PP-StructureV3 key/values) --
            # merged first so the identity fields below always win if a key
            # ever collided.
            **chunk.metadata,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "source": chunk.source,
            "document_type": chunk.document_type,
            "parser_used": chunk.parser_used,
            "page_number": chunk.page_number,
            "is_table": chunk.is_table,
        }

    def clear(self) -> None:
        self._vectorstore = None
        self._chunks = []
        self.version += 1
        if self.index_dir.exists():
            for pattern in (f"{self.index_name}.faiss", f"{self.index_name}.pkl", f"{self.index_name}.chunks.pkl"):
                path = self.index_dir / pattern
                if path.exists():
                    path.unlink()

    # -- persistence ----------------------------------------------------
    def save(self) -> None:
        if self._vectorstore is None:
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(self.index_dir), index_name=self.index_name)
        with open(self.index_dir / f"{self.index_name}.chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self) -> None:
        faiss_path = self.index_dir / f"{self.index_name}.faiss"
        chunks_path = self.index_dir / f"{self.index_name}.chunks.pkl"
        if not faiss_path.exists() or not chunks_path.exists():
            return

        from langchain_community.vectorstores import FAISS

        self._vectorstore = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            index_name=self.index_name,
            allow_dangerous_deserialization=True,
        )
        with open(chunks_path, "rb") as f:
            self._chunks = pickle.load(f)
        self.version += 1