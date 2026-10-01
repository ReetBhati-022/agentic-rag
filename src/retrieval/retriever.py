"""
retriever.py
------------
Retriever.retrieve(question) -> List[RetrievedChunk], implementing the
pipeline described by config.yaml's `retrieval` section:

    dense (FAISS/MMR) ---+
                          +--> Reciprocal Rank Fusion --> cross-encoder rerank --> top_k
    lexical (BM25)     ---+

Hybrid fusion and reranking are each individually toggleable
(retrieval.hybrid.enabled / retrieval.reranker.enabled); with both off
this degrades to plain dense MMR search.

Retriever.format_context() / Retriever.citations() are also used
directly by app.py's plain (non-agent) /api/chat route and by
src/agent/nodes.py's tools_node.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.retrieval.faiss_store import FaissStore


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page_number: Optional[int]
    chunk_id: str
    is_table: bool
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Retriever:
    def __init__(self, store: FaissStore, config: Dict[str, Any]):
        self.store = store
        r_cfg = config.get("retrieval", {})

        self.top_k = r_cfg.get("top_k", 5)
        self.fetch_k = r_cfg.get("fetch_k", 20)
        self.lambda_mult = r_cfg.get("lambda_mult", 0.5)
        self.score_threshold = r_cfg.get("score_threshold")

        hybrid_cfg = r_cfg.get("hybrid", {})
        self.hybrid_enabled = hybrid_cfg.get("enabled", True)
        self.bm25_k = hybrid_cfg.get("bm25_k", 20)
        self.rrf_k = hybrid_cfg.get("rrf_k", 60)

        rerank_cfg = r_cfg.get("reranker", {})
        self.rerank_enabled = rerank_cfg.get("enabled", True)
        self.rerank_model_name = rerank_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.candidate_k = rerank_cfg.get("candidate_k", 20)

        self._bm25 = None
        self._bm25_chunks: List = []
        self._bm25_fingerprint: tuple = ()
        self._reranker = None

    # -- public API ---------------------------------------------------
    def retrieve(self, question: str) -> List[RetrievedChunk]:
        if not self.store.is_ready:
            return []

        dense = self._dense_search(question)

        if self.hybrid_enabled:
            lexical = self._bm25_search(question)
            fused = self._reciprocal_rank_fusion(dense, lexical)
        else:
            fused = dense

        if not fused:
            return []

        if self.rerank_enabled:
            candidates = fused[: self.candidate_k]
            results = self._rerank(question, candidates)[: self.top_k]
        else:
            results = fused[: self.top_k]

        if self.score_threshold is not None and self.rerank_enabled:
            results = [r for r in results if r.score >= self.score_threshold]

        return results

    @staticmethod
    def format_context(retrieved: List[RetrievedChunk]) -> str:
        if not retrieved:
            return ""
        parts = []
        for r in retrieved:
            location = r.source if not r.page_number else f"{r.source}, page {r.page_number}"
            chunk_text = f"[Source: {location}]\n{r.text}"
            # Structured field/transaction data (see chunker.py's
            # per-page structured_fields) rides in r.metadata, not in
            # r.text. Ingestion is now a single LLMWhisperer call (see
            # src/Ingestion/llmwhisperer_extractor.py), which doesn't
            # populate structured_fields itself, so this branch is
            # effectively dormant today -- kept in case a future
            # extraction path starts populating it again. For everything
            # else (plain
            # pages with no fields, bank-statement pages whose raw text
            # is untouched), r.text is still the raw page text a parser
            # produced, so the FIELD/TRANSACTION lines below are what
            # actually gets the reconstructed values in front of the
            # LLM -- citations() below carries the same fields, but
            # that's only wired into the separate groundedness-
            # verification step, not generation. Render explicit
            # label/value and transaction lines the same way
            # _format_evidence_block already does, so both paths show
            # the same evidence.
            already_has_structured_block = "Structured fields:" in r.text
            structured_lines = []
            if not already_has_structured_block:
                for key, value in (r.metadata.get("key_values") or {}).items():
                    structured_lines.append(f"FIELD -- {key}: {value}")
            for txn in r.metadata.get("transactions") or []:
                structured_lines.append(
                    "TRANSACTION -- "
                    f"Date: {txn.get('date') or 'unknown'} | "
                    f"Description: {txn.get('description') or ''} | "
                    f"Money In: {txn.get('credit')} | "
                    f"Money Out: {txn.get('debit')} | "
                    f"Amount: {txn.get('amount')} | "
                    f"Balance After: {txn.get('balance')}"
                )
            if structured_lines:
                chunk_text += "\n" + "\n".join(structured_lines)
            parts.append(chunk_text)
        return "\n\n".join(parts)

    @staticmethod
    def citations(retrieved: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for r in retrieved:
            key = (r.source, r.page_number)
            if key in seen:
                continue
            seen.add(key)
            citation: Dict[str, Any] = {"source": r.source, "page": r.page_number, "chunk_id": r.chunk_id}
            # Carry the structured transaction/key-value fields through to
            # the citation (see chunker.py's per-page structured_fields)
            # so downstream evidence formatting (nodes.py's
            # _format_evidence_block) can render explicit
            # Date/Money In/Money Out/Balance fields instead of relying on
            # a truncated raw-text snippet that may cut the relevant row.
            if r.metadata.get("transactions"):
                citation["transactions"] = r.metadata["transactions"]
            if r.metadata.get("key_values"):
                citation["key_values"] = r.metadata["key_values"]
            out.append(citation)
        return out

    # -- dense (FAISS / MMR) -------------------------------------------
    def _dense_search(self, question: str) -> List[RetrievedChunk]:
        vectorstore = self.store.vectorstore
        if vectorstore is None:
            return []
        docs = vectorstore.max_marginal_relevance_search(
            question, k=self.top_k, fetch_k=self.fetch_k, lambda_mult=self.lambda_mult
        )
        # MMR search doesn't return a comparable similarity score; RRF
        # fusion below only needs rank order, so a simple 1/rank score
        # is enough here (real scores come from the reranker later).
        return [self._doc_to_chunk(doc, score=1.0 / (i + 1)) for i, doc in enumerate(docs)]

    @staticmethod
    def _doc_to_chunk(doc, score: float) -> RetrievedChunk:
        meta = doc.metadata or {}
        return RetrievedChunk(
            text=doc.page_content,
            source=meta.get("source", "unknown"),
            page_number=meta.get("page_number"),
            chunk_id=meta.get("chunk_id", ""),
            is_table=meta.get("is_table", False),
            score=score,
            metadata=meta,
        )

    # -- lexical (BM25) ---------------------------------------------------
    @staticmethod
    def _corpus_fingerprint(chunks) -> tuple:
        # Chunk IDs (not just count) -- two different corpora can easily
        # have the same chunk count, which previously meant a stale BM25
        # index built from a since-replaced/cleared corpus could get
        # reused silently.
        return tuple(getattr(c, "chunk_id", None) or id(c) for c in chunks)

    def _ensure_bm25(self) -> None:
        chunks = self.store.chunks
        fingerprint = self._corpus_fingerprint(chunks)
        if self._bm25 is not None and fingerprint == self._bm25_fingerprint:
            return
        from rank_bm25 import BM25Okapi

        self._bm25_chunks = list(chunks)
        self._bm25_fingerprint = fingerprint
        tokenized = [c.text.lower().split() for c in self._bm25_chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _bm25_search(self, question: str) -> List[RetrievedChunk]:
        self._ensure_bm25()
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(question.lower().split())
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.bm25_k]

        results = []
        for rank, idx in enumerate(ranked_idx):
            if scores[idx] <= 0:
                break
            c = self._bm25_chunks[idx]
            results.append(
                RetrievedChunk(
                    text=c.text, source=c.source, page_number=c.page_number,
                    chunk_id=c.chunk_id, is_table=c.is_table, score=1.0 / (rank + 1),
                    metadata=dict(c.metadata),
                )
            )
        return results

    # -- fusion -----------------------------------------------------------
    def _reciprocal_rank_fusion(
        self, dense: List[RetrievedChunk], lexical: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        rrf_scores: Dict[str, float] = {}
        items: Dict[str, RetrievedChunk] = {}

        for rank_list in (dense, lexical):
            for rank, item in enumerate(rank_list):
                key = item.chunk_id or item.text[:64]
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                items.setdefault(key, item)

        fused = sorted(items.values(), key=lambda it: rrf_scores[it.chunk_id or it.text[:64]], reverse=True)
        for it in fused:
            it.score = rrf_scores[it.chunk_id or it.text[:64]]
        return fused

    # -- warm-up -----------------------------------------------------------
    def warmup(self) -> None:
        """Loads the reranker and builds the BM25 index up front, so the
        first real question doesn't pay for them. Same objects/logic the
        first retrieve() would create lazily -- results are unchanged."""
        if self.rerank_enabled:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.rerank_model_name)
            self._reranker.predict([("warmup", "warmup")])
        if self.store.is_ready and self.store.chunks:
            self._ensure_bm25()

    # -- reranking ---------------------------------------------------------
    def _rerank(self, question: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not candidates:
            return candidates
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.rerank_model_name)

        pairs = [(question, c.text) for c in candidates]
        raw_scores = self._reranker.predict(pairs)
        for c, s in zip(candidates, raw_scores):
            c.score = float(s)
        return sorted(candidates, key=lambda c: c.score, reverse=True)