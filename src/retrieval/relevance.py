"""
relevance.py
------------
Final "are these chunks actually relevant?" gate that runs AFTER
retrieval + hybrid fusion + reranking (see retriever.py's pipeline
docstring):

    Query -> FAISS+BM25 -> RRF -> Reranker -> top_k -> [this module]

Reranker/RRF only ever produce a RANKING among whatever candidates
exist -- they always return *something*, even when nothing in the
index actually relates to the query (e.g. asking about "Capsitech"
against a Barclays bank statement: the reranker will still hand back
its "least bad" Barclays chunks, ranked with real-looking scores).

A raw cross-encoder score threshold was tried first and rejected --
see config.yaml's `retrieval.score_threshold` comment: the
ms-marco-MiniLM reranker emits negative logits even for genuinely
relevant pairs (observed roughly -11 to -8), so no single fixed cutoff
is safe across queries/documents without per-deployment calibration.

Instead this asks the LLM directly, once per retrieval call, whether
the top_k chunks actually contain information relevant to the
question. It's a single short structured call over <= top_k already-
short chunks, so the added cost/latency is small and bounded.
"""
from __future__ import annotations

import json
import logging
from typing import List, Tuple

from src.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Cap on how much of each chunk's text is shown to the grader -- the
# grader only needs enough to judge topical relevance, not the full
# passage (which the main answer LLM sees separately, unmodified).
_MAX_CHARS_PER_PASSAGE = 800

_RELEVANCE_PROMPT = """You are a strict relevance grader for a retrieval-augmented \
question-answering system. You will be given a user QUESTION and a list of \
numbered PASSAGES retrieved from the user's uploaded document(s).

For EACH passage, decide whether it contains information a careful reader \
would actually use as evidence to answer the QUESTION. A passage that merely \
comes from the same document but concerns a different entity, transaction, \
date, account, or topic than the one asked about is NOT relevant, even if it \
shares the same document type or some vocabulary with the question.

QUESTION: {question}

PASSAGES:
{passages}

Respond with ONLY a JSON object and nothing else, in this exact form:
{{"relevant_ids": [<ids of passages that are actually relevant; [] if none are>]}}
"""


def check_relevance(
    question: str, chunks: List[RetrievedChunk], llm
) -> Tuple[bool, List[RetrievedChunk]]:
    """Grade `chunks` against `question` and keep only the relevant ones.

    Returnvs (any_relevant, filtered_chunks):
      - any_relevant is True iff at least one chunk survived grading.
      - filtered_chunks is the relevant subset, in its original order.

    Fails OPEN (returns the original, ungraded chunks) if the grader
    call errors or its response can't be parsed -- a grading bug
    should never be able to silently hide real evidence from the user;
    it should only ever be able to fail to *reject* irrelevant chunks,
    which is the pre-existing (already accepted) behavior.
    """
    if not chunks:
        return False, []

    passages = "\n\n".join(
        f"[{i}] {c.text[:_MAX_CHARS_PER_PASSAGE]}" for i, c in enumerate(chunks)
    )
    prompt = _RELEVANCE_PROMPT.format(question=question, passages=passages)

    try:
        response = llm.invoke(prompt)
        content = (getattr(response, "content", None) or str(response)).strip()
        # Tolerate ```json fences or stray commentary around the object.
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"no JSON object found in grader response: {content!r}")
        parsed = json.loads(content[start : end + 1])
        relevant_ids = {int(i) for i in parsed.get("relevant_ids", [])}
    except Exception:
        logger.warning(
            "Relevance grading failed to parse; failing open (keeping all %d chunk(s)).",
            len(chunks),
            exc_info=True,
        )
        return True, chunks

    filtered = [c for i, c in enumerate(chunks) if i in relevant_ids]
    return bool(filtered), filtered