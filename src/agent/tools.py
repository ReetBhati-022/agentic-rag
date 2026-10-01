"""
tools.py
--------
Native LangChain tool wrappers around the CURRENT (unmodified) tool
implementations, so the LLM can drive them itself via
llm.bind_tools([...]) / response.tool_calls, instead of a fixed
regex-based plan computed up front.

Each tool wraps existing, already-working logic -- it does not
reimplement it:
  - retrieve_documents  -> src.retrieval.retriever.Retriever.retrieve
  - web_search           -> src.tools.web_search.run_web_search
  - calculate             -> src.tools.math_tool.run_math_tool

response_format="content_and_artifact" is used so the tool can return
a short text summary for the LLM (content) while separately carrying
structured data (citations, raw web results, error info) that
src/agent/nodes.py's tool-execution node reads back out via
ToolMessage.artifact -- this is how citation propagation and the
web-search-error surfacing from the previous implementation are
preserved without threading extra state through the LLM's messages.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.tools import tool

from src.retrieval.relevance import check_relevance
from src.retrieval.retriever import Retriever
from src.tools.math_tool import run_math_tool
from src.tools.web_search import run_web_search

_NO_EVIDENCE_MESSAGE = (
    "NO DOCUMENT EVIDENCE: the uploaded document(s) do not appear to contain "
    "information relevant to this query. Do not present the previously "
    "retrieved passages as evidence for this answer. If the question is "
    "about whether the document(s) mention this topic, you can answer now. "
    "If the question asks for real-world/external/current information "
    "about this subject, use the web_search tool before answering -- do "
    "not conclude the information is unavailable without checking it."
)


def make_retrieval_tool(retriever: Retriever, llm):
    @tool("retrieve_documents", response_format="content_and_artifact")
    def retrieve_documents(query: str) -> Tuple[str, Dict[str, Any]]:
        """Search the user's uploaded document(s) for passages relevant to
        the query. Use this ONLY when the question is about the specific
        content of the uploaded file(s) (their statements, forms, reports,
        transactions, figures, etc) -- not for general public knowledge
        about an entity, current/latest/recent information, or anything
        that would be true regardless of whether a document was uploaded.
        Returns the most relevant passages with their source/page, or an
        explicit "no evidence" signal if nothing relevant was found."""
        retrieved = retriever.retrieve(query)
        if not retrieved:
            return "No relevant information was found in the uploaded documents.", {"citations": []}

        # Relevance check (Phase 2): reranking always returns its
        # "least bad" candidates, even when none of them actually
        # relate to the query -- e.g. asking about an entity that
        # never appears in the indexed document. Grade the top_k
        # against the query and drop anything that doesn't survive,
        # so unrelated chunks can never be cited as evidence for a
        # "not found" answer.
        any_relevant, relevant = check_relevance(query, retrieved, llm)
        if not any_relevant:
            return _NO_EVIDENCE_MESSAGE, {"citations": []}

        context = Retriever.format_context(relevant)
        # `type: "document"` plus a snippet of each chunk's own text -- used
        # by evaluate_node for two DIFFERENT purposes (see nodes.py):
        #   1. citation-precision filter (LLM-graded, tolerant of truncation)
        #   2. the deterministic numeric/date pre-check (_deterministic_
        #      numeric_check), which does an EXACT string match of every
        #      £/%/date the answer states against this snippet text.
        # This used to be hard-truncated to [:500] chars. That silently
        # broke (2): a chunk can easily be >500 chars (e.g. a 2-column form
        # region), so a correct figure sitting past char 500 was invisible
        # to the exact-match check -- it would report "doesn't match any
        # amount in the evidence" even though the number was right there in
        # the full chunk, just past the cut. Snippet is now the FULL chunk
        # text so the deterministic check always sees everything the answer
        # was actually generated from. Chunks are already bounded by the
        # chunker's chunk_size, so this isn't unbounded -- it just stops
        # silently discarding part of a single chunk's own text.
        # Retriever.citations() dedupes by (source, page) but drops chunk
        # text, so chunk_id is used to look the matching chunk back up here
        # rather than duplicating Retriever's dedup logic.
        chunk_text_by_id = {c.chunk_id: c.text for c in relevant}
        citations = [
            {**c, "type": "document", "snippet": chunk_text_by_id.get(c["chunk_id"], "")}
            for c in Retriever.citations(relevant)
        ]
        return context, {"citations": citations}

    return retrieve_documents



# Small allowlist of outlets/official-source domains that are reasonable to
# treat as higher-authority for financial/company news. Not exhaustive --
# just enough to (a) sort them ahead of unfamiliar aggregator domains and
# (b) let the model's own prompt instructions distinguish "official/major
# outlet" from "unfamiliar source" without pretending to be a rigorous
# credibility classifier.
_HIGHER_AUTHORITY_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "apnews.com",
    "bbc.com", "bbc.co.uk", "cnbc.com", "barclays.co.uk", "barclays.com",
    "sec.gov", "londonstockexchange.com", "fca.org.uk",
}


def make_web_search_tool(num_results: int = 5):
    @tool("web_search", response_format="content_and_artifact")
    def web_search(query: str) -> Tuple[str, Dict[str, Any]]:
        """Search the web for current, latest, recent, or external/public
        information -- anything that is general real-world knowledge and
        not specific to the uploaded document(s) (e.g. a company's public
        details, recent news, current rates, up-to-date facts). Use this
        directly when the question is clearly external -- do not wait for
        retrieve_documents to fail first. EXCEPTION: if the question uses
        a vague reference ("this bank", "this company", "the
        organization", "the document") instead of naming its subject, you
        must resolve that reference to the specific entity/name FIRST
        (using the loaded-document hint, conversation, or a
        retrieve_documents call if the exact registered name isn't
        already known) -- never pass a vague phrase like "this bank" as
        the query itself.

        Each result's date and outlet are reported to you exactly as
        returned -- 'no date reported' means the source did not supply
        one; never guess, approximate, or infer a date instead. Results
        from well-known/official outlets are listed first and marked
        [known outlet]; unmarked results are from outlets not on that
        short list -- treat those as lower-confidence and say so if you
        rely on one, rather than presenting all sources as equally
        authoritative."""
        result = run_web_search(query, num_results=num_results)
        if not result.get("success"):
            error = result.get("error")
            return f"Web search failed: {error}", {"citations": [], "web_results": [], "error": error}

        web_results: List[Dict[str, Any]] = result.get("results", [])
        # Higher-authority domains first; stable sort preserves Exa's own
        # relevance ordering within each tier.
        web_results = sorted(
            web_results, key=lambda r: r.get("domain") not in _HIGHER_AUTHORITY_DOMAINS
        )

        citations = [
            {
                "type": "web",
                "source": r.get("title") or r.get("url", "web result"),
                "url": r.get("url"),
                "domain": r.get("domain"),
                "published_date": r.get("published_date"),
                "known_outlet": r.get("domain") in _HIGHER_AUTHORITY_DOMAINS,
                "snippet": (r.get("text") or "")[:500],
            }
            for r in web_results
        ]
        if web_results:
            lines = []
            for r in web_results:
                date_str = r.get("published_date") or "no date reported"
                outlet_tag = " [known outlet]" if r.get("domain") in _HIGHER_AUTHORITY_DOMAINS else ""
                domain = r.get("domain") or "unknown domain"
                lines.append(
                    f"- {r['title']} ({domain}{outlet_tag}, published: {date_str}): {r['text'][:300]}"
                )
            content = "\n".join(lines)
        else:
            content = "No web results found."
        return content, {"citations": citations, "web_results": web_results, "error": None}

    return web_search


@tool("calculate")
def calculate(expression: str) -> str:
    """Evaluate a numeric arithmetic expression, e.g. "15000 * 0.15" or
    "sqrt(144)". Pass ONLY the plain numeric expression -- first look up
    any values you need (e.g. via retrieve_documents), substitute them
    in yourself, then call this tool with the resulting expression."""
    result = run_math_tool(expression)
    if result.get("success"):
        return f"Result: {result['result_str']}"
    return f"Could not evaluate '{expression}': {result.get('error')}"


def make_tools(retriever: Retriever, llm, num_web_results: int = 5) -> List[Any]:
    """Builds the tool list passed to llm.bind_tools(...).

    `llm` is used as the relevance grader inside retrieve_documents
    (see src.retrieval.relevance.check_relevance) -- it's the same
    model driving the agent, not a separate provider/config.
    """
    return [make_retrieval_tool(retriever, llm), make_web_search_tool(num_web_results), calculate]