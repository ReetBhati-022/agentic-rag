from __future__ import annotations

import json
import logging
import re
import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agent.state import AgentState

logger = logging.getLogger(__name__)

# Bounds sequential/parallel tool-calling rounds within one outer pass.
# Native tool calling has no built-in bound, so one is needed here to
# guarantee the graph always terminates.
_MAX_TOOL_ROUNDS_PER_TURN = 5

# Some models (observed: gpt-oss:120b-cloud via Ollama) keep inserting
# inline pseudo-citation brackets into `answer` despite the system
# prompt explicitly telling them not to -- the real citation UI (source
# chips, built from tool artifacts) is unaffected by this either way, so
# this is purely a text-cleanup backstop for when the prompt instruction
# alone doesn't hold. U+3010/U+3011 (full-width brackets) are distinct
# enough from normal English punctuation that stripping them is safe.
_INLINE_CITATION_MARKER_RE = re.compile(r"\s*【[^】]*】")


def _strip_inline_citation_markers(text: str) -> str:
    return _INLINE_CITATION_MARKER_RE.sub("", text).strip()


def _dedupe_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for c in citations:
        key = (c.get("type"), c.get("source"), c.get("page"), c.get("url"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def make_recall_memory_node(memory_store):
    """Runs first, before start_turn. Fetches this user's long-term
    memories relevant to the incoming question and stashes them in
    state["memories"] for start_turn_node to fold into the question.

    Only does real work on a fresh turn (iteration == 0) -- an internal
    retry (revised plan for the SAME question) re-enters the graph at
    start_turn directly, not here, so this node simply doesn't run again
    within one turn; no special-casing needed."""

    def recall_memory_node(state: AgentState) -> Dict[str, Any]:
        user_id = state.get("user_id")
        if not user_id or memory_store is None:
            return {"memories": []}
        memories = memory_store.search(user_id, state["question"])
        return {"memories": memories}

    return recall_memory_node


def make_start_turn_node(system_prompt: str):
    def start_turn_node(state: AgentState) -> Dict[str, Any]:
        from langgraph.graph.message import RemoveMessage

        indexed_files = state.get("indexed_files") or []
        question = state["question"]
        if indexed_files:
            # Prepended to the question (not the system prompt) so it's
            # tied to this turn only, not a permanent instruction.
            files_list = ", ".join(indexed_files)
            question = (
                f"[Currently loaded document(s): {files_list}. If my question below uses a "
                f"vague reference like \"this bank\"/\"this company\"/\"the document\", it refers "
                f"to whatever entity these document(s) are about -- use retrieve_documents first "
                f"if you need the exact registered name before calling web_search.]\n\n{question}"
            )

        # Long-term memory recall (set once by recall_memory_node, before
        # this node runs -- see graph.py). Folded into the question, not
        # the system prompt, for the same reason as indexed_files above:
        # this is per-turn context, not a standing instruction, and the
        # system prompt is only ever sent once per thread (see
        # `is_new_turn` below) while recalled memories can legitimately
        # differ turn to turn.
        memories = state.get("memories") or []
        if memories:
            facts_list = "\n".join(f"- {m}" for m in memories)
            question = (
                f"[Known long-term facts about this user, from earlier sessions -- use them "
                f"only if relevant to the question below, don't recite them unprompted:\n"
                f"{facts_list}]\n\n{question}"
            )

        prev = state.get("messages") or []

        # iteration == 0 only on the first pass of a new question (a
        # retry via should_retry -> "start_turn" arrives with iteration
        # already >= 1); distinguishes "new turn" from "internal retry".
        is_new_turn = state.get("iteration", 0) == 0

        updates: Dict[str, Any] = {
            "tool_round": 0,
            "citations": [],
            "web_results": [],
            "web_search_error": None,
            "math_result": None,
            "answer": "",
            "iteration": state.get("iteration", 0) + 1,
        }

        if is_new_turn:
            # Prior turns' messages stay untouched; add_messages appends.
            # SystemMessage only sent once per thread, not every turn.
            new_messages: List[Any] = (
                [SystemMessage(content=system_prompt)] if not prev else []
            ) + [HumanMessage(content=question)]
            updates["messages"] = new_messages
            updates["turn_boundary_ids"] = [m.id for m in prev if getattr(m, "id", None)]
        else:
            # Retry: remove only this turn's own messages, keep earlier turns.
            boundary_ids = set(state.get("turn_boundary_ids") or [])
            to_remove = [m for m in prev if getattr(m, "id", None) and m.id not in boundary_ids]
            updates["messages"] = [RemoveMessage(id=m.id) for m in to_remove] + [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ]

        return updates

    return start_turn_node


def make_agent_node(llm, tools):
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> Dict[str, Any]:
        response = llm_with_tools.invoke(state["messages"])
        updates: Dict[str, Any] = {"messages": [response]}
        if not getattr(response, "tool_calls", None):
            updates["answer"] = getattr(response, "content", "") or ""
        return updates

    return agent_node


# Simple process-wide tool-result cache: {(tool_name, json args) -> (saved_at, msg, artifact)}.
# Plain dict, no new dependency -- avoids re-running retrieve_documents/web_search
# when the exact same tool call (same name + same args) happens again soon,
# e.g. a retry re-asking the same question, or the same query across turns.
_tool_cache: Dict[Any, Any] = {}
_TOOL_CACHE_TTL_SECONDS = 300
_TOOL_CACHE_MAX_ENTRIES = 256


def _prune_tool_cache() -> None:
    now = time.perf_counter()
    for key in [k for k, v in _tool_cache.items() if now - v[0] >= _TOOL_CACHE_TTL_SECONDS]:
        del _tool_cache[key]
    while len(_tool_cache) > _TOOL_CACHE_MAX_ENTRIES:
        _tool_cache.pop(next(iter(_tool_cache)))  # dicts keep insertion order -> oldest first


def _parse_math_result(content: str) -> Dict[str, Any]:
    """Turns the calculate tool's "Result: 300" text into the dict the verifier expects,
    including a numeric `result` (read by _format_evidence_block)."""
    parsed: Dict[str, Any] = {"success": content.startswith("Result:"), "result_str": content}
    if parsed["success"]:
        try:
            parsed["result"] = float(content[len("Result:"):].strip())
        except ValueError:
            pass
    return parsed


def make_tools_node(tools, index_version: Optional[Callable[[], Any]] = None):
    tools_by_name = {t.name: t for t in tools}

    def tools_node(state: AgentState) -> Dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        logger.info("tools_node: model called %s", [c["name"] for c in tool_calls])
        new_messages: List[ToolMessage] = []
        citations = list(state.get("citations", []))
        web_results = list(state.get("web_results", []))
        web_search_error = state.get("web_search_error")
        math_result = state.get("math_result")

        for call in tool_calls:
            tool_fn = tools_by_name.get(call["name"])
            if tool_fn is None:
                new_messages.append(
                    ToolMessage(content=f"Error: unknown tool '{call['name']}'.", tool_call_id=call["id"])
                )
                continue

            # retrieve_documents results depend on the indexed corpus, so the index version is part
            # of its key -- a new upload/clear must never serve the previous document's passages.
            corpus_version = index_version() if index_version and call["name"] == "retrieve_documents" else None
            cache_key = (call["name"], json.dumps(call.get("args", {}), sort_keys=True), corpus_version)
            cached = _tool_cache.get(cache_key)
            if cached and (time.perf_counter() - cached[0]) < _TOOL_CACHE_TTL_SECONDS:
                _saved_at, cached_content, cached_artifact = cached
                msg = ToolMessage(content=cached_content, tool_call_id=call["id"], artifact=cached_artifact)
            else:
                try:
                    msg = tool_fn.invoke(call)
                except Exception as exc:
                    new_messages.append(ToolMessage(content=f"Tool error: {exc}", tool_call_id=call["id"]))
                    continue
                # Don't cache failures (e.g. a transient web_search error) for the whole TTL.
                if not (getattr(msg, "artifact", None) or {}).get("error"):
                    _tool_cache[cache_key] = (time.perf_counter(), msg.content, getattr(msg, "artifact", None))
                    _prune_tool_cache()

            new_messages.append(msg)
            artifact = getattr(msg, "artifact", None) or {}
            if "citations" in artifact:
                citations.extend(artifact["citations"])
            if call["name"] == "web_search":
                web_results.extend(artifact.get("web_results", []))
                web_search_error = artifact.get("error")
            if call["name"] == "calculate":
                math_result = _parse_math_result(msg.content)

        return {
            "messages": new_messages,
            "citations": _dedupe_citations(citations),
            "web_results": web_results,
            "web_search_error": web_search_error,
            "math_result": math_result,
            "tool_round": state.get("tool_round", 0) + 1,
        }

    return tools_node


def make_finalize_node(llm):
    """Guarantees `answer` is set. Normal case: agent_node already stopped
    (AIMessage with no tool_calls) -- no-op passthrough. Fallback case: the
    tool_round cap was hit while the model still wanted to call tools --
    forces one plain, tool-free generation so the user gets a real answer."""

    def finalize_node(state: AgentState) -> Dict[str, Any]:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return {"tool_round": state.get("tool_round", 0)}  # no-op; every node must write >=1 key

        content = (getattr(last, "content", "") or "").strip()
        if content:
            return {"answer": content}

        nudge = HumanMessage(
            content=(
                "You have reached the tool-call limit for this turn. Based on "
                "everything gathered so far, give your best final answer now "
                "without calling any more tools."
            )
        )
        response = llm.invoke(state["messages"] + [nudge])
        return {"answer": getattr(response, "content", "") or "", "messages": [nudge, response]}

    return finalize_node


# -- citation precision --------------------------------------------------
#
# retrieve_documents' own relevance grading is TOPIC-level ("would this
# passage help answer the query"), which is coarser than what one already-
# written ANSWER actually draws on. This second gate runs once the answer
# is final: given the ANSWER and this turn's document citations (with their
# snippets), ask the LLM which passages actually support a claim the
# answer makes. Same fail-open philosophy as check_relevance.
#
# Prompt is worded around SUPPORT/corroboration, not exact-match
# containment -- an earlier stricter wording ("contains the EXACT
# information") caused the grader to drop genuinely-used corroborating
# evidence whenever a differently-worded passage looked like a closer
# literal match.
_CITATION_PRECISION_PROMPT = """You will be given an ANSWER and a numbered list of PASSAGES. \
Each passage was retrieved from the user's uploaded document(s) and judged topically \
relevant to the user's question -- but the ANSWER may only actually be supported by some \
of them.

The ANSWER may make more than one claim (e.g. it states a figure, then notes where else \
that figure appears, or adds a caveat). For EACH passage, decide whether it supports, \
corroborates, defines, or provides context for AT LEAST ONE claim the ANSWER makes -- i.e. \
would a reader trying to verify the ANSWER find this passage genuinely useful evidence for \
part of it? A passage does NOT need to contain the ANSWER's exact final wording or the \
complete answer by itself to count -- e.g. if the ANSWER says "the ending balance was \
£130.82", a passage that only says "Balance carried forward 130.82" still supports that \
claim. Only exclude a passage if it is about a genuinely different fact, entity, date, or \
transaction than anything the ANSWER actually states.

ANSWER: {answer}

PASSAGES:
{passages}

Respond with ONLY a JSON object and nothing else, in this exact form:
{{"supporting_ids": [<ids of passages that support at least one claim in the ANSWER; [] if none do>]}}
"""

# Matches "page 2", "p. 2", "p2", "pg 2" -- used only as a completeness
# backstop below, never as the primary relevance decision.
_PAGE_MENTION_RE = re.compile(r"\b(?:page|pg\.?|p\.?)\s*(\d+)\b", re.IGNORECASE)


def _explicitly_referenced_pages(answer: str) -> set[int]:
    return {int(m) for m in _PAGE_MENTION_RE.findall(answer)}


def _select_supporting_document_citations(
    answer: str, citations: List[Dict[str, Any]], llm
) -> List[Dict[str, Any]]:
    doc_citations = [c for c in citations if c.get("type") == "document"]
    other_citations = [c for c in citations if c.get("type") != "document"]

    # 0-1 document citations can't be over-broad; no answer text means
    # nothing to check them against.
    if len(doc_citations) <= 1 or not answer.strip():
        return citations

    passages = "\n\n".join(
        f"[{i}] ({c.get('source')}, page {c.get('page')}): {c.get('snippet', '')}"
        for i, c in enumerate(doc_citations)
    )
    prompt = _CITATION_PRECISION_PROMPT.format(answer=answer, passages=passages)

    try:
        response = llm.invoke(prompt)
        content = (getattr(response, "content", None) or str(response)).strip()
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"no JSON object found in citation-precision response: {content!r}")
        parsed = json.loads(content[start : end + 1])
        supporting_ids = {int(i) for i in parsed.get("supporting_ids", [])}
    except Exception:
        logger.warning(
            "Citation-precision grading failed to parse; failing open (keeping all document citations).",
            exc_info=True,
        )
        return citations

    # If the answer explicitly names a page that was actually retrieved
    # this turn, keep it even if the grader dropped it -- the answer
    # itself is a stronger signal than the grader's judgment. Only ever
    # adds a citation back, never removes one, and only for pages
    # actually seen this turn.
    referenced_pages = _explicitly_referenced_pages(answer)
    for i, c in enumerate(doc_citations):
        if i in supporting_ids:
            continue
        if c.get("page") in referenced_pages:
            supporting_ids.add(i)

    kept = [c for i, c in enumerate(doc_citations) if i in supporting_ids]
    if not kept:
        # Fail open: an answer that passed evaluate_node's sufficiency
        # check presumably came from real evidence, so zero citations
        # would be more misleading than the original unnarrowed set.
        kept = doc_citations
    return other_citations + kept


_WEB_CITATION_PRECISION_PROMPT = """You will be given an ANSWER and a numbered list of WEB RESULTS \
(title, domain, text excerpt) returned by a web search for the user's question. Search engines \
return loosely related pages, so many of them may have nothing to do with the ANSWER.

Decide which results genuinely support, corroborate, or give context for at least one claim the \
ANSWER makes. EXCLUDE a result if it is about a different company/entity/product than the one the \
ANSWER is about (e.g. a similarly named but unrelated business), or about a different fact than \
anything the ANSWER states.

ANSWER: {answer}

WEB RESULTS:
{passages}

Respond with ONLY a JSON object and nothing else, in this exact form:
{{"supporting_ids": [<ids of results that support at least one claim in the ANSWER; [] if none do>]}}
"""


def _select_supporting_web_citations(
    answer: str, citations: List[Dict[str, Any]], llm
) -> List[Dict[str, Any]]:
    """Same idea as _select_supporting_document_citations, for web results:
    without this every Exa hit from every query becomes a source chip,
    including unrelated pages. Fails open (keeps everything) on any grader
    problem, so it can only ever reduce noise, never hide real evidence
    because of an outage."""
    web = [c for c in citations if c.get("type") == "web"]
    rest = [c for c in citations if c.get("type") != "web"]
    if len(web) <= 1 or not answer.strip():
        return citations

    passages = "\n\n".join(
        f"[{i}] {c.get('source')} ({c.get('domain') or 'unknown domain'}): {c.get('snippet', '')}"
        for i, c in enumerate(web)
    )
    prompt = _WEB_CITATION_PRECISION_PROMPT.format(answer=answer, passages=passages)
    try:
        response = llm.invoke(prompt)
        content = (getattr(response, "content", None) or str(response)).strip()
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"no JSON object found in web-citation-precision response: {content!r}")
        ids = {int(i) for i in json.loads(content[start : end + 1]).get("supporting_ids", [])}
    except Exception:
        logger.warning("Web-citation precision grading failed; keeping all web citations.", exc_info=True)
        return citations

    kept = [c for i, c in enumerate(web) if i in ids]
    if not kept:
        kept = web  # fail open: answer passed sufficiency, zero sources would mislead more
    logger.info("Web citations narrowed: %d -> %d", len(web), len(kept))
    return rest + kept


# -- answer verification --------------------------------------------------
#
# evaluate_node's sufficiency check only looks at whether the answer TEXT
# looks complete -- it never checks the answer's claims against the
# evidence gathered. This third gate does that, reusing the evidence
# already in state (citation snippets, web_results text, math_result) --
# no new tool, no re-retrieval. Runs only when the answer already looked
# sufficient and there is evidence to check it against.
_ANSWER_VERIFICATION_PROMPT = """You are checking whether an ANSWER is actually supported \
by the EVIDENCE gathered for it. The evidence is split into three kinds; an answer may not \
have all three, only check the kinds that are present:

- DOCUMENT EVIDENCE: passages retrieved from the user's uploaded document(s).
- WEB EVIDENCE: passages returned by a web search.
- CALCULATION RESULT: the output of a calculator tool.

Check specifically:
1. Claim support -- is every factual claim in the ANSWER actually stated by some piece of \
evidence, not merely near it or on the same topic?
2. Numeric consistency -- do any numbers in the ANSWER (amounts, balances, percentages, \
counts) exactly match what the evidence states? A close-but-different number is a failure.
3. Date consistency -- do any dates in the ANSWER exactly match what the evidence states, \
rather than being guessed or approximated? If the ANSWER states a year that appears nowhere \
in the evidence, treat that year as unsupported/invented.
4. Financial field roles -- when evidence includes labeled TRANSACTION rows (Date, \
Description, Money In, Money Out, Amount, Balance After), these are DIFFERENT fields and must \
not be treated as interchangeable: "Balance After" is the running account balance, not a \
payment; "Money In"/credit is not the same as "Money Out"/debit. A claim about "the largest \
outgoing payment" is only supported by Money Out / debit values, never by a balance figure, \
even one that looks numerically similar.
5. If the ANSWER makes claims about current/external/real-world facts, are they backed by \
WEB EVIDENCE specifically (not just document evidence, which cannot speak to current facts)?
6. If the ANSWER makes claims about the uploaded document's own content, are they backed by \
DOCUMENT EVIDENCE specifically (not just web evidence)?

ANSWER: {answer}

{evidence_block}

Respond with ONLY a JSON object and nothing else, in this exact form:
{{"supported": true or false, "reason": "<one short sentence, empty string if supported>"}}
"""


def _format_evidence_block(
    citations: List[Dict[str, Any]], web_results: List[Dict[str, Any]], math_result: Any
) -> str:
    blocks = []

    doc_citations = [c for c in citations if c.get("type") == "document"]
    if doc_citations:
        doc_lines = []
        for c in doc_citations:
            doc_lines.append(f"- ({c.get('source')}, page {c.get('page')}): {c.get('snippet', '')}")
            # Explicit field-labeled rows so the verifier doesn't have to
            # infer which number is the payment vs. the running balance
            # from raw table text (the direct fix for balance-vs-payment
            # confusion).
            for txn in c.get("transactions") or []:
                doc_lines.append(
                    "  TRANSACTION -- "
                    f"Date: {txn.get('date') or 'unknown'} | "
                    f"Description: {txn.get('description') or ''} | "
                    f"Money In: {_fmt_amount(txn.get('credit'))} | "
                    f"Money Out: {_fmt_amount(txn.get('debit'))} | "
                    f"Amount: {_fmt_amount(txn.get('amount'))} | "
                    f"Balance After: {_fmt_amount(txn.get('balance'))}"
                )
            for key, value in (c.get("key_values") or {}).items():
                doc_lines.append(f"  FIELD -- {key}: {value}")
        blocks.append("DOCUMENT EVIDENCE:\n" + "\n".join(doc_lines))

    if web_results:
        web_lines = "\n".join(
            f"- {r.get('title', 'web result')} (published: {r.get('published_date') or 'no date reported'}): "
            f"{(r.get('text') or '')[:500]}"
            for r in web_results
        )
        blocks.append(f"WEB EVIDENCE:\n{web_lines}")

    if math_result:
        # math_tool returns a bare number ("28934915"), but the
        # deterministic numeric check below only matches £/$/€/₹-prefixed
        # tokens -- so also emit a £-formatted variant it can match against.
        result = math_result.get("result")
        money_variant = f" (£{_fmt_amount(result)})" if isinstance(result, (int, float)) else ""
        blocks.append(f"CALCULATION RESULT:\n- {math_result.get('result_str', '')}{money_variant}")

    return "\n\n".join(blocks)


def _fmt_amount(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


# -- deterministic numeric/date pre-check ---------------------------------
#
# Exact string matching is strictly more reliable than an LLM judgment for
# currency amounts, percentages, and dates, so this runs first and can
# fail the answer deterministically before the LLM call. It only ever
# produces a "failed" verdict or none at all -- it can't judge non-numeric
# claim support, so the LLM grader still runs afterward for that.
_MONEY_RE = re.compile(r"[£$€₹]\s*-?\d[\d,]*(?:\.\d+)?")
_PERCENT_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?\s*%")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)


def _normalize_numeric_token(token: str) -> str:
    cleaned = re.sub(r"[\s,]", "", token).lower()
    m = re.fullmatch(r"([£$€₹]?)(-?\d+(?:\.\d+)?)(%?)", cleaned)
    if m:  # compare numerically: 300 == 300.00
        return f"{m.group(1)}{format(Decimal(m.group(2)).normalize(), 'f')}{m.group(3)}"
    return cleaned


def _extract_numeric_facts(text: str) -> List[str]:
    facts: List[str] = []
    for pattern in (_MONEY_RE, _PERCENT_RE, _DATE_RE):
        facts.extend(pattern.findall(text))
    return [_normalize_numeric_token(f) for f in facts]


def _deterministic_numeric_check(answer: str, evidence_block: str) -> Optional[str]:
    """Returns a failure reason if the ANSWER states a currency amount,
    percentage, or date that doesn't appear verbatim in the evidence,
    else None (passed, or nothing to check -- either way not itself a
    full "verified" confirmation)."""
    answer_facts = _extract_numeric_facts(answer)
    if not answer_facts:
        return None

    evidence_facts = set(_extract_numeric_facts(evidence_block))
    mismatches = [f for f in answer_facts if f not in evidence_facts]
    if not mismatches:
        return None

    logger.info(
        "Deterministic check mismatches=%s; evidence contains %d numeric fact(s): %s",
        sorted(set(mismatches)), len(evidence_facts), sorted(evidence_facts)[:40],
    )
    return (
        "Deterministic check failed: the answer states "
        f"{', '.join(sorted(set(mismatches)))}, which doesn't exactly match any "
        "amount/percentage/date in the evidence."
    )


def _verify_answer_support(
    answer: str,
    citations: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
    math_result: Any,
    llm,
) -> tuple[str, str]:
    """Returns (verification_status, reason):
      - "verified"    ran successfully, answer supported by evidence.
      - "failed"      ran successfully, found a factual/grounding problem
                       (reason explains what).
      - "unavailable" the verifier itself couldn't complete (LLM/API
                       error, malformed response) -- distinct from
                       "verified" or "failed": nothing was actually
                       checked, so no result to trust either way.

    Caller decides what each status means for retries; this only reports
    what happened."""
    evidence_block = _format_evidence_block(citations, web_results, math_result)
    if not evidence_block:
        # Defensive fallback -- caller only calls this when has_evidence
        # is True.
        return "verified", ""

    deterministic_reason = _deterministic_numeric_check(answer, evidence_block)
    if deterministic_reason:
        # A hard numeric/date mismatch is conclusive -- skip the LLM call
        # rather than let it potentially overrule an exact-match failure.
        return "failed", deterministic_reason

    prompt = _ANSWER_VERIFICATION_PROMPT.format(answer=answer, evidence_block=evidence_block)
    try:
        response = llm.invoke(prompt)
        content = (getattr(response, "content", None) or str(response)).strip()
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"no JSON object found in verification response: {content!r}")
        parsed = json.loads(content[start : end + 1])
        supported = bool(parsed.get("supported", True))
        reason = str(parsed.get("reason", "") or "")
    except Exception:
        # A verifier outage/parsing bug must not be reported as a pass --
        # "unavailable" is its own status, never merged into "verified".
        logger.warning(
            "Answer verification could not be completed (LLM/API error or unparseable response); "
            "reporting as unavailable rather than verified.",
            exc_info=True,
        )
        return "unavailable", ""

    return ("verified" if supported else "failed"), reason


def make_evaluate_node(llm):
    def evaluate_node(state: AgentState) -> Dict[str, Any]:
        """Sufficiency check: empty/very short answer, or one that says the
        context didn't have the answer, is treated as insufficient
        (triggers a retry up to max_iterations). Also narrows document
        citations to the ones that actually support the final answer
        (see _select_supporting_document_citations), then verifies the
        answer's claims against the gathered evidence
        (see _verify_answer_support).

        verification_status handling:
          - "verified"    -> is_sufficient unchanged.
          - "failed"      -> genuine grounding problem; is_sufficient=False,
                             feeding the existing retry loop in graph.py.
          - "unavailable" -> verification infrastructure failed;
                             is_sufficient is left as-is (not forced False)
                             so an outage can't masquerade as a factual
                             failure -- a warning is attached instead."""
        answer = _strip_inline_citation_markers(state.get("answer") or "")
        warnings = []

        if state.get("web_search_error"):
            warnings.append(
                f"Web search was unavailable ({state['web_search_error']}); this answer may not reflect current information."
            )

        insufficient_markers = ("cannot be found", "don't know", "no information", "not available")
        is_sufficient = bool(answer) and not any(m in answer.lower() for m in insufficient_markers)
        if not is_sufficient:
            warnings.append("The answer may be incomplete based on the available context.")

        citations = _select_supporting_document_citations(answer, state.get("citations", []), llm)
        citations = _select_supporting_web_citations(answer, citations, llm)

        # Verification checks against the FULL citation set gathered this
        # turn, not the precision-narrowed one above -- narrowing is a
        # display concern, verification is a grounding concern. Checking
        # the narrowed set would let a precision-filter mistake make the
        # verifier blind to evidence that was actually retrieved.
        all_citations = state.get("citations", [])
        has_evidence = bool(all_citations) or bool(state.get("web_results")) or bool(state.get("math_result"))
        verification_status = None
        if is_sufficient and has_evidence:
            verification_status, reason = _verify_answer_support(
                answer, all_citations, state.get("web_results", []), state.get("math_result"), llm
            )
            if verification_status == "failed":
                is_sufficient = False
                warnings.append(f"Verification failed: {reason}")
            elif verification_status == "unavailable":
                warnings.append(
                    "Answer verification was unavailable; the answer could not be fully verified."
                )

        return {
            "answer": answer,
            "is_sufficient": is_sufficient,
            "warnings": warnings,
            "citations": citations,
            "verification_status": verification_status,
        }

    return evaluate_node


# -- long-term memory extraction ------------------------------------------
#
# Deliberately narrow: only durable facts ABOUT THE USER (identity, role,
# stated preferences, recurring context they'd want remembered next
# session) -- not document content (that already lives in FAISS/citations)
# and not one-off details of this single question (e.g. a specific
# balance asked about once isn't "memorable", "always show amounts in
# GBP" is).
_MEMORY_EXTRACTION_PROMPT = """Look at this single exchange between a user and an assistant. \
Extract only durable facts ABOUT THE USER that would be genuinely useful to remember in a \
FUTURE, unrelated conversation -- their name, role/company, stated preferences (tone, \
format, units, language), recurring goals, or context they explicitly shared about \
themselves. Do NOT extract facts about an uploaded document's contents, one-off figures \
the user merely asked about, or anything already obvious from a single question. If \
nothing durable and user-specific was shared, return an empty list.

USER: {user_message}
ASSISTANT: {assistant_message}

Respond with ONLY a JSON object and nothing else, in this exact form:
{{"facts": ["<short, self-contained fact 1>", "..."]}}
"""


def make_save_memory_node(llm, memory_store):
    """Runs once per turn, right before the graph ends (see graph.py's
    should_retry -> "save_memory" -> END). Extracts and stores any
    durable, user-specific facts from this turn's final exchange.

    Best-effort and silent on failure -- a broken extraction or an
    unreachable memory store must never surface as a chat error, since
    by this point the user's actual answer has already been produced."""

    def save_memory_node(state: AgentState) -> Dict[str, Any]:
        user_id = state.get("user_id")
        answer = state.get("answer") or ""
        if not user_id or not answer or memory_store is None:
            return {"tool_round": state.get("tool_round", 0)}  # no-op; every node must write >=1 key

        try:
            prompt = _MEMORY_EXTRACTION_PROMPT.format(
                user_message=state["question"], assistant_message=answer
            )
            response = llm.invoke(prompt)
            content = (getattr(response, "content", None) or str(response)).strip()
            start, end = content.find("{"), content.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"no JSON object found in memory-extraction response: {content!r}")
            facts = json.loads(content[start : end + 1]).get("facts", []) or []
            for fact in facts:
                if isinstance(fact, str) and fact.strip():
                    memory_store.add(user_id, fact.strip())
        except Exception:
            logger.warning("Long-term memory extraction failed for this turn; continuing.", exc_info=True)

        return {"tool_round": state.get("tool_round", 0)}  # no-op; every node must write >=1 key

    return save_memory_node


# -- follow-up question suggestions ----------------------------------------
#
# Purely a UX nicety (clickable "you might also ask" chips) -- kept as its
# own small, cheap LLM call rather than folded into the main answer
# generation, so a parsing failure here can never affect the answer
# itself, only whether chips are shown.
_SUGGESTED_QUESTIONS_PROMPT = """Given this question and answer, suggest up to 3 short, \
natural follow-up questions the user might reasonably want to ask next. Each should be a \
question the SAME assistant (with the same tools -- the user's document(s), web search, a \
calculator) could actually answer, not something unrelated. Keep each under 12 words, plain \
text, no numbering or leading dashes. If nothing sensible follows from this exchange, \
return an empty list.

QUESTION: {question}
ANSWER: {answer}

Respond with ONLY a JSON object and nothing else, in this exact form:
{{"questions": ["<question 1>", "<question 2>", "<question 3>"]}}
"""


def make_suggest_questions_node(llm):
    def suggest_questions_node(state: AgentState) -> Dict[str, Any]:
        answer = state.get("answer") or ""
        if not answer:
            return {"suggested_questions": []}
        return {"suggested_questions": generate_followup_questions(llm, state["question"], answer)}

    return suggest_questions_node


def generate_followup_questions(llm, question: str, answer: str) -> List[str]:
    """Shared by suggest_questions_node above (agentic route) and
    app.py's plain /api/chat route, so both surfaces get the same
    follow-up-chip behavior from one implementation."""
    if not answer:
        return []
    try:
        prompt = _SUGGESTED_QUESTIONS_PROMPT.format(question=question, answer=answer)
        response = llm.invoke(prompt)
        content = (getattr(response, "content", None) or str(response)).strip()
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"no JSON object found in suggested-questions response: {content!r}")
        questions = json.loads(content[start : end + 1]).get("questions", []) or []
        return [q.strip() for q in questions if isinstance(q, str) and q.strip()][:3]
    except Exception:
        logger.warning("Follow-up question suggestion failed for this turn; continuing.", exc_info=True)
        return []