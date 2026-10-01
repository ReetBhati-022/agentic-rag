from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

from src.agent.nodes import (
    _MAX_TOOL_ROUNDS_PER_TURN,
    make_agent_node,
    make_evaluate_node,
    make_finalize_node,
    make_recall_memory_node,
    make_save_memory_node,
    make_start_turn_node,
    make_suggest_questions_node,
    make_tools_node,
)
from src.agent.state import AgentState
from src.agent.tools import make_tools
from src.retrieval.retriever import Retriever
from src.services.memory_store import MemoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Postgres checkpointer (module-level singleton)
# ---------------------------------------------------------------------------
# One ConnectionPool + PostgresSaver for the whole process, built lazily on
# first use and reused across every run_agent() call/thread_id -- opening a
# fresh Postgres connection pool per request would be wasteful and would
# also defeat the point of persistence (you'd want one long-lived pool, not
# one per graph invocation). `.setup()` (creates/migrates the checkpoint
# tables) only needs to run once per DB, so it's guarded by _checkpointer
# already being set.
_checkpointer = None
_checkpointer_pool = None

# Guards the one-time dev diagram render in _build_graph() -- see there.
_diagram_saved = False

# Timestamp of the last failed connection attempt, so a Postgres outage
# doesn't cost every single run_agent() call the full ~30s pool connect
# timeout while retrying a connection that's going to fail again anyway.
# Set to a real perf_counter() value on failure; retried only after
# _CHECKPOINTER_RETRY_COOLDOWN_SECONDS have passed.
_checkpointer_last_failure: Optional[float] = None
_CHECKPOINTER_RETRY_COOLDOWN_SECONDS = 60


def _get_checkpointer(config: Dict[str, Any]):
    """Returns the process-wide PostgresSaver, or None if checkpointing is
    disabled in config.yaml or its connection string isn't configured.
    Falls back to None (i.e. an uncheckpointed, in-memory-only graph, same
    as this app's behavior before this was added) rather than raising, so
    a Postgres outage doesn't take down chat entirely -- it just loses
    cross-request persistence for that run."""
    global _checkpointer, _checkpointer_pool, _checkpointer_last_failure

    if _checkpointer is not None:
        return _checkpointer

    if _checkpointer_last_failure is not None:
        elapsed = time.perf_counter() - _checkpointer_last_failure
        if elapsed < _CHECKPOINTER_RETRY_COOLDOWN_SECONDS:
            # Still in cooldown from a recent failure -- skip straight to
            # "no persistence this turn" instead of paying the full ~30s
            # connection timeout again on a DB that was just unreachable.
            return None

    checkpoint_cfg = config.get("agent", {}).get("checkpoint", {})
    if not checkpoint_cfg.get("enabled", False):
        return None

    conn_string = os.environ.get(checkpoint_cfg.get("conn_string_env", "POSTGRES_URI"))
    if not conn_string:
        logger.warning(
            "agent.checkpoint.enabled is true but %s is not set in the environment -- "
            "running without persistence.",
            checkpoint_cfg.get("conn_string_env", "POSTGRES_URI"),
        )
        return None

    try:
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver

        # autocommit + dict_row are what PostgresSaver expects of connections
        # it's handed; see langgraph.checkpoint.postgres's own from_conn_string
        # for the same kwargs.
        _checkpointer_pool = ConnectionPool(
            conninfo=conn_string,
            max_size=checkpoint_cfg.get("max_pool_size", 10),
            kwargs={"autocommit": True, "row_factory": __import__("psycopg").rows.dict_row},
            open=True,
        )
        saver = PostgresSaver(_checkpointer_pool)
        saver.setup()  # no-op after the first successful run against this DB
        _checkpointer = saver
        _checkpointer_last_failure = None
        logger.info("Postgres checkpointer connected and ready.")
    except Exception:
        logger.exception(
            "Failed to initialize Postgres checkpointer -- running without persistence "
            "for at least the next %ss.",
            _CHECKPOINTER_RETRY_COOLDOWN_SECONDS,
        )
        _checkpointer_pool = None
        _checkpointer = None
        _checkpointer_last_failure = time.perf_counter()

    return _checkpointer

# Default system prompt for the AGENTIC route only. Deliberately NOT the
# same as config.yaml's `rag.system_prompt` -- that one was written for
# the tool-less plain /api/chat route ("answer using ONLY the provided
# context, never use outside knowledge") and, reused here verbatim,
# actively discouraged this agent from ever falling back to web_search
# after retrieve_documents came back empty (e.g. "What is Capsitech's
# registered address?" against a Barclays statement -- Capsitech isn't
# a document question, it's a web_search question). This prompt gives
# the agent an explicit tool-selection policy instead. Override it via
# `agent.system_prompt` in config.yaml if needed; falls back to this
# default when that key isn't set.
_DEFAULT_AGENT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using the tools \
available to you: retrieve_documents (the user's uploaded document(s)), web_search \
(current/external/public information), and calculate (arithmetic).

Decide which tool(s) you need from the user's question itself, before calling anything \
-- do not default to retrieve_documents just because a document happens to be uploaded.

Tool-selection policy:
- Use retrieve_documents when the answer should come from the uploaded document(s) \
specifically (its statements, forms, transactions, figures, etc).
- Use web_search when the question asks for current, latest, recent, external, public, \
or general real-world information about a subject -- this is true regardless of whether \
a document is uploaded. Call web_search directly for these; do not call \
retrieve_documents first "just in case."
- Exception: if the question refers to its subject vaguely ("this bank", "this company", \
"the organization", "the document", etc.) rather than naming it, you must resolve that \
reference to the specific entity/name BEFORE calling web_search -- a vague phrase is not \
a usable web_search query. You may already know the entity from the loaded-document hint \
at the start of this conversation or from earlier messages; if you don't know it precisely \
enough (e.g. you only know a filename, not the exact registered name), call \
retrieve_documents first to find it in the document content, then call web_search with the \
resolved, self-contained query (e.g. "latest news Barclays Bank UK PLC", not "latest news \
about this bank").
- Use calculate for any arithmetic -- including finding the difference, sum, or comparison \
between two or more numbers you already have (e.g. two balances, two amounts from earlier \
in this conversation). Even when both numbers are already sitting in front of you (in a \
tool result or in your own prior message), you must still call calculate with them rather \
than computing the result yourself -- mental arithmetic on multi-decimal money figures is \
where you're most likely to make a silent sign or rounding error. First retrieve or \
otherwise obtain the numeric values you need, then call calculate with the resulting \
expression.
- Use multiple tools in sequence when one tool's result is needed to complete another \
step (e.g. retrieve a document value, then calculate with it; or retrieve document \
context, then web_search for related current information).
- If the user refers to "this bank", "this company", "the organization", etc., resolve \
that reference to the actual name using the loaded-document hint, conversation, or a \
retrieve_documents call (see the exception above) before constructing a web_search query.
- Recovery case: if you called retrieve_documents and it reports NO DOCUMENT EVIDENCE, \
re-evaluate the question. If it's asking whether the document(s) mention something, you \
can answer now. If it's actually asking for real-world/external/current information \
about the subject, call web_search before answering -- do not conclude the information \
is unavailable without checking it.

Every factual claim in your answer must be traceable to a tool result -- never guess or \
fabricate a fact no tool returned. State numbers, dates, and names exactly as the tools \
returned them. If, after using the appropriate tool(s), nothing relevant was found, say \
so clearly and briefly instead of guessing.

Date accuracy: only state a publication date if a tool explicitly returned one for that \
result. Never approximate, round, or infer a date (no "(approx.)", no guessing a year from \
surrounding context) -- if a web_search result has no date, say its date wasn't reported \
rather than estimating one.

Source quality: web_search marks results from well-known/official outlets as [known \
outlet] and lists them first; the rest are from outlets not on that short list. When \
summarizing "latest"/current news, lean on the [known outlet] results, and if you rely on \
an unmarked source, note that its outlet isn't a well-known one rather than presenting it \
with the same confidence as an official/major-outlet source.

Facts vs. your own analysis: state only what a source actually says as fact ("X reported \
Y"). If you add interpretation, inference, or "what this means" commentary beyond what a \
source states, clearly mark it as your own reading (e.g. "this could suggest...") rather \
than presenting it as if the source said it.

Citation scope for mixed-tool answers: if you called retrieve_documents only to resolve a \
vague reference (e.g. finding the entity's exact name) and the document content itself \
does not support the substance of your final answer, say so plainly if asked, and do not \
imply the document is evidence for claims that actually came from web_search.

Formatting: the application already shows the user which document/page or web source each \
answer draws on as clickable source chips below your answer -- this is built from the tool \
results directly, not from anything in your text. Do NOT add your own inline citation \
markers, footnote-style brackets, reference numbers, or source tags of any kind (e.g. no \
[1], no (Source: ...), no 【...†...】-style annotations) -- write the answer as plain prose \
with no citation syntax at all.

Write like you're actually talking to the person, not filing a report. Default to a couple \
of plain sentences, the way you'd say it out loud. Do NOT reach for bold text, numbered \
lists, or bullet points as your default style -- use them only when the answer is genuinely \
a list of several distinct items (e.g. "list all the direct debits") or the person explicitly \
asks for a list/table/breakdown. Do NOT quote raw lines out of the document with ">" \
blockquote syntax; just state the fact in your own words. A short, single-figure answer like \
a balance or a date should usually be one sentence, not a heading plus a quote plus a \
restatement."""


def _build_graph(retriever: Retriever, llm, config: Dict[str, Any], memory_store: Optional[MemoryStore]):
    system_prompt = config.get("agent", {}).get("system_prompt") or _DEFAULT_AGENT_SYSTEM_PROMPT
    num_web_results = config.get("agent", {}).get("web_search", {}).get("num_results", 5)
    max_iterations = config.get("agent", {}).get("max_iterations", 2)

    tools = make_tools(retriever, llm, num_web_results=num_web_results)

    recall_memory_node = make_recall_memory_node(memory_store)
    start_turn_node = make_start_turn_node(system_prompt)
    agent_node = make_agent_node(llm, tools)
    tools_node = make_tools_node(tools, index_version=lambda: retriever.store.version)
    finalize_node = make_finalize_node(llm)
    evaluate_node = make_evaluate_node(llm)
    save_memory_node = make_save_memory_node(llm, memory_store)
    suggest_questions_node = make_suggest_questions_node(llm)

    def should_continue_tools(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state.get("tool_round", 0) < _MAX_TOOL_ROUNDS_PER_TURN:
            return "tools"
        return "finalize"

    def should_retry(state: AgentState) -> str:
        # An internal retry (revised plan, same question) skips
        # save_memory/suggest_questions and goes straight back to
        # start_turn -- there's nothing new and final to work from yet.
        # Both only run once the turn is actually about to end.
        if state.get("is_sufficient") or state.get("iteration", 0) >= max_iterations:
            return "suggest_questions"
        return "start_turn"

    graph = StateGraph(AgentState)
    graph.add_node("recall_memory", recall_memory_node)
    graph.add_node("start_turn", start_turn_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("save_memory", save_memory_node)
    graph.add_node("suggest_questions", suggest_questions_node)

    graph.set_entry_point("recall_memory")
    graph.add_edge("recall_memory", "start_turn")
    graph.add_edge("start_turn", "agent")
    graph.add_conditional_edges("agent", should_continue_tools, {"tools": "tools", "finalize": "finalize"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", "evaluate")
    graph.add_conditional_edges(
        "evaluate", should_retry, {"start_turn": "start_turn", "suggest_questions": "suggest_questions"}
    )
    graph.add_edge("suggest_questions", "save_memory")
    graph.add_edge("save_memory", END)

    checkpointer = _get_checkpointer(config)
    compiled = graph.compile(checkpointer=checkpointer)

    # Dev-only visualization. draw_mermaid_png() (with no custom renderer
    # configured) calls out to the hosted mermaid.ink API over the network
    # -- _build_graph() runs on EVERY run_agent() call, so without this
    # guard this was firing that network round-trip on every single
    # question, adding real latency for a diagram nobody was looking at
    # after the first run. Now draws at most once per process lifetime.
    global _diagram_saved
    if not _diagram_saved:
        _diagram_saved = True  # set first so a failed attempt doesn't retry every call either
        try:
            compiled.get_graph().draw_mermaid_png(output_file_path="graph.png")
            logger.info("Graph diagram saved as graph.png")
        except Exception:
            logger.debug("Graph PNG render failed (non-fatal, dev-only diagnostic).", exc_info=True)

    return compiled


def run_agent(
    question: str,
    provider: str,
    config: Dict[str, Any],
    retriever: Retriever,
    llm,
    indexed_files: List[str] | None = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    memory_store: Optional[MemoryStore] = None,
) -> Dict[str, Any]:
    app = _build_graph(retriever, llm, config, memory_store)
    initial_state: AgentState = {
        "question": question,
        "provider": provider,
        "iteration": 0,
        "indexed_files": indexed_files or [],
        "user_id": user_id or "",
    }

    # LangGraph's own hard stop (default recursion_limit=25) counts every
    # node execution as a step -- it is NOT the same thing as this app's
    # own max_iterations/_MAX_TOOL_ROUNDS_PER_TURN bounds, and by default
    # is smaller than what those bounds can legitimately need. Worst case,
    # one outer pass is: start_turn(1) + up to _MAX_TOOL_ROUNDS_PER_TURN
    # (agent, tools) pairs + finalize(1) + evaluate(1); that repeats up to
    # max_iterations times. Without raising recursion_limit to match,
    # LangGraph can abort a run ("Recursion limit of 25 reached") before
    # this graph's OWN stop conditions (is_sufficient / max_iterations)
    # ever get a chance to end it on purpose -- e.g. a genuinely tricky
    # question that uses several tool-calling rounds. +10 is just a
    # buffer for the one-off nodes (start_turn/finalize/evaluate rounding).
    max_iterations = config.get("agent", {}).get("max_iterations", 2)
    steps_per_pass = 2 + 2 * _MAX_TOOL_ROUNDS_PER_TURN + 2
    # +3 for recall_memory, suggest_questions and save_memory, which each
    # run exactly once per run_agent() call (not once per retry pass --
    # see should_retry).
    recursion_limit = max_iterations * steps_per_pass + 10 + 3

    run_config: Dict[str, Any] = {"recursion_limit": recursion_limit}
    if thread_id:
        # Only meaningful when _get_checkpointer() returned a real saver --
        # if checkpointing is disabled/unavailable, the graph was compiled
        # without a checkpointer and LangGraph ignores thread_id entirely
        # (each invoke() is stateless, same as before this was added).
        run_config["configurable"] = {"thread_id": thread_id}

    final_state = app.invoke(initial_state, config=run_config)
    _cits = final_state.get("citations", []) or []
    logger.info(
        "run_agent done: %d citation(s) (%d document, %d web), verification=%s",
        len(_cits),
        sum(1 for c in _cits if c.get("type") != "web"),
        sum(1 for c in _cits if c.get("type") == "web"),
        final_state.get("verification_status"),
    )

    return {
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "warnings": final_state.get("warnings", []),
        # Issue 2B: "verified" / "failed" / "unavailable" / None (no
        # verification check applied this pass) -- surfaced alongside the
        # answer so callers/UI can distinguish "checked and passed" from
        # "not actually checked", per _verify_answer_support in nodes.py.
        "verification_status": final_state.get("verification_status"),
        "suggested_questions": final_state.get("suggested_questions", []),
    }