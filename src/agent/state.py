"""
state.py
--------
Shared state passed between nodes in the agent graph (src/agent/graph.py).
Mirrors config.yaml's agent.* section: max_iterations bounds how many
outer start_turn -> agent -> tools* -> finalize -> evaluate passes can
run (same meaning as before Phase 1: "1 initial attempt + N retries").

PHASE 1 CHANGE: tool selection is no longer a fixed, pre-computed plan
(the old `tool_plan` regex heuristic). The LLM now drives tool use
itself via llm.bind_tools([...]) / response.tool_calls, in a
model -> tool_calls -> tool execution -> tool results -> model loop
(see src/agent/nodes.py, src/agent/tools.py). `tool_plan` and
`retrieved_context` are removed since nothing reads them anymore --
retrieved/web content now flows to the LLM as ToolMessages inside
`messages`, and citations/web_search_error/math_result are populated
by the tool-execution node from each tool's structured `artifact`
(see src/agent/tools.py's response_format="content_and_artifact").
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    question: str
    provider: str
    indexed_files: List[str]
    # Long-term memory (cross-session, cross-thread -- see
    # src/services/memory_store.py). `user_id` is stable across "New
    # chat"/thread_id resets; `memories` is the list of recalled fact
    # strings for THIS question, fetched once by recall_memory_node at
    # the very start of a turn and folded into the question by
    # start_turn_node (same pattern as the indexed_files hint below).
    user_id: str
    memories: List[str]
    suggested_questions: List[str]
    messages: Annotated[List[BaseMessage], add_messages]
    tool_round: int
    web_results: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    web_search_error: Optional[str] 
    math_result: Optional[Dict[str, Any]]
    answer: str
    is_sufficient: bool
    warnings: List[str]
    verification_status: Optional[str]
    iteration: int
    turn_boundary_ids: List[str]