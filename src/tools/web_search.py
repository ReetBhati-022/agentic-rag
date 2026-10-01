"""
web_search.py
-------------
Wraps the Exa API (exa-py). Never raises -- every failure mode (missing
EXA_API_KEY, exa-py not installed, network/API error) is caught and
returned as {"success": False, "error": ...} instead, so a failed call
degrades to document-only answering with a visible warning rather than
crashing the agent loop (see src/agent/nodes.py's tools_node /
synthesize_node, which read `success`/`error` explicitly).

Logged explicitly at INFO here (rather than relying on the logging
framework the underlying HTTP client happens to use) because exa-py's
main sync client calls out via `requests`, not `httpx` -- this app's
logging.basicConfig only surfaces INFO from `httpx` (which is why
Ollama/HuggingFace calls appear in the console but a real, successful
Exa call otherwise leaves no trace at all, making "did web_search even
run" impossible to answer from the console alone.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _domain_of(url: str | None) -> str:
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def run_web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        logger.info("web_search SKIPPED for query=%r -- EXA_API_KEY is not set.", query)
        return {"success": False, "error": "EXA_API_KEY is not set.", "results": []}

    try:
        from exa_py import Exa
    except Exception as exc:
        logger.info("web_search SKIPPED for query=%r -- exa-py is not installed: %s", query, exc)
        return {"success": False, "error": f"exa-py is not installed: {exc}", "results": []}

    try:
        client = Exa(api_key=api_key)
        response = client.search_and_contents(
            query, num_results=num_results, text={"max_characters": 1000}
        )
        results = []
        for r in getattr(response, "results", None) or []:
            url = getattr(r, "url", None)
            results.append(
                {
                    "title": getattr(r, "title", None) or url or "web result",
                    "url": url,
                    "text": getattr(r, "text", "") or "",
                    # None (not "") when Exa doesn't return one -- the agent must be able to
                    # tell "no date available" apart from an empty-string date, and must never
                    # invent/approximate a date when this is None (see tools.py / graph.py).
                    "published_date": getattr(r, "published_date", None),
                    "domain": _domain_of(url),
                }
            )
        logger.info("web_search OK for query=%r -- %d result(s) returned.", query, len(results))
        return {"success": True, "results": results, "error": None}
    except Exception as exc:
        logger.info("web_search FAILED for query=%r -- %s", query, exc)
        return {"success": False, "error": str(exc), "results": []}