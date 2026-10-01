"""
memory_store.py
----------------
Cross-session LONG-TERM memory, distinct from the per-thread checkpointer
in src/agent/graph.py:

  - The Postgres checkpointer remembers the messages of ONE conversation
    (one thread_id). A "New chat" / api_clear_chat regenerates thread_id,
    so that history is gone for the agent even though it's still sitting
    in the checkpoint tables.
  - MemoryStore remembers small, durable facts *about the user* (name,
    role, preferences, recurring context, etc.) keyed by a stable
    user_id that survives "New chat", index clears, and process
    restarts -- e.g. "prefers short answers", "works at Acme Corp".

Storage: a single `long_term_memory` table, one row per fact, with the
fact's embedding stored as JSON (float list) rather than a pgvector
column -- this app's own embedding model (BAAI/bge-small-en-v1.5 by
default) is already loaded in-process for FAISS, so similarity is just
computed in Python over each user's (typically small: tens-to-low-
hundreds) rows rather than pushing the search into Postgres. That
avoids requiring the pgvector extension to be installed/enabled on
whatever POSTGRES_URI points at.

Same failure philosophy as the checkpointer: if Postgres is
unreachable or agent.memory.enabled is false, every method becomes a
no-op (add() silently drops, search()/all() return []) instead of
raising -- long-term memory is a nice-to-have, not something that
should be able to take chat down.
"""
from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS long_term_memory (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    memory_text TEXT NOT NULL,
    embedding JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    use_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS long_term_memory_user_id_idx ON long_term_memory (user_id);
"""


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class MemoryStore:
    """Lazily connects on first real use (not __init__) so app boot never
    blocks on/fails because of Postgres -- same reasoning as
    graph.py's _get_checkpointer."""

    def __init__(self, config: Dict[str, Any], embeddings):
        mem_cfg = (config.get("agent", {}) or {}).get("memory", {}) or {}
        self.enabled: bool = bool(mem_cfg.get("enabled", False))
        self.conn_string_env: str = mem_cfg.get("conn_string_env", "POSTGRES_URI")
        self.max_pool_size: int = int(mem_cfg.get("max_pool_size", 5))
        self.max_recall: int = int(mem_cfg.get("max_recall", 5))
        self.recall_threshold: float = float(mem_cfg.get("recall_threshold", 0.30))
        self.dedup_threshold: float = float(mem_cfg.get("dedup_threshold", 0.92))

        self.embeddings = embeddings

        self._pool = None
        self._ready = False
        self._last_failure: Optional[float] = None
        self._retry_cooldown_seconds = 60

    # -- connection -------------------------------------------------------
    def _get_pool(self):
        import os

        if self._pool is not None:
            return self._pool
        if not self.enabled:
            return None
        if self._last_failure is not None and (
            time.perf_counter() - self._last_failure < self._retry_cooldown_seconds
        ):
            return None

        conn_string = os.environ.get(self.conn_string_env)
        if not conn_string:
            logger.warning(
                "agent.memory.enabled is true but %s is not set -- long-term "
                "memory is disabled for this process.",
                self.conn_string_env,
            )
            return None

        try:
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(
                conninfo=conn_string,
                max_size=self.max_pool_size,
                kwargs={"autocommit": True},
                timeout=3.0,  # default is 30s; fail fast when Postgres is down
                open=True,
            )
            with pool.connection() as conn:
                conn.execute(_CREATE_TABLE_SQL)
            self._pool = pool
            self._ready = True
            self._last_failure = None
            logger.info("Long-term memory store connected and ready.")
        except Exception:
            logger.exception(
                "Failed to initialize long-term memory store -- running "
                "without it for at least the next %ss.",
                self._retry_cooldown_seconds,
            )
            self._pool = None
            self._ready = False
            self._last_failure = time.perf_counter()

        return self._pool

    def warmup(self) -> None:
        """Connects at startup instead of on the first question."""
        self._get_pool()

    def _embed(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    # -- public API ---------------------------------------------------------
    def search(self, user_id: str, query: str, k: Optional[int] = None) -> List[str]:
        """Top-k memory texts for user_id most similar to query, above
        recall_threshold. Returns [] if memory is disabled/unavailable or
        the user has no memories yet -- never raises."""
        pool = self._get_pool()
        if pool is None or not query.strip():
            return []

        k = k or self.max_recall
        try:
            query_vec = self._embed(query)
            rows: List[tuple] = []
            with pool.connection() as conn:
                cur = conn.execute(
                    "SELECT id, memory_text, embedding FROM long_term_memory WHERE user_id = %s",
                    (user_id,),
                )
                rows = cur.fetchall()

            scored = []
            for row_id, text, embedding_json in rows:
                vec = embedding_json if isinstance(embedding_json, list) else json.loads(embedding_json)
                score = _cosine(query_vec, vec)
                if score >= self.recall_threshold:
                    scored.append((score, row_id, text))
            scored.sort(key=lambda t: t[0], reverse=True)
            top = scored[:k]

            if top:
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.executemany(
                            "UPDATE long_term_memory SET last_used_at = now(), "
                            "use_count = use_count + 1 WHERE id = %s",
                            [(row_id,) for _, row_id, _ in top],
                        )

            return [text for _, _, text in top]
        except Exception:
            logger.exception("Long-term memory search failed; continuing without recalled memories.")
            return []

    def add(self, user_id: str, memory_text: str) -> bool:
        """Stores a new fact for user_id unless a near-duplicate (cosine
        similarity >= dedup_threshold) already exists. Returns whether a
        row was actually inserted."""
        pool = self._get_pool()
        memory_text = (memory_text or "").strip()
        if pool is None or not memory_text:
            return False

        try:
            new_vec = self._embed(memory_text)
            with pool.connection() as conn:
                cur = conn.execute(
                    "SELECT embedding FROM long_term_memory WHERE user_id = %s", (user_id,)
                )
                for (embedding_json,) in cur.fetchall():
                    vec = embedding_json if isinstance(embedding_json, list) else json.loads(embedding_json)
                    if _cosine(new_vec, vec) >= self.dedup_threshold:
                        return False  # near-duplicate already stored

                conn.execute(
                    "INSERT INTO long_term_memory (user_id, memory_text, embedding) "
                    "VALUES (%s, %s, %s)",
                    (user_id, memory_text, json.dumps(new_vec)),
                )
            return True
        except Exception:
            logger.exception("Long-term memory write failed for one fact; continuing.")
            return False

    def all(self, user_id: str) -> List[Dict[str, Any]]:
        """All stored memories for user_id, newest first. Used by the
        /api/memories UI endpoint; never used on the hot chat path."""
        pool = self._get_pool()
        if pool is None:
            return []
        try:
            with pool.connection() as conn:
                cur = conn.execute(
                    "SELECT id, memory_text, created_at, use_count FROM long_term_memory "
                    "WHERE user_id = %s ORDER BY created_at DESC",
                    (user_id,),
                )
                return [
                    {"id": r[0], "text": r[1], "created_at": str(r[2]), "use_count": r[3]}
                    for r in cur.fetchall()
                ]
        except Exception:
            logger.exception("Long-term memory listing failed.")
            return []

    def delete(self, user_id: str, memory_id: Optional[int] = None) -> bool:
        """Deletes one memory (memory_id given) or ALL memories for
        user_id (memory_id omitted)."""
        pool = self._get_pool()
        if pool is None:
            return False
        try:
            with pool.connection() as conn:
                if memory_id is None:
                    conn.execute("DELETE FROM long_term_memory WHERE user_id = %s", (user_id,))
                else:
                    conn.execute(
                        "DELETE FROM long_term_memory WHERE user_id = %s AND id = %s",
                        (user_id, memory_id),
                    )
            return True
        except Exception:
            logger.exception("Long-term memory delete failed.")
            return False