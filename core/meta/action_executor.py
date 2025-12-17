"""Safe, deterministic executor stubs for internal action tokens.

These are small, offline-friendly stubs suitable for unit tests and later
integration with real retrieval/time services. Executors return machine-only
structured outputs; they do NOT perform any re-invocation of the LLM.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import datetime
from dataclasses import dataclass

__all__ = ["execute_search", "execute_get_time", "handle_continue"]


def execute_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Deterministic fake search results. Do NOT perform network calls here.

    Integration hook: core.memory.retrieval.search(query, max_results)
    """
    q = query.strip().lower()
    results = []
    for i in range(1, max_results + 1):
        results.append({
            "title": f"Result {i} for {q}",
            "snippet": f"Snippet {i} (deterministic) for query: {q}",
            "url": f"https://example.local/search/{i}?q={q.replace(' ', '+')}"
        })

    return {"query": q, "results": results}


def execute_get_time(timezone: Optional[str] = None, now: Optional[datetime.datetime] = None) -> str:
    """Return an ISO-formatted time string. For tests, caller can inject `now`.

    timezone: optional string like 'UTC+7' (simple offset parsed as hours)
    """
    if now is None:
        now = datetime.datetime.utcnow()
    # keep UTC for deterministic output unless explicit offset provided
    if timezone is None:
        return now.replace(microsecond=0).isoformat() + "Z"

    # simple parser for timezone offsets like UTC+7 or UTC-03
    tz = timezone.upper().strip()
    if tz.startswith("UTC") and ("+" in tz or "-" in tz):
        try:
            sign = 1 if "+" in tz else -1
            offset = int(tz.split("+")[1] if "+" in tz else tz.split("-")[1])
            adj = now + datetime.timedelta(hours=sign * offset)
            return adj.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            pass

    # fallback to UTC string
    return now.replace(microsecond=0).isoformat() + "Z"


def handle_continue(reason: Optional[str], continuation_count: int, max_allowed: int = 2) -> Dict[str, Any]:
    """Decide whether to allow a continuation and compute the new count.

    Returns a dict: {"allowed": bool, "new_count": int}
    """
    if continuation_count >= max_allowed:
        return {"allowed": False, "new_count": continuation_count}

    return {"allowed": True, "new_count": continuation_count + 1}
