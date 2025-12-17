"""Action token parser for internal machine-only action markers.

This module exposes a small, deterministic parser that extracts a single
action token from an LLM raw output. It is synchronous, pure, and has no
external dependencies.

Format supported (each must appear on its own line):
- <action:search query="...">  (query required)
- <action:get_time>
- <action:continue reason="...">  (reason optional)

Parser returns a dict: {"ok": True, "action": Action} when found,
{"ok": True, "action": None} when no action found, or
{"ok": False, "error": "message"} on validation errors.
"""
from __future__ import annotations

import re
import dataclasses
from dataclasses import dataclass
from typing import Optional, Dict, Any

__all__ = ["Action", "parse_action_from_text"]


@dataclass(frozen=True)
class Action:
    kind: str
    payload: Dict[str, Any]
    raw: str


# Regex to match action tokens on a full line
_ACTION_LINE_RE = re.compile(r"^<action:(?P<kind>[a-z_]+)(?:\s+(?P<attrs>[^>]+))?>$")

# Attribute extractors (very small, safe regexes)
_QUERY_RE = re.compile(r'query="(?P<q>[^"]+)"')
_REASON_RE = re.compile(r'reason="(?P<r>[^"]+)"')

# Sanitization helpers
_CONTROL_RE = re.compile(r"[\x00-\x1F#:\\]")

MAX_QUERY_LEN = 256
MAX_REASON_LEN = 128


def _sanitize_value(s: str, max_len: int) -> str:
    s = s.strip()
    # remove control chars and some punctuation that could act like tokens
    s = _CONTROL_RE.sub("", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s


def parse_action_from_text(text: str) -> Dict[str, Any]:
    """Parse a single action token from LLM raw output.

    Returns:
        {"ok": True, "action": Action} if a single valid action found
        {"ok": True, "action": None} if no action token is present
        {"ok": False, "error": "reason"} on parse/validation error
    """
    if not text:
        return {"ok": True, "action": None}

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    matches = []
    for ln in lines:
        m = _ACTION_LINE_RE.match(ln)
        if m:
            matches.append((ln, m))

    if not matches:
        return {"ok": True, "action": None}

    if len(matches) > 1:
        return {"ok": False, "error": "multiple actions not allowed"}

    raw_line, match = matches[0]
    kind = match.group("kind")
    attrs = match.group("attrs") or ""

    if kind == "search":
        qmatch = _QUERY_RE.search(attrs)
        if not qmatch:
            return {"ok": False, "error": "search requires query attribute"}
        raw_q = qmatch.group("q")
        if len(raw_q) > MAX_QUERY_LEN:
            return {"ok": False, "error": "query too long"}
        query = _sanitize_value(raw_q, MAX_QUERY_LEN)
        if not query:
            return {"ok": False, "error": "empty query"}
        return {"ok": True, "action": Action(kind="search", payload={"query": query}, raw=raw_line)}

    if kind == "get_time":
        # no attributes allowed
        if attrs.strip():
            return {"ok": False, "error": "get_time takes no attributes"}
        return {"ok": True, "action": Action(kind="get_time", payload={}, raw=raw_line)}

    if kind == "continue":
        rmatch = _REASON_RE.search(attrs)
        reason = None
        if rmatch:
            raw_r = rmatch.group("r")
            if len(raw_r) > MAX_REASON_LEN:
                return {"ok": False, "error": "reason too long"}
            reason = _sanitize_value(raw_r, MAX_REASON_LEN)
            if not reason:
                return {"ok": False, "error": "empty reason"}
        return {"ok": True, "action": Action(kind="continue", payload={"reason": reason}, raw=raw_line)}

    return {"ok": False, "error": f"unsupported action: {kind}"}
