"""
response_manager/overrides.py

Define forbidden runtime patterns and helpers used by runtime checks and
ModelResetController. Patterns here are intentionally conservative: they
are used to detect accidental role labels or machine-only markers that
must never appear in a user-facing assistant message.
"""
import re
from typing import List

# Forbidden top-level tokens or role labels that may be produced by broken
# prompts or model hallucination. These are used to detect role-echo and
# multi-response output.
FORBIDDEN_PATTERNS: List[str] = [
    r"^Greeting:\b",
    r"^Response:\b",
    r"^Assistant:\b",
    r"^User:\b",
    r"^System:\b",
    r"^<continue>\b",
]

FORBIDDEN_REGEX = re.compile("|".join(f"(?:{p})" for p in FORBIDDEN_PATTERNS), flags=re.I | re.M)


def contains_forbidden_labels(text: str) -> bool:
    """Return True if the text contains any forbidden role labels or markers."""
    if not text:
        return False
    return bool(FORBIDDEN_REGEX.search(text))
