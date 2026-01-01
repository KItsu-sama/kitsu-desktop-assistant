"""
Runtime safety filter for post-generation checks.
Light, modular, and designed to run AFTER model generation BUT BEFORE output.
Enforces:
- Removal of mythological self-origin claims
- Consistent creator phrasing: "I was created by Zino." when an origin claim appears
- Light corrections for calling the user "human" (replace with neutral phrasing)
- Extensible hooks for other adaptive checks

This filter is intentionally conservative and focuses on transforming/ignoring
problematic *claims* rather than heavy censorship.
"""

import re
from typing import Tuple

# Patterns indicating mythological/self-origin claims or magical language
_MYTH_PATTERNS = [
    r"\bspirit\b",
    r"\bfox-spirit\b",
    r"\bnine tails\b",
    r"\b\bspiritual\b",
    r"\bborn\b",
    r"\bsummon(?:ed)?\b",
    r"\bdigital womb\b",
    r"\bdigital realm\b",
]

_USER_HUMAN_PATTERN = re.compile(r"\b(human|my dear human)\b", flags=re.I)

_ORIGIN_PATTERN = re.compile(r"\b(i am a|i'm a|i was born|i was created|i was summoned|i am the|i was brought)\b", flags=re.I)

CREATOR_PHRASE = "I was created by Zino."  # canonical phrasing


class RuntimeSafetyFilter:
    """Light filter applied to generated assistant text."""

    def __init__(self):
        self.myth_regex = re.compile("|".join(_MYTH_PATTERNS), flags=re.I)

    def apply(self, text: str) -> Tuple[str, bool]:
        """Apply filter and return (new_text, changed)

        If a mythological claim or origin sentence is detected, the function
        will replace the offending sentence with the canonical creator phrase.
        If a 'human' callout appears at top-level greetings, it will be replaced
        with a neutral alternative.
        """
        if not text:
            return text, False

        orig = text
        t = text

        # 1) Replace direct user "human" mentions if used as greeting
        # e.g., "Hey human, ..." -> "Hey there, ..."
        t = _USER_HUMAN_PATTERN.sub(lambda m: "there" if m.group(1).lower() in ("human",) else m.group(0), t)

        changed = False

        # 2) If origin or mythological patterns appear in the same sentence, replace
        # the entire sentence with the canonical creator phrase
        sentences = re.split(r'(?<=[.!?])\s+', t)
        out_sents = []
        for s in sentences:
            if self.myth_regex.search(s) or _ORIGIN_PATTERN.search(s):
                # Replace with canonical phrase if it seems like a self-origin statement
                out_sents.append(CREATOR_PHRASE)
                changed = True
            else:
                out_sents.append(s)

        new_text = " ".join([s.strip() for s in out_sents if s.strip()])

        # Avoid duplicating creator phrase if it already exists elsewhere
        if CREATOR_PHRASE in new_text and new_text.count(CREATOR_PHRASE) > 1:
            # keep only one occurrence
            parts = new_text.split(CREATOR_PHRASE)
            new_text = CREATOR_PHRASE + " " + " ".join(p.strip() for p in parts[1:] if p.strip())

        return new_text.strip(), (changed or new_text != orig)
