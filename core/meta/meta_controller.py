"""Lightweight meta-controller for gated action execution.

This module enforces simple, deterministic rules for actions suggested by
the LLM. It is intentionally tiny and fast to keep Kitsu runnable on
low-spec hardware.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from functools import partial

from core.meta.action_parser import Action
from core.meta import action_executor

__all__ = ["MetaController"]


@dataclass
class ContinuationState:
    count: int = 0



from .response_modes import ResponseMode
from .safety_gate import safety_check

# core/meta/meta_controller.py

from enum import Enum

class ResponseMode(Enum):
    FAST = "fast"
    PLAYFUL = "playful"
    FLIRTY = "flirty"
    REASONED = "reasoned"
    MEMORY_HEAVY = "memory_heavy"
    SILENT = "silent"
    FALLBACK = "fallback"


class MetaController:
    def decide(self, *, emotion_state, emotion_analysis, user_permissions):
        """
        Decide how Kitsu should respond THIS turn.
        Very cheap. No LLM. No loops.
        """

        if emotion_state.get("is_hidden"):
            return ResponseMode.SILENT

        intent = emotion_analysis.get("intent")
        sentiment = emotion_analysis.get("sentiment")
        dominant = emotion_state.get("dominant_emotion")

        # ---- Hard redirects (not censorship) ----
        if intent == "dangerous":
            return ResponseMode.FALLBACK

        # ---- Flirty / sus control ----
        allow_nsfw = user_permissions.get("allow_nsfw", False)

        if dominant in ["teasing", "flirty", "playful"]:
            if allow_nsfw:
                return ResponseMode.FLIRTY
            return ResponseMode.PLAYFUL

        # ---- Depth control ----
        if intent in ["question", "explain", "why"]:
            return ResponseMode.REASONED

        if sentiment == "negative":
            return ResponseMode.MEMORY_HEAVY

        return ResponseMode.FAST

    def __init__(self, max_continuations: int = 2):
        # keep backward compatible default
        self.max_continuations = max_continuations

    def decide_and_handle_action(
        self,
        action: Optional[Action],
        *,
        user_permissions: Optional[Dict[str, Any]] = None,
        last_topic: Optional[str] = None,
        continuation_state: Optional[ContinuationState] = None,
        emotion_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a parsed action token and return a decision dict.

        Decision dict keys: approved, reason, execute (callable or None), requires_confirmation
        """
        user_permissions = user_permissions or {}
        if action is None:
            return {"approved": False, "reason": "no action", "execute": None, "requires_confirmation": False}

        kind = action.kind

        # SEARCH
        if kind == "search":
            if not user_permissions.get("allow_search", True):
                return {"approved": False, "reason": "permission denied: search", "execute": None, "requires_confirmation": False}

            query = action.payload.get("query", "")
            # simple topic heuristic: first alpha word
            parts = [p for p in query.lower().split() if p.isalpha()]
            topic = parts[0] if parts else ""
            if last_topic and topic and topic != last_topic.lower():
                return {"approved": False, "reason": "topic differs, confirmation required", "execute": None, "requires_confirmation": True}

            return {"approved": True, "reason": "approved", "execute": partial(action_executor.execute_search, query, 5), "requires_confirmation": False}

        # GET_TIME
        if kind == "get_time":
            return {"approved": True, "reason": "approved", "execute": partial(action_executor.execute_get_time), "requires_confirmation": False}

        # CONTINUE
        if kind == "continue":
            if not user_permissions.get("allow_continue", True):
                return {"approved": False, "reason": "permission denied: continue", "execute": None, "requires_confirmation": False}

            if continuation_state is None:
                continuation_state = ContinuationState(0)

            if continuation_state.count >= getattr(self, "max_continuations", 2):
                return {"approved": False, "reason": "continuation limit reached", "execute": None, "requires_confirmation": False}

            intensity = 0.0
            if emotion_state:
                intensity = float(emotion_state.get("intensity", 0.0))
            if intensity > 0.85:
                return {"approved": False, "reason": "emotion intensity too high for continuation", "execute": None, "requires_confirmation": False}

            reason = action.payload.get("reason") if action.payload else None
            return {"approved": True, "reason": "approved", "execute": partial(action_executor.handle_continue, reason, continuation_state.count, getattr(self, "max_continuations", 2)), "requires_confirmation": False}

        return {"approved": False, "reason": f"unsupported action: {kind}", "execute": None, "requires_confirmation": False}

    def apply_safety_gate(self, meta_input):
        """
        Apply safety gate to meta_input.
        Returns (is_blocked: bool, redirect_mode: Optional[ResponseMode])
        """
        return safety_check(meta_input)