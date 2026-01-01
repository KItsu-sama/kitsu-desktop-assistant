"""
ModelResetController

Detects instability in generated outputs (loops, role-echo, forbidden labels)
and performs a safe reset action:
 - aborts current generation
 - resets model state (attempts soft reset; never sets model to None)
 - restores last known stable prompt where possible
 - logs reset events to InteractionHistory for curator review

The reset action is intentionally conservative and side-effect free with
respect to training data and model weights.
"""
from typing import Optional, Tuple, List
import asyncio
import logging
import re

from core.response_manager.overrides import contains_forbidden_labels
from core.response_manager.history import InteractionHistory

log = logging.getLogger(__name__)


class ResetResult:
    def __init__(self, success: bool, message: str = "", restored_prompt: Optional[str] = None):
        self.success = success
        self.message = message
        self.restored_prompt = restored_prompt


class ModelResetController:
    def __init__(self, llm_interface, history: InteractionHistory, logger: Optional[logging.Logger] = None):
        self.llm = llm_interface
        self.history = history
        self.logger = logger or log
        self.last_stable_prompt: Optional[str] = None

    def save_stable_prompt(self, prompt: str) -> None:
        """Store a stable prompt checkpoint. The controller does not persist
        training or other sensitive data - this is only a runtime checkpoint."""
        self.last_stable_prompt = prompt
        self.logger.debug("Saved stable prompt checkpoint")

    def detect_instability(self, raw_response: str, recent_history: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Return (True, reason) if instability is detected.

        Heuristics implemented:
        - Forbidden role labels present (e.g., 'Greeting:', 'Response:')
        - Duplicate greetings / repeated identical assistant outputs
        - Repeated <continue> markers (loop indicators)
        """
        if not raw_response:
            return False, ""

        # 1) forbidden labels
        if contains_forbidden_labels(raw_response):
            return True, "forbidden_role_label"

        # 2) multiple <continue> markers
        continue_count = len(re.findall(r"<continue>", raw_response))
        if continue_count > 1:
            return True, "multiple_continue_markers"

        # 3) repeated recent assistant outputs -> simple loop detection
        if recent_history and recent_history:
            last = recent_history[-1]
            if last.strip() and last.strip() == raw_response.strip():
                return True, "repeated_output"

        # 4) excessively short repeated lines (echo loop)
        lines = [ln.strip() for ln in raw_response.splitlines() if ln.strip()]
        if lines:
            # detect if same line repeated, e.g., 'Hi' x3
            if any(re.fullmatch(re.escape(lines[0]), l) for l in lines[1:3]):
                return True, "short_echo_loop"

        return False, ""

    def perform_reset(self, reason: str) -> ResetResult:
        """Perform a safe reset of model state.

        Steps:
        1) Log the reset event to InteractionHistory (flagged for curator review)
        2) Attempt to revert LoRA stack to the previous stack if a LoRA manager is present
        3) Mark the LLM as unavailable and attempt a soft restart if possible
        4) Return the last stable prompt (if any) to allow the caller to retry

        This operation MUST NOT write to training sets or perform live tuning.
        """
        self.history.record_reset_event(reason, details={
            "model": getattr(self.llm, "model", None),
            "active_stack": getattr(self.llm, "lora_manager", None) and getattr(self.llm.lora_manager, "current_stack", None)
        })

        # 1) Try revert LoRA if available
        try:
            lm = getattr(self.llm, "lora_manager", None)
            if lm and hasattr(lm, "revert_last_switch"):
                reverted = lm.revert_last_switch()
                self.logger.info("LoRA revert attempted: %s", reverted)
        except Exception as e:
            self.logger.warning("LoRA revert failed: %s", e)

        # 2) Soft reset: mark unavailable and try restarting adapter/service
        try:
            self.llm.is_available = False

            # If adapter supports a soft reset, call it
            adapter = getattr(self.llm, "adapter", None)
            if adapter and hasattr(adapter, "close"):
                try:
                    adapter.close()
                except Exception:
                    pass

            # Try asynchronous restart helper when available
            try:
                coro = getattr(self.llm, "_try_restart_ollama", None)
                if coro and asyncio.iscoroutinefunction(coro):
                    # run it to completion; it's safe and returns True/False
                    res = asyncio.run(coro())
                    self.logger.info("_try_restart_ollama result: %s", res)
                else:
                    # fallback to checking availability
                    try:
                        ok = getattr(self.llm, "_check_availability", lambda: False)()
                        self.logger.info("_check_availability result: %s", ok)
                    except Exception:
                        pass
            except Exception as e:
                self.logger.warning("Restart attempt failed: %s", e)

        except Exception as e:
            self.logger.error("Model reset failed: %s", e)
            return ResetResult(False, message=str(e), restored_prompt=self.last_stable_prompt)

        # Return the last stable prompt so the caller can optionally retry
        return ResetResult(True, message="reset_performed", restored_prompt=self.last_stable_prompt)
