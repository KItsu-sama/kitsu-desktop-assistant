# File: core/dev_console/handlers/response_rater.py
# -----------------------------------------------------------------------------
"""Rate the last response and store the score for learning signals.

Ratings are written as JSONL to `logs/ratings.jsonl` and include metadata:
- timestamp
- score
- prompt
- response
- mood
- style
- lora_stack

Ratings are durable (written to disk) and are used only as observations that
can be ingested by a curator pipeline; they do not directly alter models.
"""

import time
import json
from pathlib import Path
from typing import Optional

RATINGS_LOG = Path("./logs/ratings.jsonl")
RATINGS_LOG.parent.mkdir(parents=True, exist_ok=True)


class ResponseRater:
    def __init__(self, kitsu_core=None, memory=None, logger=None):
        self.core = kitsu_core
        self.logger = logger
        # memory should be an instance of MemoryManager; allow injection or retrieve from core
        self.memory = memory if memory is not None else (getattr(kitsu_core, 'memory', None) if kitsu_core else None)

    def _get_last_exchange(self) -> dict:
        """Return last user->assistant pair (prompt, response) and metadata.
        Falls back to minimal values if memory is unavailable.
        """
        try:
            if not self.memory:
                return {"prompt": None, "response": None, "mood": None, "style": None, "lora_stack": None}

            sessions = list(self.memory.sessions)
            # Walk backwards looking for the last assistant response with a preceding user
            last_assistant = None
            prev_user = None
            for s in reversed(sessions):
                if s.get("role") in ("kitsu", "assistant") and last_assistant is None:
                    last_assistant = s
                elif s.get("role") == "user" and last_assistant is None:
                    # keep note in case assistant comes later
                    prev_user = s
                elif s.get("role") == "user" and last_assistant is not None:
                    prev_user = s
                    break

            prompt = prev_user.get("text") if prev_user else None
            response = last_assistant.get("text") if last_assistant else None

            # Mood/style from best-effort: memory state or kit self
            mood = getattr(self.memory.kitsu_self, 'mood', None) if getattr(self.memory, 'kitsu_self', None) else None
            style = getattr(self.memory.kitsu_self, 'style', None) if getattr(self.memory, 'kitsu_self', None) else None

            # LoRA stack from global manager if present
            lora_stack = None
            try:
                lora_stack = (self.core.llm.lora_manager.current_stack if (self.core and getattr(self.core, 'llm', None) and getattr(self.core.llm, 'lora_manager', None)) else None)
            except Exception:
                lora_stack = None

            return {"prompt": prompt, "response": response, "mood": mood, "style": style, "lora_stack": lora_stack}
        except Exception:
            return {"prompt": None, "response": None, "mood": None, "style": None, "lora_stack": None}

    def rate(self, score: Optional[str]) -> str:
        try:
            s = int(score)
        except Exception:
            return "invalid score"

        ts = int(time.time())
        meta = self._get_last_exchange()

        entry = {
            "ts": ts,
            "score": s,
            "prompt": meta.get("prompt"),
            "response": meta.get("response"),
            "mood": meta.get("mood"),
            "style": meta.get("style"),
            "lora_stack": meta.get("lora_stack") or [],
        }

        # Determine label
        entry["label"] = "positive" if s >= 4 else ("negative" if s < 3 else "neutral")

        try:
            with RATINGS_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            return "failed to write rating"

        if self.logger:
            self.logger.info("Response rated: %s", s)

        return "rating saved"