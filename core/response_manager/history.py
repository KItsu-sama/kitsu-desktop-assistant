"""
Minimal InteractionHistory for response storage and reset logging.
- Stores interaction records (timestamps, verdicts)
- Records reset events as learning candidates (flagged)
- Never performs training actions or writes to training datasets
"""
from pathlib import Path
import json
import datetime
from typing import List, Dict, Optional


class InteractionHistory:
    def __init__(self, storage: Optional[Path] = None):
        self.storage = Path(storage or "data/logs/resets.log")
        self.storage.parent.mkdir(parents=True, exist_ok=True)
        self._events: List[Dict] = []

    def _write_event(self, ev: Dict) -> None:
        # Append as JSONL for auditing and human review
        with open(self.storage, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def record_interaction(self, user_input: str, assistant_output: str, verdict: str = "unknown") -> None:
        ev = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "type": "interaction",
            "user_input": user_input,
            "assistant_output": assistant_output,
            "verdict": verdict,
        }
        self._events.append(ev)
        self._write_event(ev)

    def record_reset_event(self, reason: str, details: Optional[Dict] = None) -> None:
        ev = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "type": "reset_event",
            "reason": reason,
            "details": details or {},
            "learning_candidate": True,  # flagged for curator review; do NOT auto-train
        }
        self._events.append(ev)
        self._write_event(ev)

    def last_events(self, count: int = 10) -> List[Dict]:
        return list(self._events[-count:])
