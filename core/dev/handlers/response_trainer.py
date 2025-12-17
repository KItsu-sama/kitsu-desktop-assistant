# File: core/dev/handlers/response_trainer.py
# -----------------------------------------------------------------------------


import json
from pathlib import Path
from datetime import datetime

class ResponseTrainer:
    def __init__(self, kitsu_core=None, memory=None, logger=None):
        self.core = kitsu_core
        self.logger = logger
        # prefer explicit memory, otherwise try to infer from kitsu_core
        self.memory = memory if memory is not None else (getattr(kitsu_core, "memory", None) if kitsu_core else None)
        
        self.buffer_path = Path("./logs/response_overrides.jsonl")
        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)


    def save_override(self, content: str) -> str:
        """Save corrected response for training."""
        if not content:
            return "No override provided"
        
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user": self._get_user_name(),
            "original": self.memory.get_last_response() if self.memory else None,
            "override": content,
        }
        
        with self.buffer_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        return "Override saved for training"
    
    def _get_user_name(self):
        """Get current user name from memory."""
        try:
            if self.memory:
                info = self.memory.get_user_info()
            return info.get("name", "unknown") if isinstance(info, dict) else "unknown"
        except Exception:
            pass
        return "unknown"