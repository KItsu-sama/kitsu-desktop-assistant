# File: core/dev/handlers/response_trainer.py
# -----------------------------------------------------------------------------


import json
from pathlib import Path
from datetime import datetime
from typing import Optional

class ResponseTrainer:
    def __init__(self, kitsu_core=None, memory=None, logger=None):
        self.core = kitsu_core
        self.logger = logger
        # prefer explicit memory, otherwise try to infer from kitsu_core
        self.memory = memory if memory is not None else (getattr(kitsu_core, "memory", None) if kitsu_core else None)
        
        self.buffer_path = Path("./logs/response_overrides.jsonl")
        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-train settings
        self.auto_train_enabled = False
        self._training_proc = None


    def save_override(self, content: str) -> str:
        """Save corrected response for training.

        Attempt to retrieve the original assistant response from memory. Some
        memory implementations provide a `get_last_response()` helper; if that
        doesn't exist we fall back to scanning `memory.sessions` for the last
        assistant message (same approach used by ResponseRater).
        """
        if not content:
            return "No override provided"

        # Attempt to find the last assistant response robustly (support different memory APIs)
        original = None
        try:
            ex = self._get_last_exchange()
            original = ex.get("response")
        except Exception:
            original = None

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user": self._get_user_name(),
            "original": original,
            "override": content,
        }

        # Append override in a simple JSONL format
        with self.buffer_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Optionally trigger an asynchronous training job if auto-train is enabled
        try:
            if getattr(self, "auto_train_enabled", False):
                self.trigger_training_async()
        except Exception:
            if self.logger:
                self.logger.exception("Auto-train trigger failed")

        return "Override saved for training (will be applied to future matching inputs)"

    def _get_last_exchange(self) -> dict:
        """Return last user->assistant pair (prompt, response)."""
        try:
            if not self.memory:
                return {"prompt": None, "response": None}

            # If the memory provides a convenience helper, try that first
            if hasattr(self.memory, "get_last_response"):
                resp = self.memory.get_last_response()
                # helper may return a string or dict
                if isinstance(resp, dict):
                    return {"prompt": None, "response": resp.get("text")}
                return {"prompt": None, "response": resp}

            # Fallback to scanning sessions (same approach used elsewhere)
            sessions = list(self.memory.sessions)
            last_assistant = None
            prev_user = None
            for s in reversed(sessions):
                role = s.get("role")
                if role in ("kitsu", "assistant") and last_assistant is None:
                    last_assistant = s
                elif role == "user" and last_assistant is None:
                    prev_user = s
                elif role == "user" and last_assistant is not None:
                    prev_user = s
                    break

            prompt = prev_user.get("text") if prev_user else None
            response = last_assistant.get("text") if last_assistant else None
            return {"prompt": prompt, "response": response}
        except Exception:
            return {"prompt": None, "response": None}

    # Auto-train control methods
    def toggle_auto_train(self, arg: Optional[str] = None) -> str:
        """Enable/disable or query auto-train."""
        if arg is None:
            return "on" if self.auto_train_enabled else "off"

        a = str(arg).lower()
        if a in ("on", "true", "1", "enable", "enabled"):
            self.auto_train_enabled = True
            return "auto_train enabled"
        elif a in ("off", "false", "0", "disable", "disabled"):
            self.auto_train_enabled = False
            return "auto_train disabled"
        else:
            return "invalid argument, use 'on' or 'off'"

    def trigger_training_async(self, include_ratings: bool = False, min_rating: Optional[int] = None) -> str:
        """Start a background training run using the pipeline and stream output to a buffer/log.

        Args:
            include_ratings: whether to include /rate ratings when merging dev feedback
            min_rating: optional minimum rating to include
        """
        import subprocess, sys, threading
        from collections import deque

        if self._training_proc and self._training_proc.poll() is None:
            return "training already in progress"

        cmd = [sys.executable, "scripts/train_pipeline.py", "--train-only", "--include-dev-feedback"]
        if include_ratings:
            cmd.append("--include-ratings")
            if min_rating is not None:
                cmd.extend(["--min-rating", str(min_rating)])

        try:
            # Start process with pipes so we can stream output
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self._training_proc = proc

            # Prepare an in-memory buffer for tailing
            if not hasattr(self, '_output') or self._output is None:
                self._output = deque(maxlen=1000)

            # Ensure log file exists and is appended to
            log_path = Path("logs/auto_train.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)

            def _reader_thread(p, out_deque, logfile):
                try:
                    with logfile.open('a', encoding='utf-8') as lf:
                        while True:
                            raw = p.stdout.readline()
                            if not raw:
                                break
                            # Support bytes or str stdout from subprocess/fakes
                            if isinstance(raw, bytes):
                                try:
                                    line = raw.decode('utf-8', errors='ignore')
                                except Exception:
                                    line = raw.decode(errors='ignore')
                            else:
                                line = raw
                            out_deque.append(line.rstrip('\n'))
                            lf.write(line)
                            lf.flush()
                            if self.logger:
                                self.logger.info(line.rstrip('\n'))
                except Exception as e:
                    if self.logger:
                        self.logger.exception("Training log reader failed: %s", e)

            t = threading.Thread(target=_reader_thread, args=(proc, self._output, log_path), daemon=True)
            t.start()

            if self.logger:
                self.logger.info("Auto-train started (pid=%s)", proc.pid)
            return f"auto-train started (pid={proc.pid})"
        except Exception as e:
            if self.logger:
                self.logger.exception("Failed to start auto-train: %s", e)
            return f"failed to start auto-train: {e}"

    def training_status(self) -> str:
        """Return a concise string about training status."""
        if not getattr(self, '_training_proc', None):
            return "no auto-train process"
        if self._training_proc.poll() is None:
            return f"running (pid={self._training_proc.pid})"
        return f"finished (exit={self._training_proc.returncode})"

    def get_training_output(self, n_lines: int = 20) -> str:
        """Return the last n lines of training output as a single string."""
        try:
            out = list(getattr(self, '_output', []) )
            if not out:
                return ''
            return '\n'.join(out[-n_lines:])
        except Exception:
            return ''

    def find_override(self, user_input: str) -> Optional[str]:
        """Find the most recent override matching user_input (exact or substring)."""
        try:
            if not self.buffer_path.exists():
                return None
            with self.buffer_path.open("r", encoding="utf-8") as f:
                for line in reversed(f.readlines()):
                    try:
                        obj = json.loads(line)
                        orig = (obj.get("original") or "").strip()
                        if not orig:
                            continue
                        ui = (user_input or "").strip()
                        if not ui:
                            continue
                        if orig.lower() == ui.lower() or orig.lower() in ui.lower() or ui.lower() in orig.lower():
                            return obj.get("override")
                    except Exception:
                        continue
        except Exception:
            if self.logger:
                self.logger.debug("Failed to lookup override", exc_info=True)
        return None
    
    def _get_user_name(self):
        """Get current user name from memory."""
        try:
            if self.memory:
                info = self.memory.get_user_info()
            return info.get("name", "unknown") if isinstance(info, dict) else "unknown"
        except Exception:
            pass
        return "unknown"