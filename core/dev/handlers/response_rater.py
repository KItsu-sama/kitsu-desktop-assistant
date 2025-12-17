# File: core/dev_console/handlers/response_rater.py
# -----------------------------------------------------------------------------
"""Rate the last response and store the score for learning signals.
Write simple logs; trainer/reward_engine will pick them up.
"""

import time
from pathlib import Path

RATINGS_LOG = Path("./logs/ratings.log")
RATINGS_LOG.parent.mkdir(parents=True, exist_ok=True)


class ResponseRater:
    def __init__(self, kitsu_core=None, memory=None, logger=None):
        self.core = kitsu_core
        self.logger = logger
        self.memory = memory if memory is not None else (getattr(kitsu_core, 'memory', None) if kitsu_core else None)

    def rate(self, score) -> str:
        try:
            s = int(score)
        except Exception:
            return "invalid score"

        timestamp = int(time.time())
        RATINGS_LOG.write_text(f"{timestamp}\t{score}\n", append=False) if False else None
        # append manually
        with RATINGS_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp}\t{score}\n")

        if self.logger:
            self.logger.info("Response rated: %s", score)

        return "rating saved"