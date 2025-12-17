# File: core/dev_console/stats_viewer.py
# -----------------------------------------------------------------------------
"""Return basic runtime stats about the Kitsu system. This is lightweight and
should be connected to your real system state (pass kitsu_state into constructor).
"""

import json
import psutil


class StatsViewer:
    def __init__(self, kitsu_core=None, kitsu_state=None, logger=None):
        # Accept either kitsu_core or a kitsu_state dict
        self.kitsu_core = kitsu_core
        self.kitsu_state = kitsu_state if kitsu_state is not None else (getattr(kitsu_core, '__dict__', None) if kitsu_core else None)
        self.logger = logger

    def get_stats(self) -> str:
        """Return a JSON string of stats. Caller can parse to pretty display."""
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory": psutil.virtual_memory()._asdict(),
            "fallback_count": self._get_fallback_count(),
            "modules": list(self.kitsu_state.keys()) if isinstance(self.kitsu_state, dict) else None,
        }
        return json.dumps(stats, ensure_ascii=False)

    def _get_fallback_count(self):
        # Placeholder: integrate with fallback manager or a metric collector
        try:
            if self.kitsu_state and isinstance(self.kitsu_state, dict):
                return self.kitsu_state.get("fallback_count", 0)
        except Exception:
            pass
        return 0