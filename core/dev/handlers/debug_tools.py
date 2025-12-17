# File: core/dev_console/debug_tools.py
# -----------------------------------------------------------------------------
"""Small debug tools: simulate errors, show a summary, and helpers for testing the fallback.
"""

import random
from pathlib import Path
import psutil
import json
from datetime import datetime

class DebugTools:
    """Debug and diagnostic tools"""

    def __init__(self, kitsu_core=None, logger=None):
        self.core = kitsu_core
        self.logger = logger
    
    def summary(self) -> str:
        """Get system summary"""
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        
        summary = f"""
🔧 DEBUG SUMMARY
{'='*50}
CPU Usage: {cpu}%
Memory: {mem.percent}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)
Uptime: {self._get_uptime()}

Kitsu Status:
  - Memory loaded: {hasattr(self.core, 'memory')}
  - LLM loaded: {hasattr(self.core, 'llm')}
  - Personality active: {hasattr(self.core, 'personality')}
"""
        return summary
    
    def _get_uptime(self) -> str:
        """Get system uptime"""
        uptime_seconds = psutil.boot_time()
        now = datetime.now().timestamp()
        delta = now - uptime_seconds
        
        hours = int(delta // 3600)
        minutes = int((delta % 3600) // 60)
        
        return f"{hours}h {minutes}m"
    
    def simulate_error(self) -> str:
        """Simulate an error for testing"""
        error_msg = f"[TEST ERROR] {datetime.now().isoformat()}: Simulated error for testing"
        
        # Log to error viewer
        log_path = Path("logs/errors.log")
        with open(log_path, "a") as f:
            f.write(error_msg + "\n")
        
        return "✅ Test error logged"