# File: core/dev_console/error_viewer.py
# -----------------------------------------------------------------------------
"""View and export recent errors. By default reads ./logs/errors.log
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

class ErrorViewer:
    """View and export error logs"""
    
    def __init__(self, kitsu_core=None, log_path=None, logger=None):
        self.core = kitsu_core
        self.logger = logger
        self.log_path = Path(log_path) if log_path else Path("logs/errors.log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def show_last(self, n: Optional[int] = 10) -> str:
        """Show last N errors"""
        if not self.log_path.exists():
            return "No errors logged yet"
        
        lines = self.log_path.read_text().strip().split("\n")
        recent = lines[-n:] if n else lines
        
        result = f"📋 Last {len(recent)} errors:\n\n"
        for i, line in enumerate(recent, 1):
            result += f"{i}. {line}\n"
        
        return result
    
    def export_logs(self) -> str:
        """Export all logs to a file"""
        if not self.log_path.exists():
            return "No logs to export"
        
        export_path = Path(f"logs/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        export_path.write_text(self.log_path.read_text())
        
        return f"Logs exported to {export_path}"