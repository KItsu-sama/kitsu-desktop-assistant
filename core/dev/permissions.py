# File: core/dev/permissions.py
# -----------------------------------------------------------------------------
"""Permission management for dev commands."""

from typing import Optional, Set
from pathlib import Path
import json


class PermissionManager:
    """Handles user permissions for dev console."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("data/config/dev_permissions.json")
        self.admins: Set[str] = self._load_admins()
    
    def _load_admins(self) -> Set[str]:
        """Load admin list from config."""
        try:
            if self.config_path.exists():
                data = json.loads(self.config_path.read_text())
                return set(data.get("admins", ["Zino", "Natadaide"]))
        except Exception:
            pass
        return {"Zino", "Natadaide"}
    
    def check(self, user_id: Optional[str], required_level: str) -> bool:
        """Check if user has required permission level."""
        if required_level == "user":
            return True  # All users can access
        if required_level == "admin":
            return user_id in self.admins
        return False
    
    def add_admin(self, user_id: str):
        """Add a new admin."""
        self.admins.add(user_id)
        self._save_admins()
    
    def _save_admins(self):
        """Persist admin list."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps({"admins": list(self.admins)}, indent=2))