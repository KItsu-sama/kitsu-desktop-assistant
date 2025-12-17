# FILE 2: core/personality/trigger_manager.py (Your Code - Cleaned)
# =============================================================================
"""
Manages triggers with cooldowns and durations
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

log = logging.getLogger(__name__)


class TriggerManager:
    """
    Manages emotion triggers with cooldowns
    Loads from triggers.json
    """
    
    def __init__(self, triggers_path: Optional[Path] = None):
        self.triggers_path = triggers_path or Path("data/triggers.json")
        self.triggers = self._load_triggers()
        self.last_trigger_times: Dict[str, float] = {}
        
        log.info(f"TriggerManager loaded: {len(self.triggers)} triggers")
    
    def _load_triggers(self) -> Dict[str, Any]:
        """Load triggers from JSON file"""
        if not self.triggers_path.exists():
            log.warning(f"Triggers file not found: {self.triggers_path}")
            return {}
        
        try:
            data = json.loads(self.triggers_path.read_text(encoding="utf-8"))
            return data.get("triggers", {})
        except Exception as e:
            log.error(f"Failed to load triggers: {e}")
            return {}
    
    def reload(self):
        """Reload triggers from file"""
        self.triggers = self._load_triggers()
        log.info(f"Triggers reloaded: {len(self.triggers)} triggers")
    
    def can_fire(self, trigger_name: str) -> bool:
        """Check if trigger can fire (respects cooldown)"""
        trigger = self.triggers.get(trigger_name, {})
        cooldown = trigger.get("cooldown", 0.0)
        last_time = self.last_trigger_times.get(trigger_name, 0)
        
        elapsed = time.time() - last_time
        return elapsed >= cooldown
    
    def fire_trigger(self, trigger_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fire trigger if cooldown allows
        Returns list of emotion effects
        """
        if not self.can_fire(trigger_name):
            log.debug(f"Trigger {trigger_name} on cooldown")
            return None
        
        # Update last fire time
        self.last_trigger_times[trigger_name] = time.time()
        
        # Get emotions
        trigger = self.triggers.get(trigger_name, {})
        emotions = trigger.get("emotions", [])
        
        log.debug(f"Trigger fired: {trigger_name} ({len(emotions)} emotions)")
        return emotions
    
    def get_modifiers(self, trigger_name: str) -> Dict[str, float]:
        """Get personality modifiers for trigger"""
        trigger = self.triggers.get(trigger_name, {})
        return trigger.get("modifiers", {})
    
    def get_trigger_info(self, trigger_name: str) -> Optional[Dict[str, Any]]:
        """Get full trigger info"""
        return self.triggers.get(trigger_name)