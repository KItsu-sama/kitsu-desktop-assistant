# core/triggers.py
"""
triggers.py — Manage Kitsu's triggers and reactions

Role:
- Load triggers from data/triggers.json
- Apply emotions, modifiers, and actions
- Respect cooldowns per trigger
- Forward actions to executor
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional

class Triggers:
    def __init__(self, emotion_engine, executor, trigger_file: str = "data/triggers.json"):
        self.emotion_engine = emotion_engine
        self.executor = executor
        self.trigger_file = Path(trigger_file)

        self.triggers: Dict[str, Any] = {}
        self.symbols: Dict[str, str] = {}
        self.keywords: Dict[str, str] = {}
        self._last_trigger_times: Dict[str, float] = {}

        self._load_triggers()

    def _load_triggers(self):
        if not self.trigger_file.exists():
            self.triggers, self.symbols, self.keywords = {}, {}, {}
            return
        with open(self.trigger_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.triggers = data.get("triggers", {})
        self.symbols = data.get("symbols", {})
        self.keywords = data.get("keywords", {})

    def reload(self):
        self._load_triggers()

    def resolve_trigger(self, text: str) -> Optional[str]:
        """
        Try to resolve a trigger name from raw input text (symbols / keywords).
        """
        # check symbol mappings
        for symbol, emotion in self.symbols.items():
            if symbol in text:
                return emotion

        # check keywords
        for word, trigger_name in self.keywords.items():
            if word in text.lower():
                return trigger_name

        return None

    def get_triggered_action(self, trigger_name: str) -> Optional[Dict[str, Any]]:
        """
        Apply trigger by name → return structured plan:
        {
            "emotions": [...],
            "modifiers": {...},
            "actions": [...]
        }
        """
        trigger = self.triggers.get(trigger_name)
        if not trigger:
            return None

        now = time.time()
        last_time = self._last_trigger_times.get(trigger_name, 0)
        cooldown = trigger.get("cooldown", 0)

        # respect cooldown
        if now - last_time < cooldown:
            return None

        # update trigger timestamp
        self._last_trigger_times[trigger_name] = now

        plan = {
            "emotions": [],
            "modifiers": trigger.get("modifiers", {}),
            "actions": []
        }

        # === Emotions ===
        for emo in trigger.get("emotions", []):
            name = emo.get("name")
            intensity = emo.get("intensity", 0.5)
            duration = emo.get("duration", 5)
            self.emotion_engine.add_emotion(name, intensity, duration)
            plan["emotions"].append(emo)

        # === Actions ===
        for action in trigger.get("actions", []):
            chance = action.get("chance", 1.0)
            if random.random() <= chance:
                plan["actions"].append(action)

        return plan

    async def fire_trigger(self, trigger_name: str) -> Optional[Dict[str, Any]]:
        """
        High-level: runs trigger and executes actions immediately.
        """
        plan = self.get_triggered_action(trigger_name)
        if not plan:
            return None

        if plan.get("actions"):
            for action in plan["actions"]:
                await self.executor.execute_plan({"action": action})

        return plan
    
    async def handle_emotion(self, emotion: Optional[str]):
        """
        React to emotion updates (if a trigger is mapped to this emotion).
        Looks up emotion in triggers.json and fires the trigger if found.
        """
        if not emotion:
            return None

        # If emotion name matches a trigger key → fire it
        if emotion in self.triggers:
            return await self.fire_trigger(emotion)

        # Later: could add fuzzy matching, intensity-based lookup, etc.
        return None


