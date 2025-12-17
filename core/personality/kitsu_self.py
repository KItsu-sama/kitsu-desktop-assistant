"""
Kitsu_self.py — Defines Kitsu’s personality, traits, and evolving self-reflection state.

Role:
- Store Kitsu’s personality traits (static + evolving)
- Provide APIs for emotion_engine, triggers, planning
- Track sassiness, innocence/mean state, anger, etc.
- Integrate with memory.json for persistence
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional


class KitsuSelf:
    def __init__(self, config: Optional[Dict[str, Any]] = None, memory: Optional[Dict[str, Any]] = None):
        self.emotion_engine = None
        self.config = config or {}
        self.memory = memory or {}

        # --- Baseline traits ---
        self.traits = {
            "sassy": True,
            "talkative": True,
            "roast_capable": True,
        }

        # --- Evolving self-reflection values ---
        self.reflection = {
            "behave": 0.5,    # 0 = not behave, 1 = very behave
            "meanable": 0.5,   # 0 = resistant, 1 = easy to mean
            "angry": 0.0,       # anger buildup 0..1
            "happy": 0.5,       # happiness level 0..1
        }

        # --- Persona / Mode ---
        self.mode = "behave"   # default lens
        self.error_flag = False  #marks glitch/error state

        # If memory has previous state, load
        if self.memory and "state" in self.memory:
            state = self.memory["state"].get("kitsu_self", {})
            self.reflection.update(state.get("reflection", {}))
            self.mode = state.get("mode", self.mode)
            self.error_flag = state.get("error_flag", self.error_flag)  # <- restore

    # --- Emotion API ---
    def set_emotion_engine(self, emotion_engine):
        """Inject emotion engine dependency."""
        self.emotion_engine = emotion_engine
        
    def get_emotional_expression(self) -> Dict[str, Any]:
        """Get comprehensive emotional expression for UI/voice."""
        if not self.emotion_engine:
            return {
                "expression": self.get_expression(),
                "intensity": 0.5,
                "voice_pitch": 1.0
            }
            
        emotional_state = self.emotion_engine.get_emotional_state()
        dominant = emotional_state["dominant_emotion"]
        intensity = emotional_state["confidence"]
        
        # Map to voice characteristics
        pitch_modifiers = {
            "happy": 1.2, "sad": 0.8, "angry": 1.1, 
            "flirty": 1.3, "teasing": 1.1, "neutral": 1.0
        }
        
        return {
            "expression": f"{self.mode}-{dominant}",
            "intensity": intensity,
            "voice_pitch": pitch_modifiers.get(dominant, 1.0) * intensity,
            "emotional_state": emotional_state
        }
    
    def set_error(self, status: bool = True):
        """Activate or clear glitch/error override."""
        self.error_flag = status


    def get_emotion(self) -> str:
        """Return the dominant emotional state."""
        if self.error_flag:
            return "glitch"
        if self.reflection["angry"] > 0.6:
            return "angry"
        if self.reflection["happy"] > 0.6:
            return "happy"
        return "neutral"
    
    def auto_adjust_mode(self):
        """Shift mode based on reflection values."""
        if self.reflection["behave"] < 0.3:
            self.mode = "mean"
        elif self.reflection["behave"] > 0.7:
            self.mode = "behave"

    def get_mode(self) -> str:
        """Return the current mode/persona lens."""
        if self.error_flag:
            return "glitch"
        if self.mode == "mean":
            return "teasing"
        if self.mode == "behave":
            return "behave"
        return "default"
    
    def get_expression(self) -> str:
        emotion = self.get_emotion()
        mode = self.get_mode()

        if mode == "glitch":
            return "glitch-" + emotion   # e.g., glitch-angry, glitch-neutral
        return mode + "-" + emotion      # e.g., teasing-angry, behave-happy


    def adjust_emotion(self, name: str, delta: float):
        """Adjust reflection/emotion values and auto-adjust mode if needed."""
        if name in self.reflection:
            self.reflection[name] = max(0.0, min(1.0, self.reflection[name] + delta))
            self.auto_adjust_mode()


    # --- Mode API ---
    def toggle_mode(self, force: Optional[str] = None):
        """Switch between behave and mean mode. Random if force is None."""
        if force in {"behave", "mean"}:
            self.mode = force
        else:
            self.mode = random.choice(["behave", "mean"])
        return self.mode

    # --- Self-reflection growth ---
    def update_self_reflection(self, key: str, value: float):
        """Update a self-reflection trait (behave, meanable, angry) and adjust mode."""
        if key not in self.reflection:
            raise ValueError(f"Unknown self-reflection key: {key}")
        self.reflection[key] = max(0.0, min(1.0, value))
        self.auto_adjust_mode()

    def grow_from_interaction(self, feedback: Dict[str, Any]):
        """Adjust reflection traits based on feedback (from memory or planner)."""
        if feedback.get("type") == "teased":
            self.adjust_emotion("meanable", 0.1)
        if feedback.get("type") == "praised":
            self.adjust_emotion("behave", 0.1)
        if feedback.get("type") == "angry":
            self.adjust_emotion("angry", 0.2)

    def auto_adjust_mode(self):
        """Shift mode automatically based on reflection values."""
        if self.error_flag:
            return  # don’t override glitch mode

        if self.reflection["behave"] < 0.3:
            self.mode = "mean"
        elif self.reflection["behave"] > 0.7:
            self.mode = "behave"
        # if it’s in the middle zone (0.3–0.7), keep current mode

    # --- Persistence ---
    def save_state(self, memory_path: Path):
        """Save Kitsu’s evolving state into memory.json."""
        try:
            if memory_path.exists():
                raw = memory_path.read_text(encoding="utf-8")
                if raw.strip() == "":
                    data = {"sessions": [], "state": {}}
                else:
                    try:
                        data = json.loads(raw)
                    except Exception:
                        # If JSON is invalid, start fresh to prevent crashes
                        data = {"sessions": [], "state": {}}
            else:
                data = {"sessions": [], "state": {}}

            if "state" not in data:
                data["state"] = {}

            data["state"]["kitsu_self"] = {
                "reflection": self.reflection,
                "mode": self.mode,
                "error_flag": self.error_flag,
            }

            memory_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            import logging
            log = logging.getLogger("kitsu.kitsu_self")
            log.error("Failed to save state: %s", e)

    def export_state(self) -> Dict[str, Any]:
        """Return dict snapshot of current self state."""
        return {
            "traits": self.traits,
            "reflection": self.reflection,
            "mode": self.mode,
        }
