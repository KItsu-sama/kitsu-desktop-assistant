"""
Integrated Planner System
planning logic adapted for the new modular architecture

File location: core/cognition/planner.py
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

log = logging.getLogger(__name__)


# =============================================================================
# Action Plan (Output Format)
# =============================================================================

@dataclass
class ActionPlan:
    """
    Unified planning output
    Represents what Kitsu will do in response to input
    """
    action_type: str  # "dialogue", "prank", "tease", "hug", "system", etc.
    text: str = ""    # Response text (if dialogue)
    emotion: str = "neutral"  # Current emotion
    voice_pitch: str = "default"  # Voice modulation
    params: Optional[Dict[str, Any]] = None  # Additional metadata
    confidence: float = 0.8  # Confidence in this plan
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for executor"""
        return {
            "type": self.action_type,
            "text": self.text,
            "emotion": self.emotion,
            "voice_pitch": self.voice_pitch,
            "params": self.params,
            "confidence": self.confidence
        }


# =============================================================================
# Sass Generator
# =============================================================================

class SassGenerator:
    """
    Generates sass, roasts, puns, and personality lines
    Loads from sass.json
    """
    
    def __init__(self, sass_path: str = "data/sass.json"):
        self.sass_path = Path(sass_path)
        self.sass_data = self._load_sass()
        
        log.info(f"SassGenerator loaded: {len(self.sass_data.get('roasts', []))} roasts")
    
    def _load_sass(self) -> Dict[str, List[str]]:
        """Load sass lines from JSON"""
        if not self.sass_path.exists():
            log.warning(f"Sass file not found: {self.sass_path}")
            return {
                "roasts": ["You're testing my patience..."],
                "fox_puns": ["What does the fox say? Not much right now!"],
                "behave": ["I'm being good, I promise! 🦊"],
                "mean": ["Aww, did I hurt your feelings?"],
                "flirty": ["You're kinda cute when you're flustered~ 💕"]
            }
        
        try:
            with open(self.sass_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to load sass file: {e}")
            return {}
    
    def pick(self, category: str) -> str:
        """Pick random line from category"""
        lines = self.sass_data.get(category, [])
        if not lines:
            log.warning(f"No sass lines for category: {category}")
            return f"[No {category} line available]"
        return random.choice(lines)
    
    def reload(self):
        """Reload sass data from file"""
        self.sass_data = self._load_sass()
        log.info("Sass data reloaded")


# =============================================================================
# Behavior Modes
# =============================================================================

class BehaviorModes:
    """
    Manages behavior modes (behave, mean, flirty)
    Maps to emotion engine's mood system
    """
    
    def __init__(self, emotion_engine: Optional[Any] = None):
        self.emotion_engine = emotion_engine
        self._override_mode: Optional[str] = None
    
    def get_mode(self) -> str:
        """
        Get current behavior mode
        
        Returns mode in priority order:
        1. Manual override (if set)
        2. Emotion engine mood (if available)
        3. Default "behave"
        """
        # Manual override takes priority
        if self._override_mode:
            return self._override_mode
        
        # Get from emotion engine
        if self.emotion_engine:
            try:
                return self.emotion_engine.mood
            except AttributeError:
                pass
        
        # Default
        return "behave"
    
    def set_mode(self, mode: str):
        """Manually set mode (overrides emotion engine)"""
        if mode in ["behave", "mean", "flirty"]:
            self._override_mode = mode
            log.info(f"Mode manually set: {mode}")
        else:
            log.warning(f"Invalid mode: {mode}")
    
    def clear_override(self):
        """Clear manual override, return to emotion engine control"""
        self._override_mode = None
        log.info("Mode override cleared")


# =============================================================================
# Main Planner
# =============================================================================

class Planner:
    """
    Main planning system - decides how to respond
    
    Flow:
    1. Analyze input → Detect triggers
    2. Get current emotion + mood
    3. Choose action type (dialogue, prank, etc.)
    4. Generate response text
    5. Return ActionPlan
    """
    
    def __init__(
        self,
        kitsu_self: Optional[Any] = None,
        memory: Optional[Any] = None,
        emotion_engine: Optional[Any] = None,
        trigger_manager: Optional[Any] = None,
        llm_interface: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.kitsu_self = kitsu_self
        self.memory = memory
        self.emotion_engine = emotion_engine
        self.trigger_manager = trigger_manager
        self.llm = llm_interface
        self.config = config or {}
        
        # Initialize subsystems
        self.sass = SassGenerator(
            sass_path=self.config.get("sass_path", "data/sass.json")
        )
        self.behavior_modes = BehaviorModes(emotion_engine)
        
        log.info("Planner initialized")
    
    # =========================================================================
    # Voice Pitch
    # =========================================================================
    
    def _derive_pitch(self, emotion: str) -> str:
        """Map emotion to voice pitch"""
        mapping = {
            "happy": "high",
            "excited": "higher",
            "angry": "low",
            "sad": "lower",
            "embarrassed": "wobbly",
            "flustered": "wobbly",
            "teasing": "playful",
            "affection": "soft",
            "neutral": "default"
        }
        return mapping.get(emotion, "default")
    
    def _get_pitch(self, emotion: str) -> str:
        """Get voice pitch based on config mode"""
        mode = self.config.get("voice_mode", "auto_pitch")
        
        if mode == "auto_pitch":
            return self._derive_pitch(emotion)
        elif mode == "manual_pitch":
            return self.config.get("voice_pitch", "default")
        elif mode == "text_only":
            return f"[pitch: {self._derive_pitch(emotion)}]"
        
        return "default"
    
    # =========================================================================
    # Text Decision Logic
    # =========================================================================
    
# core/cognition/planner.py (modified)

    async def _decide_text_with_llm(
        self,
        user_text: str,
        emotion: str,
        mode: str
    ) -> str:
        """Generate response using trained model"""
        if not self.llm:
            return self._decide_text_fallback(user_text, emotion, mode)
        
        try:
            style = self.emotion_engine.style if self.emotion_engine else "chaotic"
            
            # Use MINIMAL prompt builder
            from core.llm.prompt_builder import MinimalPromptBuilder
            builder = MinimalPromptBuilder(self.memory)
            
            prompt = builder.build_prompt(
                user_input=user_text,
                emotion=emotion,
                mood=mode,
                style=style
            )
            
            # Model should respond in-character from training
            response = await self.llm.generate_raw(prompt, stream=False)
            return response
            
        except Exception as e:
            log.warning(f"LLM generation failed: {e}, using fallback")
            return self._decide_text_fallback(user_text, emotion, mode)
    
    def _decide_text_fallback(
        self,
        user_text: str,
        emotion: str,
        mode: str
    ) -> str:
        """
        Fallback text generation
        Used when LLM is unavailable or fails
        """
        lower = user_text.lower()
        
        # 1. Insult detection → Roast back
        if any(word in lower for word in ["dumb", "stupid", "loser", "idiot"]):
            return self.sass.pick("roasts")
        
        # 2. Fox puns (20% chance)
        if random.random() < 0.2:
            return self.sass.pick("fox_puns")
        
        # 3. Mode-based sass lines
        if mode == "mean":
            if random.random() < 0.3:
                return self.sass.pick("mean")
        
        elif mode == "flirty":
            if random.random() < 0.3:
                return self.sass.pick("flirty")
        
        elif mode == "behave":
            if random.random() < 0.1:
                return self.sass.pick("behave")
        
        # 4. Default personality lines
        if self.kitsu_self:
            try:
                personality = self.kitsu_self.export_state()
                default_lines = personality.get("default_lines", ["..."])
                return random.choice(default_lines)
            except Exception:
                pass
        
        # 5. Final fallback
        return "..."
    
    # =========================================================================
    # Main Planning Logic
    # =========================================================================
    
    async def plan(self, user_text: str) -> ActionPlan:
        """
        Main planning method - decides what to do
        
        Args:
            user_text: User input
            
        Returns:
            ActionPlan with action type and parameters
        """
        try:
            # Save to memory
            if self.memory:
                try:
                    # Get current emotion for memory tagging
                    current_emotion = (
                        self.emotion_engine.get_current_emotion()
                        if self.emotion_engine else "neutral"
                    )
                    self.memory.remember("user", user_text, emotion=current_emotion)
                except Exception as e:
                    log.warning(f"Memory save failed: {e}")
            
            # Get current state
            emotion = (
                self.emotion_engine.get_current_emotion()
                if self.emotion_engine else "neutral"
            )
            mode = self.behavior_modes.get_mode()
            
            # Check for triggers first (priority over normal dialogue)
            if self.trigger_manager and self.emotion_engine:
                try:
                    trigger_name = self._detect_trigger(user_text, emotion)
                    if trigger_name:
                        action = await self._resolve_trigger(trigger_name)
                        if action:
                            return action
                except Exception as e:
                    log.warning(f"Trigger check failed: {e}")
            
            # No trigger → Generate normal dialogue
            text = await self._decide_text_with_llm(user_text, emotion, mode)
            pitch = self._get_pitch(emotion)
            
            # Create action plan
            plan = ActionPlan(
                action_type="dialogue",
                text=text,
                emotion=emotion,
                voice_pitch=pitch,
                params={
                    "mode": mode,
                    "style": (
                        self.emotion_engine.style
                        if self.emotion_engine else "chaotic"
                    )
                }
            )
            
            # Save Kitsu's response to memory
            if self.memory:
                try:
                    self.memory.remember("kitsu", text, emotion=emotion)
                except Exception as e:
                    log.warning(f"Memory save failed: {e}")
            
            return plan
            
        except Exception as e:
            log.exception(f"Planning failed: {e}")
            # Emergency fallback
            return ActionPlan(
                action_type="dialogue",
                text="*tilts head* I'm having trouble thinking right now...",
                emotion="confused",
                confidence=0.1
            )
    
    def _detect_trigger(self, user_text: str, emotion: str) -> Optional[str]:
        """
        Detect if input contains a trigger
        
        Args:
            user_text: User input
            emotion: Current emotion
            
        Returns:
            Trigger name if found, None otherwise
        """
        # Check trigger manager's keyword mappings
        if not self.trigger_manager:
            return None
        
        # Use trigger manager's resolve_trigger method
        if hasattr(self.trigger_manager, 'triggers'):
            # Check symbols
            for symbol, trigger_name in self.trigger_manager.triggers.get("symbols", {}).items():
                if symbol in user_text:
                    return trigger_name
            
            # Check keywords
            lower = user_text.lower()
            for keyword, trigger_name in self.trigger_manager.triggers.get("keywords", {}).items():
                if keyword in lower:
                    return trigger_name
        
        return None
    
    async def _resolve_trigger(self, trigger_name: str) -> Optional[ActionPlan]:
        """
        Resolve trigger to action plan
        
        Args:
            trigger_name: Name of trigger to fire
            
        Returns:
            ActionPlan for trigger action, or None if trigger fails
        """
        if not self.emotion_engine:
            return None
        
        try:
            # Fire trigger (updates emotion stack)
            self.emotion_engine.fire_trigger(trigger_name)
            
            # Get trigger info
            trigger_info = self.trigger_manager.get_trigger_info(trigger_name)
            if not trigger_info:
                return None
            
            # Determine action type from trigger
            action_type = trigger_info.get("action_type", "dialogue")
            
            # Generate appropriate response
            if action_type == "prank":
                text = self.sass.pick("mean")
            elif action_type == "tease":
                text = self.sass.pick("roasts")
            elif action_type == "affection":
                text = self.sass.pick("flirty")
            else:
                # Default dialogue
                emotion = self.emotion_engine.get_current_emotion()
                mode = self.behavior_modes.get_mode()
                text = await self._decide_text_with_llm("", emotion, mode)
            
            # Get current emotion after trigger
            emotion = self.emotion_engine.get_current_emotion()
            pitch = self._get_pitch(emotion)
            
            return ActionPlan(
                action_type=action_type,
                text=text,
                emotion=emotion,
                voice_pitch=pitch,
                params={"trigger": trigger_name}
            )
            
        except Exception as e:
            log.error(f"Trigger resolution failed: {e}")
            return None
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """Get planner state"""
        return {
            "mode": self.behavior_modes.get_mode(),
            "has_llm": self.llm is not None,
            "has_memory": self.memory is not None,
            "has_emotions": self.emotion_engine is not None
        }
