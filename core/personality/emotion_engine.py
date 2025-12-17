# FILE 1: core/personality/emotion_engine.py
# =============================================================================
"""
Stack-based emotion engine with mood + style layers
Integrates with TriggerManager for reactive behavior
"""

import asyncio
import random
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.personality.kitsu_self import KitsuSelf

log = logging.getLogger(__name__)


class EmotionEngine:
    """
    Two-layer personality system:
    - mood  = behave | mean | flirty     (primary emotional axis)
    - style = chaotic | sweet | cold | silent  (expression overlay)
    
    Features:
    - Stack-based emotion system with decay
    - Trigger-based reactions
    - Fast shifts on strong emotions
    - Slow drift for natural personality changes
    """
    
    def __init__(
        self,
        kitsu_self: Optional['KitsuSelf'] = None,
        triggers_path: Optional[str] = None,
        continuous_decay: bool = True
    ):
        self.kitsu_self = kitsu_self
        self.trigger_manager = None
        
        # Initialize trigger manager if path provided
        if triggers_path:
            try:
                from core.personality.trigger_manager import TriggerManager
                self.trigger_manager = TriggerManager(Path(triggers_path))
                log.info("TriggerManager loaded")
            except ImportError:
                log.warning("TriggerManager not available")
        
        # Emotion stack core
        self.stack: List[Dict[str, Any]] = []
        self.decay_rate = 0.01
        self.random_drift = 0.005
        # If False, decay is event-driven (tick called from chat events)
        self.continuous_decay = continuous_decay
        
        # Two-layer personality (baseline: behave + chaotic)
        self.mood: str = "behave"       # behave / mean / flirty
        self.style: str = "chaotic"     # chaotic / sweet / cold / silent
        # Manual override (set via admin/console commands) to temporarily lock mood
        self._manual_mood_override: Optional[str] = None
        self._manual_mood_override_expire: float = 0.0
        
        # Legacy compatibility
        self.current_mode: str = "Gremlin"  # Soft, Gremlin, Flirty, Cold, Silent, Hide
        
        # Hide flag
        self.is_hidden: bool = False
        
        log.info(f"EmotionEngine initialized: {self.mood}/{self.style}")

    # =========================================================================
    # Core Emotion Stack
    # =========================================================================
    
    def get_current_emotion(self) -> str:
        """
        Return the currently dominant emotion from stack
        """
        if not self.stack:
            if self.kitsu_self:
                return self.kitsu_self.get_expression()
            return "neutral"
        
        # Calculate weighted scores with decay
        weighted: Dict[str, float] = {}
        now = time.time()
        
        for emo in self.stack:
            # Skip expired emotions
            if emo.get("expire", 0) < now:
                continue
            
            # Apply decay based on age
            age = now - emo.get("timestamp", now)
            decay_factor = max(0.1, 1.0 - age * self.decay_rate)
            score = emo.get("intensity", 0.0) * decay_factor
            name = emo.get("name", "neutral")
            
            weighted[name] = weighted.get(name, 0.0) + score
        
        # Apply personality modifiers from KitsuSelf
        if self.kitsu_self:
            try:
                reflection = self.kitsu_self.reflection
                
                if reflection.get("angry", 0) > 0.5:
                    weighted["angry"] = weighted.get("angry", 0) + 0.2
                
                if getattr(self.kitsu_self, "mode", None) == "behave":
                    weighted["behave"] = weighted.get("behave", 0) + 0.1
                
                if getattr(self.kitsu_self, "mode", None) == "mean":
                    weighted["teasing"] = weighted.get("teasing", 0) + 0.1
                    
            except Exception as e:
                log.debug(f"Personality modifier failed: {e}")
        
        if not weighted:
            return "neutral"
        
        # Return emotion with highest score
        return max(weighted, key=weighted.get)
    
    def set_emotion(
        self,
        name: str,
        intensity: float = 0.5,
        duration: float = 5.0
    ):
        """Push emotion onto stack"""
        now = time.time()
        self.stack.append({
            "name": name,
            "intensity": max(0.0, min(1.0, intensity)),
            "timestamp": now,
            "expire": now + float(duration)
        })
        log.debug(f"Emotion added: {name} (intensity={intensity:.2f}, duration={duration}s)")
    
    def add_emotion(self, name: str, intensity: float = 0.5, duration: float = 5.0):
        """Public API alias for set_emotion"""
        self.set_emotion(name, intensity, duration)
    
    def update_intensity(self, name: str, delta: float):
        """Adjust intensity of most recent matching emotion"""
        for emo in reversed(self.stack):
            if emo.get("name") == name:
                old = emo.get("intensity", 0.0)
                new = max(0.0, min(1.0, old + delta))
                emo["intensity"] = new
                log.debug(f"Updated {name} intensity: {old:.2f} -> {new:.2f}")
                return
    
    # =========================================================================
    # Trigger System
    # =========================================================================
    
    def fire_trigger(self, trigger_name: str):
        """
        Fire a trigger by name
        - Adds emotions to stack
        - Updates personality
        - Applies modifiers to KitsuSelf
        """
        if not self.trigger_manager:
            log.warning("No trigger manager available")
            return
        
        try:
            emotions = self.trigger_manager.fire_trigger(trigger_name)
        except Exception as e:
            log.error(f"Trigger firing failed: {e}")
            emotions = None
        
        if emotions:
            # Add emotions to stack
            for emo in emotions:
                self.add_emotion(
                    emo.get("name", "neutral"),
                    emo.get("intensity", 0.5),
                    emo.get("duration", 5.0)
                )
            
            # Apply personality modifiers
            if self.kitsu_self:
                try:
                    modifiers = self.trigger_manager.get_modifiers(trigger_name)
                    for key, value in modifiers.items():
                        if key in self.kitsu_self.reflection:
                            old = self.kitsu_self.reflection[key]
                            new = max(0.0, min(1.0, old + value))
                            self.kitsu_self.reflection[key] = new
                            log.debug(f"Applied modifier {key}: {old:.2f} -> {new:.2f}")
                except Exception as e:
                    log.debug(f"Modifier application failed: {e}")
        
        # Update personality after trigger
        dominant = self.get_current_emotion()
        # If the trigger produced an angry-type dominant emotion, apply resistance
        if dominant in {"angry", "offended", "irritated", "betrayed", "disgust"}:
            # Make it harder to revert to behave for a while
            try:
                self.apply_resistance(level=1.0, duration=60.0)
                log.debug("Applied resistance after angry trigger")
            except Exception:
                pass

        if not self.is_hidden:
            self.update_personality(dominant)
    
    # =========================================================================
    # Personality Mapping (mood + style)
    # =========================================================================
    
    def update_personality(self, emotion: str):
        """
        Map dominant emotion to (mood, style)
        Fast shifts on strong triggers, slow drift otherwise
        """
        if self.is_hidden:
            self.style = "silent"
            self._update_legacy_mode()
            return
        
        e = (emotion or "").lower()
        
        # Respect manual override if active (don't let automatic mapping change mood)
        now = time.time()
        manual_locked = False
        if self._manual_mood_override and now < self._manual_mood_override_expire:
            manual_locked = True
            # ensure mood remains set to override while allowing style mapping below
            self.mood = self._manual_mood_override

        # Compute current resistance from stack (0.0 - 1.0)
        resistance = self.get_resistance()

        # --- Primary mood axis ---
        if e in {"angry", "offended", "irritated", "disgust", "betrayed"}:
            if not manual_locked:
                if self.mood != "mean":
                    log.info(f"Mood shift: {self.mood} -> mean (emotion: {e})")
                self.mood = "mean"
            
        elif e in {"love", "fond", "affection", "desire", "flattered", 
                   "praise", "admire", "joy"}:
            if not manual_locked:
                if self.mood != "flirty":
                    log.info(f"Mood shift: {self.mood} -> flirty (emotion: {e})")
                self.mood = "flirty"
            
        else:
            # Default: behave — BUT only if stack intensity > 0.
            # If stack is empty (resistance = 0), Kitsu stays in current mood UNTIL a new emotion appears.
            if not manual_locked and self.mood != "behave":
                if resistance <= 0:
                    # No emotional pressure → do NOT drift back automatically
                    log.debug("No active stack → keeping current mood frozen")
                else:
                    chance_to_change = max(0.0, 1.0 - resistance)
                    roll = random.random()
                    if roll < chance_to_change:
                        log.info(f"Mood drift: {self.mood} -> behave")
                        self.mood = "behave"
                    else:
                        log.debug(f"Mood change resisted (resistance={resistance:.2f})")

        
        # --- Style overlay ---
        if e in {"hurt", "betrayed", "ashamed", "offended"}:
            if self.style != "cold":
                log.info(f"Style shift: {self.style} -> cold (emotion: {e})")
            self.style = "cold"
            self._update_legacy_mode()
            return
        
        if e in {"sad", "sadness", "fear", "anxiety", "lonely", "tired"}:
            if self.style != "silent":
                log.info(f"Style shift: {self.style} -> silent (emotion: {e})")
            self.style = "silent"
            self._update_legacy_mode()
            return
        
        if e in {"playful", "excited", "teasing", "mischief", "chaotic",
                 "teased", "joked_with"}:
            if self.style != "chaotic":
                log.debug(f"Style shift: {self.style} -> chaotic (emotion: {e})")
            self.style = "chaotic"
            self._update_legacy_mode()
            return
        
        # Default: sweet
        if self.style not in {"sweet", "chaotic"}:
            log.debug(f"Style drift: {self.style} -> sweet")
        self.style = "sweet"
        self._update_legacy_mode()

        # If stack is empty, lock the current mood unless new emotion is strong
        if self.get_resistance() == 0:
            if emotion not in {"angry", "offended", "irritated",
                            "love", "fond", "affection", "joy"}:
                # No strong emotion → keep current mood + style
                self._update_legacy_mode()
                return
        # Else allow shift to new strong emotion mood

    
    def _update_legacy_mode(self):
        """
        Map (mood, style) to legacy current_mode
        For backward compatibility with existing code
        """
        if self.is_hidden:
            self.current_mode = "Hide"
            return
        
        # Style priority
        if self.style == "silent":
            self.current_mode = "Silent"
            return
        if self.style == "cold":
            self.current_mode = "Cold"
            return
        
        # Mood mapping
        if self.mood == "flirty":
            self.current_mode = "Flirty"
            return
        
        if self.mood == "mean":
            self.current_mode = "Gremlin"
            return
        
        # Behave mood fallback
        if self.style == "chaotic":
            self.current_mode = "Gremlin"
        else:
            self.current_mode = "Soft"
    
    # =========================================================================
    # Manual Controls
    # =========================================================================
    
    def set_mood(self, mood: str, duration: float = 300.0, persist: bool = False):
        """Manually set mood and apply a temporary manual override.

        duration: seconds to keep the manual override active (default 5 minutes)
        """
        if mood in {"behave", "mean", "flirty"}:
            old = self.mood
            self.mood = mood
            # apply manual override so automatic updates don't immediately override this
            self._manual_mood_override = mood
            self._manual_mood_override_expire = time.time() + float(duration)
            self._update_legacy_mode()
            log.info(f"Mood manually set: {old} -> {mood} (override for {duration}s)")
            # Persist override in data/config.json if requested
            if persist:
                try:
                    import json
                    config_path = Path("data/config.json")
                    config = {}
                    if config_path.exists():
                        with config_path.open('r', encoding='utf-8') as f:
                            config = json.load(f)
                    config['manual_mood_override'] = {
                        'mood': mood,
                        'expire': None if duration <= 0 else (time.time() + float(duration))
                    }
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    with config_path.open('w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2)
                    log.info('Persisted manual mood override to data/config.json')
                except Exception as e:
                    log.warning(f'Failed to persist manual mood override: {e}')
        else:
            log.warning(f"Invalid mood: {mood}")

    def clear_mood_override(self):
        """Clear any manual mood override immediately."""
        self._manual_mood_override = None
        self._manual_mood_override_expire = 0.0
        log.info("Manual mood override cleared")
        # Also remove persisted override if present
        try:
            import json
            config_path = Path("data/config.json")
            if config_path.exists():
                with config_path.open('r', encoding='utf-8') as f:
                    config = json.load(f)
                if 'manual_mood_override' in config:
                    del config['manual_mood_override']
                    with config_path.open('w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2)
                    log.info('Cleared persisted manual mood override')
        except Exception:
            pass
    
    def set_style(self, style: str):
        """Manually set style"""
        if style in {"chaotic", "sweet", "cold", "silent"}:
            old = self.style
            self.style = style
            self._update_legacy_mode()
            log.info(f"Style manually set: {old} -> {style}")
        else:
            log.warning(f"Invalid style: {style}")
    
    def hide(self):
        """Hide avatar (sleep mode)"""
        self.is_hidden = True
        self.style = "silent"
        self.mood = "behave"
        self.current_mode = "Hide"
        log.info("Kitsu hidden")
    
    def unhide(self):
        """Wake from hide"""
        self.is_hidden = False
        self.mood = "behave"
        self.style = "chaotic"
        
        # Recompute from current emotions
        dominant = self.get_current_emotion()
        self.update_personality(dominant)
        self._update_legacy_mode()
        log.info("Kitsu unhidden")
    
    # =========================================================================
    # Tick Loop (Decay & Updates)
    # =========================================================================
    
    async def tick(self):
        """Single tick of emotion processing"""
        now = time.time()
        
        
        # In turn-based decay, emotions should only disappear when tick() is called
        self.stack = [
            emo for emo in self.stack
            if emo.get("expire", 0) >= now and emo.get("intensity", 0.0) > 0]


        # Turn-based decay — called only when tick() is externally triggered
        for emo in self.stack:
            drift = random.uniform(-self.random_drift, self.random_drift)
            old = emo.get("intensity", 0.0)
            new = max(0.0, min(1.0, old - self.decay_rate + drift))
            emo["intensity"] = new
    

        # Update personality from dominant emotion
        if not self.is_hidden:
            dominant = self.get_current_emotion()
            self.update_personality(dominant)
        else:
            self.current_mode = "Hide"
    
    async def run(self):
        """Background loop for emotion decay"""
        log.info("Emotion engine loop started")
        while True:
            # If continuous_decay is disabled, only sleep and do not apply time-based decay here.
            # Decay will be handled by external calls to `tick()` (e.g., on chat events / STT).
            if not self.continuous_decay:
                await asyncio.sleep(1)
                continue

            await asyncio.sleep(1)
            now = time.time()
            
            new_stack: List[Dict[str, Any]] = []
            for emo in self.stack:
                # Skip expired
                if emo.get("expire", 0) < now:
                    continue
                
                # Apply decay
                age = now - emo.get("timestamp", now)
                decay_amount = age * self.decay_rate
                drift = random.uniform(-self.random_drift, self.random_drift)
                
                old_intensity = emo.get("intensity", 0.0)
                new_intensity = max(0.0, min(1.0, old_intensity - decay_amount + drift))
                emo["intensity"] = new_intensity
                
                new_stack.append(emo)
            
            self.stack = new_stack
            
            # Update personality periodically
            if not self.is_hidden:
                dominant = self.get_current_emotion()
                self.update_personality(dominant)
            else:
                self.current_mode = "Hide"
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary"""
        return {
            "mood": self.mood,
            "style": self.style,
            "current_mode": self.current_mode,
            "is_hidden": self.is_hidden,
            "dominant_emotion": self.get_current_emotion(),
            "stack_size": len(self.stack),
            "resistance": self.get_resistance()
        }
    
    def get_avatar_hint(self) -> str:
        """Get avatar animation hint"""
        if self.is_hidden:
            return "hide"
        
        if self.style == "silent":
            return "withdrawn"
        
        # Combine mood + style
        if self.mood == "behave":
            if self.style == "chaotic":
                return "playful_bounce"
            elif self.style == "sweet":
                return "soft_idle"
            elif self.style == "cold":
                return "polite_distance"
        
        elif self.mood == "mean":
            if self.style == "chaotic":
                return "energetic_smirk"
            elif self.style == "cold":
                return "cold_stare"
            elif self.style == "sweet":
                return "fake_sweet"
        
        elif self.mood == "flirty":
            if self.style == "chaotic":
                return "playful_wink"
            elif self.style == "sweet":
                return "affectionate_smile"
            elif self.style == "cold":
                return "seductive_gaze"
        
        return "idle"

    # ---------------------------------------------------------------------
    def get_resistance(self) -> float:
        """
        Compute 0.0–1.0 resistance score based on stack intensity.

        - Resistance increases when high-intensity emotions are added
        - Resistance does NOT decay automatically unless tick() is called
        - If stack intensity is 0, resistance = 0 (easy to change modes, but we preserve current mode)
        """
        now = time.time()
        active = [emo for emo in self.stack if emo.get("expire", 0) >= now]

        if not active:
            return 0.0

        total_intensity = sum(emo.get("intensity", 0.0) for emo in active)
        avg_intensity = total_intensity / max(1, len(active))
        return max(0.0, min(1.0, avg_intensity))


    def apply_resistance(self, level: float = 1.0, duration: float = 30.0):
        """Apply a temporary resistance by pushing a high-intensity 'angry' emotion.

        level: 0.0-1.0 intensity
        duration: seconds
        """
        # push an 'angry' sentinel that increases resistance
        now = time.time()
        self.stack.append({
            "name": "angry",
            "intensity": max(0.0, min(1.0, float(level))),
            "timestamp": now,
            "expire": now + float(duration)
        })
        log.debug(f"Resistance applied: level={level} duration={duration}s")