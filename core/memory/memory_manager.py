"""
Integrated Memory System
Your plugin-based memory system adapted for the new modular architecture

File location: core/memory/memory_manager.py
"""

import json
import random
import logging
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Protocol
from core.memory.scoring import compute_score
from difflib import SequenceMatcher

log = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class MemoryConfig:
    """Memory system configuration"""
    
    def __init__(
        self,
        max_history: int = 200,
        auto_save: bool = True,
        save_interval: int = 300,
        compression_enabled: bool = True
    ):
        self.max_history = max_history
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.compression_enabled = compression_enabled


# =============================================================================
# Plugin Interface (Your Design - Preserved)
# =============================================================================

class MemoryPlugin(Protocol):
    """Interface for memory plugins"""
    
    @property
    def priority(self) -> int:
        """Execution order priority (lower = earlier)"""
        return 0
    
    def on_remember(
        self,
        role: str,
        text: str,
        emotion: Optional[str] = None
    ) -> Optional[str]:
        """Modify or filter incoming memory"""
        pass
    
    def on_recall(
        self,
        sessions: List[Dict[str, Any]],
        context_length: int
    ) -> List[Dict[str, Any]]:
        """Filter or reorder sessions on recall"""
        pass
    
    def on_save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add plugin data before save"""
        pass
    
    def on_load(self, data: Dict[str, Any]) -> None:
        """Restore plugin data after load"""
        pass
    
    def on_optimize(
        self,
        sessions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Refine sessions during optimization"""
        pass


# =============================================================================
# Core Plugins (Your Implementations)
# =============================================================================

class PlayfulForgetfulnessPlugin:
    """
    Kitsu's playful forgetfulness behavior
    Sometimes 'forgets' details in a cute way
    """
    
    def __init__(self, kitsu_self: Optional[Any] = None):
        self.kitsu_self = kitsu_self
        self.priority = 10  # High priority
    
    def on_remember(
        self,
        role: str,
        text: str,
        emotion: Optional[str] = None
    ) -> Optional[str]:
        """Occasionally replace memory with playful forgetfulness"""
        if random.random() < 0.05:  # 5% chance
            if self.kitsu_self:
                playful_level = getattr(self.kitsu_self, 'playfulness', 0.5)
                if playful_level > 0.7:
                    return "[got distracted by something shiny]"
                else:
                    return "[forgot a detail]"
        return text
    
    def on_recall(
        self,
        sessions: List[Dict[str, Any]],
        context_length: int
    ) -> List[Dict[str, Any]]:
        return sessions
    
    def on_save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data
    
    def on_load(self, data: Dict[str, Any]) -> None:
        pass
    
    def on_optimize(
        self,
        sessions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return sessions


class EmotionalMemoryPlugin:
    """
    Prioritize emotionally significant memories
    Strong emotions are more likely to be recalled
    """
    
    def __init__(self):
        self.emotional_weights = {
            "joy": 1.5,
            "anger": 1.8,
            "surprise": 1.3,
            "sadness": 1.4,
            "fear": 1.2,
            "love": 1.7,
            "embarrassed": 1.4,
            "neutral": 1.0
        }
        self.priority = 20
    
    def on_remember(
        self,
        role: str,
        text: str,
        emotion: Optional[str] = None
    ) -> Optional[str]:
        return text
    
    def on_recall(
        self,
        sessions: List[Dict[str, Any]],
        context_length: int
    ) -> List[Dict[str, Any]]:
        """Prioritize emotional memories in recall"""
        if len(sessions) <= context_length:
            return sessions
        
        # Keep recent memories + most emotional older ones
        recent = sessions[-(context_length // 2):]
        older = sessions[:-(context_length // 2)]
        
        # Sort older by emotional weight
        def emotion_score(session):
            emotion = session.get("emotion", "neutral")
            return self.emotional_weights.get(emotion, 1.0)
        
        older_sorted = sorted(older, key=emotion_score, reverse=True)
        remaining_slots = context_length - len(recent)
        
        return older_sorted[:remaining_slots] + recent
    
    def on_save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data
    
    def on_load(self, data: Dict[str, Any]) -> None:
        pass
    
    def on_optimize(
        self,
        sessions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Preserve emotional memories during optimization"""
        preserved = []
        compressed = []
        threshold = 1.2
        
        for session in sessions:
            emotion = session.get("emotion", "neutral")
            weight = self.emotional_weights.get(emotion, 1.0)
            
            if weight > threshold:
                preserved.append(session)
            else:
                compressed.append(session)
        
        # Add summary for compressed memories
        if compressed:
            preserved.append({
                "role": "system",
                "text": f"[compressed {len(compressed)} neutral memories]",
                "emotion": "neutral",
                "compressed_count": len(compressed)
            })
        
        return preserved


class SleepOptimizationPlugin:
    """
    Compress neutral memories during idle/sleep
    Saves context space for important memories
    """
    
    def __init__(self):
        self.compression_threshold = 50
        self.priority = 30
    
    def on_remember(
        self,
        role: str,
        text: str,
        emotion: Optional[str] = None
    ) -> Optional[str]:
        return text
    
    def _is_important(self, session):
        return session.get("score", 0) > 0.4

    def on_recall(
        self,
        sessions: List[Dict[str, Any]],
        context_length: int
    ) -> List[Dict[str, Any]]:
        return sessions
    
    def on_save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data
    
    def on_load(self, data: Dict[str, Any]) -> None:
        pass
    
    def on_optimize(
        self,
        sessions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compress neutral memories during optimization
        Also forget very low-value short-term memories
        """

        # 🔽 STEP 1: Soft-forget unimportant short memories
        sessions = [
            s for s in sessions
            if not (
                s.get("score", 0) < 0.15 and
                s.get("type") == "SHORT"
            )
        ]

        # 🔽 STEP 2: Skip compression if too few memories remain
        if len(sessions) < self.compression_threshold:
            return sessions

        # 🔽 STEP 3: Compression
        compressed = []
        current_group = []
        group_emotions = []

        for session in sessions:
            if self._is_important(session):
                # Important memory — flush group first
                if current_group:
                    compressed.append(
                        self._create_summary_entry(current_group, group_emotions)
                    )
                    current_group = []
                    group_emotions = []

                compressed.append(session)

            else:
                # Neutral memory — group for compression
                current_group.append(session)
                group_emotions.append(session.get("emotion", "neutral"))

        # Flush remaining group
        if current_group:
            compressed.append(
                self._create_summary_entry(current_group, group_emotions)
            )

        return compressed
    
    def _create_summary_entry(self, sessions, emotions):
        dominant_emotion = (
            max(set(emotions), key=emotions.count)
            if emotions else "neutral"
        )

        return {
            "role": "system",
            "text": f"[Compressed {len(sessions)} routine interactions — {dominant_emotion}]",
            "emotion": dominant_emotion,
            "timestamp": time.time(),
            "type": "EPISODIC",
            "uses": 0,
            "score": 0.45,
            "compressed_count": len(sessions),
        }


# =============================================================================
# Main Memory Manager
# =============================================================================

class MemoryManager:
    """
    Main memory management system with plugin support
    
    Features:
    - Plugin architecture for extensibility
    - Emotional memory prioritization
    - Sleep/idle optimization
    - Playful forgetfulness
    - Thread-safe operations
    - Auto-save
    """
    
    def __init__(
        self,
        kitsu_self: Optional[Any] = None,
        memory_path: Optional[Path] = None,
        config: Optional[MemoryConfig] = None
    ):
        self.kitsu_self = kitsu_self
        self.memory_path = memory_path or Path("data/memory/memory.json")
        self.config = config or MemoryConfig()
        
        # Memory storage (deque for efficient FIFO)
        self.sessions = deque(maxlen=self.config.max_history)
        self.state: Dict[str, Any] = {}
        
        # Plugin system
        self.plugins: List[MemoryPlugin] = []
        
        # Thread safety
        self._lock = threading.RLock()
        self._last_save_time = time.time()
        
        # Initialize core plugins
        self._init_core_plugins()
        
        # Load existing memory
        self.load()
        
        log.info(f"MemoryManager initialized ({len(self.sessions)} memories loaded)")
    
    def _init_core_plugins(self):
        """Initialize core memory plugins"""
        self.add_plugin(PlayfulForgetfulnessPlugin(self.kitsu_self))
        self.add_plugin(EmotionalMemoryPlugin())
        self.add_plugin(SleepOptimizationPlugin())
    
    # =========================================================================
    # Plugin Management
    # =========================================================================
    
    def add_plugin(self, plugin: MemoryPlugin):
        """Add a memory plugin"""
        with self._lock:
            self.plugins.append(plugin)
            # Sort by priority
            self.plugins.sort(key=lambda p: getattr(p, 'priority', 0))
            log.info(f"Added memory plugin: {plugin.__class__.__name__}")

    def _mark_dirty(self):
        """Mark the manager as needing save; triggers save if auto_save set."""
        # For simplicity: if autosave is enabled, call save(); otherwise do nothing yet
        if getattr(self.config, 'auto_save', False):
            try:
                self.save()
            except Exception:
                log.exception("Failed to autosave memory on mark dirty")
    
    def remove_plugin(self, plugin_class):
        """Remove a plugin by class"""
        with self._lock:
            self.plugins = [
                p for p in self.plugins
                if not isinstance(p, plugin_class)
            ]
            log.info(f"Removed memory plugin: {plugin_class.__name__}")
    
    def has_plugin(self, plugin_class) -> bool:
        """Check if plugin is loaded"""
        with self._lock:
            return any(isinstance(p, plugin_class) for p in self.plugins)
    
    # =========================================================================
    # Core Memory APIs
    # =========================================================================
    
    def remember(
        self,
        role: str,
        text: str,
        emotion: Optional[str] = None
    ):
        """
        Store a new memory
        
        Args:
            role: Who said it (user, kitsu, system)
            text: What was said
            emotion: Associated emotion (optional)
        """
        with self._lock:

            # Add emotional pressure
            if self.kitsu_self:
                current_emotion = getattr(self.kitsu_self, "emotion", None)
                if current_emotion and emotion == current_emotion:
                    entry["score"] += 0.1

            # Validation
            if not text or not text.strip():
                log.warning("Attempted to remember empty text")
                return
            
            # Normalize role
            if role not in ["user", "kitsu", "self", "assistant", "system"]:
                log.warning(f"Invalid role: {role}, using 'system'")
                role = "system"
            
            original_text = text
            
            # Process through plugins (in priority order)
            for plugin in self.plugins:
                modified_text = plugin.on_remember(role, text, emotion)
                if modified_text is not None:
                    text = modified_text
            
            # Create memory entry
            entry = {
                "role": role,
                "text": text,
                "emotion": emotion or "neutral",
                "timestamp": time.time(),


                "score": 0.0,
                "uses": 0,
                "type": "SHORT"   # SHORT | EPISODIC | LONG
            }

            entry["score"] = compute_score(entry, time.time())

            # Track if modified by plugins
            if text != original_text:
                entry["original_text"] = original_text
                entry["modified_by"] = "plugins"
            
            # Add to memory
            self.sessions.append(entry)
            
            # Auto-save check
            if self.config.auto_save:
                elapsed = time.time() - self._last_save_time
                if elapsed > self.config.save_interval:
                    self.save()
    

    def recall(self, context_length: int = 5) -> List[Dict[str, Any]]:
        now = time.time()

        with self._lock:
            sessions = list(self.sessions)

            # Update scores
            for m in sessions:
                m["score"] = compute_score(m, now)

            # Sort by importance
            sessions.sort(key=lambda m: m.get("score", 0), reverse=True)

            if context_length > 0:
                sessions = sessions[:context_length]

            # Plugins (post-ranking)
            for plugin in self.plugins:
                sessions = plugin.on_recall(sessions, context_length)

            # Reinforce usage
            for m in sessions:
                m["uses"] = m.get("uses", 0) + 1
                self._promote_memory(m)

            return sessions

    
    def format_context(self, context_length: int = 5) -> str:
        """
        Format memories as conversation context
        
        Args:
            context_length: Number of memories to include
            
        Returns:
            Formatted string for LLM prompt
        """
        recent = self.recall(context_length)
        if not recent:
            return ""
        
        formatted = []
        for entry in recent:
            role = entry.get("role", "unknown")
            
            # Map internal roles to display names
            display_role = {
                "kitsu": "Kitsu",
                "self": "Kitsu",
                "assistant": "Kitsu",
                "user": "User",
                "system": "System"
            }.get(role, role.capitalize())
            
            text = entry.get("text", "")
            emotion = entry.get("emotion")
            
            # Include emotion if present and not neutral
            if emotion and emotion != "neutral":
                formatted.append(f"{display_role} ({emotion}): {text}")
            else:
                formatted.append(f"{display_role}: {text}")
        
        return "\n".join(formatted)
    
    def optimize_memory(self):
        """
        Optimize memory by compressing neutral entries
        Preserves emotional and important memories
        """
        with self._lock:
            sessions = list(self.sessions)
            
            # Process through plugins
            for plugin in self.plugins:
                sessions = plugin.on_optimize(sessions)
            
            # Update deque
            self.sessions = deque(
                sessions[-self.config.max_history:],
                maxlen=self.config.max_history
            )
        for m in sessions:
            if m.get("uses", 0) == 0 and m.get("score", 0) < 0.25:
                m["score"] *= 0.85
        emotion_counts = {}
        for m in sessions:
            emotion_counts[m["emotion"]] = emotion_counts.get(m["emotion"], 0) + 1
        for m in sessions:
            if emotion_counts.get(m["emotion"], 0) > 20:
                m["score"] *= 0.9

            
            log.info(f"Memory optimized: {len(self.sessions)} sessions remaining")
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save(self):
        """Save memory to disk"""
        with self._lock:
            try:
                data = {
                    "sessions": list(self.sessions),
                    "state": self.state,
                    "metadata": {
                        "version": "1.1",
                        "saved_at": time.time(),
                        "total_entries": len(self.sessions)
                    }
                }
                
                # Process through plugins
                for plugin in self.plugins:
                    data = plugin.on_save(data)
                
                # Ensure directory exists
                self.memory_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Atomic write (write to temp, then rename)
                temp_path = self.memory_path.with_suffix('.tmp')
                temp_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                temp_path.replace(self.memory_path)
                
                self._last_save_time = time.time()
                log.info(f"Memory saved: {len(self.sessions)} entries")
                
            except Exception as e:
                log.error(f"Failed to save memory: {e}")
    
    def load(self):
        """Load memory from disk"""
        with self._lock:
            try:
                if self.memory_path.exists():
                    data = json.loads(
                        self.memory_path.read_text(encoding="utf-8")
                    )
                    
                    self.sessions = deque(
                        data.get("sessions", []),
                        maxlen=self.config.max_history
                    )
                    self.state = data.get("state", {})
                    
                    # Process through plugins
                    for plugin in self.plugins:
                        plugin.on_load(data)
                    
                    log.info(f"Memory loaded: {len(self.sessions)} entries")
                else:
                    log.info("No existing memory file, starting fresh")
                    
            except Exception as e:
                log.error(f"Failed to load memory: {e}")
                self.sessions = deque(maxlen=self.config.max_history)
                self.state = {}
    
    # =========================================================================
    # Advanced Features
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self._lock:
            emotions = {}
            roles = {}
            text_lengths = []
            timestamps = []
            
            for session in self.sessions:
                emotion = session.get("emotion", "neutral")
                role = session.get("role", "unknown")
                text = session.get("text", "")
                
                emotions[emotion] = emotions.get(emotion, 0) + 1
                roles[role] = roles.get(role, 0) + 1
                text_lengths.append(len(text))
                
                if 'timestamp' in session:
                    timestamps.append(session['timestamp'])
            
            avg_length = (
                sum(text_lengths) / len(text_lengths)
                if text_lengths else 0
            )
            
            age_range = (
                (min(timestamps), max(timestamps))
                if timestamps else (0, 0)
            )
            
            memory_json = json.dumps(list(self.sessions), ensure_ascii=False)
            memory_bytes = len(memory_json.encode('utf-8'))
            
            return {
                "total_sessions": len(self.sessions),
                "emotion_distribution": emotions,
                "role_distribution": roles,
                "avg_text_length": avg_length,
                "memory_age_range": age_range,
                "active_plugins": [p.__class__.__name__ for p in self.plugins],
                "memory_usage_bytes": memory_bytes
            }
    
    def search(self, query: str, limit: int = 10, emotion_filter: Optional[str] = None):
        now = time.time()

        with self._lock:
            matches = []
            q = query.lower()

            for i, session in enumerate(self.sessions):
                text = session.get("text", "").lower()
                if q not in text:
                    continue

                if emotion_filter and session.get("emotion") != emotion_filter:
                    continue

                relevance = self._calculate_relevance(session, query)
                memory_score = compute_score(session, now)

                final_score = relevance * 0.7 + memory_score * 0.3

                matches.append({
                    **session,
                    "session_index": i,
                    "relevance_score": relevance,
                    "final_score": final_score
                })

            matches.sort(key=lambda x: x["final_score"], reverse=True)
            return matches[:limit]

    
    def _calculate_relevance(
        self,
        session: Dict[str, Any],
        query: str
    ) -> float:
        """Calculate search relevance score"""
        text = session.get("text", "").lower()
        query_terms = query.lower().split()
        
        score = 0
        for term in query_terms:
            if term in text:
                score += 1
                # Exact word match bonus
                if f" {term} " in f" {text} ":
                    score += 0.5
        
        # Emotional memory boost
        emotion = session.get("emotion", "neutral")
        if emotion != "neutral":
            score *= 1.2
        
        return score
    
    def clear(self):
        """Clear all memories"""
        with self._lock:
            self.sessions.clear()
            self.state = {}
            log.info("Memory cleared")
    
    async def close(self):
        """Cleanup (save and close)"""
        self.save()
        log.info("Memory manager closed")

# =============================================================================
# extra
# =============================================================================

    def set_user_info(self, **updates) -> None:
        """Update user information safely with nested merge."""
        with self._lock:
            user = self.state.setdefault("user", {})

            for key, value in updates.items():
                # Handle nested dicts
                if isinstance(value, dict) and isinstance(user.get(key), dict):
                    user[key].update(value)
                else:
                    user[key] = value

            self._mark_dirty()   # If use autosave
            # Persist config changes as appropriate
            # Separate permission updates vs profile updates
            profile_updates = {}
            permission_updates = {}
            for key, value in updates.items():
                if key == 'permissions' and isinstance(value, dict):
                    permission_updates.update(value)
                elif key.startswith('permissions.'):
                    # dotted notation for permission e.g. permissions.can_shutdown
                    _, perm_key = key.split('.', 1)
                    permission_updates[perm_key] = value
                else:
                    profile_updates[key] = value

            if profile_updates:
                try:
                    self._update_config_profile(profile_updates)
                except Exception:
                    log.exception('Failed to update config user profile')
            if permission_updates:
                try:
                    self._update_config_permissions(permission_updates)
                except Exception:
                    log.exception('Failed to update config permissions')

    def _promote_memory(self, mem):
        if mem["score"] >= 0.75 and mem["uses"] >= 3:
            mem["type"] = "LONG"
        elif mem["score"] >= 0.45:
            mem["type"] = "EPISODIC"
        if mem["uses"] >= 6 and mem["score"] >= 0.8:
            mem["type"] = "LONG"
        elif mem["uses"] >= 3 and mem["score"] >= 0.45:
            mem["type"] = "EPISODIC"

    def _load_json_safe(self, path: Path) -> dict:
        """Load JSON from file, return {} if missing or invalid."""
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.debug(f"Failed to load JSON from {path}", exc_info=True)
        return {}

    def _save_json_safe(self, path: Path, data: dict) -> None:
        """Save JSON data to a file safely (atomic write)."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix('.tmp')
            temp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            temp_path.replace(path)
        except Exception as e:
            log.error(f"Failed to save JSON to {path}: {e}")

    def get_user_info(self) -> dict:
        """Return merged user info from config and permissions, falling back to defaults."""
        with self._lock:
            # Load defaults
            default_profile = self._load_json_safe(Path("data/default/user_profile.json"))
            default_permissions = self._load_json_safe(Path("data/default/permissions.json")).get("permissions", {})

            # Load config overrides
            profile = self._load_json_safe(Path("data/config/user_profile.json"))
            permissions = self._load_json_safe(Path("data/config/permissions.json")).get("permissions", {})

            # Merge
            merged = default_profile.copy()
            merged.update(profile)

            merged_permissions = default_permissions.copy()
            merged_permissions.update(permissions)
            merged["permissions"] = merged_permissions

            # Also include any runtime overrides stored in memory state
            user_state = self.state.get("user", {})
            # deep merge of nested dicts
            for k, v in user_state.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k].update(v)
                else:
                    merged[k] = v

            return merged

    def _update_config_profile(self, updates: dict) -> None:
        """Apply updates to data/config/user_profile.json (merge) and persist."""
        with self._lock:
            cfg_path = Path("data/config/user_profile.json")
            current = self._load_json_safe(cfg_path)

            # Apply merge
            for key, value in updates.items():
                # support dotted keys e.g. relationship.trust_level
                if "." in key:
                    top, rest = key.split('.', 1)
                    cur_sub = current.setdefault(top, {})
                    if isinstance(cur_sub, dict):
                        # nested assignment
                        # if rest contains dots, create nested dicts
                        sub_ref = cur_sub
                        parts = rest.split('.')
                        for p in parts[:-1]:
                            sub_ref = sub_ref.setdefault(p, {})
                        sub_ref[parts[-1]] = value
                    else:
                        current[top] = {rest: value}
                else:
                    current[key] = value

            # Save back
            self._save_json_safe(cfg_path, current)

    def _update_config_permissions(self, updates: dict) -> None:
        """Apply updates to data/config/permissions.json (merge under 'permissions') and persist."""
        with self._lock:
            cfg_path = Path("data/config/permissions.json")
            current = self._load_json_safe(cfg_path)
            perms = current.get("permissions", {})
            perms.update(updates)
            current["permissions"] = perms
            self._save_json_safe(cfg_path, current)

    def save_user(self) -> None:
        """Persist runtime user state to config files (profile + permissions)."""
        with self._lock:
            user_state = self.state.get("user", {})
            # Split profile and permissions
            profile_updates = {}
            permission_updates = {}

            for k, v in user_state.items():
                if k == 'permissions' and isinstance(v, dict):
                    permission_updates.update(v)
                else:
                    profile_updates[k] = v

            if profile_updates:
                try:
                    # merge into config/user_profile.json
                    cfg_path = Path("data/config/user_profile.json")
                    current = self._load_json_safe(cfg_path)
                    current.update(profile_updates)
                    self._save_json_safe(cfg_path, current)
                except Exception:
                    log.exception("Failed to save user profile to config")

            if permission_updates:
                try:
                    self._update_config_permissions(permission_updates)
                except Exception:
                    log.exception("Failed to save permissions to config")

    def reset_user_info(self, what: Optional[str] = None) -> None:
        """Reset user profile or permissions to defaults.

        what: 'profile'|'permissions'|None (both)
        """
        with self._lock:
            if what in (None, 'profile', 'all'):
                default_profile = self._load_json_safe(Path("data/default/user_profile.json"))
                self._save_json_safe(Path("data/config/user_profile.json"), default_profile)
            if what in (None, 'permissions', 'all'):
                default_perms = self._load_json_safe(Path("data/default/permissions.json"))
                self._save_json_safe(Path("data/config/permissions.json"), default_perms)

            # Clear runtime overrides for user
            if "user" in self.state:
                self.state.pop("user")

            self._mark_dirty()

    def extract_training_data(self, min_score=0.6):
        """
        Extract high-quality (input → response) pairs
        """
        pairs = []

        prev = None
        for m in self.sessions:
            if m.get("score", 0) < min_score:
                prev = m
                continue

            if prev and prev["role"] == "user" and m["role"] in ("kitsu", "assistant"):
                pairs.append({
                    "prompt": prev["text"],
                    "response": m["text"],
                    "emotion": m.get("emotion"),
                    "weight": m["score"]
                })
            prev = m

        return pairs


# =============================================================================
# Async Interface (for compatibility)
# =============================================================================

async def initialize_memory(
    kitsu_self: Optional[Any] = None,
    config: Optional[MemoryConfig] = None
) -> MemoryManager:
    """
    Initialize memory manager (async-friendly)
    """
    memory = MemoryManager(kitsu_self=kitsu_self, config=config)
    return memory