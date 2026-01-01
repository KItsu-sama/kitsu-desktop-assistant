"""LoRA manager and meta-controller

Provides a tiny, safe manager for discovering LoRA adapters, selecting
styles based on emotional state, switching active adapters, and
persisting the chosen style to data/config.json.

This module is intentionally small, defensive and side-effect free so
it can be safely imported during tests and runtime.
"""
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Any, List, Tuple


from core.llm.lora_registry import LoRARegistry

log = logging.getLogger(__name__)


class LoRAManager:
    """Discover and manage LoRA adapters through a registry.

    - Enforces strict separation of adapter types: base | style | emotion
    - Ensures only one base adapter active at a time
    - Switches are logged, counted, and reversible via `revert_last_switch()`
    """

    def __init__(self, adapters_dir: Path = Path("data/lora"), llm_interface: Optional[Any] = None, config: Optional[dict] = None):
        self.registry = LoRARegistry(adapters_dir)
        self.current_stack: List[str] = []
        self.switch_count: int = 0
        self.llm_interface = llm_interface
        self._history: List[List[str]] = []
        self.config = config or {}

    def discover_adapters(self) -> None:
        """Populate registry and register adapters with LLM router."""
        try:
            self.registry.discover()
            # validate base uniqueness
            try:
                self.registry.ensure_single_base()
            except Exception as e:
                log.warning("LoRA registry base check failed: %s", e)

            # Register with llm_interface's router if available
            try:
                if self.llm_interface and hasattr(self.llm_interface, 'register_lora'):
                    for adapter in self.registry.list_all():
                        self.llm_interface.register_lora(adapter.name, str(adapter.path))
            except Exception:
                pass

            log.info("Discovered %d LoRA adapters: %s", len(self.registry.list_all()), list(self.registry.to_simple_dict().keys()))
        except Exception as e:
            log.exception("Failed to discover adapters: %s", e)

    def select_for_emotion(self, emotion_state: dict) -> List[str]:
        """Return an ordered stack (base -> style(s) -> emotion(s)).

        Deterministic selection using registry priorities. If explicit style
        provided in emotion_state and available, it will be used as the style
        layer.
        """
        try:
            stack: List[str] = []

            # Always include base if present
            bases = self.registry.list_by_type('base')
            if bases:
                # choose highest priority (lowest priority number)
                base_choice = bases[0]
                stack.append(base_choice.name)

            # style selection
            requested = (emotion_state.get('style') or '').strip()
            styles = []
            if requested and self.registry.get(requested) and self.registry.get(requested).type == 'style':
                styles.append(requested)
            else:
                # fallback mapping based on mood
                mood = (emotion_state.get('dominant_emotion') or '').lower()
                if mood in ('playful', 'happy'):
                    for candidate in ('chaotic', 'sweet'):
                        if self.registry.get(candidate):
                            styles.append(candidate)
                            break
                if mood in ('hurt', 'angry') and self.registry.get('cold'):
                    styles.append('cold')

            # emotion selection
            emotions = []
            if self.registry.get((emotion_state.get('emotion') or '').strip()):
                emotions.append(emotion_state.get('emotion'))

            # assemble stack: base -> styles -> emotions
            for s in styles + emotions:
                if s not in stack:
                    stack.append(s)

                if not stack and self.registry.has_base():
                    base = self.registry.list_by_type("base")[0]
                    return [base.name]


            return stack
        except Exception as e:
            log.debug("select_for_emotion error: %s", e)
            return []

    def validate_stack(self, stack: list[str]) -> Tuple[bool, str]:
        """Validate a proposed LoRA stack without modifying state.

        Returns (True, "") if valid, otherwise (False, reason).
        Rules:
            - Every adapter name must exist in the registry
            - No more than one base adapter in the requested stack
            - Adapter types must be one of the valid types
        """
        try:
            if isinstance(stack, str):
                stack = [stack]

            # Check presence
            for name in stack:
                if not self.get(name):
                    return False, f"adapter_not_found:{name}"

            # Count bases
            bases_in_stack = [name for name in stack if self.get(name).type == 'base']
            if len(bases_in_stack) > 1:
                return False, "multiple_bases_in_stack"

            # Types validated by registry on discovery; additional checks can go here
            return True, ""
        except Exception as e:
            log.debug("validate_stack error: %s", e)
            return False, str(e)

    def switch_adapter(self, styles, force: bool = False) -> Optional[bool]:
        """Switch to a new stack of adapters.

        `styles` may be a single name or an ordered list. This will not unload
        base adapters; base must remain if present.
        """
        try:
            if isinstance(styles, str):
                styles = [styles]

            # Validate presence and types
            for name in styles:
                adapter = self.registry.get(name)
                if not adapter:
                    log.debug("Requested adapter not available: %s", name)
                    return False

            # Ensure base is always present if registry has a base
            if self.registry.has_base():
                bases = [a.name for a in self.registry.list_by_type('base')]
                if bases and bases[0] not in styles:
                    # prepend base
                    styles = [bases[0]] + styles

            # No-op check
            if styles == self.current_stack and not force:
                return None

            # Save for revert capability
            self._history.append(list(self.current_stack))

            self.current_stack = list(styles)
            self.switch_count += 1

            # Inform LLM router if available
            try:
                if self.llm_interface and hasattr(self.llm_interface, 'lora_router'):
                    self.llm_interface.lora_router.set_active_stack(self.current_stack)
            except Exception:
                pass

            self.persist_config()
            log.info("Switched LoRA stack to: %s", self.current_stack)
            return True
        except Exception as e:
            log.exception("Failed to switch adapter: %s", e)
            return False

    def revert_last_switch(self) -> bool:
        """Revert to previous stack if available."""
        try:
            if not self._history:
                return False
            prev = self._history.pop()
            self.current_stack = prev
            try:
                if self.llm_interface and hasattr(self.llm_interface, 'lora_router'):
                    self.llm_interface.lora_router.set_active_stack(self.current_stack)
            except Exception:
                pass
            self.persist_config()
            log.info("Reverted LoRA stack to: %s", self.current_stack)
            return True
        except Exception as e:
            log.exception("Failed to revert LoRA stack: %s", e)
            return False

    def get_stats(self) -> dict:
        try:
            all_adapters = self.registry.list_all()
            available_styles = [a.name for a in all_adapters]
            current_stack = list(self.current_stack) if isinstance(self.current_stack, (list, tuple)) else []
            current_adapter = current_stack[-1] if current_stack else None

            return {
                "total_adapters": len(all_adapters),
                "available_styles": available_styles,
                "switch_count": self.switch_count,
                "current_stack": current_stack,
                "current_adapter": current_adapter,
            }
        except Exception as e:
            log.debug(f"get_stats failed: {e}")
            return {
                "total_adapters": 0,
                "available_styles": [],
                "switch_count": self.switch_count,
                "current_stack": [],
                "current_adapter": None,
            }

    def persist_config(self) -> None:
        """Write data/config.json under {"model": {"stack": current_stack}}"""
        try:
            cfg = {"model": {"stack": self.current_stack}}
            target = Path("data/config.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            tmp.replace(target)
        except Exception as e:
            log.exception("Failed to persist config: %s", e)

    def load_config(self) -> None:
        """Restore current_stack from data/config.json if present."""
        try:
            target = Path("data/config.json")
            if not target.exists():
                return
            with open(target, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                model_entry = cfg.get("model", {})
                stack = None
                # Support legacy format where model is a string, or new format where model is a dict with 'stack'
                if isinstance(model_entry, dict):
                    stack = model_entry.get("stack")
                elif isinstance(model_entry, str):
                    stack = [model_entry]
                if stack:
                    if isinstance(stack, str):
                        stack = [stack]
                    # only restore styles that are still available
                    stack = [s for s in stack if self.registry.get(s)]
                    if stack:
                        self.current_stack = stack
                        try:
                            if self.llm_interface and hasattr(self.llm_interface, 'lora_router'):
                                self.llm_interface.lora_router.set_active_stack(self.current_stack)
                        except Exception:
                            pass
        except Exception as e:
            log.exception("Failed to load LoRA config: %s", e)

    def get_stats(self) -> dict:
        try:
            all_adapters = self.registry.list_all()
            available_styles = [a.name for a in all_adapters]
            current_stack = list(self.current_stack) if isinstance(self.current_stack, (list, tuple)) else []
            current_adapter = current_stack[-1] if current_stack else None

            return {
                "total_adapters": len(all_adapters),
                "available_styles": available_styles,
                "switch_count": self.switch_count,
                "current_stack": current_stack,
                "current_adapter": current_adapter,
            }
        except Exception as e:
            log.debug(f"get_stats failed: {e}")
            return {
                "total_adapters": 0,
                "available_styles": [],
                "switch_count": self.switch_count,
                "current_stack": [],
                "current_adapter": None,
            }

    def persist_config(self) -> None:
        """Write data/config.json under {"model": {"stack": current_stack}} atomically."""
        try:
            cfg = {"model": {"stack": self.current_stack}}
            target = Path("data/config.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            tmp.replace(target)
        except Exception as e:
            log.exception("Failed to persist config: %s", e)

    def load_config(self) -> None:
        """Restore current_stack from data/config.json if present."""
        try:
            target = Path("data/config.json")
            if not target.exists():
                return
            with open(target, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                model_entry = cfg.get("model", {})
                stack = None
                # Support legacy format where model is a string, or new format where model is a dict with 'stack'
                if isinstance(model_entry, dict):
                    stack = model_entry.get("stack")
                elif isinstance(model_entry, str):
                    stack = [model_entry]
                if stack:
                    if isinstance(stack, str):
                        stack = [stack]
                    # only restore styles that are still available
                    stack = [s for s in stack if self.registry.get(s)]
                    if stack:
                        self.current_stack = stack
                        # inform router
                        try:
                            if self.llm_interface and hasattr(self.llm_interface, 'lora_router'):
                                self.llm_interface.lora_router.set_active_stack(self.current_stack)
                        except Exception:
                            pass
        except Exception as e:
            log.exception("Failed to load LoRA config: %s", e)