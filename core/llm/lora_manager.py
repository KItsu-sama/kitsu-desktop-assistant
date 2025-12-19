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
from typing import Dict, Optional, Any

log = logging.getLogger(__name__)


class LoRAManager:
    """Discover LoRA adapters and perform safe, emotion-driven switching.

    Public attributes:
        adapters: Dict[str, Path]
        current_style: Optional[str]
        switch_count: int
    """

    def __init__(self, adapters_dir: Path = Path("data/lora"), llm_interface: Optional[Any] = None, config: Optional[dict] = None):
        self.adapters_dir = Path(adapters_dir)
        self.adapters: Dict[str, Path] = {}
        self.current_style: Optional[str] = None
        self.switch_count: int = 0
        self.llm_interface = llm_interface
        self.config = config or {}

    def discover_adapters(self) -> None:
        """Scan adapters_dir for subdirectories or metadata.json files.

        If metadata.json contains {"style": "<name>"} use that; otherwise
        use the directory name.
        """
        try:
            self.adapters = {}
            if not self.adapters_dir.exists():
                log.debug("LoRA adapters dir not found: %s", self.adapters_dir)
                return

            for child in self.adapters_dir.iterdir():
                try:
                    if child.is_dir():
                        style = child.name
                        meta = child / "metadata.json"
                        if meta.exists():
                            try:
                                with open(meta, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                    style = data.get("style", style)
                            except Exception:
                                log.debug("Failed to read metadata for %s", child)

                        self.adapters[str(style)] = child

                    elif child.is_file() and child.name == "metadata.json":
                        try:
                            with open(child, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                style = data.get("style")
                                if style:
                                    self.adapters[str(style)] = child.parent
                        except Exception:
                            log.debug("Failed to read top-level metadata: %s", child)
                except Exception:
                    # skip problematic entries
                    continue

            log.info("Discovered %d LoRA adapters: %s", len(self.adapters), list(self.adapters.keys()))
        except Exception as e:
            log.exception("Failed to discover adapters: %s", e)

    def select_for_emotion(self, emotion_state: dict) -> Optional[str]:
        """Return a safe mapping from emotion_state to a style name.

        Uses emotion_state.get('style') as primary mapping; otherwise falls
        back to rules based on dominant_emotion.
        """
        try:
            style = emotion_state.get("style")
            if style and style in self.adapters:
                return style

            dominant = (emotion_state.get("dominant_emotion") or "").lower()

            if dominant in ("playful", "happy"):
                for candidate in ("chaotic", "sweet"):
                    if candidate in self.adapters:
                        return candidate

            if dominant in ("hurt", "angry"):
                if "cold" in self.adapters:
                    return "cold"

            if dominant in ("tired", "silent"):
                if "silent" in self.adapters:
                    return "silent"

            return self.current_style
        except Exception as e:
            log.debug("select_for_emotion error: %s", e)
            return None

    def switch_adapter(self, style: str, force: bool = False) -> Optional[bool]:
        """Switch to the named style.

        Returns False if style not found; None if no-op; True if switched.
        """
        try:
            if style not in self.adapters:
                log.debug("Requested style not available: %s", style)
                return False

            if style == self.current_style and not force:
                return None

            self.current_style = style
            self.switch_count += 1
            self.persist_config()
            log.info("Switched LoRA style to: %s", style)
            return True
        except Exception as e:
            log.exception("Failed to switch adapter: %s", e)
            return False

    def get_stats(self) -> dict:
        return {
            "total_adapters": len(self.adapters),
            "available_styles": list(self.adapters.keys()),
            "switch_count": self.switch_count,
            "current_style": self.current_style,
        }

    def persist_config(self) -> None:
        """Write data/config.json under {"model": {"style": current_style}} atomically."""
        try:
            cfg = {"model": {"style": self.current_style}}
            target = Path("data/config.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            tmp.replace(target)
        except Exception as e:
            log.exception("Failed to persist config: %s", e)

    def load_config(self) -> None:
        """Restore current_style from data/config.json if present."""
        try:
            target = Path("data/config.json")
            if not target.exists():
                return
            with open(target, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                style = cfg.get("model", {}).get("style")
                if style and style in self.adapters:
                    self.current_style = style
        except Exception as e:
            log.exception("Failed to load LoRA config: %s", e)