"""LoRA Registry

Single source-of-truth registry for LoRA adapters.

Responsibilities:
- Maintain adapter metadata (name, type, path, priority, weight, notes)
- Validate adapter types and uniqueness constraints (only one base)
- Provide query APIs used by LoRAManager and LoRaRouter

Design notes:
- Lightweight, in-memory registry that reads from on-disk layout (metadata.json)
- No global state mutations outside instance methods
- Strictly uses absolute imports and project-root paths
"""
from pathlib import Path
from typing import Dict, Optional, Any, List
import json
import logging

log = logging.getLogger(__name__)

VALID_TYPES = ("base", "style", "emotion")


class LoRAAdapter:
    """Small data holder for an adapter's metadata."""

    def __init__(self, name: str, path: Path, type_: str = "style", priority: int = 0, weight: float = 1.0, notes: Optional[str] = None):
        if type_ not in VALID_TYPES:
            raise ValueError(f"Invalid adapter type: {type_}")
        self.name = name
        self.path = Path(path)
        self.type = type_
        self.priority = int(priority)
        self.weight = float(weight)
        self.notes = notes or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "type": self.type,
            "priority": self.priority,
            "weight": self.weight,
            "notes": self.notes,
        }


class LoRARegistry:
    """Registry managing adapters and lookup.

    Important: keep operations idempotent and fail-safe for runtime usage.
    """

    def __init__(self, adapters_dir: Path = Path("data/lora")):
        self.adapters_dir = Path(adapters_dir)
        self._adapters: Dict[str, LoRAAdapter] = {}

    def discover(self) -> None:
        """Discover adapters on-disk and populate registry.

        Expects each adapter to live in a subdirectory under adapters_dir and
        optionally include a metadata.json with fields:
            {"name": "...", "type": "base|style|emotion", "priority": 0, "weight": 1.0, "notes": "..."}
        """
        self._adapters = {}
        if not self.adapters_dir.exists():
            log.debug("LoRA adapters dir not found: %s", self.adapters_dir)
            return

        for child in sorted(self.adapters_dir.iterdir()):
            try:
                if not child.is_dir():
                    continue

                name = child.name
                type_ = "style"
                priority = 0
                weight = 1.0
                notes = ""

                meta = child / "metadata.json"
                if meta.exists():
                    try:
                        with open(meta, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            name = data.get("name", name)
                            type_ = data.get("type", type_)
                            priority = data.get("priority", priority)
                            weight = data.get("weight", weight)
                            notes = data.get("notes", notes)
                    except Exception as e:
                        log.debug("Failed to read metadata for %s: %s", child, e)

                # Validate
                if type_ not in VALID_TYPES:
                    log.debug("Ignoring adapter with invalid type %s: %s", type_, child)
                    continue

                adapter = LoRAAdapter(name=name, path=child, type_=type_, priority=priority, weight=weight, notes=notes)
                self._adapters[adapter.name] = adapter

            except Exception:
                continue

        log.info("LoRA registry discovered %d adapters", len(self._adapters))

    def register(self, adapter: LoRAAdapter) -> None:
        """Register an adapter instance explicitly."""
        if adapter.name in self._adapters:
            log.debug("Overwriting adapter in registry: %s", adapter.name)
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> Optional[LoRAAdapter]:
        return self._adapters.get(name)

    def list_all(self) -> List[LoRAAdapter]:
        return list(self._adapters.values())

    def list_by_type(self, type_: str) -> List[LoRAAdapter]:
        if type_ not in VALID_TYPES:
            raise ValueError(f"Invalid adapter type: {type_}")
        return sorted([a for a in self._adapters.values() if a.type == type_], key=lambda x: x.priority)

    def has_base(self) -> bool:
        return any(a.type == "base" for a in self._adapters.values())

    def ensure_single_base(self) -> None:
        bases = [a for a in self._adapters.values() if a.type == "base"]
        if len(bases) > 1:
            names = [b.name for b in bases]
            raise RuntimeError(f"Multiple base adapters found in registry: {names}")

    def to_simple_dict(self) -> Dict[str, Dict[str, Any]]:
        return {name: a.to_dict() for name, a in self._adapters.items()}
