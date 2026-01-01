"""Simple LoRA routing helper.

This module keeps a mapping of style -> LoRA adapter path and provides
utilities to apply a LoRA adapter to an in-process HF model (PEFT).
Design is minimal and non-intrusive: applying adapters is best-effort and
keeps the base model object intact to avoid reloading base weights.
"""
from __future__ import annotations

from typing import Dict, Optional
import logging

log = logging.getLogger(__name__)


class LoRaRouter:
    """Manage style -> LoRA adapter registrations and lightweight switching.

    This router now supports stacking multiple adapters (e.g., base -> mood -> style).
    Application to an in-memory HF model will attempt to apply adapters in-order.
    """

    def __init__(self, default_style: str = "chaotic"):
        self.mapping: Dict[str, str] = {}
        self.default_style = default_style
        self.active_style = default_style
        self.active_stack = [default_style]

    def register(self, style: str, lora_path: str) -> None:
        """Register a LoRA adapter path for a style."""
        self.mapping[style] = lora_path
        log.info(f"Registered LoRA for style={style}: {lora_path}")

    def get_for_style(self, style: Optional[str]) -> Optional[str]:
        """Return LoRA path for style or default."""
        if not style:
            style = self.default_style
        return self.mapping.get(style, self.mapping.get(self.default_style))

    def set_active(self, style: str) -> None:
        """Set active style (single selection) (does not mutate models)."""
        self.active_style = style
        self.active_stack = [style]

    def set_active_stack(self, styles: list) -> None:
        """Set an ordered stack of styles (e.g., ['base','mood','style']).

        The first element is treated as primary for status reporting, but
        all adapters will be attempted when applying to an in-memory model.
        """
        if not styles:
            styles = [self.default_style]
        self.active_stack = styles
        self.active_style = styles[0]

    def apply_to_peft_model(self, base_model, style: Optional[str] = None):
        """Backward compatible: apply a single style or the configured stack.

        If `style` is a list, it will be applied as a stack; otherwise the
        configured stack (self.active_stack) is used.
        """
        # Support list input to maintain compatibility
        if isinstance(style, (list, tuple)):
            return self.apply_stack_to_peft_model(base_model, list(style))

        # If no explicit style provided, apply the active stack
        return self.apply_stack_to_peft_model(base_model, self.active_stack)

    def apply_stack_to_peft_model(self, base_model, styles: list):
        """Sequentially apply multiple LoRA adapters to the base_model.

        Attempts to apply adapters in-order: the resulting wrapped model from
        one adapter is used as the base for the next. If `peft` is not
        available, or an adapter fails, the original base_model is returned
        or the last-successful wrapped model is used.
        """
        try:
            from peft import PeftModel
        except Exception:
            log.debug("peft not available; cannot apply LoRA in-process")
            return base_model

        model = base_model
        for style in styles:
            lora_path = self.get_for_style(style)
            if not lora_path:
                log.debug("No LoRA registered for style %s; skipping", style)
                continue
            try:
                model = PeftModel.from_pretrained(model, lora_path)
                log.info(f"Applied LoRA adapter {lora_path} for style {style}")
            except Exception as e:
                log.warning(f"Failed to apply LoRA adapter {lora_path}: {e}")
                # Continue attempting remaining adapters
                continue

        return model
