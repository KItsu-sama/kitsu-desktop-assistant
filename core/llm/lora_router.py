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
    """Manage style -> LoRA adapter registrations and lightweight switching."""

    def __init__(self, default_style: str = "chaotic"):
        self.mapping: Dict[str, str] = {}
        self.default_style = default_style
        self.active_style = default_style

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
        """Set active style (selection only). This does not mutate models.

        Actual application to a model should be done via apply_to_peft_model()
        or via runtime-specific wiring. This separation keeps switching cheap.
        """
        self.active_style = style

    def apply_to_peft_model(self, base_model, style: Optional[str] = None):
        """Apply the PEFT adapter for style to an in-memory HF model.

        Returns the wrapped model with LoRA weights applied on success, or the
        original base_model on failure. This operation is designed to be cheap
        compared to reloading full base weights.
        """
        try:
            from peft import PeftModel
        except Exception:
            log.debug("peft not available; cannot apply LoRA in-process")
            return base_model

        lora_path = self.get_for_style(style)
        if not lora_path:
            log.debug("No LoRA registered for style; skipping apply")
            return base_model

        try:
            wrapped = PeftModel.from_pretrained(base_model, lora_path)
            log.info(f"Applied LoRA adapter {lora_path} to model for style {style}")
            return wrapped
        except Exception as e:
            log.warning(f"Failed to apply LoRA adapter {lora_path}: {e}")
            return base_model
