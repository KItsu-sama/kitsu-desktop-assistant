"""
State encoder for failure/error states. Encodes error metadata without
including raw prompt text (only a short stable hash is stored).
"""
import hashlib
from typing import Dict, Any, Optional


class StateEncoder:
    @staticmethod
    def hash_prompt(prompt: Optional[str]) -> Optional[str]:
        if prompt is None:
            return None
        h = hashlib.sha256(prompt.encode("utf-8")[:1024]).hexdigest()
        # return a short digest to avoid storing raw text
        return h[:16]

    @staticmethod
    def encode_failure_state(error_type: str, emotion: str, lora_stack: list, prompt_profile: Optional[str]) -> Dict[str, Any]:
        return {
            "error_type": error_type,
            "emotion": emotion,
            "lora_stack": list(lora_stack or []),
            "prompt_profile_hash": StateEncoder.hash_prompt(prompt_profile),
        }
