from enum import Enum

class ResponseMode(Enum):
    FAST_CHAT = "fast_chat"
    PLAYFUL = "playful"
    FLIRTY = "flirty"
    SARCASTIC = "sarcastic"
    REASONED = "reasoned"
    REFLECTIVE = "reflective"
    MEMORY_HEAVY = "memory_heavy"
    TOOL_USE = "tool_use"
    FALLBACK = "fallback"
