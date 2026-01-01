"""
Appearance helpers - ensure presentation (skins/avatars) is a UI layer only.
Provides a safe API to change avatar/skin without touching personality or memory.
"""
from typing import Optional, Any, Dict

VALID_UI_SKINS = {"chibi", "live2d", "3d", "default"}


def set_avatar_skin(skin: str, kitsu_self: Optional[Any] = None, memory: Optional[Any] = None) -> Dict[str, str]:
    """Set a UI skin and return a small descriptor.

    Important: This function MUST NOT modify `kitsu_self` or `memory`.
    It exists as a single place to document and enforce the UI-only policy.
    """
    s = skin.lower().strip()
    if s not in VALID_UI_SKINS:
        s = "default"

    # Return descriptor for UI layer to consume; do NOT mutate personality/memory
    return {"skin": s, "note": "UI-only skin applied"}
