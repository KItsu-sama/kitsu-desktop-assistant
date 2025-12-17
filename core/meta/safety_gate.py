from .response_modes import ResponseMode

def safety_check(meta_input):
    # Example rules (you control these)
    if not meta_input.user_permissions.get("allow_nsfw", False):
        if meta_input.nsfw_level > 0:
            return True, ResponseMode.PLAYFUL  # redirect, not block

    if meta_input.intent == "dangerous":
        return True, ResponseMode.FALLBACK

    return False, None
