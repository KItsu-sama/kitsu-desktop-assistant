# File: core/dev_console/settings.py
# -----------------------------------------------------------------------------
"""Settings for dev_console (admin list, toggles).
Replace ADMIN_USERS with your real admin names or check memory-based auth.
"""


ADMIN_USERS = ["Zino", "Alice", "Bob"]  # Replace with real admin usernames


# Toggle which commands are enabled (useful for production)
ENABLED_COMMANDS = {
"train_response": True,
"rate_response": True,
"errors": True,
"stats": True,
"reset_module": True,
"debug": True,
"simulate_error": True,
"export_logs": True,
}


# Limits / configuration
MAX_ERROR_HISTORY = 200
RATING_RANGE = (1, 5)