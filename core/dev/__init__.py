# File: core/dev/__init__.py
"""
Lightweight helpers to expose a ConsoleRouter class for importing by KitsuCore.
Developer tools for Kitsu - commands, debugging, response training
"""


from .console_router import ConsoleRouter
from .settings import ADMIN_USERS


__all__ = ["ConsoleRouter", "ADMIN_USERS"]
