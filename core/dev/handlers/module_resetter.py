# File: core/dev_console/module_resetter.py
# -----------------------------------------------------------------------------
"""Utility to hot-reload or reset modules by name. Works best when modules are
provided as a dict{name: object} and those objects expose `reload()` or `reset()`.
"""

import importlib
import sys

class ModuleResetter:
    """Reset/reload Python modules without restarting"""

    def __init__(self, kitsu_core=None, modules=None, logger=None):
        self.core = kitsu_core
        self.logger = logger
        self.modules = modules or {}
    
    def reset_module(self, module_name: str) -> str:
        """Reload a Python module"""
        if not module_name:
            return "❌ Module name required"
        
        try:
            # Find module
            if module_name in sys.modules:
                module = sys.modules[module_name]
                importlib.reload(module)
                return f"✅ Reloaded: {module_name}"
            else:
                return f"❌ Module not found: {module_name}"
        except Exception as e:
            return f"❌ Failed to reload: {e}"
    
    def list_modules(self) -> str:
        """List all loaded modules"""
        core_modules = [m for m in sys.modules.keys() if m.startswith("core.")]
        return f"📦 Loaded modules:\n" + "\n".join(f"  - {m}" for m in core_modules)