# File: core/dev_console/console_router.py
# -----------------------------------------------------------------------------
"""A small command router that maps text commands to handlers.
Intended to be integrated with Kitsu's command parsing (slash or text).
"""


from typing import Optional
from .settings import ADMIN_USERS, ENABLED_COMMANDS
from .handlers.response_trainer import ResponseTrainer
from .handlers.response_rater import ResponseRater
from .handlers.error_viewer import ErrorViewer
from .handlers.stats_viewer import StatsViewer
from .handlers.module_resetter import ModuleResetter
from .handlers.debug_tools import DebugTools

from typing import Optional, Dict, Any, Callable
import asyncio
from .permissions import PermissionManager
from .handlers import (
    ResponseTrainer, ResponseRater, ErrorViewer, 
    StatsViewer, ModuleResetter, DebugTools
)



class ConsoleRouter:

    def __init__(self, memory=None, logger=None, modules=None, admin_check=None):
        """Construct router. Pass memory and module registry for richer commands.


        Args:
        memory: optional memory manager (object with get_user_info etc.)
        logger: optional logger
        modules: optional dict{name: module_obj} to support /reset_module
        admin_check: optional function(user_id) -> bool to replace ADMIN_USERS check
        """
        self.memory = memory
        self.logger = logger
        self.modules = modules or {}
        self.admin_check = admin_check


        # Instantiate handlers with standardized signature
        self.trainer = ResponseTrainer(kitsu_core=None, memory=self.memory, logger=self.logger)
        self.rater = ResponseRater(kitsu_core=None, memory=self.memory, logger=self.logger)
        self.errors = ErrorViewer(kitsu_core=None, log_path=None, logger=self.logger)
        self.stats = StatsViewer(kitsu_core=None, kitsu_state=None, logger=self.logger)
        self.resetter = ModuleResetter(kitsu_core=None, modules=self.modules, logger=self.logger)
        self.debug = DebugTools(kitsu_core=None, logger=self.logger)


    def is_admin(self, user: Optional[str]) -> bool:
        if self.admin_check:
            return self.admin_check(user)
        return user in ADMIN_USERS


    def route(self, command: str, user: Optional[str] = None, **kwargs):
        """Route a textual command to the appropriate handler.
        command should be something like 'train_response', 'rate_response 4', etc.
        Returns a dict {ok: bool, result: str}
        """
        parts = command.strip().split()
        if not parts:
            return {"ok": False, "result": "empty command"}


        name = parts[0].lstrip('/').lower()
        args = parts[1:]


        if not self.is_admin(user) and name not in ("stats",):
            return {"ok": False, "result": "permission denied"}


        if name == "train_response" and ENABLED_COMMANDS.get("train_response"):
            return {"ok": True, "result": self.trainer.save_override(' '.join(args))}


        if name == "rate_response" and ENABLED_COMMANDS.get("rate_response"):
            score = args[0] if args else None
            return {"ok": True, "result": self.rater.rate(score)}


        if name in ("errors", "debug_errors") and ENABLED_COMMANDS.get("errors"):
            return {"ok": True, "result": self.errors.show_last(n=int(args[0]) if args else None)}


        if name == "stats" and ENABLED_COMMANDS.get("stats"):
            return {"ok": True, "result": self.stats.get_stats()}


        if name == "reset_module" and ENABLED_COMMANDS.get("reset_module"):
            module_name = args[0] if args else None
            return {"ok": True, "result": self.resetter.reset_module(module_name)}


        if name == "debug" and ENABLED_COMMANDS.get("debug"):
            return {"ok": True, "result": self.debug.summary()}


        if name == "simulate_error" and ENABLED_COMMANDS.get("simulate_error"):
            return {"ok": True, "result": self.debug.simulate_error()}


        if name == "export_logs" and ENABLED_COMMANDS.get("export_logs"):
            return {"ok": True, "result": self.errors.export_logs()}


        return {"ok": False, "result": f"unknown or disabled command: {name}"}
    


class DevCommandRouter:
    """Routes dev commands to appropriate handlers with permission checking."""
    
    def __init__(
        self, 
        kitsu_core,
        permission_manager: Optional[PermissionManager] = None
    ):
        self.core = kitsu_core
        self.permissions = permission_manager or PermissionManager()
        
        # Initialize all handlers
        self.handlers = {
            'trainer': ResponseTrainer(self.core),
            'rater': ResponseRater(self.core),
            'errors': ErrorViewer(self.core),
            'stats': StatsViewer(self.core),
            'resetter': ModuleResetter(self.core),
            'debug': DebugTools(self.core)
        }
        
        # Command registry: command_name -> (handler, method_name, required_permission)
        self.commands: Dict[str, tuple] = {
            'train': ('trainer', 'save_override', 'admin'),
            'rate': ('rater', 'rate', 'user'),
            'errors': ('errors', 'show_last', 'admin'),
            'stats': ('stats', 'get_stats', 'user'),
            'reset': ('resetter', 'reset_module', 'admin'),
            'debug': ('debug', 'summary', 'admin'),
            'simulate_error': ('debug', 'simulate_error', 'admin'),
            'export_logs': ('errors', 'export_logs', 'admin'),
        }
    
    async def route(
        self, 
        command: str, 
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route a command to its handler.
        
        Returns:
            {"ok": bool, "result": str, "data": Any}
        """
        parts = command.strip().lstrip('/').split()
        if not parts:
            return {"ok": False, "result": "Empty command"}
        
        cmd_name = parts[0].lower()
        args = parts[1:]
        
        # Check if command exists
        if cmd_name not in self.commands:
            return {"ok": False, "result": f"Unknown command: {cmd_name}"}
        
        handler_name, method_name, required_perm = self.commands[cmd_name]
        
        # Check permissions
        if not self.permissions.check(user_id, required_perm):
            return {"ok": False, "result": "Permission denied"}
        
        # Execute command
        try:
            handler = self.handlers[handler_name]
            method = getattr(handler, method_name)
            result = await method(*args, **kwargs) if asyncio.iscoroutinefunction(method) else method(*args, **kwargs)
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "result": f"Command failed: {e}"}