# File: core/dev/console_router.py
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
        """Construct router. Pass memory and module registry for richer commands."""
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
        
        # prompt inspector
        from core.dev.handlers.prompt_inspector import PromptInspector
        llm_interface = self.modules.get("llm") if self.modules else None
        self.prompt_inspector = PromptInspector(
            kitsu_core=None, 
            llm_interface=llm_interface, 
            logger=self.logger
        )


    def is_admin(self, user: Optional[str]) -> bool:
        if self.admin_check:
            return self.admin_check(user)
        return user in ADMIN_USERS


    def route(self, command: str, user: Optional[str] = None, **kwargs):
        """Route a textual command to the appropriate handler."""
        parts = command.strip().split()
        if not parts:
            return {"ok": False, "result": "empty command"}

        name = parts[0].lstrip('/').lower()
        args = parts[1:]

        if not self.is_admin(user) and name not in ("stats",):
            return {"ok": False, "result": "permission denied"}


        if name in ("train_response", "train") and ENABLED_COMMANDS.get("train_response"):
            res = self.trainer.save_override(' '.join(args))
            # Auto-train if enabled
            try:
                if getattr(self.trainer, "auto_train_enabled", False):
                    start_msg = self.trainer.trigger_training_async(include_ratings=False)
                    tail = self.trainer.get_training_output(10)
                    if tail:
                        res = f"{res} | {start_msg}\n--- training output ---\n{tail}"
                    else:
                        res = f"{res} | {start_msg}"
            except Exception:
                if self.logger:
                    self.logger.exception("Auto-train trigger failed")
            return {"ok": True, "result": res}


        if name in ("rate_response", "rate") and ENABLED_COMMANDS.get("rate_response"):
            score = args[0] if args else None
            res = self.rater.rate(score)
            # Auto-train if enabled (include ratings)
            try:
                if getattr(self.trainer, "auto_train_enabled", False):
                    start_msg = self.trainer.trigger_training_async(include_ratings=True, min_rating=4)
                    tail = self.trainer.get_training_output(10)
                    if tail:
                        res = f"{res} | {start_msg}\n--- training output ---\n{tail}"
                    else:
                        res = f"{res} | {start_msg}"
            except Exception:
                if self.logger:
                    self.logger.exception("Auto-train trigger failed")
            return {"ok": True, "result": res}


        if name in ("errors", "debug_errors") and ENABLED_COMMANDS.get("errors"):
            return {"ok": True, "result": self.errors.show_last(n=int(args[0]) if args else None)}


        if name == "stats" and ENABLED_COMMANDS.get("stats"):
            return {"ok": True, "result": self.stats.get_stats()}


        if name in ("reset_module", "reset") and ENABLED_COMMANDS.get("reset_module"):
            module_name = args[0] if args else None
            return {"ok": True, "result": self.resetter.reset_module(module_name)}


        if name == "debug" and ENABLED_COMMANDS.get("debug"):
            return {"ok": True, "result": self.debug.summary()}


        if name == "simulate_error" and ENABLED_COMMANDS.get("simulate_error"):
            return {"ok": True, "result": self.debug.simulate_error()}


        if name == "export_logs" and ENABLED_COMMANDS.get("export_logs"):
            return {"ok": True, "result": self.errors.export_logs()}


    
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

        # ADD THIS BLOCK - /auto_train command
        if name in ("auto_train", "autotrain") and ENABLED_COMMANDS.get("auto_train", True):
            arg = args[0] if args else None
            res = self.trainer.toggle_auto_train(arg)
            return {"ok": True, "result": res}

        if name in ("train_response", "train") and ENABLED_COMMANDS.get("train_response"):
            res = self.trainer.save_override(' '.join(args))
            # Auto-train if enabled
            try:
                if getattr(self.trainer, "auto_train_enabled", False):
                    start_msg = self.trainer.trigger_training_async(include_ratings=False)
                    tail = self.trainer.get_training_output(10)
                    if tail:
                        res = f"{res} | {start_msg}\n--- training output ---\n{tail}"
                    else:
                        res = f"{res} | {start_msg}"
            except Exception:
                if self.logger:
                    self.logger.exception("Auto-train trigger failed")
            return {"ok": True, "result": res}

        if name in ("rate_response", "rate") and ENABLED_COMMANDS.get("rate_response"):
            score = args[0] if args else None
            res = self.rater.rate(score)
            # Auto-train if enabled (include ratings)
            try:
                if getattr(self.trainer, "auto_train_enabled", False):
                    self.trainer.trigger_training_async(include_ratings=True, min_rating=4)
            except Exception:
                if self.logger:
                    self.logger.exception("Auto-train trigger failed")
            return {"ok": True, "result": res}
        
        # PROMPT INSPECTION COMMANDS (NEW)
        if name in ("show_pre_prompt", "show_prompt", "last_prompt"):
            format_arg = args[0] if args else "pretty"
            if format_arg not in ("pretty", "raw", "json"):
                format_arg = "pretty"
            res = self.prompt_inspector.show_last_prompt(format=format_arg)
            return {"ok": True, "result": res}

        if name in ("prompt_breakdown", "breakdown"):
            res = self.prompt_inspector.show_prompt_breakdown()
            return {"ok": True, "result": res}

        if name in ("model_config", "show_model"):
            res = self.prompt_inspector.show_model_config()
            return {"ok": True, "result": res}

        if name in ("compare_modes", "compare_prompts"):
            test_input = " ".join(args) if args else "Hello!"
            res = self.prompt_inspector.compare_modes(test_input)
            return {"ok": True, "result": res}

        if name in ("export_prompts", "save_prompts"):
            res = self.prompt_inspector.export_prompt_history()
            return {"ok": True, "result": res}
        
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
            'auto_train': ('trainer', 'toggle_auto_train', 'admin'),
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

            # normalize args for certain methods
            call_args = args
            if method_name == 'save_override' and args:
                call_args = [' '.join(args)]

            result = await method(*call_args, **kwargs) if asyncio.iscoroutinefunction(method) else method(*call_args, **kwargs)

            # If command was train or rate, optionally trigger auto-train
            if cmd_name in ("train", "rate"):
                try:
                    trainer = self.handlers.get('trainer')
                    if trainer and getattr(trainer, 'auto_train_enabled', False):
                        start_msg = trainer.trigger_training_async(include_ratings=(cmd_name=='rate'), min_rating=(4 if cmd_name=='rate' else None))
                        tail = trainer.get_training_output(10)
                        if tail:
                            result = f"{result} | {start_msg}\n--- training output ---\n{tail}"
                        else:
                            result = f"{result} | {start_msg}"
                except Exception:
                    # swallow errors to avoid failing the command
                    pass

            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "result": f"Command failed: {e}"}