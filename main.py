#!/usr/bin/env python3
"""
Kitsu AI — main.py
Entry point: initializes all subsystems, wires the async event loop, and launches
voice or text mode based on config.

This file is dependency-light and resilient:
- Safe imports with helpful error messages
- Graceful shutdown via signals
- Background tasks for emotion ticking & memory autosave
- Event-driven plumbing between input → planning → executor → output
"""

import asyncio
import signal
import sys
import logging
from pathlib import Path
from typing import Optional, List
from core.kitsu_core import KitsuIntegrated
import argparse

# Setup logging first
from utils.logger import setup_logger, get_logger
from rich.console import Console

console = Console()
log = setup_logger("kitsu.main", level=logging.INFO)

# Track async tasks for cleanup
BACKGROUND_TASKS: List[asyncio.Task] = []


# =============================================================================
# Safe Imports with Error Handling
# =============================================================================

def safe_import(module_path: str, error_msg: str = None):
    """
    Safely import a module with helpful error message
    
    Args:
        module_path: Module import path (e.g., "core.kitsu_core")
        error_msg: Custom error message
        
    Returns:
        Imported module or None if failed
    """
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[parts[-1]])
        return module
    except ImportError as e:
        msg = error_msg or f"Failed to import {module_path}: {e}"
        log.error(msg)
        log.error(f"Make sure {module_path.replace('.', '/')} exists")
        return None
    except Exception as e:
        log.exception(f"Unexpected error importing {module_path}: {e}")
        return None


# Import core Kitsu system
kitsu_core_module = safe_import(
    "core.kitsu_core",
    "Could not import KitsuIntegrated. Check core/kitsu_core.py exists"
)

if not kitsu_core_module:
    log.critical("Cannot start without KitsuIntegrated. Exiting.")
    sys.exit(1)

KitsuIntegrated = kitsu_core_module.KitsuIntegrated


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config() -> dict:
    """Load configuration from data/config.json"""
    config_path = Path("data/config.json")
    
    if not config_path.exists():
        # If no config exists, attempt interactive setup wizard, else fall back to defaults
        defaults = {
            "mode": "text",  # text or voice
            "model": "gemma:2b",
            "temperature": 0.8,
            "voice_mode": "auto_pitch",
            "enable_tts": False,
            "enable_stt": False,
            "enable_avatar": False,
            "enable_browser_hooks": False,
            "enable_system_control": False,
            "safe_mode": True,  # Disable dangerous operations
            "streaming": True,
            "greet_on_startup": True,
            "paths": {
                "personality": "data/config/personality.json",
                "user_profile": "data/config/user_profile.json",
                "permissions": "data/config/permissions.json",
                "kitsu_config": "data/config/kitsu_config.json"
            }
        }

        log.warning("No config.json found — attempting interactive setup wizard")
        try:
            from scripts.setup_wizard import SetupWizard
            wiz = SetupWizard()
            wiz.run()
            # after wizard completes, attempt to reload config
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            log.info(f"Configuration loaded: mode={config.get('mode', 'text')}")
            return config
        except Exception as e:
            log.warning(f"Setup wizard not available or failed: {e}. Using defaults")
            log.info(f"Configuration loaded: mode={defaults.get('mode')}")
            return defaults
    
    try:
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        log.info(f"Configuration loaded: mode={config.get('mode', 'text')}")
        return config
    except Exception as e:
        log.error(f"Failed to load config: {e}")
        return {}


# =============================================================================
# Background Task Management
# =============================================================================

async def start_background_task(coro, name: str):
    """Start a background task and track it"""
    task = asyncio.create_task(coro, name=name)
    BACKGROUND_TASKS.append(task)
    log.info(f"Started background task: {name}")
    return task


async def stop_all_background_tasks():
    """Stop all background tasks gracefully"""
    log.info(f"Stopping {len(BACKGROUND_TASKS)} background tasks...")
    
    for task in BACKGROUND_TASKS:
        if not task.done():
            task.cancel()
    
    # Wait for all to finish
    results = await asyncio.gather(*BACKGROUND_TASKS, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            log.error(f"Task {BACKGROUND_TASKS[i].get_name()} error: {result}")
    
    BACKGROUND_TASKS.clear()
    log.info("All background tasks stopped")


# =============================================================================
# Graceful Shutdown Handler
# =============================================================================

class GracefulShutdown:
    """Handle graceful shutdown on signals"""
    
    def __init__(self, kitsu = KitsuIntegrated):
        self.kitsu = kitsu
        self.shutdown_event = asyncio.Event()
    
    def request_shutdown(self, signum=None, frame=None):
        """Request graceful shutdown"""
        log.info(f"Shutdown requested (signal: {signum})")
        self.shutdown_event.set()
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        await self.shutdown_event.wait()
    
    async def cleanup(self):
        """Perform cleanup"""
        log.info("Starting cleanup...")
        
        try:
            # Stop background tasks
            await stop_all_background_tasks()
            
            # Shutdown Kitsu
            await self.kitsu.shutdown()
            
            log.info("Cleanup complete")
        except Exception as e:
            log.exception(f"Error during cleanup: {e}")


# =============================================================================
# Main Application
# =============================================================================

class KitsuApplication:
    """Main application orchestrator"""
    
    def __init__(self, config: dict):
        self.config = config
        self.kitsu: Optional[KitsuIntegrated] = None
        self.shutdown_handler: Optional[GracefulShutdown] = None
        self.mode = config.get("mode", "text")
    
    async def initialize(self):
        """Initialize all systems"""
        # Nice startup banner using rich Panel for emphasis
        from rich.panel import Panel

        panel_text = (
            "[bold magenta]With Emotion + Personality Systems[/bold magenta]\n\n"
            "[cyan]Type[/cyan] [bold yellow]/help[/bold yellow] for a full list of commands.\n"
            "[cyan]Type[/cyan] [bold yellow]/mood -h[/bold yellow] or [bold yellow]/style -h[/bold yellow] for command-specific help.\n"
            "[cyan]Tip[/cyan]: Most commands support -h / --help."
        )

        console.print(Panel(panel_text, title="🦊 KITSU AI - Desktop VTuber Assistant", border_style="magenta"))
        
        # Create KitsuIntegrated instance
        try:
            log.info("Initializing KitsuIntegrated...")
            # Determine continuous decay behavior:
            # - If running in voice mode, keep continuous decay on
            # - If both STT and TTS are enabled, keep continuous decay on
            # - Otherwise (text-only), use event-driven decay (tick on chat events)
            enable_stt = self.config.get('enable_stt', False)
            enable_tts = self.config.get('enable_tts', False)
            continuous_decay = True if (self.mode == 'voice' or (enable_stt and enable_tts)) else False

            # Model value can be a string (legacy) or a dict (new config format
            # where model is an object like {"style": "sweet"}). Normalize to a
            # string model name for LLM initialization.
            model_cfg = self.config.get("model", "gemma:2b")
            if isinstance(model_cfg, dict):
                model_value = model_cfg.get("style") or model_cfg.get("name") or "gemma:2b"
            else:
                model_value = model_cfg

            self.kitsu = KitsuIntegrated(
                model=model_value,
                temperature=self.config.get("temperature", 0.8),
                streaming=self.config.get('streaming', True),
                continuous_decay=continuous_decay
            )
            
            # Initialize all subsystems
            await self.kitsu.initialize()
            # Apply greeting configuration (suppress if config says so)
            self.kitsu.greet_on_startup = self.config.get('greet_on_startup', True)
            
            log.info("✅ KitsuIntegrated ready!")
            
        except Exception as e:
            log.exception(f"Failed to initialize Kitsu: {e}")
            raise
        
        # Setup shutdown handler
        self.shutdown_handler = GracefulShutdown(self.kitsu)
        
        # Register signal handlers (Unix/Windows compatible)
        try:
            signal.signal(signal.SIGINT, self.shutdown_handler.request_shutdown)
            signal.signal(signal.SIGTERM, self.shutdown_handler.request_shutdown)
            log.info("Signal handlers registered")
        except Exception as e:
            log.warning(f"Could not register signals: {e}")
        
        # Start background tasks
        await self._start_background_tasks()
    
    async def _start_background_tasks(self):
        """Start all background tasks"""
        log.info("Starting background tasks...")
        
        # Emotion engine decay loop (if available)
        if hasattr(self.kitsu, '_decay_task') and self.kitsu._decay_task:
            # Already started in initialize
            BACKGROUND_TASKS.append(self.kitsu._decay_task)
        
        # Memory auto-save loop
        if self.kitsu.memory:
            await start_background_task(
                self._memory_autosave_loop(),
                "memory_autosave"
            )
        
        # Emotion tick loop (updates personality periodically)
        if self.kitsu.emotion_engine:
            await start_background_task(
                self._emotion_tick_loop(),
                "emotion_tick"
            )
    
    async def _memory_autosave_loop(self):
        """Auto-save memory periodically"""
        try:
            while True:
                await asyncio.sleep(1)   
                if self.kitsu.memory:
                    try:
                        await self.kitsu.memory.save_async_aio()
                        log.debug("Memory auto-saved")
                    except Exception as e:
                        log.error(f"Memory auto-save failed: {e}")
        except asyncio.CancelledError:
            log.debug("Memory auto-save loop cancelled")
    
    async def _emotion_tick_loop(self):
        """Tick emotion engine periodically"""
        try:
            while True:
                await asyncio.sleep(1)  # Every second
                if self.kitsu.emotion_engine:
                    try:
                        await self.kitsu.emotion_engine.tick()
                    except Exception as e:
                        log.error(f"Emotion tick failed: {e}")
        except asyncio.CancelledError:
            log.debug("Emotion tick loop cancelled")
    
    async def run(self):
        """Main run loop - launches appropriate interface"""
        log.info(f"Running in {self.mode} mode")
        log.info("")
        
        try:
            if self.mode == "voice":
                await self._run_voice_mode()
            else:
                await self._run_text_mode()
        except Exception as e:
            log.exception(f"Run loop error: {e}")
        finally:
            # Cleanup
            if self.shutdown_handler:
                await self.shutdown_handler.cleanup()
    
    async def _run_text_mode(self):
        """Run text chat interface"""
        # Use the chat loop from KitsuIntegrated
        try:
            # Create concurrent tasks
            chat_task = asyncio.create_task(
                self.kitsu.chat_loop(),
                name="chat_interface"
            )
            shutdown_task = asyncio.create_task(
                self.shutdown_handler.wait_for_shutdown(),
                name="shutdown_watcher"
            )
            
            # Wait for either chat to end or shutdown signal
            done, pending = await asyncio.wait(
                [chat_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            log.exception(f"Text mode error: {e}")
    
    async def _run_voice_mode(self):
        """Run voice interface"""
        log.warning("Voice mode not yet implemented")
        log.info("Falling back to text mode...")
        await self._run_text_mode()


# =============================================================================
# Entry Point
# =============================================================================

async def async_main(run_setup: bool = False, no_greet: bool = False, auto_train: bool = False, model_swicht: bool = False) -> int:
    """Async main entry point"""
    # Optionally run setup wizard first
    if run_setup:
        try:
            from scripts.setup_wizard import SetupWizard
            log.info("--setup passed: running setup wizard")
            wiz = SetupWizard()
            wiz.run()
        except Exception as e:
            log.warning(f"Failed to run setup wizard via --setup: {e}")

    config = load_config()
    if no_greet:
        config['greet_on_startup'] = False

    if model_swicht:
        from core.llm.llm_interface import LLMInterface
        lmf = LLMInterface()
        lmf.switch_to_character_model()

    # Create application
    app = KitsuApplication(config)
    
    try:
        # Initialize
        await app.initialize()
        
        # Enable auto-train if flag is set
        if auto_train:
            if app.kitsu and hasattr(app.kitsu, 'dev_router') and app.kitsu.dev_router:
                trainer = app.kitsu.dev_router.trainer
                trainer.auto_train_enabled = True
                log.info("✅ Auto-train enabled via --auto-train flag")
                console.print("[green]✅ Auto-train enabled[/green]")
            else:
                log.warning("⚠️  Could not enable auto-train (dev_router not available)")

        
        # Run
        await app.run()
        
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        return 1
    
    return 0


def main():
    """Synchronous entry point"""
    # Python 3.7+ compatibility
    if sys.version_info < (3, 7):
        log.error("Python 3.7+ required")
        return 1
    
    try:
        # Parse CLI args
        parser = argparse.ArgumentParser(description='Kitsu AI')
        parser.add_argument('--model-swicht', action='store_true', help='Switch model on startup') # default to kitsu character model = kitsu:character
        parser.add_argument('--setup', action='store_true', help='Run setup wizard before starting')
        parser.add_argument('--no-greet', action='store_true', help='Disable greeting on startup (override config)')
        parser.add_argument('--auto-train', action='store_true', help='Enable automatic fine-tuning after /train or /rate')
        args = parser.parse_args()

        # Run async main
        exit_code = asyncio.run(async_main(
            run_setup=args.setup, 
            no_greet=args.no_greet,
            auto_train=args.auto_train,
            model_swicht=args.model_swicht
        ))
        return exit_code
    except KeyboardInterrupt:
        log.info("\n\n👋 Kitsu: Bye bye! See you later! 🦊\n")
        return 0
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())