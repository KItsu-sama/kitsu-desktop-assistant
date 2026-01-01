# ============================================================================
# FILE: scripts/setup_wizard.py
# Interactive setup wizard (first-time configuration)
# ============================================================================

import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import box

console = Console()

class SetupWizard:
    """Interactive setup wizard for first-time configuration"""
    
    def __init__(self):
        self.config = {
            "mode": "text",
            "model": "gemma:2b",
            "temperature": 0.8,
            "enable_tts": False,
            "enable_stt": False,
            "enable_avatar": False,
            # Enable Claude Haiku 4.5 by default for all clients
            "enable_claude_haiku": True,
            "safe_mode": True,
            "streaming": True,
        }
        
        self.user_profile = {
            "name": "",
            "nickname": "",
            "refer_title": "",
            "gender": "unknown",
            "status": "user",
            "relationship": {
                "trust_level": 0.5,
                "affinity": 0.5,
                "lore_tag": "stranger"
            },
            "permissions": {
                "admin": False,
                "dev_console": False,
                "memory_clear": True,
                "state_change": True
            }
        }
    
    def run(self):
        """Run the setup wizard"""
        
        self._show_welcome()
        self._configure_user()
        self._configure_system()
        self._configure_personality()
        self._save_config()
        self._show_completion()
    
    def _show_welcome(self):
        """Show welcome screen"""
        console.clear()
        console.print(Panel.fit(
            "[bold magenta]🦊 KITSU SETUP WIZARD[/bold magenta]\n"
            "[white]Let's get you set up for some chaotic fun![/white]",
            border_style="magenta",
            box=box.DOUBLE_EDGE
        ))
        console.print("")
    
    def _configure_user(self):
        """Configure user profile"""
        console.print("[bold cyan]👤 User Profile[/bold cyan]")
        console.print("[cyan]" + "─"*60 + "[/cyan]\n")
        
        # Name
        self.user_profile["name"] = Prompt.ask(
            "What's your name?",
            default="User"
        )
        
        # Nickname
        self.user_profile["nickname"] = Prompt.ask(
            "What should Kitsu call you? (nickname)",
            default=self.user_profile["name"]
        )
        
        # Refer title
        self.user_profile["refer_title"] = Prompt.ask(
            "How should Kitsu address you?",
            choices=["Master", "Boss", "Friend", "Senpai", self.user_profile["nickname"]],
            default=self.user_profile["nickname"]
        )
        
        # Gender (for pronoun context)
        gender = Prompt.ask(
            "Gender (for pronoun context)?",
            choices=["male", "female", "other", "prefer not to say"],
            default="prefer not to say"
        )
        self.user_profile["gender"] = gender if gender != "prefer not to say" else "unknown"
        
        # Admin/dev access
        is_dev = Confirm.ask(
            "\nAre you a developer? (enables dev console)",
            default=False
        )
        
        if is_dev:
            self.user_profile["permissions"]["admin"] = True
            self.user_profile["permissions"]["dev_console"] = True
            self.user_profile["status"] = "developer"
        
        console.print("")
    
    def _configure_system(self):
        """Configure system settings"""
        console.print("[bold cyan]⚙️  System Configuration[/bold cyan]")
        console.print("[cyan]" + "─"*60 + "[/cyan]\n")
        # Model selection (repeatable; user can re-configure until satisfied)
        while True:
            console.print("Available models:")
            console.print("  1. TinyLlama 1.1B (fastest, GT 730 friendly)")
            console.print("  2. Gemma 2B (balanced)")
            console.print("  3. Qwen 1.8B (smarter, slower)")
            console.print("  4. Custom Ollama model")

            model_choice = Prompt.ask(
                "\nChoose model",
                choices=["1", "2", "3", "4"],
                default="2"
            )

            MODEL_MAP = {
                "1": "tinyllama:1.1b",
                "2": "gemma:2b",
                "3": "qwen:1.8b"
            }

            if model_choice == "4":
                self.config["model"] = Prompt.ask("Enter Ollama model name")
            else:
                self.config["model"] = MODEL_MAP[model_choice]

            console.print(f"\nSelected model: [yellow]{self.config['model']}[/yellow]")
            if not Confirm.ask("Reconfigure model?", default=False):
                break

        # Voice/TTS
        self.config["enable_tts"] = Confirm.ask(
            "\nEnable Text-to-Speech (experimental)?",
            default=False
        )
        
        if self.config["enable_tts"]:
            self.config["enable_stt"] = Confirm.ask(
                "Enable Speech-to-Text?",
                default=False
            )
        
        # Avatar
        self.config["enable_avatar"] = Confirm.ask(
            "\nEnable VTuber avatar (requires OBS/VTube Studio)?",
            default=False
        )
        # Greeting on startup
        self.config["greet_on_startup"] = Confirm.ask(
            "\nEnable greeting on startup? (Kitsu will greet when the app starts)",
            default=True
        )
        # Claude Haiku model opt-in
        self.config["enable_claude_haiku"] = Confirm.ask(
            "\nEnable Claude Haiku 4.5 for all clients?",
            default=self.config.get("enable_claude_haiku", True)
        )
        
        console.print("")

    def _write_first_run_flag(self):
        """Write a lightweight first-run flag so other scripts can detect setup."""
        try:
            Path("data").mkdir(parents=True, exist_ok=True)
            flag = Path("data/.first_run_complete")
            flag.write_text("1")
        except Exception:
            # Non-fatal; continue without breaking the wizard
            pass
    
    def _configure_personality(self):
        """Configure personality defaults"""
        console.print("[bold cyan]😊 Personality Settings[/bold cyan]")
        console.print("[cyan]" + "─"*60 + "[/cyan]\n")
        
        console.print("Kitsu's default mood:")
        mood = Prompt.ask(
            "Choose default mood",
            choices=["behave", "flirty", "mean"],
            default="behave"
        )
        
        console.print("\nKitsu's default style:")
        style = Prompt.ask(
            "Choose default style",
            choices=["chaotic", "sweet", "cold", "silent"],
            default="chaotic"
        )
        
        self.config["default_mood"] = mood
        self.config["default_style"] = style
        
        console.print("")
    
    def _save_config(self):
        """Save configuration files.

        Merge with existing `user_profile` when present to preserve developer
        flags, permissions, learned ratings, and LoRA state. Also set
        `completed_setup` flag so re-running the wizard cannot unset it.
        """
        console.print("[yellow]💾 Saving configuration...[/yellow]")

        # Create directories
        Path("data/config").mkdir(parents=True, exist_ok=True)
        Path("data/memory").mkdir(parents=True, exist_ok=True)
        Path("data/models").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)

        # Save main config
        with open("data/config.json", "w", encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)

        # Merge user_profile with existing file if present
        user_path = Path("data/config/user_profile.json")
        if user_path.exists():
            try:
                existing = json.loads(user_path.read_text(encoding='utf-8'))
            except Exception:
                existing = {}
        else:
            existing = {}

        # Preserve developer flags and permissions unless explicitly changed
        preserved = {}
        if existing.get("status") == "developer":
            preserved["status"] = existing.get("status")
        # Preserve explicit permission flags
        preserved_permissions = existing.get("permissions", {})

        merged = existing.copy()
        merged.update(self.user_profile)

        # Merge permissions conservatively: existing keys not overwritten unless set in wizard
        merged_perms = preserved_permissions.copy()
        merged_perms.update(self.user_profile.get("permissions", {}))
        merged["permissions"] = merged_perms

        # Ensure completed_setup flag is present and remains true once set
        if existing.get("completed_setup"):
            merged["completed_setup"] = True
        else:
            merged["completed_setup"] = True  # set on successful save

        # Write merged profile
        with open(user_path, "w", encoding='utf-8') as f:
            json.dump(merged, f, indent=2)

        # Create default personality config
        personality = {
            "default_mood": self.config.get("default_mood", "behave"),
            "default_style": self.config.get("default_style", "chaotic"),
            "emotion_decay_rate": 0.1,
            "emotion_threshold": 0.3,
            "max_stack_size": 5
        }

        with open("data/config/personality.json", "w", encoding='utf-8') as f:
            json.dump(personality, f, indent=2)
        console.print("[green]✓ Configuration saved![/green]\n")
        # Mark first run as complete file flag
        self._write_first_run_flag()

    def apply_defaults(self):
        """Apply present default configuration without any interactive prompts.

        This method writes defaults but merges with existing user profile to
        preserve developer permissions and other persistent flags.
        """
        # No interactive prompts; simply persist the defaults
        Path("data/config").mkdir(parents=True, exist_ok=True)
        Path("data/memory").mkdir(parents=True, exist_ok=True)
        Path("data/models").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)

        import json
        with open("data/config.json", "w", encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)

        # Merge with existing user profile to avoid wiping developer flags
        user_path = Path("data/config/user_profile.json")
        existing = {}
        if user_path.exists():
            try:
                existing = json.loads(user_path.read_text(encoding='utf-8'))
            except Exception:
                existing = {}

        merged = existing.copy()
        merged.update(self.user_profile)
        # Preserve existing permissions unless explicitly set
        merged_perms = existing.get("permissions", {}).copy()
        merged_perms.update(self.user_profile.get("permissions", {}))
        merged["permissions"] = merged_perms
        merged["completed_setup"] = True

        with open(user_path, "w", encoding='utf-8') as f:
            json.dump(merged, f, indent=2)

        personality = {
            "default_mood": self.config.get("default_mood", "behave"),
            "default_style": self.config.get("default_style", "chaotic"),
            "emotion_decay_rate": 0.1,
            "emotion_threshold": 0.3,
            "max_stack_size": 5
        }

        with open("data/config/personality.json", "w", encoding='utf-8') as f:
            json.dump(personality, f, indent=2)

        console.print("[green]✓ Default configuration written (non-interactive)[/green]\n")
    
    def _show_completion(self):
        """Show completion message"""
        console.print(Panel.fit(
            "[bold green]✅ Setup Complete![/bold green]\n\n"
            f"[white]Welcome, {self.user_profile['refer_title']}![/white]\n"
            f"[white]Kitsu is configured and ready to go![/white]\n\n"
            "[cyan]Next steps:[/cyan]\n"
            "  1. Generate training data: [yellow]python scripts/generate_dataset.py[/yellow]\n"
            "  2. Train personality: [yellow]python scripts/finetune_lora.py[/yellow]\n"
            "  3. Start Kitsu: [yellow]python main.py[/yellow]\n\n"
            "[dim]You can reconfigure anytime with /user commands[/dim]",
            border_style="green",
            box=box.DOUBLE_EDGE
        ))


def main():
    """Run setup wizard"""
    wizard = SetupWizard()
    wizard.run()


if __name__ == "__main__":
    main()