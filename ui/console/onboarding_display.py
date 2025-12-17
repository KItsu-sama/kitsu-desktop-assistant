# /ui/console/onboarding_display.py - PRESENTATION ONLY
# ============================================================================

from rich.console import Console
from rich.panel import Panel
from core.onboarding.flow import OnboardingFlowController

class OnboardingDisplay:
    """Renders onboarding UI - uses flow controller for logic."""
    
    def __init__(self, flow_controller: OnboardingFlowController):
        self.flow = flow_controller
        self.console = Console()
    
    async def run(self):
        """Run the onboarding display loop."""
        while not self.flow.is_complete():
            prompt_data = self.flow.get_current_prompt()
            self._render_prompt(prompt_data)
            
            user_input = input("> ")
            moved = self.flow.process_input(user_input)
            
            if not moved:
                self.console.print("[red]Invalid input, please try again[/red]")
    
    def _render_prompt(self, prompt_data: dict):
        """Render a single prompt."""
        text = prompt_data.get("text", "")
        self.console.print(Panel(text, border_style="yellow", title="Kitsu"))
        
        if "options" in prompt_data:
            for i, opt in enumerate(prompt_data["options"], 1):
                self.console.print(f"  {i}. {opt}")


import json
from pathlib import Path
from .ascii_ui import AsciiUI
from .kitsu_speaker import KitsuSpeaker

class OnboardingManager:
    def __init__(self, config, speaker):
        self.config = config
        self.speaker = speaker
        self.ui = AsciiUI()
        self.profile_file = Path("data/config/user_profile.json")

    def is_first_time(self):
        return not self.profile_file.exists()

    async def run(self):
        self.ui.banner("Welcome to KITSU", "Your digital fox companion")

        await self.speaker.say(
            "Hiii! I'm Kitsu! You're seeing this because it's our first time meeting!"
        )

        await self.speaker.say("Let me set everything up for you~")

        # Ask for name
        name = input("🦊 What should I call you? → ")
        await self.speaker.say(f"Oki! {name} it is~")

        self._save_profile({"name": name})

        await self.speaker.say("All set! Let's have fun together!")
        return True

    def _save_profile(self, data):
        self.profile_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.profile_file, "w", encoding="utf8") as f:
            json.dump(data, f, indent=4)
