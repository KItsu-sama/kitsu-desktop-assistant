# File: core/onboarding/flow_controller.py
"""Onboarding state machine - pure logic, no UI."""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

class OnboardingStep(Enum):
    WELCOME = "welcome"
    NAME_INPUT = "name_input"
    UI_MODE = "ui_mode"
    TTS_SETUP = "tts_setup"
    PERSONALITY = "personality"
    COMPLETE = "complete"


@dataclass
class OnboardingState:
    current_step: OnboardingStep
    user_name: Optional[str] = None
    ui_mode: str = "ascii"
    tts_enabled: bool = False
    personality_style: str = "friendly"
    completed: bool = False


class OnboardingFlowController:
    """Pure logic for onboarding flow - no rendering."""
    
    def __init__(self):
        self.state = OnboardingState(current_step=OnboardingStep.WELCOME)
    
    def get_current_prompt(self) -> dict:
        """Return prompt data for current step (UI will render this)."""
        prompts = {
            OnboardingStep.WELCOME: {
                "text": "Hiii~ I'm Kitsu! Your chaotic little fox assistant. What should I call you?",
                "expects_input": True,
                "input_type": "text"
            },
            OnboardingStep.UI_MODE: {
                "text": "Choose your UI style:",
                "options": ["ascii", "minimal", "full"],
                "expects_input": True,
                "input_type": "choice"
            },
            # ... more prompts
        }
        return prompts.get(self.state.current_step, {})
    
    def process_input(self, user_input: str) -> bool:
        """Process user input for current step. Returns True if moved to next step."""
        if self.state.current_step == OnboardingStep.WELCOME:
            self.state.user_name = user_input.strip()
            self.state.current_step = OnboardingStep.UI_MODE
            return True
        
        elif self.state.current_step == OnboardingStep.UI_MODE:
            if user_input.lower() in ["ascii", "minimal", "full"]:
                self.state.ui_mode = user_input.lower()
                self.state.current_step = OnboardingStep.TTS_SETUP
                return True
        
        # ... handle other steps
        return False
    
    def is_complete(self) -> bool:
        return self.state.completed