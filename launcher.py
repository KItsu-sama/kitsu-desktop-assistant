from ui.console.onboarding import OnboardingManager
from ui.console.kitsu_speaker import KitsuSpeaker
from ui.console.ui_config import UIConfig

async def launch():
    ui_config = UIConfig(mode="text", tts_enabled=False)
    speaker = KitsuSpeaker(io_manager, ui_config)
    onboarding = OnboardingManager(config, speaker)

    if onboarding.is_first_time():
        await onboarding.run()
