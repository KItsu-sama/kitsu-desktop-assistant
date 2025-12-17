class KitsuSpeaker:
    def __init__(self, io_manager, config):
        self.io = io_manager
        self.config = config

    async def say(self, text):
        if self.config.tts_enabled:
            await self.io.voice.speak(text)
        else:
            print(f"\n🦊 Kitsu: {text}\n")
