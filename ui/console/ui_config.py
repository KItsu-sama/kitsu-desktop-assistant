class UIConfig:
    def __init__(self, mode="text", tts_enabled=False):
        self.mode = mode
        self.tts_enabled = tts_enabled
    def is_text_mode(self) -> bool:
        return self.mode == "text"