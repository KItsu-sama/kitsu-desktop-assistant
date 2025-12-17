# core/fallback_manager.py

import random
from core.memory.user_manager import UserManager


class FallbackManager:
    """
    Generates fallback responses when the LLM fails.
    Handles:
        - personalization (name/nickname/title)
        - mood styling
        - glitch effects
        - fox noises & emojis
        - Kitsu personality quirks
        - spam-safe formatting
    """

    def __init__(self, memory=None):
        self.memory = memory or UserManager if isinstance(memory, UserManager) else None   

    # ----------------------------------------------------------
    #  Retrieve stored user data (name, nickname, refer_title)
    # ----------------------------------------------------------
    def _get_personal_info(self):
        user_name = nickname = refer_title = None
        try:
            if self.memory:
                info = self.memory.get_user_info()
                if isinstance(info, dict):
                    user_name = info.get("name")
                    nickname = info.get("nickname")
                    refer_title = info.get("refer_title")
        except Exception:
            pass

        return user_name, nickname, refer_title

    # ----------------------------------------------------------
    #  Glitch effect helper
    # ----------------------------------------------------------
    def _glitch(self, text: str) -> str:
        if random.random() >= 0.20:  # 20% glitch chance
            return text

        glitch_variants = [ 
            text.replace("not", "n—not"),
            text.replace("AI-ing", "A͟I͟-͟i͟n͟g͟"),
            text.replace("fox-ing", "fo͢x͢-i͢n͢g͢"),
            text[:len(text)//2] + "..." + text[len(text)//2:],
            "▉ " + text,
        ]
        return random.choice(glitch_variants)

    # ----------------------------------------------------------
    #  Pick personalization target by mood
    # ----------------------------------------------------------
    def _choose_personal(self, mood, name, nickname, refer_title):
        if mood == "flirty":
            choices = [nickname, refer_title, name]  # flirty prefers nickname
        elif mood == "mean":
            choices = [name, refer_title, nickname]  # mean prefers name
        elif mood == "behave":
            choices = [refer_title, name, nickname]  # behave prefers title
        else:
            choices = [name, nickname, refer_title]

        choices = [c for c in choices if c]

        # Still chance-based
        if choices and random.random() < 0.75:
            return random.choice(choices)

        return None  # fallback to generic

    # ----------------------------------------------------------
    #  Fox quirks & emojis
    # ----------------------------------------------------------
    def _fox_noise(self):
        noises = [" *nyah*", " *mrrp*", " *fox-chirp*", " 🦊"]
        return random.choice(noises) if random.random() < 0.30 else ""

    # ----------------------------------------------------------
    #  Generate the base fallback phrase
    # ----------------------------------------------------------
    def _generate_base_phrase(self, target):
        # 50/50 split between foxing and AI-ing for personalized lines
        if target:
            if random.random() < 0.5:
                return f"{target}, your fox is not fox-ing"
            else:
                return f"{target}, my AI is not AI-ing"

        # generic fallback
        return "my AI is not AI-ing"

    # ----------------------------------------------------------
    #  Main fallback generator
    # ----------------------------------------------------------
    def generate(self, mood: str = "", style: str = "") -> str:
        name, nickname, refer_title = self._get_personal_info()

        # choose personalization target
        target = self._choose_personal(mood, name, nickname, refer_title)

        # generate base phrase
        base = self._generate_base_phrase(target)

        # apply glitch effects
        base = self._glitch(base)

        # fox emotion noise
        noise = self._fox_noise()

        # -------------------------
        #   Mood-specific edits
        # -------------------------
        if mood == "behave":
            prefix = "umm " if random.random() < 0.85 else ""
            base = f"{prefix}{base} anymore{noise}"

        elif mood == "mean":
            mean_endings = [
                " again.",
                " ... what did you break this time?",
                " (seriously?)",
                " unbelievable.",
                " this is your fault.",
                " probably because of you.",
            ]
            base = f"{base}{random.choice(mean_endings)}{noise}"

        elif mood == "flirty":
            flirty_endings = ["~", " hehe~", " ufufu~", " <3", " mmh~", ""]
            base = f"{base} {random.choice(flirty_endings)}{noise}".strip()

        else:
            # neutral fallback
            base = f"{base}{noise}"

        return base
