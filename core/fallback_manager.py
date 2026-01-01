# core/fallback_manager.py

import random
import json
from pathlib import Path
import logging
from core.memory.user_manager import UserManager

log = logging.getLogger(__name__)


class FallbackManager:
    """
    Generates fallback responses when the LLM fails.

    Core rules:
        - Primary failure is ALWAYS identity-based:
            * "my AI is not AI-ing"
            * "your fox is not fox-ing"
        - Cause is appended softly using "cuz / c.u.z"
        - Cause never replaces the core phrase
    """

    # --------------------------------------------
    # Init
    # --------------------------------------------
    def __init__(self, memory: UserManager | None = None):
        self.memory = memory

    # --------------------------------------------
    # User personalization
    # --------------------------------------------
    def _get_personal_info(self):
        try:
            if self.memory:
                info = self.memory.get_user_info() or {}
                if isinstance(info, dict):
                    return (
                        info.get("name"),
                        info.get("nickname"),
                        info.get("refer_title"),
                    )
        except Exception:
            log.exception("Failed to fetch user info from memory")

        # Disk fallback (tests / early boot)
        try:
            cfg = Path("data/config/user_profile.json")
            if cfg.exists():
                data = json.loads(cfg.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return (
                        data.get("name"),
                        data.get("nickname"),
                        data.get("refer_title"),
                    )
            else:
                log.warning("Fallback missing user profile: %s", cfg)
        except Exception:
            log.exception("Failed to read user profile from disk")

        return None, None, None

    def _choose_personal(self, mood, name, nickname, refer_title):
        if mood == "flirty":
            order = [nickname, refer_title, name]
        elif mood == "mean":
            order = [name, refer_title, nickname]
        elif mood == "behave":
            order = [refer_title, name, nickname]
        else:
            order = [name, nickname, refer_title]

        choices = [c for c in order if c]
        return random.choice(choices) if choices and random.random() < 0.75 else None

    # --------------------------------------------
    # Cause suffix (NEVER replaces core)
    # --------------------------------------------
    def _cause_suffix(self, cause: str | None):
        cause = (cause or "").lower().strip()

        if cause == "timeout":
            return random.choice([
                " cuz of being stuck thinking for too long",
                " cuz of thinking too hard",
                " cuz of being stuck",
            ])

        if cause == "crash":
            return random.choice([
                " cuz of felling over internally",
                " cuz of having something snapped inside",
                " cuz my code tripped",
            ])

        if cause in ("rate_limit", "overload"):
            return random.choice([
                " cuz of being pushed too fast",
                " cuz there is too much happening at once",
                " cuz of being overwhelmed",
            ])

        return ""

    # --------------------------------------------
    # Glitch effect
    # --------------------------------------------
    def _glitch(self, text: str) -> str:
        if random.random() >= 0.20:
            return text

        effects = [
            lambda t: t.replace("not", "n—not"),
            lambda t: t.replace("AI-ing", "A͟I͟-͟i͟n͟g͟"), 
            lambda t: t.replace("fox-ing", "fo͢x͢-i͢n͢g͢"),
            lambda t: t.replace(" ", " … ", 1),
            lambda t: "▉ " + t,
            lambda t: "".join(
                c + random.choice(["", "͟"]) if random.random() < 0.06 else c
                for c in t
            ),
        ]
        return random.choice(effects)(text)

    # --------------------------------------------
    # Fox quirks
    # --------------------------------------------
    def _fox_noise(self):
        noises = [" *nyah*", " *mrrp*", " *fox-chirp*", " 🦊"]
        return random.choice(noises) if random.random() < 0.30 else ""

    # --------------------------------------------
    # Base phrase (identity failure FIRST)
    # --------------------------------------------
    def _generate_base_phrase(self, target, mood, cause_suffix):
        extra = "I think " if random.random() < 0.5 else ""

        playful = ""
        ending = ""

        if mood == "flirty" and random.random() < 0.5:
            playful = "sweet " if random.random() < 0.5 else "dear "
            ending = " anymore" if random.random() < 0.5 else ""

        # Choose identity failure
        core = (
            f"your {playful}fox is not fox-ing"
            if random.random() < 0.5
            else "my AI is not AI-ing"
        )

        if target:
            return f"{target}, {extra}{core}{ending}{cause_suffix}".strip()

        return f"{extra}{core}{cause_suffix}".strip()

    # --------------------------------------------
    # Public API
    # --------------------------------------------
    def generate(self, mood: str = "", style: str = "", cause: str | None = None) -> str:
        cause = cause or "unknown"

        name, nickname, refer_title = self._get_personal_info()
        target = self._choose_personal(mood, name, nickname, refer_title)

        cause_suffix = self._cause_suffix(cause)
        base = self._generate_base_phrase(target, mood, cause_suffix)

        base = self._glitch(base)
        noise = self._fox_noise()

        # Mood polish
        if mood == "behave":
            prefix = "umm " if random.random() < 0.85 else ""
            base = f"{prefix}{base}{noise}"

        elif mood == "mean":
            base += random.choice([
                " again.",
                " (seriously?)",
                " unbelievable.",
                " this is your fault.",
                " probably because of you.",
                ". Ugh.",
            ]) + noise

        elif mood == "flirty":
            starts = ["Oh no~ ", "Hmm~ ", "Hehe~ ", "Ufu~ ", "Mmh~ ", "Oh dear~ "]
            ends = ["~", " hehe~", " ufufu~", " <3", " mmh~", ""]
            base = f"{random.choice(starts)}{base}{random.choice(ends)}{noise}".strip()

        else:
            if noise and random.random() < 0.9:
                base += noise

        return base
