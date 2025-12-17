# core/llm/prompt_builder.py

"""
Prompt construction with personality and context
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds prompts with personality, memory, and emotion context
    """
    
    def __init__(
        self,
        character_context: str,
        memory_manager: Optional[Any] = None,
        templates_path: Optional[Path] = None
    ):
        self.character_context = character_context
        self.memory = memory_manager
        
        # Set templates path (default to data/templates)
        if templates_path is None:
            templates_path = Path("data/templates")
        self.templates_path = templates_path
        
        # Load mode templates
        self.mode_templates = self._load_mode_templates()
        
        log.info(f"PromptBuilder initialized with {len(self.mode_templates)} mode templates")
    
    def _load_mode_templates(self) -> Dict[str, str]:
        """
        Load mode-specific templates from files
        
        Returns:
            Dict mapping mode names to template content
        """
        templates = {}
        mode_dir = self.templates_path / "mode_templates"
        
        if not mode_dir.exists():
            log.warning(f"Mode templates directory not found: {mode_dir}")
            return self._get_fallback_templates()
        
        # Load each mode template
        for mode_file in ["behave.txt", "mean.txt", "flirty.txt"]:
            mode_path = mode_dir / mode_file
            mode_name = mode_file.replace(".txt", "")
            
            try:
                with open(mode_path, "r", encoding="utf-8") as f:
                    templates[mode_name] = f.read().strip()
                log.debug(f"Loaded template: {mode_name}")
            except FileNotFoundError:
                log.warning(f"Template not found: {mode_path}")
                templates[mode_name] = self._get_fallback_templates()[mode_name]
            except Exception as e:
                log.error(f"Error loading template {mode_path}: {e}")
                templates[mode_name] = self._get_fallback_templates()[mode_name]
        
        return templates
    
    def _get_fallback_templates(self) -> Dict[str, str]:
        """
        Fallback templates if files can't be loaded
        """
        return {
            "behave": "You are friendly, playful, and supportive. Be kind but teasing.",
            "mean": "You are mischievous and enjoy teasing. Be playfully mean but not hurtful.",
            "flirty": "You are flirtatious and charming. Be playful and affectionate."
        }
    
    def _get_style_modifier(self, style: str) -> str:
        """
        Get style-specific behavioral modifiers
        
        Args:
            style: Current style (chaotic/sweet/cold/silent)
            
        Returns:
            Style guidance text
        """
        modifiers = {
            "chaotic": "Be energetic, unpredictable, and playful. Use exclamations and emotes.",
            "sweet": "Be warm, gentle, and caring. Use soft language and affection.",
            "cold": "Be polite but distant. Keep responses brief and emotionally reserved.",
            "silent": "Be extremely quiet. Respond with minimal words, mostly emotes or sounds."
        }
        return modifiers.get(style, "Use your natural personality.")
    
    def _build_greeting_prompt(self, user_title: str, mood: str, style: str) -> str:
        """Build a custom greeting prompt"""
        
        # Get mode-specific greeting flavor
        mode_flavors = {
            "behave": "friendly, energetic, and welcoming",
            "mean": "playfully bratty and teasing, but still excited to see them",
            "flirty": "charming, coy, and affectionate with fox-spirit mystique"
        }
        
        style_hints = {
            "chaotic": "Lots of energy! Exclamations! Playful and bouncy!",
            "sweet": "Warm, gentle, soft and caring.",
            "cold": "Brief, polite, subtle warmth.",
            "silent": "Minimal words. Mostly emotes or sounds."
        }
        
        flavor = mode_flavors.get(mood, "friendly and playful")
        style_hint = style_hints.get(style, "Be natural and expressive.")
        
        greeting_prompt = f"""{self.character_context}

    ## Current Mode: {mood.upper()}
    Be {flavor}.

    ## Current Style: {style.upper()}
    {style_hint}

    {user_title} has just woken you up. Respond with a brief, natural greeting.

    ## Requirements
    - ONE OR TWO sentences MAXIMUM
    - Be casual and conversational
    - Match your {mood} mood and {style} style
    - DO NOT describe actions in third person (no "spreads across my face", "purr-fectly", etc.)
    - DO NOT introduce yourself or explain who you are
    - Just greet them naturally like you're excited to chat
    - Use dialogue format, not narration

    Now generate a short greeting (one or two sentences)."""
        
        return greeting_prompt

    
    def build_conversational_prompt(
        self,
        user_input: str,
        mood: str = "behave",
        style: str = "chaotic",
        memory_limit: int = 5
    ) -> str:
        """
        Build a conversational response prompt
        
        Args:
            user_input: User's message
            mood: Current mood (behave/mean/flirty)
            style: Current style (chaotic/sweet/cold/silent)
            memory_limit: Number of past messages to include
            
        Returns:
            Complete prompt string
        """
        # Get memory context
        memory_context = ""
        if self.memory:
            try:
                memory_context = self.memory.format_context(memory_limit)
            except Exception as e:
                log.warning(f"Memory context failed: {e}")
        
        # Get mode template
        mode_template = self.mode_templates.get(mood, self.mode_templates["behave"])
        
        # Get style modifier
        style_modifier = self._get_style_modifier(style)
        
        # Get user info for personalization
        user_info = ""
        if self.memory:
            try:
                user_data = self.memory.get_user_info()
                name = user_data.get("name", "Unknown")
                nickname = user_data.get("nickname") or name
                title = user_data.get("refer_title") or nickname
                status = user_data.get("status", "User")
                
                # Relationship
                rel = user_data.get("relationship", {})
                trust = int(rel.get("trust_level", 0) * 100)
                affinity = int(rel.get("affinity", 0) * 100)
                lore = rel.get("lore", "").strip()

                user_info = f"""
        ## User Profile
        - Name: {name}
        - Kitsu calls you: {title}
        - Status: {status}
        - Trust: {trust}/100
        - Affinity: {affinity}/100"""

                if lore:
                    user_info += f"\n- Lore: {lore}"

                # Permissions
                perms = user_data.get("permissions", {})
                if perms:
                    user_info += "\n- Permissions:\n"
                    for k, v in perms.items():
                        user_info += f"   - {k}: {v}"

            except Exception as e:
                log.warning(f"User info retrieval failed: {e}")
        
        # Build complete prompt
        prompt = f"""{self.character_context}

## Mode: {mood.upper()}
{mode_template}

## Style: {style.upper()}
{style_modifier}
{user_info}

{memory_context}

## Current Message
User: {user_input}

Respond as Kitsu. Stay in character. Keep response natural and conversational.
Reply to the user in the same tone and style as the examples.
Do not summarize your identity unless directly asked.

Kitsu:"""
        
        return prompt
    
    def build_emotion_analysis_prompt(self, text: str) -> str:
        """
        Build prompt for emotion classification
        
        Args:
            text: Text to analyze
            
        Returns:
            Emotion analysis prompt
        """
        prompt = f"""Analyze the emotional content and intent of this message.

Input: "{text}"

Return ONLY valid JSON with these fields:
{{
  "intent": "one word intent (ask, joke, flirt, insult, praise, compliment, command, etc.)",
  "sentiment": "positive | neutral | negative",
  "hf_emotion": "joy | sadness | anger | fear | surprise | disgust | neutral",
  "trigger": "emotional trigger if any (teased, praised, insulted, ignored, complimented, etc.) or null"
}}

JSON:"""
        return prompt
    
    def build_reaction_planning_prompt(
        self,
        user_input: str,
        emotion_analysis: Dict[str, Any],
        mood: str,
        style: str
    ) -> str:
        """
        Build prompt for reaction planning
        
        Args:
            user_input: User's message
            emotion_analysis: Emotion analysis results
            mood: Current mood
            style: Current style
            
        Returns:
            Reaction planning prompt
        """
        import json
        
        prompt = f"""Plan Kitsu's reaction to this interaction.

Current State:
- Mood: {mood}
- Style: {style}

User Input: "{user_input}"

Emotion Analysis:
{json.dumps(emotion_analysis, indent=2)}

Return ONLY valid JSON with:
{{
  "plan": "brief description of intended approach",
  "expression": "primary emotion to display (happy, annoyed, flirty, shy, smug, etc.)",
  "retaliation": "none | mild | strong | playful"
}}

JSON:"""
        return prompt
    
    def reload_templates(self):
        """
        Reload mode templates from disk (useful for live editing)
        """
        log.info("Reloading mode templates...")
        self.mode_templates = self._load_mode_templates()
        log.info(f"Reloaded {len(self.mode_templates)} templates")

#==============================================================================

"""
Minimal prompt builder for trained character models
No personality instructions - model knows Kitsu from training
"""

class MinimalPromptBuilder:
    def __init__(self, memory_manager=None):
        self.memory = memory_manager
    
    def build_prompt(
        self,
        user_input: str,
        emotion: str,
        mood: str,
        style: str,
        memory_limit: int = 3
    ) -> str:
        """Build context-only prompt (no personality)"""
        
        # Get recent memory
        memory_context = ""
        if self.memory:
            try:
                recent = self.memory.get_recent(memory_limit)
                if recent:
                    memory_context = f"memory: {', '.join(recent)}"
            except Exception:
                pass
        
        #  just context metadata
        prompt = f"""emotion: {emotion}
    mood: {mood}
    style: {style}
    {memory_context}

    User: {user_input}

    Respond as Kitsu, keeping responses natural and in-character."""
        
        return prompt
    
    def _build_greeting_prompt(self, user_title: str, mood: str, style: str) -> str:
        """
        Build minimal greeting prompt for character model
        Character model already knows personality, just needs context
        """
        # Minimal context - personality is in the weights!
        prompt = f"""emotion: happy
    mood: {mood}
    style: {style}
    {user_title} just woke you up. Greet them warmly and naturally.

    Respond with a brief, natural greeting (one or two sentences)."""
        
        return prompt
    
    def build_conversational_prompt(self, user_input: str, mood: str = "behave", 
                                   style: str = "chaotic", memory_limit: int = 5) -> str:
        """Alias for compatibility with full PromptBuilder"""
        return self.build_prompt(
            user_input=user_input,
            emotion="neutral",
            mood=mood,
            style=style,
            memory_limit=memory_limit
        )