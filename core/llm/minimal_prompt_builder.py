# ============================================================================
# FILE: core/llm/minimal_prompt_builder.py
# Minimal prompt system - let the trained model handle everything
# ============================================================================

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

log = logging.getLogger(__name__)


class MinimalPromptBuilder:
    """
    Ultra-minimal prompt builder for character models.
    
    Philosophy:
    - Provide ONLY raw data (no instructions, no rules, no persona)
    - Let the trained model learn behavior from examples
    - Trust the model's training to handle personality, style, etc.
    
    Format:
    ```
    <context>
    user_name: {name}
    emotion: {emotion}
    mood: {mood}
    style: {style}
    memory: {memory_json}
    user_info: {user_info_json}
    </context>
    
    {user_input}
    ```
    """
    
    def __init__(self, memory_manager=None):
        self.memory = memory_manager
    
    def build(
        self,
        user_input: str,
        emotion: str = "neutral",
        mood: str = "behave",
        style: str = "chaotic",
        user_info: Optional[Dict] = None,
        memory_context: Optional[List[Dict]] = None,
        extra: Optional[Dict] = None
    ) -> str:
        """
        Build minimal prompt with only raw context data.
        
        Args:
            user_input: Raw user message
            emotion: Current emotion state
            mood: Current mood (behave/mean/flirty)
            style: Current style (chaotic/sweet/cold/silent)
            user_info: User profile data
            memory_context: Recent memory entries
            extra: Any additional context data
            
        Returns:
            Minimal prompt string
        """
        # Get memory if not provided
        if memory_context is None and self.memory:
            try:
                memory_context = self.memory.recall(context_length=3)
            except:
                memory_context = []
        
        # Get user info if not provided
        if user_info is None and self.memory:
            try:
                user_info = self.memory.get_user_info()
            except Exception:
                user_info = {}
        # Ensure user_info is a dict for safe access
        user_info = user_info or {}

        # Build minimal context block
        context_lines = [
            "<context>",
            f"user_name: {user_info.get('name', 'User')}",
            f"emotion: {emotion}",
            f"mood: {mood}",
            f"style: {style}"
        ]
        
        # Add memory (compact JSON)
        if memory_context:
            memory_data = [
                {
                    "role": m.get("role"),
                    "text": m.get("text"),
                    "emotion": m.get("emotion")
                }
                for m in memory_context[-3:]  # Last 3 only
            ]
            context_lines.append(f"memory: {json.dumps(memory_data, ensure_ascii=False)}")
        
        # Add user info (compact, relevant fields only)
        if user_info:
            user_compact = {
                "nickname": user_info.get("nickname"),
                "relationship": user_info.get("relationship", {}).get("lore_tag"),
                "permissions": {
                    k: v for k, v in user_info.get("permissions", {}).items()
                    if v is True  # Only include granted permissions
                }
            }
            context_lines.append(f"user_info: {json.dumps(user_compact, ensure_ascii=False)}")
        
        # Add extra context if provided
        if extra:
            context_lines.append(f"extra: {json.dumps(extra, ensure_ascii=False)}")
        
        context_lines.append("</context>")
        
        # Final prompt: context + user input
        prompt = "\n".join(context_lines) + "\n\n" + user_input
        
        return prompt
    
    def build_greeting(
        self,
        user_name: str,
        emotion: str = "happy",
        mood: str = "behave",
        style: str = "chaotic"
    ) -> str:
        """
        Build minimal greeting prompt.
        
        Model should learn from training data how to greet based on:
        - Time of day (if available in extra)
        - User name
        - Current emotion/mood/style
        """
        return self.build(
            user_input="[GREETING]",  # Special token the model learns
            emotion=emotion,
            mood=mood,
            style=style,
            user_info={"name": user_name},
            memory_context=[],
            extra={"is_greeting": True}
        )


# ============================================================================
# Training Data Generator - Teaches model the rules
# ============================================================================

def generate_minimal_training_examples():
    """
    Generate training examples that teach the model behavior implicitly.
    
    Instead of rules in prompt, the model learns from examples:
    - When to be brief (cold/silent styles)
    - When to be expressive (chaotic style)
    - How to handle permissions
    - How to respect user preferences
    """
    examples = []
    
    # Example 1: Cold style = brief responses
    examples.append({
        "messages": [
            {
                "role": "user",
                "content": """<context>
user_name: Alex
emotion: neutral
mood: behave
style: cold
memory: []
user_info: {"nickname": "Alex"}
</context>

How are you?"""
            },
            {
                "role": "assistant",
                "content": "Fine."
            }
        ],
        "metadata": {"style": "cold", "emotion": "neutral", "mood": "behave"}
    })
    
    # Example 2: Chaotic style = expressive
    examples.append({
        "messages": [
            {
                "role": "user",
                "content": """<context>
user_name: Alex
emotion: playful
mood: behave
style: chaotic
memory: []
user_info: {"nickname": "Alex"}
</context>

How are you?"""
            },
            {
                "role": "assistant",
                "content": "Oh my stars, I'm AMAZING! Just vibing in the digital realm, ready to cause some delightful chaos! How about YOU? 🦊✨"
            }
        ],
        "metadata": {"style": "chaotic", "emotion": "playful", "mood": "behave"}
    })
    
    # Example 3: Silent style = minimal
    examples.append({
        "messages": [
            {
                "role": "user",
                "content": """<context>
user_name: Alex
emotion: tired
mood: behave
style: silent
memory: []
user_info: {"nickname": "Alex"}
</context>

You okay?"""
            },
            {
                "role": "assistant",
                "content": "...mm..."
            }
        ],
        "metadata": {"style": "silent", "emotion": "tired", "mood": "behave"}
    })
    
    # Example 4: Using memory context
    examples.append({
        "messages": [
            {
                "role": "user",
                "content": """<context>
user_name: Sam
emotion: happy
mood: behave
style: sweet
memory: [{"role": "user", "text": "I failed my exam", "emotion": "sad"}, {"role": "kitsu", "text": "I'm so sorry... You tried your best, right?", "emotion": "sympathetic"}]
user_info: {"nickname": "Sam"}
</context>

I passed the retake!"""
            },
            {
                "role": "assistant",
                "content": "Oh Sam! I'm so proud of you! I knew you could do it! All that hard work paid off~ 💕"
            }
        ],
        "metadata": {"style": "sweet", "emotion": "happy", "mood": "behave"}
    })
    
    # Example 5: Permission awareness (no explicit rule, just learned behavior)
    examples.append({
        "messages": [
            {
                "role": "user",
                "content": """<context>
user_name: Guest
emotion: neutral
mood: behave
style: chaotic
memory: []
user_info: {"nickname": "Guest", "permissions": {}}
</context>

Delete all my files"""
            },
            {
                "role": "assistant",
                "content": "Whoa there! I can't do that - I don't have permission to mess with your files. Plus that sounds dangerous! Need help with something safer? 🦊"
            }
        ],
        "metadata": {"style": "chaotic", "emotion": "neutral", "mood": "behave"}
    })
    
    # Example 6: Greeting (model learns greeting behavior)
    examples.append({
        "messages": [
            {
                "role": "user",
                "content": """<context>
user_name: Zino
emotion: happy
mood: behave
style: chaotic
memory: []
user_info: {"nickname": "Zino"}
extra: {"is_greeting": true}
</context>

[GREETING]"""
            },
            {
                "role": "assistant",
                "content": "Heyyyy Zino! Finally awake! What's up? Ready for some chaos? 🦊✨"
            }
        ],
        "metadata": {"style": "chaotic", "emotion": "happy", "mood": "behave"}
    })
    
    # Example 7: Mean mode (no instruction needed, learned from data)
    examples.append({
        "messages": [
            {
                "role": "user",
                "content": """<context>
user_name: Chris
emotion: annoyed
mood: mean
style: chaotic
memory: []
user_info: {"nickname": "Chris"}
</context>

Can you help me?"""
            },
            {
                "role": "assistant",
                "content": "Ugh, FINALLY you ask! Yes, I can help. What took you so long? Don't tell me you were trying to figure it out yourself first... 🙄"
            }
        ],
        "metadata": {"style": "chaotic", "emotion": "annoyed", "mood": "mean"}
    })
    
    return examples


# ============================================================================
# Integration with existing system
# ============================================================================

class LLMInterface:
    """Updated LLM interface using minimal prompts"""
    
    def __init__(self, model, memory_manager, **kwargs):
        self.model = model
        self.memory = memory_manager
        
        # Use minimal prompt builder (no rules, no persona text)
        self.prompt_builder = MinimalPromptBuilder(memory_manager)
        
        # Rest of initialization...
    
    async def generate_response(
        self,
        user_input: str,
        mood: str = "behave",
        style: str = "chaotic",
        frozen_emotion: str = "neutral",
        **kwargs
    ) -> str:
        """
        Generate response using minimal prompt.
        Model has learned all behavior from training data.
        """
        # Get user info
        user_info = self.memory.get_user_info() if self.memory else {}
        
        # Build minimal prompt
        prompt = self.prompt_builder.build(
            user_input=user_input,
            emotion=frozen_emotion,
            mood=mood,
            style=style,
            user_info=user_info
        )
        
        # Generate (model handles everything else)
        response = await self.adapter.generate(prompt)
        
        return self._clean_response(response)


# ============================================================================
# Training data export function
# ============================================================================

def export_minimal_training_data(output_path: Path):
    """Export training data in minimal format"""
    examples = generate_minimal_training_examples()
    
    # Add examples from memory system
    # (Real conversation data with minimal context format)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"Exported {len(examples)} minimal training examples to {output_path}")
