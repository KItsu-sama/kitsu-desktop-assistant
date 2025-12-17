# scripts/generate_training_data.py

import asyncio
import json
import random
from pathlib import Path


class TrainingDataGenerator:
    def __init__(self, kitsu_core):
        self.core = kitsu_core
        self.output_path = Path("training_data/kitsu_dataset.jsonl")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def generate_conversation(
        self,
        prompt: str,
        mood: str,
        style: str,
        num_turns: int = 5
    ):
        """Generate a multi-turn conversation"""
        conversation = []
        
        for turn in range(num_turns):
            # Get Kitsu's response using CURRENT system
            response = await self.core.llm.generate_response(
                user_input=prompt,
                mood=mood,
                style=style,
                stream=False
            )
            
            conversation.append({
                "role": "user",
                "content": prompt
            })
            conversation.append({
                "role": "assistant", 
                "content": response
            })
            
            # Generate follow-up prompt
            prompt = self._generate_followup(response)
        
        return conversation
    
    def save_training_sample(
        self,
        conversation: list,
        emotion: str,
        mood: str,
        style: str,
        memory: list = None
    ):
        """Save in training format"""
        sample = {
            "messages": conversation,
            "metadata": {
                "emotion": emotion,
                "mood": mood,
                "style": style,
                "memory": memory or []
            }
        }
        
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    async def generate_dataset(self, num_samples: int = 10000):
        """Generate full training dataset"""
        moods = ["behave", "mean", "flirty"]
        styles = ["chaotic", "sweet", "cold", "silent"]
        
        # Seed prompts covering different scenarios
        seed_prompts = self._load_seed_prompts()
        
        for i in range(num_samples):
            mood = random.choice(moods)
            style = random.choice(styles)
            prompt = random.choice(seed_prompts)
            
            conversation = await self.generate_conversation(
                prompt, mood, style, num_turns=random.randint(3, 7)
            )
            
            self.save_training_sample(
                conversation=conversation,
                emotion=self.core.emotion_engine.get_current_emotion(),
                mood=mood,
                style=style
            )
            
            if i % 100 == 0:
                print(f"Generated {i}/{num_samples} samples")