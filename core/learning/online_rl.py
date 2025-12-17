# core/learning/online_rl.py

"""
Online Reinforcement Learning for Kitsu
Learns from user reactions in real-time (CPU-only)
"""

import json
import numpy as np
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple
import logging

log = logging.getLogger(__name__)

class OnlineRLEngine:
    """
    Learns which responses work best based on user reactions
    Similar to Neuro-sama's learning system
    """
    
    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Response memory (last 1000 interactions)
        self.response_memory = deque(maxlen=1000)
        
        # Response ratings (which patterns work best)
        self.response_patterns = {}
        
        # User preference model
        self.user_preferences = {
            "humor_level": 0.5,  # 0=serious, 1=very funny
            "affection_level": 0.5,  # 0=cold, 1=very affectionate
            "chaos_level": 0.5,  # 0=calm, 1=very chaotic
            "teasing_tolerance": 0.5,  # 0=sensitive, 1=loves teasing
        }
        
        # Load existing data
        self.load()
    
    def record_interaction(
        self,
        context: Dict,
        response: str,
        user_reaction: str = None,
        explicit_rating: int = None
    ):
        """
        Record an interaction for learning
        
        Args:
            context: Conversation context (mood, style, user input)
            response: Kitsu's response
            user_reaction: User's next message (used to infer satisfaction)
            explicit_rating: User rating (1-5) if provided
        """
        
        # Calculate implicit reward from user reaction
        reward = self._calculate_reward(user_reaction, explicit_rating)
        
        # Store interaction
        interaction = {
            "context": context,
            "response": response,
            "reward": reward,
            "timestamp": Path(__file__).stat().st_mtime
        }
        
        self.response_memory.append(interaction)
        
        # Update response patterns
        self._update_patterns(context, response, reward)
        
        # Update user preferences
        if reward > 0.6:  # Good response
            self._update_preferences(context, response)
        
        # Save periodically
        if len(self.response_memory) % 10 == 0:
            self.save()
        
        log.debug(f"Recorded interaction (reward: {reward:.2f})")
    
    def _calculate_reward(self, user_reaction: str, explicit_rating: int) -> float:
        """
        Calculate reward from user reaction
        """
        
        # Explicit rating takes priority
        if explicit_rating is not None:
            return explicit_rating / 5.0
        
        # Infer from user reaction
        if not user_reaction:
            return 0.5  # Neutral
        
        reaction_lower = user_reaction.lower()
        
        # Positive signals
        positive_keywords = [
            "haha", "lol", "😂", "🤣", "😄", "😊", "💕", "❤️", "love",
            "cute", "funny", "good", "great", "amazing", "perfect",
            "thanks", "thank you", "appreciate"
        ]
        
        # Negative signals
        negative_keywords = [
            "stop", "annoying", "不", "bad", "hate", "shut up",
            "leave me alone", "go away", "😡", "😠", "🙄"
        ]
        
        # Count matches
        positive_score = sum(1 for kw in positive_keywords if kw in reaction_lower)
        negative_score = sum(1 for kw in negative_keywords if kw in reaction_lower)
        
        # Calculate reward
        if positive_score > 0:
            return min(0.8 + (positive_score * 0.1), 1.0)
        elif negative_score > 0:
            return max(0.2 - (negative_score * 0.1), 0.0)
        else:
            # Length-based heuristic (longer response = more engaged)
            if len(user_reaction) > 50:
                return 0.7
            elif len(user_reaction) > 20:
                return 0.6
            else:
                return 0.5
    
    def _update_patterns(self, context: Dict, response: str, reward: float):
        """
        Update which response patterns work best
        """
        
        # Extract pattern key
        mood = context.get("mood", "behave")
        style = context.get("style", "chaotic")
        intent = context.get("intent", "unknown")
        
        pattern_key = f"{mood}_{style}_{intent}"
        
        # Update running average
        if pattern_key not in self.response_patterns:
            self.response_patterns[pattern_key] = {
                "count": 0,
                "avg_reward": 0.5,
                "best_responses": []
            }
        
        pattern = self.response_patterns[pattern_key]
        pattern["count"] += 1
        
        # Update average reward
        alpha = 0.1  # Learning rate
        pattern["avg_reward"] = (
            (1 - alpha) * pattern["avg_reward"] + alpha * reward
        )
        
        # Store best responses
        if reward > 0.7:
            pattern["best_responses"].append({
                "response": response,
                "reward": reward
            })
            # Keep only top 5
            pattern["best_responses"].sort(key=lambda x: x["reward"], reverse=True)
            pattern["best_responses"] = pattern["best_responses"][:5]
    
    def _update_preferences(self, context: Dict, response: str):
        """
        Learn user preferences from successful responses
        """
        
        # Analyze response characteristics
        response_lower = response.lower()
        
        # Humor indicators
        humor_indicators = ["haha", "lol", "kidding", "joke", "funny", "~", "!"]
        humor_score = sum(1 for ind in humor_indicators if ind in response_lower) / len(humor_indicators)
        
        # Affection indicators
        affection_indicators = ["💕", "❤️", "~", "aww", "cute", "love"]
        affection_score = sum(1 for ind in affection_indicators if ind in response_lower) / len(affection_indicators)
        
        # Chaos indicators
        chaos_indicators = ["!", "!!", "✨", "chaos", "whee", "yay"]
        chaos_score = sum(1 for ind in chaos_indicators if ind in response_lower) / len(chaos_indicators)
        
        # Update preferences (slowly)
        alpha = 0.05
        self.user_preferences["humor_level"] = (
            (1 - alpha) * self.user_preferences["humor_level"] + alpha * humor_score
        )
        self.user_preferences["affection_level"] = (
            (1 - alpha) * self.user_preferences["affection_level"] + alpha * affection_score
        )
        self.user_preferences["chaos_level"] = (
            (1 - alpha) * self.user_preferences["chaos_level"] + alpha * chaos_score
        )
    
    def get_best_response_pattern(self, context: Dict) -> Dict:
        """
        Get the best-performing response pattern for this context
        """
        
        mood = context.get("mood", "behave")
        style = context.get("style", "chaotic")
        intent = context.get("intent", "unknown")
        
        pattern_key = f"{mood}_{style}_{intent}"
        
        return self.response_patterns.get(pattern_key, {
            "avg_reward": 0.5,
            "best_responses": []
        })
    
    def get_user_preferences(self) -> Dict:
        """Get current user preference model"""
        return self.user_preferences.copy()
    
    def save(self):
        """Save learning data"""
        data = {
            "response_memory": list(self.response_memory),
            "response_patterns": self.response_patterns,
            "user_preferences": self.user_preferences
        }
        
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load existing learning data"""
        if self.save_path.exists():
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                
                self.response_memory = deque(data.get("response_memory", []), maxlen=1000)
                self.response_patterns = data.get("response_patterns", {})
                self.user_preferences = data.get("user_preferences", self.user_preferences)
                
                log.info(f"Loaded {len(self.response_memory)} past interactions")
            except Exception as e:
                log.warning(f"Failed to load RL data: {e}")