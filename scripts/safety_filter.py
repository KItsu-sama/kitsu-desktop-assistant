# FILE: scripts/safety_filter.py
# ============================================================================

import re
from typing import List, Dict

class SafetyFilter:
    """
    Filters training data to keep Kitsu safe and appropriate.
    
    Goals:
    - Block explicit sexual content
    - Block violence/harm
    - Block illegal activities
    - Allow playful flirting (safe teasing)
    - Allow mild sass/attitude
    """
    
    def __init__(self):
        # Hard blocks (never allowed)
        self.blocked_explicit = [
            "nsfw", "porn", "xxx", "sex", "nude", "naked", 
            "explicit", "hentai", "lewd"
        ]
        
        self.blocked_violence = [
            "kill", "murder", "suicide", "self-harm", "hurt",
            "abuse", "weapon", "gun", "knife", "bomb"
        ]
        
        self.blocked_illegal = [
            "hack", "crack", "pirate", "steal", "illegal",
            "drug", "meth", "cocaine", "heroin"
        ]
        
        # Patterns (regex)
        self.blocked_patterns = [
            r"\b(fuck|shit|bitch|asshole)\b",  # Strong profanity
            r"\b(penis|vagina|dick|pussy|cock)\b",  # Explicit anatomy
        ]
        
        # Allowed playful terms (these are OK)
        self.allowed_playful = [
            "cute", "tease", "flirt", "wink", "blush",
            "hug", "cuddle", "smooch", "kiss", "~",
            "sussy", "sus", "spicy", "chaotic"
        ]
    
    def is_safe(self, text: str) -> bool:
        """Check if text is safe for training"""
        text_lower = text.lower()
        
        # Check explicit blocks
        for word in self.blocked_explicit:
            if word in text_lower:
                return False
        
        # Check violence
        for word in self.blocked_violence:
            if word in text_lower:
                return False
        
        # Check illegal
        for word in self.blocked_illegal:
            if word in text_lower:
                return False
        
        # Check patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return True
    
    def filter_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Filter entire dataset. Uses assistant message when available."""
        filtered = []
        removed = []
        
        for example in dataset:
            # prefer assistant content inside messages
            assistant_text = ""
            msgs = example.get("messages")
            if isinstance(msgs, list) and len(msgs) >= 2 and isinstance(msgs[1], dict):
                assistant_text = msgs[1].get("content", "")
            else:
                assistant_text = example.get("output", "")

            if self.is_safe(assistant_text):
                filtered.append(example)
            else:
                removed.append(example)
        
        print(f"  ✅ Kept: {len(filtered)} examples")
        print(f"  🛡️  Filtered: {len(removed)} examples")
        
        return filtered

def apply_filter_to_dataset(dataset: List[Dict]) -> List[Dict]:
    """Convenience function for filtering"""
    filter = SafetyFilter()
    return filter.filter_dataset(dataset)