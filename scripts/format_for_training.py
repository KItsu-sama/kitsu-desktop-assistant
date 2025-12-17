# scripts/format_for_training.py

import json
from pathlib import Path
from typing import List, Dict

def convert_to_chat_template(samples: List[Dict]) -> List[Dict]:
    """
    Convert to Llama-3 chat format for training
    
    Format:
    <|system|>
    emotion: happy | mood: behave | style: chaotic
    <|user|>
    Hello!
    <|assistant|>
    Hi there! 🦊
    """
    formatted = []
    
    for sample in samples:
        context = sample["context"]
        messages = sample["messages"]
        
        # Build chat format
        text = f"<|system|>\n{context}\n"
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|{role}|>\n{content}\n"
        
        formatted.append({
            "text": text,
            "metadata": sample["metadata"]
        })
    
    return formatted

def save_training_data(output_path: Path):
    """Generate and save complete training dataset"""
    from scripts.dataset_kitsu import create_expanded_dataset
    from scripts.safety_filter import apply_filter_to_dataset
    
    print("🦊 Generating training dataset...")
    
    # Generate raw dataset
    raw_dataset = create_expanded_dataset()
    print(f"✅ Generated {len(raw_dataset)} raw examples")
    
    # Apply safety filter
    safe_dataset = apply_filter_to_dataset(raw_dataset)
    print(f"🛡️  Filtered to {len(safe_dataset)} safe examples")
    
    # Convert to training format
    formatted = convert_to_chat_template(safe_dataset)
    
    # Save as JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"💾 Saved to {output_path}")
    print(f"📊 Total training samples: {len(formatted)}")

if __name__ == "__main__":
    save_training_data(Path("training_data/kitsu_formatted.jsonl"))