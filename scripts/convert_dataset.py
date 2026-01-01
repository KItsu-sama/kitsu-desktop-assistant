# scripts/convert_dataset.py
"""
Convert dataset_kitsu.py format to proper JSONL training format
"""

import json
import sys
from pathlib import Path
from typing import List, Dict


def convert_to_jsonl(samples: List[Dict], output_path: Path):
    """
    Convert training samples to JSONL format
    
    Input format (from dataset_kitsu.py):
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "metadata": {
            "mood": "behave",
            "style": "chaotic",
            "emotion": "happy"
        }
    }
    
    Output format (JSONL):
    Same structure, one JSON per line
    """
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            # Validate structure
            if "messages" not in sample or "metadata" not in sample:
                print(f"⚠️  Skipping invalid sample: missing messages or metadata")
                continue
            
            messages = sample["messages"]
            if len(messages) < 2:
                print(f"⚠️  Skipping sample: less than 2 messages")
                continue
            
            # Check first two messages are user/assistant
            if messages[0].get("role") != "user":
                print(f"⚠️  Skipping sample: first message not 'user'")
                continue
            
            if messages[1].get("role") != "assistant":
                print(f"⚠️  Skipping sample: second message not 'assistant'")
                continue
            
            # Check for content
            user_content = messages[0].get("content", "").strip()
            asst_content = messages[1].get("content", "").strip()
            
            if not user_content or not asst_content:
                print(f"⚠️  Skipping sample: empty content")
                continue
            
            # Write valid sample
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            valid_count += 1
    
    print(f"✅ Converted {valid_count} valid samples to {output_path}")
    return valid_count


def main():
    """Main conversion function"""
    
    print("\n🦊 Dataset Converter (FIXED)")
    print("="*60)
    
    # Import dataset generator
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from dataset_kitsu import create_expanded_dataset
    except ImportError as e:
        print(f"❌ Failed to import dataset_kitsu: {e}")
        print("\nMake sure dataset_kitsu.py exists in scripts/")
        return 1
    
    # Generate dataset
    print("\n📚 Generating dataset...")
    try:
        samples = create_expanded_dataset()
        print(f"✅ Generated {len(samples)} samples")
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Validate and convert
    print("\n🔍 Validating samples...")
    
    output_path = Path("data/training/kitsu_personality.jsonl")
    valid_count = convert_to_jsonl(samples, output_path)
    
    if valid_count == 0:
        print("\n❌ No valid samples!")
        return 1
    
    # Show sample
    print("\n📄 Sample output (first entry):")
    with open(output_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        print(json.dumps(sample, indent=2, ensure_ascii=False)[:300] + "...")
    
    print(f"\n✅ Conversion complete!")
    print(f"   Output: {output_path}")
    print(f"   Valid samples: {valid_count}")
    
    print("\n🎯 Next step:")
    print(f"   python scripts/finetune_lora.py --data-path {output_path}")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)