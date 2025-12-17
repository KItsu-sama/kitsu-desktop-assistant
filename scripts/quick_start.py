# FILE: scripts/quick_start.py
# ============================================================================

def quick_start_guide():
    """Print quick start guide"""
    
    guide = """
╔══════════════════════════════════════════════════════════════╗
║               🦊 KITSU QUICK START GUIDE 🦊                 ║
╚══════════════════════════════════════════════════════════════╝

📋 STEP-BY-STEP SETUP:

1️⃣  INSTALL DEPENDENCIES (5-10 minutes)
    python scripts/setup_complete.py
    → Choose option 1

2️⃣  GENERATE DATASET (instant)
    python scripts/setup_complete.py
    → Choose option 2
    → Creates data/training/kitsu_personality.json

3️⃣  CHOOSE TRAINING METHOD:

    A) GT 730 Training (2-4 hours, local)
       python scripts/finetune_lora.py
       → Uses TinyLlama 1.1B
       → Output: ~50MB adapter
       → Speed: Slow but FREE

    B) Colab Training (4-6 hours, free cloud)
       → Upload to Google Colab
       → Run notebooks/kitsu_training_colab.ipynb
       → Uses T4 GPU (much faster!)
       → Download trained model

4️⃣  TEST YOUR MODEL
    python scripts/run_kitsu.py
    → Loads trained model
    → Chat with Kitsu!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️  RECOMMENDED SETTINGS (GT 730):

Model: TinyLlama 1.1B
Quantization: 4-bit
LoRA Rank: 8
Batch Size: 1
Context: 512 tokens

Expected Performance:
  - Training: 2-4 hours
  - Inference: 5-10 tokens/sec
  - VRAM usage: ~1.5GB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🆘 TROUBLESHOOTING:

❌ Out of Memory?
    → Reduce batch_size to 1
    → Reduce max_seq_length to 256
    → Use Qwen 0.5B instead of TinyLlama

❌ Training too slow?
    → Use Google Colab (free T4 GPU)
    → Or wait patiently (it's worth it!)

❌ Model not responding well?
    → Add more training examples
    → Train for more epochs
    → Check safety filter (might be too strict)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 NEXT STEPS:

1. Train base personality (this guide)
2. Implement memory system (core/memory/)
3. Add emotion engine (core/personality/)
4. Connect to VTuber avatar
5. Deploy as desktop companion

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Need help? Check:
  - docs/ folder for detailed guides
  - logs/ folder for error messages
  - Discord/GitHub for community support

Good luck! You got this! 🦊✨
"""
    
    print(guide)

if __name__ == "__main__":
    quick_start_guide()