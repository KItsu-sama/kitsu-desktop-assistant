"""Backward-compatible wrapper for CPU-friendly LoRA training.

This script forwards args to `scripts/finetune_lora.py` so tools that call
`finetune_lora_cpu.py` (legacy scripts) will continue to work.
"""
import subprocess
import sys

if __name__ == "__main__":
    # Forward all arguments to the main finetune script
    args = [sys.executable, "scripts/finetune_lora.py"] + sys.argv[1:]
    try:
        res = subprocess.run(args, check=False)
        sys.exit(res.returncode)
    except KeyboardInterrupt:
        print("\n⚠️ Training wrapper interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error launching finetune: {e}")
        sys.exit(1)
