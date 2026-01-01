# Changelog

## 2025-12-22 - Maintenance

- Removed duplicate / deprecated scripts and replaced them with safe deprecation stubs:
  - `scripts/run_pipeline.py` -> delegate to `scripts/train_pipeline.py`
  - `scripts/generate_dataset.py` -> delegate to `scripts/train_pipeline.py`
  - `scripts/format_for_training.py` -> deprecated
  - `scripts/load_to_ollama_direct.py` -> delegate to `scripts/load_to_ollama_direct.py`
  - `scripts/quickstart_lora.py` -> consolidated into `scripts/train_pipeline.py`
  - `scripts/sanitize_training_data.py` -> replaced by built-in safety filter

- Added `ModelResetController` to detect instability and perform safe resets.
- Added LoRA stack validation to prevent malformed stacks from loading at runtime.
- Hardened learning modules: `state_encoder`, `online_rl` (observations saved as curator-only), `trainer` (enforces sanitized datasets only).
- Added unit tests for reset behavior, trainer sanitization checks, and LoRA validation.
