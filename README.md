Kitsu - Desktop VTuber Assistant
================================

README
-----------

Run the setup wizard (interactive):

```powershell
python main.py --setup
```

Start Kitsu normally (text mode):

```powershell
Kitsu — Desktop VTuber Assistant
================================

Kitsu is a small, privacy-friendly assistant focused on characterful conversation and easy experimentation with LoRA adapters. She is designed to work on modest hardware (CPU and low-end GPUs) and provides scripts to generate data, train lightweight LoRA adapters, and load models into Ollama or run them directly.

Key principles
-------------
- Friendly to low-end hardware (works on CPU; supports small models like TinyLlama).
- Small, modular codebase with clear scripts for common tasks.
- Safety-first defaults: safe_mode enabled by default and careful training data handling.

Quick start
-----------
1. Install Python 3.9+ and the listed dependencies (see `requirements.txt`).
2. Run the one-time setup: `python scripts/first_run.py --run` or `python main.py --setup`.
3. Generate data, train, and load a model using the pipeline or individual scripts (see `docs/TUTORIAL.md`).

Project layout (high-level commented tree)
-----------------------------------------
Note: important files and folders with short descriptions. Do not invent files that aren't present.

```
CONTRIBUTING.md        # Contribution guidelines
README.md              # This file (overview + quick start)
main.py                # Application entry point and runtime orchestration
launcher.py            # Alternate launcher (platform-specific helpers)
pyproject.toml         # Project metadata
requirements.txt       # Python dependencies (runtime/dev)

assets/                # Static assets used by the app (models, sounds, etc.)
	models/kitsu/        # Example or packaged model artifacts
	sounds/               # Audio assets for TTS/notifications

core/                  # Core runtime components and glue
	kitsu_core.py        # High-level KitsuIntegrated implementation
	fallback_manager.py  # Fallback handlers for missing subsystems
	cognition/           # Planning, intent classification, and NLP

data/                  # Runtime data and generated artifacts
	config.json          # Main runtime configuration (created by setup)
	config/              # Per-subsystem config files and user profile
	training/            # Generated training dataset (JSON)
	models/              # Trained/merged LoRA and merged models

docs/                  # Documentation and tutorials (you'll add TUTORIAL.md)

scripts/               # Helpful utilities and pipelines
	generate_training_data.py  # Generate training examples from templates
	finetune_lora.py          # CPU-friendly LoRA training script
	load_to_ollama.py         # Preferred Ollama loader
	load_to_ollama_direct.py  # Direct import fallback (no GGUF step)
	run_pipeline.py           # Run the full generate->train->load pipeline
	first_run.py              # Manage first-run status and reset

tests/                 # Smoke and unit tests

utils/                 # Small utility helpers (logging, file utils)

```

Core features
-------------
- Lightweight LoRA training pipeline optimized for CPU and small GPUs.
- Modular loading into Ollama (preferred) with a direct fallback loader.
- Simple personality/emotion system; memory persistence and a small command console for interaction.

System architecture (high-level)
--------------------------------
- main.py initializes the KitsuIntegrated core which wires: input → planner → executor → output.
- A small emotion engine and memory system run as background tasks and persist to disk.
- Models are represented as base model + optional LoRA adapters; adapters are merged for deployment.

Models & LoRA strategy
----------------------
- Use small base models (TinyLlama/TinyLlama-1.1B) for CPU-first workflows.
- Train LoRA adapters per personality/style to keep them small and fast to train.
- Merged models (base + LoRA) are written to `data/models/` and can be loaded into Ollama or used locally.

Scripts overview
----------------
- `scripts/first_run.py` — manage first-run state (status, run, reset).
- `scripts/generate_training_data.py` — prepare JSON training examples.
- `scripts/finetune_lora.py` — CPU-friendly LoRA training (writes into `data/models/`).
- `scripts/load_to_ollama.py` — preferred path to load a model into Ollama.
- `scripts/load_to_ollama_direct.py` — fallback importer if Ollama is not available.
- `scripts/run_pipeline.py` — convenience runner: generate → train → load (stops on failure).

First-time setup (high level)
-----------------------------
1. Run: `python scripts/first_run.py --run` (interactive) or `python main.py --setup`.
2. The wizard writes configuration to `data/config.json` and marks first-run with `data/.first_run_complete`.
3. Re-run: `python scripts/first_run.py --run` or `python scripts/first_run.py --reset` to clear and reconfigure.

Common workflows
----------------
- Quick pipeline (all-in-one): `python scripts/run_pipeline.py`
- Manual sequence:
	1. `python scripts/generate_training_data.py`
	2. `python scripts/finetune_lora.py --style chaotic`
	3. `python scripts/load_to_ollama.py` (or direct fallback)
	4. `python main.py`

Reset / rebuild instructions
----------------------------
- Reset configuration: `python scripts/first_run.py --reset` (removes `data/config.json` and first-run flag).
- If you want to remove only training outputs: delete `data/training/` and `data/models/`.

Notes on safety & design philosophy
----------------------------------
- Safe-mode defaults are enabled to reduce the risk of dangerous or unsafe behavior.
- Training scripts include conservative token handling and probe callbacks to detect overfitting early.
- Kitsu is intentionally simple and auditable — prefer transparent, file-based workflows.

Where to go next
----------------
- Read `docs/TUTORIAL.md` for a step-by-step zero-to-running guide.
- Use `scripts/run_pipeline.py` to automate standard workflows.

If something is unclear or a script fails, open an issue with the failing command and any console output — the code is designed to be readable and easy to debug.

