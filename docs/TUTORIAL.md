# Kitsu Quick Tutorial — Zero to Running

This guide provides a concise, step-by-step path from a fresh machine to a running Kitsu instance. It assumes a technical but non-expert user and recommends CPU-friendly options.

1./Environment setup
--------------------;

- Install Python 3.9+ (Windows: use the official installer; enable "Add to PATH").
- Create and activate a virtual environment:
  - Windows (PowerShell):
    - `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
  - Linux/macOS:
    - `python -m venv .venv; source .venv/bin/activate`
- Install dependencies:
  - `python -m pip install -r requirements.txt`
- Optional: if you plan to use Ollama, install it following https://ollama.ai/download and ensure `ollama` is on your PATH.

2./Dataset generation
---------------------;

- Prepare training examples used by the LoRA training script:
  - `python scripts/generate_training_data.py`
- Output will be written to `data/training/kitsu_personality.json`.

3./LoRA training (CPU-friendly)
-------------------------------;

- Train a LoRA adapter (style named `chaotic` as example):
  - `python scripts/finetune_lora.py --style chaotic --probe-interval 500`
- Output will be saved to `data/models/kitsu-lora-chaotic`.
- Notes:
  - CPU training can be slow but is supported. The script prints ETA estimates.
  - Use `--probe-interval 0` to disable periodic probe checks.

4./Model loading (Ollama preferred, direct fallback available)
-------------------------------------------------------------;

- Preferred: use Ollama to host and serve the model.
  - `python scripts/load_to_ollama_direct.py`
  - If Ollama is not running: start it (`ollama serve`) and re-run the script.
- Direct fallback (if Ollama is unavailable):
  - `python scripts/load_to_ollama_direct.py`
  - This imports a merged model directly and updates `data/config.json` to use `kitsu:character`.

5./Running Kitsu
----------------;

- Start Kitsu in text mode:
  - `python main.py`
- One-time setup (interactive):
  - `python scripts/first_run.py --run` or `python main.py --setup`
- Re-run setup or check status:
  - `python scripts/first_run.py --status`
  - Reset: `python scripts/first_run.py --reset`

6./Updating / retraining
------------------------;

- Regenerate data if you made changes to templates or sources:
  - `python scripts/generate_training_data.py`
- Retrain LoRA with the same or a new style:
  - `python scripts/finetune_lora.py --style calm`
- Re-load model into Ollama or use direct loader after a merge.

7./Resetting everything safely
-----------------------------;

- Reset configuration (keeps models and training data):
  - `python scripts/first_run.py --reset`
- Remove generated datasets and trained models if you want a full rebuild:
  - `rm -r data/training/ data/models/` (Windows PowerShell: `Remove-Item -Recurse data\training, data\models`)

8./Convenience: run full pipeline

--------------------------------;

- A convenience script runs the generate → finetune → load pipeline and stops on failure:
  - `python scripts/run_pipeline.py`

Troubleshooting tips
--------------------;

- If a script fails, read console output and check `logs/` for additional details.
- For slow CPU training, reduce dataset size or train for fewer epochs for experimentation.
- If Ollama fails to import, try the direct loader or inspect the Modelfile created at `data/models/Modelfile.kitsu`.

This tutorial is intentionally short and actionable — for more context, see `README.md` and the script docstrings inside `scripts/`.
