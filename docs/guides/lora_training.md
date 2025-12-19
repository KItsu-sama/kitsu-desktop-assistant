# LoRA Training (Automated Overnight Pipeline)

This guide explains how to generate training data and run unattended LoRA training for Kitsu.

## Dataset generation ✅

- Script: `scripts/generate_training_data.py`
- Output: JSONL at `data/training/kitsu_dataset.jsonl` (one conversation per line)
- Each line:
```
{ "messages": [{"role":"user","content":"..."}, ...], "metadata": {"mood":"...","style":"...","emotion":"..."} }
```

- Example run (generate 10k samples):
```
python scripts/generate_training_data.py --num-samples 10000 --seed 42
```
- Run for X hours:
```
python scripts/generate_training_data.py --duration-hours 8
```
- Notes:
  - Generator keeps assistant replies short and avoids obedience phrases
  - No system/persona text is written to the dataset
  - Files flushed frequently; safe to Ctrl+C (graceful shutdown)

## Training ✅

- Script: `scripts/finetune_lora.py`
- Default dataset path: `data/training/kitsu_dataset.jsonl`
- Core identity LoRA (always first): `kitsu-core` (output: `data/models/kitsu-lora-core`)
- Optional single style LoRA per run: `--style chaotic|sweet|cold|silent` (output: `data/models/kitsu-lora-<style>`)

- Example: Train core and `kitsu-chaotic` style if dataset large enough:
```
python scripts/finetune_lora.py --style chaotic --min-samples 500
```

- Safety features:
  - Skips training if dataset smaller than `--min-samples`
  - Graceful checkpoint save on Ctrl+C
  - Deterministic runs possible with `--seed`

---

For details and advanced usage, see the script docstrings.
