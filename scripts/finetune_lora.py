# scripts/finetune_lora_cpu.py
"""
Clean CPU-friendly LoRA training for Kitsu
- Single stable chat format
- No instruction headers
- No padding poisoning
- Real ETA estimation
- Safe for CPU / GT 730
"""

import argparse
import json
import math
import time
import random
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)

# =========================
# Hardware detection
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🦊 Kitsu LoRA Training")
print("=" * 60)
print(f"💻 Device: {device}")

if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
else:
    print("   ⚠️ CPU mode (slow but stable)")

# =========================
# Dataset loading
# =========================


def load_training_data(path: Path, style_filter: Optional[str] = None) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    rows = []
    # Expect JSONL where each line is a conversation with messages[] and metadata
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            item = json.loads(ln)
            meta = item.get("metadata", {})
            mood = meta.get("mood", "behave")
            style = meta.get("style", "calm")
            emotion = meta.get("emotion", "neutral")

            if style_filter and style != style_filter:
                continue

            messages = item.get("messages", [])
            # Validate and extract user/assistant pairs
            for i in range(0, len(messages) - 1, 2):
                u = messages[i]
                a = messages[i + 1]
                if u.get("role") != "user" or a.get("role") != "assistant":
                    # skip malformed pairs
                    continue

                user = u.get("content", "").strip()
                assistant = a.get("content", "").strip()

                if not user or not assistant:
                    continue

                # Ensure no system/persona markers leaked into content
                forbidden = ["you are kitsu", "you are the assistant", "as an ai"]
                combined_lower = (user + " " + assistant).lower()
                if any(f in combined_lower for f in forbidden):
                    continue

                text = (
                    f"emotion: {emotion} | mood: {mood} | style: {style}\n"
                    f"<user>\n{user}\n\n"
                    f"<assistant>\n{assistant}"
                )

                rows.append({"text": text})

    return Dataset.from_list(rows)


# =========================
# Tokenization
# =========================


def tokenize(batch):
    texts = batch["text"]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=256,
        padding=False,
    )

    labels = []
    for i, (text, ids) in enumerate(zip(texts, enc["input_ids"])):
        # Everything before <assistant> is ignored
        assistant_idx = text.find("<assistant>")
        if assistant_idx == -1:
            labels.append([-100] * len(ids))
            continue

        prefix = tokenizer(
            text[:assistant_idx], add_special_tokens=False
        )["input_ids"]
        lbl = [-100] * len(prefix) + ids[len(prefix):]

        # Ensure assistant outputs always end with explicit EOS token so the
        # model learns a clear end-of-response. We append eos token id when
        # missing (quiet, training-only change; no effect on eval/run-time).
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and ids[-1] != eos_id:
            ids = ids + [eos_id]
            # Update the encoded input ids in-place
            enc["input_ids"][i] = ids
            if "attention_mask" in enc:
                enc["attention_mask"][i] = enc["attention_mask"][i] + [1]

        # Recompute label slice to match potentially updated ids
        lbl = lbl[: len(ids)]
        labels.append(lbl)

    enc["labels"] = labels
    return enc



# =========================
# ETA estimation
# =========================


def estimate_eta(num_samples, args):
    steps_per_epoch = math.ceil(
        num_samples
        / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * args.num_train_epochs

    # Conservative real-world numbers
    sec_per_step = 120 if device == "cpu" else 0.4
    total_seconds = total_steps * sec_per_step

    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)

    return total_steps, h, m, s


# =========================
# Training
# =========================


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default=None, help="Optional single style to train (chaotic/sweet/cold/silent)")
    parser.add_argument("--min-samples", type=int, default=400, help="Minimum samples required to start a training job")
    parser.add_argument("--probe-interval", type=int, default=500, help="Probe interval in steps (0 to disable)")
    parser.add_argument("--data-path", type=str, default="data/training/kitsu_dataset.jsonl", help="JSONL dataset path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic runs (0 = system)")
    parser.add_argument("--no-core", action="store_true", help="Do not train kitsu-core; only train specified style")
    args_cli = parser.parse_args()

    STYLE = args_cli.style
    PROBE_INTERVAL = max(0, int(args_cli.probe_interval))
    MIN_SAMPLES = int(args_cli.min_samples)
    DATA_PATH = Path(args_cli.data_path)
    SEED = None if args_cli.seed == 0 else int(args_cli.seed)

    # Sanity: allowed styles
    ALLOWED_STYLES = {"chaotic", "sweet", "cold", "silent"}
    if STYLE and STYLE not in ALLOWED_STYLES:
        raise ValueError(f"Unknown style: {STYLE}. Allowed: {', '.join(sorted(ALLOWED_STYLES))}")

    # Output dirs
    core_output = Path("data/models/kitsu-lora-core")
    style_output = Path(f"data/models/kitsu-lora-{STYLE}") if STYLE else None

    print("\n📚 Loading dataset...")
    # Core training uses all samples
    all_ds = load_training_data(DATA_PATH, style_filter=None)
    print(f"✅ Total samples available: {len(all_ds)}")

    print("\n📥 Loading model...")
    # Use the non-chat base model to avoid chat-template behavior during fine-tuning
    model_name = "TinyLlama/TinyLlama-1.1B"

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    print("✅ Model loaded")

    print("\n🔧 Applying LoRA...")
    lora = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # deterministic seed
    if SEED is not None:
        random.seed(SEED)
        torch.manual_seed(SEED)

    # =========================
    # Probe callback for early overfit detection (lightweight, training-only)
    # =========================
    PROBE_PROMPT = "Please respond in one short sentence: Hello!"

    class ProbeCallback(TrainerCallback):
        def __init__(self, prompt: str, interval: int, tokenizer, device: str):
            self.prompt = prompt
            self.interval = interval
            self.tokenizer = tokenizer
            self.device = device

        def on_step_end(self, args, state, control, **kwargs):
            # kwargs may contain the model
            model = kwargs.get("model")
            if model is None or self.interval <= 0:
                return
            step = int(state.global_step)
            if step % self.interval != 0:
                return

            # Run a tiny generation probe without affecting gradients
            try:
                model.eval()
                with torch.no_grad():
                    tokens = self.tokenizer(self.prompt, return_tensors="pt").to(self.device)
                    out = model.generate(**tokens, max_new_tokens=24)
                    text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    print(f"[PROBE] step={step} probe_out={text}")
            except Exception as e:
                print(f"[PROBE] failed: {e}")
            finally:
                model.train()

    print("\n📝 Tokenizing...")
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=20,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=device == "cuda",
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    steps, h, m, s = estimate_eta(len(tokenized), args)

    print("\n⏳ Training estimate")
    print(f"   Steps: {steps}")
    print(f"   ETA: ~{h}h {m}m {s}s")
    print("   (Estimate stabilizes after a few steps)")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
        callbacks=[ProbeCallback(PROBE_PROMPT, PROBE_INTERVAL, tokenizer, device)],
    )

    print("\n🎯 Training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user — saving checkpoint...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        raise

    print("\n💾 Saving...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": model_name,
        "samples": len(dataset),
        "lora_r": 8,
        "device": device,
        "format": "emotion | mood | style + <user>/<assistant>",
        "model_type": "lora",
        "style": STYLE if STYLE else "core",
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Training complete")
    print(f"📁 Saved to: {output_dir}")


def _run_training_job(dataset: Dataset, output_dir: Path, style_name: str, probe_interval: int, seed: int):
    # Create a minimal wrapper around the original training body so we can
    # call it for core and for an optional style adapter.
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device_local = device
    if device_local == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B",
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    print("✅ Model loaded")

    print("\n🔧 Applying LoRA...")
    lora = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=20,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=device_local == "cuda",
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    steps, h, m, s = estimate_eta(len(tokenized), args)

    print("\n⏳ Training estimate")
    print(f"   Steps: {steps}")
    print(f"   ETA: ~{h}h {m}m {s}s")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[ProbeCallback("Please respond in one short sentence: Hello!", probe_interval, tokenizer, device_local)],
    )

    print("\n🎯 Training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user — saving checkpoint...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        return False

    print("\n💾 Saving...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": "TinyLlama/TinyLlama-1.1B",
        "samples": len(dataset),
        "lora_r": 8,
        "device": device_local,
        "format": "emotion | mood | style + <user>/<assistant>",
        "model_type": "lora",
        "style": style_name,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Training complete")
    print(f"📁 Saved to: {output_dir}")
    return True


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(130)

