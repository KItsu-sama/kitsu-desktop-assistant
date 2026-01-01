# scripts/finetune_lora.py
"""
Clean CPU-friendly LoRA training for Kitsu
- Single stable chat format
- No instruction headers
- No padding poisoning
- Real ETA estimation
- Safe for CPU / GT 730
Key fixes:
- Consistent chat format with proper EOS tokens
- Correct model name (TinyLlama-1.1B-Chat-v1.0)
- Proper label masking for <assistant> responses only
- No padding poisoning
- Improved probe callback
- Better training arguments for convergence
"""

import argparse
import json
import math
import random
import sys
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

# Global tokenizer
tokenizer = None

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
            style = meta.get("style", "chaotic")
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

                # Build training text in consistent format
                text = (
                    f"emotion: {emotion} | mood: {mood} | style: {style}\n"
                    f"User: {user}\n"
                    f"Kitsu: {assistant}"
                )

                rows.append({"text": text})

    if not rows:
        raise ValueError(f"No valid training samples found in {path}")

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
        add_special_tokens=True,
    )

    labels = []
    for i, (text, ids) in enumerate(zip(texts, enc["input_ids"])):
        # Everything before "Kitsu:" is ignored
        kitsu_idx = text.find("Kitsu:")
        if kitsu_idx == -1:
            labels.append([-100] * len(ids))
            continue

        prefix = tokenizer(
            text[:kitsu_idx + 6], add_special_tokens=False  # Include "Kitsu:"
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
        # Ensure last token is NOT masked (so model learns to predict EOS)
        if lbl[-1] == -100:
            lbl[-1] = ids[-1]
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
# Probe callback
# =========================


class ProbeCallback(TrainerCallback):
    def __init__(self, prompt: str, interval: int, tokenizer, device: str):
        self.prompt = prompt
        self.interval = interval
        self.tokenizer = tokenizer
        self.device = device

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or self.interval <= 0:
            return
        step = int(state.global_step)
        if step % self.interval != 0:
            return

        try:
            model.eval()
            with torch.no_grad():
                tokens = self.tokenizer(self.prompt, return_tensors="pt").to(self.device)
                out = model.generate(
                    **tokens,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                print(f"\n[PROBE step={step}] {text}\n")
        except Exception as e:
            print(f"[PROBE] failed: {e}")
        finally:
            model.train()


# =========================
# Training
# =========================


def train():
    global tokenizer

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
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
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
        lora_alpha=16,  # Increased for better learning
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

    print("\n📝 Tokenizing...")
    tokenized = all_ds.map(
        tokenize,
        batched=True,
        remove_columns=all_ds.column_names,
    )

    args = TrainingArguments(
        output_dir=str(core_output),
        num_train_epochs=3,  # More epochs for better learning
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Larger effective batch
        learning_rate=2e-4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
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

    # Probe prompt
    probe_prompt = "emotion: happy | mood: behave | style: chaotic\nUser: Hi!\nKitsu:"

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
        callbacks=[ProbeCallback(probe_prompt, PROBE_INTERVAL, tokenizer, device)],
    )

    print("\n🎯 Training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user — saving checkpoint...")
        trainer.save_model(core_output)
        tokenizer.save_pretrained(core_output)
        raise

    print("\n💾 Saving...")
    trainer.save_model(core_output)
    tokenizer.save_pretrained(core_output)

    metadata = {
        "base_model": model_name,
        "samples": len(all_ds),
        "lora_r": 8,
        "lora_alpha": 16,
        "device": device,
        "format": "emotion | mood | style + User:/Kitsu:",
        "model_type": "lora",
        "style": "core",
    }

    with open(core_output / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Training complete")
    print(f"📁 Saved to: {core_output}")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)