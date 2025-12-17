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

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for item in raw:
        emotion = item.get("emotion", "neutral")
        mood = item.get("mood", "behave")
        style = item.get("style", "calm")
        # If a style_filter is provided, skip items that don't match. This
        # enables training separate LoRA adapters per style without mixing.
        if style_filter and style != style_filter:
            continue
        memory = item.get("memory", [])
        user = item.get("user", "")
        assistant = item.get("assistant", "")

        # Inject up to 2 memories before the user turn; keep them as internal context
        memory_line = ""
        if memory:
            memory_line = "memory: " + ", ".join(memory[:2]) + "\n\n"

        # Enforce <continue> placement rules for training: if <continue> appears
        # inside an assistant chunk with trailing text, split into two assistant
        # segments so that the first ends with <continue> and is followed by a
        # second <assistant> chunk. This makes continuation behavior explicit
        # during training and avoids learning unsafe placements.
        if "<continue>" in assistant:
            idx = assistant.find("<continue>")
            before = assistant[:idx].strip()
            after = assistant[idx + len("<continue>") :].strip()
            if after:
                # Create two assistant segments in the same sample
                assistant = f"{before} <continue>\n\n<assistant>\n{after}"
            else:
                assistant = f"{before} <continue>"

        text = (
            f"emotion: {emotion} | mood: {mood} | style: {style}\n"
            f"{memory_line}"
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
    parser.add_argument("--style", type=str, default="chaotic", help="Style name for this LoRA adapter (chaotic/calm)")
    parser.add_argument("--probe-interval", type=int, default=500, help="Probe interval in steps (0 to disable)")
    args_cli = parser.parse_args()

    STYLE = args_cli.style or "chaotic"
    PROBE_INTERVAL = max(0, int(args_cli.probe_interval))

    data_path = Path("data/training/kitsu_personality.json")
    # Output dir includes style to allow multiple LoRA adapters on the same base
    output_dir = Path(f"data/models/kitsu-lora-{STYLE}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📚 Loading dataset...")
    dataset = load_training_data(data_path, style_filter=STYLE)
    print(f"✅ Samples: {len(dataset)}")

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
    trainer.train()

    print("\n💾 Saving...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": model_name,
        "samples": len(dataset),
        "lora_r": 8,
        "device": device,
        "format": "emotion | mood | style + <user>/<assistant>",
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Training complete")
    print(f"📁 Saved to: {output_dir}")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
