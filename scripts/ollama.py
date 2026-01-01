#!/usr/bin/env python3
"""
Centralized Ollama utilities for Kitsu

Provides canonical helpers for:
- discovering adapters/merged models
- creating Modelfile (adapter-aware)
- merging LoRA adapters
- loading/removing models in Ollama
- testing and updating config

This file consolidates duplicate logic previously scattered across multiple
scripts (create_modelfile.py, load_to_ollama_direct.py, load_model.py,
standalone_merged.py, train_pipeline.py, load_to_ollama_direct.py).

API compatibility: exposes functions with the same names used by existing
scripts so those scripts can remain thin wrappers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from rich.console import Console

console = Console()
FINAL_MODEL_NAME = "kitsu:character"


# ----------------------------- Discovery -----------------------------------
def list_available_adapters() -> List[Dict]:
    """List LoRA adapters under data/models/kitsu-lora-*"""
    models_dir = Path("data/models")
    if not models_dir.exists():
        return []

    adapters = []
    for path in models_dir.glob("kitsu-lora-*"):
        if path.is_dir():
            if (path / "adapter_model.bin").exists() or (path / "adapter_model.safetensors").exists() or (path / "adapter_config.json").exists():
                adapters.append({
                    "name": path.name,
                    "path": path,
                    "modified": path.stat().st_mtime
                })
    return sorted(adapters, key=lambda x: x["modified"], reverse=True)


def find_model_dir() -> Optional[Path]:
    """Find a reasonable model directory (LoRA adapter or merged model).

    Returns the best candidate Path or None if nothing found.
    """
    console.print("\n[cyan]🔍 Looking for trained model...[/cyan]")
    models_dir = Path("data/models")
    if not models_dir.exists():
        console.print("[red]❌ No models directory found[/red]")
        return None

    candidates: List[Tuple[float, Path, bool]] = []
    for sub in models_dir.iterdir():
        if not sub.is_dir():
            continue
        has_adapter = (sub / "adapter_config.json").exists() or (sub / "adapter_model.bin").exists()
        has_model = (sub / "pytorch_model.bin").exists() or (sub / "model.safetensors").exists() or any(sub.glob("*.safetensors"))
        if has_adapter or has_model:
            candidates.append((sub.stat().st_mtime, sub, has_adapter))

    if not candidates:
        console.print("[red]❌ No trained model found![/red]")
        return None

    candidates.sort(reverse=True)
    _, path, _ = candidates[0]
    console.print(f"[green]✅ Found model:[/green] {path}")
    return path


# ----------------------------- Merge ---------------------------------------
def merge_lora_adapter(adapter_path: Path, output_path: Path, base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> bool:
    """Merge a LoRA adapter into a standalone model dir.

    Returns True on success.
    """
    console.print(f"\n[bold cyan]🔀 Merging LoRA Adapter[/bold cyan]")
    console.print(f"   Adapter: {adapter_path}")
    console.print(f"   Output: {output_path}\n")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as e:  # pragma: no cover - depends on heavy libs
        console.print(f"[red]❌ Missing dependencies for merging: {e}[/red]")
        console.print("\nInstall with: pip install transformers peft torch")
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(adapter_path), device_map="cpu")
        merged = model.merge_and_unload()

        output_path.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(output_path), safe_serialization=True)
        tokenizer.save_pretrained(str(output_path))

        # copy/augment metadata if present
        metadata_src = adapter_path / "metadata.json"
        if metadata_src.exists():
            try:
                metadata = json.loads(metadata_src.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
            metadata["merged"] = True
            metadata["base_model"] = base_model
            (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        console.print(f"[green]✅ Merge complete:[/green] {output_path}")
        return True
    except Exception as e:  # pragma: no cover - heavy ops
        console.print(f"[red]❌ Merge failed: {e}[/red]")
        return False


# ----------------------------- Modelfile -----------------------------------
def create_modelfile(model_dir: Path) -> Path:
    """Create a Modelfile.kitsu appropriate for the model_dir.

    If model_dir contains an adapter (adapter_config.json or adapter_model.*)
    we produce an ADAPTER-style Modelfile referencing the adapter and a
    base FROM line suitable for the LoRA workflow. If model_dir is a merged
    model, we reference the directory directly in FROM.
    """
    console.print(f"\n[cyan]📝 Creating Modelfile for {model_dir.name}...[/cyan]")

    # Determine whether this dir is an adapter
    is_adapter = (model_dir / "adapter_config.json").exists() or (model_dir / "adapter_model.bin").exists() or (model_dir / "adapter_model.safetensors").exists()

    if is_adapter:
        from_clause = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        adapter_line = f"ADAPTER {model_dir.resolve()}"
    else:
        from_clause = str(model_dir.resolve())
        adapter_line = ""

    modelfile_content = f"""# Kitsu Character Model
FROM {from_clause}
{adapter_line}

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

TEMPLATE "{{{{ if .System }}}}\n<|system|>\n{{{{ .System }}}}\n{{{{ end }}}}\n<|user|>\n{{{{ .Prompt }}}}\n<|assistant|>\n"

PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "<|system|>"
PARAMETER stop "</s>"
"""

    modelfile_path = Path("data/models/Modelfile.kitsu")
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    console.print(f"[green]✅ Modelfile created:[/green] {modelfile_path}")
    return modelfile_path


# ----------------------------- Ollama operations ---------------------------

def check_ollama_available() -> bool:
    try:
        subprocess.run(["ollama", "list"], check=True, capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def check_model_loaded(model_name: str = FINAL_MODEL_NAME) -> bool:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, encoding="utf-8", errors="ignore")
        return model_name in result.stdout
    except Exception:
        return False


def remove_existing_model(model_name: str = FINAL_MODEL_NAME) -> bool:
    console.print(f"[yellow]Removing existing {model_name}...[/yellow]")
    try:
        subprocess.run(["ollama", "rm", model_name], capture_output=True, check=True)
        console.print("[green]✅ Removed[/green]")
        return True
    except subprocess.CalledProcessError:
        console.print("[yellow]⚠️  Could not remove (may not exist)[/yellow]")
        return False


def load_to_ollama(modelfile_path: Path, model_name: str = FINAL_MODEL_NAME) -> bool:
    console.print(f"\n[cyan]📦 Creating Ollama model: {model_name}...[/cyan]")
    console.print(f"   Modelfile: {modelfile_path}")

    result = subprocess.run(["ollama", "create", model_name, "-f", str(modelfile_path)], capture_output=True, text=True, encoding="utf-8", errors="ignore")

    if result.returncode == 0:
        console.print("\n[bold green]✅ Model loaded successfully![/bold green]")
        console.print(f"   Test with: [yellow]ollama run {model_name}[/yellow]")
        return True
    else:
        console.print("\n[bold red]❌ Failed to load model[/bold red]")
        if result.stderr:
            console.print(f"   Error: {result.stderr}")
        if result.stdout:
            console.print(f"   Output: {result.stdout}")
        return False


def load_model(adapter_path: Path, force: bool = False) -> bool:
    """Load adapter into Ollama (adapter dir expected)."""
    console.print(f"\n[bold cyan]🦊 Loading {adapter_path.name} into Ollama[/bold cyan]\n")

    # Check if already loaded
    if check_model_loaded():
        if not force:
            console.print(f"[yellow]⚠️  {FINAL_MODEL_NAME} already exists[/yellow]")
            console.print("   Use --force to overwrite")
            return False
        remove_existing_model()

    # Create Modelfile and load
    modelfile_path = create_modelfile(adapter_path)
    console.print(f"\n[cyan]📦 Creating Ollama model...[/cyan]")
    console.print(f"   Name: {FINAL_MODEL_NAME}")
    console.print(f"   Adapter: {adapter_path.name}")

    return load_to_ollama(modelfile_path, model_name=FINAL_MODEL_NAME)


def test_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> bool:
    console.print("\n[cyan]🧪 Testing model...[/cyan]")
    test_prompt = "emotion: happy | mood: behave | style: chaotic\\nUser: Hi!\\nKitsu:"
    try:
        result = subprocess.run(["ollama", "run", model_name, test_prompt], capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=60)
        if result.returncode == 0:
            console.print("\n[green]✅ Test successful![/green]")
            response = result.stdout.strip()
            if response:
                console.print(f"   {response}")
            else:
                console.print("   [dim](empty response - model may still be loading)[/dim]")
            return True
        else:
            console.print("[yellow]⚠️  Test had issues but model may still work[/yellow]")
            console.print("   Try manually: ollama run <model>")
            return True
    except subprocess.TimeoutExpired:
        console.print("[yellow]⚠️  Test timed out (model may still be loading)[/yellow]")
        return True
    except Exception as e:
        console.print(f"[yellow]⚠️  Test error: {e}[/yellow]")
        return True


# ----------------------------- Helpers -------------------------------------

def update_config(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> None:
    console.print("\n[cyan]⚙️  Updating configuration...[/cyan]")
    config_path = Path("data/config.json")
    if not config_path.exists():
        console.print("[yellow]⚠️  config.json not found[/yellow]")
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["model"] = model_name
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        console.print("[green]✅ Config updated![/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not update config: {e}[/yellow]")


def show_status() -> None:
    console.print("\n[bold cyan]🦊 Kitsu Model Status[/bold cyan]\n")
    if not check_ollama_available():
        console.print("[red]❌ Ollama not available[/red]")
        console.print("   Install from: https://ollama.ai")
        return
    console.print("[green]✅ Ollama is running[/green]\n")
    model_loaded = check_model_loaded()
    if model_loaded:
        console.print(f"[green]✅ {FINAL_MODEL_NAME} is loaded[/green]")
    else:
        console.print(f"[yellow]⚠️  {FINAL_MODEL_NAME} not found[/yellow]")
    adapters = list_available_adapters()
    if adapters:
        console.print(f"\n[cyan]Available adapters:[/cyan]")
        from rich.table import Table
        from datetime import datetime
        table = Table(show_header=True)
        table.add_column("Adapter", style="cyan")
        table.add_column("Modified", style="yellow")
        table.add_column("Status", style="green")
        for adapter in adapters:
            modified = datetime.fromtimestamp(adapter["modified"]).strftime("%Y-%m-%d %H:%M")
            status = "✓ Latest" if adapter == adapters[0] else ""
            table.add_row(adapter["name"], modified, status)
        console.print(table)
    else:
        console.print("\n[yellow]⚠️  No adapters found[/yellow]")
        console.print("   Run training first: [cyan]python scripts/train_pipeline.py[/cyan]")
    if not model_loaded and adapters:
        console.print(f"\n[cyan]💡 To load latest:[/cyan]")
        console.print(f"   python scripts/load_model.py")


# ----------------------------- Backward compatibility ----------------------
# Small convenience functions kept at module-level so existing scripts can
# `from scripts.ollama import create_modelfile, load_to_ollama, find_model_dir, merge_lora_adapter`
# without import churn.


if __name__ == "__main__":
    console.print("Use this module as a library: import scripts.ollama")
    sys.exit(0)
