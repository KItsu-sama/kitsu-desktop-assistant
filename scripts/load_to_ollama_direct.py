# scripts/load_to_ollama_direct.py
"""
Load trained model directly to Ollama
No GGUF conversion needed - Ollama handles it!
Thin CLI wrapper for model loading that delegates to scripts.ollama.
This wrapper preserves the previous CLI interface while using a single
canonical implementation in `scripts.ollama`.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.prompt import Confirm

# Ensure project root is on sys.path when running this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts import ollama
except ImportError:
    ollama = None  # Fallback to local implementations

console = Console()

def find_model_dir():
    """Find trained model directory"""
    console.print("\n[cyan]🔍 Looking for trained model...[/cyan]")
    
    possible_paths = [
        Path("data/models/kitsu-lora-tinyllama"),
        Path("data/models/kitsu-merged"),
        Path("data/models/kitsu-character"),
    ]
    
    for path in possible_paths:
        if path.exists():
            # Check if it's a merged model or LoRA
            has_adapter = (path / "adapter_config.json").exists()
            has_model = (path / "pytorch_model.bin").exists() or \
                       (path / "model.safetensors").exists() or \
                       list(path.glob("*.safetensors"))
            
            if has_model or has_adapter:
                console.print(f"[green]✅ Found model:[/green] {path}")
                model_type = "LoRA adapters" if has_adapter else "Merged model"
                console.print(f"   Type: {model_type}")
                return path
    
    console.print("[red]❌ No trained model found![/red]")
    console.print("\n[yellow]💡 Train a model first:[/yellow]")
    console.print("   python scripts/finetune_lora.py")
    return None

def merge_lora_if_needed(model_dir: Path):
    """Merge LoRA adapters if needed"""
    adapter_config = model_dir / "adapter_config.json"
    
    if not adapter_config.exists():
        console.print("[green]✅ Model already merged[/green]")
        return model_dir
    
    console.print("\n[cyan]🔗 LoRA adapters detected - merging...[/cyan]")
    
    merged_dir = Path("data/models/kitsu-merged")
    
    if merged_dir.exists():
        if Confirm.ask(f"Merged model exists at {merged_dir}. Use existing?", default=True):
            return merged_dir
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        # Load metadata
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        base_model_name = metadata.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        console.print(f"   Base model: {base_model_name}")
        
        # Load and merge
        console.print("   Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        console.print("   Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        
        console.print("   Merging...")
        model = model.merge_and_unload()
        
        console.print(f"   Saving to {merged_dir}...")
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        
        console.print("[green]✅ Model merged successfully[/green]")
        return merged_dir
        
    except Exception as e:
        console.print(f"[red]❌ Merge failed: {e}[/red]")
        return None

def create_modelfile(model_dir: Path):
    """Create Ollama Modelfile automatically (always overwrite)"""
    console.print("\n[cyan]📝 Creating Modelfile (auto)...[/cyan]")

    modelfile_content = f"""# Kitsu Character Model
FROM {model_dir.resolve()}

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

TEMPLATE \"\"\"{{{{ if .System }}}}
<|system|>
{{{{ .System }}}}
{{{{ end }}}}
<|user|>
{{{{ .Prompt }}}}
<|assistant|>
\"\"\"

PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "<|system|>"
PARAMETER stop "</s>"
"""

    modelfile_path = Path("data/models/Modelfile.kitsu")
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    console.print(f"[green]✅ Modelfile ready:[/green] {modelfile_path}")
    return modelfile_path

def load_to_ollama(modelfile_path: Path):
    """Load model into Ollama"""
    console.print("\n[cyan]📦 Loading model to Ollama...[/cyan]")
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
        # Check if Ollama is running
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=5
        )
        
        if result.returncode != 0:
            console.print("[red]❌ Ollama not running![/red]")
            console.print("\n[yellow]💡 Start Ollama:[/yellow]")
            console.print("   ollama serve")
            return False
        
        # Create model
        console.print(f"   Creating model: {model_name}")
        console.print("   This may take a few minutes...")
        
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            console.print(f"[green]✅ Model loaded successfully![/green]")
            console.print(f"   Model name: {model_name}")
            return True
        else:
            console.print(f"[red]❌ Failed to load model[/red]")
            console.print(f"   Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        console.print("[red]❌ Ollama not installed![/red]")
        console.print("\n[yellow]💡 Install Ollama:[/yellow]")
        console.print("   https://ollama.ai/download")
        return False
    except subprocess.TimeoutExpired:
        console.print("[red]❌ Ollama not responding[/red]")
        return False

def test_model():
    """Test the loaded model"""
    console.print("\n[cyan]🧪 Testing model...[/cyan]")
    
    test_prompt = "emotion: happy | mood: behave | style: chaotic\\nUser: Hi!\\nKitsu:"
    
    try:
        result = subprocess.run(
            ["ollama", "run", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", test_prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=60  # Increased timeout for first run
        )
        
        if result.returncode == 0:
            console.print("\n[green]✅ Test successful![/green]")
            console.print(f"\n[cyan]Response:[/cyan]")
            response = result.stdout.strip()
            if response:
                console.print(f"   {response}")
            else:
                console.print("   [dim](empty response - model may still be loading)[/dim]")
            return True
        else:
            console.print("[yellow]⚠️  Test had issues but model may still work[/yellow]")
            console.print(f"   Try manually: ollama run TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            return True  # Return True anyway since model loaded
            
    except subprocess.TimeoutExpired:
        console.print("[yellow]⚠️  Test timed out (model may still be loading)[/yellow]")
        console.print("   Model is installed, try: [yellow]ollama run TinyLlama/TinyLlama-1.1B-Chat-v1.0[/yellow]")
        return True  # Return True anyway since model loaded
    except Exception as e:
        console.print(f"[yellow]⚠️  Test error: {e}[/yellow]")
        console.print("   Model should still work, test manually")
        return True

def update_config():
    """Update Kitsu config to use new model"""
    console.print("\n[cyan]⚙️  Updating configuration...[/cyan]")
    
    config_path = Path("data/config.json")
    
    if not config_path.exists():
        console.print("[yellow]⚠️  config.json not found[/yellow]")
        return
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        config["model"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        console.print("[green]✅ Config updated![/green]")
        console.print("   Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not update config: {e}[/yellow]")
        console.print("   Update manually: data/config.json")

def main():
    """Main function with CLI support"""
    parser = argparse.ArgumentParser(description="Load Kitsu models into Ollama")

    parser.add_argument(
        "--adapter",
        type=str,
        help="Specific adapter to load (e.g., 'chaotic', 'core')"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check status only (don't load)"
    )

    args = parser.parse_args()

    # Check status mode
    if args.check:
        if ollama:
            ollama.show_status()
        else:
            console.print("[yellow]⚠️  scripts.ollama not available - status check skipped[/yellow]")
        return 0

    # Check Ollama available
    if ollama and not ollama.check_ollama_available():
        console.print("\n[red]❌ Ollama not available[/red]")
        console.print("   Make sure Ollama is installed and running")
        console.print("   Install from: https://ollama.ai")
        return 1

    # Get available adapters
    if ollama:
        adapters = ollama.list_available_adapters()
    else:
        # Fallback: use local find_model_dir
        model_dir = find_model_dir()
        adapters = [{"name": "local-model", "path": model_dir}] if model_dir else []

    if not adapters:
        console.print("\n[red]❌ No adapters found[/red]")
        console.print("   Run training first:")
        console.print("   [cyan]python scripts/train_pipeline.py[/cyan]")
        return 1

    # Select adapter
    if args.adapter:
        selected = None
        for adapter in adapters:
            if args.adapter.lower() in adapter["name"].lower():
                selected = adapter
                break

        if not selected:
            console.print(f"\n[red]❌ Adapter '{args.adapter}' not found[/red]")
            console.print("\nAvailable:")
            for adapter in adapters:
                console.print(f"   - {adapter['name']}")
            return 1
    else:
        selected = adapters[0]
        console.print(f"[cyan]Using latest adapter: {selected['name']}[/cyan]")

    # Load model (delegated or local)
    if ollama:
        success = ollama.load_model(selected["path"], force=args.force)
    else:
        # Fallback to local flow
        model_dir = selected["path"]
        model_dir = merge_lora_if_needed(model_dir)
        if not model_dir:
            return 1
        modelfile = create_modelfile(model_dir)
        success = load_to_ollama(modelfile)
        if success:
            test_model()
            update_config()

    return 0 if success else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)