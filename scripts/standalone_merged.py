#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for Ollama

This creates a standalone merged model that Ollama can use directly
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def merge_lora_adapter(adapter_path: Path, output_path: Path, base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> bool:
    """Delegate merging to canonical implementation."""
    from scripts import ollama as _ollama
    return _ollama.merge_lora_adapter(adapter_path, output_path, base_model=base_model)


def create_modelfile_for_merged(merged_path: Path) -> Path:
    """Create Modelfile for merged model"""
    console.print("\n[cyan]📝 Creating Modelfile...[/cyan]")
    
    modelfile_content = f"""# Kitsu Character Model (Merged)
FROM {merged_path.resolve()}

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

TEMPLATE \"\"\"{{{{ if .System }}}}{{{{ .System }}}}{{{{ end }}}}{{{{ .Prompt }}}}\"\"\"

PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
"""
    
    modelfile_path = Path("data/models/Modelfile.kitsu")
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    
    console.print(f"[green]✅ Modelfile created[/green]")
    return modelfile_path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter for Ollama")
    
    parser.add_argument(
        "--adapter",
        type=str,
        help="Adapter name (e.g., 'core', 'chaotic')"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/kitsu-merged"),
        help="Output directory for merged model"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model to merge with"
    )
    
    parser.add_argument(
        "--load",
        action="store_true",
        help="Also load into Ollama after merging"
    )
    
    args = parser.parse_args()
    
    # Find adapter
    models_dir = Path("data/models")
    adapters = list(models_dir.glob("kitsu-lora-*"))
    
    if not adapters:
        console.print("\n[red]❌ No adapters found[/red]")
        console.print("   Run training first:")
        console.print("   [cyan]python scripts/train_pipeline.py[/cyan]")
        return 1
    
    # Select adapter
    if args.adapter:
        selected = None
        for adapter in adapters:
            if args.adapter.lower() in adapter.name.lower():
                selected = adapter
                break
        
        if not selected:
            console.print(f"\n[red]❌ Adapter '{args.adapter}' not found[/red]")
            console.print("\nAvailable:")
            for adapter in adapters:
                console.print(f"   - {adapter.name}")
            return 1
    else:
        # Use most recent
        selected = sorted(adapters, key=lambda p: p.stat().st_mtime)[-1]
        console.print(f"[cyan]Using latest: {selected.name}[/cyan]")
    
    # Check if output exists
    if args.output.exists():
        console.print(f"\n[yellow]⚠️  Output already exists: {args.output}[/yellow]")
        
        import sys
        if sys.stdin.isatty():
            response = input("Overwrite? [y/N]: ").strip().lower()
            if response != 'y':
                console.print("[yellow]Cancelled[/yellow]")
                return 0
        
        shutil.rmtree(args.output)
    
    # Merge
    success = merge_lora_adapter(
        selected,
        args.output,
        base_model=args.base_model
    )
    
    if not success:
        return 1
    
    # Load into Ollama if requested
    if args.load:
        console.print("\n[cyan]📦 Loading into Ollama...[/cyan]")
        
        try:
            import subprocess
            
            # Create Modelfile
            modelfile = create_modelfile_for_merged(args.output)
            
            # Check if model exists
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            
            if "kitsu:character" in result.stdout:
                console.print("[yellow]Removing existing model...[/yellow]")
                subprocess.run(["ollama", "rm", "kitsu:character"])
            
            # Create model
            console.print("Creating kitsu:character...")
            result = subprocess.run(
                ["ollama", "create", "kitsu:character", "-f", str(modelfile)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                console.print("\n[bold green]✅ Loaded into Ollama![/bold green]")
                console.print("   Test: [yellow]ollama run kitsu:character[/yellow]")
            else:
                console.print(f"\n[red]❌ Ollama load failed[/red]")
                if result.stderr:
                    console.print(f"   {result.stderr}")
        
        except Exception as e:
            console.print(f"\n[yellow]⚠️  Could not load into Ollama: {e}[/yellow]")
    
    console.print("\n[cyan]💡 To load manually:[/cyan]")
    console.print(f"   ollama create kitsu:character -f data/models/Modelfile.kitsu")
    
    return 0


from scripts import ollama as _ollama

def create_modelfile_for_merged(merged_path: Path) -> Path:
    return _ollama.create_modelfile(merged_path)


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