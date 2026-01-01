#!/usr/bin/env python3
"""
Complete training pipeline for Kitsu
Single unified script that handles everything

Usage:
  python scripts/train_pipeline.py --full          # Generate + Train + Load
  python scripts/train_pipeline.py --generate-only # Just generate dataset
  python scripts/train_pipeline.py --train-only    # Just train (dataset must exist)
  python scripts/train_pipeline.py --style chaotic # Train specific style
  python scripts/train_pipeline.py --all-styles    # Train all LoRA styles
  python scripts/train_pipeline.py --quickstart    # Quick start mode

"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import shutil
import time

try:
    from scripts import ollama
except ImportError:
    ollama = None  # Fallback

console = Console()
FINAL_MODEL_NAME = "kitsu:character"

# ============================================================================
# STEP 1: Dataset Generation (with safety filter)
# ============================================================================

def generate_dataset(target: str = "base", variant: str = None, num_samples: int = 200):
    """Generate training dataset with safety filtering"""
    
    console.print("\n[bold cyan]📚 STEP 1: Dataset Generation[/bold cyan]\n")
    
    # Import modules
    try:
        from dataset_kitsu import create_expanded_dataset
        from safety_filter import SafetyFilter
    except ImportError:
        console.print("[red]❌ Missing dataset_kitsu.py or safety_filter.py[/red]")
        return False
    
    # Generate raw data
    console.print("Generating samples...")
    raw_dataset = create_expanded_dataset()
    console.print(f"  ✓ Generated {len(raw_dataset)} raw samples")
    
    # Apply safety filter
    console.print("\n🛡️  Applying safety filter...")
    filter = SafetyFilter()
    safe_dataset = filter.filter_dataset(raw_dataset)
    
    if len(safe_dataset) < 50:
        console.print("[red]❌ Not enough safe samples![/red]")
        return False
    
    # Determine output path
    if target == "base":
        output_path = Path("data/training/kitsu_personality.jsonl")
    else:
        variant_name = variant or "all"
        output_path = Path(f"data/training/{target}/kitsu_{variant_name}.jsonl")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in safe_dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    console.print(f"\n[green]✅ Dataset saved:[/green] {output_path}")
    console.print(f"   Safe samples: {len(safe_dataset)}")
    
    return output_path


# ============================================================================
# STEP 2: Training (CPU-friendly LoRA)
# ============================================================================

def train_model(data_path: Path, style: str = None):
    """Train LoRA adapter"""
    
    console.print("\n[bold cyan]🎯 STEP 2: Training[/bold cyan]\n")
    
    if not data_path.exists():
        console.print(f"[red]❌ Dataset not found: {data_path}[/red]")
        return False
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/finetune_lora_direct.py",
        "--data-path", str(data_path),
        "--min-samples", "50",
        "--probe-interval", "500"
    ]
    
    if style:
        cmd.extend(["--style", style])
    
    console.print(f"Command: {' '.join(cmd)}\n")
    
    # Run training
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        console.print("[red]❌ Training failed![/red]")
        return False
    
    console.print("[green]✅ Training complete![/green]")
    return True


# ============================================================================
# STEP 3: Load to Ollama
# ============================================================================

def load_to_ollama():
    """Load merged model into Ollama as kitsu:character"""
    if ollama:
        return ollama.load_to_ollama(ollama.create_modelfile(Path("data/models/kitsu-merged")), model_name=FINAL_MODEL_NAME)
    else:
        console.print("[yellow]⚠️  scripts.ollama not available - using legacy implementation[/yellow]")
        # Legacy implementation (preserved for fallback)
        try:
            result = subprocess.run(
                ["ollama", "list"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )
            console.print("   ✅ Ollama is available")
        except FileNotFoundError:
            console.print("[red]❌ Ollama not installed[/red]")
            console.print("   Install from: https://ollama.ai")
            return False
        except subprocess.CalledProcessError as e:
            console.print(f"[red]❌ Ollama error: {e}[/red]")
            return False
        
        merged_path = Path("data/models/kitsu-merged")
        if not merged_path.exists():
            console.print("[red]❌ Merged model not found[/red]")
            console.print("   Run: python scripts/merge_lora.py")
            return False
        
        console.print(f"   Using: {merged_path.name}")
        
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
        
        console.print(f"   ✅ Modelfile ready")
        
        try:
            list_result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )
            
            if FINAL_MODEL_NAME in list_result.stdout:
                console.print(f"   ⚠️  Model {FINAL_MODEL_NAME} already exists")
                console.print("   Removing old version...")
                subprocess.run(
                    ["ollama", "rm", FINAL_MODEL_NAME],
                    capture_output=True
                )
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not check existing models: {e}[/yellow]")
        
        console.print(f"   Creating model: {FINAL_MODEL_NAME}")
        
        result = subprocess.run(
            ["ollama", "create", FINAL_MODEL_NAME, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )
        
        if result.returncode == 0:
            console.print("[green]✅ Ollama model created successfully![/green]")
            console.print(f"   Name: {FINAL_MODEL_NAME}")
            console.print(f"   Test: [yellow]ollama run {FINAL_MODEL_NAME}[/yellow]")
            return True
        else:
            console.print("[red]❌ Ollama create failed[/red]")
            if result.stderr:
                console.print(f"   Error: {result.stderr}")
            if result.stdout:
                console.print(f"   Output: {result.stdout}")
            return False


# ============================================================================
# Style Training Pipeline (from train_all_styles.py)
# ============================================================================

class StyleTrainingPipeline:
    """
    Complete pipeline for training style-specific LoRA adapters
    
    Workflow:
    1. Generate style-specific training data
    2. Apply safety filters
    3. Train LoRA for each style
    4. Validate outputs
    5. Load to Ollama (optional)
    """
    
    def __init__(self):
        self.data_dir = Path("data/training")
        self.models_dir = Path("data/models")
        self.styles = ["chaotic", "sweet", "cold", "silent"]
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_style_datasets(self):
        """Generate training data for each style"""
        console.print("\n[bold cyan]📚 Step 1: Generating Training Data[/bold cyan]\n")
        
        try:
            from scripts.dataset_kitsu import create_expanded_dataset
        except ImportError:
            console.print("[red]❌ Missing scripts.dataset_kitsu[/red]")
            return {}
        
        console.print("   Generating base dataset...")
        full_dataset = create_expanded_dataset()
        console.print(f"   [green]✓[/green] {len(full_dataset)} total samples\n")
        
        style_datasets = {}
        for style in self.styles:
            style_samples = [
                sample for sample in full_dataset
                if sample.get("metadata", {}).get("style") == style
            ]
            style_datasets[style] = style_samples
        
        for style, samples in style_datasets.items():
            output_path = self.data_dir / f"kitsu_{style}.json"
            formatted = self._format_for_training(samples, style)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted, f, indent=2, ensure_ascii=False)
            
            console.print(f"   [green]✓[/green] {style:10} → {len(formatted):3} samples → {output_path.name}")
        
        return style_datasets
    
    def _format_for_training(self, samples: list, style: str) -> list:
        formatted = []
        for sample in samples:
            metadata = sample.get("metadata", {})
            messages = sample.get("messages", [])
            if len(messages) != 2:
                continue
            user_msg = messages[0].get("content", "")
            assistant_msg = messages[1].get("content", "")
            emotion = metadata.get("emotion", "neutral")
            mood = metadata.get("mood", "behave")
            memory = metadata.get("memory", [])
            context = f"emotion: {emotion} | mood: {mood} | style: {style}"
            if memory:
                memory_str = ", ".join(memory[:2])
                context += f"\nmemory: {memory_str}"
            formatted.append({
                "emotion": emotion,
                "mood": mood,
                "style": style,
                "user": user_msg,
                "assistant": assistant_msg,
                "memory": memory
            })
        return formatted
    
    def train_all_styles(self, probe_interval: int = 500):
        """Train LoRA adapter for each style"""
        console.print("\n[bold cyan]🎯 Step 2: Training LoRA Adapters[/bold cyan]\n")
        
        results = {}
        for style in self.styles:
            console.print(f"\n[yellow]Training style: {style}[/yellow]")
            success = self._train_single_style(style, probe_interval)
            results[style] = success
            status = "[green]✓ Success[/green]" if success else "[red]✗ Failed[/red]"
            console.print(f"   {status}")
        
        from rich.table import Table
        console.print("\n[bold]Training Summary:[/bold]")
        table = Table()
        table.add_column("Style")
        table.add_column("Status")
        table.add_column("Model Path")
        for style, success in results.items():
            status = "[green]✓ Success[/green]" if success else "[red]✗ Failed[/red]"
            model_path = f"data/models/kitsu-lora-{style}"
            table.add_row(style, status, model_path)
        console.print(table)
        return results
    
    def _train_single_style(self, style: str, probe_interval: int) -> bool:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/finetune_lora.py",
                    "--style", style,
                    "--probe-interval", str(probe_interval)
                ],
                capture_output=True,
                text=True,
                timeout=7200
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            console.print(f"[red]Training timeout for {style}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Training error: {e}[/red]")
            return False
    
    def validate_adapters(self):
        """Validate trained adapters"""
        console.print("\n[bold cyan]🧪 Step 3: Validating Adapters[/bold cyan]\n")
        
        test_prompts = {
            "chaotic": "emotion: playful | mood: behave | style: chaotic\n<user>\nHi!\n\n<assistant>\n",
            "sweet": "emotion: happy | mood: behave | style: sweet\n<user>\nHow are you?\n\n<assistant>\n",
            "cold": "emotion: hurt | mood: mean | style: cold\n<user>\nHey\n\n<assistant>\n",
            "silent": "emotion: tired | mood: behave | style: silent\n<user>\nYou okay?\n\n<assistant>\n"
        }
        
        results = {}
        for style in self.styles:
            model_path = self.models_dir / f"kitsu-lora-{style}"
            if not model_path.exists():
                console.print(f"   [red]✗[/red] {style:10} → Model not found")
                results[style] = False
                continue
            required = ["adapter_config.json", "adapter_model.bin", "metadata.json"]
            missing = [f for f in required if not (model_path / f).exists()]
            if missing:
                console.print(f"   [red]✗[/red] {style:10} → Missing: {', '.join(missing)}")
                results[style] = False
                continue
            try:
                response = self._test_generation(model_path, test_prompts.get(style, ""))
                if response and len(response) > 10:
                    console.print(f"   [green]✓[/green] {style:10} → Valid")
                    console.print(f"      Sample: {response[:60]}...")
                    results[style] = True
                else:
                    console.print(f"   [yellow]?[/yellow] {style:10} → Weak output")
                    results[style] = True
            except Exception as e:
                console.print(f"   [yellow]?[/yellow] {style:10} → Cannot test ({e})")
                results[style] = True
        return results
    
    def _test_generation(self, model_path: Path, prompt: str) -> str:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            with open(model_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            base_model_name = metadata.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="cpu"
            )
            model = PeftModel.from_pretrained(base_model, str(model_path))
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.8, do_sample=True)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        except Exception as e:
            raise Exception(f"Generation test failed: {e}")
    
    def load_to_ollama(self):
        """Load trained adapters to Ollama"""
        console.print("\n[bold cyan]📦 Step 4: Loading to Ollama[/bold cyan]\n")
        for style in self.styles:
            console.print(f"   Loading {style}...")
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/load_to_ollama_direct.py",
                        "--style", style
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0:
                    console.print(f"   [green]✓[/green] {style} loaded to Ollama")
                else:
                    console.print(f"   [red]✗[/red] {style} failed to load")
            except Exception as e:
                console.print(f"   [red]✗[/red] {style}: {e}")
    
    def run_full_pipeline(self, load_ollama: bool = False, probe_interval: int = 500):
        """Run complete training pipeline"""
        console.print("\n[bold magenta]🦊 KITSU LORA TRAINING PIPELINE[/bold magenta]")
        console.print("[dim]Training style-specific adapters for emotion-driven behavior[/dim]\n")
        
        self.generate_style_datasets()
        train_results = self.train_all_styles(probe_interval)
        valid_results = self.validate_adapters()
        if load_ollama:
            self.load_to_ollama()
        
        console.print("\n[bold green]✅ Pipeline Complete![/bold green]\n")
        successful = sum(1 for v in train_results.values() if v)
        console.print(f"   Trained: {successful}/{len(self.styles)} styles")
        console.print(f"   Models saved to: {self.models_dir}")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("   1. Update data/config.json to use trained models")
        console.print("   2. Start Kitsu: [yellow]python main.py[/yellow]")
        console.print("   3. Emotion-driven LoRA switching will be automatic!")


# ============================================================================
# Quickstart Flow (from deprecated quickstart_lora.py)
# ============================================================================

def run_quickstart():
    """Quickstart flow for training (deprecated but preserved)"""
    console.print("\n[bold magenta]🦊 KITSU LORA QUICK START[/bold magenta]\n")
    console.print("[dim]Complete setup in one run (deprecated - use --full instead)[/dim]\n")
    
    # Check files
    required_files = [
        "scripts/dataset_kitsu.py",
        "scripts/convert_dataset.py",
        "scripts/finetune_lora.py",
        "core/llm/lora_manager.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if not path.exists():
            console.print(f"   [red]✗[/red] {file_path} [red]MISSING[/red]")
            all_exist = False
        else:
            console.print(f"   [green]✓[/green] {file_path}")
    
    if not all_exist:
        console.print("\n[red]❌ Missing required files![/red]")
        return False
    
    # Generate dataset
    if not generate_dataset():
        return False
    
    # Train core
    dataset_path = Path("data/training/kitsu_personality.jsonl")
    if not train_model(dataset_path):
        return False
    
    # Load to Ollama
    load_to_ollama()
    
    console.print("\n[bold green]✨ Quickstart Complete![/bold green]")
    return True


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Kitsu Training Pipeline")
    
    # Mode flags
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--generate-only", action="store_true", help="Only generate dataset")
    parser.add_argument("--train-only", action="store_true", help="Only train (dataset must exist)")
    parser.add_argument("--all-styles", action="store_true", help="Train all LoRA styles")
    parser.add_argument("--quickstart", action="store_true", help="Quickstart mode (deprecated)")
    
    # Dataset options
    parser.add_argument("--target", choices=["base", "mood", "style"], default="base")
    parser.add_argument("--variant", type=str, help="Specific variant (e.g., chaotic, behave)")
    parser.add_argument("--num-samples", type=int, default=200)
    
    # Training options
    parser.add_argument("--style", type=str, help="Train specific style adapter")
    parser.add_argument("--data-path", type=Path, help="Override dataset path")
    parser.add_argument("--include-dev-feedback", action="store_true", help="Append developer overrides/ratings to dataset before training")
    parser.add_argument("--include-ratings", action="store_true", help="Include /rate ratings when merging dev feedback")
    parser.add_argument("--min-rating", type=int, help="Minimum numeric rating to include when merging ratings", default=None)
    parser.add_argument("--load-ollama", action="store_true", help="Load trained models to Ollama after training")
    parser.add_argument("--probe-interval", type=int, default=500, help="Probe interval for training (0 to disable)")
    
    args = parser.parse_args()
    
    # Default to full pipeline if no mode specified
    if not (args.full or args.generate_only or args.train_only or args.all_styles or args.quickstart):
        args.full = True
    
    console.print("\n[bold magenta]🦊 KITSU TRAINING PIPELINE[/bold magenta]\n")
    
    # Handle deprecated quickstart
    if args.quickstart:
        console.print("[yellow]⚠️  --quickstart is deprecated. Use --full instead.[/yellow]\n")
        return 0 if run_quickstart() else 1
    
    # Handle all-styles
    if args.all_styles:
        pipeline = StyleTrainingPipeline()
        try:
            pipeline.run_full_pipeline(
                load_ollama=args.load_ollama,
                probe_interval=args.probe_interval
            )
            return 0
        except Exception as e:
            console.print(f"\n[red]❌ Pipeline error: {e}[/red]")
            import traceback
            traceback.print_exc()
            return 1
    
    success = True
    dataset_path = args.data_path or Path("data/training/kitsu_personality.jsonl")
    
    # Step 1: Generate
    if args.full or args.generate_only:
        dataset_path = generate_dataset(
            target=args.target,
            variant=args.variant,
            num_samples=args.num_samples
        )
        
        if not dataset_path:
            console.print("\n[red]❌ Pipeline failed at generation[/red]")
            return 1
        
        if args.generate_only:
            console.print("\n[green]✅ Generation complete![/green]")
            return 0
    
    # Step 2: Train
    if args.full or args.train_only:
        # Optionally merge developer feedback into the dataset
        if args.include_dev_feedback:
            try:
                from scripts.convert_dev_feedback import merge_feedback_into_dataset
                merged = Path("data/training/merged_with_dev_feedback.jsonl")
                console.print("\n[cyan]🔁 Merging developer feedback into dataset...[/cyan]")
                merge_feedback_into_dataset(
                    dataset_path,
                    merged,
                    include_overrides=True,
                    include_ratings=args.include_ratings,
                    min_rating=args.min_rating,
                )
                dataset_path = merged
                console.print(f"   Merged dataset: {merged}")
            except Exception as e:
                console.print(f"\n[red]❌ Failed to merge developer feedback: {e}[/red]")
                return 1

        if not train_model(dataset_path, style=args.style):
            console.print("\n[red]❌ Pipeline failed at training[/red]")
            return 1
    
    # Step 3: Load
    if args.full and args.load_ollama:
        load_to_ollama()
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold green]✅ PIPELINE COMPLETE![/bold green]")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  1. Test: [yellow]python main.py[/yellow]")
    console.print("  2. Check: [yellow]/lora status[/yellow] in chat")
    console.print("\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)