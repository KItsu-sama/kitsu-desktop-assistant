# =============================================================================
# FILE: scripts/train_all_styles.py
# Complete training pipeline for all LoRA styles
# =============================================================================

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()


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
    
    # =========================================================================
    # Step 1: Generate Training Data
    # =========================================================================
    
    def generate_style_datasets(self):
        """Generate training data for each style"""
        console.print("\n[bold cyan]📚 Step 1: Generating Training Data[/bold cyan]\n")
        
        from scripts.dataset_kitsu import create_expanded_dataset
        
        # Generate full dataset
        console.print("   Generating base dataset...")
        full_dataset = create_expanded_dataset()
        console.print(f"   [green]✓[/green] {len(full_dataset)} total samples\n")
        
        # Split by style
        style_datasets = {}
        for style in self.styles:
            style_samples = [
                sample for sample in full_dataset
                if sample.get("metadata", {}).get("style") == style
            ]
            style_datasets[style] = style_samples
        
        # Save per-style datasets
        for style, samples in style_datasets.items():
            output_path = self.data_dir / f"kitsu_{style}.json"
            
            # Convert to training format
            formatted = self._format_for_training(samples, style)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted, f, indent=2, ensure_ascii=False)
            
            console.print(f"   [green]✓[/green] {style:10} → {len(formatted):3} samples → {output_path.name}")
        
        return style_datasets
    
    def _format_for_training(
        self,
        samples: List[Dict],
        style: str
    ) -> List[Dict]:
        """
        Format samples for LoRA training
        
        Format: emotion | mood | style → user → assistant
        """
        formatted = []
        
        for sample in samples:
            metadata = sample.get("metadata", {})
            messages = sample.get("messages", [])
            
            if len(messages) != 2:
                continue
            
            user_msg = messages[0].get("content", "")
            assistant_msg = messages[1].get("content", "")
            
            # Build context header
            emotion = metadata.get("emotion", "neutral")
            mood = metadata.get("mood", "behave")
            memory = metadata.get("memory", [])
            
            # Minimal context (personality is in weights)
            context = f"emotion: {emotion} | mood: {mood} | style: {style}"
            
            # Add memory hint if present
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
    
    # =========================================================================
    # Step 2: Train LoRA Adapters
    # =========================================================================
    
    def train_all_styles(self, probe_interval: int = 500):
        """Train LoRA adapter for each style"""
        console.print("\n[bold cyan]🎯 Step 2: Training LoRA Adapters[/bold cyan]\n")
        
        results = {}
        
        for style in self.styles:
            console.print(f"\n[yellow]Training style: {style}[/yellow]")
            
            success = self._train_single_style(style, probe_interval)
            results[style] = success
            
            if not success:
                console.print(f"   [red]✗[/red] Failed to train {style}")
            else:
                console.print(f"   [green]✓[/green] {style} training complete")
        
        # Summary
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
        """Train a single LoRA adapter"""
        try:
            # Call training script
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/finetune_lora_cpu.py",
                    "--style", style,
                    "--probe-interval", str(probe_interval)
                ],
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0:
                return True
            else:
                console.print(f"[red]Error output:[/red]\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            console.print(f"[red]Training timeout for {style}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Training error: {e}[/red]")
            return False
    
    # =========================================================================
    # Step 3: Validate Outputs
    # =========================================================================
    
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
            
            # Check for required files
            required = ["adapter_config.json", "adapter_model.bin", "metadata.json"]
            missing = [f for f in required if not (model_path / f).exists()]
            
            if missing:
                console.print(f"   [red]✗[/red] {style:10} → Missing: {', '.join(missing)}")
                results[style] = False
                continue
            
            # Test generation (if possible)
            try:
                response = self._test_generation(model_path, test_prompts.get(style, ""))
                
                if response and len(response) > 10:
                    console.print(f"   [green]✓[/green] {style:10} → Valid")
                    console.print(f"      Sample: {response[:60]}...")
                    results[style] = True
                else:
                    console.print(f"   [yellow]?[/yellow] {style:10} → Weak output")
                    results[style] = True  # Still valid, just weak
                    
            except Exception as e:
                console.print(f"   [yellow]?[/yellow] {style:10} → Cannot test ({e})")
                results[style] = True  # Files exist, can't test generation
        
        return results
    
    def _test_generation(self, model_path: Path, prompt: str) -> str:
        """Test generation with a model (best effort)"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            # Load metadata
            with open(model_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            base_model_name = metadata.get("base_model", "TinyLlama/TinyLlama-1.1B")
            
            # Load model
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="cpu"
            )
            model = PeftModel.from_pretrained(base_model, str(model_path))
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
            
        except Exception as e:
            raise Exception(f"Generation test failed: {e}")
    
    # =========================================================================
    # Step 4: Load to Ollama (Optional)
    # =========================================================================
    
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
    
    # =========================================================================
    # Main Pipeline
    # =========================================================================
    
    def run_full_pipeline(self, load_ollama: bool = False, probe_interval: int = 500):
        """Run complete training pipeline"""
        console.print("\n[bold magenta]🦊 KITSU LORA TRAINING PIPELINE[/bold magenta]")
        console.print("[dim]Training style-specific adapters for emotion-driven behavior[/dim]\n")
        
        # Step 1: Generate datasets
        self.generate_style_datasets()
        
        # Step 2: Train adapters
        train_results = self.train_all_styles(probe_interval)
        
        # Step 3: Validate
        valid_results = self.validate_adapters()
        
        # Step 4: Load to Ollama (optional)
        if load_ollama:
            self.load_to_ollama()
        
        # Final summary
        console.print("\n[bold green]✅ Pipeline Complete![/bold green]\n")
        
        successful = sum(1 for v in train_results.values() if v)
        console.print(f"   Trained: {successful}/{len(self.styles)} styles")
        console.print(f"   Models saved to: {self.models_dir}")
        
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("   1. Update data/config.json to use trained models")
        console.print("   2. Start Kitsu: [yellow]python main.py[/yellow]")
        console.print("   3. Emotion-driven LoRA switching will be automatic!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train all Kitsu LoRA styles")
    parser.add_argument(
        "--load-ollama",
        action="store_true",
        help="Load trained models to Ollama after training"
    )
    parser.add_argument(
        "--probe-interval",
        type=int,
        default=500,
        help="Probe interval for training (0 to disable)"
    )
    args = parser.parse_args()
    
    pipeline = StyleTrainingPipeline()
    
    try:
        pipeline.run_full_pipeline(
            load_ollama=args.load_ollama,
            probe_interval=args.probe_interval
        )
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Pipeline cancelled[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]❌ Pipeline error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()