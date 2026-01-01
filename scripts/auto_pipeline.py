# ============================================================================
# FILE: scripts/auto_pipeline.py
# auto-training system with proper imports and error handling
# ============================================================================

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import subprocess
from typing import Optional, Dict, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# ============================================================================
# Knowledge Base Processor
# ============================================================================

class KnowledgeProcessor:
    """Process raw text files into training samples"""
    
    def __init__(self):
        self.knowledge_dir = Path("data/knowledge")
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-tagging rules
        self.file_tags = {
            "facts.txt": {"emotion": "neutral", "mood": "behave", "style": "chaotic"},
            "trivia.txt": {"emotion": "playful", "mood": "behave", "style": "chaotic"},
            "technical.txt": {"emotion": "helpful", "mood": "behave", "style": "sweet"},
            "science.txt": {"emotion": "curious", "mood": "behave", "style": "sweet"},
            "history.txt": {"emotion": "interested", "mood": "behave", "style": "chaotic"},
            "gaming.txt": {"emotion": "playful", "mood": "behave", "style": "chaotic"},
            "culture.txt": {"emotion": "interested", "mood": "behave", "style": "sweet"},
            "language.txt": {"emotion": "helpful", "mood": "behave", "style": "chaotic"},
        }
    
    def process_all(self) -> List[Dict]:
        """Process all knowledge files"""
        all_samples = []
        
        console.print("\n[cyan]📚 Processing knowledge base...[/cyan]")
        
        for file_path in self.knowledge_dir.glob("*.txt"):
            try:
                samples = self._process_file(file_path)
                all_samples.extend(samples)
                console.print(f"   ✅ {file_path.name}: {len(samples)} samples")
            except Exception as e:
                console.print(f"   [yellow]⚠️  {file_path.name}: {e}[/yellow]")
        
        if not all_samples:
            console.print("   [yellow]No valid samples found[/yellow]")
        else:
            console.print(f"[green]✅ Total knowledge samples: {len(all_samples)}[/green]")
        
        return all_samples
    
    def _process_file(self, file_path: Path) -> List[Dict]:
        samples = []

        tags = self.file_tags.get(file_path.name, {
            "emotion": "neutral",
            "mood": "behave",
            "style": "chaotic"
        })

        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if len(line) < 10:
                continue

            sample = self._create_qa_pair(line, tags)
            if sample:
                samples.append(sample)

        return samples

    
    def _create_qa_pair(self, fact: str, tags: Dict) -> Optional[Dict]:
        """Convert fact into question/answer pair - FIXED"""
        
        # Generate natural question
        words = fact.split()[:5]
        question = f"Tell me about {' '.join(words)}"  # Fixed: join words first
        
        # Build minimal context
        context = (
            f"<context>\n"
            f"user_name: User\n"
            f"emotion: {tags['emotion']}\n"
            f"mood: {tags['mood']}\n"
            f"style: {tags['style']}\n"
            f"memory: []\n"
            f"user_info: {{}}\n"
            f"</context>\n\n"
        )
        
        return {
            "messages": [
                {"role": "user", "content": context + question},
                {"role": "assistant", "content": fact}
            ],
            "metadata": tags
        }


# ============================================================================
# Memory Miner
# ============================================================================

class MemoryMiner:
    """Extract training samples from conversations"""
    
    def __init__(self, memory_path: Path = Path("data/memory/memory.json")):
        self.memory_path = memory_path
    
    def mine_samples(self, min_score: float = 0.6) -> List[Dict]:
        """Extract high-quality conversation pairs"""
        
        console.print("\n[cyan]⛏️  Mining memory...[/cyan]")
        
        if not self.memory_path.exists():
            console.print("   [yellow]No memory file - skipping[/yellow]")
            return []
        
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            sessions = data.get("sessions", [])
            
            samples = []
            prev_msg = None
            
            for msg in sessions:
                if prev_msg and \
                   prev_msg.get("role") == "user" and \
                   msg.get("role") in ["kitsu", "assistant"] and \
                   msg.get("score", 0) >= min_score:
                    
                    emotion = msg.get("emotion", "neutral")
                    context = (
                        f"<context>\n"
                        f"user_name: User\n"
                        f"emotion: {emotion}\n"
                        f"mood: behave\n"
                        f"style: chaotic\n"
                        f"memory: []\n"
                        f"user_info: {{}}\n"
                        f"</context>\n\n"
                    )
                    
                    samples.append({
                        "messages": [
                            {"role": "user", "content": context + prev_msg["text"]},
                            {"role": "assistant", "content": msg["text"]}
                        ],
                        "metadata": {
                            "emotion": emotion,
                            "mood": "behave",
                            "style": "chaotic",
                            "source": "memory"
                        }
                    })
                
                prev_msg = msg
            
            console.print(f"[green]✅ Mined {len(samples)} samples[/green]")
            return samples
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Mining failed: {e}[/yellow]")
            return []


# ============================================================================
# Auto Training Pipeline
# ============================================================================

class AutoPipeline:
    """Automated training pipeline - fixed version"""
    
    def __init__(self, config_path: Path = Path("data/training/auto_config.json")):
        self.config_path = config_path
        self.config = self._load_config()
        self.start_time = time.time()
        
        # Output paths
        self.output_dir = Path("data/training/auto")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.combined_dataset = self.output_dir / "combined_dataset.jsonl"
        self.log_file = self.output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    def _load_config(self) -> Dict:
        """Load or create config"""
        default_config = {
            "num_synthetic_samples": 500,
            "enable_knowledge_processing": True,
            "enable_memory_mining": True,
            "min_memory_score": 0.6,
            "training": {
                "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Fixed model name
                "style": None,
                "probe_interval": 500,
                "min_samples": 50  # Lower minimum
            },
            "auto_load_ollama": True,
            "auto_test": True
        }
        
        if self.config_path.exists():
            try:
                config = json.loads(self.config_path.read_text(encoding="utf-8"))
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except:
                pass
        
        # Save default config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(default_config, indent=2))
        return default_config
    
    async def run(self):
        """Run complete pipeline"""
        
        console.print("\n[bold magenta]" + "="*60 + "[/bold magenta]")
        console.print("[bold magenta]🦊 KITSU AUTO TRAINING PIPELINE[/bold magenta]")
        console.print("[bold magenta]" + "="*60 + "[/bold magenta]\n")
        
        console.print(f"[cyan]Started:[/cyan] {datetime.now():%Y-%m-%d %H:%M:%S}")
        console.print(f"[cyan]Config:[/cyan] {self.config_path}\n")
        
        try:
            # Step 1: Gather training data
            all_samples = await self._gather_data()
            
            # Check if we have enough data
            if len(all_samples) < self.config["training"]["min_samples"]:
                console.print(f"\n[red]❌ Not enough samples ({len(all_samples)} < {self.config['training']['min_samples']})[/red]")
                console.print("\n[yellow]💡 Add more facts:[/yellow]")
                console.print("   python scripts/quick_add.py --init-samples")
                return False
            
            # Step 2: Safety filter
            all_samples = self._apply_safety_filter(all_samples)
            
            # Step 3: Save dataset
            self._save_dataset(all_samples)
            
            # Step 4: Train model
            success = await self._train_model()
            
            if not success:
                console.print("\n[red]❌ Training failed[/red]")
                return False
            
            # Step 5: Load to Ollama
            if self.config.get("auto_load_ollama", True):
                await self._load_to_ollama()
            
            # Step 6: Test
            if self.config.get("auto_test", True):
                await self._test_model()
            
            # Done!
            self._print_summary()
            return True
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
            return False
        except Exception as e:
            console.print(f"\n[red]❌ Pipeline failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False
    
    async def _gather_data(self) -> List[Dict]:
        """Gather all training data"""
        all_samples = []
        
        # 1. Knowledge base
        if self.config.get("enable_knowledge_processing", True):
            processor = KnowledgeProcessor()
            knowledge_samples = processor.process_all()
            all_samples.extend(knowledge_samples)
        
        # 2. Memory mining
        if self.config.get("enable_memory_mining", True):
            miner = MemoryMiner()
            memory_samples = miner.mine_samples(
                min_score=self.config.get("min_memory_score", 0.6)
            )
            all_samples.extend(memory_samples)
        
        # 3. Synthetic samples - Load inline to avoid import issues
        console.print("\n[cyan]🎭 Generating synthetic samples...[/cyan]")
        synthetic_samples = self._generate_synthetic_samples()
        all_samples.extend(synthetic_samples)
        
        console.print(f"\n[bold green]📊 Total samples: {len(all_samples)}[/bold green]")
        return all_samples
    
    def _generate_synthetic_samples(self) -> List[Dict]:
        """Generate synthetic samples inline - no external imports"""
        samples = []
        
        # Basic greeting samples
        greetings = [
            ("Hi!", "Heyyyy! What's up? ✨"),
            ("Hello", "Oh. Hi."),
            ("Good morning!", "Good morning~ How are you? 💕"),
            ("Hey there", "Oh my~ Someone's here! 💕"),
        ]
        
        contexts = [
            {"emotion": "happy", "mood": "behave", "style": "chaotic"},
            {"emotion": "neutral", "mood": "behave", "style": "cold"},
            {"emotion": "calm", "mood": "behave", "style": "sweet"},
            {"emotion": "playful", "mood": "flirty", "style": "chaotic"},
        ]
        
        for (user, assistant), ctx in zip(greetings, contexts):
            context = (
                f"<context>\n"
                f"user_name: User\n"
                f"emotion: {ctx['emotion']}\n"
                f"mood: {ctx['mood']}\n"
                f"style: {ctx['style']}\n"
                f"memory: []\n"
                f"user_info: {{}}\n"
                f"</context>\n\n"
            )
            
            samples.append({
                "messages": [
                    {"role": "user", "content": context + user},
                    {"role": "assistant", "content": assistant}
                ],
                "metadata": ctx
            })
        
        console.print(f"[green]✅ Generated {len(samples)} synthetic samples[/green]")
        return samples
    
    def _apply_safety_filter(self, samples: List[Dict]) -> List[Dict]:
        """Basic safety filter inline"""
        console.print("\n[cyan]🛡️  Applying safety filter...[/cyan]")
        
        blocked_words = [
            "nsfw", "porn", "xxx", "sex", "nude", "explicit",
            "kill", "murder", "suicide", "weapon", "drug"
        ]
        
        filtered = []
        for sample in samples:
            messages = sample.get("messages", [])
            if len(messages) >= 2:
                assistant_text = messages[1].get("content", "").lower()
                
                # Check if any blocked word present
                if not any(word in assistant_text for word in blocked_words):
                    filtered.append(sample)
        
        removed = len(samples) - len(filtered)
        console.print(f"   ✅ Kept: {len(filtered)} samples")
        console.print(f"   🛡️  Filtered: {removed} samples")
        
        return filtered
    
    def _save_dataset(self, samples: List[Dict]):
        """Save combined dataset"""
        console.print(f"\n[cyan]💾 Saving dataset...[/cyan]")
        
        with open(self.combined_dataset, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        console.print(f"[green]✅ Saved {len(samples)} samples to {self.combined_dataset}[/green]")
    
    async def _train_model(self) -> bool:
        """Train model"""
        console.print("\n[cyan]🎯 Starting training...[/cyan]")
        console.print("[dim]This will take a while...[/dim]\n")
        
        cmd = [
            sys.executable,
            "scripts/finetune_lora.py",
            "--data-path", str(self.combined_dataset),
            "--probe-interval", str(self.config["training"]["probe_interval"]),
            "--min-samples", str(self.config["training"]["min_samples"])
        ]
        
        if self.config["training"].get("style"):
            cmd.extend(["--style", self.config["training"]["style"]])
        
        try:
            result = subprocess.run(cmd, check=False)
            
            if result.returncode == 0:
                console.print("\n[green]✅ Training complete![/green]")
                return True
            else:
                console.print("\n[red]❌ Training failed[/red]")
                return False
                
        except Exception as e:
            console.print(f"\n[red]❌ Training error: {e}[/red]")
            return False
    
    async def _load_to_ollama(self):
        """Load to Ollama"""
        console.print("\n[cyan]📦 Loading to Ollama...[/cyan]")
        
        try:
            result = subprocess.run(
                [sys.executable, "scripts/load_to_ollama_direct.py"],
                check=False
            )
            
            if result.returncode == 0:
                console.print("[green]✅ Loaded to Ollama[/green]")
            else:
                console.print("[yellow]⚠️  Ollama load issues[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Ollama load failed: {e}[/yellow]")
    
    async def _test_model(self):
        """Test model"""
        console.print("\n[cyan]🧪 Testing...[/cyan]")
        
        test_prompt = (
            "<context>\n"
            "user_name: User\n"
            "emotion: happy\n"
            "mood: behave\n"
            "style: chaotic\n"
            "memory: []\n"
            "user_info: {}\n"
            "</context>\n\n"
            "Hi!"
        )
        
        try:
            result = subprocess.run(
                ["ollama", "run", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", test_prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and result.stdout.strip():
                console.print("\n[green]✅ Test successful![/green]")
                console.print(f"\n[cyan]Response:[/cyan] {result.stdout.strip()}")
            else:
                console.print("[yellow]⚠️  Test had issues[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Test failed: {e}[/yellow]")
    
    def _print_summary(self):
        """Print summary"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        console.print("\n[bold green]" + "="*60 + "[/bold green]")
        console.print("[bold green]✅ PIPELINE COMPLETE![/bold green]")
        console.print("[bold green]" + "="*60 + "[/bold green]\n")
        
        console.print(f"[cyan]Total time:[/cyan] {hours}h {minutes}m")
        console.print(f"[cyan]Dataset:[/cyan] {self.combined_dataset}")
        
        console.print("\n[bold cyan]🎯 Next:[/bold cyan]")
        console.print("   python main.py")


# ============================================================================
# CLI
# ============================================================================

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto training pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/training/auto_config.json"),
        help="Config file"
    )
    
    args = parser.parse_args()
    
    pipeline = AutoPipeline(config_path=args.config)
    success = await pipeline.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye! 🦊[/yellow]")
        sys.exit(0)