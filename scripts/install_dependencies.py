# ============================================================================
# FILE: scripts/install_dependencies.py
# One-time installation of all required packages
# ============================================================================

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def check_python_version():
    """Ensure Python 3.8+"""
    if sys.version_info < (3, 8):
        console.print("[red]❌ Python 3.8+ required![/red]")
        console.print(f"   Current: {sys.version}")
        sys.exit(1)
    console.print(f"[green]✓[/green] Python {sys.version_info.major}.{sys.version_info.minor}")

def install_package(package: str, progress):
    """Install a single package with progress indicator"""
    task = progress.add_task(f"Installing {package}...", total=None)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        progress.update(task, completed=True)
        return True
    except subprocess.CalledProcessError:
        progress.update(task, completed=True)
        return False

def main():
    """Install all dependencies for Kitsu"""
    
    console.print("\n[bold magenta]🦊 KITSU DEPENDENCY INSTALLER[/bold magenta]\n")
    
    # Check Python version
    check_python_version()
    
    # Package categories
    packages = {
        "Core ML & Training": [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "datasets>=2.14.0",
            "peft>=0.6.0",
            "bitsandbytes>=0.41.0",
            "accelerate>=0.24.0",
            "trl>=0.7.0",
            "sentence-transformers>=2.2.0",
        ],
        "Optimized Inference": [
            "llama-cpp-python",
            "optimum",
            "onnxruntime",
        ],
        "NLP & Embeddings": [
            "spacy>=3.7.0",
            "nltk>=3.8.0",
        ],
        "Data Processing": [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
        ],
        "UI & Visualization": [
            "rich>=13.0.0",
            "prompt_toolkit>=3.0.0",
        ],
        "System Monitoring": [
            "psutil>=5.9.0",
            "gputil>=1.4.0",
        ],
        "Utilities": [
            "tqdm>=4.65.0",
            "requests>=2.31.0",
            "aiohttp>=3.9.0",
        ]
    }
    
    # Optional packages (for full features)
    optional_packages = {
        "Voice & TTS": [
            "pyttsx3",
            "SpeechRecognition",
        ],
        "API Integration": [
            "anthropic",
            "openai",
        ]
    }
    
    total_packages = sum(len(pkgs) for pkgs in packages.values())
    
    console.print(f"📦 Installing {total_packages} core packages...\n")
    
    failed = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for category, pkgs in packages.items():
            console.print(f"\n[cyan]{category}:[/cyan]")
            
            for pkg in pkgs:
                success = install_package(pkg, progress)
                if not success:
                    failed.append(pkg)
                    console.print(f"  [red]✗[/red] {pkg}")
                else:
                    console.print(f"  [green]✓[/green] {pkg}")
    
    # Optional packages
    console.print("\n[yellow]📦 Optional packages (type 'y' to install, 'n' to skip):[/yellow]\n")
    
    for category, pkgs in optional_packages.items():
        response = input(f"Install {category}? (y/n): ").lower().strip()
        if response == 'y':
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                for pkg in pkgs:
                    install_package(pkg, progress)
    
    # Summary
    console.print("\n" + "="*60)
    if failed:
        console.print(f"[yellow]⚠️  {len(failed)} packages failed to install:[/yellow]")
        for pkg in failed:
            console.print(f"  - {pkg}")
        console.print("\n[yellow]You can install them manually later.[/yellow]")
    else:
        console.print("[green]✅ All packages installed successfully![/green]")
    
    console.print("="*60 + "\n")
    
    # Next steps
    console.print("[bold cyan]🚀 Next Steps:[/bold cyan]")
    console.print("  1. Run setup wizard: [white]python scripts/setup_wizard.py[/white]")
    console.print("  2. Generate dataset: [white]python scripts/generate_dataset.py[/white]")
    console.print("  3. Train model: [white]python scripts/finetune_lora.py[/white]")
    console.print("  4. Start Kitsu: [white]python main.py[/white]\n")

if __name__ == "__main__":
    main()

