# ============================================================================
# FILE: scripts/generate_dataset.py
# Generate training dataset for Kitsu personality
# ============================================================================

import json
import sys
from pathlib import Path

# Add parent directory to path so we can import from scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def generate_dataset(): 
    """Generate comprehensive training dataset"""
    
    console.print("\n[bold magenta]🦊 KITSU DATASET GENERATOR[/bold magenta]\n")
    console.print("Generating personality training data...\n")
    
    # Import dataset generation
    try:
        # Try relative import first
        from dataset_kitsu import create_expanded_dataset
        from safety_filter import SafetyFilter
    except ImportError:
        # If that fails, try absolute import
        try:
            from scripts.dataset_kitsu import create_expanded_dataset
            from scripts.safety_filter import SafetyFilter
        except ImportError:
            console.print("[red]❌ Error:[/red] Could not import dataset modules")
            console.print("[yellow]Make sure dataset_kitsu.py and safety_filter.py exist in scripts/[/yellow]")
            sys.exit(1)
    
    # Generate raw dataset
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("📚 Creating dataset...", total=None)
        raw_dataset = create_expanded_dataset()
        progress.update(task, completed=True)
    
    console.print(f"  [green]✓[/green] Generated {len(raw_dataset)} examples")
    
    # Apply safety filter
    console.print("\n🛡️  Applying safety filter...")
    filter = SafetyFilter()
    safe_dataset = filter.filter_dataset(raw_dataset)

    console.print(f"  [green]✓[/green] {len(safe_dataset)} examples passed safety filter")

    # Save filtered dataset as JSONL (messages + metadata only)
    output_path = Path("data/training/kitsu_dataset.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in safe_dataset:
            # Write only messages + metadata; safety filter already cleared content
            sample = {"messages": ex.get("messages", []), "metadata": ex.get("metadata", {})}
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    console.print(f"\n[green]✅ Dataset saved:[/green] {output_path}")
    console.print(f"   Total examples: {len(safe_dataset)}")

    
    # Statistics
    moods = {}
    styles = {}
    
    for ex in safe_dataset:
        inst = ex.get("instruction", "")
        if "Mode:" in inst:
            parts = inst.split("|")
            for part in parts:
                if "Mode:" in part:
                    mood = part.split(":")[-1].strip()
                    moods[mood] = moods.get(mood, 0) + 1
                elif "Style:" in part:
                    style = part.split(":")[-1].strip()
                    styles[style] = styles.get(style, 0) + 1
    
    console.print(f"\n📊 Dataset breakdown:")
    if moods:
        console.print(f"   Moods: {dict(sorted(moods.items()))}")
    if styles:
        console.print(f"   Styles: {dict(sorted(styles.items()))}")
    
    console.print("\n[cyan]Next step:[/cyan]")
    console.print("   Run: [yellow]python scripts/finetune_lora.py[/yellow]\n")


if __name__ == "__main__":
    try:
        generate_dataset()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Generation cancelled[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]❌ Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)