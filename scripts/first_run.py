"""
scripts/first_run.py

Utility to manage first-run state:
- status: show whether initial setup completed
- run: run the interactive setup wizard
- reset: remove saved config and first-run flag (prompts before destructive actions)

No external dependencies; uses the existing SetupWizard.
"""
import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.prompt import Confirm

console = Console()

FLAG = Path("data/.first_run_complete")
CONFIG = Path("data/config.json")


def status():
    if FLAG.exists():
        console.print("[green]First-run setup completed[/green]")
        try:
            with open(CONFIG, 'r', encoding='utf-8') as f:
                import json
                cfg = json.load(f)
                console.print(f"Model: [yellow]{cfg.get('model')}[/yellow]")
                console.print(f"Mode: [yellow]{cfg.get('mode', 'text')}[/yellow]")
        except Exception:
            pass
    else:
        console.print("[yellow]No first-run completion flag found. Run setup to create one.[/yellow]")


def run_wizard():
    try:
        from scripts.setup_wizard import SetupWizard
        wiz = SetupWizard()
        wiz.run()
        console.print("[green]Setup finished.[/green]")
    except Exception as e:
        console.print(f"[red]Failed to run setup wizard: {e}[/red]")
        sys.exit(1)


def reset(confirm: bool = False):
    if not confirm:
        if not Confirm.ask("This will remove data/config.json and the first-run flag. Continue?", default=False):
            console.print("[cyan]Aborted.[/cyan]")
            return
    # Remove config file and flag, but keep models and training data
    try:
        if CONFIG.exists():
            CONFIG.unlink()
        flag = FLAG
        if flag.exists():
            flag.unlink()
        # Also remove config folder if empty
        cfg_dir = Path("data/config")
        if cfg_dir.exists() and not any(cfg_dir.iterdir()):
            cfg_dir.rmdir()
        console.print("[green]Reset complete. You can re-run setup with: python scripts/first_run.py --run[/green]")
    except Exception as e:
        console.print(f"[red]Reset failed: {e}[/red]")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Manage first-run setup state')
    parser.add_argument('--status', action='store_true', help='Show first-run status')
    parser.add_argument('--run', action='store_true', help='Run the interactive setup wizard')
    parser.add_argument('--reset', action='store_true', help='Reset the first-run flag and config (destructive)')
    args = parser.parse_args()

    if args.status:
        status()
    elif args.run:
        run_wizard()
    elif args.reset:
        reset()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
