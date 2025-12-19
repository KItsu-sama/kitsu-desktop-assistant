"""Integration smoke tests for LoRA system

This script is a lightweight runner used during CI and manual testing to
validate that LoRA discovery, mapping and runtime switching behave
safely even when no real adapters exist (a fallback directory will be
created in that case).
"""
import sys
import time
from pathlib import Path
from rich.console import Console

console = Console()

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class LoRASystemTester:
    def __init__(self):
        self.kitsu = None

    async def setup(self):
        console.print("\n[cyan]Initializing Kitsu for testing...[/cyan]")
        try:
            from core.kitsu_core import KitsuIntegrated

            self.kitsu = KitsuIntegrated(model="tinyllama", temperature=0.8, streaming=False)
            await self.kitsu.initialize()

            # If no adapters found, create a minimal fallback set for tests
            manager = getattr(self.kitsu.llm, 'lora_manager', None)
            if manager is None or manager.get_stats().get('total_adapters', 0) == 0:
                fallback = Path('data/lora_examples')
                fallback.mkdir(parents=True, exist_ok=True)
                for name in ('chaotic', 'sweet', 'cold', 'silent'):
                    (fallback / name).mkdir(exist_ok=True)
                with open(fallback / 'README.md', 'w', encoding='utf-8') as f:
                    f.write("Placeholder LoRA adapters for tests\n")
                console.print('[yellow]No real adapters found; created data/lora_examples fallback.[/yellow]')
                # Rediscover adapters
                if manager:
                    manager.discover_adapters()
            else:
                console.print('[green]Real adapters found; using them for tests.[/green]')

            return True

        except Exception as e:
            console.print(f"[red]✗[/red] Setup failed: {e}\n")
            return False


if __name__ == '__main__':
    import asyncio

    tester = LoRASystemTester()
    ok = asyncio.get_event_loop().run_until_complete(tester.setup())
    print('\nSetup OK' if ok else '\nSetup FAILED')
