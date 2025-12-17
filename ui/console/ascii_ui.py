from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

console = Console()

class AsciiUI:
    def banner(self, title="KITSU", subtitle="Desktop VTuber Assistant"):
        console.print(
            Panel.fit(
                f"[bold magenta]🦊 {title}[/bold magenta]\n[white]{subtitle}[/white]",
                border_style="magenta",
                box=box.DOUBLE_EDGE
            )
        )

    def section(self, title):
        console.print(f"\n[bold cyan]{title}[/bold cyan]")
        console.print("[cyan]" + "─"*60 + "[/cyan]")

    def scrollable(self, text):
        console.print(Markdown(text))
