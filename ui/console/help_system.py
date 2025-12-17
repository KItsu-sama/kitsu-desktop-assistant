from .ascii_ui import AsciiUI
from rich.panel import Panel

class HelpSystem:
    def __init__(self):
        self.ui = AsciiUI()

    def show(self, dev_mode=False):
        self.ui.banner("Help Menu")

        basic = """
### User Commands
- /mood <type> — Change emotion mode
- /style <type> — Change speaking style
- /state — Show current state
- /clear — Reset memory
- /quit — Exit program
"""

        dev = """
### Developer Commands
- /res_fine_tune — Overwrite last response
- /rate_res — Rate response quality
- /error_log — View recent errors
- /devtools — Open full dev console
"""
        self.ui.section("User Commands")
        self.ui.scrollable(basic)

        if dev_mode:
            self.ui.section("Developer Tools")
            self.ui.scrollable(dev)
