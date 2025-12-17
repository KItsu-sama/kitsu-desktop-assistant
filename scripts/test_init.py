import asyncio
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import load_config, KitsuApplication

async def run_test():
    config = load_config()
    app = KitsuApplication(config)
    await app.initialize()
    # Wait a moment to let background tasks start
    await asyncio.sleep(2)
    # Test /first_meet - run setup wizard and apply defaults to running instance
    try:
        # Force non-interactive mode for headless tests
        import sys
        orig_isatty = getattr(sys.stdin, 'isatty', None)
        try:
            sys.stdin.isatty = lambda: False
            await app.kitsu._handle_command('/first_meet')
        finally:
            if orig_isatty is not None:
                sys.stdin.isatty = orig_isatty
    except Exception as e:
        print('first_meet failed:', e)
    # Call cleanup
    # Create an intentionally invalid/empty state file and verify save_state handles it
    from pathlib import Path
    bad_state_path = Path('data/memory/kitsu_state.json')
    bad_state_path.parent.mkdir(parents=True, exist_ok=True)
    bad_state_path.write_text('')
    try:
        app.kitsu.kitsu_self.save_state(bad_state_path)
        content = bad_state_path.read_text()
        import json
        parsed = json.loads(content)
        print('kitsu_state.json sanity: OK')
    except Exception as e:
        print('kitsu_state.json sanity: FAILED', e)

    await app.shutdown_handler.cleanup()

if __name__ == '__main__':
    asyncio.run(run_test())
