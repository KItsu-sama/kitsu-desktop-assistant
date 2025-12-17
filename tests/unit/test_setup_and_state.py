import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from main import load_config, KitsuApplication
from core.personality.kitsu_self import KitsuSelf
from core.personality.emotion_engine import EmotionEngine


def cleanup_files(paths):
    for p in paths:
        try:
            Path(p).unlink()
        except Exception:
            pass


def test_save_state_handles_empty(tmp_path, monkeypatch):
    # Prepare empty state file
    mem_dir = Path('data/memory')
    mem_dir.mkdir(parents=True, exist_ok=True)
    bad_state = mem_dir / 'kitsu_state.json'
    bad_state.write_text('')

    ks = KitsuSelf()
    ks.save_state(bad_state)

    content = bad_state.read_text(encoding='utf-8')
    data = json.loads(content)
    assert 'state' in data
    assert 'kitsu_self' in data['state']

    # clean up
    cleanup_files([bad_state])


def test_first_meet_applies_defaults(tmp_path, monkeypatch):
    # Force config absence
    cfg_path = Path('data/config.json')
    if cfg_path.exists():
        cfg_path.unlink()

    # Load defaults and initialize app
    config = load_config()
    app = KitsuApplication(config)

    async def run_and_apply():
        await app.initialize()
        # Force non-interactive by patching isatty
        import sys
        orig_isatty = getattr(sys.stdin, 'isatty', None)
        try:
            sys.stdin.isatty = lambda: False
            await app.kitsu._handle_command('/first_meet')
        finally:
            if orig_isatty is not None:
                sys.stdin.isatty = orig_isatty

        # After applying defaults, config and personality should exist
        assert Path('data/config.json').exists()
        persona = json.loads(Path('data/config/personality.json').read_text(encoding='utf-8'))
        assert 'default_mood' in persona
        assert 'default_style' in persona

        # If engine present, default_mood should reflect override persistence
        if app.kitsu.emotion_engine:
            state = app.kitsu.emotion_engine.get_state_dict()
            # The wizard default is 'behave'; if a default was set, mood should be set or persisted
            assert isinstance(state['mood'], str)

        await app.shutdown_handler.cleanup()

    asyncio.run(run_and_apply())


def test_mood_persist_and_clear(tmp_path):
    cfg = Path('data/config.json')
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({'mode':'text','greet_on_startup': True}, indent=2))

    engine = EmotionEngine()
    engine.set_mood('flirty', duration=10, persist=True)
    # Confirm runtime mood
    assert engine.mood == 'flirty'

    with open('data/config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        assert cfg.get('manual_mood_override') is not None

    engine.clear_mood_override()
    with open('data/config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        assert cfg.get('manual_mood_override') is None
