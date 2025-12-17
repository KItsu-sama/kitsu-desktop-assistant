"""
Simple smoke tests for Kitsu functionalities:
- Create emotion engine
- Test set_mood with persistence
- Test clear_mood_override
- Test ConsoleRouter dev commands instantiation
"""

import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.personality.emotion_engine import EmotionEngine
from core.dev.console_router import ConsoleRouter

# Ensure config file exists for test
config_path = Path('data/config.json')
if not config_path.exists():
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({'mode':'text','greet_on_startup':True}, indent=2))

# Test emotion engine persist
engine = EmotionEngine()
engine.set_mood('flirty', duration=10, persist=True)
print('Mood after set (runtime):', engine.mood)
# Read persisted config
with open('data/config.json','r',encoding='utf-8') as f:
    cfg = json.load(f)
print('Manual override in config:', cfg.get('manual_mood_override'))

# Clear override and check
engine.clear_mood_override()
with open('data/config.json','r',encoding='utf-8') as f:
    cfg = json.load(f)
print('Manual override after clear:', cfg.get('manual_mood_override'))

# Test console router instantiation
router = ConsoleRouter(memory=None, logger=None)
print('ConsoleRouter OK - trainer:', hasattr(router, 'trainer'))

print('Smoke tests completed')
