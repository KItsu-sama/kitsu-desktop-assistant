"""Unit tests for LoRAManager

Lightweight tests that exercise discovery, selection, switching and
config persistence using the provided `data/lora_examples` placeholder
adapters.
"""
from pathlib import Path
import json
import os

from core.llm.lora_manager import LoRAManager


def test_discover_and_switch(tmp_path, monkeypatch):
    # Ensure fallback examples are available
    base = Path('data/lora_examples')
    base.mkdir(parents=True, exist_ok=True)
    for name in ('chaotic', 'sweet', 'cold', 'silent'):
        (base / name).mkdir(exist_ok=True)

    mgr = LoRAManager(adapters_dir=base)
    mgr.discover_adapters()

    stats = mgr.get_stats()
    assert stats['total_adapters'] >= 4
    for required in ('chaotic', 'sweet', 'cold', 'silent'):
        assert required in stats['available_styles']

    # Test mapping
    s = mgr.select_for_emotion({'dominant_emotion': 'playful', 'style': 'chaotic'})
    assert s in ('chaotic', 'sweet', None)

    # Test switching and persistence
    cfg = Path('data/config.json')
    if cfg.exists():
        cfg.unlink()

    ok = mgr.switch_adapter('sweet', force=True)
    assert ok is True
    assert mgr.current_style == 'sweet'

    # Check config file created
    assert cfg.exists()
    data = json.loads(cfg.read_text(encoding='utf-8'))
    assert data.get('model', {}).get('style') == 'sweet'
