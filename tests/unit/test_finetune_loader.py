import json
from pathlib import Path
# Import module by path so tests don't require package installation
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("finetune_lora", Path("scripts/finetune_lora.py"))
finetune_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune_mod)
load_training_data = finetune_mod.load_training_data


def make_jsonl(tmp_path, conversations):
    p = tmp_path / "data.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    return p


def test_load_training_data_filters_and_parses(tmp_path):
    convs = [
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "metadata": {"mood": "behave", "style": "chaotic", "emotion": "happy"},
        },
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "metadata": {"mood": "behave", "style": "sweet", "emotion": "calm"},
        },
    ]

    path = make_jsonl(tmp_path, convs)
    ds_all = load_training_data(path, style_filter=None)
    assert len(ds_all) == 2

    ds_chaotic = load_training_data(path, style_filter="chaotic")
    assert len(ds_chaotic) == 1
