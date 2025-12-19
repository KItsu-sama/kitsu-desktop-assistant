# Import by path so tests run without installing package
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("dataset_kitsu", Path("scripts/dataset_kitsu.py"))
dk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dk)
create_training_sample = dk.create_training_sample
create_expanded_dataset = dk.create_expanded_dataset


def test_create_training_sample_structure():
    s = create_training_sample("behave", "sweet", "calm", "Hi", "Hello", memory=["m1"])
    assert "messages" in s and "metadata" in s
    assert isinstance(s["messages"], list)
    assert s["messages"][0]["role"] == "user"
    assert s["messages"][1]["role"] == "assistant"
    assert s["metadata"]["mood"] == "behave"
    assert s["metadata"]["style"] == "sweet"


def test_create_expanded_dataset_has_expected_fields():
    ds = create_expanded_dataset()
    assert isinstance(ds, list)
    assert len(ds) > 0
    ex = ds[0]
    assert "messages" in ex and "metadata" in ex
    msgs = ex["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
