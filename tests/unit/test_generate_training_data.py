import asyncio
import json
from pathlib import Path
import tempfile

# Import module by path so tests don't require package installation
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("generate_training_data", Path("scripts/generate_training_data.py"))
generate_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_mod)
TrainingDataGenerator = generate_mod.TrainingDataGenerator


class DummyLLM:
    async def generate_response(self, user_input, mood=None, style=None, max_tokens=48, temperature=0.7, stream=False):
        # return a short, in-character reply
        return "Sure! I can help."


class DummyCore:
    def __init__(self):
        self.llm = DummyLLM()
        class E:
            def get_current_emotion(self):
                return "neutral"
        self.emotion_engine = E()


def test_sanitize_and_save(tmp_path):
    core = DummyCore()
    out = tmp_path / "out.jsonl"
    gen = TrainingDataGenerator(core, output_path=out)

    async def run_once():
        await gen.generate(num_samples=1, duration_hours=None, seed=42, include_memory=False, report_interval=1)

    asyncio.run(run_once())

    assert out.exists()
    with open(out, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert "messages" in obj and "metadata" in obj
    assert obj["metadata"]["style"] in {"chaotic","sweet","cold","silent"}


def test_sanitize_response_removes_obedience():
    core = DummyCore()
    gen = TrainingDataGenerator(core)
    s = gen._sanitize_response("As an AI, I cannot do that, but I can suggest...")
    assert "as an ai" not in s.lower()
