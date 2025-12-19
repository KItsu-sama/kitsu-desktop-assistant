# scripts/generate_training_data.py

import asyncio
import json
import os
import random
import signal
import time
from pathlib import Path
from typing import List


DEFAULT_OUTPUT = Path("data/training/kitsu_dataset.jsonl")


class TrainingDataGenerator:
    """Robust, unattended dataset generator for Kitsu.

    Produces JSONL lines of the form:
    {"messages": [{"role":"user","content":"..."}, ...], "metadata": {"mood":"...","style":"...","emotion":"..."}}

    The generator avoids system/persona text, keeps assistant replies short, and flushes frequently.
    """

    def __init__(self, kitsu_core, output_path: Path = DEFAULT_OUTPUT):
        self.core = kitsu_core
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # safety
        self._stop = asyncio.Event()

    async def _generate_conversation(self, prompt: str, mood: str, style: str, num_turns: int = 5) -> List[dict]:
        conversation = []

        for _ in range(num_turns):
            # Generate a short, natural assistant response (runtime params are not saved)
            response = await self.core.llm.generate_response(
                user_input=prompt,
                mood=mood,
                style=style,
                max_tokens=48,
                temperature=0.7,
                stream=False,
            )

            # Sanitize assistant response: remove common obedience phrases, keep it short
            response = self._sanitize_response(response)

            conversation.append({"role": "user", "content": prompt})
            conversation.append({"role": "assistant", "content": response})

            prompt = self._generate_followup(response)

            if self._stop.is_set():
                break

        return conversation

    def _sanitize_response(self, text: str) -> str:
        if not text:
            return ""

        # Remove leading obedience/AI disclaimers
        forbidden_prefixes = [
            "as an ai",
            "as an assistant",
            "as a language model",
            "certainly",
            "i'm a",
            "i am a",
        ]

        t = text.strip()
        low = t.lower()
        for p in forbidden_prefixes:
            if low.startswith(p):
                # drop the prefix
                cut = len(p)
                # skip punctuation/spaces
                while cut < len(t) and t[cut] in ":,._ -":
                    cut += 1
                t = t[cut:].strip()
                break

        # Keep only the first short sentence (avoid rambling)
        # Split on sentence terminators
        for sep in (". ", "! ", "? "):
            if sep in t:
                t = t.split(sep)[0].strip()
                break

        # Limit to ~30 words
        words = t.split()
        if len(words) > 30:
            t = " ".join(words[:30])

        return t

    def _generate_followup(self, response: str) -> str:
        # Lightweight follow-up: ask a related question or continue thread
        # Keep deterministic choices for reproducibility if seed set
        followups = [
            "Can you tell me more?",
            "What's next?",
            "Why do you think that?",
            "How does that make you feel?",
            "Anything else you'd add?",
        ]
        return random.choice(followups)

    def _save_sample(self, conversation: List[dict], emotion: str, mood: str, style: str, include_memory: bool = False):
        sample = {
            "messages": conversation,
            "metadata": {"emotion": emotion, "mood": mood, "style": style, "memory": []},
        }

        # Write safely and flush to disk immediately to avoid loss on crash
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    async def generate(self, *, num_samples: int = None, duration_hours: float = None, seed: int = None, include_memory: bool = False, report_interval: int = 100):
        """Run generation until num_samples produced or duration_hours elapsed or until Ctrl+C."""
        moods = ["behave", "mean", "flirty"]
        styles = ["chaotic", "sweet", "cold", "silent"]

        if seed is not None:
            random.seed(seed)

        seed_prompts = self._load_seed_prompts()

        start = time.time()
        produced = 0

        async def _stop_on_signal():
            loop = asyncio.get_running_loop()
            stop_future = loop.create_future()

            def _signal_handler(_signum, _frame):
                if not stop_future.done():
                    stop_future.set_result(True)

            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
            await stop_future
            self._stop.set()

        stopper = asyncio.create_task(_stop_on_signal())

        try:
            while True:
                if self._stop.is_set():
                    break

                if num_samples is not None and produced >= num_samples:
                    break

                if duration_hours is not None and (time.time() - start) >= duration_hours * 3600:
                    break

                mood = random.choice(moods)
                style = random.choice(styles)
                prompt = random.choice(seed_prompts)
                num_turns = random.randint(3, 7)

                # Immediate progress indicator so users can see activity in realtime
                print(
                    f"⏳ Generating sample {produced+1}: mood={mood}, style={style} — elapsed {int(time.time()-start)}s",
                    flush=True,
                )

                conversation = await self._generate_conversation(prompt, mood, style, num_turns=num_turns)

                emotion = getattr(self.core, "emotion_engine", None)
                emotion = emotion.get_current_emotion() if emotion is not None else "neutral"

                # Save only user/assistant messages (already enforced) and metadata
                self._save_sample(conversation, emotion, mood, style, include_memory=include_memory)

                produced += 1

                # Short confirmation after each saved sample
                print(
                    f"✅ Saved sample {produced} — output={self.output_path} (emotion={emotion})",
                    flush=True,
                )

                if produced % report_interval == 0:
                    elapsed = time.time() - start
                    print(f"Generated {produced} samples — elapsed {int(elapsed)}s", flush=True)

            # final report
            elapsed = time.time() - start
            print(f"Generation stopped. Produced={produced} elapsed={int(elapsed)}s")

        finally:
            stopper.cancel()

    def _load_seed_prompts(self) -> List[str]:
        # try loading from assets/seeds.txt else use an internal list
        seeds_path = Path("data/training/seed_prompts.txt")
        if seeds_path.exists():
            return [l.strip() for l in seeds_path.read_text(encoding="utf-8").splitlines() if l.strip()]

        return [
            "Tell me about your day.",
            "What's something interesting you've learned?",
            "How would you react if someone surprised you?",
            "Describe the weather in a single sentence.",
            "Recommend a quick snack.",
            "What's a small secret you keep?",
        ]


def _parse_cli():
    import argparse

    p = argparse.ArgumentParser(description="Kitsu training dataset generator (JSONL)")
    p.add_argument("--num-samples", type=int, default=None, help="Number of conversations to generate (default: run until interrupted)")
    p.add_argument("--duration-hours", type=float, default=None, help="Run duration in hours")
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSONL path")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0 = system random)")
    p.add_argument("--report-interval", type=int, default=100, help="How often to report progress")
    p.add_argument("--include-memory", action="store_true", help="Include memory in metadata (disabled by default)")
    return p.parse_args()


async def main(core=None):
    args = _parse_cli()

    # Core object must be provided by the project runtime; if not, try importing minimal stub
    if core is None:
        try:
            # Import minimal Kitsu core if available (new name: KitsuIntegrated)
            from core.kitsu_core import KitsuIntegrated

            core = KitsuIntegrated()
            # Initialize the runtime components before generating responses
            await core.initialize()
        except Exception:
            raise RuntimeError("Kitsu core object required to generate realistic responses; pass core instance or run inside project runtime")

    seed = args.seed if args.seed != 0 else None
    gen = TrainingDataGenerator(core, output_path=Path(args.output))
    await gen.generate(num_samples=args.num_samples, duration_hours=args.duration_hours, seed=seed, include_memory=args.include_memory, report_interval=args.report_interval)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")