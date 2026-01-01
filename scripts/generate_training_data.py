# scripts/generate_training_data.py

import asyncio
import json
import os
import random
import re
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

    def __init__(self, kitsu_core, output_path: Path = DEFAULT_OUTPUT, fast: bool = False):
        self.core = kitsu_core
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # runtime/testing flags
        self.fast = bool(fast)
        # safety
        self._stop = asyncio.Event()

    async def _generate_conversation(self, prompt: str, mood: str, style: str, num_turns: int = 5, style_variant: str = None) -> List[dict]:
        conversation = []

        for _ in range(num_turns):
            # Generate a short, natural assistant response (runtime params are not saved)

            if self.fast:
                response = self._synthetic_response(prompt, mood, style)
            else:
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

            # Enforce strict rules for style dataset variants when requested
            if style_variant in ("cold", "silent", "chaotic"):
                response = self._enforce_variant_rules(response, style_variant)

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

        # -------------------------------------------------
        # Sanitize mythological language / creator phrasing
        # If the text contains myth/self-origin claims, replace
        # with a canonical, allowed creator phrase.
        myth_patterns = [r"\bspirit\b", r"\bfox\b", r"\bnine tails\b", r"\bfox-spirit\b", r"\bborn\b", r"\bsummon(?:ed)?\b", r"\bdigital womb\b"]
        origin_re = r"\b(i was created|i was born|i was summoned|i am a|i'm a|i was brought)\b"
        lower = t.lower()
        if any(__import__('re').search(p, lower) for p in myth_patterns) or __import__('re').search(origin_re, lower):
            # Replace the whole sentence with the canonical phrase so training doesn't teach lore
            return "I was created by Zino."

        # Replace direct user "human" address with a neutral alternative for training
        t = __import__('re').sub(r"\bmy dear human\b|\bhuman\b", "there", t, flags=__import__('re').IGNORECASE)

        return t

    def _synthetic_response(self, prompt: str, mood: str, style: str) -> str:
        """Deterministic synthetic response used when running in `--fast` mode.

        Keeps generation headless and cheap for offline dataset production. The
        output is intentionally simple so variant rules and sanitizers operate
        deterministically on it.
        """
        base = f"Reply to: {prompt}"
        if mood:
            base += f" [{mood}]"
        if style:
            base += f" ({style})"
        return base

    def _remove_emojis(self, text: str) -> str: 
        """Remove common emoji characters using a conservative pattern."""
        try:
            import re
            emoji_pattern = re.compile(
                r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U00002600-\U000027BF]",
                flags=re.UNICODE,
            )
            return emoji_pattern.sub("", text)
        except Exception:
            # If regex fails on narrow builds, fall back to no-op
            return text

    def _enforce_variant_rules(self, text: str, style_variant: str) -> str:
        """Apply strict output rules for `style` dataset variants.

        Rules (as requested):
        - cold: ≤12 words, no emojis
        - silent: ≤5 words
        - chaotic: ≤20 words, emojis allowed (try to keep emojis if present)

        This function makes a best-effort enforcement by trimming words and
        removing emojis when necessary. It is intentionally simple, deterministic
        and safe for use in bulk generation tasks.
        """
        if not text:
            return ""

        t = text.strip()

        if style_variant == "cold":
            # remove emojis and enforce short length
            t = self._remove_emojis(t)
            words = t.split()
            if len(words) > 12:
                t = " ".join(words[:12])
            # also strip trailing punctuation
            return t.strip()

        if style_variant == "silent":
            words = t.split()
            if len(words) > 5:
                t = " ".join(words[:5])
            return t.strip()

        if style_variant == "chaotic":
            # chaotic allows emojis but still cap length
            words = t.split()
            # Find trailing non-word characters (emojis/punctuation) in original text
            trailing = ""
            m = re.search(r"([^\w\s]+)\s*$", t)
            if m:
                trailing = m.group(1)

            if len(words) > 20:
                t = " ".join(words[:20])
                # append any trailing non-word chunk (likely emojis)
                if trailing:
                    # append trailing (emoji/punct) without adding an extra token
                    t = f"{t}{trailing}"
            return t.strip()

        # Unknown variants: return sanitized text
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

    def _save_sample(self, conversation: List[dict], emotion: str, mood: str, style: str, include_memory: bool = False, target: str = "base"):
        # Prefer the messages+metadata format for training datasets
        # New canonical format: messages + metadata
        sample = {
            "messages": conversation,
            "metadata": {
                "emotion": emotion or "",
                "mood": mood or "",
                "style": style or "",
            }
        }

        # Also provide legacy flat fields for backward compatibility
        instruction = ""
        input_text = ""
        outputs = []
        for msg in conversation:
            if msg.get("role") == "user" and not instruction:
                instruction = msg.get("content", "")
            if msg.get("role") == "assistant":
                outputs.append(msg.get("content", ""))

        sample.update({
            "instruction": instruction,
            "input": input_text,
            "output": "\n".join(outputs).strip(),
            "meta": {"base": "" if target != "base" else (style or ""), "style": style or "", "emotion": emotion or ""}
        })

        # Write safely and flush to disk immediately to avoid loss on crash
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()

    async def generate(self, *, num_samples: int = None, duration_hours: float = None, seed: int = None, include_memory: bool = False, report_interval: int = 100, target: str = "base", variant: str = None, heartbeat: int = 30, resume_existing: int = 0):
        """Run generation until num_samples produced or duration_hours elapsed or until Ctrl+C.

        Added parameters:
        - target: 'base'|'mood'|'style' to select dataset folder and behavior
        - variant: optional specific variant for mood/style datasets
        - heartbeat: seconds between heartbeat prints while waiting for LLM
        - resume_existing: number of samples already present in output file (skip these)
        """
        moods = ["behave", "mean", "flirty"]
        styles = ["chaotic", "sweet", "cold", "silent"]
        style_variants = ["chaotic", "cold", "silent","sweet"]

        if seed is not None:
            random.seed(seed)

        seed_prompts = self._load_seed_prompts()

        # Validate variant for the selected target
        if variant is not None:
            if target == 'mood' and variant not in moods:
                raise ValueError(f"Invalid mood variant '{variant}'. Valid: {moods}")
            if target == 'style' and variant not in style_variants:
                raise ValueError(f"Invalid style variant '{variant}'. Valid: {style_variants}")

        start = time.time()
        produced = 0
        if resume_existing:
            produced = resume_existing

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

                # Select mood/style depending on target dataset
                if target == "base":
                    mood_choice = None
                    style_choice = None
                elif target == "mood":
                    mood_choice = variant or random.choice(moods)
                    style_choice = None
                else:  # target == 'style'
                    style_choice = variant or random.choice(style_variants)
                    mood_choice = None

                prompt = random.choice(seed_prompts)
                # num_turns = random.randint(3, 7) -save for later if needed
                num_turns = 1


                # Immediate progress indicator so users can see activity in realtime
                print(
                    f"⏳ Generating sample {produced+1}: target={target}, mood={mood_choice or ''}, style={style_choice or ''} — elapsed {int(time.time()-start)}s",
                    flush=True,
                )

                # Heartbeat task to indicate liveness during slow LLM calls
                heartbeat_event = asyncio.Event()

                async def _heartbeat_loop():
                    while not heartbeat_event.is_set():
                        print(f"⏱ Heartbeat: generating sample {produced+1} (target={target})", flush=True)
                        try:
                            await asyncio.wait_for(heartbeat_event.wait(), timeout=heartbeat)
                        except asyncio.TimeoutError:
                            continue

                heartbeat_task = None
                if heartbeat and heartbeat > 0:
                    heartbeat_task = asyncio.create_task(_heartbeat_loop())

                # Pass fallback defaults for generation parameters so LLM works as expected
                mood_param = mood_choice or "behave"
                style_param = style_choice or "chaotic"

                # If a strict style_variant is requested, pass the variant to enforce rules afterwards
                style_variant = style_choice if style_choice in style_variants else None

                conversation = await self._generate_conversation(
                    prompt,
                    mood_param,
                    style_param,
                    num_turns=num_turns,
                    style_variant=style_variant,
                )

                # stop heartbeat once generation for this sample is done
                if heartbeat_task:
                    heartbeat_event.set()
                    try:
                        await heartbeat_task
                    except Exception:
                        # ignore cancellation errors
                        pass

                emotion = getattr(self.core, "emotion_engine", None)
                emotion = emotion.get_current_emotion() if emotion is not None else "neutral"

                # Save only user/assistant messages (already enforced) and metadata
                # Use resolved params (mood_param/style_param) so metadata is filled correctly
                self._save_sample(conversation, emotion, mood_param, style_param, include_memory=include_memory, target=target)

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
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSONL path (overridden by --target)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0 = system random)")
    p.add_argument("--report-interval", type=int, default=100, help="How often to report progress")
    p.add_argument("--include-memory", action="store_true", help="Include memory in metadata (disabled by default)")

    # New options for dataset splitting
    p.add_argument("--target", choices=["base", "mood", "style"], default="base", help="Which dataset to generate: base|mood|style")
    p.add_argument("--variant", type=str, default=None, help="Variant for mood/style (e.g., behave, mean, flirty or chaotic, cold, silent). If omitted a random variant is used per sample")

    # Heartbeat frequency (seconds) while waiting for LLM
    p.add_argument("--heartbeat", type=int, default=30, help="How often (s) to print a heartbeat message while waiting for LLM responses (0 to disable)")

    # 
    p.add_argument("--fast", action="store_true", help="Use synthetic fast responses for testing (no LLM calls)")

    return p.parse_args()


async def main(core=None):
    args = _parse_cli()

    seed = args.seed if args.seed != 0 else None

    # Compute output path base on target to keep datasets separate
    if args.target == "base":
        out_dir = Path("data/training/base")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "kitsu_dataset.jsonl"
    else:
        out_dir = Path("data/training") / args.target
        out_dir.mkdir(parents=True, exist_ok=True)
        variant_suffix = args.variant or "all"
        out_file = out_dir / f"kitsu_dataset_{variant_suffix}.jsonl"

    # If no core provided, use a lightweight mock core so datasets can be generated
    # without requiring the full runtime. This keeps dataset generation headless
    # and safe for Auto runs on low-spec machines.
    if core is None:
        class MockLLM:
            def __init__(self, seed=None):
                self._seed = seed

            async def generate_response(self, user_input: str, mood: str = "behave", style: str = "chaotic", **kwargs) -> str:
                # Deterministic, cheap fake response useful for dataset generation
                # Format: a short template that can be post-processed by rules
                base = f"Reply to: {user_input}"
                if mood:
                    base += f" [{mood}]"
                if style:
                    base += f" ({style})"
                return base

        class MockCore:
            def __init__(self, seed=None):
                self.llm = MockLLM(seed=seed)
                # minimal emotion engine stubs
                self.emotion_engine = None

        core = MockCore(seed=seed)

    # Resume support: count lines in existing file and continue producing new samples
    produced_existing = 0
    if out_file.exists():
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                for _ in f:
                    produced_existing += 1
        except Exception:
            produced_existing = 0

    gen = TrainingDataGenerator(core, output_path=out_file, fast=args.fast)

    # Start generation with awareness of existing produced count
    await gen.generate(
        num_samples=args.num_samples,
        duration_hours=args.duration_hours,
        seed=seed,
        include_memory=args.include_memory,
        report_interval=args.report_interval,
        target=args.target,
        variant=args.variant,
        heartbeat=args.heartbeat,
        resume_existing=produced_existing,
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")