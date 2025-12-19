"""
scripts/run_pipeline.py

Simple cross-platform pipeline runner:
1. scripts/generate_training_data.py
2. scripts/finetune_lora.py
3. scripts/load_to_ollama.py (fallback to load_to_ollama_direct.py)

Stops on failure and prints clear status messages.
"""
import subprocess
import sys
from pathlib import Path
import shutil

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    Path("scripts/generate_training_data.py"),
    Path("scripts/finetune_lora.py"),
    Path("scripts/load_to_ollama.py"),
]

FALLBACK = Path("scripts/load_to_ollama_direct.py")


def run_step(script: Path, args=None):
    """
    Run a script. If it's a module under the `scripts` package, run it with -m
    from the project root so package imports (e.g., `import core`) work.
    """
    # Prefer module execution for scripts in the scripts/ package so the
    # project root is on sys.path and imports like `import core` succeed.
    if script.suffix == ".py" and script.parts and script.parts[0] == "scripts":
        module = ".".join(script.with_suffix("").parts)
        cmd = [PY, "-m", module]
    else:
        cmd = [PY, str(script)]

    if args:
        cmd.extend(args)

    print(f"\n🔷 Running: {script} \n   cmd: {cmd}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"❌ Step failed: {script} (exit {result.returncode})")
        return False
    print(f"✅ Step succeeded: {script}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default=None, help='Optional style to pass to finetune step')
    args = parser.parse_args()

    # 1 - generate training data
    if not SCRIPTS[0].exists():
        print(f"⚠️ Missing script: {SCRIPTS[0]} - skipping generation step")
    else:
        if not run_step(SCRIPTS[0]):
            return 1

    # 2 - finetune (CPU-friendly script)
    finetune_script = SCRIPTS[1]
    if not finetune_script.exists():
        print(f"⚠️ Missing finetune script: {finetune_script}")
        return 1
    finetune_args = []
    if args.style:
        finetune_args.extend(['--style', args.style])
    if not run_step(finetune_script, args=finetune_args):
        return 1

    # 3 - load to Ollama (detect availability first)
    ollama_present = shutil.which("ollama") is not None
    load_script = SCRIPTS[2]
    if ollama_present and load_script.exists():
        if not run_step(load_script):
            # Try fallback
            if FALLBACK.exists():
                print("⚠️ load_to_ollama failed; trying fallback direct loader...")
                if not run_step(FALLBACK):
                    return 1
            else:
                return 1
    else:
        print("⚠️ Ollama not found or load_to_ollama missing; using direct loader if available")
        if FALLBACK.exists():
            if not run_step(FALLBACK):
                return 1
        else:
            print("❌ No loader script available. Aborting.")
            return 1

    print("\n🎉 Pipeline complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
