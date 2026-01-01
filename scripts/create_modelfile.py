# scripts/create_modelfile.py
"""
Create a proper Ollama Modelfile for Kitsu
"""

from pathlib import Path
from rich.console import Console as console


def create_modelfile(model_dir: Path):
    """Create Ollama Modelfile automatically (always overwrite)"""
    console.print("\n[cyan]📝 Creating Modelfile (auto)...[/cyan]")

    modelfile_content = f"""# Kitsu Character Model
FROM {model_dir.resolve()}

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

TEMPLATE \"\"\"{{{{ if .System }}}}
<|system|>
{{{{ .System }}}}
{{{{ end }}}}
<|user|>
{{{{ .Prompt }}}}
<|assistant|>
\"\"\"

PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "<|system|>"
PARAMETER stop "</s>"
"""

    modelfile_path = Path("data/models/Modelfile.kitsu")
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    console.print(f"[green]✅ Modelfile ready:[/green] {modelfile_path}")
    return modelfile_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Ollama Modelfile for Kitsu")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/models/kitsu-merged"),
        help="Path to merged model directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/Modelfile.kitsu"),
        help="Output Modelfile path"
    )
    
    args = parser.parse_args()
    
    print("\n🦊 Creating Ollama Modelfile")
    print("="*60)
    
    if not args.model_path.exists():
        print(f"❌ Model not found: {args.model_path}")
        print("\nAvailable models:")
        models_dir = Path("data/models")
        if models_dir.exists():
            for path in models_dir.iterdir():
                if path.is_dir():
                    print(f"   - {path.name}")
        return 1
    
    create_modelfile(args.model_path, args.output)
    
    print("\n🎯 Next step:")
    print(f"   ollama create kitsu:character -f {args.output}")
    
    return 0


# Backward-compatible alias that delegates to canonical implementation
from scripts import ollama as _ollama

def create_modelfile(model_dir: Path):
    return _ollama.create_modelfile(model_dir)


if __name__ == "__main__":
    import sys
    sys.exit(main())