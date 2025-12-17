# scripts/load_to_ollama.py

import subprocess
from pathlib import Path

def create_modelfile():
    """Create Ollama Modelfile for Kitsu"""
    
    modelfile_content = """FROM ./data/models/kitsu-gguf/kitsu-q4_k_m.gguf

# Character model - personality in weights
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System prompt (minimal - just format)
TEMPLATE \"\"\"{{- if .System }}
<|system|>
{{ .System }}
{{- end }}
<|user|>
{{ .Prompt }}
<|assistant|>
\"\"\"

# Stop tokens
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "<|system|>"
"""
    
    modelfile_path = Path("data/models/Modelfile")
    modelfile_path.write_text(modelfile_content)
    return modelfile_path

def load_to_ollama():
    """Load character model into Ollama"""
    
    print("🦊 Loading Kitsu into Ollama...")
    
    # Create Modelfile
    modelfile = create_modelfile()
    print(f"✅ Created Modelfile: {modelfile}")
    
    # Create Ollama model
    print("\n📦 Creating Ollama model...")
    result = subprocess.run([
        "ollama", "create", "kitsu:character",
        "-f", str(modelfile)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Model loaded successfully!")
        print("\n🎯 Test it with:")
        print('   ollama run kitsu:character "emotion: happy | mood: behave | style: chaotic\\nUser: Hi!"')
    else:
        print(f"❌ Failed: {result.stderr}")

if __name__ == "__main__":
    load_to_ollama()