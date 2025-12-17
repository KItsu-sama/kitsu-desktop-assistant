# ============================================================================
# FILE: scripts/quick_start.sh (Unix/Linux/Mac)
# One-command setup for everything
# ============================================================================

"""
#!/bin/bash
# Quick start script for Kitsu

echo "🦊 KITSU QUICK START"
echo "===================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
python3 scripts/install_dependencies.py

# Run setup wizard
echo ""
echo "⚙️  Running setup wizard..."
python3 scripts/setup_wizard.py

# Generate dataset
echo ""
echo "📚 Generating training data..."
python3 scripts/generate_dataset.py

# Done
echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start Kitsu:"
echo "   python3 main.py"
echo ""
"""