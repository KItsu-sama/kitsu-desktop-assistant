# ============================================================================
# FILE: scripts/quick_start.bat (Windows)
# ============================================================================

"""
@echo off
REM Quick start script for Kitsu (Windows)

echo 🦊 KITSU QUICK START
echo ====================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    exit /b 1
)

echo ✓ Python found

REM Install dependencies
echo.
echo 📦 Installing dependencies...
python scripts\install_dependencies.py

REM Run setup wizard
echo.
echo ⚙️ Running setup wizard...
python scripts\setup_wizard.py

REM Generate dataset
echo.
echo 📚 Generating training data...
python scripts\generate_dataset.py

REM Done
echo.
echo ✅ Setup complete!
echo.
echo 🚀 To start Kitsu:
echo    python main.py
echo.
"""