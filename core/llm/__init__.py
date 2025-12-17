"""
Refactored LLM System for Kitsu
Split into clean modules following new architecture:

core/llm/llm_interface.py      - Main interface
core/llm/ollama_adapter.py     - Ollama backend
core/llm/prompt_builder.py     - Prompt construction
core/personality/emotion_engine.py - Emotion state machine
core/cognition/planner.py      - Response planning
"""
