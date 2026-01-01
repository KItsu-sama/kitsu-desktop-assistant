# File: core/dev/handlers/prompt_inspector.py
# -----------------------------------------------------------------------------
"""
Prompt inspection and debugging tools for developers.

Provides commands to view:
- Last prompt sent to LLM
- Prompt building process
- Model configuration
- Generation options
"""

import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

log = logging.getLogger(__name__)


class PromptInspector:
    """Handler for prompt inspection and debugging commands"""
    
    def __init__(self, kitsu_core=None, llm_interface=None, logger=None):
        self.core = kitsu_core
        self.llm = llm_interface or (kitsu_core.llm if kitsu_core else None)
        self.logger = logger or log
        
        # Store last prompt data
        self._last_prompt_data: Optional[Dict[str, Any]] = None
        
        # Hook into LLM if available
        if self.llm:
            self._hook_llm()
    
    def _hook_llm(self):
        """Hook into LLM to capture prompts"""
        # Store original generate method
        original_generate = self.llm.generate_response
        
        async def wrapped_generate(*args, **kwargs):
            """Wrapped generate that captures prompt data"""
            try:
                # Capture inputs
                self._capture_prompt_data(args, kwargs)
            except Exception as e:
                self.logger.debug(f"Failed to capture prompt data: {e}")
            
            # Call original
            return await original_generate(*args, **kwargs)
        
        # Replace method
        self.llm.generate_response = wrapped_generate
        self.logger.debug("Prompt inspector hooked into LLM")
    
    def _capture_prompt_data(self, args: tuple, kwargs: dict):
        """Capture prompt construction data"""
        try:
            # Extract common parameters
            user_input = args[0] if args else kwargs.get('user_input', '')
            mood = kwargs.get('mood', 'behave')
            style = kwargs.get('style', 'chaotic')
            frozen_emotion = kwargs.get('frozen_emotion', 'neutral')
            custom_prompt = kwargs.get('custom_prompt')
            is_greeting = kwargs.get('is_greeting', False)
            
            # Build the actual prompt using LLM's method
            if self.llm:
                prompt = self.llm._build_prompt(
                    user_input=user_input,
                    mood=mood,
                    style=style,
                    emotion=frozen_emotion,
                    custom_prompt=custom_prompt,
                    is_greeting=is_greeting,
                    user_title=kwargs.get('user_title', ''),
                    length_hint=kwargs.get('length_hint'),
                    preferences=kwargs.get('preferences'),
                )
                
                # Get generation options
                options = self.llm._get_generation_options(mood, style, frozen_emotion, kwargs.get('length_hint'))
                
                # Store everything
                self._last_prompt_data = {
                    'timestamp': self._get_timestamp(),
                    'user_input': user_input,
                    'mood': mood,
                    'style': style,
                    'emotion': frozen_emotion,
                    'is_greeting': is_greeting,
                    'prompt': prompt,
                    'prompt_length': len(prompt),
                    'options': options,
                    'model': self.llm.model,
                    'is_character_model': self.llm.is_character_model,
                }
                
        except Exception as e:
            self.logger.exception(f"Error capturing prompt data: {e}")
    
    def show_last_prompt(self, format: str = "pretty") -> str:
        """Show the last prompt sent to LLM
        
        Args:
            format: Output format (pretty|raw|json)
        """
        if not self._last_prompt_data:
            return "❌ No prompt data available yet. Send a message first."
        
        data = self._last_prompt_data
        
        if format == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)
        
        elif format == "raw":
            return data['prompt']
        
        else:  # pretty
            lines = [
                "=" * 80,
                "📝 LAST PROMPT INSPECTION",
                "=" * 80,
                "",
                f"⏰ Timestamp: {data['timestamp']}",
                f"🤖 Model: {data['model']} ({'CHARACTER' if data['is_character_model'] else 'STANDARD'})",
                "",
                "📊 PARAMETERS:",
                f"  User Input: {self._truncate(data['user_input'], 60)}",
                f"  Mood: {data['mood']}",
                f"  Style: {data['style']}",
                f"  Emotion: {data['emotion']}",
                f"  Is Greeting: {data['is_greeting']}",
                "",
                "⚙️  GENERATION OPTIONS:",
            ]
            
            for key, value in data['options'].items():
                lines.append(f"  {key}: {value}")
            
            lines.extend([
                "",
                f"📏 PROMPT LENGTH: {data['prompt_length']} characters",
                "",
                "=" * 80,
                "📄 FULL PROMPT:",
                "=" * 80,
                "",
                data['prompt'],
                "",
                "=" * 80,
            ])
            
            return "\n".join(lines)
    
    def show_prompt_breakdown(self) -> str:
        """Show detailed breakdown of prompt construction"""
        if not self._last_prompt_data:
            return "❌ No prompt data available yet."
        
        data = self._last_prompt_data
        prompt = data['prompt']
        
        lines = [
            "=" * 80,
            "🔍 PROMPT BREAKDOWN ANALYSIS",
            "=" * 80,
            "",
        ]
        
        if data['is_character_model']:
            # Parse control header
            lines.append("📋 CHARACTER MODEL FORMAT")
            lines.append("")
            
            if "<kitsu.control>" in prompt:
                control_start = prompt.find("<kitsu.control>")
                control_end = prompt.find("</kitsu.control>")
                
                if control_end > control_start:
                    control_block = prompt[control_start:control_end + len("</kitsu.control>")]
                    rest = prompt[control_end + len("</kitsu.control>"):].strip()
                    
                    lines.append("📦 CONTROL HEADER:")
                    lines.append("-" * 40)
                    lines.append(control_block)
                    lines.append("")
                    
                    # Check for memory block
                    if "<kitsu.memory>" in rest:
                        mem_start = rest.find("<kitsu.memory>")
                        mem_end = rest.find("</kitsu.memory>")
                        
                        if mem_end > mem_start:
                            mem_block = rest[mem_start:mem_end + len("</kitsu.memory>")]
                            user_input = rest[mem_end + len("</kitsu.memory>"):].strip()
                            
                            lines.append("🧠 MEMORY INJECTION:")
                            lines.append("-" * 40)
                            lines.append(mem_block)
                            lines.append("")
                            lines.append("💬 USER INPUT:")
                            lines.append("-" * 40)
                            lines.append(user_input)
                        else:
                            lines.append("💬 USER INPUT:")
                            lines.append("-" * 40)
                            lines.append(rest)
                    else:
                        lines.append("💬 USER INPUT:")
                        lines.append("-" * 40)
                        lines.append(rest)
            else:
                lines.append("⚠️  No control header found (unexpected)")
                lines.append(prompt)
        else:
            # Standard model with full prompt
            lines.append("📋 STANDARD MODEL FORMAT")
            lines.append("")
            lines.append("Full natural language prompt with character context,")
            lines.append("mode templates, style modifiers, and memory.")
            lines.append("")
            lines.append("💬 FULL PROMPT:")
            lines.append("-" * 40)
            lines.append(prompt)
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def show_model_config(self) -> str:
        """Show current model configuration"""
        if not self.llm:
            return "❌ LLM interface not available"
        
        lines = [
            "=" * 80,
            "🤖 MODEL CONFIGURATION",
            "=" * 80,
            "",
            f"Model Name: {self.llm.model}",
            f"Model Type: {'CHARACTER' if self.llm.is_character_model else 'STANDARD'}",
            f"Temperature: {self.llm.temperature}",
            f"Streaming: {self.llm.adapter.streaming}",
            f"Available: {self.llm.is_available}",
            "",
            "📋 PROMPT MODE:",
            f"  Using: {'Minimal Control Headers' if self.llm.is_character_model else 'Full Prompts'}",
            f"  Builder: {'None (Control Mode)' if self.llm.prompt_builder is None else 'PromptBuilder'}",
            "",
        ]
        
        # Show LoRA status if available
        if hasattr(self.llm, 'lora_manager') and self.llm.lora_manager:
            lora_stats = self.llm.lora_manager.get_stats()
            lines.extend([
                "🎭 LORA STATUS:",
                f"  Current Stack: {lora_stats.get('current_stack', [])}",
                f"  Available Styles: {', '.join(lora_stats.get('available_styles', []))}",
                f"  Total Switches: {lora_stats.get('switch_count', 0)}",
                "",
            ])
        
        # Show config
        if hasattr(self.llm, 'config'):
            lines.extend([
                "⚙️  LLM CONFIG:",
                f"  Auto Restart: {self.llm.config.auto_restart}",
                f"  Fallback on Failure: {self.llm.config.fallback_on_failure}",
                f"  Max Retries: {self.llm.config.max_retries}",
                f"  Retry Delay: {self.llm.config.retry_delay}s",
                "",
            ])
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def export_prompt_history(self, output_path: Optional[Path] = None) -> str:
        """Export prompt history to file"""
        if output_path is None:
            output_path = Path("logs/prompt_history.jsonl")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self._last_prompt_data:
            return "❌ No prompt data to export"
        
        try:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(self._last_prompt_data, ensure_ascii=False) + "\n")
            
            return f"✅ Exported prompt to: {output_path}"
        except Exception as e:
            return f"❌ Export failed: {e}"
    
    def compare_modes(self, user_input: str = "Hello!") -> str:
        """Show how prompt differs across modes/styles"""
        if not self.llm:
            return "❌ LLM interface not available"
        
        modes = ["behave", "mean", "flirty"]
        styles = ["chaotic", "sweet", "cold", "silent"]
        
        lines = [
            "=" * 80,
            "🔄 PROMPT COMPARISON",
            "=" * 80,
            f"",
            f"Test Input: '{user_input}'",
            "",
        ]
        
        for mood in modes:
            for style in styles:
                try:
                    prompt = self.llm._build_prompt(
                        user_input=user_input,
                        mood=mood,
                        style=style,
                        emotion="neutral"
                    )
                    
                    lines.append(f"📊 {mood.upper()} / {style.upper()}")
                    lines.append("-" * 40)
                    
                    if self.llm.is_character_model:
                        # Extract just the control header
                        if "<kitsu.control>" in prompt:
                            control_end = prompt.find("</kitsu.control>")
                            if control_end > 0:
                                control = prompt[:control_end + len("</kitsu.control>")]
                                lines.append(control)
                    else:
                        # Show first 200 chars for standard prompts
                        lines.append(self._truncate(prompt, 200))
                    
                    lines.append("")
                    
                except Exception as e:
                    lines.append(f"❌ Error: {e}")
                    lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _truncate(self, text: str, length: int) -> str:
        """Truncate text with ellipsis"""
        if len(text) <= length:
            return text
        return text[:length - 3] + "..."
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")