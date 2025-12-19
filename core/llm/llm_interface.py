# core/llm/llm_interface.py 

"""
Main LLM interface - coordinates all LLM interactions
NOW WITH CHARACTER MODEL SUPPORT!
"""

import asyncio
import logging
import json
import re
from pathlib import Path
import threading
import datetime
from typing import Dict, Any, Optional, AsyncGenerator

from core.llm.lora_manager import LoRAManager

from core.llm.ollama_adapter import OllamaAdapter
from core.llm.prompt_builder import PromptBuilder
from core.fallback_manager import FallbackManager
from core.llm.lora_router import LoRaRouter

from core.meta.action_parser import parse_action_from_text
from core.meta.meta_controller import MetaController
from core.meta.action_executor import (
    execute_search,
    execute_get_time,
)
from core.meta.meta_controller import ContinuationState

 
log = logging.getLogger(__name__)


class LLMConfig:
    """Configuration for LLM behavior"""
    
    def __init__(
        self,
        auto_restart: bool = True,
        fallback_on_failure: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.auto_restart = auto_restart
        self.fallback_on_failure = fallback_on_failure
        self.max_retries = max_retries
        self.retry_delay = retry_delay


class LLMInterface:
    """
    Main interface for LLM interactions
    
    NEW: Automatically detects character models and uses machine-readable control
    prompts for Neuro-sama style character models. Character prompts are emitted
    as a compact <kitsu.control> header with no persona, instructions, or
    natural-language system text. LoRA selections and sampling params are
    controlled via emotion/mood/style.
    """
    
    def __init__(
        self,
        model: str = "gemma:2b",
        temperature: float = 0.8,
        character_context: str = "",
        memory_manager: Optional[Any] = None,
        templates_path: Optional[Path] = None,
        streaming: bool = True,
        config: Optional[LLMConfig] = None,
        memory: Optional[Any] = None,
        emotion_engine: Optional[Any] = None,  # NEW: Need emotion engine for minimal prompts
        lora_manager: Optional[Any] = None
    ):
        # Model may be provided as a string or a dict (legacy vs new config).
        # Normalize to a simple string name here so downstream code expecting
        # a string does not error when passed a dict.
        if isinstance(model, dict):
            model = model.get("style") or model.get("name") or "gemma:2b"

        self.model = model
        self.temperature = temperature
        self.emotion_engine = emotion_engine  # Store emotion engine reference

        self.config = config or LLMConfig()
        self.fallback = FallbackManager(memory)
        
        # Create adapter (respect streaming preference)
        self.adapter = OllamaAdapter(
            model=self.model,
            temperature=self.temperature,
            streaming=streaming
        )

        # Initialize LoRA manager (allow injection)
        try:
            if lora_manager is not None:
                self.lora_manager = lora_manager
            else:
                # only create if not injected
                self.lora_manager = getattr(self, 'lora_manager', None) or LoRAManager()
                log.info("✅ LoRA manager initialized")

                # Auto-load default adapter if configured
                config_style = getattr(config, 'default_style', None)
                if config_style and config_style in self.lora_manager.adapters:
                    self.lora_manager.switch_adapter(config_style)
                    log.info(f"Loaded default LoRA: {config_style}")

        except Exception as e:
            log.warning(f"⚠️  LoRA manager unavailable: {e}")
            self.lora_manager = None


        
        # Detect if using character model
        self.is_character_model = self._detect_character_model(model)
        
        if self.is_character_model:
            log.info("🦊 CHARACTER MODEL DETECTED - Using machine-readable control prompts")
            # For Neuro-sama-style character models we DO NOT use a natural-language
            # prompt builder. Prompts are machine-readable control headers only.
            self.prompt_builder = None
            self.minimal_mode = True
        else:
            log.info("📝 Standard model - Using full prompts")
            # Use full prompt builder for standard models
            self.prompt_builder = PromptBuilder(
                character_context=character_context,
                memory_manager=memory_manager,
                templates_path=templates_path or Path("data/templates")
            )
            self.minimal_mode = False
        
        # Keep a reference to memory manager for personalized fallbacks
        self.memory = memory_manager
        
        # State tracking
        self.is_available = False
        # Meta-controller for gating actions
        self.meta_controller = MetaController()
        # Last action decision (machine-only) for UI integration
        self.last_action_decision: Optional[Dict[str, Any]] = None
        # LoRA router (register style->adapter paths elsewhere, cheap selection)
        self.lora_router = LoRaRouter()
        self.active_lora_style = "chaotic"
        self._check_availability()
        
        log.info(f"LLMInterface initialized with {self.model}")

    # ==================== generate_response ====================

    async def generate_response(
        self,
        user_input: str,
        mood: str = "behave",
        style: str = "chaotic",
        stream: bool = False,
        custom_prompt: Optional[str] = None,
        is_greeting: bool = False,
        user_title: str = "",
        preferences: Optional[Dict[str, Any]] = None,
        user_permissions: Optional[Dict[str, Any]] = None,
        last_topic: Optional[str] = None,
        continuation_state: Optional[ContinuationState] = None,
    ) -> str:
        """
        Generate conversational response with automatic retry and fallback
        
        NOW WITH EMOTION-DRIVEN LORA SWITCHING!
        """
        
        # === LoRA SWITCHING (NEW) ===
        if self.lora_manager and self.emotion_engine:
            try:
                # Get current emotion state
                emotion_state = self.emotion_engine.get_state_dict()
                
                # Select best LoRA for current state
                target_style = self.lora_manager.select_for_emotion(emotion_state)
                
                # Switch if needed (fast, no model reload)
                if target_style:
                    switched = self.lora_manager.switch_adapter(target_style)
                    if switched:
                        log.debug(f"🦊 LoRA switched to: {target_style}")
            
            except Exception as e:
                log.debug(f"LoRA switching failed: {e}")
        
        # === REST OF METHOD (UNCHANGED) ===
        
        # Freeze emotion once per response
        emotion = (
            self.emotion_engine.get_current_emotion()
            if self.emotion_engine
            else "neutral"
        )
        
        # Apply meta-controller
        mood, style, length_hint = self._apply_meta_controller(mood, style, preferences)
        user_permissions = user_permissions or {}
        
        if continuation_state is None:
            continuation_state = ContinuationState(0)
        
        for attempt in range(self.config.max_retries):
            try:
                # Check availability
                if not self.is_available:
                    log.warning(f"LLM unavailable, attempt {attempt + 1}/{self.config.max_retries}")
                    
                    if await self._try_restart_ollama():
                        continue
                    
                    if self.config.fallback_on_failure:
                        return self._get_fallback_response(mood, style)
                    
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                
                # Build prompt
                prompt = self._build_prompt(
                    user_input,
                    mood,
                    style,
                    emotion=emotion,
                    custom_prompt=custom_prompt,
                    is_greeting=is_greeting,
                    user_title=user_title,
                    length_hint=length_hint,
                    preferences=preferences,
                )
                
                # Generation options (now depend on mood/style/emotion)
                options = self._get_generation_options(mood, style, emotion, length_hint)
                
                # Generate
                if stream:
                    chunks = []
                    async for chunk in self.stream_response(prompt, **options):
                        chunks.append(chunk)
                    raw_response = "".join(chunks)
                else:
                    raw_response = await asyncio.to_thread(
                        self.adapter.generate,
                        prompt,
                        **options
                    )
                
                # Action handling (existing code)
                try:
                    parsed = parse_action_from_text(raw_response)
                except Exception as e:
                    log.warning(f"Action parser exception: {e}")
                    parsed = {"ok": False, "error": "parser exception"}
                
                self.last_action_decision = None
                
                if not parsed.get("ok"):
                    log.debug(f"Action parse failed: {parsed.get('error')}")
                    return self._sanitize_final_response(raw_response)
                
                action = parsed.get("action")
                
                # Continue marker detection
                if action is None:
                    lines = [ln.strip() for ln in raw_response.splitlines() if ln.strip()]
                    continue_lines = [ln for ln in lines if ln == "<continue>"]
                    if len(continue_lines) == 1:
                        from core.meta.action_parser import Action as ActionClass
                        action = ActionClass(
                            kind="continue",
                            payload={"reason": None},
                            raw="<continue>"
                        )
                    elif len(continue_lines) > 1:
                        log.warning("Multiple <continue> markers found")
                        return self._sanitize_final_response(raw_response)
                
                if action is None:
                    return self._sanitize_final_response(raw_response)
                
                # Meta-controller decision
                emotion_state = {"intensity": 0.0}
                try:
                    if self.emotion_engine and hasattr(self.emotion_engine, "get_current_intensity"):
                        emotion_state["intensity"] = float(
                            self.emotion_engine.get_current_intensity()
                        )
                except Exception:
                    pass
                
                decision = self.meta_controller.decide_and_handle_action(
                    action,
                    user_permissions=user_permissions,
                    last_topic=last_topic,
                    continuation_state=continuation_state,
                    emotion_state=emotion_state,
                )
                
                self.last_action_decision = decision
                
                # Handle decision
                if decision.get("requires_confirmation"):
                    log.info("Action requires confirmation")
                    return self._sanitize_final_response(raw_response)
                
                if not decision.get("approved") or not callable(decision.get("execute")):
                    log.info("Action denied or no executor")
                    return self._sanitize_final_response(raw_response)
                
                # Execute action
                try:
                    exec_fn = decision.get("execute")
                    exec_result = exec_fn()
                except Exception as e:
                    log.warning(f"Action execution failed: {e}")
                    return self._sanitize_final_response(raw_response)
                
                # Build tool result
                tool_result = {"action": action.kind, "result": exec_result}
                
                # Update continuation state
                if action.kind == "continue" and isinstance(exec_result, dict):
                    if exec_result.get("allowed"):
                        continuation_state.count = exec_result.get(
                            "new_count",
                            continuation_state.count
                        )
                
                # Single continuation
                try:
                    cont_prompt = (
                        prompt + 
                        "\n[INTERNAL] Tool result (machine-only): " + 
                        json.dumps(tool_result)
                    )
                    cont_resp = await asyncio.to_thread(
                        self.adapter.generate,
                        cont_prompt,
                        **options
                    )
                    final_raw = cont_resp
                except Exception as e:
                    log.warning(f"Continuation generation failed: {e}")
                    final_raw = raw_response
                
                return self._sanitize_final_response(final_raw)
            
            except Exception as e:
                log.error(f"Generation failed (attempt {attempt + 1}): {e}")
                self.is_available = False
                
                if attempt == self.config.max_retries - 1:
                    if self.config.fallback_on_failure:
                        log.warning("All retries exhausted, using fallback")
                        return self._get_fallback_response(mood, style)
                    else:
                        raise
                
                await asyncio.sleep(self.config.retry_delay)
        
        if self.config.fallback_on_failure:
            return self._get_fallback_response(mood, style)
        else:
            raise Exception("LLM generation failed after all retries")


    
    def _detect_character_model(self, model_name: Optional[Any]) -> bool:
        """
        Detect if model is a character model. Accepts string or dict-like values.

        Character models are identified by:
        - Name contains "kitsu:character"
        - Name contains ":character"
        - Metadata file exists with model_type=character
        """
        try:
            if isinstance(model_name, dict):
                # try common keys
                model_name = model_name.get("style") or model_name.get("name") or model_name.get("model")

            if not model_name:
                return False

            model_name_str = str(model_name)

            # Check name patterns
            if "kitsu:character" in model_name_str.lower():
                return True
            if ":character" in model_name_str.lower():
                return True

            # Check for metadata file
            model_dir = Path("data/models") / model_name_str.replace(":", "_")
            metadata_file = model_dir / "metadata.json"

            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        if metadata.get("model_type") == "character":
                            return True
                except Exception:
                    pass

            return False
        except Exception:
            return False
        
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    if metadata.get("model_type") == "character":
                        return True
            except Exception:
                pass
        
        return False
    
    def _check_availability(self) -> bool:
        """Check if LLM is available"""
        try:
            self.adapter.generate("test", num_predict=1)
            self.is_available = True
            log.info("✅ LLM is available")
            return True
        except Exception as e:
            self.is_available = False
            log.warning(f"⚠️ LLM not available: {e}")
            return False
    
    async def _try_restart_ollama(self) -> bool:
        """Attempt to restart Ollama service"""
        if not self.config.auto_restart:
            log.info("Auto-restart disabled, skipping")
            return False
        
        log.info("🔄 Attempting to restart Ollama...")
        
        try:
            import subprocess
            import platform
            
            system = platform.system()
            
            if system == "Linux" or system == "Darwin":
                try:
                    subprocess.run(
                        ["systemctl", "restart", "ollama"],
                        check=True,
                        capture_output=True,
                        timeout=10
                    )
                    log.info("✅ Ollama restarted via systemctl")
                except:
                    subprocess.Popen(
                        ["ollama", "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    log.info("✅ Ollama started as background process")
            
            elif system == "Windows":
                subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                log.info("✅ Ollama started on Windows")
            
            await asyncio.sleep(3)
            
            if self._check_availability():
                log.info("🎉 Ollama successfully restarted!")
                return True
            else:
                log.warning("❌ Ollama restart failed - service not responding")
                return False
                
        except Exception as e:
            log.error(f"❌ Failed to restart Ollama: {e}")
            return False
    
    def _get_fallback_response(self, mood, style):
        """Get fallback response when LLM fails"""
        return self.fallback.generate(mood, style)
    
    def _build_prompt(
        self,
        user_input: str,
        mood: str,
        style: str,
        emotion: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        is_greeting: bool = False,
        user_title: str = "",
        length_hint: Optional[Any] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build prompt based on model type

        For character models we emit a compact, machine-readable control block and
        the raw user input. No persona, no instructions, no natural-language
        system text.
        """
        if custom_prompt:
            return custom_prompt

        if self.is_character_model:
            # Machine-readable control header (deterministic, non-natural language)
            if emotion is None:
                emotion = "neutral"

            # Normalize length hint for the control block
            lh = length_hint if length_hint is not None else "medium"
            if isinstance(lh, int):
                # numeric token count
                length_field = str(lh)
            else:
                length_field = str(lh)

            control = (
                "<kitsu.control>\n"
                f"emotion={emotion}\n"
                f"mood={mood}\n"
                f"style={style}\n"
                f"length={length_field}\n"
                "</kitsu.control>\n\n"
            )

            # Memory injection only when explicitly requested (avoid per-turn injection)
            memory_block = ""
            if preferences and preferences.get("inject_memory") and self.memory:
                try:
                    # Attempt a couple of common memory APIs; be tolerant of missing methods.
                    if hasattr(self.memory, "get_memory_for_prompt"):
                        mem = self.memory.get_memory_for_prompt()
                    elif hasattr(self.memory, "summarize"):
                        mem = self.memory.summarize()
                    elif hasattr(self.memory, "get_recent_memory"):
                        mem = self.memory.get_recent_memory()
                    else:
                        mem = None
                except Exception:
                    mem = None

                if mem:
                    # Insert as a separate machine-only block
                    memory_block = f"\n<kitsu.memory>{mem}</kitsu.memory>\n\n"

            return control + memory_block + (user_input or "")
        else:
            # FULL PROMPT for standard model
            if is_greeting and user_title:
                # Use a cleaned title (avoid header tokens)
                safe_title = self._sanitize_user_title(user_title)
                return self.prompt_builder._build_greeting_prompt(safe_title, mood, style)
            else:
                return self.prompt_builder.build_conversational_prompt(
                    user_input=user_input,
                    mood=mood,
                    style=style,
                    memory_limit=3
                )
    
    async def generate_response(
        self,
        user_input: str,
        mood: str = "behave",
        style: str = "chaotic",
        stream: bool = False,
        custom_prompt: Optional[str] = None,
        is_greeting: bool = False,
        user_title: str = "",
        preferences: Optional[Dict[str, Any]] = None,  # Add this parameter
        user_permissions: Optional[Dict[str, Any]] = None,
        last_topic: Optional[str] = None,
        continuation_state: Optional[ContinuationState] = None,
        mode: Optional[str] = None,
        frozen_emotion: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate conversational response with automatic retry and fallback
        
        NOW AUTOMATICALLY USES MINIMAL PROMPTS FOR CHARACTER MODELS!
        
        Args:
            user_input: User's message
            mood: Current mood
            style: Current style
            stream: Whether to stream response
            custom_prompt: Optional custom prompt override
            is_greeting: Whether this is a greeting
            user_title: User's preferred title (for greetings)
            preferences: User preferences (ignored, for compatibility)
            
        Returns:
            Generated response text
        """
        # Ignore preferences parameter (for compatibility with old code)
        
        # Freeze emotion once per response (do not re-query during generation)
        if frozen_emotion is not None:
            emotion = frozen_emotion
        else:
            emotion = (
                self.emotion_engine.get_current_emotion()
                if self.emotion_engine
                else "neutral"
            )

        # Apply lightweight meta-controller to validate/normalize mood/style and length hints
        mood, style, length_hint = self._apply_meta_controller(mood, style, preferences)
        user_permissions = user_permissions or {}
        if continuation_state is None:
            continuation_state = ContinuationState(0)
        
        for attempt in range(self.config.max_retries):
            try:
                # Check if LLM is available
                if not self.is_available:
                    log.warning(f"LLM unavailable, attempt {attempt + 1}/{self.config.max_retries}")
                    
                    if await self._try_restart_ollama():
                        continue
                    
                    if self.config.fallback_on_failure:
                        return self._get_fallback_response(mood, style)
                    
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                
                # Build prompt (automatically chooses control header for character models or full prompt for standard models)
                prompt = self._build_prompt(
                    user_input,
                    mood,
                    style,
                    emotion=emotion,
                    custom_prompt=custom_prompt,
                    is_greeting=is_greeting,
                    user_title=user_title,
                    length_hint=length_hint,
                    preferences=preferences,
                )

                # Log prompt type for debugging (do not print user input)
                if self.is_character_model:
                    log.debug(f"📝 Control prompt ({len(prompt)} chars)")
                else:
                    log.debug(f"📝 Full prompt ({len(prompt)} chars)")

                # LoRA switching should consider mood/style/emotion as well
                if self.lora_manager:
                    try:
                        state = self.emotion_engine.get_state_dict() if self.emotion_engine else {}
                        state.update({"mood": mood, "style": style, "emotion": emotion})
                        target_style = self.lora_manager.select_for_emotion(state)
                        if target_style:
                            switched = self.lora_manager.switch_adapter(target_style)
                            if switched:
                                log.debug(f"🦊 LoRA switched to: {target_style}")
                    except Exception as e:
                        log.debug(f"LoRA switching failed: {e}")

                # Adjust generation options based on mood/style/emotion and length hint
                options = self._get_generation_options(mood, style, emotion, length_hint)
                
                if stream:
                    # Stream response (collect chunks internally, then handle actions deterministically)
                    chunks = []
                    async for chunk in self.stream_response(prompt, **options):
                        chunks.append(chunk)
                    raw_response = "".join(chunks)
                else:
                    # Non-streaming: single generation
                    raw_response = await asyncio.to_thread(self.adapter.generate, prompt, **options)

                # --- ACTION TOKEN FLOW (strict, single action, single continuation) ---
                try:
                    parsed = parse_action_from_text(raw_response)
                except Exception as e:
                    log.warning(f"Action parser exception: {e}")
                    parsed = {"ok": False, "error": "parser exception"}

                # Reset last action decision (machine-only)
                self.last_action_decision = None

                # If parser error, ignore actions and return sanitized raw response
                if not parsed.get("ok"):
                    log.debug(f"Action parse failed: {parsed.get('error')}")
                    return self._sanitize_final_response(raw_response)

                action = parsed.get("action")
                # Detect plain '<continue>' marker (training-time token) on its own line
                if action is None:
                    # find exact line matches for '<continue>'
                    lines = [ln.strip() for ln in raw_response.splitlines() if ln.strip()]
                    continue_lines = [ln for ln in lines if ln == "<continue>"]
                    if len(continue_lines) == 1:
                        # Treat as an implicit continue action
                        from core.meta.action_parser import Action as ActionClass

                        action = ActionClass(kind="continue", payload={"reason": None}, raw="<continue>")
                    elif len(continue_lines) > 1:
                        # Multiple continue markers -> treat as parse error
                        log.warning("Multiple <continue> markers found; ignoring action")
                        return self._sanitize_final_response(raw_response)
                if action is None:
                    # No action; return sanitized output
                    return self._sanitize_final_response(raw_response)

                # Ask meta-controller (deterministic)
                emotion_state = {"intensity": 0.0}
                try:
                    if self.emotion_engine and hasattr(self.emotion_engine, "get_current_intensity"):
                        emotion_state["intensity"] = float(self.emotion_engine.get_current_intensity())
                except Exception:
                    pass

                decision = self.meta_controller.decide_and_handle_action(
                    action,
                    user_permissions=user_permissions,
                    last_topic=last_topic,
                    continuation_state=continuation_state,
                    emotion_state=emotion_state,
                )

                # Store decision for UI (machine-only)
                self.last_action_decision = decision

                # If requires confirmation: do not execute; return sanitized original reply
                if decision.get("requires_confirmation"):
                    log.info("Action requires confirmation; not executing")
                    return self._sanitize_final_response(raw_response)

                # If not approved or no execute callable: ignore silently
                if not decision.get("approved") or not callable(decision.get("execute")):
                    log.info("Action denied or no executor; ignoring")
                    return self._sanitize_final_response(raw_response)

                # Execute approved action once (safe stub)
                try:
                    exec_fn = decision.get("execute")
                    exec_result = exec_fn()
                except Exception as e:
                    log.warning(f"Action execution failed: {e}")
                    return self._sanitize_final_response(raw_response)

                # Build machine-only tool result
                tool_result = {"action": action.kind, "result": exec_result}

                # If continue action, update continuation_state if allowed
                if action.kind == "continue" and isinstance(exec_result, dict):
                    if exec_result.get("allowed"):
                        continuation_state.count = exec_result.get("new_count", continuation_state.count)

                # Single continuation: inject internal tool result and re-invoke LLM once
                try:
                    cont_prompt = prompt + "\n[INTERNAL] Tool result (machine-only): " + json.dumps(tool_result)
                    cont_resp = await asyncio.to_thread(self.adapter.generate, cont_prompt, **options)
                    # Ignore any actions present in continuation (no recursion)
                    final_raw = cont_resp
                except Exception as e:
                    log.warning(f"Continuation generation failed: {e}")
                    final_raw = raw_response

                # Strip machine-only sequences and return
                return self._sanitize_final_response(final_raw)
                    
            except Exception as e:
                log.error(f"Generation failed (attempt {attempt + 1}): {e}")
                self.is_available = False
                
                if attempt == self.config.max_retries - 1:
                    if self.config.fallback_on_failure:
                        log.warning("All retries exhausted, using fallback")
                        return self._get_fallback_response(mood, style)
                    else:
                        raise
                
                await asyncio.sleep(self.config.retry_delay)
        
        if self.config.fallback_on_failure:
            return self._get_fallback_response(mood, style)
        else:
            raise Exception("LLM generation failed after all retries")
    
    async def stream_response(
        self,
        prompt: str,
        **options
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens"""
        def _stream_sync():
            for chunk in self.adapter.stream(prompt, **options):
                yield chunk
        
        loop = asyncio.get_event_loop()
        gen = await loop.run_in_executor(None, _stream_sync)
        
        for chunk in gen:
            yield chunk
    
    async def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotional content of text with fallback
        
        NOTE: This still uses standard prompts even for character models
        because emotion analysis is a classification task, not dialogue
        """
        fallback_result = {
            "intent": "chat",
            "sentiment": "neutral",
            "trigger": None,
            "hf_emotion": "neutral"
        }
        
        if not self.is_available:
            log.warning("LLM unavailable for emotion analysis, using fallback")
            await self._try_restart_ollama()
            return fallback_result
        
        # Use full prompt builder for analysis (even for character models)
        if self.is_character_model:
            # Temporarily create full prompt builder for analysis
            from core.llm.prompt_builder import PromptBuilder
            temp_builder = PromptBuilder("", self.memory)
            prompt = temp_builder.build_emotion_analysis_prompt(text)
        else:
            prompt = self.prompt_builder.build_emotion_analysis_prompt(text)
        
        try:
            response = await asyncio.to_thread(
                self.adapter.generate,
                prompt,
                temperature=0.1,
                num_predict=128
            )
            
            result = self._parse_json_response(response)
            
            result.setdefault("intent", "unknown")
            result.setdefault("sentiment", "neutral")
            result.setdefault("trigger", None)
            result.setdefault("hf_emotion", "neutral")
            
            return result
            
        except Exception as e:
            log.warning(f"Emotion analysis failed: {e}")
            self.is_available = False
            return fallback_result
    
    async def plan_reaction(
        self,
        user_input: str,
        emotion_analysis: Dict[str, Any],
        mood: str,
        style: str
    ) -> Dict[str, Any]:
        """Plan reaction based on emotion analysis with fallback"""
        fallback_result = {
            "plan": f"respond in {mood}/{style} mode",
            "expression": "neutral",
            "retaliation": "none"
        }
        
        if not self.is_available:
            log.warning("LLM unavailable for reaction planning, using fallback")
            return fallback_result
        
        # Use full prompt for planning (even for character models)
        if self.is_character_model:
            from core.llm.prompt_builder import PromptBuilder
            temp_builder = PromptBuilder("", self.memory)
            prompt = temp_builder.build_reaction_planning_prompt(
                user_input, emotion_analysis, mood, style
            )
        else:
            prompt = self.prompt_builder.build_reaction_planning_prompt(
                user_input, emotion_analysis, mood, style
            )
        
        try:
            response = await asyncio.to_thread(
                self.adapter.generate,
                prompt,
                temperature=0.3,
                num_predict=128
            )
            
            result = self._parse_json_response(response)
            
            result.setdefault("plan", f"respond in {mood}/{style} mode")
            result.setdefault("expression", "neutral")
            result.setdefault("retaliation", "none")
            
            return result
            
        except Exception as e:
            log.warning(f"Reaction planning failed: {e}")
            self.is_available = False
            return fallback_result
    
    def _get_generation_options(self, mood: str, style: str, emotion: Optional[str] = None, length_hint: Optional[Any] = None) -> Dict[str, Any]:
        """Return options tuned deterministically by mood/style/emotion and a length hint.

        Emotion/mood/style affect both sampling temperature and token budget. Keep
        defaults conservative for low-spec machines.
        """
        def _length_to_num(hint: Optional[Any]) -> int:
            if hint is None:
                return 128
            if isinstance(hint, int):
                return max(16, int(hint))
            if hint == "short":
                return 64
            if hint == "long":
                return 256
            return 128  # medium/default

        # Base token budget
        num_predict = _length_to_num(length_hint)

        # Base temperature from configured temperature, adjusted by style/mood
        temp = float(self.temperature)

        # Style-based temperature nudges
        if style == "silent":
            temp = min(temp, 0.25)
            num_predict = min(64, num_predict)
        elif style == "chaotic":
            temp = max(temp, 0.9)
            num_predict = max(128, num_predict)
        elif style == "cold":
            temp = min(max(temp * 0.6, 0.2), 0.5)
            num_predict = min(128, num_predict)
        elif style == "sweet":
            temp = min(max(temp * 0.8, 0.4), 0.8)

        # Mood influences
        if mood == "mean":
            temp = max(0.2, temp * 0.8)
        elif mood == "flirty":
            temp = min(0.95, temp * 1.1)

        # Emotion may further tweak sampling (intensity or valence)
        try:
            if isinstance(emotion, dict) and "intensity" in emotion:
                intensity = float(emotion.get("intensity", 0.0))
                # more intense -> slightly higher temp
                temp = min(1.0, temp + 0.15 * intensity)
        except Exception:
            pass

        options = {
            "temperature": float(round(temp, 3)),
            "num_predict": int(num_predict),
            "stop": ["\nUser:", "\n###", "\nRESPONSE:"]
        }

        return options

    def _apply_meta_controller(self, mood: str, style: str, preferences: Optional[Dict[str, Any]] = None):
        """Lightweight meta-controller that validates mood/style and extracts a length hint.

        This keeps decisions deterministic, tiny, and outside the model prompt.
        """
        valid_moods = {"behave", "mean", "flirty"}
        valid_styles = {"chaotic", "sweet", "cold", "silent"}

        mood = mood if mood in valid_moods else "behave"
        style = style if style in valid_styles else "chaotic"

        # length hint can be provided in preferences or derived from style
        length_hint = None
        if preferences:
            lh = preferences.get("length") or preferences.get("max_tokens")
            if isinstance(lh, int):
                length_hint = lh
            elif isinstance(lh, str) and lh in {"short", "medium", "long"}:
                length_hint = lh

        if length_hint is None:
            if style == "silent" or style == "cold":
                length_hint = "short"
            elif style == "chaotic":
                length_hint = "medium"
            else:
                length_hint = "medium"

        return mood, style, length_hint

    def _sanitize_user_title(self, title: str) -> str:
        """Remove header-like tokens from a user supplied title to avoid prompt injection."""
        if not title:
            return title
        bad_tokens = ["System:", "Assistant:", "AI:", "Kitsu:", "###"]
        clean = title
        for t in bad_tokens:
            clean = clean.replace(t, "")
        clean = clean.strip().replace("\n", " ")
        return clean[:64]

    
    def _clean_response(self, response: str) -> str:
        """Clean up LLM response"""
        prefixes = ["Kitsu:", "Assistant:", "AI:", "Response:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response.strip()

    def _sanitize_final_response(self, text: str) -> str:
        """Remove any machine-only tokens before returning to the user.

        Strips:
          - Any full-line <action:...> tokens
          - Any <thought>...</thought> blocks (multiline)
          - Any lines beginning with [INTERNAL
        """
        if not text:
            return ""

        # Remove <thought>...</thought> blocks (multiline, case-insensitive)
        text = re.sub(r"(?is)<thought>.*?</thought>", "", text)

        # Remove any full-line action tokens
        text = re.sub(r"(?m)^\s*<action:[^>]+>\s*$", "", text)

        # Remove lines starting with [INTERNAL
        text = re.sub(r"(?m)^\s*\[INTERNAL[^\n]*\n?", "", text)

        # Remove leftover angled tags on their own lines
        text = re.sub(r"(?m)^\s*<[^>]+>\s*$", "", text)

        # Clean common assistant prefixes
        text = self._clean_response(text)

        return text.strip()
    
    def get_lora_status(self) -> Dict[str, Any]:
        """Get current LoRA adapter status"""
        if not self.lora_manager:
            return {"available": False}
        
        try:
            stats = self.lora_manager.get_stats()
            return {
                "available": True,
                "current_style": stats["current_style"],
                "current_adapter": stats["current_adapter"],
                "total_switches": stats["switch_count"],
                "available_styles": stats["available_styles"]
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            return json.loads(response.strip())
        except:
            pass
        
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            return json.loads(response[start:end])
        except:
            pass
        
        log.warning(f"Failed to parse JSON from response: {response[:100]}...")
        return {}
    
    def reload_templates(self):
        """Reload mode templates (only for standard models)"""
        if not self.is_character_model:
            self.prompt_builder.reload_templates()
        else:
            log.info("Character model - no templates to reload")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current LLM status"""
        status = {
            "available": self.is_available,
            "model": self.model,
            "is_character_model": self.is_character_model,
            "prompt_mode": "control" if self.is_character_model else "full",
            "temperature": self.temperature,
            "auto_restart": self.config.auto_restart,
            "fallback_enabled": self.config.fallback_on_failure,
            "max_retries": self.config.max_retries,
            "active_lora_style": getattr(self, "active_lora_style", None)
        }
        
        # Add LoRA status
        if self.lora_manager:
            status["lora"] = self.get_lora_status()
        
        return status


    def register_lora(self, style: str, path: str) -> None:
        """Register a LoRA adapter path for a style at runtime.

        This is a cheap registry operation. Applying the LoRA to an in-memory
        HF model can be done via `apply_active_lora_to_model`.
        """
        self.lora_router.register(style, path)

    def select_lora_style(self, style: str) -> None:
        """Select which style's LoRA should be active (cheap local op).

        This does not reload base model by itself; apply_active_lora_to_model
        will perform a best-effort PEFT adapter application to the in-memory
        model if available.
        """
        self.active_lora_style = style or "chaotic"
        self.lora_router.set_active(self.active_lora_style)

    def apply_active_lora_to_model(self, base_model) -> Any:
        """Best-effort: apply the active LoRA adapter to an in-memory HF model.

        Returns wrapped model on success, or original base_model on failure.
        """
        return self.lora_router.apply_to_peft_model(base_model, self.active_lora_style)
    
    def switch_to_character_model(self, model_name: str = "kitsu:character"):
        """
        Switch to character model at runtime
        
        Args:
            model_name: Name of character model
        """
        log.info(f"🔄 Switching to character model: {model_name}")
        
        self.model = model_name
        self.is_character_model = True
        
        # Character models use machine-readable control prompts; no natural-language prompt builder
        self.prompt_builder = None

        # Update adapter (keep streaming behavior)
        self.adapter = OllamaAdapter(
            model=self.model,
            temperature=self.temperature,
            streaming=self.adapter.streaming
        )
        
        # Check availability
        self._check_availability()
        
        log.info("✅ Switched to character model with minimal prompts")
    
    def switch_to_standard_model(
        self,
        model_name: str = "gemma:2b",
        character_context: str = ""
    ):
        """
        Switch to standard model at runtime
        
        Args:
            model_name: Name of standard model
            character_context: Character context for full prompts
        """
        log.info(f"🔄 Switching to standard model: {model_name}")
        
        self.model = model_name
        self.is_character_model = False
        
        # Replace prompt builder
        self.prompt_builder = PromptBuilder(
            character_context=character_context,
            memory_manager=self.memory,
            templates_path=Path("data/templates")
        )
        
        # Update adapter
        self.adapter = OllamaAdapter(
            model=self.model,
            temperature=self.temperature,
            streaming=self.adapter.streaming
        )
        
        # Check availability
        self._check_availability()
        
        log.info("✅ Switched to standard model with full prompts")