#  core/llm/ollama_adapter.py

"""
Ollama adapter with proper UTF-8 handling and streaming support
"""

import logging
import subprocess
import json
from typing import Any, Dict, Optional, Generator

log = logging.getLogger(__name__)

# Try imports
try:
    from langchain_ollama import OllamaLLM
    from langchain_core.callbacks.base import BaseCallbackHandler
    _HAS_LANGCHAIN_OLLAMA = True
except Exception:
    OllamaLLM = None
    BaseCallbackHandler = None
    _HAS_LANGCHAIN_OLLAMA = False


def _run_ollama_cli(model_name: str, prompt: str, options: Optional[dict] = None, timeout: Optional[int] = None) -> str:
    """
    CLI fallback with proper UTF-8 handling.

    Supports a limited set of options passed via the Ollama CLI flags (when
    available). If the CLI does not support a given flag, we at minimum ensure
    a reasonable timeout is used to avoid infinite hangs.
    """
    try:
        cmd = ["ollama", "run", model_name]
        # Append prompt as a single argument (safer quoting)
        cmd.append(prompt)

        # Map some options to CLI flags when present
        if options:
            num_predict = options.get("num_predict") or options.get("max_new_tokens")
            if num_predict is not None:
                # Newer Ollama CLI uses kebab-case flags ("--num-predict").
                # Use the kebab-case flag and fall back to running without the flag
                # if the CLI reports it as unknown.
                cmd += ["--num-predict", str(int(num_predict))]
            temp = options.get("temperature")
            if temp is not None:
                cmd += ["--temperature", str(float(temp))]
            top_p = options.get("top_p")
            if top_p is not None:
                cmd += ["--top-p", str(float(top_p))]
            top_k = options.get("top_k")
            if top_k is not None:
                cmd += ["--top-k", str(int(top_k))]

        # Derive a sensible timeout if none provided (avoid unbounded runs)
        if timeout is None:
            timeout =  max(10, int((options.get("num_predict", 128) // 4))) if options else 30

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )

        # Retry without num-predict flag if CLI reports it as unknown
        if proc.returncode != 0 and proc.stderr and "unknown flag" in proc.stderr.lower() and ("--num_predict" in proc.stderr or "--num-predict" in proc.stderr):
            # Rebuild command without num_predict flags
            cmd_no_num = [c for c in cmd if c not in ("--num-predict", "--num_predict") and not (c.startswith("--num-predict=") or c.startswith("--num_predict="))]
            try:
                proc2 = subprocess.run(
                    cmd_no_num,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout
                )
                proc = proc2
            except subprocess.TimeoutExpired:
                raise RuntimeError("ollama CLI timed out")

        if proc.returncode != 0:
            raise RuntimeError(f"ollama CLI error: {proc.stderr.strip()}")
        return proc.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("ollama CLI not found")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ollama CLI timed out")
    except Exception as e:
        raise RuntimeError(f"ollama CLI failed: {e}")


class OllamaAdapter:
    """
    Clean adapter for Ollama with streaming support
    Handles both langchain and CLI modes
    """
    
    def __init__(
        self,
        model: str = "gemma:2b",
        temperature: float = 0.8,
        streaming: bool = True,
        timeout: Optional[int] = None,
        callback_handler: Optional[Any] = None
    ):
        self.model = model
        self.temperature = temperature
        self.streaming = streaming
        self.timeout = timeout
        self._llm = None
        
        # Try to initialize langchain
        if _HAS_LANGCHAIN_OLLAMA:
            try:
                callbacks = [callback_handler] if callback_handler else []
                self._llm = OllamaLLM(
                    model=self.model,
                    temperature=self.temperature,
                    streaming=self.streaming,
                    callbacks=callbacks
                )
                log.info(f"OllamaAdapter: using langchain for {self.model}")
            except Exception as e:
                log.warning(f"OllamaAdapter init failed: {e}, using CLI fallback")
    
    def generate(self, prompt: str, **options) -> str:
        """
        Generate response (non-streaming)
        
        Args:
            prompt: Input prompt
            **options: Ollama options (num_predict, top_p, etc.)
        
        Returns:
            Generated text
        """
        # Normalize options
        ollama_options = self._normalize_options(options)
        
        # Try langchain first
        if self._llm is not None:
            try:
                result = self._llm.invoke(prompt, options=ollama_options)
                return self._extract_text(result)
            except Exception as e:
                log.warning(f"Langchain generation failed: {e}, trying CLI")
        
        # Fallback to CLI and pass normalized options so token limits/temperature are applied
        return _run_ollama_cli(self.model, prompt, options=ollama_options, timeout=self.timeout)
    
    def stream(self, prompt: str, **options) -> Generator[str, None, None]:
        """
        Generate response with streaming
        
        Args:
            prompt: Input prompt
            **options: Ollama options
            
        Yields:
            Text chunks as they arrive
        """
        ollama_options = self._normalize_options(options)
        
        # Try langchain streaming
        if self._llm is not None and hasattr(self._llm, "stream"):
            try:
                for chunk in self._llm.stream(prompt, options=ollama_options):
                    text = self._extract_text(chunk)
                    if text:
                        yield text
                return
            except Exception as e:
                log.warning(f"Langchain streaming failed: {e}")
        
        # Fallback: non-streaming response (pass options through)
        result = self.generate(prompt, **options)
        yield result
    
    def _normalize_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize various option formats to Ollama format
        """
        normalized = {}
        
        # Temperature
        if "temperature" in options:
            normalized["temperature"] = float(options["temperature"])
        else:
            normalized["temperature"] = self.temperature
        
        # Max tokens (map HF to Ollama)
        if "max_new_tokens" in options:
            normalized["num_predict"] = int(options["max_new_tokens"])
        elif "num_predict" in options:
            normalized["num_predict"] = int(options["num_predict"])
        
        # Top-p and top-k
        if "top_p" in options:
            normalized["top_p"] = float(options["top_p"])
        if "top_k" in options:
            normalized["top_k"] = int(options["top_k"])
        
        # Drop unsupported options
        supported = {"temperature", "num_predict", "top_p", "top_k", "stop"}
        return {k: v for k, v in normalized.items() if k in supported}
    
    def _extract_text(self, result: Any) -> str:
        """
        Extract text from various result formats
        """
        if isinstance(result, str):
            return result
        
        if isinstance(result, dict):
            # Try common keys
            for key in ["content", "text", "response", "output"]:
                if key in result:
                    return str(result[key])
            return json.dumps(result)
        
        if hasattr(result, "content"):
            return str(result.content)
        
        return str(result)
