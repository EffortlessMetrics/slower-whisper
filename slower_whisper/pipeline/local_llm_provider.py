"""Local LLM provider with lazy loading and optional dependency handling (#89).

This module provides a standalone LocalLLMProvider that:
1. Does NOT require torch/transformers at import time
2. Lazily loads models only when first used
3. Handles missing dependencies gracefully
4. Supports configurable model selection (qwen, smollm, etc.)

Usage:
    >>> from transcription.local_llm_provider import LocalLLMProvider, is_available
    >>> if is_available():
    ...     provider = LocalLLMProvider("Qwen/Qwen2.5-3B-Instruct")
    ...     response = provider.generate("What is 2+2?")
    ... else:
    ...     print("Local LLM not available - install torch and transformers")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Optional dependency availability flags
_TORCH_AVAILABLE: bool | None = None
_TRANSFORMERS_AVAILABLE: bool | None = None


def _check_torch_available() -> bool:
    """Check if torch is available (cached)."""
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401

            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _check_transformers_available() -> bool:
    """Check if transformers is available (cached)."""
    global _TRANSFORMERS_AVAILABLE
    if _TRANSFORMERS_AVAILABLE is None:
        try:
            import transformers  # noqa: F401

            _TRANSFORMERS_AVAILABLE = True
        except ImportError:
            _TRANSFORMERS_AVAILABLE = False
    return _TRANSFORMERS_AVAILABLE


def is_available() -> bool:
    """Check if local LLM provider is available.

    Returns True if both torch and transformers are installed.

    This is a fast check that does not load any models.
    """
    return _check_torch_available() and _check_transformers_available()


def get_availability_status() -> dict[str, bool]:
    """Get detailed availability status for all dependencies.

    Returns:
        Dictionary with 'torch', 'transformers', and 'available' keys.
    """
    return {
        "torch": _check_torch_available(),
        "transformers": _check_transformers_available(),
        "available": is_available(),
    }


@dataclass
class LocalLLMResponse:
    """Response from local LLM generation.

    Attributes:
        text: Generated text.
        tokens_used: Number of tokens generated.
        duration_ms: Generation time in milliseconds.
        model: Model identifier used.
    """

    text: str
    tokens_used: int = 0
    duration_ms: int = 0
    model: str = ""


class LocalLLMProvider:
    """Local LLM provider with lazy loading.

    Supports models like:
    - Qwen/Qwen2.5-7B-Instruct (good balance of quality/speed)
    - Qwen/Qwen2.5-3B-Instruct (faster, slightly lower quality)
    - HuggingFaceTB/SmolLM2-1.7B-Instruct (very fast, smaller context)
    - microsoft/phi-3-mini-4k-instruct (Microsoft's efficient model)

    Model loading is lazy - the model is only loaded on the first generate() call.
    This ensures that importing this module does not require torch/transformers.

    Attributes:
        model_name: Model identifier from HuggingFace Hub.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens to generate.
        device: Device for inference ("auto", "cuda", "cpu").
    """

    # Default model if none specified
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    # Smaller models suitable for testing or resource-constrained environments
    SMALL_MODELS = [
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/phi-3-mini-4k-instruct",
    ]

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        device: str = "auto",
    ) -> None:
        """Initialize the local LLM provider.

        Args:
            model_name: Model identifier from HuggingFace Hub.
                       Defaults to Qwen/Qwen2.5-7B-Instruct.
            temperature: Sampling temperature. Use 0.0 for deterministic output.
            max_tokens: Maximum tokens to generate.
            device: Device for inference. Options:
                   - "auto": Use CUDA if available, else CPU
                   - "cuda": Force CUDA (fails if unavailable)
                   - "cpu": Force CPU
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device

        # Lazy-loaded components
        self._model: Any = None
        self._tokenizer: Any = None
        self._resolved_device: str | None = None
        self._load_lock = asyncio.Lock()
        self._sync_load_lock: Any = None  # threading.Lock, loaded lazily

    def _ensure_available(self) -> None:
        """Check that dependencies are available.

        Raises:
            ImportError: If torch or transformers is not installed.
        """
        if not _check_torch_available():
            raise ImportError(
                "torch is required for local LLM inference. "
                "Install with: pip install 'slower-whisper[emotion]'"
            )
        if not _check_transformers_available():
            raise ImportError(
                "transformers is required for local LLM inference. "
                "Install with: pip install 'slower-whisper[emotion]'"
            )

    def _load_model_sync(self) -> tuple[Any, Any, str]:
        """Load model synchronously (blocking).

        Returns:
            Tuple of (tokenizer, model, device).

        Raises:
            ImportError: If dependencies not available.
            RuntimeError: If model loading fails.
        """
        self._ensure_available()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading local LLM model: {self.model_name}")
        start_time = time.time()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Resolve device
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        # Select dtype based on device
        dtype = torch.float16 if device == "cuda" else torch.float32

        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,  # Required for some models like Qwen
            )

            if device == "cpu":
                model = model.to(device)

        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}") from e

        load_time = time.time() - start_time
        logger.info(f"Loaded {self.model_name} on {device} in {load_time:.1f}s")

        return tokenizer, model, device

    def _ensure_model_loaded_sync(self) -> None:
        """Ensure model is loaded (thread-safe, blocking).

        This uses a threading lock for synchronous contexts.
        """
        if self._model is not None:
            return

        # Lazy import threading to avoid issues
        import threading

        if self._sync_load_lock is None:
            self._sync_load_lock = threading.Lock()

        with self._sync_load_lock:
            if self._model is not None:
                return  # type: ignore[unreachable]  # Double-check pattern
            self._tokenizer, self._model, self._resolved_device = self._load_model_sync()

    async def _ensure_model_loaded_async(self) -> None:
        """Ensure model is loaded (async-safe).

        Uses asyncio.Lock for async contexts and runs loading in executor.
        """
        if self._model is not None:
            return

        async with self._load_lock:
            if self._model is not None:
                return  # type: ignore[unreachable]  # Double-check pattern

            loop = asyncio.get_event_loop()
            self._tokenizer, self._model, self._resolved_device = await loop.run_in_executor(
                None,
                self._load_model_sync,
            )

    def _generate_sync(self, prompt: str) -> tuple[str, int]:
        """Generate text synchronously.

        Args:
            prompt: Formatted prompt string.

        Returns:
            Tuple of (generated_text, token_count).
        """
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only new tokens (not the prompt)
        input_length = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_length:]
        response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)

        return response_text, len(response_tokens)

    def _build_chat_prompt(self, system: str, user: str) -> str:
        """Build chat prompt using model's template.

        Args:
            system: System prompt.
            user: User prompt.

        Returns:
            Formatted prompt string.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Use chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            return str(
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            # Fallback for models without chat template
            return f"{system}\n\nUser: {user}\n\nAssistant:"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> LocalLLMResponse:
        """Generate text using the local LLM (synchronous).

        Args:
            prompt: The user prompt/query.
            system_prompt: Optional system prompt for context.

        Returns:
            LocalLLMResponse with generated text and metadata.

        Raises:
            ImportError: If dependencies not available.
            RuntimeError: If generation fails.
        """
        self._ensure_model_loaded_sync()

        start_time = time.perf_counter_ns()

        # Build prompt
        chat_prompt = self._build_chat_prompt(system_prompt, prompt)

        # Generate
        text, token_count = self._generate_sync(chat_prompt)

        duration_ms = (time.perf_counter_ns() - start_time) // 1_000_000

        return LocalLLMResponse(
            text=text,
            tokens_used=token_count,
            duration_ms=duration_ms,
            model=self.model_name,
        )

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> LocalLLMResponse:
        """Generate text using the local LLM (asynchronous).

        Args:
            prompt: The user prompt/query.
            system_prompt: Optional system prompt for context.

        Returns:
            LocalLLMResponse with generated text and metadata.

        Raises:
            ImportError: If dependencies not available.
            RuntimeError: If generation fails.
        """
        await self._ensure_model_loaded_async()

        start_time = time.perf_counter_ns()

        # Build prompt
        chat_prompt = self._build_chat_prompt(system_prompt, prompt)

        # Generate in executor to avoid blocking
        loop = asyncio.get_event_loop()
        text, token_count = await loop.run_in_executor(
            None,
            self._generate_sync,
            chat_prompt,
        )

        duration_ms = (time.perf_counter_ns() - start_time) // 1_000_000

        return LocalLLMResponse(
            text=text,
            tokens_used=token_count,
            duration_ms=duration_ms,
            model=self.model_name,
        )

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded.

        Returns:
            True if model is loaded, False otherwise.
        """
        return self._model is not None

    def get_device(self) -> str | None:
        """Get the device the model is loaded on.

        Returns:
            Device string ("cuda" or "cpu") if loaded, None otherwise.
        """
        return self._resolved_device


class MockLocalLLMProvider:
    """Mock provider for testing without real model.

    Returns predefined responses for testing semantic extraction.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str | None = None,
    ) -> None:
        """Initialize mock provider.

        Args:
            responses: Dictionary mapping prompt substrings to responses.
            default_response: Default response when no match found.
        """
        self.responses = responses or {}
        self.default_response = default_response or '{"topics": [], "risks": [], "actions": []}'
        self.model_name = "mock-local-llm"
        self.call_count = 0
        self.calls: list[tuple[str, str]] = []

    def generate(self, prompt: str, system_prompt: str = "") -> LocalLLMResponse:
        """Return mock response."""
        self.call_count += 1
        self.calls.append((system_prompt, prompt))

        # Look for matching response
        for key, response in self.responses.items():
            if key in prompt or key in system_prompt:
                return LocalLLMResponse(
                    text=response,
                    tokens_used=len(response.split()),
                    duration_ms=50,
                    model=self.model_name,
                )

        return LocalLLMResponse(
            text=self.default_response,
            tokens_used=len(self.default_response.split()),
            duration_ms=50,
            model=self.model_name,
        )

    async def generate_async(self, prompt: str, system_prompt: str = "") -> LocalLLMResponse:
        """Return mock response (async)."""
        return self.generate(prompt, system_prompt)

    def is_loaded(self) -> bool:
        """Mock is always 'loaded'."""
        return True

    def get_device(self) -> str:
        """Mock device."""
        return "cpu"


# Type guard for checking provider availability at runtime
if TYPE_CHECKING:
    # For type checking, assume full types
    pass


__all__ = [
    "LocalLLMProvider",
    "LocalLLMResponse",
    "MockLocalLLMProvider",
    "is_available",
    "get_availability_status",
]
