"""
Custom Ollama client that extends OpenAI client to support Ollama-specific model parameters.

This module provides an OllamaClient that can pass model-specific parameters
like num_ctx, top_p, etc. to the Ollama API. It uses a hybrid approach:
- Uses native Ollama API for completions to preserve parameters
- Converts responses to OpenAI format for compatibility
"""

import typing
from typing import Dict, Any
import httpx

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig
from graphiti_core.llm_client.openai_base_client import BaseOpenAIClient


class OllamaClient(BaseOpenAIClient):
    """
    OllamaClient extends the BaseOpenAIClient to support Ollama-specific model parameters.

    This client can pass additional model parameters like num_ctx, top_p, repeat_penalty,
    etc. to the Ollama API through the OpenAI-compatible interface.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_parameters: Dict[str, Any] | None = None,
    ):
        """
        Initialize the OllamaClient with the provided configuration and model parameters.

        Args:
            config: The configuration for the LLM client
            cache: Whether to use caching for responses
            client: An optional async client instance to use
            max_tokens: Maximum tokens for responses
            model_parameters: Ollama-specific model parameters (num_ctx, top_p, etc.)
        """
        super().__init__(config, cache, max_tokens)

        if config is None:
            config = LLMConfig()

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

        # Store Ollama-specific model parameters
        self.model_parameters = model_parameters or {}

        # Store base URL for native Ollama API calls
        self.ollama_base_url = config.base_url if config else "http://localhost:11434"

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
    ):
        """Create a structured completion with Ollama model parameters.

        Since Ollama doesn't support OpenAI's structured output API properly,
        we fall back to using the regular completion API with the native Ollama API.
        """
        # Fall back to regular completion since Ollama doesn't support structured output properly
        return await self._create_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model,
        )

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ):
        """Create a regular completion using native Ollama API to preserve parameters."""
        # Convert messages to a prompt for native API
        prompt = self._messages_to_prompt(messages)

        # Use native Ollama API with parameters
        native_url = self.ollama_base_url.replace("/v1", "")
        api_url = f"{native_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self.model_parameters if self.model_parameters else {}
        }

        # Add keep_alive if specified in model_parameters
        if self.model_parameters and "keep_alive" in self.model_parameters:
            payload["keep_alive"] = self.model_parameters["keep_alive"]

        # Add temperature and max_tokens to options if provided
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=60.0)
            response.raise_for_status()
            response_data = response.json()

            # Convert native response to OpenAI format
            return self._convert_native_response_to_openai(response_data, model)

    def _messages_to_prompt(self, messages: list[ChatCompletionMessageParam]) -> str:
        """Convert OpenAI messages format to a simple prompt for native API."""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n".join(prompt_parts)

    def _convert_native_response_to_openai(self, native_response: dict, model: str):
        """Convert native Ollama response to OpenAI format."""
        import time

        # Create a mock OpenAI response structure
        class MockChoice:
            def __init__(self, message_content: str):
                self.message = MockMessage(message_content)
                self.index = 0
                self.finish_reason = "stop"

        class MockMessage:
            def __init__(self, content: str):
                self.content = content
                self.role = "assistant"
                self.parsed = None  # Ollama doesn't support structured output
                self.refusal = None  # Ollama doesn't have refusal mechanism

        class MockResponse:
            def __init__(self, content: str, model: str):
                self.choices = [MockChoice(content)]
                self.model = model
                self.id = f"chatcmpl-{int(time.time())}"
                self.created = int(time.time())
                self.object = "chat.completion"

        return MockResponse(native_response.get("response", ""), model)
