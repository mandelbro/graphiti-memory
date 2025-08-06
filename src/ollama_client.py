"""
Custom Ollama client that extends OpenAI client to support Ollama-specific model parameters.

This module provides an OllamaClient that can pass model-specific parameters
like num_ctx, top_p, etc. to the Ollama API through the OpenAI-compatible interface.
"""

import typing
from typing import Dict, Any

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

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
    ):
        """Create a structured completion with Ollama model parameters."""
        # Build the request parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_model,  # type: ignore
        }

        # Add Ollama-specific model parameters if any
        if self.model_parameters:
            # In Ollama's OpenAI-compatible API, additional model parameters
            # can be passed through the "extra_body" parameter
            params["extra_body"] = self.model_parameters

        return await self.client.beta.chat.completions.parse(**params)

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ):
        """Create a regular completion with Ollama model parameters."""
        # Build the request parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        # Add Ollama-specific model parameters if any
        if self.model_parameters:
            # In Ollama's OpenAI-compatible API, additional model parameters
            # can be passed through the "extra_body" parameter
            params["extra_body"] = self.model_parameters

        return await self.client.chat.completions.create(**params)
