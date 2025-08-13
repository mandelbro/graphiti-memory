"""
Custom Ollama client that extends OpenAI client to support Ollama-specific model parameters.

This module provides an OllamaClient that can pass model-specific parameters
like num_ctx, top_p, etc. to the Ollama API. It uses a hybrid approach:
- Uses native Ollama API for completions to preserve parameters
- Converts responses to OpenAI format for compatibility
"""

import json
import logging
import typing
from typing import Any

import httpx
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig
from graphiti_core.llm_client.openai_base_client import BaseOpenAIClient
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from .utils.ollama_health_validator import OllamaHealthValidator
from .utils.ollama_response_converter import OllamaResponseConverter

logger = logging.getLogger(__name__)


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
        model_parameters: dict[str, Any] | None = None,
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

        # Initialize health validator utility
        self._health_validator = OllamaHealthValidator(
            self.ollama_base_url or "http://localhost:11434"
        )

        # Initialize response converter utility
        self._response_converter = OllamaResponseConverter()

        # Connection pooling infrastructure
        self._http_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - close HTTP client and health validator."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
        # Close health validator resources
        await self._health_validator.__aexit__(exc_type, exc_val, exc_tb)

    async def _get_http_client(self) -> httpx.AsyncClient:
        """
        Get or create a shared HTTP client with connection pooling.

        Delegates to health validator's HTTP client for consistency.

        Returns:
            httpx.AsyncClient: The shared HTTP client instance
        """
        if self._http_client is None or self._http_client.is_closed:
            # Create new HTTP client with connection pooling configuration
            limits = httpx.Limits(
                max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0
            )

            timeout = httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=2.0)

            self._http_client = httpx.AsyncClient(limits=limits, timeout=timeout)

        return self._http_client

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
    ):
        """Enhanced structured completion with JSON parsing for Ollama.

        This method attempts to parse JSON responses from Ollama and populate
        the parsed field when successful, providing better structured output
        handling while maintaining compatibility with the base client.
        """
        # Get regular completion from Ollama
        response = await self._create_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model,
        )

        # Attempt to parse structured response
        try:
            content = response.choices[0].message.content
            if content and content.strip().startswith("{"):
                # Try to parse as JSON
                parsed_data = json.loads(content.strip())
                # Validate against the response model
                parsed_model = response_model(**parsed_data)
                # Update the response with parsed data using converter utility
                self._response_converter.set_parsed_response(response, parsed_model)
                logger.debug(
                    f"Successfully parsed structured response for model {model}"
                )

        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON response from Ollama model {model}: {e}"
            )
            # Continue with parsed=None, which is fine for fallback handling
        except ValidationError as e:
            logger.warning(
                f"Failed to validate parsed data against {response_model.__name__} for model {model}: {e}"
            )
            # Continue with parsed=None, which is fine for fallback handling
        except Exception as e:
            logger.warning(
                f"Unexpected error during structured response parsing for model {model}: {e}"
            )
            # Continue with parsed=None, which is fine for fallback handling

        return response

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
        prompt = self._response_converter.messages_to_prompt(messages)

        # Use native Ollama API with parameters
        base_url = self.ollama_base_url or "http://localhost:11434"
        native_url = base_url.replace("/v1", "").rstrip("/")
        api_url = f"{native_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self.model_parameters.copy() if self.model_parameters else {},
        }

        # Add keep_alive if specified in model_parameters
        if self.model_parameters and "keep_alive" in self.model_parameters:
            payload["keep_alive"] = self.model_parameters["keep_alive"]

        # Add temperature and max_tokens to options if provided
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        client = await self._get_http_client()
        response = await client.post(api_url, json=payload, timeout=60.0)
        response.raise_for_status()
        response_data = response.json()

        # Convert native response to OpenAI format
        return self._response_converter.convert_native_response_to_openai(
            response_data, model
        )

    async def check_health(self) -> tuple[bool, str]:
        """
        Check Ollama server health.

        Delegates to health validator utility for health checking with caching.

        Returns:
            tuple[bool, str]: (is_healthy, message)
        """
        return await self._health_validator.check_health()

    async def validate_model_available(self, model: str) -> tuple[bool, str]:
        """
        Validate if a model is available on the Ollama server.

        Delegates to health validator utility for model validation with caching.

        Args:
            model: The model name to validate

        Returns:
            tuple[bool, str]: (is_available, message)
        """
        return await self._health_validator.validate_model_available(model)

    def _messages_to_prompt(self, messages: list[ChatCompletionMessageParam]) -> str:
        """
        Convert OpenAI messages format to a simple prompt for native API.

        Compatibility method that delegates to the response converter utility.

        Args:
            messages: List of chat completion messages in OpenAI format

        Returns:
            str: Formatted prompt string for Ollama native API
        """
        return self._response_converter.messages_to_prompt(messages)
