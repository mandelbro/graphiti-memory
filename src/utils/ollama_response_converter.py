"""
Ollama Response Conversion Utilities.

This module provides utilities for converting between Ollama native responses
and OpenAI-compatible formats, including message formatting and mock response classes.
"""

import time
from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel


class MockMessage:
    """Mock OpenAI message class for Ollama compatibility."""

    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"
        self.parsed: BaseModel | None = None  # Can hold parsed structured output
        self.refusal = None  # Ollama doesn't have refusal mechanism

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation compatible with Pydantic model_dump()."""
        return {
            "content": self.content,
            "role": self.role,
            "parsed": self.parsed,
            "refusal": self.refusal,
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": None,
        }


class MockChoice:
    """Mock OpenAI choice class for Ollama compatibility."""

    def __init__(self, message_content: str):
        self.message = MockMessage(message_content)
        self.index = 0
        self.finish_reason = "stop"


class MockResponse:
    """Mock OpenAI response class for Ollama compatibility."""

    def __init__(self, content: str, model: str):
        self.choices = [MockChoice(content)]
        self.model = model
        self.id = f"chatcmpl-{int(time.time())}"
        self.created = int(time.time())
        self.object = "chat.completion"


class OllamaResponseConverter:
    """
    Utility class for converting between Ollama and OpenAI response formats.

    Provides methods to convert messages, responses, and maintain compatibility
    with OpenAI's chat completion format while using Ollama's native API.
    """

    @staticmethod
    def messages_to_prompt(messages: list[ChatCompletionMessageParam]) -> str:
        """
        Convert OpenAI messages format to a simple prompt for native Ollama API.

        Args:
            messages: List of chat completion messages in OpenAI format

        Returns:
            str: Formatted prompt string for Ollama native API
        """
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

    @staticmethod
    def convert_native_response_to_openai(
        native_response: dict, model: str
    ) -> MockResponse:
        """
        Convert native Ollama response to OpenAI-compatible format.

        Args:
            native_response: Response from Ollama native API
            model: Model name used for the request

        Returns:
            MockResponse: OpenAI-compatible response object
        """
        return MockResponse(native_response.get("response", ""), model)

    @staticmethod
    def set_parsed_response(response: MockResponse, parsed_model: BaseModel) -> None:
        """
        Set the parsed field on a mock response for structured output.

        Args:
            response: The mock response to update
            parsed_model: The parsed Pydantic model instance
        """
        if response.choices and response.choices[0].message:
            response.choices[0].message.parsed = parsed_model
