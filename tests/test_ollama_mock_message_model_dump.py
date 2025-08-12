"""
Test for OllamaClient MockMessage model_dump() compatibility.

This test verifies that MockMessage objects have the model_dump() method
required for compatibility with BaseOpenAIClient response validation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.llm_client.config import LLMConfig
from openai.types.chat import ChatCompletionMessageParam

from src.ollama_client import OllamaClient


class TestOllamaMockMessageModelDump:
    """Test suite for MockMessage model_dump() compatibility."""

    @pytest.fixture
    def llm_config(self):
        """Create a test LLM configuration."""
        return LLMConfig(
            api_key="test_key",
            model="test_model",
            base_url="http://localhost:11434/v1",
            temperature=0.5,
            max_tokens=1000,
        )

    @pytest.fixture
    def ollama_client(self, llm_config):
        """Create an OllamaClient instance for testing."""
        return OllamaClient(config=llm_config)

    @pytest.mark.asyncio
    async def test_mock_message_has_model_dump_method(self, ollama_client):
        """Test that MockMessage objects have model_dump() method."""
        # Arrange
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test message"}
        ]

        # Mock the httpx response
        mock_response_data = {"response": "Test response from Ollama"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()  # Use MagicMock for response, not AsyncMock
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()  # Synchronous method
            mock_response.json.return_value = mock_response_data
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Act - Create completion which should return MockResponse with MockMessage
            result = await ollama_client._create_completion(
                model="test_model", messages=messages, temperature=0.7, max_tokens=100
            )

            # Assert - Verify MockMessage has model_dump() method
            mock_message = result.choices[0].message
            assert hasattr(mock_message, "model_dump"), (
                "MockMessage should have model_dump() method"
            )

            # Test that model_dump() returns a dictionary
            model_dump_result = mock_message.model_dump()
            assert isinstance(model_dump_result, dict), (
                "model_dump() should return a dictionary"
            )

            # Verify required fields are in the model_dump output
            expected_fields = ["content", "role", "parsed", "refusal"]
            for field in expected_fields:
                assert field in model_dump_result, (
                    f"model_dump() should include '{field}' field"
                )

            # Verify field values
            assert model_dump_result["content"] == "Test response from Ollama"
            assert model_dump_result["role"] == "assistant"
            assert model_dump_result["parsed"] is None
            assert model_dump_result["refusal"] is None

    @pytest.mark.asyncio
    async def test_mock_message_model_dump_compatibility_fields(self, ollama_client):
        """Test that MockMessage model_dump() includes all OpenAI compatibility fields."""
        # Arrange
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test compatibility"}
        ]

        mock_response_data = {"response": "Compatibility test response"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()  # Use MagicMock for response, not AsyncMock
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()  # Synchronous method
            mock_response.json.return_value = mock_response_data
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Act
            result = await ollama_client._create_completion(
                model="test_model", messages=messages, temperature=0.7, max_tokens=100
            )

            # Assert - Test all expected OpenAI compatibility fields
            mock_message = result.choices[0].message
            model_dump_result = mock_message.model_dump()

            # All OpenAI ChatCompletionMessage fields that should be present
            expected_openai_fields = [
                "content",
                "role",
                "parsed",
                "refusal",
                "annotations",
                "audio",
                "function_call",
                "tool_calls",
            ]

            for field in expected_openai_fields:
                assert field in model_dump_result, (
                    f"Missing OpenAI compatibility field: {field}"
                )

            # Verify specific values for key fields
            assert model_dump_result["content"] == "Compatibility test response"
            assert model_dump_result["role"] == "assistant"

            # Verify optional fields are None (as expected for Ollama)
            optional_fields = [
                "parsed",
                "refusal",
                "annotations",
                "audio",
                "function_call",
                "tool_calls",
            ]
            for field in optional_fields:
                assert model_dump_result[field] is None, (
                    f"Optional field '{field}' should be None"
                )
