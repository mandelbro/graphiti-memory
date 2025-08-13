"""
Tests for Phase 2: Enhanced Structured Response Handling in OllamaClient.

This module tests the enhanced structured completion functionality that was
implemented in Phase 2 of the Ollama OpenAI compatibility fix.
"""

from unittest.mock import MagicMock, patch

import pytest
from graphiti_core.llm_client.config import LLMConfig
from pydantic import BaseModel, Field

from src.ollama_client import OllamaClient


class EntityTestModel(BaseModel):
    """Test entity for structured response validation."""

    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    confidence: float = Field(..., description="Confidence score")


class TestOllamaPhase2StructuredResponses:
    """Test suite for Phase 2 enhanced structured response handling."""

    @pytest.fixture
    def ollama_config(self):
        """Create test configuration for Ollama."""
        return LLMConfig(
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="test-key",
            temperature=0.1,
        )

    @pytest.fixture
    def ollama_client(self, ollama_config):
        """Create OllamaClient instance for testing."""
        return OllamaClient(
            config=ollama_config, model_parameters={"num_ctx": 4096, "top_p": 0.9}
        )

    @pytest.mark.asyncio
    async def test_structured_completion_with_valid_json(self, ollama_client):
        """Test that valid JSON responses are properly parsed and structured."""
        # Mock the native API call to return valid JSON
        json_response = '{"name": "John Doe", "type": "person", "confidence": 0.95}'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": json_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify the response structure
            assert result is not None
            assert result.choices[0].message.content == json_response

            # Verify that parsed field is populated with the correct model
            parsed = result.choices[0].message.parsed
            assert parsed is not None
            assert isinstance(parsed, EntityTestModel)
            assert parsed.name == "John Doe"
            assert parsed.type == "person"
            assert parsed.confidence == 0.95

    @pytest.mark.asyncio
    async def test_structured_completion_with_invalid_json(self, ollama_client):
        """Test graceful handling of invalid JSON responses."""
        # Mock the native API call to return invalid JSON
        invalid_json = 'This is not valid JSON: {"name": "incomplete'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": invalid_json}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify that parsing failed gracefully
            assert result is not None
            assert result.choices[0].message.content == invalid_json
            assert result.choices[0].message.parsed is None

    @pytest.mark.asyncio
    async def test_structured_completion_with_validation_error(self, ollama_client):
        """Test handling of JSON that doesn't match the response model."""
        # Mock the native API call to return JSON with wrong structure
        invalid_structure = '{"wrong_field": "value", "another_field": 123}'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": invalid_structure}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify that validation failed gracefully
            assert result is not None
            assert result.choices[0].message.content == invalid_structure
            assert result.choices[0].message.parsed is None

    @pytest.mark.asyncio
    async def test_structured_completion_with_non_json_content(self, ollama_client):
        """Test handling of non-JSON responses."""
        # Mock the native API call to return regular text
        text_response = "This is a regular text response without JSON structure."

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": text_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify that non-JSON content is handled gracefully
            assert result is not None
            assert result.choices[0].message.content == text_response
            assert result.choices[0].message.parsed is None

    @pytest.mark.asyncio
    async def test_mock_message_model_dump_with_parsed_data(self, ollama_client):
        """Test that MockMessage properly implements model_dump() with parsed data."""
        # Mock the native API call to return structured JSON data
        json_response = '{"name": "Test Entity", "type": "test", "confidence": 0.8}'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": json_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Test model_dump method with parsed data
            dump = result.choices[0].message.model_dump()

            assert isinstance(dump, dict)
            assert dump["content"] == json_response
            assert dump["role"] == "assistant"
            assert dump["parsed"] is not None
            assert isinstance(dump["parsed"], EntityTestModel)
            assert dump["refusal"] is None

    @pytest.mark.asyncio
    async def test_logging_during_structured_parsing(self, ollama_client, caplog):
        """Test that appropriate log messages are generated during parsing."""
        import logging

        caplog.set_level(logging.WARNING)

        # Mock the native API call to return invalid JSON
        invalid_json = '{"invalid": json'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": invalid_json}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify warning was logged
            assert (
                "Failed to parse JSON response from Ollama model llama3.1:8b"
                in caplog.text
            )

    @pytest.mark.asyncio
    async def test_structured_completion_preserves_original_functionality(
        self, ollama_client
    ):
        """Test that the enhanced method still works when JSON parsing is not applicable."""
        # Mock the native API call to return a standard text response
        text_response = "Standard text response without JSON"

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": text_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Generate text"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify original functionality is preserved
            assert result is not None
            assert result.choices[0].message.content == text_response
            assert result.choices[0].message.parsed is None

    def test_mock_message_type_annotations(self):
        """Test that MockMessage has proper type annotations for parsed field."""
        from src.ollama_client import OllamaClient

        # This test verifies that the MockMessage class properly accepts BaseModel instances
        # in the parsed field, which was the main issue fixed in the type annotations
        client = OllamaClient()

        # Create a mock response to access MockMessage class
        mock_response = client._response_converter.convert_native_response_to_openai(
            {"response": "test"}, "test-model"
        )
        message = mock_response.choices[0].message

        # Test that we can assign a BaseModel instance to parsed
        test_entity = EntityTestModel(name="test", type="test", confidence=0.5)
        message.parsed = test_entity

        # Verify the assignment worked
        assert message.parsed == test_entity
        assert isinstance(message.parsed, EntityTestModel)
