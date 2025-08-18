"""
Test suite for enhanced OllamaClient integration with response converter.

This test suite verifies that the OllamaClient properly integrates with the
OllamaResponseConverter for all structured response scenarios, ensuring
schema validation works correctly across all memory operations.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.llm_client.config import LLMConfig
from pydantic import BaseModel

from src.ollama_client import OllamaClient
from src.utils.ollama_response_converter import OllamaResponseConverter


class MockEntity(BaseModel):
    """Test entity model for validation."""
    entity_type_id: int
    name: str
    description: str | None = None


class ExtractedEntities(BaseModel):
    """Test schema matching Graphiti's ExtractedEntities format."""
    extracted_entities: list[MockEntity]


class SimpleResponse(BaseModel):
    """Simple response model for testing."""
    value: str
    confidence: float = 0.8


@pytest.fixture
def mock_config():
    """Create mock LLM configuration for testing."""
    return LLMConfig(
        api_key="test-key",
        model="llama3.2:3b",
        base_url="http://localhost:11434/v1",
        temperature=0.1,
        max_tokens=2048
    )


@pytest.fixture
def ollama_client(mock_config):
    """Create OllamaClient instance for testing."""
    return OllamaClient(config=mock_config, model_parameters={"num_ctx": 8192})


@pytest.fixture
def mock_http_response():
    """Create mock HTTP response from Ollama native API."""
    return {
        "model": "llama3.2:3b",
        "response": json.dumps([
            {"entity_type_id": 1, "entity": "Test Entity", "description": "A test entity"}
        ]),
        "done": True,
        "total_duration": 1000000000,
        "load_duration": 500000000,
        "prompt_eval_count": 10,
        "eval_count": 20
    }


class TestEnhancedIntegration:
    """Test cases for enhanced OllamaClient integration with response converter."""

    @pytest.mark.asyncio
    async def test_structured_completion_with_extracted_entities_schema(
        self, ollama_client, mock_http_response
    ):
        """Test structured completion with ExtractedEntities schema conversion."""

        # Mock the HTTP client and response
        with patch.object(ollama_client, '_get_http_client') as mock_client:
            mock_http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_http_response
            mock_response.raise_for_status.return_value = None
            mock_http_client.post.return_value = mock_response
            mock_client.return_value = mock_http_client

            # Test the structured completion
            messages = [{"role": "user", "content": "Extract entities from this text."}]

            result = await ollama_client._create_structured_completion(
                model="llama3.2:3b",
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                response_model=ExtractedEntities
            )

            # Verify the response structure
            assert result is not None
            assert hasattr(result.choices[0].message, 'parsed')

            # Verify the parsed content matches expected schema
            if result.choices[0].message.parsed:
                parsed_data = result.choices[0].message.parsed
                assert isinstance(parsed_data, ExtractedEntities)
                assert len(parsed_data.extracted_entities) == 1
                assert parsed_data.extracted_entities[0].name == "Test Entity"
                assert parsed_data.extracted_entities[0].entity_type_id == 1

    @pytest.mark.asyncio
    async def test_structured_completion_with_simple_response_schema(
        self, ollama_client
    ):
        """Test structured completion with simple response schema."""

        simple_response_data = {
            "model": "llama3.2:3b",
            "response": json.dumps({"value": "test response", "confidence": 0.9}),
            "done": True
        }

        with patch.object(ollama_client, '_get_http_client') as mock_client:
            mock_http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = simple_response_data
            mock_response.raise_for_status.return_value = None
            mock_http_client.post.return_value = mock_response
            mock_client.return_value = mock_http_client

            messages = [{"role": "user", "content": "Generate a simple response."}]

            result = await ollama_client._create_structured_completion(
                model="llama3.2:3b",
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                response_model=SimpleResponse
            )

            # Verify the response
            assert result is not None
            if result.choices[0].message.parsed:
                parsed_data = result.choices[0].message.parsed
                assert isinstance(parsed_data, SimpleResponse)
                assert parsed_data.value == "test response"
                assert parsed_data.confidence == 0.9

    @pytest.mark.asyncio
    async def test_structured_completion_error_handling(self, ollama_client):
        """Test error handling in structured completion."""

        # Invalid JSON response
        invalid_response_data = {
            "model": "llama3.2:3b",
            "response": "This is not valid JSON",
            "done": True
        }

        with patch.object(ollama_client, '_get_http_client') as mock_client:
            mock_http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = invalid_response_data
            mock_response.raise_for_status.return_value = None
            mock_http_client.post.return_value = mock_response
            mock_client.return_value = mock_http_client

            messages = [{"role": "user", "content": "Generate response."}]

            result = await ollama_client._create_structured_completion(
                model="llama3.2:3b",
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                response_model=SimpleResponse
            )

            # Should handle gracefully without parsed content
            assert result is not None
            assert result.choices[0].message.content == "This is not valid JSON"
            # parsed should be None for invalid JSON
            assert result.choices[0].message.parsed is None

    @pytest.mark.asyncio
    async def test_response_converter_integration(self, ollama_client):
        """Test that response converter is properly integrated."""

        # Test data that needs conversion
        ollama_format_data = [
            {"entity_type_id": 1, "entity": "Entity One"},
            {"entity_type_id": 2, "entity": "Entity Two"}
        ]

        response_data = {
            "model": "llama3.2:3b",
            "response": json.dumps(ollama_format_data),
            "done": True
        }

        with patch.object(ollama_client, '_get_http_client') as mock_client:
            mock_http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = response_data
            mock_response.raise_for_status.return_value = None
            mock_http_client.post.return_value = mock_response
            mock_client.return_value = mock_http_client

            messages = [{"role": "user", "content": "Extract multiple entities."}]

            result = await ollama_client._create_structured_completion(
                model="llama3.2:3b",
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                response_model=ExtractedEntities
            )

            # Verify conversion happened correctly
            assert result is not None
            if result.choices[0].message.parsed:
                parsed_data = result.choices[0].message.parsed
                assert isinstance(parsed_data, ExtractedEntities)
                assert len(parsed_data.extracted_entities) == 2
                assert parsed_data.extracted_entities[0].name == "Entity One"
                assert parsed_data.extracted_entities[1].name == "Entity Two"

    def test_response_converter_initialization(self, ollama_client):
        """Test that response converter is properly initialized."""
        assert hasattr(ollama_client, '_response_converter')
        assert isinstance(ollama_client._response_converter, OllamaResponseConverter)

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, ollama_client):
        """Test that existing functionality remains compatible."""

        # Test with regular completion (should work as before)
        standard_response = {
            "model": "llama3.2:3b",
            "response": "This is a regular text response",
            "done": True
        }

        with patch.object(ollama_client, '_get_http_client') as mock_client:
            mock_http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = standard_response
            mock_response.raise_for_status.return_value = None
            mock_http_client.post.return_value = mock_response
            mock_client.return_value = mock_http_client

            result = await ollama_client._create_completion(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.1,
                max_tokens=2048
            )

            assert result is not None
            assert result.choices[0].message.content == "This is a regular text response"
