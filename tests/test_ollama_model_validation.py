"""
Tests for Ollama model availability validation functionality.

This module tests the model validation system that checks if requested models
are available on the Ollama server and provides helpful error messages.
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from graphiti_core.llm_client.config import LLMConfig

from src.ollama_client import OllamaClient


class TestOllamaModelValidation:
    """Test suite for Ollama model availability validation functionality."""

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
    def model_parameters(self):
        """Create test model parameters."""
        return {
            "num_ctx": 4096,
            "num_predict": 100,
            "top_p": 0.9,
            "temperature": 0.7,
        }

    @pytest.fixture
    async def ollama_client(self, llm_config, model_parameters):
        """Create an OllamaClient instance for testing."""
        client = OllamaClient(
            config=llm_config,
            model_parameters=model_parameters,
        )
        yield client
        # Cleanup
        await client.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_model_cache_initialization(self, ollama_client):
        """Test that model cache is properly initialized."""
        assert hasattr(ollama_client._health_validator, "_model_cache")
        assert isinstance(ollama_client._health_validator._model_cache, dict)
        assert len(ollama_client._health_validator._model_cache) == 0

    @pytest.mark.asyncio
    async def test_validate_model_available_success(self, ollama_client):
        """Test successful model validation when model is available."""
        # Mock successful response with available models
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "codellama:7b"},
                {"name": "test_model"},
            ]
        }

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            is_available, message = await ollama_client.validate_model_available(
                "test_model"
            )

            assert is_available is True
            assert message == "Model available"

            # Verify the call was made correctly
            mock_client.get.assert_called_once_with(
                "http://localhost:11434/api/tags", timeout=5.0
            )

    @pytest.mark.asyncio
    async def test_validate_model_unavailable(self, ollama_client):
        """Test model validation when model is not available."""
        # Mock response with different available models
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "codellama:7b"},
                {"name": "mistral:7b"},
                {"name": "gemma:2b"},
                {"name": "qwen2:7b"},
                {"name": "phi3:mini"},  # More than 5 models to test truncation
            ]
        }

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            is_available, message = await ollama_client.validate_model_available(
                "nonexistent_model"
            )

            assert is_available is False
            assert "Model 'nonexistent_model' not found" in message
            assert "Available models:" in message
            # Should show first 5 models
            assert "llama3.1:8b" in message
            assert "codellama:7b" in message
            assert "mistral:7b" in message
            assert "gemma:2b" in message
            assert "qwen2:7b" in message
            # Should NOT show the 6th model (phi3:mini)
            assert "phi3:mini" not in message

    @pytest.mark.asyncio
    async def test_validate_model_caching_positive(self, ollama_client):
        """Test that positive model validation results are cached."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"models": [{"name": "test_model"}]}

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # First call
            is_available1, message1 = await ollama_client.validate_model_available(
                "test_model"
            )

            # Second call (should use cache)
            is_available2, message2 = await ollama_client.validate_model_available(
                "test_model"
            )

            # Results should be identical
            assert is_available1 is True and is_available2 is True
            assert message1 == message2 == "Model available"

            # HTTP client should only be called once (cached second time)
            assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_validate_model_caching_negative(self, ollama_client):
        """Test that negative model validation results are cached."""
        # Mock response with different models
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"models": [{"name": "different_model"}]}

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # First call
            is_available1, message1 = await ollama_client.validate_model_available(
                "nonexistent"
            )

            # Second call (should use cache)
            is_available2, message2 = await ollama_client.validate_model_available(
                "nonexistent"
            )

            # Results should be identical
            assert is_available1 is False and is_available2 is False
            assert message1 == message2
            assert "not found" in message1

            # HTTP client should only be called once (cached second time)
            assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_validate_model_connection_error(self, ollama_client):
        """Test model validation with connection error (graceful fallback)."""
        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")
            mock_get_client.return_value = mock_client

            is_available, message = await ollama_client.validate_model_available(
                "test_model"
            )

            # Should fallback gracefully - allow request to proceed
            assert is_available is True
            assert "Model validation skipped due to error" in message
            assert "Connection failed" in message

    @pytest.mark.asyncio
    async def test_validate_model_timeout_error(self, ollama_client):
        """Test model validation with timeout error (graceful fallback)."""
        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")
            mock_get_client.return_value = mock_client

            is_available, message = await ollama_client.validate_model_available(
                "test_model"
            )

            # Should fallback gracefully - allow request to proceed
            assert is_available is True
            assert "Model validation skipped due to error" in message
            assert "Timeout" in message

    @pytest.mark.asyncio
    async def test_validate_model_http_error(self, ollama_client):
        """Test model validation with HTTP error (graceful fallback)."""
        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "HTTP Error", request=AsyncMock(), response=AsyncMock()
            )
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            is_available, message = await ollama_client.validate_model_available(
                "test_model"
            )

            # Should fallback gracefully - allow request to proceed
            assert is_available is True
            assert "Model validation skipped due to error" in message

    @pytest.mark.asyncio
    async def test_validate_model_malformed_json(self, ollama_client):
        """Test model validation with malformed JSON response (graceful fallback)."""
        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            is_available, message = await ollama_client.validate_model_available(
                "test_model"
            )

            # Should fallback gracefully - allow request to proceed
            assert is_available is True
            assert "Model validation skipped due to error" in message
            assert "Invalid JSON" in message

    @pytest.mark.asyncio
    async def test_validate_model_missing_models_field(self, ollama_client):
        """Test model validation with missing 'models' field in response."""
        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {}  # Missing 'models' field
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            is_available, message = await ollama_client.validate_model_available(
                "test_model"
            )

            # Should fallback gracefully - allow request to proceed
            assert is_available is True
            assert "Model validation skipped due to error" in message

    @pytest.mark.asyncio
    async def test_validate_model_empty_models_list(self, ollama_client):
        """Test model validation with empty models list."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"models": []}

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            is_available, message = await ollama_client.validate_model_available(
                "test_model"
            )

            assert is_available is False
            assert "Model 'test_model' not found" in message
            assert "Available models: (none)" in message

    @pytest.mark.asyncio
    async def test_validate_model_case_sensitivity(self, ollama_client):
        """Test that model validation is case sensitive."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "models": [{"name": "Llama3.1:8b"}]  # Capital L
        }

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Test lowercase version should not match
            is_available, message = await ollama_client.validate_model_available(
                "llama3.1:8b"
            )

            assert is_available is False
            assert "Model 'llama3.1:8b' not found" in message
            assert "Llama3.1:8b" in message

    @pytest.mark.asyncio
    async def test_validate_different_models_separate_cache(self, ollama_client):
        """Test that different models have separate cache entries."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"models": [{"name": "model1"}]}

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Validate model1 (should be available)
            is_available1, _ = await ollama_client.validate_model_available("model1")

            # Validate model2 (should not be available) - should make another API call
            is_available2, _ = await ollama_client.validate_model_available("model2")

            assert is_available1 is True
            assert is_available2 is False

            # Should make 2 API calls since they're different models
            assert mock_client.get.call_count == 2
