"""
Tests for Ollama server health check functionality.

This module tests the health check system with intelligent caching
to validate Ollama server availability before making requests.
"""

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from graphiti_core.llm_client.config import LLMConfig

from src.ollama_client import OllamaClient


class TestOllamaHealthCheck:
    """Test suite for Ollama health check functionality."""

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
    async def test_health_check_cache_initialization(self, ollama_client):
        """Test that health check cache is properly initialized."""
        assert hasattr(ollama_client._health_validator, "_health_check_cache")
        assert isinstance(ollama_client._health_validator._health_check_cache, dict)
        assert len(ollama_client._health_validator._health_check_cache) == 0

    @pytest.mark.asyncio
    async def test_health_check_successful(self, ollama_client):
        """Test successful health check response."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            is_healthy, message = await ollama_client.check_health()

            assert is_healthy is True
            assert "healthy and accessible" in message
            assert "http://localhost:11434/v1" in message

            # Verify the call was made correctly
            mock_client.get.assert_called_once_with(
                "http://localhost:11434/api/tags", timeout=1.0
            )

    @pytest.mark.asyncio
    async def test_health_check_connect_error(self, ollama_client):
        """Test health check with connection error."""
        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")
            mock_get_client.return_value = mock_client

            is_healthy, message = await ollama_client.check_health()

            assert is_healthy is False
            assert "Cannot connect to Ollama server" in message
            assert "Is Ollama running?" in message

    @pytest.mark.asyncio
    async def test_health_check_timeout_error(self, ollama_client):
        """Test health check with timeout error."""
        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")
            mock_get_client.return_value = mock_client

            is_healthy, message = await ollama_client.check_health()

            assert is_healthy is False
            assert "is not responding" in message
            assert "Server may be overloaded" in message

    @pytest.mark.asyncio
    async def test_health_check_generic_error(self, ollama_client):
        """Test health check with generic error."""
        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Generic error")
            mock_get_client.return_value = mock_client

            is_healthy, message = await ollama_client.check_health()

            assert is_healthy is False
            assert "Ollama health check failed" in message
            assert "Generic error" in message

    @pytest.mark.asyncio
    async def test_health_check_caching_behavior(self, ollama_client):
        """Test that health check results are cached properly."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # First call
            is_healthy1, message1 = await ollama_client.check_health()

            # Second call (should use cache)
            is_healthy2, message2 = await ollama_client.check_health()

            # Results should be identical
            assert is_healthy1 is True and is_healthy2 is True
            assert message1 == message2

            # HTTP client should only be called once (cached second time)
            assert mock_client.get.call_count == 1

            # Verify cache contains the result
            cache_key = f"health_{ollama_client.ollama_base_url}"
            assert cache_key in ollama_client._health_validator._health_check_cache

    @pytest.mark.asyncio
    async def test_health_check_cache_expiration(self, ollama_client):
        """Test that health check cache expires after 5 minutes."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # First call
            await ollama_client.check_health()

            # Manually expire the cache by setting an old timestamp
            cache_key = f"health_{ollama_client.ollama_base_url}"
            if cache_key in ollama_client._health_validator._health_check_cache:
                result, _ = ollama_client._health_validator._health_check_cache[
                    cache_key
                ]
                ollama_client._health_validator._health_check_cache[cache_key] = (
                    result,
                    time.time() - 400,
                )  # 400 seconds ago

            # Second call (should make new request due to expired cache)
            await ollama_client.check_health()

            # HTTP client should be called twice
            assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_health_check_cache_negative_results(self, ollama_client):
        """Test that negative health check results are also cached."""
        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")
            mock_get_client.return_value = mock_client

            # First call (will fail and be cached)
            is_healthy1, message1 = await ollama_client.check_health()

            # Second call (should use cached failure)
            is_healthy2, message2 = await ollama_client.check_health()

            # Results should be identical failures
            assert is_healthy1 is False and is_healthy2 is False
            assert message1 == message2

            # HTTP client should only be called once (cached second time)
            assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_health_check_url_construction(self, ollama_client):
        """Test that health check URL is constructed correctly."""
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            await ollama_client.check_health()

            # Verify the URL construction removes /v1 and adds /api/tags
            expected_url = "http://localhost:11434/api/tags"
            mock_client.get.assert_called_once_with(expected_url, timeout=1.0)

    @pytest.mark.asyncio
    async def test_health_check_performance_timeout(self, ollama_client):
        """Test that health check completes quickly (< 1 second timeout)."""
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None

        with patch.object(
            ollama_client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            start_time = time.time()
            await ollama_client.check_health()
            elapsed_time = time.time() - start_time

            # Should complete very quickly (under 0.1 seconds in mocked scenario)
            assert elapsed_time < 0.1

            # Verify timeout parameter is set to 1.0 second
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert call_args.kwargs.get("timeout") == 1.0

    @pytest.mark.asyncio
    async def test_health_check_with_custom_base_url(self):
        """Test health check with custom base URL configuration."""
        custom_config = LLMConfig(
            api_key="test_key",
            model="test_model",
            base_url="http://custom-ollama:8080/v1",
        )

        client = OllamaClient(config=custom_config)
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None

        with patch.object(
            client._health_validator, "_get_http_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            is_healthy, message = await client.check_health()

            assert is_healthy is True
            assert "http://custom-ollama:8080/v1" in message

            # Verify URL construction for custom base URL
            expected_url = "http://custom-ollama:8080/api/tags"
            mock_client.get.assert_called_once_with(expected_url, timeout=1.0)

        await client.__aexit__(None, None, None)
