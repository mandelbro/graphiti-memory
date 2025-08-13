"""
Test suite for OllamaClient connection pooling functionality.

This module tests the connection pooling infrastructure added to OllamaClient
to ensure proper HTTP client lifecycle management and connection reuse.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.llm_client.config import LLMConfig
from openai.types.chat import ChatCompletionMessageParam

from src.ollama_client import OllamaClient


class TestOllamaConnectionPooling:
    """Test suite for OllamaClient connection pooling functionality."""

    @pytest.fixture
    def llm_config(self):
        """Create a test LLM configuration."""
        return LLMConfig(
            api_key="test_key",
            model="llama3.2",
            base_url="http://localhost:11434/v1",
            temperature=0.5,
            max_tokens=1000,
        )

    @pytest.fixture
    def model_parameters(self):
        """Create test model parameters."""
        return {
            "num_ctx": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }

    def test_init_connection_pooling_variables(self, llm_config, model_parameters):
        """Test that OllamaClient initializes connection pooling variables."""
        client = OllamaClient(config=llm_config, model_parameters=model_parameters)

        # Check that connection pooling variables are initialized
        assert hasattr(client, "_http_client")
        assert client._http_client is None

    @pytest.mark.asyncio
    async def test_async_context_manager_protocol(self, llm_config):
        """Test that OllamaClient implements async context manager protocol."""
        client = OllamaClient(config=llm_config)

        # Test __aenter__
        context_client = await client.__aenter__()
        assert context_client is client

        # Test __aexit__
        await client.__aexit__(None, None, None)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_get_http_client_creates_new_client(
        self, mock_async_client, llm_config
    ):
        """Test that _get_http_client creates a new HTTP client when none exists."""
        mock_client_instance = AsyncMock()
        mock_client_instance.is_closed = False
        mock_async_client.return_value = mock_client_instance

        client = OllamaClient(config=llm_config)

        # Get HTTP client should create a new one
        http_client = await client._get_http_client()

        assert http_client is mock_client_instance
        assert client._http_client is mock_client_instance

        # Verify the client was created with proper configuration
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args.kwargs

        # Check connection limits
        assert call_kwargs["limits"].max_keepalive_connections == 5
        assert call_kwargs["limits"].max_connections == 10
        assert call_kwargs["limits"].keepalive_expiry == 30.0

        # Check timeouts
        timeout = call_kwargs["timeout"]
        assert timeout.connect == 5.0
        assert timeout.read == 60.0
        assert timeout.write == 5.0
        assert timeout.pool == 2.0

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_get_http_client_reuses_existing_client(
        self, mock_async_client, llm_config
    ):
        """Test that _get_http_client reuses existing HTTP client when available."""
        mock_client_instance = AsyncMock()
        mock_client_instance.is_closed = False
        mock_async_client.return_value = mock_client_instance

        client = OllamaClient(config=llm_config)

        # First call creates client
        http_client1 = await client._get_http_client()

        # Second call should reuse the same client
        http_client2 = await client._get_http_client()

        assert http_client1 is http_client2
        assert http_client1 is mock_client_instance

        # Verify AsyncClient was only called once
        mock_async_client.assert_called_once()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_get_http_client_recreates_closed_client(
        self, mock_async_client, llm_config
    ):
        """Test that _get_http_client recreates HTTP client when existing one is closed."""
        # Second mock client (open)
        mock_client2 = AsyncMock()
        mock_client2.is_closed = False

        # Configure the mock to return the new client
        mock_async_client.return_value = mock_client2

        client = OllamaClient(config=llm_config)

        # First mock client (closed) - simulate existing closed client
        mock_client1 = AsyncMock()
        mock_client1.is_closed = True
        client._http_client = mock_client1

        # Get HTTP client should create a new one since existing is closed
        http_client = await client._get_http_client()

        assert http_client is mock_client2
        assert client._http_client is mock_client2

        # Verify AsyncClient was called to create new client
        mock_async_client.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.ollama_client.OllamaClient._get_http_client")
    async def test_create_completion_uses_shared_client(
        self, mock_get_http_client, llm_config
    ):
        """Test that _create_completion uses the shared HTTP client."""
        # Set up mocks
        mock_http_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "Test response"}
        mock_http_client.post.return_value = mock_response

        mock_get_http_client.return_value = mock_http_client

        client = OllamaClient(config=llm_config)

        # Test messages
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test message"}
        ]

        # Call _create_completion
        await client._create_completion(
            model="llama3.2", messages=messages, temperature=0.7, max_tokens=100
        )

        # Verify that _get_http_client was called
        mock_get_http_client.assert_called_once()

        # Verify that the HTTP client was used for the request
        mock_http_client.post.assert_called_once()

        # Verify the request was made to the correct URL
        call_args = mock_http_client.post.call_args
        assert call_args[0][0].endswith("/api/generate")

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, llm_config):
        """Test that context manager properly cleans up HTTP client."""
        with patch("httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.is_closed = False
            mock_async_client.return_value = mock_client_instance

            client = OllamaClient(config=llm_config)

            # Use as context manager
            async with client:
                # Get HTTP client to initialize it
                await client._get_http_client()
                assert client._http_client is not None

            # After exiting context, HTTP client should be closed
            mock_client_instance.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_handles_none_client(self, llm_config):
        """Test that __aexit__ handles case where HTTP client is None."""
        client = OllamaClient(config=llm_config)

        # Ensure _http_client is None
        assert client._http_client is None

        # __aexit__ should not raise an exception
        await client.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_aexit_handles_already_closed_client(self, llm_config):
        """Test that __aexit__ handles already closed HTTP client gracefully."""
        with patch("httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.is_closed = True
            mock_async_client.return_value = mock_client_instance

            client = OllamaClient(config=llm_config)
            client._http_client = mock_client_instance

            # __aexit__ should not attempt to close already closed client
            await client.__aexit__(None, None, None)

            # aclose should not be called on already closed client
            mock_client_instance.aclose.assert_not_called()
