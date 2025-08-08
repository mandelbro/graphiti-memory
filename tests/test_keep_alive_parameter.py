"""
Specific tests for the keep_alive parameter functionality in OllamaClient.
Tests the new functionality that allows model_parameters to include keep_alive
which controls how long Ollama keeps models loaded in memory.
"""

import pytest
from unittest.mock import patch, AsyncMock
from src.ollama_client import OllamaClient
from graphiti_core.llm_client.config import LLMConfig


class TestKeepAliveParameter:
    """Test suite for keep_alive parameter functionality in OllamaClient."""

    @pytest.fixture
    def llm_config(self):
        """Create a test LLM configuration."""
        return LLMConfig(
            api_key="test_key",
            model="test_model",
            base_url="http://localhost:11434/v1",
            temperature=0.5,
            max_tokens=1000
        )

    @pytest.mark.asyncio
    async def test_keep_alive_parameter_flows_through_completion(self, llm_config):
        """Integration test that keep_alive parameter works in completion method."""
        model_parameters = {
            "num_ctx": 8192,
            "keep_alive": "10m"
        }

        client = OllamaClient(
            config=llm_config,
            model_parameters=model_parameters
        )

        with patch('httpx.AsyncClient') as mock_client_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            mock_response.json.return_value = {"response": "Integration test response"}
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Test completion
            messages = [{"role": "user", "content": "Test message"}]
            await client._create_completion(
                model="test-model",
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )

            # Verify keep_alive is included
            assert mock_client.post.called
            payload = mock_client.post.call_args[1]['json']
            assert 'keep_alive' in payload
            assert payload['keep_alive'] == "10m"

    @pytest.mark.asyncio
    async def test_keep_alive_parameter_with_config_integration(self, llm_config):
        """Test keep_alive parameter integration with real configuration patterns."""
        # Simulate configuration that might come from YAML
        model_parameters = {
            "num_ctx": 15000,
            "num_predict": -1,
            "repeat_penalty": 1.1,
            "top_k": 50,
            "top_p": 0.9,
            "keep_alive": "15m"  # Keep model loaded for 15 minutes
        }

        client = OllamaClient(
            config=llm_config,
            model_parameters=model_parameters
        )

        with patch('httpx.AsyncClient') as mock_client_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            mock_response.json.return_value = {"response": "Config integration response"}
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Test completion with all parameters
            messages = [{"role": "user", "content": "Test with full config"}]
            await client._create_completion(
                model="gpt-oss:latest",
                messages=messages,
                temperature=0.1,
                max_tokens=512
            )

            # Verify all parameters are included
            payload = mock_client.post.call_args[1]['json']

            # Verify keep_alive is at top level
            assert 'keep_alive' in payload
            assert payload['keep_alive'] == "15m"

            # Verify all model parameters are in options
            options = payload['options']
            assert options['num_ctx'] == 15000
            assert options['num_predict'] == -1
            assert options['repeat_penalty'] == 1.1
            assert options['top_k'] == 50
            assert options['top_p'] == 0.9

            # Verify temperature and max_tokens overrides
            assert options['temperature'] == 0.1
            assert options['num_predict'] == 512  # max_tokens override

    def test_keep_alive_parameter_documentation_example(self, llm_config):
        """Test keep_alive parameter with values commonly used in documentation."""
        # Common keep_alive values from Ollama documentation
        test_values = [
            "5m",      # 5 minutes
            "30s",     # 30 seconds
            "2h",      # 2 hours
            "0",       # Unload immediately
            "-1",      # Keep forever
            300,       # 5 minutes in seconds
            0,         # Unload immediately (integer)
        ]

        for keep_alive_value in test_values:
            model_parameters = {"keep_alive": keep_alive_value}
            client = OllamaClient(
                config=llm_config,
                model_parameters=model_parameters
            )

            # Verify the parameter is stored correctly
            assert client.model_parameters["keep_alive"] == keep_alive_value

    @pytest.mark.asyncio
    async def test_keep_alive_parameter_priority_over_defaults(self, llm_config):
        """Test that keep_alive from model_parameters takes priority."""
        model_parameters = {
            "num_ctx": 4096,
            "keep_alive": "custom_value"
        }

        client = OllamaClient(
            config=llm_config,
            model_parameters=model_parameters
        )

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            mock_response.json.return_value = {"response": "Priority test response"}
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Test completion
            messages = [{"role": "user", "content": "Priority test"}]
            await client._create_completion(
                model="test-model",
                messages=messages,
                temperature=0.5,
                max_tokens=100
            )

            # Verify our custom keep_alive value is used
            payload = mock_client.post.call_args[1]['json']
            assert payload['keep_alive'] == "custom_value"
