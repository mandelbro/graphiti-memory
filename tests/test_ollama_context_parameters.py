"""
Specific tests for Ollama context parameter functionality.
This module tests the fix for ensuring num_ctx and other parameters are correctly sent to Ollama.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
import yaml
import httpx

from src.config_loader import ConfigLoader
from src.graphiti_mcp_server import GraphitiLLMConfig
from src.ollama_client import OllamaClient
from graphiti_core.llm_client.config import LLMConfig

# Create a config_loader instance for the tests
config_loader = ConfigLoader()


class TestOllamaContextParameters:
    """Test the specific functionality for handling Ollama context parameters."""

    @pytest.fixture
    def sample_context_config(self):
        """Sample configuration with context parameters."""
        return {
            'llm': {
                'model': 'gpt-oss:latest',
                'base_url': 'http://localhost:11434/v1',
                'temperature': 0.1,
                'max_tokens': 32768,
                'model_parameters': {
                    'num_ctx': 12000,
                    'num_predict': -1,
                    'repeat_penalty': 1.1,
                    'top_k': 50,
                    'top_p': 0.9,
                    'num_threads': 40
                }
            }
        }

    def test_context_parameter_loading_from_yaml(self, sample_context_config):
        """Test that context parameters are correctly loaded from YAML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            providers_dir = config_dir / 'providers'
            providers_dir.mkdir()

            config_file = providers_dir / 'ollama.yml'
            with open(config_file, 'w') as f:
                yaml.dump(sample_context_config, f)

            # Temporarily override the config loader's config directory
            original_config_dir = config_loader.config_dir
            config_loader.config_dir = config_dir

            try:
                os.environ['USE_OLLAMA'] = 'true'

                # Load configuration
                llm_config = GraphitiLLMConfig.from_yaml_and_env()

                # Verify all parameters are loaded
                assert llm_config.ollama_model_parameters['num_ctx'] == 12000
                assert llm_config.ollama_model_parameters['num_threads'] == 40
                assert llm_config.ollama_model_parameters['num_predict'] == -1
                assert llm_config.ollama_model_parameters['repeat_penalty'] == 1.1
                assert llm_config.ollama_model_parameters['top_k'] == 50
                assert llm_config.ollama_model_parameters['top_p'] == 0.9

            finally:
                if 'USE_OLLAMA' in os.environ:
                    del os.environ['USE_OLLAMA']
                config_loader.config_dir = original_config_dir

    def test_ollama_client_receives_context_parameters(self, sample_context_config):
        """Test that OllamaClient receives the context parameters correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            providers_dir = config_dir / 'providers'
            providers_dir.mkdir()

            config_file = providers_dir / 'ollama.yml'
            with open(config_file, 'w') as f:
                yaml.dump(sample_context_config, f)

            original_config_dir = config_loader.config_dir
            config_loader.config_dir = config_dir

            try:
                os.environ['USE_OLLAMA'] = 'true'

                llm_config = GraphitiLLMConfig.from_yaml_and_env()
                client = llm_config.create_client()

                # Verify client setup
                assert client.__class__.__name__ == 'OllamaClient'
                assert hasattr(client, 'model_parameters')
                assert len(client.model_parameters) == 6  # All 6 parameters

                # Verify specific context parameters
                assert client.model_parameters['num_ctx'] == 12000
                assert client.model_parameters['num_threads'] == 40

            finally:
                if 'USE_OLLAMA' in os.environ:
                    del os.environ['USE_OLLAMA']
                config_loader.config_dir = original_config_dir

    @pytest.mark.asyncio
    async def test_native_api_call_with_context_parameters(self, sample_context_config):
        """Test that the native API call includes context parameters correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            providers_dir = config_dir / 'providers'
            providers_dir.mkdir()

            config_file = providers_dir / 'ollama.yml'
            with open(config_file, 'w') as f:
                yaml.dump(sample_context_config, f)

            original_config_dir = config_loader.config_dir
            config_loader.config_dir = config_dir

            try:
                os.environ['USE_OLLAMA'] = 'true'

                llm_config = GraphitiLLMConfig.from_yaml_and_env()
                client = llm_config.create_client()

                # Mock the native API call
                with patch('httpx.AsyncClient') as mock_client_class:
                    mock_client = AsyncMock()
                    mock_response = AsyncMock()
                    mock_response.status_code = 200
                    mock_response.raise_for_status = AsyncMock()
                    mock_response.json.return_value = {
                        "model": "gpt-oss:latest",
                        "response": "Test response with correct context",
                        "done": True
                    }
                    mock_client.post.return_value = mock_response
                    mock_client.__aenter__.return_value = mock_client
                    mock_client.__aexit__.return_value = None
                    mock_client_class.return_value = mock_client

                    # Trigger the native API call
                    result = await client._create_completion(
                        "gpt-oss:latest",
                        [{"role": "user", "content": "Test prompt"}],
                        0.1,
                        100
                    )

                    # Verify the native API call was made
                    mock_client.post.assert_called_once()
                    call_args = mock_client.post.call_args

                    # Verify URL
                    assert call_args[0][0] == "http://localhost:11434/api/generate"

                    # Verify payload structure
                    payload = call_args[1]['json']
                    assert payload['model'] == "gpt-oss:latest"
                    assert payload['prompt'] == "User: Test prompt"
                    assert payload['stream'] is False

                    # Verify all context parameters are in options
                    options = payload['options']
                    assert options['num_ctx'] == 12000
                    assert options['num_threads'] == 40
                    assert options['num_predict'] == -1
                    assert options['repeat_penalty'] == 1.1
                    assert options['top_k'] == 50
                    assert options['top_p'] == 0.9

                    # Verify response
                    assert result.choices[0].message.content == "Test response with correct context"

            finally:
                if 'USE_OLLAMA' in os.environ:
                    del os.environ['USE_OLLAMA']
                config_loader.config_dir = original_config_dir

    def test_url_conversion_edge_cases(self):
        """Test URL conversion from OpenAI-compatible to native format."""
        test_cases = [
            ("http://localhost:11434/v1", "http://localhost:11434"),
            ("http://localhost:11434/v1/", "http://localhost:11434"),
            ("http://localhost:11434", "http://localhost:11434"),
            ("https://my-ollama-server.com/v1", "https://my-ollama-server.com"),
            ("https://my-ollama-server.com:8080/v1", "https://my-ollama-server.com:8080"),
        ]

        for input_url, expected_native_url in test_cases:
            llm_config = LLMConfig(
                api_key="test_key",
                model="test_model",
                base_url=input_url
            )

            client = OllamaClient(config=llm_config, model_parameters={"num_ctx": 4096})

            # Test URL conversion
            expected_api_url = f"{expected_native_url}/api/generate"

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_response.raise_for_status = AsyncMock()
                mock_response.json.return_value = {"model": "test", "response": "test", "done": True}
                mock_client.post.return_value = mock_response
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client_class.return_value = mock_client

                async def run_test():
                    await client._create_completion(
                        "test_model",
                        [{"role": "user", "content": "Test"}],
                        0.5,
                        100
                    )

                import asyncio
                asyncio.run(run_test())

                # Verify the correct URL was called
                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                actual_url = call_args[0][0]
                assert actual_url == expected_api_url, f"Expected {expected_api_url}, got {actual_url} for input {input_url}"

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens_override(self):
        """Test that temperature and max_tokens parameters override config values."""
        config = LLMConfig(
            api_key="test_key",
            model="test_model",
            base_url="http://localhost:11434/v1"
        )

        base_parameters = {"num_ctx": 8192, "temperature": 0.5}
        client = OllamaClient(config=config, model_parameters=base_parameters)

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            mock_response.json.return_value = {"model": "test", "response": "test", "done": True}
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Call with different temperature and max_tokens
            await client._create_completion(
                "test_model",
                [{"role": "user", "content": "Test"}],
                temperature=0.8,  # Override the 0.5 in base_parameters
                max_tokens=150    # This should become num_predict
            )

            # Verify the overrides were applied
            call_args = mock_client.post.call_args
            payload = call_args[1]['json']
            options = payload['options']

            assert options['num_ctx'] == 8192  # From base_parameters
            assert options['temperature'] == 0.8  # Overridden
            assert options['num_predict'] == 150  # From max_tokens
