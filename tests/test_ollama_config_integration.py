"""
Integration tests for Ollama configuration with model parameters.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest
import yaml

from src.config_loader import ConfigLoader
from src.graphiti_mcp_server import GraphitiLLMConfig


class TestOllamaConfigIntegration:
    """Test Ollama configuration integration with YAML and environment variables."""

    def test_ollama_yaml_config_loading(self):
        """Test that Ollama configuration loads model parameters from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create providers directory and ollama config
            providers_dir = config_dir / 'providers'
            providers_dir.mkdir()

            ollama_config = {
                'llm': {
                    'model': 'test-model:7b',
                    'base_url': 'http://localhost:11434/v1',
                    'temperature': 0.2,
                    'max_tokens': 4096,
                    'model_parameters': {
                        'num_ctx': 8192,
                        'num_predict': 100,
                        'repeat_penalty': 1.2,
                        'top_k': 50,
                        'top_p': 0.95,
                        'temperature': 0.15,  # Model-level temperature
                        'seed': 42
                    }
                }
            }

            config_file = providers_dir / 'ollama.yml'
            with open(config_file, 'w') as f:
                yaml.dump(ollama_config, f)

            # Temporarily override the config loader's config directory
            from src.config_loader import config_loader
            original_config_dir = config_loader.config_dir
            config_loader.config_dir = config_dir

            try:
                # Clear any existing OLLAMA environment variables (including from .env)
                saved_env_vars = {}
                for key in list(os.environ.keys()):
                    if key.startswith('OLLAMA') or key.startswith('LLM_'):
                        saved_env_vars[key] = os.environ[key]
                        del os.environ[key]

                # Set environment variables for Ollama
                os.environ['USE_OLLAMA'] = 'true'

                # Create LLM config from YAML and env
                llm_config = GraphitiLLMConfig.from_yaml_and_env()

                # Verify basic configuration
                assert llm_config.use_ollama is True
                assert llm_config.ollama_llm_model == 'test-model:7b'
                assert llm_config.ollama_base_url == 'http://localhost:11434/v1'
                assert llm_config.temperature == 0.2
                assert llm_config.max_tokens == 4096

                # Verify model parameters were loaded
                assert llm_config.ollama_model_parameters['num_ctx'] == 8192
                assert llm_config.ollama_model_parameters['num_predict'] == 100
                assert llm_config.ollama_model_parameters['repeat_penalty'] == 1.2
                assert llm_config.ollama_model_parameters['top_k'] == 50
                assert llm_config.ollama_model_parameters['top_p'] == 0.95
                assert llm_config.ollama_model_parameters['temperature'] == 0.15
                assert llm_config.ollama_model_parameters['seed'] == 42

            finally:
                # Clean up environment variables
                if 'USE_OLLAMA' in os.environ:
                    del os.environ['USE_OLLAMA']

                # Restore saved environment variables
                for key, value in saved_env_vars.items():
                    os.environ[key] = value

                # Restore original config directory
                config_loader.config_dir = original_config_dir

    def test_environment_variables_override_yaml(self):
        """Test that environment variables override YAML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create providers directory and ollama config
            providers_dir = config_dir / 'providers'
            providers_dir.mkdir()

            ollama_config = {
                'llm': {
                    'model': 'yaml-model:7b',
                    'base_url': 'http://yaml:11434/v1',
                    'temperature': 0.1,
                    'max_tokens': 2048,
                    'model_parameters': {
                        'num_ctx': 4096
                    }
                }
            }

            config_file = providers_dir / 'ollama.yml'
            with open(config_file, 'w') as f:
                yaml.dump(ollama_config, f)

            # Temporarily override the config loader's config directory
            from src.config_loader import config_loader
            original_config_dir = config_loader.config_dir
            config_loader.config_dir = config_dir

            try:
                # Set environment variables that should override YAML
                os.environ['USE_OLLAMA'] = 'true'
                os.environ['OLLAMA_LLM_MODEL'] = 'env-model:13b'
                os.environ['OLLAMA_BASE_URL'] = 'http://env:11434/v1'
                os.environ['LLM_TEMPERATURE'] = '0.5'
                os.environ['LLM_MAX_TOKENS'] = '16384'

                # Create LLM config from YAML and env
                llm_config = GraphitiLLMConfig.from_yaml_and_env()

                # Verify that environment variables override YAML
                assert llm_config.ollama_llm_model == 'env-model:13b'  # From env
                assert llm_config.ollama_base_url == 'http://env:11434/v1'  # From env
                assert llm_config.temperature == 0.5  # From env
                assert llm_config.max_tokens == 16384  # From env

                # Verify that model parameters still come from YAML
                assert llm_config.ollama_model_parameters['num_ctx'] == 4096

            finally:
                # Clean up environment variables
                for key in ['USE_OLLAMA', 'OLLAMA_LLM_MODEL', 'OLLAMA_BASE_URL', 'LLM_TEMPERATURE', 'LLM_MAX_TOKENS']:
                    if key in os.environ:
                        del os.environ[key]

                # Restore original config directory
                config_loader.config_dir = original_config_dir

    def test_ollama_client_creation(self):
        """Test that OllamaClient is created with model parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create providers directory and ollama config
            providers_dir = config_dir / 'providers'
            providers_dir.mkdir()

            ollama_config = {
                'llm': {
                    'model': 'test-model:7b',
                    'model_parameters': {
                        'num_ctx': 8192,
                        'top_p': 0.9
                    }
                }
            }

            config_file = providers_dir / 'ollama.yml'
            with open(config_file, 'w') as f:
                yaml.dump(ollama_config, f)

            # Temporarily override the config loader's config directory
            from src.config_loader import config_loader
            original_config_dir = config_loader.config_dir
            config_loader.config_dir = config_dir

            try:
                # Set environment variables for Ollama
                os.environ['USE_OLLAMA'] = 'true'

                # Create LLM config from YAML and env
                llm_config = GraphitiLLMConfig.from_yaml_and_env()

                # Create the client
                client = llm_config.create_client()

                # Verify it's an OllamaClient (check by class name since we can't import it here easily)
                assert client.__class__.__name__ == 'OllamaClient'

                # Verify the model parameters were passed
                assert hasattr(client, 'model_parameters')
                assert client.model_parameters['num_ctx'] == 8192
                assert client.model_parameters['top_p'] == 0.9

            finally:
                # Clean up environment variables
                if 'USE_OLLAMA' in os.environ:
                    del os.environ['USE_OLLAMA']

                # Restore original config directory
                config_loader.config_dir = original_config_dir

    def test_ollama_client_native_api_integration(self):
        """Test that OllamaClient properly integrates with native API for parameter loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create providers directory and ollama config with context parameters
            providers_dir = config_dir / 'providers'
            providers_dir.mkdir()

            ollama_config = {
                'llm': {
                    'model': 'test-model:latest',
                    'base_url': 'http://localhost:11434/v1',
                    'model_parameters': {
                        'num_ctx': 12000,
                        'num_threads': 40,
                        'repeat_penalty': 1.1,
                        'top_p': 0.9
                    }
                }
            }

            config_file = providers_dir / 'ollama.yml'
            with open(config_file, 'w') as f:
                yaml.dump(ollama_config, f)

            # Temporarily override the config loader's config directory
            original_config_dir = config_loader.config_dir
            config_loader.config_dir = config_dir

            try:
                # Set environment variables
                os.environ['USE_OLLAMA'] = 'true'

                # Create the configuration and client
                llm_config = GraphitiLLMConfig.from_yaml_and_env()
                client = llm_config.create_client()

                # Verify the client has the expected configuration
                assert client.__class__.__name__ == 'OllamaClient'
                assert hasattr(client, 'model_parameters')
                assert client.model_parameters['num_ctx'] == 12000
                assert client.model_parameters['num_threads'] == 40
                assert client.model_parameters['repeat_penalty'] == 1.1
                assert client.model_parameters['top_p'] == 0.9

                # Verify the ollama_base_url is set correctly
                assert hasattr(client, 'ollama_base_url')
                assert client.ollama_base_url == 'http://localhost:11434/v1'

            finally:
                # Clean up environment variables
                if 'USE_OLLAMA' in os.environ:
                    del os.environ['USE_OLLAMA']

                # Restore original config directory
                config_loader.config_dir = original_config_dir

    @pytest.mark.asyncio
    async def test_ollama_native_api_call_integration(self):
        """Test the complete flow of native API calls with parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create providers directory and ollama config
            providers_dir = config_dir / 'providers'
            providers_dir.mkdir()

            ollama_config = {
                'llm': {
                    'model': 'integration-test:latest',
                    'base_url': 'http://localhost:11434/v1',
                    'model_parameters': {
                        'num_ctx': 8192,
                        'num_threads': 20,
                        'temperature': 0.5
                    }
                }
            }

            config_file = providers_dir / 'ollama.yml'
            with open(config_file, 'w') as f:
                yaml.dump(ollama_config, f)

            # Temporarily override the config loader's config directory
            original_config_dir = config_loader.config_dir
            config_loader.config_dir = config_dir

            try:
                # Set environment variables
                os.environ['USE_OLLAMA'] = 'true'

                # Create the configuration and client
                llm_config = GraphitiLLMConfig.from_yaml_and_env()
                client = llm_config.create_client()

                # Mock the native API call
                with patch('httpx.AsyncClient') as mock_client_class:
                    mock_client = AsyncMock()
                    mock_response = AsyncMock()
                    mock_response.status_code = 200
                    mock_response.raise_for_status = AsyncMock()
                    mock_response.json.return_value = {
                        "model": "integration-test:latest",
                        "response": "Integration test response",
                        "done": True
                    }
                    mock_client.post.return_value = mock_response
                    mock_client.__aenter__.return_value = mock_client
                    mock_client.__aexit__.return_value = None
                    mock_client_class.return_value = mock_client

                    # Test the native API completion
                    result = await client._create_completion(
                        "integration-test:latest",
                        [{"role": "user", "content": "Integration test prompt"}],
                        0.5,
                        100
                    )

                    # Verify the native API call was made correctly
                    mock_client.post.assert_called_once()
                    call_args = mock_client.post.call_args

                    # Check URL (should be converted from /v1 to native)
                    assert call_args[0][0] == "http://localhost:11434/api/generate"

                    # Check payload structure
                    payload = call_args[1]['json']
                    assert payload['model'] == "integration-test:latest"
                    assert payload['prompt'] == "User: Integration test prompt"
                    assert payload['stream'] is False

                    # Check that all parameters were included
                    options = payload['options']
                    assert options['num_ctx'] == 8192
                    assert options['num_threads'] == 20
                    assert options['temperature'] == 0.5  # From payload overrides config

                    # Verify response conversion
                    assert result.choices[0].message.content == "Integration test response"

            finally:
                # Clean up environment variables
                if 'USE_OLLAMA' in os.environ:
                    del os.environ['USE_OLLAMA']

                # Restore original config directory
                config_loader.config_dir = original_config_dir
