"""
Tests for the configuration loader system.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config_loader import ConfigLoader


class TestConfigLoader:
    """Test the ConfigLoader functionality."""

    def test_load_yaml_config_existing_file(self):
        """Test loading an existing YAML configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create a test YAML file
            test_config = {
                'llm': {
                    'model': 'test-model',
                    'temperature': 0.5,
                    'model_parameters': {
                        'num_ctx': 2048,
                        'top_p': 0.8
                    }
                }
            }

            providers_dir = config_dir / 'providers'
            providers_dir.mkdir()

            config_file = providers_dir / 'test.yml'
            with open(config_file, 'w') as f:
                yaml.dump(test_config, f)

            # Test loading the configuration
            loader = ConfigLoader(config_dir)
            loaded_config = loader.load_yaml_config('providers/test.yml')

            assert loaded_config == test_config
            assert loaded_config['llm']['model'] == 'test-model'
            assert loaded_config['llm']['model_parameters']['num_ctx'] == 2048

    def test_load_yaml_config_missing_file(self):
        """Test loading a non-existent YAML configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            loader = ConfigLoader(config_dir)

            # Try to load a non-existent file
            loaded_config = loader.load_yaml_config('providers/nonexistent.yml')

            assert loaded_config == {}

    def test_load_provider_config(self):
        """Test loading provider-specific configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create providers directory and test config
            providers_dir = config_dir / 'providers'
            providers_dir.mkdir()

            test_config = {
                'llm': {
                    'model': 'ollama-model',
                    'base_url': 'http://localhost:11434/v1',
                    'model_parameters': {
                        'num_ctx': 4096,
                        'temperature': 0.1
                    }
                }
            }

            config_file = providers_dir / 'ollama.yml'
            with open(config_file, 'w') as f:
                yaml.dump(test_config, f)

            # Test loading the provider configuration
            loader = ConfigLoader(config_dir)
            loaded_config = loader.load_provider_config('ollama')

            assert loaded_config == test_config

    def test_merge_configs(self):
        """Test merging multiple configuration dictionaries."""
        base_config = {
            'llm': {
                'model': 'base-model',
                'temperature': 0.0,
                'model_parameters': {
                    'num_ctx': 2048,
                    'top_p': 0.9
                }
            },
            'embedder': {
                'model': 'base-embedder'
            }
        }

        override_config = {
            'llm': {
                'temperature': 0.5,
                'model_parameters': {
                    'num_ctx': 4096,
                    'top_k': 40
                }
            }
        }

        merged = ConfigLoader.merge_configs(base_config, override_config)

        # Check that values were merged correctly
        assert merged['llm']['model'] == 'base-model'  # From base
        assert merged['llm']['temperature'] == 0.5  # Overridden
        assert merged['llm']['model_parameters']['num_ctx'] == 4096  # Overridden
        assert merged['llm']['model_parameters']['top_p'] == 0.9  # From base
        assert merged['llm']['model_parameters']['top_k'] == 40  # Added
        assert merged['embedder']['model'] == 'base-embedder'  # From base

    def test_get_env_value(self):
        """Test getting environment variable values with type conversion."""
        # Test string value
        os.environ['TEST_STRING'] = 'test_value'
        assert ConfigLoader.get_env_value('TEST_STRING') == 'test_value'

        # Test integer conversion
        os.environ['TEST_INT'] = '42'
        assert ConfigLoader.get_env_value('TEST_INT', convert_type=int) == 42

        # Test float conversion
        os.environ['TEST_FLOAT'] = '3.14'
        assert ConfigLoader.get_env_value('TEST_FLOAT', convert_type=float) == 3.14

        # Test boolean conversion
        os.environ['TEST_BOOL_TRUE'] = 'true'
        os.environ['TEST_BOOL_FALSE'] = 'false'
        assert ConfigLoader.get_env_value('TEST_BOOL_TRUE', convert_type=bool) is True
        assert ConfigLoader.get_env_value('TEST_BOOL_FALSE', convert_type=bool) is False

        # Test default value
        assert ConfigLoader.get_env_value('NONEXISTENT_VAR', default='default') == 'default'

        # Clean up
        for key in ['TEST_STRING', 'TEST_INT', 'TEST_FLOAT', 'TEST_BOOL_TRUE', 'TEST_BOOL_FALSE']:
            if key in os.environ:
                del os.environ[key]
