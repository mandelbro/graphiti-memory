"""
Tests for error handling and edge cases in the configuration system.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest
import yaml

from src.config_loader import ConfigLoader, config_loader
from src.graphiti_mcp_server import GraphitiLLMConfig


class TestConfigErrorHandling:
    """Test error handling and edge cases in configuration system."""

    def test_yaml_file_permission_denied(self):
        """Test handling of permission denied when reading YAML files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            loader = ConfigLoader(config_dir)

            # Mock open to raise PermissionError
            with patch("builtins.open", side_effect=PermissionError("Permission denied")):
                result = loader.load_yaml_config("test.yml")
                assert result == {}

    def test_yaml_file_io_error(self):
        """Test handling of IO errors when reading YAML files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            loader = ConfigLoader(config_dir)

            # Mock open to raise IOError
            with patch("builtins.open", side_effect=IOError("IO error")):
                result = loader.load_yaml_config("test.yml")
                assert result == {}

    def test_yaml_file_corrupted_content(self):
        """Test handling of corrupted YAML content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "corrupted.yml"

            # Write corrupted YAML content
            with open(config_file, 'w') as f:
                f.write("corrupted: yaml: content:\n  - unmatched\n    bracket: {\n")

            loader = ConfigLoader(config_dir)
            result = loader.load_yaml_config("corrupted.yml")

            # Should return empty dict on parse error
            assert result == {}

    def test_yaml_file_unicode_decode_error(self):
        """Test handling of unicode decode errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "unicode_error.yml"

            # Write invalid UTF-8 bytes
            with open(config_file, 'wb') as f:
                f.write(b'\xff\xfe\x00\x00invalid unicode')

            loader = ConfigLoader(config_dir)
            result = loader.load_yaml_config("unicode_error.yml")

            # Should return empty dict on decode error
            assert result == {}

    def test_merge_configs_with_none_values(self):
        """Test merging configurations with None values."""
        base_config = {
            'key1': 'value1',
            'key2': None,
            'nested': {
                'sub_key': 'sub_value'
            }
        }

        override_config = {
            'key2': 'new_value',
            'key3': None,
            'nested': None  # Override entire nested dict with None
        }

        result = ConfigLoader.merge_configs(base_config, override_config)

        assert result['key1'] == 'value1'
        assert result['key2'] == 'new_value'
        assert result['key3'] is None
        assert result['nested'] is None  # Should be overridden

    def test_get_env_value_with_invalid_type(self):
        """Test environment variable conversion with invalid type."""
        os.environ['TEST_VAR'] = 'test_value'

        # Test with unsupported type conversion
        class CustomType:
            def __init__(self, value):
                if value == 'fail':
                    raise ValueError("Custom conversion error")
                self.value = value

        result = ConfigLoader.get_env_value('TEST_VAR', default='default', convert_type=CustomType)

        # Should return the converted value
        assert isinstance(result, CustomType)
        assert result.value == 'test_value'

        # Test with conversion that raises an exception
        os.environ['TEST_VAR'] = 'fail'
        result = ConfigLoader.get_env_value('TEST_VAR', default='default', convert_type=CustomType)
        assert result == 'default'

        # Clean up
        del os.environ['TEST_VAR']

    def test_graphiti_llm_config_yaml_load_error(self):
        """Test GraphitiLLMConfig when YAML loading fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Override config loader directory
            original_config_dir = config_loader.config_dir
            config_loader.config_dir = config_dir

            # Clear all relevant environment variables
            saved_env_vars = {}
            for key in list(os.environ.keys()):
                if key.startswith('OLLAMA') or key.startswith('LLM_'):
                    saved_env_vars[key] = os.environ[key]
                    del os.environ[key]

            try:
                # Mock load_provider_config to raise an exception
                with patch.object(config_loader, 'load_provider_config', side_effect=Exception("YAML load error")):
                    os.environ['USE_OLLAMA'] = 'true'

                    # Should still work with default values when YAML loading fails
                    config = GraphitiLLMConfig.from_yaml_and_env()

                    assert config.use_ollama is True
                    assert config.ollama_llm_model == 'deepseek-r1:7b'  # Default value

            finally:
                config_loader.config_dir = original_config_dir
                if 'USE_OLLAMA' in os.environ:
                    del os.environ['USE_OLLAMA']

                # Restore saved environment variables
                for key, value in saved_env_vars.items():
                    os.environ[key] = value

    def test_config_loader_invalid_path_type(self):
        """Test ConfigLoader with invalid path types."""
        # Test with None
        loader = ConfigLoader(None)
        assert loader.config_dir.name == "config"

        # Test with string path
        loader = ConfigLoader("/tmp/test")
        assert loader.config_dir == Path("/tmp/test")

    def test_yaml_config_deeply_nested_error(self):
        """Test YAML configuration with deeply nested structure that causes errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            providers_dir = config_dir / 'providers'
            providers_dir.mkdir()

            # Create a deeply nested config that might cause issues
            complex_config = {
                'llm': {
                    'model_parameters': {
                        'nested': {
                            'level1': {
                                'level2': {
                                    'level3': {
                                        'value': 'deep_value'
                                    }
                                }
                            }
                        }
                    }
                }
            }

            config_file = providers_dir / 'ollama.yml'
            with open(config_file, 'w') as f:
                yaml.dump(complex_config, f)

            # Override config loader directory
            original_config_dir = config_loader.config_dir
            config_loader.config_dir = config_dir

            try:
                os.environ['USE_OLLAMA'] = 'true'

                config = GraphitiLLMConfig.from_yaml_and_env()

                # Should handle deeply nested structures
                assert config.ollama_model_parameters['nested']['level1']['level2']['level3']['value'] == 'deep_value'

            finally:
                config_loader.config_dir = original_config_dir
                if 'USE_OLLAMA' in os.environ:
                    del os.environ['USE_OLLAMA']

    def test_environment_variable_circular_dependency(self):
        """Test handling of environment variables that might create circular dependencies."""
        # This is a theoretical test - in practice, env vars don't create circular deps
        # but testing the robustness of the config system
        os.environ['TEST_VAR1'] = '${TEST_VAR2}'
        os.environ['TEST_VAR2'] = '${TEST_VAR1}'

        # Should just return the literal string values
        result1 = ConfigLoader.get_env_value('TEST_VAR1')
        result2 = ConfigLoader.get_env_value('TEST_VAR2')

        assert result1 == '${TEST_VAR2}'
        assert result2 == '${TEST_VAR1}'

        # Clean up
        del os.environ['TEST_VAR1']
        del os.environ['TEST_VAR2']

    def test_config_with_very_large_values(self):
        """Test configuration system with very large values."""
        large_value = 'x' * 10000  # 10KB string
        very_large_number = 999999999999999999

        os.environ['LARGE_STRING'] = large_value
        os.environ['LARGE_NUMBER'] = str(very_large_number)

        # Should handle large values gracefully
        string_result = ConfigLoader.get_env_value('LARGE_STRING')
        number_result = ConfigLoader.get_env_value('LARGE_NUMBER', convert_type=int)

        assert string_result == large_value
        assert number_result == very_large_number

        # Clean up
        del os.environ['LARGE_STRING']
        del os.environ['LARGE_NUMBER']

    def test_config_with_special_characters(self):
        """Test configuration with special characters and unicode."""
        special_chars = "!@#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        unicode_chars = "测试 🚀 café naïve résumé"

        os.environ['SPECIAL_CHARS'] = special_chars
        os.environ['UNICODE_CHARS'] = unicode_chars

        special_result = ConfigLoader.get_env_value('SPECIAL_CHARS')
        unicode_result = ConfigLoader.get_env_value('UNICODE_CHARS')

        assert special_result == special_chars
        assert unicode_result == unicode_chars

        # Clean up
        del os.environ['SPECIAL_CHARS']
        del os.environ['UNICODE_CHARS']

    def test_merge_configs_with_circular_references(self):
        """Test merge_configs with potential circular reference structures."""
        # Create configs that reference each other (though Python dicts can't actually be circular)
        config1 = {'a': {'ref': 'config1'}}
        config2 = {'a': {'ref': 'config2', 'new_key': 'value'}}

        result = ConfigLoader.merge_configs(config1, config2)

        # Should merge without issues
        assert result['a']['ref'] == 'config2'
        assert result['a']['new_key'] == 'value'

    def test_yaml_config_with_yaml_injection_attempt(self):
        """Test YAML configuration security against injection attempts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "injection.yml"

            # Attempt to create YAML with potentially malicious content
            malicious_yaml = """
!!python/object/apply:os.system
- "echo malicious command"
model: "safe_model"
"""

            with open(config_file, 'w') as f:
                f.write(malicious_yaml)

            loader = ConfigLoader(config_dir)

            # yaml.safe_load should prevent execution
            result = loader.load_yaml_config("injection.yml")

            # Should either return empty dict (safe_load blocks it) or just the safe content
            if result:
                assert 'model' in result
                assert result['model'] == 'safe_model'

    def test_config_loader_memory_usage_with_large_files(self):
        """Test memory usage with large YAML files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "large.yml"

            # Create a moderately large config file
            large_config = {
                'model_parameters': {
                    f'param_{i}': f'value_{i}' for i in range(1000)
                }
            }

            with open(config_file, 'w') as f:
                yaml.dump(large_config, f)

            loader = ConfigLoader(config_dir)
            result = loader.load_yaml_config("large.yml")

            # Should handle large files gracefully
            assert len(result['model_parameters']) == 1000
            assert result['model_parameters']['param_500'] == 'value_500'
