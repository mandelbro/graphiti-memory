"""
Comprehensive tests for the configuration loader system covering edge cases and error conditions.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest
import yaml

from src.config_loader import ConfigLoader


class TestConfigLoaderComprehensive:
    """Comprehensive test suite for ConfigLoader including edge cases and error conditions."""

    def test_config_loader_default_config_dir(self):
        """Test that ConfigLoader uses correct default config directory."""
        loader = ConfigLoader()
        expected_config_dir = Path(__file__).parent.parent / "config"
        assert loader.config_dir == expected_config_dir

    def test_config_loader_custom_config_dir(self):
        """Test ConfigLoader with custom config directory."""
        custom_dir = Path("/custom/config/path")
        loader = ConfigLoader(custom_dir)
        assert loader.config_dir == custom_dir

    def test_load_yaml_config_invalid_yaml(self):
        """Test loading a YAML file with invalid syntax."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "invalid.yml"

            # Create invalid YAML file
            with open(config_file, 'w') as f:
                f.write("invalid: yaml: content:\n  - bad\n    indentation")

            loader = ConfigLoader(config_dir)
            result = loader.load_yaml_config("invalid.yml")

            # Should return empty dict on YAML parse error
            assert result == {}

    def test_load_yaml_config_permission_error(self):
        """Test loading a YAML file with permission issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            loader = ConfigLoader(config_dir)

            # Mock open to raise PermissionError
            with patch("builtins.open", side_effect=PermissionError("Permission denied")):
                result = loader.load_yaml_config("test.yml")
                assert result == {}

    def test_load_yaml_config_empty_file(self):
        """Test loading an empty YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "empty.yml"

            # Create empty file
            config_file.touch()

            loader = ConfigLoader(config_dir)
            result = loader.load_yaml_config("empty.yml")

            # Should return empty dict for empty file
            assert result == {}

    def test_load_database_config_custom_database(self):
        """Test loading custom database configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            database_dir = config_dir / "database"
            database_dir.mkdir()

            config_data = {
                'uri': 'bolt://custom:7687',
                'user': 'custom_user',
                'connection_pool_size': 100
            }

            config_file = database_dir / "postgres.yml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)

            loader = ConfigLoader(config_dir)
            result = loader.load_database_config("postgres")

            assert result == config_data

    def test_load_server_config(self):
        """Test loading server configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            server_config = {
                'mcp': {
                    'transport': 'stdio',
                    'port': 9000
                },
                'logging': {
                    'level': 'DEBUG'
                }
            }

            config_file = config_dir / "server.yml"
            with open(config_file, 'w') as f:
                yaml.dump(server_config, f)

            loader = ConfigLoader(config_dir)
            result = loader.load_server_config()

            assert result == server_config

    def test_merge_configs_complex_nesting(self):
        """Test merging configurations with complex nesting."""
        base_config = {
            'level1': {
                'level2': {
                    'level3': {
                        'value1': 'base',
                        'value2': 'base_only'
                    },
                    'other_level3': 'base'
                }
            },
            'top_level': 'base'
        }

        override_config = {
            'level1': {
                'level2': {
                    'level3': {
                        'value1': 'override',
                        'value3': 'override_only'
                    }
                },
                'new_level2': 'new'
            }
        }

        result = ConfigLoader.merge_configs(base_config, override_config)

        # Check deep merge behavior
        assert result['level1']['level2']['level3']['value1'] == 'override'  # Overridden
        assert result['level1']['level2']['level3']['value2'] == 'base_only'  # Preserved
        assert result['level1']['level2']['level3']['value3'] == 'override_only'  # Added
        assert result['level1']['level2']['other_level3'] == 'base'  # Preserved
        assert result['level1']['new_level2'] == 'new'  # Added
        assert result['top_level'] == 'base'  # Preserved

    def test_merge_configs_list_replacement(self):
        """Test that lists are replaced, not merged."""
        base_config = {
            'list_value': [1, 2, 3],
            'nested': {
                'list_value': ['a', 'b']
            }
        }

        override_config = {
            'list_value': [4, 5],
            'nested': {
                'list_value': ['c']
            }
        }

        result = ConfigLoader.merge_configs(base_config, override_config)

        # Lists should be replaced, not merged
        assert result['list_value'] == [4, 5]
        assert result['nested']['list_value'] == ['c']

    def test_merge_configs_type_conflicts(self):
        """Test merging when types conflict."""
        base_config = {
            'conflict': {
                'nested': 'value'
            }
        }

        override_config = {
            'conflict': 'string_value'
        }

        result = ConfigLoader.merge_configs(base_config, override_config)

        # Override should win on type conflicts
        assert result['conflict'] == 'string_value'

    def test_get_env_value_conversion_errors(self):
        """Test environment variable conversion with invalid values."""
        # Test invalid integer
        os.environ['TEST_INVALID_INT'] = 'not_a_number'
        result = ConfigLoader.get_env_value('TEST_INVALID_INT', default=42, convert_type=int)
        assert result == 42

        # Test invalid float
        os.environ['TEST_INVALID_FLOAT'] = 'not_a_float'
        result = ConfigLoader.get_env_value('TEST_INVALID_FLOAT', default=3.14, convert_type=float)
        assert result == 3.14

        # Test invalid boolean (should still convert)
        os.environ['TEST_INVALID_BOOL'] = 'maybe'
        result = ConfigLoader.get_env_value('TEST_INVALID_BOOL', default=True, convert_type=bool)
        assert result is False  # Invalid boolean values should be False

        # Clean up
        for key in ['TEST_INVALID_INT', 'TEST_INVALID_FLOAT', 'TEST_INVALID_BOOL']:
            if key in os.environ:
                del os.environ[key]

    def test_get_env_value_boolean_variations(self):
        """Test various boolean value interpretations."""
        boolean_tests = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', True),
            ('yes', True),
            ('Yes', True),
            ('on', True),
            ('On', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('0', False),
            ('no', False),
            ('off', False),
            ('anything_else', False),
            ('', False),
        ]

        for test_value, expected in boolean_tests:
            os.environ['TEST_BOOL'] = test_value
            result = ConfigLoader.get_env_value('TEST_BOOL', convert_type=bool)
            assert result == expected, f"Failed for value '{test_value}'"

        # Clean up
        if 'TEST_BOOL' in os.environ:
            del os.environ['TEST_BOOL']

    def test_get_env_value_with_none_default(self):
        """Test environment variable with None as default."""
        result = ConfigLoader.get_env_value('NONEXISTENT_VAR', default=None)
        assert result is None

    def test_config_loader_file_encoding(self):
        """Test YAML file loading with different encodings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Test with UTF-8 content including special characters
            config_data = {
                'description': 'Tëst wîth spëcîäl chäracters: 中文 日本語',
                'emoji': '🚀 Configuration test',
                'value': 42
            }

            config_file = config_dir / "encoding_test.yml"
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, allow_unicode=True)

            loader = ConfigLoader(config_dir)
            result = loader.load_yaml_config("encoding_test.yml")

            assert result == config_data

    def test_merge_configs_empty_inputs(self):
        """Test merging with empty configurations."""
        config1 = {'key': 'value'}
        empty_config = {}

        # Empty base
        result1 = ConfigLoader.merge_configs(empty_config, config1)
        assert result1 == config1

        # Empty override
        result2 = ConfigLoader.merge_configs(config1, empty_config)
        assert result2 == config1

        # Both empty
        result3 = ConfigLoader.merge_configs(empty_config, empty_config)
        assert result3 == {}

    def test_merge_configs_multiple_sources(self):
        """Test merging multiple configuration sources."""
        config1 = {'a': 1, 'b': {'x': 1}}
        config2 = {'b': {'y': 2}, 'c': 3}
        config3 = {'b': {'z': 3}, 'd': 4}

        result = ConfigLoader.merge_configs(config1, config2, config3)

        expected = {
            'a': 1,
            'b': {'x': 1, 'y': 2, 'z': 3},
            'c': 3,
            'd': 4
        }

        assert result == expected
