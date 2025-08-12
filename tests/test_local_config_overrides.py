"""
Tests for local configuration override functionality.

This module tests the new local override system for provider configurations,
ensuring that providers/{name}.local.yml files properly override base configurations.
"""

import tempfile
from pathlib import Path

from src.config_loader import ConfigLoader


class TestLocalConfigOverrides:
    """Test suite for local configuration override functionality."""

    def test_load_provider_config_with_no_local_override(self):
        """Test loading provider config when no local override exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            providers_dir = config_dir / "providers"
            providers_dir.mkdir()

            # Create base config
            base_config = {
                "llm": {
                    "model": "base-model",
                    "temperature": 0.1,
                    "base_url": "http://localhost:11434/v1",
                }
            }

            import yaml

            with open(providers_dir / "test.yml", "w") as f:
                yaml.dump(base_config, f)

            # Initialize config loader
            loader = ConfigLoader(config_dir)

            # Load config
            result = loader.load_provider_config("test")

            # Should return base config as-is
            assert result == base_config

    def test_load_provider_config_with_local_override(self):
        """Test loading provider config with local override."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            providers_dir = config_dir / "providers"
            providers_dir.mkdir()

            # Create base config
            base_config = {
                "llm": {
                    "model": "base-model",
                    "temperature": 0.1,
                    "base_url": "http://localhost:11434/v1",
                    "max_tokens": 8192,
                }
            }

            # Create local override config
            local_config = {
                "llm": {
                    "model": "override-model",
                    "temperature": 0.3,
                    "max_tokens": 4096,
                    # Note: base_url not overridden
                }
            }

            import yaml

            with open(providers_dir / "test.yml", "w") as f:
                yaml.dump(base_config, f)

            with open(providers_dir / "test.local.yml", "w") as f:
                yaml.dump(local_config, f)

            # Initialize config loader
            loader = ConfigLoader(config_dir)

            # Load config
            result = loader.load_provider_config("test")

            # Should have merged config with local overrides taking precedence
            expected = {
                "llm": {
                    "model": "override-model",  # Overridden
                    "temperature": 0.3,  # Overridden
                    "base_url": "http://localhost:11434/v1",  # From base
                    "max_tokens": 4096,  # Overridden
                }
            }

            assert result == expected

    def test_load_provider_config_deep_merge(self):
        """Test that deep merging works correctly for nested configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            providers_dir = config_dir / "providers"
            providers_dir.mkdir()

            # Create base config with nested structure
            base_config = {
                "llm": {
                    "model": "base-model",
                    "model_parameters": {
                        "num_ctx": 4096,
                        "temperature": 0.1,
                        "top_k": 50,
                    },
                }
            }

            # Create local override with partial nested override
            local_config = {
                "llm": {
                    "model_parameters": {
                        "num_ctx": 8192,
                        "temperature": 0.3,
                        # Note: top_k not overridden
                    }
                }
            }

            import yaml

            with open(providers_dir / "test.yml", "w") as f:
                yaml.dump(base_config, f)

            with open(providers_dir / "test.local.yml", "w") as f:
                yaml.dump(local_config, f)

            # Initialize config loader
            loader = ConfigLoader(config_dir)

            # Load config
            result = loader.load_provider_config("test")

            # Should have deep merged config
            expected = {
                "llm": {
                    "model": "base-model",  # From base
                    "model_parameters": {
                        "num_ctx": 8192,  # Overridden
                        "temperature": 0.3,  # Overridden
                        "top_k": 50,  # From base
                    },
                }
            }

            assert result == expected

    def test_load_provider_config_missing_base_file(self):
        """Test loading provider config when base file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            providers_dir = config_dir / "providers"
            providers_dir.mkdir()

            # Initialize config loader
            loader = ConfigLoader(config_dir)

            # Load non-existent config
            result = loader.load_provider_config("nonexistent")

            # Should return empty dict
            assert result == {}

    def test_load_provider_config_only_local_file(self):
        """Test loading provider config when only local file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            providers_dir = config_dir / "providers"
            providers_dir.mkdir()

            # Create only local config (no base)
            local_config = {"llm": {"model": "local-only-model", "temperature": 0.5}}

            import yaml

            with open(providers_dir / "test.local.yml", "w") as f:
                yaml.dump(local_config, f)

            # Initialize config loader
            loader = ConfigLoader(config_dir)

            # Load config
            result = loader.load_provider_config("test")

            # Should return only the local config
            assert result == local_config
