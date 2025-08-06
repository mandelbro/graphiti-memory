"""
Configuration loader for YAML-based configuration files.

This module provides utilities for loading and merging YAML configuration files
with environment variables and CLI arguments, following the precedence:
1. Default values (lowest priority)
2. YAML configuration files
3. Environment variables
4. CLI arguments (highest priority)
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Utility class for loading and merging configuration from multiple sources.

    Supports loading configuration from YAML files and merging with environment
    variables following a clear precedence hierarchy.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the ConfigLoader.

        Args:
            config_dir: Path to the configuration directory. Defaults to 'config/'
                       relative to the current working directory.
        """
        if config_dir is None:
            # Default to config/ directory relative to this file's location
            self.config_dir = Path(__file__).parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)

    def load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            config_path: Path to the YAML file relative to config_dir

        Returns:
            Dictionary containing the loaded configuration, or empty dict if file not found
        """
        full_path = self.config_dir / config_path

        if not full_path.exists():
            logger.debug(f"Configuration file not found: {full_path}")
            return {}

        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file) or {}
                logger.debug(f"Loaded configuration from {full_path}")
                return config
        except Exception as e:
            logger.warning(f"Failed to load configuration from {full_path}: {e}")
            return {}

    def load_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Load provider-specific configuration.

        Args:
            provider: Provider name (e.g., 'ollama', 'openai', 'azure_openai')

        Returns:
            Dictionary containing the provider configuration
        """
        return self.load_yaml_config(f"providers/{provider}.yml")

    def load_database_config(self, database: str = "neo4j") -> Dict[str, Any]:
        """
        Load database configuration.

        Args:
            database: Database name (default: 'neo4j')

        Returns:
            Dictionary containing the database configuration
        """
        return self.load_yaml_config(f"database/{database}.yml")

    def load_server_config(self) -> Dict[str, Any]:
        """
        Load server configuration.

        Returns:
            Dictionary containing the server configuration
        """
        return self.load_yaml_config("server.yml")

    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries with deep merging.

        Later configurations override earlier ones. Nested dictionaries are merged
        recursively, while lists and primitive values are replaced entirely.

        Args:
            *configs: Configuration dictionaries to merge

        Returns:
            Merged configuration dictionary
        """
        result = {}

        for config in configs:
            result = ConfigLoader._deep_merge(result, config)

        return result

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries.

        Args:
            base: Base dictionary
            override: Dictionary to merge into base

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def get_env_value(key: str, default: Any = None, convert_type: type = str) -> Any:
        """
        Get environment variable value with type conversion.

        Args:
            key: Environment variable name
            default: Default value if not found
            convert_type: Type to convert the value to

        Returns:
            Environment variable value converted to specified type, or default
        """
        value = os.environ.get(key)
        if value is None:
            return default

        try:
            if convert_type == bool:
                return value.lower() in ('true', '1', 'yes', 'on')
            elif convert_type == int:
                return int(value)
            elif convert_type == float:
                return float(value)
            else:
                return convert_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert environment variable {key}={value} to {convert_type}: {e}")
            return default


# Global config loader instance
config_loader = ConfigLoader()
