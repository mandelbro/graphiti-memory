"""Tests for max_tokens configuration functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.config.llm_config import GraphitiLLMConfig


class TestMaxTokensConfig:
    """Test cases for max_tokens configuration."""

    def test_default_max_tokens(self):
        """Test that default max_tokens is set correctly."""
        config = GraphitiLLMConfig()
        assert config.max_tokens == 8192

    def test_environment_variable_max_tokens(self):
        """Test that max_tokens can be set via environment variable."""
        with patch.dict(os.environ, {"LLM_MAX_TOKENS": "32768"}):
            config = GraphitiLLMConfig.from_env()
            assert config.max_tokens == 32768

    def test_from_env_with_max_tokens(self):
        """Test that from_env method respects max_tokens environment variable."""
        with patch.dict(os.environ, {"USE_OLLAMA": "true", "LLM_MAX_TOKENS": "16384"}):
            config = GraphitiLLMConfig.from_env()
            assert config.max_tokens == 16384

    def test_cli_override_max_tokens(self):
        """Test that CLI arguments can override max_tokens."""
        import argparse

        # Create a proper argparse.Namespace object
        args = argparse.Namespace()
        args.max_tokens = 65536

        # Test with environment variable set
        with patch.dict(os.environ, {"LLM_MAX_TOKENS": "16384"}):
            config = GraphitiLLMConfig.from_cli_and_env(args)
            # CLI should override environment variable
            assert config.max_tokens == 65536

    def test_invalid_max_tokens_environment(self):
        """Test that invalid max_tokens environment variable is handled gracefully."""
        with patch.dict(os.environ, {"LLM_MAX_TOKENS": "invalid"}):
            # Should raise ValueError when trying to convert to int
            with pytest.raises(ValueError):
                GraphitiLLMConfig.from_env()

    def test_max_tokens_in_create_client(self):
        """Test that max_tokens is passed to LLMConfig in create_client."""
        config = GraphitiLLMConfig(max_tokens=32768)

        # Mock the LLMConfig to capture the parameters
        with patch("src.config.llm_config.LLMConfig") as mock_llm_config:
            with patch("src.config.llm_config.OpenAIClient") as mock_client:
                # Create a mock LLMConfig instance with required attributes
                mock_config_instance = MagicMock()
                mock_config_instance.api_key = "test_key"
                mock_config_instance.base_url = "http://localhost:11434/v1"
                mock_llm_config.return_value = mock_config_instance
                mock_client.return_value = mock_client

                # Call create_client
                config.create_client()

                # Verify LLMConfig was called with max_tokens
                mock_llm_config.assert_called()
                call_args = mock_llm_config.call_args
                assert "max_tokens" in call_args[1]
                assert call_args[1]["max_tokens"] == 32768
