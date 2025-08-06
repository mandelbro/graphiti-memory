"""
Comprehensive tests for the OllamaClient implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from graphiti_core.llm_client.config import LLMConfig
from pydantic import BaseModel

from src.ollama_client import OllamaClient


class MockResponseModel(BaseModel):
    """Mock response model for testing."""
    test_field: str = "test_value"


class TestOllamaClientComprehensive:
    """Comprehensive test suite for OllamaClient."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock()
        mock_client.beta.chat.completions.parse = AsyncMock()
        return mock_client

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

    @pytest.fixture
    def model_parameters(self):
        """Create test model parameters."""
        return {
            "num_ctx": 4096,
            "num_predict": 100,
            "repeat_penalty": 1.1,
            "top_k": 40,
            "top_p": 0.9,
            "temperature": 0.2,
            "seed": 42
        }

    def test_ollama_client_initialization_with_config(self, llm_config, model_parameters):
        """Test OllamaClient initialization with configuration."""
        client = OllamaClient(
            config=llm_config,
            cache=False,  # Caching not implemented in base class
            max_tokens=2000,
            model_parameters=model_parameters
        )

        assert client.model_parameters == model_parameters
        assert hasattr(client, 'client')

    def test_ollama_client_initialization_with_mock_client(self, mock_openai_client, llm_config, model_parameters):
        """Test OllamaClient initialization with mock client."""
        client = OllamaClient(
            config=llm_config,
            client=mock_openai_client,
            model_parameters=model_parameters
        )

        assert client.client == mock_openai_client
        assert client.model_parameters == model_parameters

    def test_ollama_client_initialization_no_config(self, model_parameters):
        """Test OllamaClient initialization without config."""
        client = OllamaClient(model_parameters=model_parameters)

        assert client.model_parameters == model_parameters
        assert hasattr(client, 'client')

    def test_ollama_client_initialization_no_model_parameters(self, llm_config):
        """Test OllamaClient initialization without model parameters."""
        client = OllamaClient(config=llm_config)

        assert client.model_parameters == {}

    @pytest.mark.asyncio
    async def test_create_structured_completion_with_model_parameters(self, mock_openai_client, llm_config, model_parameters):
        """Test structured completion with model parameters."""
        # Setup
        expected_response = MagicMock()
        mock_openai_client.beta.chat.completions.parse.return_value = expected_response

        client = OllamaClient(
            config=llm_config,
            client=mock_openai_client,
            model_parameters=model_parameters
        )

        # Test data
        model = "test_model"
        messages = [{"role": "user", "content": "test message"}]
        temperature = 0.7
        max_tokens = 500

        # Execute
        result = await client._create_structured_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=MockResponseModel
        )

        # Verify
        assert result == expected_response
        mock_openai_client.beta.chat.completions.parse.assert_called_once_with(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=MockResponseModel,
            extra_body=model_parameters
        )

    @pytest.mark.asyncio
    async def test_create_structured_completion_without_model_parameters(self, mock_openai_client, llm_config):
        """Test structured completion without model parameters."""
        # Setup
        expected_response = MagicMock()
        mock_openai_client.beta.chat.completions.parse.return_value = expected_response

        client = OllamaClient(
            config=llm_config,
            client=mock_openai_client,
            model_parameters={}  # Empty model parameters
        )

        # Test data
        model = "test_model"
        messages = [{"role": "user", "content": "test message"}]
        temperature = 0.7
        max_tokens = 500

        # Execute
        result = await client._create_structured_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=MockResponseModel
        )

        # Verify - should not include extra_body when no model parameters
        assert result == expected_response
        mock_openai_client.beta.chat.completions.parse.assert_called_once_with(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=MockResponseModel
        )

    @pytest.mark.asyncio
    async def test_create_completion_with_model_parameters(self, mock_openai_client, llm_config, model_parameters):
        """Test regular completion with model parameters."""
        # Setup
        expected_response = MagicMock()
        mock_openai_client.chat.completions.create.return_value = expected_response

        client = OllamaClient(
            config=llm_config,
            client=mock_openai_client,
            model_parameters=model_parameters
        )

        # Test data
        model = "test_model"
        messages = [{"role": "user", "content": "test message"}]
        temperature = 0.7
        max_tokens = 500

        # Execute
        result = await client._create_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Verify
        assert result == expected_response
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body=model_parameters
        )

    @pytest.mark.asyncio
    async def test_create_completion_without_model_parameters(self, mock_openai_client, llm_config):
        """Test regular completion without model parameters."""
        # Setup
        expected_response = MagicMock()
        mock_openai_client.chat.completions.create.return_value = expected_response

        client = OllamaClient(
            config=llm_config,
            client=mock_openai_client,
            model_parameters=None  # No model parameters
        )

        # Test data
        model = "test_model"
        messages = [{"role": "user", "content": "test message"}]
        temperature = 0.7
        max_tokens = 500

        # Execute
        result = await client._create_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=MockResponseModel
        )

        # Verify - should not include extra_body when no model parameters
        assert result == expected_response
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

    def test_ollama_client_model_parameters_property(self, llm_config, model_parameters):
        """Test that model parameters are accessible as a property."""
        client = OllamaClient(
            config=llm_config,
            model_parameters=model_parameters
        )

        # Test getting model parameters
        assert client.model_parameters == model_parameters

        # Test modifying model parameters
        new_parameters = {"num_ctx": 8192}
        client.model_parameters = new_parameters
        assert client.model_parameters == new_parameters

    def test_ollama_client_inheritance(self, llm_config):
        """Test that OllamaClient properly inherits from BaseOpenAIClient."""
        client = OllamaClient(config=llm_config)

        # Should have inherited methods and properties
        assert hasattr(client, '_create_completion')
        assert hasattr(client, '_create_structured_completion')
        assert hasattr(client, 'client')

    @pytest.mark.asyncio
    async def test_create_completion_with_response_model(self, mock_openai_client, llm_config, model_parameters):
        """Test completion with response model parameter."""
        # Setup
        expected_response = MagicMock()
        mock_openai_client.chat.completions.create.return_value = expected_response

        client = OllamaClient(
            config=llm_config,
            client=mock_openai_client,
            model_parameters=model_parameters
        )

        # Test data
        model = "test_model"
        messages = [{"role": "user", "content": "test message"}]
        temperature = 0.7
        max_tokens = 500

        # Execute with response_model (should be ignored in regular completion)
        result = await client._create_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=MockResponseModel
        )

        # Verify
        assert result == expected_response
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body=model_parameters
        )

    def test_ollama_client_with_complex_model_parameters(self, llm_config):
        """Test OllamaClient with complex nested model parameters."""
        complex_parameters = {
            "num_ctx": 4096,
            "num_predict": -1,
            "repeat_penalty": 1.1,
            "top_k": 40,
            "top_p": 0.9,
            "temperature": 0.2,
            "seed": 42,
            "stop": ["Human:", "Assistant:"],
            "nested_config": {
                "sub_param": "value",
                "sub_list": [1, 2, 3]
            }
        }

        client = OllamaClient(
            config=llm_config,
            model_parameters=complex_parameters
        )

        assert client.model_parameters == complex_parameters
        assert client.model_parameters["nested_config"]["sub_param"] == "value"
        assert client.model_parameters["stop"] == ["Human:", "Assistant:"]
