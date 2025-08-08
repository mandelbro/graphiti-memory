"""
Comprehensive tests for the OllamaClient implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any
import httpx

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
        # Setup - mock the _create_completion method since that's what gets called now
        expected_response = MagicMock()

        client = OllamaClient(
            config=llm_config,
            client=mock_openai_client,
            model_parameters=model_parameters
        )

        # Mock the _create_completion method that _create_structured_completion now calls
        client._create_completion = AsyncMock(return_value=expected_response)

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

        # Verify - the method now falls back to _create_completion
        assert result == expected_response
        client._create_completion.assert_called_once_with(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=MockResponseModel
        )
        # Beta API should not be called
        mock_openai_client.beta.chat.completions.parse.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_structured_completion_without_model_parameters(self, mock_openai_client, llm_config):
        """Test structured completion without model parameters."""
        # Setup - mock the _create_completion method instead since that's what gets called now
        expected_response = MagicMock()

        client = OllamaClient(
            config=llm_config,
            client=mock_openai_client,
            model_parameters={}  # Empty model parameters
        )

        # Mock the _create_completion method that _create_structured_completion now calls
        client._create_completion = AsyncMock(return_value=expected_response)

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

        # Verify - the method now falls back to _create_completion
        assert result == expected_response
        client._create_completion.assert_called_once_with(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=MockResponseModel
        )
        # Beta API should not be called
        mock_openai_client.beta.chat.completions.parse.assert_not_called()

    @pytest.mark.asyncio
    async def test_structured_completion_integration_with_native_api(self, llm_config):
        """Test complete integration of structured completion with native API fallback."""
        model_parameters = {
            "num_ctx": 8192,
            "temperature": 0.3,
            "top_p": 0.9
        }

        client = OllamaClient(
            config=llm_config,
            model_parameters=model_parameters
        )

        # Mock both the model loading and actual completion calls
        with patch('httpx.AsyncClient') as mock_native_client_class:
            # Setup native API mock for model loading and completion
            mock_native_client = AsyncMock()
            mock_native_response = AsyncMock()
            mock_native_response.status_code = 200
            mock_native_response.raise_for_status = AsyncMock()
            mock_native_response.json.return_value = {
                "response": "Integration test response from native API",
                "done": True
            }
            mock_native_client.post.return_value = mock_native_response
            mock_native_client.__aenter__.return_value = mock_native_client
            mock_native_client.__aexit__.return_value = None
            mock_native_client_class.return_value = mock_native_client

            # Test data
            model = "integration_test_model"
            messages = [{"role": "user", "content": "Integration test message"}]
            temperature = 0.2
            max_tokens = 200

            # Execute structured completion
            result = await client._create_structured_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_model=MockResponseModel
            )

            # Verify that native API was called for the completion
            # Since structured completion falls back to _create_completion, which uses native API
            assert mock_native_client.post.call_count >= 1  # At least one call for completion

            # Verify result structure
            assert hasattr(result, 'choices')
            assert len(result.choices) > 0
            assert hasattr(result.choices[0], 'message')
            assert result.choices[0].message.content == "Integration test response from native API"

    @pytest.mark.asyncio
    async def test_structured_completion_error_handling(self, llm_config):
        """Test error handling in structured completion fallback."""
        client = OllamaClient(config=llm_config)

        # Mock _create_completion to raise an exception
        error_message = "Native API error"
        client._create_completion = AsyncMock(side_effect=Exception(error_message))

        # Test data
        model = "error_test_model"
        messages = [{"role": "user", "content": "Error test message"}]

        # Execute and verify exception is propagated
        with pytest.raises(Exception) as exc_info:
            await client._create_structured_completion(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=100,
                response_model=MockResponseModel
            )

        assert str(exc_info.value) == error_message

    @pytest.mark.asyncio
    async def test_graphiti_episode_processing_scenario(self, llm_config):
        """Test the specific scenario that was failing in Graphiti episode processing."""
        # This test simulates the real-world scenario where Graphiti processes episodes
        # and uses structured completions which were returning empty responses

        model_parameters = {
            "num_ctx": 12000,
            "num_predict": -1,
            "repeat_penalty": 1.1,
            "top_k": 50,
            "top_p": 0.9,
            "num_threads": 40
        }

        client = OllamaClient(
            config=llm_config,
            model_parameters=model_parameters
        )

        with patch('httpx.AsyncClient') as mock_native_client_class:
            # Setup native API mock to return a proper response (not empty)
            mock_native_client = AsyncMock()
            mock_native_response = AsyncMock()
            mock_native_response.status_code = 200
            mock_native_response.raise_for_status = AsyncMock()
            mock_native_response.json.return_value = {
                "response": "Processed episode content with entities and relationships extracted.",
                "done": True
            }
            mock_native_client.post.return_value = mock_native_response
            mock_native_client.__aenter__.return_value = mock_native_client
            mock_native_client.__aexit__.return_value = None
            mock_native_client_class.return_value = mock_native_client

            # Simulate Graphiti episode processing call
            messages = [{
                "role": "system",
                "content": "Extract entities and relationships from the following episode content."
            }, {
                "role": "user",
                "content": "User Documentation Standards Preference: The user prefers clear, concise documentation."
            }]

            # Execute structured completion (as Graphiti would do)
            result = await client._create_structured_completion(
                model="gpt-oss:latest",
                messages=messages,
                temperature=0.1,
                max_tokens=8192,
                response_model=MockResponseModel
            )

            # Verify we get a non-empty response (the original issue was empty responses)
            assert result is not None
            assert hasattr(result, 'choices')
            assert len(result.choices) > 0
            assert hasattr(result.choices[0], 'message')
            assert result.choices[0].message.content != ""  # This was the main issue - empty content
            assert "Processed episode content" in result.choices[0].message.content

            # Verify native API was called with correct parameters
            assert mock_native_client.post.call_count >= 1

            # Check that model parameters were included
            for call in mock_native_client.post.call_args_list:
                if 'json' in call[1]:
                    payload = call[1]['json']
                    if 'options' in payload:
                        assert payload['options']['num_ctx'] == 12000
                        break

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

    @pytest.mark.asyncio
    async def test_native_api_create_completion(self, llm_config, model_parameters):
        """Test that create_completion uses native API with parameters."""
        with patch('httpx.AsyncClient') as mock_client_class:
            # Setup mock response
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            mock_response.json.return_value = {
                "model": "test_model",
                "response": "Test response from native API",
                "done": True
            }
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Create client
            client = OllamaClient(
                config=llm_config,
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

            # Verify native API call was made
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args

            # Check URL
            assert call_args[0][0] == "http://localhost:11434/api/generate"

            # Check payload
            payload = call_args[1]['json']
            assert payload['model'] == "test_model"
            assert payload['prompt'] == "User: test message"
            assert payload['stream'] is False
            assert payload['options'] == model_parameters
            assert payload['options']['temperature'] == 0.7
            assert payload['options']['num_predict'] == 500

            # Check response format conversion
            assert hasattr(result, 'choices')
            assert len(result.choices) == 1
            assert result.choices[0].message.content == "Test response from native API"
            assert result.choices[0].message.role == "assistant"

    @pytest.mark.asyncio
    async def test_messages_to_prompt_conversion(self, llm_config):
        """Test conversion of OpenAI messages to Ollama prompt format."""
        client = OllamaClient(config=llm_config)

        # Test various message types
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "What's the weather like?"}
        ]

        prompt = client._messages_to_prompt(messages)
        expected = (
            "System: You are a helpful assistant\n"
            "User: Hello, how are you?\n"
            "Assistant: I'm doing well, thank you!\n"
            "User: What's the weather like?"
        )

        assert prompt == expected

    @pytest.mark.asyncio
    async def test_url_conversion_from_openai_format(self, model_parameters):
        """Test that base URL is correctly converted from OpenAI-compatible to native."""
        # Test with /v1 suffix
        config_with_v1 = LLMConfig(
            api_key="test_key",
            model="test_model",
            base_url="http://localhost:11434/v1"
        )

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

            client = OllamaClient(
                config=config_with_v1,
                model_parameters=model_parameters
            )

            await client._create_completion(
                "test_model",
                [{"role": "user", "content": "test"}],
                0.5,
                100
            )

            # Verify the URL was converted correctly
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/generate"

    @pytest.mark.asyncio
    async def test_keep_alive_parameter_in_create_completion(self, llm_config):
        """Test that keep_alive parameter is included in _create_completion payload."""
        model_parameters_with_keep_alive = {
            "num_ctx": 8192,
            "keep_alive": 300  # 5 minutes in seconds
        }

        client = OllamaClient(
            config=llm_config,
            model_parameters=model_parameters_with_keep_alive
        )

        with patch('httpx.AsyncClient') as mock_client_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            mock_response.json.return_value = {"response": "Test completion response"}
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Test completion
            messages = [{"role": "user", "content": "Test message"}]
            result = await client._create_completion(
                model="test-model",
                messages=messages,
                temperature=0.5,
                max_tokens=100
            )

            # Verify the API call
            assert mock_client.post.called
            call_args = mock_client.post.call_args_list[0]
            payload = call_args[1]['json']

            # Verify keep_alive is in the payload
            assert 'keep_alive' in payload
            assert payload['keep_alive'] == 300
            assert payload['options']['num_ctx'] == 8192
            assert payload['options']['temperature'] == 0.5
            assert payload['options']['num_predict'] == 100

    @pytest.mark.asyncio
    async def test_no_keep_alive_parameter_handling(self, llm_config):
        """Test that payload doesn't include keep_alive when not in model_parameters."""
        model_parameters_without_keep_alive = {
            "num_ctx": 4096,
            "temperature": 0.7
        }

        client = OllamaClient(
            config=llm_config,
            model_parameters=model_parameters_without_keep_alive
        )

        with patch('httpx.AsyncClient') as mock_client_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            mock_response.json.return_value = {"response": "Test response"}
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Test completion
            messages = [{"role": "user", "content": "Test message"}]
            await client._create_completion(
                model="test-model",
                messages=messages,
                temperature=0.5,
                max_tokens=100
            )

            # Verify the API call
            assert mock_client.post.called
            payload = mock_client.post.call_args[1]['json']

            # Verify keep_alive is NOT in the payload
            assert 'keep_alive' not in payload
            assert payload['options']['num_ctx'] == 4096

    @pytest.mark.asyncio
    async def test_keep_alive_parameter_edge_cases(self, llm_config):
        """Test keep_alive parameter with various value formats."""
        test_cases = [
            ("5m", "5m"),           # String duration
            (300, 300),             # Integer seconds
            ("0", "0"),             # Zero (unload immediately)
            ("-1", "-1"),           # Keep forever
        ]

        for keep_alive_input, expected_output in test_cases:
            model_parameters = {
                "num_ctx": 2048,
                "keep_alive": keep_alive_input
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
                mock_response.json.return_value = {"response": "Test response"}
                mock_client.post.return_value = mock_response
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client_class.return_value = mock_client

                # Test completion
                messages = [{"role": "user", "content": "Test message"}]
                await client._create_completion(
                    model="test-model",
                    messages=messages,
                    temperature=0.5,
                    max_tokens=50
                )

                # Verify the keep_alive value
                payload = mock_client.post.call_args[1]['json']
                assert 'keep_alive' in payload
                assert payload['keep_alive'] == expected_output

    @pytest.mark.asyncio
    async def test_context_parameter_preservation(self, llm_config):
        """Test that num_ctx and other parameters are preserved in native API calls."""
        context_parameters = {
            "num_ctx": 12000,
            "num_threads": 40,
            "repeat_penalty": 1.1,
            "top_k": 50,
            "top_p": 0.9
        }

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

            client = OllamaClient(
                config=llm_config,
                model_parameters=context_parameters
            )

            await client._create_completion(
                "gpt-oss:latest",
                [{"role": "user", "content": "Hello"}],
                0.1,
                100
            )

            # Verify all context parameters were sent
            call_args = mock_client.post.call_args
            payload = call_args[1]['json']
            options = payload['options']

            assert options['num_ctx'] == 12000
            assert options['num_threads'] == 40
            assert options['repeat_penalty'] == 1.1
            assert options['top_k'] == 50
            assert options['top_p'] == 0.9

    @pytest.mark.asyncio
    async def test_native_api_error_handling(self, llm_config, model_parameters):
        """Test error handling in native API calls."""
        with patch('httpx.AsyncClient') as mock_client_class:
            # Setup mock to raise exception
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.RequestError("Connection failed")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            client = OllamaClient(
                config=llm_config,
                model_parameters=model_parameters
            )

            # The method should raise the exception (not handle it silently)
            with pytest.raises(httpx.RequestError):
                await client._create_completion(
                    "test_model",
                    [{"role": "user", "content": "test"}],
                    0.5,
                    100
                )
