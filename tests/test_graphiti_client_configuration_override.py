"""
Test suite for Graphiti client configuration override implementation.

This test suite verifies that the configuration override functionality
correctly detects Ollama configurations and creates enhanced clients
for all memory operations.
"""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.initialization.graphiti_client import (
    _create_enhanced_ollama_client,
    _detect_ollama_configuration,
    _validate_ollama_configuration,
    create_graphiti_client,
    initialize_graphiti,
)

if TYPE_CHECKING:
    pass


@pytest.fixture
def mock_ollama_config():
    """Create mock configuration for Ollama testing."""
    config = MagicMock()

    # LLM configuration
    config.llm.use_ollama = True
    config.llm.ollama_llm_model = "llama3.2:3b"
    config.llm.ollama_base_url = "http://localhost:11434/v1"
    config.llm.temperature = 0.1
    config.llm.max_tokens = 2048
    config.llm.ollama_model_parameters = {"num_ctx": 8192, "top_p": 0.9}
    config.llm.create_client = MagicMock()

    # Embedder configuration
    config.embedder.use_ollama = False
    config.embedder.create_client = MagicMock(return_value=MagicMock())

    # Neo4j configuration
    config.neo4j.uri = "bolt://localhost:7687"
    config.neo4j.user = "neo4j"
    config.neo4j.password = "password"

    # Other configuration
    config.use_custom_entities = True
    config.destroy_graph = False
    config.group_id = "test-group"

    return config


@pytest.fixture
def mock_openai_config():
    """Create mock configuration for OpenAI testing."""
    config = MagicMock()

    # LLM configuration
    config.llm.use_ollama = False
    config.llm.base_url = "https://api.openai.com/v1"
    config.llm.ollama_base_url = None
    config.llm.api_key = "test-openai-key"
    config.llm.model = "gpt-4"
    config.llm.temperature = 0.7
    config.llm.max_tokens = 4096
    config.llm.create_client = MagicMock(return_value=MagicMock())

    # Embedder configuration
    config.embedder.use_ollama = False
    config.embedder.create_client = MagicMock(return_value=MagicMock())

    # Neo4j configuration
    config.neo4j.uri = "bolt://localhost:7687"
    config.neo4j.user = "neo4j"
    config.neo4j.password = "password"

    # Other configuration
    config.use_custom_entities = True
    config.destroy_graph = False
    config.group_id = "test-group"

    return config


@pytest.fixture
def mock_base_url_config():
    """Create mock configuration with Ollama detected via base_url."""
    config = MagicMock()

    # LLM configuration - no explicit use_ollama flag
    config.llm.use_ollama = False
    config.llm.base_url = "http://localhost:11434/v1"  # Ollama URL
    config.llm.ollama_base_url = None
    config.llm.ollama_llm_model = "llama3.2:3b"
    config.llm.temperature = 0.1
    config.llm.max_tokens = 2048
    config.llm.ollama_model_parameters = {}

    # Embedder configuration
    config.embedder.use_ollama = False
    config.embedder.create_client = MagicMock(return_value=MagicMock())

    # Neo4j configuration
    config.neo4j.uri = "bolt://localhost:7687"
    config.neo4j.user = "neo4j"
    config.neo4j.password = "password"

    # Other configuration
    config.use_custom_entities = True
    config.destroy_graph = False
    config.group_id = "test-group"

    return config


class TestOllamaDetection:
    """Test cases for Ollama configuration detection."""

    def test_detect_ollama_explicit_flag(self, mock_ollama_config):
        """Test detection via explicit use_ollama flag."""
        result = _detect_ollama_configuration(mock_ollama_config)
        assert result is True

    def test_detect_ollama_base_url_localhost(self, mock_base_url_config):
        """Test detection via base_url with localhost:11434."""
        result = _detect_ollama_configuration(mock_base_url_config)
        assert result is True

    def test_detect_ollama_base_url_variations(self):
        """Test detection with various URL formats."""
        test_cases = [
            "http://localhost:11434/v1",
            "http://127.0.0.1:11434/v1",
            "http://0.0.0.0:11434",
            "https://ollama-server:11434/api",
        ]

        for base_url in test_cases:
            config = MagicMock()
            config.llm.use_ollama = False
            config.llm.base_url = base_url
            config.llm.ollama_base_url = None

            result = _detect_ollama_configuration(config)
            assert result is True, f"Failed to detect Ollama for URL: {base_url}"

    def test_detect_ollama_via_ollama_base_url(self):
        """Test detection via ollama_base_url attribute."""
        config = MagicMock()
        config.llm.use_ollama = False
        config.llm.base_url = None
        config.llm.ollama_base_url = "http://localhost:11434/v1"

        result = _detect_ollama_configuration(config)
        assert result is True

    def test_no_ollama_detection(self, mock_openai_config):
        """Test that OpenAI configuration is not detected as Ollama."""
        result = _detect_ollama_configuration(mock_openai_config)
        assert result is False

    def test_no_ollama_detection_different_port(self):
        """Test that non-11434 ports are not detected as Ollama."""
        config = MagicMock()
        config.llm.use_ollama = False
        config.llm.base_url = "http://localhost:8080/v1"
        config.llm.ollama_base_url = None

        result = _detect_ollama_configuration(config)
        assert result is False


class TestOllamaValidation:
    """Test cases for Ollama configuration validation."""

    def test_validate_ollama_success(self, mock_ollama_config):
        """Test successful validation of Ollama configuration."""
        # Should not raise an exception
        _validate_ollama_configuration(mock_ollama_config)

    def test_validate_ollama_missing_model(self, mock_ollama_config):
        """Test validation failure when model is missing."""
        mock_ollama_config.llm.ollama_llm_model = None

        with pytest.raises(ValueError, match="OLLAMA_LLM_MODEL must be set"):
            _validate_ollama_configuration(mock_ollama_config)

    def test_validate_ollama_empty_model(self, mock_ollama_config):
        """Test validation failure when model is empty."""
        mock_ollama_config.llm.ollama_llm_model = "   "

        with pytest.raises(ValueError, match="OLLAMA_LLM_MODEL must be set"):
            _validate_ollama_configuration(mock_ollama_config)

    def test_validate_ollama_invalid_base_url(self, mock_ollama_config):
        """Test validation failure for invalid base URL format."""
        mock_ollama_config.llm.ollama_base_url = "invalid-url"

        with pytest.raises(ValueError, match="Invalid Ollama base URL format"):
            _validate_ollama_configuration(mock_ollama_config)

    def test_validate_ollama_valid_https_url(self, mock_ollama_config):
        """Test validation success for HTTPS URL."""
        mock_ollama_config.llm.ollama_base_url = "https://secure-ollama:11434/v1"

        # Should not raise an exception
        _validate_ollama_configuration(mock_ollama_config)


class TestEnhancedClientCreation:
    """Test cases for enhanced Ollama client creation."""

    @patch('src.ollama_client.OllamaClient')
    @patch('graphiti_core.llm_client.config.LLMConfig')
    def test_create_enhanced_ollama_client_success(self, mock_llm_config_class, mock_ollama_client_class, mock_ollama_config):
        """Test successful creation of enhanced Ollama client."""
        mock_llm_config_instance = MagicMock()
        mock_llm_config_class.return_value = mock_llm_config_instance

        mock_ollama_client_instance = MagicMock()
        mock_ollama_client_class.return_value = mock_ollama_client_instance

        result = _create_enhanced_ollama_client(mock_ollama_config)

        # Verify LLMConfig was created with correct parameters
        mock_llm_config_class.assert_called_once_with(
            api_key="abc",
            model="llama3.2:3b",
            small_model="llama3.2:3b",
            temperature=0.1,
            max_tokens=2048,
            base_url="http://localhost:11434/v1",
        )

        # Verify OllamaClient was created with correct parameters
        mock_ollama_client_class.assert_called_once_with(
            config=mock_llm_config_instance,
            model_parameters={"num_ctx": 8192, "top_p": 0.9}
        )

        assert result == mock_ollama_client_instance

    @patch('src.ollama_client.OllamaClient')
    def test_create_enhanced_ollama_client_failure(self, mock_ollama_client_class, mock_ollama_config):
        """Test handling of client creation failure."""
        mock_ollama_client_class.side_effect = Exception("Connection failed")

        with pytest.raises(ValueError, match="Failed to create Ollama client"):
            _create_enhanced_ollama_client(mock_ollama_config)


class TestGraphitiClientCreation:
    """Test cases for Graphiti client creation with configuration override."""

    @patch('src.initialization.graphiti_client.Graphiti')
    @patch('src.initialization.graphiti_client._create_enhanced_ollama_client')
    @patch('src.initialization.graphiti_client._validate_ollama_configuration')
    @patch('src.initialization.graphiti_client._detect_ollama_configuration')
    @pytest.mark.asyncio
    async def test_create_graphiti_client_ollama(
        self,
        mock_detect,
        mock_validate,
        mock_create_client,
        mock_graphiti_class,
        mock_ollama_config
    ):
        """Test Graphiti client creation with Ollama configuration."""
        mock_detect.return_value = True
        mock_ollama_client = MagicMock()
        mock_create_client.return_value = mock_ollama_client
        mock_embedder = MagicMock()
        mock_ollama_config.embedder.create_client.return_value = mock_embedder

        mock_graphiti_instance = MagicMock()
        mock_graphiti_class.return_value = mock_graphiti_instance

        result = await create_graphiti_client(mock_ollama_config)

        # Verify detection and validation were called
        mock_detect.assert_called_once_with(mock_ollama_config)
        mock_validate.assert_called_once_with(mock_ollama_config)
        mock_create_client.assert_called_once_with(mock_ollama_config)

        # Verify Graphiti was created with enhanced client
        mock_graphiti_class.assert_called_once()
        call_kwargs = mock_graphiti_class.call_args[1]
        assert call_kwargs['llm_client'] == mock_ollama_client
        assert call_kwargs['embedder'] == mock_embedder
        assert call_kwargs['uri'] == "bolt://localhost:7687"
        assert call_kwargs['user'] == "neo4j"
        assert call_kwargs['password'] == "password"

        assert result == mock_graphiti_instance

    @patch('src.initialization.graphiti_client.Graphiti')
    @patch('src.initialization.graphiti_client._detect_ollama_configuration')
    @pytest.mark.asyncio
    async def test_create_graphiti_client_openai(
        self,
        mock_detect,
        mock_graphiti_class,
        mock_openai_config
    ):
        """Test Graphiti client creation with OpenAI configuration."""
        mock_detect.return_value = False
        mock_llm_client = MagicMock()
        mock_openai_config.llm.create_client.return_value = mock_llm_client
        mock_embedder = MagicMock()
        mock_openai_config.embedder.create_client.return_value = mock_embedder

        mock_graphiti_instance = MagicMock()
        mock_graphiti_class.return_value = mock_graphiti_instance

        result = await create_graphiti_client(mock_openai_config)

        # Verify standard client creation was used
        mock_detect.assert_called_once_with(mock_openai_config)
        mock_openai_config.llm.create_client.assert_called_once()

        # Verify Graphiti was created with standard client
        mock_graphiti_class.assert_called_once()
        call_kwargs = mock_graphiti_class.call_args[1]
        assert call_kwargs['llm_client'] == mock_llm_client
        assert call_kwargs['embedder'] == mock_embedder

        assert result == mock_graphiti_instance

    @patch('src.initialization.graphiti_client._detect_ollama_configuration')
    @pytest.mark.asyncio
    async def test_create_graphiti_client_no_llm_custom_entities(
        self,
        mock_detect,
        mock_openai_config
    ):
        """Test error when no LLM client and custom entities enabled."""
        mock_detect.return_value = False
        mock_openai_config.llm.create_client.return_value = None
        mock_openai_config.use_custom_entities = True

        with pytest.raises(ValueError, match="LLM client is required when custom entities are enabled"):
            await create_graphiti_client(mock_openai_config)


class TestFullIntegration:
    """Test cases for full initialization integration."""

    @patch('src.initialization.graphiti_client.create_graphiti_client')
    @patch('src.tools.memory_tools.set_globals')
    @patch('src.tools.search_tools.set_globals')
    @patch('src.tools.management_tools.set_globals')
    @pytest.mark.asyncio
    async def test_initialize_graphiti_success(
        self,
        mock_management_globals,
        mock_search_globals,
        mock_memory_globals,
        mock_create_client,
        mock_ollama_config
    ):
        """Test successful full initialization with Ollama."""
        mock_graphiti_client = AsyncMock()
        mock_graphiti_client.llm_client = MagicMock()
        mock_graphiti_client.embedder = MagicMock()
        mock_create_client.return_value = mock_graphiti_client

        result = await initialize_graphiti(mock_ollama_config)

        # Verify client creation was called
        mock_create_client.assert_called_once_with(mock_ollama_config)

        # Verify initialization steps
        mock_graphiti_client.build_indices_and_constraints.assert_called_once()

        # Verify tool globals were set
        mock_memory_globals.assert_called_once_with(mock_graphiti_client, mock_ollama_config)
        mock_search_globals.assert_called_once_with(mock_graphiti_client, mock_ollama_config)
        mock_management_globals.assert_called_once_with(mock_graphiti_client, mock_ollama_config)

        assert result == mock_graphiti_client

    @pytest.mark.asyncio
    async def test_initialize_graphiti_missing_neo4j_config(self, mock_ollama_config):
        """Test initialization failure with missing Neo4j configuration."""
        mock_ollama_config.neo4j.uri = None

        with pytest.raises(ValueError, match="NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set"):
            await initialize_graphiti(mock_ollama_config)

    @patch('src.initialization.graphiti_client.create_graphiti_client')
    @pytest.mark.asyncio
    async def test_initialize_graphiti_client_creation_failure(
        self,
        mock_create_client,
        mock_ollama_config
    ):
        """Test initialization failure when client creation fails."""
        mock_create_client.side_effect = Exception("Client creation failed")

        with pytest.raises(Exception, match="Client creation failed"):
            await initialize_graphiti(mock_ollama_config)


class TestBackwardCompatibility:
    """Test cases for backward compatibility."""

    def test_existing_functionality_preserved(self, mock_openai_config):
        """Test that existing OpenAI functionality is preserved."""
        # Should detect as non-Ollama
        result = _detect_ollama_configuration(mock_openai_config)
        assert result is False

        # Standard client creation should work
        mock_openai_config.llm.create_client.assert_not_called()  # Only called when needed

    @patch('src.initialization.graphiti_client._detect_ollama_configuration')
    @patch('src.initialization.graphiti_client.Graphiti')
    @pytest.mark.asyncio
    async def test_fallback_to_standard_initialization(
        self,
        mock_graphiti_class,
        mock_detect,
        mock_openai_config
    ):
        """Test fallback to standard initialization for non-Ollama configs."""
        mock_detect.return_value = False
        mock_llm_client = MagicMock()
        mock_openai_config.llm.create_client.return_value = mock_llm_client

        mock_graphiti_instance = MagicMock()
        mock_graphiti_class.return_value = mock_graphiti_instance

        result = await create_graphiti_client(mock_openai_config)

        # Should use standard client creation
        mock_openai_config.llm.create_client.assert_called_once()
        assert result == mock_graphiti_instance
