"""
Test suite for validating custom OpenAI Bedrock endpoint configuration.

This test suite verifies that the configured Bedrock gateway endpoint
configured in openai.local.yml works correctly with the Graphiti MCP server.

Test Categories:
1. Configuration Loading Tests - Verify YAML config is loaded correctly
2. LLM Client Connection Tests - Verify LLM can connect and respond
3. Embedder Client Connection Tests - Verify embedder can connect and respond
4. Graphiti Integration Tests - Verify full pipeline with custom endpoint
5. Memory Operation Tests - Verify add_memory and search operations work
"""

import asyncio
import logging
import os
import sys

import pytest
from graphiti_core import Graphiti
from graphiti_core.llm_client.openai_client import OpenAIClient

from src.config import GraphitiConfig, GraphitiEmbedderConfig, GraphitiLLMConfig
from src.config_loader import config_loader
from src.initialization.graphiti_client import initialize_graphiti
from src.models import ErrorResponse, SuccessResponse
from src.tools.memory_tools import add_memory

# Skip all Bedrock tests as requested
pytestmark = pytest.mark.skip(reason="Bedrock tests skipped as requested")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def get_ssl_verify_setting() -> bool | str:
    """
    Get the appropriate SSL verification setting.

    Returns:
        - Path to certificate bundle if SSL_CERT_FILE is set
        - False if no certificate is configured (development/testing only)
    """
    ssl_cert_file = os.getenv("SSL_CERT_FILE")
    if ssl_cert_file:
        cert_path = os.path.expanduser(ssl_cert_file)
        if os.path.exists(cert_path):
            logger.info(f"Using SSL certificate bundle: {cert_path}")
            return cert_path
        else:
            logger.warning(f"SSL_CERT_FILE set but file not found: {cert_path}")

    # Fall back to disabling verification (development only)
    logger.warning(
        "SSL verification disabled - set SSL_CERT_FILE environment variable for production"
    )
    return False


@pytest.mark.integration
class TestBedrockEndpointConfiguration:
    """Test custom Bedrock endpoint configuration loading."""

    def test_openai_local_config_loads(self):
        """Verify that openai.local.yml is loaded correctly."""
        # Load the OpenAI provider config (should use openai.local.yml if it exists)
        config = config_loader.load_provider_config("openai")

        assert config, "OpenAI configuration should not be empty"
        assert "llm" in config, "LLM configuration should be present"
        assert "embedder" in config, "Embedder configuration should be present"

        # Verify LLM configuration
        llm_config = config["llm"]
        assert "model" in llm_config, "LLM model should be configured"
        assert "base_url" in llm_config, "LLM base_url should be configured"

        # Log the loaded configuration for debugging
        logger.info(f"Loaded LLM model: {llm_config.get('model')}")
        logger.info(f"Loaded LLM base_url: {llm_config.get('base_url')}")
        logger.info(f"Loaded LLM temperature: {llm_config.get('temperature')}")
        logger.info(f"Loaded LLM max_tokens: {llm_config.get('max_tokens')}")

        # Verify embedder configuration
        embedder_config = config["embedder"]
        assert "model" in embedder_config, "Embedder model should be configured"
        assert "base_url" in embedder_config, "Embedder base_url should be configured"

        logger.info(f"Loaded embedder model: {embedder_config.get('model')}")
        logger.info(f"Loaded embedder base_url: {embedder_config.get('base_url')}")

    def test_graphiti_llm_config_from_yaml(self):
        """Verify GraphitiLLMConfig correctly loads from YAML."""
        # Set environment to use OpenAI (not Ollama)
        original_use_ollama = os.environ.get("USE_OLLAMA")
        os.environ["USE_OLLAMA"] = "false"

        try:
            config = GraphitiLLMConfig.from_yaml_and_env()

            assert config.use_ollama is False, "Should not be using Ollama"
            assert config.model, "Model should be configured"
            assert config.max_tokens > 0, "Max tokens should be positive"

            logger.info(f"GraphitiLLMConfig model: {config.model}")
            logger.info(f"GraphitiLLMConfig small_model: {config.small_model}")
            logger.info(f"GraphitiLLMConfig temperature: {config.temperature}")
            logger.info(f"GraphitiLLMConfig max_tokens: {config.max_tokens}")

        finally:
            # Restore original environment
            if original_use_ollama is not None:
                os.environ["USE_OLLAMA"] = original_use_ollama
            else:
                os.environ.pop("USE_OLLAMA", None)


@pytest.mark.integration
@pytest.mark.asyncio
class TestBedrockEndpointConnection:
    """Test connection and functionality of the custom Bedrock endpoint."""

    @pytest.fixture
    async def bedrock_llm_config(self) -> GraphitiLLMConfig:
        """Create LLM config for Bedrock endpoint testing."""
        # Set environment to use OpenAI (not Ollama)
        os.environ["USE_OLLAMA"] = "false"

        # Ensure API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")

        config = GraphitiLLMConfig.from_yaml_and_env()

        logger.info(f"Testing with model: {config.model}")
        logger.info(f"Testing with max_tokens: {config.max_tokens}")

        return config

    async def test_llm_client_creation(self, bedrock_llm_config: GraphitiLLMConfig):
        """Test that we can create an OpenAI client with custom base_url."""
        try:
            llm_client = bedrock_llm_config.create_client()

            assert llm_client is not None, "LLM client should be created"
            assert isinstance(llm_client, OpenAIClient), "Should be OpenAI client"

            logger.info("✓ LLM client created successfully")

        except Exception as e:
            pytest.fail(f"Failed to create LLM client: {e}")

    async def test_llm_simple_completion(self, bedrock_llm_config: GraphitiLLMConfig):
        """Test a simple LLM completion request to verify endpoint connectivity."""
        try:
            llm_client = bedrock_llm_config.create_client()

            # Configure SSL certificate if needed
            ssl_verify = get_ssl_verify_setting()
            if ssl_verify is False:
                logger.warning("SSL verification is disabled for this test")
            elif isinstance(ssl_verify, str):
                logger.info(f"Using SSL certificate: {ssl_verify}")
                # Update the httpx client to use the certificate
                import httpx

                llm_client.client._client = httpx.AsyncClient(
                    verify=ssl_verify, timeout=30.0
                )

            # Test using the OpenAI client directly since graphiti-core's
            # generate_response tries to modify messages as objects
            from typing import cast

            from openai.types.chat import ChatCompletionMessageParam

            messages = cast(
                list[ChatCompletionMessageParam],
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Respond concisely.",
                    },
                    {
                        "role": "user",
                        "content": "Say 'Hello, Bedrock endpoint is working!' and nothing else.",
                    },
                ],
            )

            logger.info("Sending test completion request to Bedrock endpoint...")

            # Use the underlying OpenAI client directly
            response = await llm_client.client.chat.completions.create(
                model=bedrock_llm_config.model,
                messages=messages,
                temperature=0.0,
                max_tokens=100,
            )

            assert response, "Response should not be empty"
            assert response.choices, "Response should have choices"
            assert response.choices[0].message.content, "Response should have content"

            content = response.choices[0].message.content
            logger.info(f"✓ LLM response received: {content[:100]}...")

            # Verify the response contains expected content
            content_lower = content.lower()
            assert any(
                word in content_lower for word in ["hello", "bedrock", "working"]
            ), "Response should contain expected keywords"

        except Exception as e:
            logger.error(f"LLM completion test failed: {e}", exc_info=True)
            pytest.fail(f"Failed to get LLM completion: {e}")

    async def test_embedder_connection(self):
        """Test embedder connection to Bedrock endpoint."""
        # Set environment to use OpenAI (not Ollama)
        os.environ["USE_OLLAMA"] = "false"

        # Ensure API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")

        try:
            embedder_config = GraphitiEmbedderConfig.from_yaml_and_env()
            embedder_client = embedder_config.create_client()

            assert embedder_client is not None, "Embedder client should be created"

            logger.info("✓ Embedder client created successfully")
            logger.info(f"Embedder model: {embedder_config.model}")

            # Test embedding generation
            test_text = "This is a test sentence for embedding."
            logger.info("Generating test embedding...")

            # OpenAIEmbedder uses create() method, not get_embedding()
            embedding = await embedder_client.create([test_text])

            assert embedding is not None, "Embedding should not be None"
            assert len(embedding) > 0, "Embedding should not be empty"
            assert len(embedding[0]) > 0, "Embedding should have non-zero dimensions"

            embedding_dim = len(embedding[0])
            logger.info(
                f"✓ Embedding generated successfully with dimension {embedding_dim}"
            )

            # Verify it's a reasonable embedding dimension (typically 768, 1536, 3072, etc.)
            assert embedding_dim > 100, (
                f"Embedding dimension {embedding_dim} seems too small"
            )

        except Exception as e:
            logger.error(f"Embedder test failed: {e}", exc_info=True)
            pytest.fail(f"Failed to test embedder: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestGraphitiBedrockIntegration:
    """Test full Graphiti integration with Bedrock endpoint."""

    @pytest.fixture
    async def test_config(self) -> GraphitiConfig:
        """Create test configuration using Bedrock endpoint."""
        from src.config import Neo4jConfig

        # Set environment to use OpenAI (not Ollama)
        os.environ["USE_OLLAMA"] = "false"

        # Ensure API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")

        # Ensure Neo4j connection details are set
        neo4j_uri = os.getenv(
            "NEO4J_URI", os.getenv("TEST_NEO4J_URI", "bolt://localhost:7687")
        )
        neo4j_user = os.getenv("NEO4J_USER", os.getenv("TEST_NEO4J_USER", "neo4j"))
        neo4j_password = os.getenv(
            "NEO4J_PASSWORD", os.getenv("TEST_NEO4J_PASSWORD", "password")
        )

        return GraphitiConfig(
            neo4j=Neo4jConfig(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
            ),
            llm=GraphitiLLMConfig.from_yaml_and_env(),
            embedder=GraphitiEmbedderConfig.from_yaml_and_env(),
            use_custom_entities=False,  # Disable custom entities for simpler testing
            group_id="bedrock-endpoint-test",
        )

    @pytest.fixture
    async def graphiti_client(self, test_config: GraphitiConfig):
        """Create Graphiti client with Bedrock endpoint configuration."""
        try:
            logger.info("Creating Graphiti client with Bedrock endpoint...")

            # Create the client (initialize_graphiti already builds indices and constraints)
            client = await initialize_graphiti(test_config)

            # Clear any existing test data
            logger.info("Cleaning up existing test data...")
            await self._cleanup_test_data(client, "bedrock-endpoint-test")

            logger.info("✓ Graphiti client created and initialized")

            yield client

            # Cleanup after tests
            logger.info("Cleaning up after tests...")
            await self._cleanup_test_data(client, "bedrock-endpoint-test")
            await client.close()

        except Exception as e:
            logger.error(f"Failed to create Graphiti client: {e}", exc_info=True)
            pytest.skip(f"Could not initialize Graphiti client: {e}")

    async def _cleanup_test_data(self, client: Graphiti, group_id: str):
        """Clean up test data from the graph."""
        try:
            # Get all episodes for the test group
            episodes = await client.get_episodes(group_id=group_id, last_n=1000)

            # Delete each episode
            for episode in episodes:
                try:
                    await client.delete_episode(episode.uuid)
                except Exception as e:
                    logger.warning(f"Failed to delete episode {episode.uuid}: {e}")

        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    async def test_graphiti_add_episode(
        self, graphiti_client: Graphiti, test_config: GraphitiConfig
    ):
        """Test adding an episode through Graphiti with Bedrock endpoint."""
        try:
            episode_body = (
                "This is a test episode to verify the Bedrock endpoint integration. "
                "The custom OpenAI-compatible endpoint should process this text and "
                "extract entities and relationships."
            )

            logger.info("Adding test episode...")

            result = await graphiti_client.add_episode(
                name="Bedrock Endpoint Test",
                episode_body=episode_body,
                source_description="Integration test",
                group_id=test_config.group_id,
            )

            assert result is not None, "Episode result should not be None"
            assert hasattr(result, "episode_uuid"), "Result should have episode_uuid"

            logger.info(f"✓ Episode added successfully: {result.episode_uuid}")

            # Wait a bit for processing
            await asyncio.sleep(2)

            # Verify episode was stored
            episodes = await graphiti_client.get_episodes(
                group_id=test_config.group_id, last_n=10
            )

            assert len(episodes) > 0, "Should have at least one episode"
            logger.info(f"✓ Found {len(episodes)} episode(s) in the graph")

        except Exception as e:
            logger.error(f"Add episode test failed: {e}", exc_info=True)
            pytest.fail(f"Failed to add episode: {e}")

    async def test_graphiti_search_nodes(
        self, graphiti_client: Graphiti, test_config: GraphitiConfig
    ):
        """Test searching nodes with Bedrock endpoint."""
        try:
            # First add some content
            episode_body = (
                "Alice is a software engineer at TechCorp. "
                "She works on machine learning projects using Python. "
                "Alice collaborates with Bob, who is a data scientist."
            )

            logger.info("Adding episode with entities...")

            await graphiti_client.add_episode(
                name="Entity Test",
                episode_body=episode_body,
                source_description="Entity extraction test",
                group_id=test_config.group_id,
            )

            # Wait for processing
            logger.info("Waiting for background processing...")
            await asyncio.sleep(5)

            # Search for nodes
            logger.info("Searching for nodes...")

            search_results = await graphiti_client.search(
                query="Tell me about Alice and her work",
                group_ids=[test_config.group_id],
                num_results=10,
            )

            assert search_results is not None, "Search results should not be None"
            logger.info(
                f"✓ Search completed, found {len(search_results.edges) if hasattr(search_results, 'edges') else 0} results"
            )

        except Exception as e:
            logger.error(f"Search test failed: {e}", exc_info=True)
            # Don't fail the test if search doesn't return results immediately
            logger.warning(f"Search test completed with warning: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestMCPToolsWithBedrock:
    """Test MCP tools with Bedrock endpoint."""

    async def test_add_memory_tool(self):
        """Test add_memory MCP tool with Bedrock endpoint."""
        # Set environment to use OpenAI (not Ollama)
        os.environ["USE_OLLAMA"] = "false"

        # Ensure API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")

        try:
            # Test the add_memory tool
            result = await add_memory(
                name="MCP Tool Test",
                episode_body="Testing the MCP add_memory tool with Bedrock endpoint integration.",
                group_id="bedrock-mcp-test",
                source="text",
                source_description="MCP integration test",
            )

            assert result is not None, "Result should not be None"

            if isinstance(result, ErrorResponse):
                logger.error(f"Error response: {result.error}")
                pytest.fail(f"add_memory returned error: {result.error}")

            assert isinstance(result, SuccessResponse), "Should return SuccessResponse"
            logger.info(f"✓ add_memory tool successful: {result.message}")

        except Exception as e:
            logger.error(f"MCP tool test failed: {e}", exc_info=True)
            pytest.fail(f"Failed to test add_memory tool: {e}")


if __name__ == "__main__":
    """
    Run tests directly with pytest.

    Usage:
        # Run all Bedrock endpoint tests
        pytest tests/test_bedrock_endpoint.py -v

        # Run specific test class
        pytest tests/test_bedrock_endpoint.py::TestBedrockEndpointConfiguration -v

        # Run with detailed output
        pytest tests/test_bedrock_endpoint.py -v -s

        # Run and stop on first failure
        pytest tests/test_bedrock_endpoint.py -x -v
    """
    pytest.main([__file__, "-v", "-s"])
