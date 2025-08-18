"""
Graphiti client initialization module.

Contains functions for initializing and configuring the Graphiti client
with proper validation and logging.
"""

import logging
import os
import re
from typing import TYPE_CHECKING

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

if TYPE_CHECKING:
    from src.config import GraphitiConfig

# Semaphore limit for concurrent Graphiti operations.
# Decrease this if you're experiencing 429 rate limit errors from your LLM provider.
# Increase if you have high rate limits.
SEMAPHORE_LIMIT = int(os.getenv("SEMAPHORE_LIMIT", 10))

logger = logging.getLogger(__name__)


def _detect_ollama_configuration(config: "GraphitiConfig") -> bool:
    """
    Detect if the configuration specifies Ollama usage.

    Uses multiple detection methods for robust identification:
    1. Explicit use_ollama flag (primary method)
    2. Base URL contains Ollama default port 11434 (fallback method)
    3. Various URL format handling (localhost, 127.0.0.1, etc.)

    Args:
        config: The GraphitiConfig instance to check

    Returns:
        bool: True if Ollama configuration is detected, False otherwise
    """
    # Primary detection: explicit use_ollama flag
    if hasattr(config.llm, 'use_ollama') and config.llm.use_ollama:
        logger.debug("Ollama detected via explicit use_ollama flag")
        return True

    # Check for port 11434 in various URL formats
    ollama_patterns = [
        r':11434',  # Direct port match
        r'localhost:11434',  # Localhost with port
        r'127\.0\.0\.1:11434',  # Loopback with port
        r'0\.0\.0\.0:11434',  # All interfaces with port
    ]

    # Fallback detection: check ollama_base_url for Ollama port 11434
    if hasattr(config.llm, 'ollama_base_url') and config.llm.ollama_base_url:
        base_url = config.llm.ollama_base_url.lower()

        for pattern in ollama_patterns:
            if re.search(pattern, base_url):
                logger.debug(f"Ollama detected via ollama_base_url pattern: {pattern}")
                return True

    # Also check standard ollama_base_url attribute for Ollama port 11434
    if hasattr(config.llm, 'ollama_base_url') and config.llm.ollama_base_url:
        base_url = config.llm.ollama_base_url.lower()

        for pattern in ollama_patterns:
            if re.search(pattern, base_url):
                logger.debug(f"Ollama detected via ollama_base_url pattern: {pattern}")
                return True

    logger.debug("No Ollama configuration detected")
    return False


def _create_dummy_cross_encoder():
    """Create a dummy cross encoder that returns documents in original order."""
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    class DummyCrossEncoder(CrossEncoderClient):
        """A dummy cross encoder that returns documents in original order."""

        async def rank(
            self,
            query: str,
            documents: list[str],
        ) -> list[tuple[str, float]]:
            # Return documents in original order with dummy scores
            return [(doc, 1.0 - i * 0.01) for i, doc in enumerate(documents)]

    return DummyCrossEncoder()


def _validate_ollama_configuration(config: "GraphitiConfig") -> None:
    """
    Validate Ollama configuration parameters.

    Args:
        config: The GraphitiConfig instance to validate

    Raises:
        ValueError: If required Ollama configuration is missing or invalid
    """
    if not config.llm.ollama_llm_model or not config.llm.ollama_llm_model.strip():
        raise ValueError(
            "OLLAMA_LLM_MODEL must be set when using Ollama for LLM. "
            "Please specify a valid Ollama model name (e.g., 'llama3.2:3b')"
        )

    # Validate base URL format if provided
    if hasattr(config.llm, 'ollama_base_url') and config.llm.ollama_base_url:
        base_url = config.llm.ollama_base_url
        if not (base_url.startswith('http://') or base_url.startswith('https://')):
            raise ValueError(
                f"Invalid Ollama base URL format: {base_url}. "
                "URL must start with 'http://' or 'https://'"
            )

    logger.info(f"Validated Ollama LLM configuration: model={config.llm.ollama_llm_model}")


def _create_enhanced_ollama_client(config: "GraphitiConfig"):
    """
    Create enhanced Ollama client with proper configuration and response converter integration.

    Args:
        config: The GraphitiConfig instance

    Returns:
        OllamaClient: Configured Ollama client with enhanced features
    """
    try:
        # Import here to avoid circular imports
        from graphiti_core.llm_client.config import LLMConfig

        from src.ollama_client import OllamaClient

        # Create LLM configuration for the Ollama client
        llm_client_config = LLMConfig(
            api_key="abc",  # Ollama doesn't require a real API key
            model=config.llm.ollama_llm_model,
            small_model=config.llm.ollama_llm_model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            base_url=getattr(config.llm, 'ollama_base_url', "http://localhost:11434/v1"),
        )

        # Create enhanced Ollama client with model parameters and response converter
        ollama_client = OllamaClient(
            config=llm_client_config,
            model_parameters=getattr(config.llm, 'ollama_model_parameters', {})
        )

        logger.info(
            f"Created enhanced OllamaClient: model={config.llm.ollama_llm_model}, "
            f"base_url={llm_client_config.base_url}, "
            f"parameters={getattr(config.llm, 'ollama_model_parameters', {})}"
        )

        return ollama_client

    except Exception as e:
        logger.error(f"Failed to create enhanced Ollama client: {e}")
        raise ValueError(f"Failed to create Ollama client: {e}") from e


async def create_graphiti_client(config: "GraphitiConfig") -> Graphiti:
    """
    Create Graphiti client with enhanced configuration override handling.

    This function implements the configuration override pattern to ensure all
    Graphiti Core LLM operations use our enhanced clients instead of creating
    their own client instances.

    Args:
        config: The GraphitiConfig instance

    Returns:
        Graphiti: Configured Graphiti client
    """
    try:
        # Detect and handle Ollama configuration
        if _detect_ollama_configuration(config):
            logger.info("Ollama configuration detected - using enhanced OllamaClient")

            # Validate Ollama configuration
            _validate_ollama_configuration(config)

            # Create enhanced Ollama client with schema mapping
            llm_client = _create_enhanced_ollama_client(config)

            # Create embedder client
            embedder_client = config.embedder.create_client()

            # Use shared dummy cross encoder
            dummy_cross_encoder = _create_dummy_cross_encoder()

            # Override Graphiti's internal LLM client creation by passing our enhanced client
            graphiti_client = Graphiti(
                uri=config.neo4j.uri,
                user=config.neo4j.user,
                password=config.neo4j.password,
                llm_client=llm_client,  # Force use of our enhanced client
                embedder=embedder_client,
                cross_encoder=dummy_cross_encoder,
                max_coroutines=SEMAPHORE_LIMIT,
            )

            logger.info("Graphiti client created with enhanced Ollama client override")

        else:
            # Standard initialization for other LLM providers
            logger.info("Using standard LLM provider configuration")
            llm_client = config.llm.create_client()

            if not llm_client and config.use_custom_entities:
                raise ValueError(
                    "LLM client is required when custom entities are enabled. "
                    "Please configure OPENAI_API_KEY or Ollama settings."
                )

            embedder_client = config.embedder.create_client()

            # Use shared dummy cross encoder
            dummy_cross_encoder = _create_dummy_cross_encoder()

            graphiti_client = Graphiti(
                uri=config.neo4j.uri,
                user=config.neo4j.user,
                password=config.neo4j.password,
                llm_client=llm_client,
                embedder=embedder_client,
                cross_encoder=dummy_cross_encoder,
                max_coroutines=SEMAPHORE_LIMIT,
            )

        return graphiti_client

    except Exception as e:
        logger.error(f"Failed to create Graphiti client with configuration override: {e}")
        raise


async def initialize_graphiti(
    config: "GraphitiConfig",
) -> Graphiti:
    """Initialize the Graphiti client with the configured settings."""
    # Import tools here to avoid circular imports
    from src.tools import management_tools, memory_tools
    from src.tools import search_tools as search_tools_module

    try:
        # Validate embedder configuration if using Ollama
        if config.embedder.use_ollama:
            if (
                not config.embedder.ollama_embedding_model
                or not config.embedder.ollama_embedding_model.strip()
            ):
                raise ValueError(
                    "OLLAMA_EMBEDDING_MODEL must be set when using Ollama for embeddings"
                )
            logger.info(
                f"Validated Ollama embedding model: {config.embedder.ollama_embedding_model}"
            )

        # Validate Neo4j configuration
        if not config.neo4j.uri or not config.neo4j.user or not config.neo4j.password:
            raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")

        # Create Graphiti client with enhanced configuration override
        graphiti_client = await create_graphiti_client(config)

        # Destroy graph if requested
        if config.destroy_graph:
            logger.info("Destroying graph...")
            assert graphiti_client is not None
            await clear_data(graphiti_client.driver)

        # Initialize the graph database with Graphiti's indices
        assert graphiti_client is not None
        await graphiti_client.build_indices_and_constraints()
        logger.info("Graphiti client initialized successfully")

        # Log configuration details for transparency
        has_llm = graphiti_client.llm_client is not None
        has_embedder = graphiti_client.embedder is not None

        if has_llm:
            if _detect_ollama_configuration(config):
                logger.info(f"Using Ollama LLM model: {config.llm.ollama_llm_model}")
            else:
                logger.info(f"Using OpenAI/Azure OpenAI model: {config.llm.model}")
            logger.info(f"Using temperature: {config.llm.temperature}")
        else:
            logger.info("No LLM client configured - entity extraction will be limited")

        if has_embedder:
            if config.embedder.use_ollama:
                logger.info(
                    f"Using Ollama embedding model: {config.embedder.ollama_embedding_model}"
                )
            else:
                logger.info(
                    f"Using OpenAI/Azure OpenAI embedding model: {config.embedder.model}"
                )
        else:
            logger.info(
                "No embedder client configured - embeddings will not be available"
            )

        logger.info(f"Using group_id: {config.group_id}")
        logger.info(
            f"Custom entity extraction: {'enabled' if config.use_custom_entities else 'disabled'}"
        )
        logger.info(f"Using concurrency limit: {SEMAPHORE_LIMIT}")

        # Set globals for tool modules
        memory_tools.set_globals(graphiti_client, config)
        search_tools_module.set_globals(graphiti_client, config)
        management_tools.set_globals(graphiti_client, config)

        return graphiti_client

    except Exception as e:
        logger.error(f"Failed to initialize Graphiti: {str(e)}")
        raise
