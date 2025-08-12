"""
Graphiti client initialization module.

Contains functions for initializing and configuring the Graphiti client
with proper validation and logging.
"""

import logging
import os
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


async def initialize_graphiti(
    config: "GraphitiConfig",
) -> Graphiti:
    """Initialize the Graphiti client with the configured settings."""
    # Import tools here to avoid circular imports
    from src.tools import management_tools, memory_tools
    from src.tools import search_tools as search_tools_module

    try:
        # Validate Ollama configuration if using Ollama
        if config.llm.use_ollama:
            if (
                not config.llm.ollama_llm_model
                or not config.llm.ollama_llm_model.strip()
            ):
                raise ValueError(
                    "OLLAMA_LLM_MODEL must be set when using Ollama for LLM"
                )
            logger.info(f"Validated Ollama LLM model: {config.llm.ollama_llm_model}")

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

        # Create LLM client if possible
        llm_client = config.llm.create_client()
        if not llm_client and config.use_custom_entities:
            # If custom entities are enabled, we must have an LLM client
            raise ValueError(
                "OPENAI_API_KEY must be set when custom entities are enabled"
            )

        # Validate Neo4j configuration
        if not config.neo4j.uri or not config.neo4j.user or not config.neo4j.password:
            raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")

        embedder_client = config.embedder.create_client()

        # Initialize Graphiti client
        graphiti_client = Graphiti(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            llm_client=llm_client,
            embedder=embedder_client,
            max_coroutines=SEMAPHORE_LIMIT,
        )

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
        if llm_client:
            if config.llm.use_ollama:
                logger.info(f"Using Ollama LLM model: {config.llm.ollama_llm_model}")
            else:
                logger.info(f"Using OpenAI/Azure OpenAI model: {config.llm.model}")
            logger.info(f"Using temperature: {config.llm.temperature}")
        else:
            logger.info("No LLM client configured - entity extraction will be limited")

        if embedder_client:
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
