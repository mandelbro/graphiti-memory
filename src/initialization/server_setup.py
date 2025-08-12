"""
Server setup and configuration module.

Contains functions for parsing CLI arguments, initializing server configuration,
and running the MCP server with the appropriate transport.
"""

import argparse
import logging
import os
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from ..utils.initialization_state import initialization_manager
from .graphiti_client import initialize_graphiti

if TYPE_CHECKING:
    from src.config import GraphitiConfig, MCPConfig

# Default model configurations
DEFAULT_LLM_MODEL = "deepseek-r1:7b"
SMALL_LLM_MODEL = "deepseek-r1:7b"
DEFAULT_EMBEDDER_MODEL = "nomic-embed-text"

logger = logging.getLogger(__name__)


async def initialize_server(
    mcp: FastMCP,
) -> tuple["MCPConfig", "GraphitiConfig", object]:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    # Import config classes here to avoid circular imports
    from src.config import GraphitiConfig, MCPConfig

    parser = argparse.ArgumentParser(
        description="Run the Graphiti MCP server with optional LLM client"
    )
    parser.add_argument(
        "--group-id",
        help="Namespace for the graph. This is an arbitrary string used to organize related data. "
        "If not provided, a random UUID will be generated.",
    )
    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default="sse",
        help="Transport to use for communication with the client. (default: sse)",
    )
    parser.add_argument(
        "--model",
        help=f"Model name to use with the LLM client. (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--small-model",
        help=f"Small model name to use with the LLM client. (default: {SMALL_LLM_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature setting for the LLM (0.0-2.0). Lower values make output more deterministic. (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens for LLM responses (default: 8192)",
    )
    parser.add_argument(
        "--destroy-graph", action="store_true", help="Destroy all Graphiti graphs"
    )
    parser.add_argument(
        "--use-custom-entities",
        action="store_true",
        help="Enable entity extraction using the predefined ENTITY_TYPES",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MCP_SERVER_HOST"),
        help="Host to bind the MCP server to (default: MCP_SERVER_HOST environment variable)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MCP_SERVER_PORT", "8020")),
        help="Port to bind the MCP server to (default: MCP_SERVER_PORT environment variable or 8020)",
    )
    # Ollama configuration arguments
    parser.add_argument(
        "--use-ollama",
        type=lambda x: x.lower() == "true",
        help="Use Ollama for LLM and embeddings (default: true)",
    )
    parser.add_argument(
        "--ollama-base-url",
        help="Ollama base URL (default: http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--ollama-llm-model",
        help=f"Ollama LLM model name (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--ollama-embedding-model",
        help=f"Ollama embedding model name (default: {DEFAULT_EMBEDDER_MODEL})",
    )
    parser.add_argument(
        "--ollama-embedding-dim",
        type=int,
        help="Ollama embedding dimension (default: 768)",
    )

    args = parser.parse_args()

    # Build configuration from CLI arguments and environment variables
    config = GraphitiConfig.from_cli_and_env(args)

    # Log the group ID configuration
    if args.group_id:
        logger.info(f"Using provided group_id: {config.group_id}")
    else:
        logger.info(f"Generated random group_id: {config.group_id}")

    # Log entity extraction configuration
    if config.use_custom_entities:
        logger.info("Entity extraction enabled using predefined ENTITY_TYPES")
    else:
        logger.info("Entity extraction disabled (no custom entities will be used)")

    # Log LLM configuration
    if config.llm.use_ollama:
        logger.info(f"Using Ollama LLM: {config.llm.ollama_llm_model}")
        logger.info(f"Ollama base URL: {config.llm.ollama_base_url}")
        logger.info(f"LLM temperature: {config.llm.temperature}")
        logger.info(f"LLM max tokens: {config.llm.max_tokens}")
    else:
        logger.info(f"Using OpenAI/Azure OpenAI LLM: {config.llm.model}")
        logger.info(f"LLM temperature: {config.llm.temperature}")
        logger.info(f"LLM max tokens: {config.llm.max_tokens}")

    # Log embedder configuration
    if config.embedder.use_ollama:
        logger.info(f"Using Ollama embedder: {config.embedder.ollama_embedding_model}")
        logger.info(f"Embedding dimension: {config.embedder.ollama_embedding_dim}")
    else:
        logger.info(f"Using OpenAI/Azure OpenAI embedder: {config.embedder.model}")

    # Initialize Graphiti
    graphiti_client = await initialize_graphiti(config)

    if args.host:
        logger.info(f"Setting MCP server host to: {args.host}")
        # Set MCP server host from CLI or env
        mcp.settings.host = args.host

    if args.port:
        logger.info(f"Setting MCP server port to: {args.port}")
        # Set MCP server port from CLI or env
        mcp.settings.port = args.port

    # Return MCP configuration and other objects
    return MCPConfig.from_cli(args), config, graphiti_client


async def run_mcp_server(
    mcp: FastMCP,
) -> None:
    """Run the MCP server in the current event loop."""
    try:
        # Mark initialization as started
        await initialization_manager.start_initialization()

        # Initialize the server
        mcp_config, config, graphiti_client = await initialize_server(mcp)

        # Mark initialization as completed
        await initialization_manager.complete_initialization()

        # Run the server with stdio transport for MCP in the same event loop
        logger.info(f"Starting MCP server with transport: {mcp_config.transport}")
        if mcp_config.transport == "stdio":
            await mcp.run_stdio_async()
        elif mcp_config.transport == "sse":
            logger.info(
                f"Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}"
            )
            await mcp.run_sse_async()
    except Exception as e:
        # Mark initialization as failed
        await initialization_manager.fail_initialization(str(e))
        logger.error(f"Failed to start MCP server: {str(e)}")
        raise
