#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import asyncio
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Import configuration classes
from src.config import (
    GraphitiConfig,
    GraphitiEmbedderConfig,
    GraphitiLLMConfig,
    Neo4jConfig,
)

# Import initialization functions
from src.initialization import run_mcp_server

# Import model definitions from the models package
from src.models import (
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeSearchResponse,
    Preference,
    Procedure,
    Requirement,
    StatusResponse,
    SuccessResponse,
)

# Import tools from the tools package
from src.tools import (
    add_memory as tools_add_memory,
)
from src.tools import (
    clear_graph as tools_clear_graph,
)
from src.tools import (
    delete_entity_edge as tools_delete_entity_edge,
)
from src.tools import (
    delete_episode as tools_delete_episode,
)
from src.tools import (
    get_entity_edge as tools_get_entity_edge,
)
from src.tools import (
    get_episodes as tools_get_episodes,
)
from src.tools import (
    get_status as tools_get_status,
)
from src.tools import (
    search_memory_facts as tools_search_memory_facts,
)
from src.tools import (
    search_memory_nodes as tools_search_memory_nodes,
)

load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create global config instance - will be properly initialized later
config = GraphitiConfig()

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph that transforms information into a richly connected network of episodes (content), nodes (entities), and facts (relationships). It supports text, JSON, and message formats with temporal metadata tracking.

Key capabilities:
1. Add episodes with add_memory tool
2. Search nodes with search_memory_nodes
3. Find facts with search_memory_facts
4. Manage data with delete/clear operations
5. Retrieve specific entities/episodes by UUID

Organized by group_id for separate knowledge domains. Requires proper database configuration and API keys.
"""

# MCP server instance
mcp = FastMCP(
    "Graphiti Agent Memory",
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)

# Set default port from environment variable if available
default_port = int(os.environ.get("MCP_SERVER_PORT", "8020"))
mcp.settings.port = default_port


# Register MCP tools as simple wrappers around imported tool functions
@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = "text",
    source_description: str = "",
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph."""
    return await tools_add_memory(
        name, episode_body, group_id, source, source_description, uuid
    )


@mcp.tool()
async def search_memory_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    center_node_uuid: str | None = None,
    entity: str = "",
) -> NodeSearchResponse | ErrorResponse:
    return await tools_search_memory_nodes(
        query, group_ids, max_nodes, center_node_uuid, entity
    )


@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    return await tools_search_memory_facts(
        query, group_ids, max_facts, center_node_uuid
    )


@mcp.tool()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    return await tools_delete_entity_edge(uuid)


@mcp.tool()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    return await tools_delete_episode(uuid)


@mcp.tool()
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    return await tools_get_entity_edge(uuid)


@mcp.tool()
async def get_episodes(
    group_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    return await tools_get_episodes(group_id, last_n)


@mcp.tool()
async def clear_graph() -> SuccessResponse | ErrorResponse:
    return await tools_clear_graph()


@mcp.resource("http://graphiti/status")
async def get_status() -> StatusResponse:
    return await tools_get_status()


def main():
    try:
        asyncio.run(run_mcp_server(mcp))
    except Exception as e:
        logger.error(f"Error initializing Graphiti MCP server: {str(e)}")
        raise


__all__ = [
    "mcp",  # MCP server instance
    # Tool functions
    "search_memory_nodes",
    "search_memory_facts",
    "add_memory",
    "get_episodes",
    "delete_entity_edge",
    "delete_episode",
    "get_entity_edge",
    "clear_graph",
    "get_status",
    # Configuration classes
    "GraphitiConfig",
    "GraphitiLLMConfig",
    "GraphitiEmbedderConfig",
    "Neo4jConfig",
    # Response models
    "ErrorResponse",
    "SuccessResponse",
    "NodeSearchResponse",
    "FactSearchResponse",
    "EpisodeSearchResponse",
    "StatusResponse",
    # Entity types
    "Preference",
    "Procedure",
    "Requirement",
]


if __name__ == "__main__":
    main()
