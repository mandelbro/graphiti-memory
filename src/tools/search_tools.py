"""
Search-related MCP tools for Graphiti.

This module contains MCP tool logic related to search operations:
- Searching nodes in memory
- Searching facts (edges) in memory
"""

import logging
from typing import cast

from graphiti_core import Graphiti
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters

from src.config import GraphitiConfig
from src.models import (
    ErrorResponse,
    FactSearchResponse,
    NodeResult,
    NodeSearchResponse,
)
from src.utils import format_fact_result
from src.utils.initialization_state import initialization_manager

# Get logger for this module
logger = logging.getLogger(__name__)

# These will be imported/set from the main module
graphiti_client: Graphiti | None = None
config: GraphitiConfig | None = None


def set_globals(client: Graphiti | None, server_config: GraphitiConfig) -> None:
    """Set global variables from the main module."""
    global graphiti_client, config
    graphiti_client = client
    config = server_config


async def search_memory_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    center_node_uuid: str | None = None,
    entity: str = "",
) -> NodeSearchResponse | ErrorResponse:
    """Search the graph memory for relevant node summaries.

    These contain a summary of all of a node's relationships with other nodes.
    """
    global graphiti_client, config

    # Check initialization state first
    if not initialization_manager.is_ready:
        not_ready_info = initialization_manager.get_not_ready_response()
        return ErrorResponse(error=not_ready_info["error"], details=not_ready_info)

    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    try:
        # Use the provided group_ids or fall back to the default from config if none provided
        if group_ids is not None:
            effective_group_ids = group_ids
        else:
            effective_group_ids = (
                [config.group_id] if (config and config.group_id) else []
            )

        # Configure the search
        if center_node_uuid is not None:
            search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
        else:
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_nodes

        filters = SearchFilters()
        if entity != "":
            filters.node_labels = [entity]

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Perform the search using the _search method
        search_results = await client._search(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=filters,
        )

        if not search_results.nodes:
            return NodeSearchResponse(message="No relevant nodes found", nodes=[])

        # Format the node results
        formatted_nodes: list[NodeResult] = [
            NodeResult(
                uuid=node.uuid,
                name=node.name,
                summary=getattr(node, "summary", ""),
                labels=getattr(node, "labels", []),
                group_id=node.group_id,
                created_at=node.created_at.isoformat(),
                attributes=getattr(node, "attributes", {}),
            )
            for node in search_results.nodes
        ]

        return NodeSearchResponse(
            message="Nodes retrieved successfully", nodes=formatted_nodes
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error searching nodes: {error_msg}")
        return ErrorResponse(error=f"Error searching nodes: {error_msg}")


async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the graph memory for relevant facts."""
    global graphiti_client, config

    # Check initialization state first
    if not initialization_manager.is_ready:
        not_ready_info = initialization_manager.get_not_ready_response()
        return ErrorResponse(error=not_ready_info["error"], details=not_ready_info)

    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    try:
        # Validate max_facts parameter
        if max_facts <= 0:
            return ErrorResponse(error="max_facts must be a positive integer")

        # Use the provided group_ids or fall back to the default from config if none provided
        if group_ids is not None:
            effective_group_ids = group_ids
        else:
            effective_group_ids = (
                [config.group_id] if (config and config.group_id) else []
            )

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        relevant_edges = await client.search(
            group_ids=effective_group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return FactSearchResponse(message="No relevant facts found", facts=[])

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return FactSearchResponse(message="Facts retrieved successfully", facts=facts)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error searching facts: {error_msg}")
        return ErrorResponse(error=f"Error searching facts: {error_msg}")
