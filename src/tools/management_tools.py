"""
Management tools for entity and episode retrieval and status monitoring.
"""

import logging
from datetime import UTC, datetime
from typing import Any, cast

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge

# Import configuration types
from src.config import GraphitiConfig

# Import response models from the models package
from src.models.response_models import (
    EpisodeSearchResponse,
    ErrorResponse,
    StatusResponse,
)

# Import utilities from the utils package
from src.utils.formatting_utils import format_fact_result
from src.utils.initialization_state import initialization_manager

logger = logging.getLogger(__name__)

# Global variables (set by main server)
graphiti_client: Graphiti | None = None
config: GraphitiConfig | None = None


def set_globals(client: Graphiti | None, configuration: GraphitiConfig):
    """Set global variables for use by tool functions."""
    global graphiti_client, config
    graphiti_client = client
    config = configuration


async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    global graphiti_client

    # Check initialization state first
    if not initialization_manager.is_ready:
        not_ready_info = initialization_manager.get_not_ready_response()
        return ErrorResponse(error=not_ready_info["error"], details=not_ready_info)

    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting entity edge: {error_msg}")
        return ErrorResponse(error=f"Error getting entity edge: {error_msg}")


async def get_episodes(
    group_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    """Get the most recent memory episodes for a specific group.

    Args:
        group_id: ID of the group to retrieve episodes from. If not provided, uses the default group_id.
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    global graphiti_client

    # Check initialization state first
    if not initialization_manager.is_ready:
        not_ready_info = initialization_manager.get_not_ready_response()
        return ErrorResponse(error=not_ready_info["error"], details=not_ready_info)

    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    if config is None:
        return ErrorResponse(error="Configuration not initialized")

    try:
        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        if not isinstance(effective_group_id, str):
            return ErrorResponse(error="Group ID must be a string")

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        episodes = await client.retrieve_episodes(
            group_ids=[effective_group_id],
            last_n=last_n,
            reference_time=datetime.now(UTC),
        )

        if not episodes:
            return EpisodeSearchResponse(
                message=f"No episodes found for group {effective_group_id}", episodes=[]
            )

        # Use Pydantic's model_dump method for EpisodicNode serialization
        formatted_episodes = [
            # Use mode='json' to handle datetime serialization
            episode.model_dump(mode="json")
            for episode in episodes
        ]

        # Return the Python list directly - MCP will handle serialization
        return formatted_episodes
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting episodes: {error_msg}")
        return ErrorResponse(error=f"Error getting episodes: {error_msg}")


async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and Neo4j connection."""
    global graphiti_client

    # Check initialization state first
    if not initialization_manager.is_ready:
        not_ready_info = initialization_manager.get_not_ready_response()
        return StatusResponse(
            status="error", message=not_ready_info["error"], details=not_ready_info
        )

    if graphiti_client is None:
        return StatusResponse(status="error", message="Graphiti client not initialized")

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Test database connection
        await client.driver.client.verify_connectivity()  # type: ignore

        return StatusResponse(
            status="ok", message="Graphiti MCP server is running and connected to Neo4j"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error checking Neo4j connection: {error_msg}")
        return StatusResponse(
            status="error",
            message=f"Graphiti MCP server is running but Neo4j connection failed: {error_msg}",
        )
