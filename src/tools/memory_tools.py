"""
Memory management MCP tools for Graphiti.

This module contains MCP tools related to memory operations:
- Adding episodes to memory
- Deleting entity edges and episodes
- Clearing the entire graph
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import cast

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from src.config import GraphitiConfig
from src.models import ENTITY_TYPES, ErrorResponse, SuccessResponse
from src.utils import episode_queues, process_episode_queue, queue_workers
from src.utils.initialization_state import initialization_manager

# Get logger for this module
logger = logging.getLogger(__name__)

# These will be imported from the main module
graphiti_client: Graphiti | None = None
config: GraphitiConfig | None = None


def set_globals(client: Graphiti | None, server_config: GraphitiConfig) -> None:
    """Set global variables from the main module."""
    global graphiti_client, config
    graphiti_client = client
    config = server_config


async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = "text",
    source_description: str = "",
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode to persist to memory. When source='json', this must be a
                           properly escaped JSON string, not a raw Python dictionary. The JSON data will be
                           automatically processed to extract entities and relationships.
        group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default)
                               - 'json': For structured data
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode

    Examples:
        # Adding plain text content
        add_memory(
            name="Company News",
            episode_body="Acme Corp announced a new product line today.",
            source="text",
            source_description="news article",
            group_id="some_arbitrary_string"
        )

        # Adding structured JSON data
        # NOTE: episode_body must be a properly escaped JSON string. Note the triple backslashes
        add_memory(
            name="Customer Profile",
            episode_body="{\\\"company\\\": {\\\"name\\\": \\\"Acme Technologies\\\"}, \\\"products\\\": [{\\\"id\\\": \\\"P001\\\", \\\"name\\\": \\\"CloudSync\\\"}, {\\\"id\\\": \\\"P002\\\", \\\"name\\\": \\\"DataMiner\\\"}]}",
            source="json",
            source_description="CRM data"
        )

        # Adding message-style content
        add_memory(
            name="Customer Conversation",
            episode_body="user: What's your return policy?\nassistant: You can return items within 30 days.",
            source="message",
            source_description="chat transcript",
            group_id="some_arbitrary_string"
        )

    Notes:
        When using source='json':
        - The JSON must be a properly escaped string, not a raw Python dictionary
        - The JSON will be automatically processed to extract entities and relationships
        - Complex nested structures are supported (arrays, nested objects, mixed data types), but keep nesting to a minimum
        - Entities will be created from appropriate JSON properties
        - Relationships between entities will be established based on the JSON structure
    """
    global graphiti_client, episode_queues, queue_workers

    # Check initialization state first
    if not initialization_manager.is_ready:
        not_ready_info = initialization_manager.get_not_ready_response()
        return ErrorResponse(error=not_ready_info["error"], details=not_ready_info)

    # Check initialization state first
    if not initialization_manager.is_ready:
        not_ready_info = initialization_manager.get_not_ready_response()
        return ErrorResponse(error=not_ready_info["error"], details=not_ready_info)

    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    if config is None:
        return ErrorResponse(error="Server configuration not initialized")

    try:
        # Map string source to EpisodeType enum
        source_type = EpisodeType.text
        if source.lower() == "message":
            source_type = EpisodeType.message
        elif source.lower() == "json":
            source_type = EpisodeType.json

        # We've already checked that graphiti_client is not None above
        # This also ensures config is not None since they are set together
        assert config is not None, "config should not be None here"

        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        # Cast group_id to str to satisfy type checker
        # The Graphiti client expects a str for group_id, not Optional[str]
        group_id_str = str(effective_group_id) if effective_group_id is not None else ""

        # We've already checked that graphiti_client is not None above
        # This assert statement helps type checkers understand that graphiti_client is defined
        assert graphiti_client is not None, "graphiti_client should not be None here"

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Define the episode processing function
        async def process_episode():
            try:
                logger.info(
                    f"Processing queued episode '{name}' for group_id: {group_id_str}"
                )
                # Config should be available here since it's set with graphiti_client
                assert config is not None, "config should not be None here"

                # Use all entity types if use_custom_entities is enabled, otherwise use empty dict
                entity_types = ENTITY_TYPES if config.use_custom_entities else {}

                await client.add_episode(
                    name=name,
                    episode_body=episode_body,
                    source=source_type,
                    source_description=source_description,
                    group_id=group_id_str,  # Using the string version of group_id
                    uuid=uuid,
                    reference_time=datetime.now(UTC),
                    entity_types=entity_types,
                )
                logger.info(f"Episode '{name}' added successfully")

                logger.info(f"Episode '{name}' processed successfully")
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error processing episode '{name}' for group_id {group_id_str}: {error_msg}"
                )

        # Initialize queue for this group_id if it doesn't exist
        if group_id_str not in episode_queues:
            episode_queues[group_id_str] = asyncio.Queue()

        # Add the episode processing function to the queue
        await episode_queues[group_id_str].put(process_episode)

        # Start a worker for this queue if one isn't already running
        if not queue_workers.get(group_id_str, False):
            asyncio.create_task(process_episode_queue(group_id_str))

        # Return immediately with a success message
        return SuccessResponse(
            message=f"Episode '{name}' queued for processing (position: {episode_queues[group_id_str].qsize()})"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error queuing episode task: {error_msg}")
        return ErrorResponse(error=f"Error queuing episode task: {error_msg}")


async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory.

    Args:
        uuid: UUID of the entity edge to delete
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

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return SuccessResponse(
            message=f"Entity edge with UUID {uuid} deleted successfully"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error deleting entity edge: {error_msg}")
        return ErrorResponse(error=f"Error deleting entity edge: {error_msg}")


async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory.

    Args:
        uuid: UUID of the episode to delete
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

        # Get the episodic node by UUID - EpisodicNode is already imported at the top
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return SuccessResponse(message=f"Episode with UUID {uuid} deleted successfully")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error deleting episode: {error_msg}")
        return ErrorResponse(error=f"Error deleting episode: {error_msg}")


async def clear_graph() -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph memory and rebuild indices."""
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

        # clear_data is already imported at the top
        await clear_data(client.driver)
        await client.build_indices_and_constraints()
        return SuccessResponse(message="Graph cleared successfully and indices rebuilt")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error clearing graph: {error_msg}")
        return ErrorResponse(error=f"Error clearing graph: {error_msg}")
