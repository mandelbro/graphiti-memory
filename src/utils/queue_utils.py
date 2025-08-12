"""Queue management utilities for Graphiti MCP Server.

This module contains functionality for managing episode processing queues,
ensuring that episodes for each group are processed sequentially to avoid
race conditions in the graph database.
"""

import asyncio
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

# Dictionary to store queues for each group_id
# Each queue is a list of tasks to be processed sequentially
episode_queues: dict[str, asyncio.Queue] = {}

# Dictionary to track if a worker is running for each group_id
queue_workers: dict[str, bool] = {}


async def process_episode_queue(group_id: str):
    """Process episodes for a specific group_id sequentially.

    This function runs as a long-lived task that processes episodes
    from the queue one at a time.

    Args:
        group_id: The group identifier for which to process episodes
    """
    global queue_workers

    logger.info(f"Starting episode queue worker for group_id: {group_id}")
    queue_workers[group_id] = True

    try:
        while True:
            # Get the next episode processing function from the queue
            # This will wait if the queue is empty
            process_func = await episode_queues[group_id].get()

            try:
                # Process the episode
                await process_func()
            except Exception as e:
                logger.error(
                    f"Error processing queued episode for group_id {group_id}: {str(e)}"
                )
            finally:
                # Mark the task as done regardless of success/failure
                episode_queues[group_id].task_done()
    except asyncio.CancelledError:
        logger.info(f"Episode queue worker for group_id {group_id} was cancelled")
    except Exception as e:
        logger.error(
            f"Unexpected error in queue worker for group_id {group_id}: {str(e)}"
        )
    finally:
        queue_workers[group_id] = False
        logger.info(f"Stopped episode queue worker for group_id: {group_id}")
