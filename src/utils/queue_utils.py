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
        while queue_workers.get(group_id, False):
            # Check if queue exists before accessing it
            if group_id not in episode_queues:
                logger.warning(f"Queue for group_id {group_id} no longer exists, stopping worker")
                break

            try:
                # Get the next episode processing function from the queue with timeout
                # This allows for proper cancellation handling
                try:
                    process_func = await asyncio.wait_for(
                        episode_queues[group_id].get(),
                        timeout=1.0  # 1 second timeout to allow cancellation checks
                    )
                except asyncio.TimeoutError:
                    # No work available, continue loop to check for cancellation
                    continue

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
            except KeyError:
                logger.warning(f"Queue for group_id {group_id} was removed during processing")
                break
    except asyncio.CancelledError:
        logger.info(f"Episode queue worker for group_id {group_id} was cancelled")
        raise  # Re-raise CancelledError to ensure proper task cancellation
    except Exception as e:
        logger.error(
            f"Unexpected error in queue worker for group_id {group_id}: {str(e)}"
        )
    finally:
        queue_workers[group_id] = False
        logger.info(f"Stopped episode queue worker for group_id: {group_id}")
