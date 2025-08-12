"""
Initialization state management for the Graphiti MCP server.

This module provides a centralized way to track and manage the server's
initialization state to prevent race conditions between request handling
and server startup.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class InitializationState(Enum):
    """Enumeration of possible initialization states."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class InitializationManager:
    """
    Manages the initialization state of the Graphiti MCP server.

    This class provides:
    - State tracking for server initialization
    - Graceful handling of requests during startup
    - Proper error responses with initialization status
    """

    def __init__(self):
        self._state: InitializationState = InitializationState.NOT_STARTED
        self._start_time: float | None = None
        self._completion_time: float | None = None
        self._error_message: str | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> InitializationState:
        """Get the current initialization state."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """Check if the server is ready to handle requests."""
        return self._state == InitializationState.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if initialization has failed."""
        return self._state == InitializationState.FAILED

    @property
    def initialization_time(self) -> float | None:
        """Get the time taken for initialization in seconds."""
        if self._start_time and self._completion_time:
            return self._completion_time - self._start_time
        return None

    async def start_initialization(self) -> None:
        """Mark the beginning of initialization process."""
        async with self._lock:
            if self._state != InitializationState.NOT_STARTED:
                logger.warning(
                    f"Initialization already started, current state: {self._state.value}"
                )
                return

            self._state = InitializationState.IN_PROGRESS
            self._start_time = time.time()
            logger.info("Graphiti MCP server initialization started")

    async def complete_initialization(self) -> None:
        """Mark successful completion of initialization."""
        async with self._lock:
            if self._state != InitializationState.IN_PROGRESS:
                logger.warning(
                    f"Cannot complete initialization, current state: {self._state.value}"
                )
                return

            self._state = InitializationState.COMPLETED
            self._completion_time = time.time()
            init_time = self.initialization_time
            logger.info(
                f"Graphiti MCP server initialization completed successfully in {init_time:.2f}s"
            )

    async def fail_initialization(self, error_message: str) -> None:
        """Mark initialization as failed with an error message."""
        async with self._lock:
            self._state = InitializationState.FAILED
            self._error_message = error_message
            self._completion_time = time.time()
            logger.error(f"Graphiti MCP server initialization failed: {error_message}")

    def get_status_info(self) -> dict[str, Any]:
        """Get detailed status information for debugging."""
        status_info = {
            "state": self._state.value,
            "is_ready": self.is_ready,
            "is_failed": self.is_failed,
        }

        if self._start_time:
            status_info["start_time"] = self._start_time

        if self._completion_time:
            status_info["completion_time"] = self._completion_time

        if self.initialization_time:
            status_info["initialization_duration_seconds"] = self.initialization_time

        if self._error_message:
            status_info["error_message"] = self._error_message

        # Add current elapsed time if still in progress
        if self._state == InitializationState.IN_PROGRESS and self._start_time:
            status_info["elapsed_seconds"] = time.time() - self._start_time

        return status_info

    def get_not_ready_response(self) -> dict[str, Any]:
        """
        Get a standardized response for when the server is not ready.

        Returns appropriate error information based on current state.
        """
        if self._state == InitializationState.NOT_STARTED:
            return {
                "error": "Server initialization has not started",
                "state": self._state.value,
                "retry_advice": "Please wait for server to begin initialization",
            }
        elif self._state == InitializationState.IN_PROGRESS:
            elapsed = time.time() - self._start_time if self._start_time else 0
            return {
                "error": "Server is still initializing",
                "state": self._state.value,
                "elapsed_seconds": round(elapsed, 2),
                "retry_advice": "Please wait for initialization to complete and retry",
            }
        elif self._state == InitializationState.FAILED:
            return {
                "error": "Server initialization failed",
                "state": self._state.value,
                "failure_reason": self._error_message or "Unknown error",
                "retry_advice": "Server restart may be required",
            }
        else:
            return {
                "error": "Server is in an unexpected state",
                "state": self._state.value,
            }


# Global initialization manager instance
initialization_manager = InitializationManager()
