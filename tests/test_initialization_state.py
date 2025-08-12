"""
Tests for initialization state management.

This test module verifies that the initialization state manager properly
tracks server startup progress and handles requests during initialization.
"""

import asyncio
from unittest.mock import patch

import pytest

from src.utils.initialization_state import InitializationManager, InitializationState


class TestInitializationManager:
    """Test suite for InitializationManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh InitializationManager for each test."""
        return InitializationManager()

    def test_initial_state(self, manager):
        """Test that manager starts in correct initial state."""
        assert manager.state == InitializationState.NOT_STARTED
        assert not manager.is_ready
        assert not manager.is_failed
        assert manager.initialization_time is None

    @pytest.mark.asyncio
    async def test_initialization_lifecycle(self, manager):
        """Test complete initialization lifecycle."""
        # Start initialization
        await manager.start_initialization()

        assert manager.state == InitializationState.IN_PROGRESS
        assert not manager.is_ready
        assert not manager.is_failed

        # Wait a small amount to allow time tracking
        await asyncio.sleep(0.01)

        # Complete initialization
        await manager.complete_initialization()

        assert manager.state == InitializationState.COMPLETED
        assert manager.is_ready
        assert not manager.is_failed
        assert manager.initialization_time is not None
        assert manager.initialization_time > 0

    @pytest.mark.asyncio
    async def test_initialization_failure(self, manager):
        """Test initialization failure handling."""
        error_message = "Database connection failed"

        # Start initialization
        await manager.start_initialization()

        # Fail initialization
        await manager.fail_initialization(error_message)

        assert manager.state == InitializationState.FAILED
        assert not manager.is_ready
        assert manager.is_failed

        status_info = manager.get_status_info()
        assert status_info["error_message"] == error_message

    @pytest.mark.asyncio
    async def test_double_start_warning(self, manager):
        """Test that starting initialization twice logs a warning."""
        # Start initialization
        await manager.start_initialization()
        assert manager.state == InitializationState.IN_PROGRESS

        # Try to start again - should warn but not change state
        with patch("src.utils.initialization_state.logger") as mock_logger:
            await manager.start_initialization()
            mock_logger.warning.assert_called_once()

        assert manager.state == InitializationState.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_complete_without_start_warning(self, manager):
        """Test that completing without starting logs a warning."""
        with patch("src.utils.initialization_state.logger") as mock_logger:
            await manager.complete_initialization()
            mock_logger.warning.assert_called_once()

        assert manager.state == InitializationState.NOT_STARTED

    def test_get_not_ready_response_not_started(self, manager):
        """Test get_not_ready_response for NOT_STARTED state."""
        response = manager.get_not_ready_response()

        assert "not started" in response["error"].lower()
        assert response["state"] == "not_started"
        assert "retry_advice" in response

    @pytest.mark.asyncio
    async def test_get_not_ready_response_in_progress(self, manager):
        """Test get_not_ready_response for IN_PROGRESS state."""
        await manager.start_initialization()

        response = manager.get_not_ready_response()

        assert "initializing" in response["error"].lower()
        assert response["state"] == "in_progress"
        assert "elapsed_seconds" in response
        assert "retry_advice" in response

    @pytest.mark.asyncio
    async def test_get_not_ready_response_failed(self, manager):
        """Test get_not_ready_response for FAILED state."""
        error_message = "Test failure"
        await manager.start_initialization()
        await manager.fail_initialization(error_message)

        response = manager.get_not_ready_response()

        assert "failed" in response["error"].lower()
        assert response["state"] == "failed"
        assert response["failure_reason"] == error_message
        assert "retry_advice" in response

    @pytest.mark.asyncio
    async def test_status_info_comprehensive(self, manager):
        """Test that status info includes all relevant information."""
        # Start initialization
        await manager.start_initialization()
        await asyncio.sleep(0.01)  # Small delay for timing

        # Get status while in progress
        status_info = manager.get_status_info()
        assert status_info["state"] == "in_progress"
        assert status_info["is_ready"] is False
        assert status_info["is_failed"] is False
        assert "start_time" in status_info
        assert "elapsed_seconds" in status_info

        # Complete initialization
        await manager.complete_initialization()

        # Get final status
        final_status = manager.get_status_info()
        assert final_status["state"] == "completed"
        assert final_status["is_ready"] is True
        assert final_status["is_failed"] is False
        assert "start_time" in final_status
        assert "completion_time" in final_status
        assert "initialization_duration_seconds" in final_status
        assert final_status["initialization_duration_seconds"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_access_thread_safety(self, manager):
        """Test that concurrent access to manager is thread-safe."""

        async def start_task():
            await manager.start_initialization()

        async def complete_task():
            # Wait a bit to ensure start happens first
            await asyncio.sleep(0.01)
            await manager.complete_initialization()

        # Run both tasks concurrently
        await asyncio.gather(start_task(), complete_task())

        assert manager.state == InitializationState.COMPLETED
        assert manager.is_ready


class TestInitializationIntegration:
    """Integration tests for initialization state with server components."""

    @pytest.mark.asyncio
    async def test_error_response_model_supports_details(self):
        """Test that ErrorResponse model supports details field."""
        from src.models.response_models import ErrorResponse

        # Test creating ErrorResponse with details
        error = ErrorResponse(
            error="Test error",
            details={"state": "not_started", "retry_advice": "Wait for initialization"},
        )

        assert error.error == "Test error"
        assert error.details is not None
        assert error.details["state"] == "not_started"
        assert "retry_advice" in error.details

    @pytest.mark.asyncio
    async def test_status_response_model_supports_details(self):
        """Test that StatusResponse model supports details field."""
        from src.models.response_models import StatusResponse

        # Test creating StatusResponse with details
        status = StatusResponse(
            status="error",
            message="Not ready",
            details={"state": "in_progress", "elapsed_seconds": 5.2},
        )

        assert status.status == "error"
        assert status.message == "Not ready"
        assert status.details is not None
        assert status.details["state"] == "in_progress"
        assert status.details["elapsed_seconds"] == 5.2

    def test_initialization_manager_get_not_ready_response_structure(self):
        """Test that get_not_ready_response returns expected structure for tools."""
        manager = InitializationManager()

        response = manager.get_not_ready_response()

        # Verify the response has the expected structure for tool integration
        assert "error" in response
        assert "state" in response
        assert "retry_advice" in response
        assert isinstance(response["error"], str)
        assert isinstance(response["state"], str)
        assert isinstance(response["retry_advice"], str)
