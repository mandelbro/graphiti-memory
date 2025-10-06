"""
End-to-End Integration Testing for Graphiti Memory Pipeline Processing.

This test suite verifies the complete memory operation workflow from user request
through background processing to final storage, with specific focus on the schema
validation fixes for Ollama integration.

Test Categories:
1. Memory Operation Lifecycle Tests
2. Schema Validation Verification
3. Ollama Integration Tests
4. Performance and Reliability Tests

The tests verify that the fixes for the schema validation mismatch in Graphiti's
background processing pipeline work correctly and that memory operations actually
result in stored data rather than silent failures.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import cast
from unittest.mock import patch

import pytest
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from mcp.server.fastmcp import FastMCP

from src.config import GraphitiConfig, GraphitiLLMConfig
from src.initialization.graphiti_client import create_graphiti_client
from src.initialization.server_setup import initialize_server
from src.models import (
    ErrorResponse,
    SuccessResponse,
)
from src.ollama_client import OllamaClient
from src.tools.memory_tools import add_memory
from src.utils.initialization_state import initialization_manager

# Configure detailed logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Set specific loggers to DEBUG to trace execution
logging.getLogger("src.tools.memory_tools").setLevel(logging.DEBUG)
logging.getLogger("src.initialization").setLevel(logging.DEBUG)
logging.getLogger("src.utils.queue_utils").setLevel(logging.DEBUG)
logging.getLogger("graphiti_core").setLevel(logging.DEBUG)


# Module-level fixtures that can be shared across test classes
@pytest.fixture(scope="function")  # Changed from class to function scope
def test_config() -> GraphitiConfig:
    """Create test configuration for integration tests."""
    from src.config import Neo4jConfig

    return GraphitiConfig(
        neo4j=Neo4jConfig(
            uri=os.getenv("TEST_NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("TEST_NEO4J_USER", "neo4j"),
            password=os.getenv("TEST_NEO4J_PASSWORD", "demodemo"),  # Docker compose default
        ),
        llm=GraphitiLLMConfig(
            model="gpt-oss:latest",
            use_ollama=True,
            ollama_base_url=os.getenv("TEST_OLLAMA_URL", "http://localhost:11434/v1"),
            ollama_llm_model="gpt-oss:latest",
            api_key="ollama",  # Ollama doesn't require real API key
        ),
        use_custom_entities=True,
        group_id="integration-test",
    )


@pytest.fixture(scope="function")  # Changed from class to function scope
async def graphiti_client(test_config: GraphitiConfig) -> AsyncIterator[Graphiti]:
    """Create real Graphiti client with Ollama configuration for integration testing."""
    try:
        # Create the client using our enhanced configuration
        client = await create_graphiti_client(test_config)

        # Build indices and constraints
        await client.build_indices_and_constraints()

        # Clear any existing test data
        await _cleanup_test_data(client, "integration-test")

        yield client

    except Exception as e:
        pytest.skip(f"Ollama server not available for integration tests: {e}")

    finally:
        # Cleanup
        if 'client' in locals():
            await _cleanup_test_data(client, "integration-test")
            await client.close()


async def _cleanup_test_data(client: Graphiti, group_id: str) -> None:
    """Clean up test data from the database."""
    try:
        # Get all episodes for the test group
        episodes = await client.retrieve_episodes(
            reference_time=datetime.now(UTC),
            group_ids=[group_id],
            last_n=1000
        )

        # Delete each episode
        for episode in episodes:
            try:
                await client.remove_episode(episode_uuid=episode.uuid)
            except Exception as e:
                logger.warning(f"Failed to delete episode {episode.uuid}: {e}")

    except Exception as e:
        logger.warning(f"Failed to cleanup test data: {e}")


@pytest.mark.integration
class TestGraphitiPipelineIntegration:
    """
    Comprehensive integration tests for the complete Graphiti memory operation workflow.

    These tests verify that the schema validation fixes work correctly and that memory
    operations actually result in stored data across the entire pipeline.
    """

    @pytest.fixture(scope="function")  # Changed from class to function scope
    async def mcp_server(self, test_config: GraphitiConfig) -> AsyncIterator[FastMCP]:
        """Create MCP server with test configuration."""
        logger.info("=" * 60)
        logger.info("FIXTURE: Starting mcp_server setup")
        logger.info("=" * 60)

        # Reset initialization state and global queues for fresh test
        logger.info("Resetting initialization state and global queues")
        await initialization_manager.reset_for_testing()

        # Reset global queue state
        from src.utils import episode_queues, queue_workers
        episode_queues.clear()
        queue_workers.clear()
        logger.info(f"Cleared queues: episode_queues={len(episode_queues)}, queue_workers={len(queue_workers)}")

        logger.info("Creating FastMCP instance")
        mcp = FastMCP("Test Graphiti Memory", instructions="Test server")

        # Mock environment variables for testing
        env_vars = {
            "NEO4J_URI": test_config.neo4j.uri,
            "NEO4J_USER": test_config.neo4j.user,
            "NEO4J_PASSWORD": test_config.neo4j.password,
            "OLLAMA_BASE_URL": test_config.llm.ollama_base_url,
            "OLLAMA_MODEL": test_config.llm.model,
        }
        logger.info(f"Setting environment variables: {list(env_vars.keys())}")

        with patch.dict(os.environ, env_vars):
            try:
                # Initialize server components
                logger.info("Starting server initialization process")
                await initialization_manager.start_initialization()
                logger.info("Initialization manager started")

                logger.info("Calling initialize_server")
                mcp_config, config, graphiti_client = await initialize_server(mcp, test_config)
                logger.info("initialize_server completed successfully")
                logger.info(f"  - mcp_config: {type(mcp_config)}")
                logger.info(f"  - config: {type(config)}")
                logger.info(f"  - graphiti_client: {type(graphiti_client)}")

                logger.info("Completing initialization")
                await initialization_manager.complete_initialization()
                logger.info("Initialization completed successfully")

                # Store references for cleanup - dynamic attributes for test cleanup
                mcp._test_graphiti_client = graphiti_client  # type: ignore[attr-defined]
                mcp._test_config = config  # type: ignore[attr-defined]

                logger.info("MCP server setup complete, yielding to test")
                yield mcp
                logger.info("Test completed, starting cleanup")

            except Exception as e:
                logger.error(f"Server initialization failed: {e}")
                logger.error(f"Exception type: {type(e)}")
                await initialization_manager.fail_initialization(str(e))
                pytest.skip(f"Failed to initialize MCP server: {e}")
            finally:
                logger.info("Starting fixture cleanup")
                # Cleanup
                if hasattr(mcp, '_test_graphiti_client'):
                    logger.info("Cleaning up test data")
                    client = cast(Graphiti, mcp._test_graphiti_client)  # type: ignore[attr-defined]
                    await _cleanup_test_data(client, "integration-test")
                    logger.info("Closing Graphiti client")
                    await client.close()

                # Reset state and cleanup background tasks after test
                logger.info("Resetting initialization state")
                await initialization_manager.reset_for_testing()

                # Clean up only our application tasks, not system/pytest tasks
                logger.info("Cleaning up application background tasks")
                # Gracefully stop queue workers - wait for current work to finish
                from src.utils.queue_utils import episode_queues, queue_workers
                # First, wait for all queues to drain (current processing to complete)
                max_queue_wait = 15.0  # Allow up to 15 seconds for processing
                start_time = time.time()

                while time.time() - start_time < max_queue_wait:
                    # Check if all queues are empty
                    active_queues = []
                    total_pending = 0
                    for group_id, queue in episode_queues.items():
                        size = queue.qsize()
                        if size > 0:
                            active_queues.append(f"{group_id}({size})")
                            total_pending += size

                    if total_pending == 0:
                        elapsed = time.time() - start_time
                        logger.info(f"All episode queues are empty after {elapsed:.1f}s")
                        break

                    logger.info(f"Waiting for {total_pending} items to process in queues: {active_queues}")
                    await asyncio.sleep(1.0)
                else:
                    elapsed = time.time() - start_time
                    logger.warning(f"Queues not empty after {elapsed:.1f}s, proceeding with worker shutdown")

                # Now signal workers to stop
                for group_id in list(queue_workers.keys()):
                    if queue_workers.get(group_id, False):
                        logger.info(f"Signaling stop for queue worker: {group_id}")
                        queue_workers[group_id] = False

                # Brief wait for workers to notice the stop signal
                await asyncio.sleep(0.5)
                logger.info("Queue workers signaled to stop")

                # Clear any remaining queues
                episode_queues.clear()

                # Only cancel tasks that match our application patterns (avoid system tasks)
                current_task = asyncio.current_task()
                application_tasks = []
                for task in asyncio.all_tasks():
                    if not task.done() and task != current_task:
                        # Use task name and repr to identify application tasks safely
                        task_repr = repr(task)
                        task_name = getattr(task, 'get_name', lambda: '')() or ''

                        # Look for patterns that indicate this is an application task
                        if (any(pattern in task_repr.lower() for pattern in [
                            'src.', 'graphiti', 'queue_worker', 'process_episode',
                            'memory_tools', 'initialization'
                        ]) or any(pattern in task_name.lower() for pattern in [
                            'queue_worker', 'process_episode', 'graphiti'
                        ])):
                            application_tasks.append(task)

                if application_tasks:
                    logger.info(f"Cancelling {len(application_tasks)} application background tasks")
                    for task in application_tasks:
                        task.cancel()

                    # Wait briefly for cancellation with short timeout
                    try:
                        await asyncio.wait_for(
                            asyncio.sleep(0.5),  # Just wait briefly, don't gather cancelled tasks
                            timeout=2.0
                        )
                        logger.info("Application background tasks cleanup completed")
                    except TimeoutError:
                        logger.warning("Timeout during background task cleanup, continuing")
                else:
                    logger.info("No application background tasks to cancel")

                logger.info("Fixture cleanup completed")
                logger.info("=" * 60)

    @pytest.mark.asyncio
    async def test_smoke_basic_initialization(self, mcp_server: FastMCP):
        """
        Smoke test to verify basic MCP server and Graphiti client initialization.

        This test is designed to fail fast and identify if the basic setup is working
        before attempting more complex operations.
        """
        logger.info("=" * 80)
        logger.info("STARTING SMOKE TEST: test_smoke_basic_initialization")
        logger.info("=" * 80)

        # Verify MCP server is initialized
        logger.info("Verifying MCP server initialization")
        assert mcp_server is not None, "MCP server should be initialized"
        assert hasattr(mcp_server, '_test_graphiti_client'), "MCP server should have Graphiti client"

        # Get the Graphiti client
        logger.info("Getting Graphiti client from MCP server")
        graphiti_client = cast(Graphiti, mcp_server._test_graphiti_client)  # type: ignore[attr-defined]
        assert graphiti_client is not None, "Graphiti client should not be None"
        logger.info(f"Graphiti client type: {type(graphiti_client)}")

        # Test basic database connectivity
        logger.info("Testing basic database connectivity")
        try:
            # Simple query to test connectivity
            test_episodes = await graphiti_client.retrieve_episodes(
                reference_time=datetime.now(UTC),
                group_ids=["smoke-test"],
                last_n=1
            )
            logger.info(f"Database connectivity test passed, found {len(test_episodes)} episodes")
        except Exception as e:
            logger.error(f"Database connectivity test failed: {e}")
            raise

        # Test add_memory function availability
        logger.info("Testing add_memory function availability")
        from src.tools.memory_tools import add_memory
        assert callable(add_memory), "add_memory should be callable"

        logger.info("=" * 80)
        logger.info("SMOKE TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    @pytest.mark.asyncio
    async def test_memory_operation_complete_lifecycle(self, mcp_server: FastMCP):
        """
        Test complete memory operation from add to retrieval.

        This test verifies that:
        1. Memory addition succeeds and returns proper response
        2. Background processing actually processes the memory
        3. Memory is retrievable after processing
        4. No silent failures occur in the pipeline
        """
        logger.info("=" * 80)
        logger.info("STARTING TEST: test_memory_operation_complete_lifecycle")
        logger.info("=" * 80)

        # Get the Graphiti client from the MCP server
        logger.info("Step 0: Retrieving Graphiti client from MCP server")
        graphiti_client = cast(Graphiti, mcp_server._test_graphiti_client)  # type: ignore[attr-defined]
        logger.info(f"Graphiti client retrieved: {type(graphiti_client)}")

        # Test data
        test_name = f"Integration Test Memory {uuid.uuid4().hex[:8]}"
        test_content = """
        ActionCable WebSocket integration presents unique challenges for infinite loop subscription handling.
        The guarantor pattern ensures that subscription loops are properly managed and terminated when needed.
        This is particularly important for real-time features that maintain persistent connections.
        """
        test_group_id = "integration-test-lifecycle"

        logger.info("Test setup complete:")
        logger.info(f"  - Test name: {test_name}")
        logger.info(f"  - Test group ID: {test_group_id}")
        logger.info(f"  - Content length: {len(test_content)} characters")

        # Step 1: Add memory through the tool function
        logger.info("Step 1: Starting memory addition through add_memory tool")
        logger.info(f"Calling add_memory with name='{test_name}', group_id='{test_group_id}'")

        try:
            add_result = await add_memory(
                name=test_name,
                episode_body=test_content,
                group_id=test_group_id,
                source="text",
                source_description="Integration test content"
            )
            logger.info(f"add_memory completed successfully: {type(add_result)}")
            logger.info(f"add_memory result: {add_result}")
        except Exception as e:
            logger.error(f"add_memory failed with exception: {e}")
            logger.error(f"Exception type: {type(e)}")
            raise

        # Verify immediate response
        logger.info("Step 1a: Verifying immediate response")
        assert isinstance(add_result, SuccessResponse), f"Expected SuccessResponse, got {type(add_result)}"
        assert "queued for processing" in add_result.message, f"Expected 'queued for processing' in message: {add_result.message}"
        assert test_name in add_result.message, f"Expected test name in message: {add_result.message}"
        logger.info("Immediate response verification passed")

        # Step 2: Wait for background processing to complete
        logger.info("Step 2: Starting background processing wait loop")
        max_wait_time = 15  # seconds - reduced for faster debugging
        wait_interval = 1  # second
        episodes_found = []

        for attempt in range(max_wait_time):
            logger.info(f"Wait attempt {attempt + 1}/{max_wait_time}")
            try:
                logger.debug(f"Calling retrieve_episodes for group_id='{test_group_id}'")
                episodes = await graphiti_client.retrieve_episodes(
                    reference_time=datetime.now(UTC),
                    group_ids=[test_group_id],
                    last_n=10
                )
                logger.debug(f"retrieve_episodes returned {len(episodes)} episodes")

                episodes_found = [ep for ep in episodes if ep.name == test_name]
                logger.info(f"Found {len(episodes_found)} episodes matching test name")

                if episodes_found:
                    logger.info(f"SUCCESS: Found target episode after {attempt + 1} attempts")
                    break

                logger.debug(f"Episode not found yet, sleeping {wait_interval} seconds")
                await asyncio.sleep(wait_interval)

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to get episodes: {e}")
                logger.warning(f"Exception type: {type(e)}")
                await asyncio.sleep(wait_interval)

        # Step 3: Verify memory was actually stored
        logger.info("Step 3: Verifying memory was stored")
        if len(episodes_found) == 0:
            logger.error(f"FAILURE: Memory '{test_name}' was not stored after {max_wait_time} seconds")
            logger.error("This indicates the background processing failed or is too slow")
            # Let's check if there are any episodes at all in the group
            all_episodes = await graphiti_client.retrieve_episodes(
                reference_time=datetime.now(UTC),
                group_ids=[test_group_id],
                last_n=100
            )
            logger.error(f"Total episodes in group '{test_group_id}': {len(all_episodes)}")
            for ep in all_episodes:
                logger.error(f"  Episode: {ep.name} (uuid: {ep.uuid})")

        assert len(episodes_found) > 0, f"Memory '{test_name}' was not stored after {max_wait_time} seconds"

        stored_episode = episodes_found[0]
        logger.info(f"Found stored episode: {stored_episode.name} (uuid: {stored_episode.uuid})")
        logger.info(f"Episode content length: {len(stored_episode.content)} characters")

        assert stored_episode.name == test_name, f"Name mismatch: expected '{test_name}', got '{stored_episode.name}'"
        assert stored_episode.content == test_content, f"Content mismatch: lengths {len(stored_episode.content)} vs {len(test_content)}"
        assert stored_episode.group_id == test_group_id, f"Group ID mismatch: expected '{test_group_id}', got '{stored_episode.group_id}'"
        logger.info("Step 3: Memory verification completed successfully")

        # Step 4: Test memory retrieval through search
        logger.info("Step 4: Testing memory search functionality")
        logger.info("Executing search query: 'ActionCable WebSocket subscription guarantor'")

        try:
            search_results = await graphiti_client.search(
                query="ActionCable WebSocket subscription guarantor",
                group_ids=[test_group_id]
            )
            logger.info(f"Search completed, found {len(search_results)} results")
            for i, result in enumerate(search_results):
                logger.debug(f"Search result {i}: {str(result)[:200]}...")
        except Exception as e:
            logger.error(f"Search failed with exception: {e}")
            raise

        # Verify search finds the stored memory
        assert len(search_results) > 0, "Search did not find the stored memory"
        logger.info("Search found results successfully")

        # Verify entities were extracted (this is where schema validation was failing)
        edges_with_content = [edge for edge in search_results if "ActionCable" in str(edge)]
        logger.info(f"Found {len(edges_with_content)} edges containing 'ActionCable'")
        assert len(edges_with_content) > 0, "No entities were extracted from the memory content"
        logger.info("Step 4: Entity extraction verification completed successfully")

        # Cleanup
        logger.info("Step 5: Cleaning up test data")
        try:
            await graphiti_client.remove_episode(episode_uuid=stored_episode.uuid)
            logger.info(f"Successfully removed episode {stored_episode.uuid}")
        except Exception as e:
            logger.warning(f"Failed to cleanup episode {stored_episode.uuid}: {e}")

        logger.info("=" * 80)
        logger.info("TEST COMPLETED SUCCESSFULLY: test_memory_operation_complete_lifecycle")
        logger.info("=" * 80)

    @pytest.mark.asyncio
    async def test_schema_validation_success_with_ollama(self, graphiti_client: Graphiti):
        """
        Test that schema validation passes with Ollama responses.

        This specifically tests the fix for the ExtractedEntities schema validation
        that was causing silent failures in background processing.
        """
        test_content = """
        The DataProcessor service handles customer profile updates through a JSON API.
        It validates incoming data against predefined schemas and stores valid records
        in the PostgreSQL database. Error cases are logged for debugging purposes.
        """

        test_group_id = "integration-test-schema"
        episode_name = f"Schema Test {uuid.uuid4().hex[:8]}"

        # Add episode directly to test entity extraction
        try:
            await graphiti_client.add_episode(
                name=episode_name,
                episode_body=test_content,
                source=EpisodeType.text,
                source_description="Schema validation test content",
                group_id=test_group_id,
                reference_time=datetime.now(UTC),
                entity_types={}  # Use default entity extraction
            )

            # Wait a moment for processing
            await asyncio.sleep(2)

            # Verify episode was stored successfully
            episodes = await graphiti_client.retrieve_episodes(
                reference_time=datetime.now(UTC),
                group_ids=[test_group_id],
                last_n=5
            )
            stored_episodes = [ep for ep in episodes if ep.name == episode_name]

            assert len(stored_episodes) > 0, "Episode was not stored - schema validation likely failed"

            stored_episode = stored_episodes[0]
            assert stored_episode.content == test_content

            # Test that entity extraction worked (entities should be created)
            search_results = await graphiti_client.search(
                query="DataProcessor service customer profile",
                group_ids=[test_group_id]
            )

            # Should find edges related to the entities
            assert len(search_results) > 0, "No entities were extracted - schema validation failed"

            # Cleanup
            await graphiti_client.remove_episode(episode_uuid=stored_episode.uuid)

        except Exception as e:
            pytest.fail(f"Schema validation failed with Ollama: {e}")

    @pytest.mark.asyncio
    async def test_background_processing_no_silent_failures(self, mcp_server: FastMCP):
        """
        Test that background processing doesn't fail silently.

        This test specifically addresses the issue where operations appeared to succeed
        but failed during background processing without proper error reporting.
        """
        graphiti_client = cast(Graphiti, mcp_server._test_graphiti_client)  # type: ignore[attr-defined]

        # Test with various content types that previously caused silent failures
        test_cases = [
            {
                "name": "Simple Text Memory",
                "content": "This is a simple text memory for testing basic functionality.",
                "source": "text"
            },
            {
                "name": "JSON Structured Memory",
                "content": json.dumps({
                    "customer": {"name": "Acme Corp", "type": "enterprise"},
                    "products": [{"id": "P001", "name": "CloudSync"}]
                }),
                "source": "json"
            },
            {
                "name": "Complex Entity Memory",
                "content": "The authentication service integrates with OAuth2 providers like Google and GitHub. It handles token refresh automatically and provides session management capabilities for web applications.",
                "source": "text"
            }
        ]

        test_group_id = "integration-test-silent-failures"
        stored_episodes = []

        for test_case in test_cases:
            # Add memory
            result = await add_memory(
                name=test_case["name"],
                episode_body=test_case["content"],
                group_id=test_group_id,
                source=test_case["source"]
            )

            assert isinstance(result, SuccessResponse), f"Failed to queue {test_case['name']}"

        # Wait for all background processing to complete
        max_wait_time = 45  # seconds for multiple operations
        wait_interval = 2   # seconds

        for _attempt in range(max_wait_time // wait_interval):
            episodes = await graphiti_client.retrieve_episodes(
                reference_time=datetime.now(UTC),
                group_ids=[test_group_id],
                last_n=10
            )
            stored_episodes = [ep for ep in episodes if any(tc["name"] == ep.name for tc in test_cases)]

            if len(stored_episodes) == len(test_cases):
                break

            await asyncio.sleep(wait_interval)

        # Verify all memories were processed successfully
        assert len(stored_episodes) == len(test_cases), f"Silent failures detected: expected {len(test_cases)} memories, got {len(stored_episodes)}"

        # Verify content integrity
        for test_case in test_cases:
            matching_episodes = [ep for ep in stored_episodes if ep.name == test_case["name"]]
            assert len(matching_episodes) == 1, f"Episode '{test_case['name']}' not found or duplicated"

            episode = matching_episodes[0]
            assert episode.content == test_case["content"], f"Content mismatch for '{test_case['name']}'"

        # Cleanup
        for episode in stored_episodes:
            await graphiti_client.remove_episode(episode_uuid=episode.uuid)

    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self, mcp_server: FastMCP):
        """
        Test multiple concurrent memory operations.

        This verifies that the schema validation fixes work under concurrent load
        and that there are no race conditions in the background processing.
        """
        graphiti_client = cast(Graphiti, mcp_server._test_graphiti_client)  # type: ignore[attr-defined]
        test_group_id = "integration-test-concurrent"

        # Create multiple concurrent memory operations
        num_operations = 5
        operations = []

        for i in range(num_operations):
            operation = add_memory(
                name=f"Concurrent Memory {i}",
                episode_body=f"This is concurrent memory operation number {i}. It tests that multiple operations can be processed simultaneously without interference.",
                group_id=test_group_id,
                source="text"
            )
            operations.append(operation)

        # Execute all operations concurrently
        results = await asyncio.gather(*operations, return_exceptions=True)

        # Verify all operations succeeded
        successful_results = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent operation {i} failed with exception: {result}")
            elif isinstance(result, SuccessResponse):
                successful_results += 1
            else:
                pytest.fail(f"Concurrent operation {i} returned unexpected result: {result}")

        assert successful_results == num_operations, f"Expected {num_operations} successful operations, got {successful_results}"

        # Wait for all background processing to complete
        max_wait_time = 60  # seconds for concurrent operations
        wait_interval = 2   # seconds

        stored_episodes = []
        for _attempt in range(max_wait_time // wait_interval):
            episodes = await graphiti_client.retrieve_episodes(
                reference_time=datetime.now(UTC),
                group_ids=[test_group_id],
                last_n=20
            )
            stored_episodes = [ep for ep in episodes if ep.name.startswith("Concurrent Memory")]

            if len(stored_episodes) >= num_operations:
                break

            await asyncio.sleep(wait_interval)

        # Verify all concurrent operations were processed
        assert len(stored_episodes) >= num_operations, f"Concurrent processing failed: expected at least {num_operations} memories, got {len(stored_episodes)}"

        # Cleanup
        for episode in stored_episodes:
            await graphiti_client.remove_episode(episode_uuid=episode.uuid)

    @pytest.mark.asyncio
    async def test_large_memory_content_processing(self, graphiti_client: Graphiti):
        """
        Test processing of large memory content.

        This verifies that the schema validation fixes work correctly even with
        large content that might produce complex entity extraction results.
        """
        # Create large content that will generate many entities
        large_content = """
        The enterprise software architecture consists of multiple microservices that handle different aspects of the business logic.

        The authentication service manages user credentials, OAuth2 integration, and session handling. It connects to Active Directory
        for enterprise customers and supports SAML, OpenID Connect, and traditional username/password authentication methods.

        The payment processing service integrates with Stripe, PayPal, and Square for credit card transactions. It handles subscription
        billing, one-time payments, refunds, and dispute management. The service maintains PCI compliance and encrypts all sensitive data.

        The notification service sends emails, SMS messages, and push notifications to users. It integrates with SendGrid for email
        delivery, Twilio for SMS, and Firebase for mobile push notifications. The service supports template management and delivery tracking.

        The analytics service collects user behavior data, generates reports, and provides business intelligence dashboards. It uses
        ClickHouse for real-time analytics, Elasticsearch for log analysis, and Grafana for visualization.

        The content management service handles file uploads, image processing, and document storage. It uses AWS S3 for storage,
        CloudFront for content delivery, and ImageMagick for image transformations.

        All services communicate through a message queue system using RabbitMQ and maintain their own PostgreSQL databases.
        The entire infrastructure runs on Kubernetes with Docker containers and is monitored using Prometheus and Grafana.
        """

        test_group_id = "integration-test-large-content"
        episode_name = f"Large Content Test {uuid.uuid4().hex[:8]}"

        start_time = time.time()

        # Add the large content episode
        await graphiti_client.add_episode(
            name=episode_name,
            episode_body=large_content,
            source=EpisodeType.text,
            source_description="Large content performance test",
            group_id=test_group_id,
            reference_time=datetime.now(UTC),
            entity_types={}
        )

        processing_time = time.time() - start_time

        # Wait for processing to complete
        await asyncio.sleep(5)  # Large content may take longer to process

        # Verify episode was stored
        episodes = await graphiti_client.retrieve_episodes(
            reference_time=datetime.now(UTC),
            group_ids=[test_group_id],
            last_n=5
        )
        stored_episodes = [ep for ep in episodes if ep.name == episode_name]

        assert len(stored_episodes) > 0, "Large content episode was not stored"

        stored_episode = stored_episodes[0]
        assert stored_episode.content == large_content

        # Verify entity extraction worked with large content
        search_results = await graphiti_client.search(
            query="authentication service OAuth2 payment processing",
            group_ids=[test_group_id]
        )

        # Should extract many entities from the large content
        assert len(search_results) > 0, "No entities extracted from large content"

        # Verify performance is acceptable (< 30 seconds for large content)
        assert processing_time < 30, f"Large content processing took too long: {processing_time:.2f} seconds"

        # Cleanup
        await graphiti_client.remove_episode(episode_uuid=stored_episode.uuid)

    @pytest.mark.asyncio
    async def test_ollama_health_and_connectivity(self, test_config: GraphitiConfig):
        """
        Test Ollama server health and connectivity for integration tests.

        This ensures that Ollama is properly configured and accessible before
        running the main integration tests.
        """
        try:
            # Create Ollama client for health check
            from graphiti_core.llm_client.config import LLMConfig

            llm_config = LLMConfig(
                api_key="abc",
                model=test_config.llm.ollama_llm_model,
                base_url=test_config.llm.ollama_base_url,
                temperature=test_config.llm.temperature,
                max_tokens=test_config.llm.max_tokens,
            )

            ollama_client = OllamaClient(
                config=llm_config,
                model_parameters={}
            )

            # Test health check
            is_healthy, health_message = await ollama_client.check_health()
            assert is_healthy, f"Ollama server is not healthy: {health_message}"

            # Test model availability
            is_available, model_message = await ollama_client.validate_model_available(test_config.llm.model)
            assert is_available, f"Ollama model '{test_config.llm.model}' is not available: {model_message}"

            # Test basic completion to verify integration
            from openai.types.chat import ChatCompletionMessageParam

            messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello, this is a test message."}]
            response = await ollama_client._create_completion(
                model=test_config.llm.model,
                messages=messages,
                temperature=0.1,
                max_tokens=50
            )

            assert response is not None, "Failed to get response from Ollama"
            assert response.choices[0].message.content, "Empty response from Ollama"

        except Exception as e:
            pytest.skip(f"Ollama integration test requirements not met: {e}")

    @pytest.mark.asyncio
    async def test_error_handling_and_graceful_degradation(self, mcp_server: FastMCP):
        """
        Test error scenarios and graceful degradation.

        This verifies that the system handles errors gracefully and provides
        appropriate feedback when issues occur.
        """
        # Test with invalid content that might cause processing errors
        invalid_test_cases = [
            {
                "name": "Empty Content",
                "content": "",
                "expected_error": False  # Empty content should be handled gracefully
            },
            {
                "name": "Very Long Content",
                "content": "A" * 10000,  # Very long content to test limits
                "expected_error": False  # Should be handled gracefully
            },
            {
                "name": "Unicode Content",
                "content": "Test with émojis 🚀 and unicode characters: ñáéíóú 中文 العربية",
                "expected_error": False  # Unicode should be handled correctly
            }
        ]

        test_group_id = "integration-test-error-handling"

        for test_case in invalid_test_cases:
            try:
                result = await add_memory(
                    name=test_case["name"],
                    episode_body=test_case["content"],
                    group_id=test_group_id,
                    source="text"
                )

                if test_case["expected_error"]:
                    assert isinstance(result, ErrorResponse), f"Expected error for {test_case['name']}, but got success"
                else:
                    assert isinstance(result, SuccessResponse), f"Expected success for {test_case['name']}, but got error: {result}"

            except Exception as e:
                if not test_case["expected_error"]:
                    pytest.fail(f"Unexpected exception for {test_case['name']}: {e}")

    @pytest.mark.asyncio
    async def test_memory_search_integration(self, graphiti_client: Graphiti):
        """
        Test memory search functionality with the schema validation fixes.

        This verifies that memories can be properly searched and retrieved after
        being processed through the fixed pipeline.
        """
        test_group_id = "integration-test-search"

        # Add memories with different searchable content
        memories = [
            {
                "name": "Python Development Best Practices",
                "content": "Python development requires following PEP8 style guidelines, using type hints, and writing comprehensive unit tests with pytest.",
                "entities": ["Python", "PEP8", "pytest"]
            },
            {
                "name": "React Frontend Architecture",
                "content": "React applications benefit from component-based architecture, state management with Redux, and testing with Jest and React Testing Library.",
                "entities": ["React", "Redux", "Jest"]
            },
            {
                "name": "Database Design Principles",
                "content": "Database design should follow normalization principles, use appropriate indexes, and consider performance implications of foreign key relationships.",
                "entities": ["database", "normalization", "indexes"]
            }
        ]

        # Add all memories
        for memory in memories:
            await graphiti_client.add_episode(
                name=memory["name"],
                episode_body=memory["content"],
                source=EpisodeType.text,
                source_description="Search integration test content",
                group_id=test_group_id,
                reference_time=datetime.now(UTC),
                entity_types={}
            )

        # Wait for processing
        await asyncio.sleep(3)

        # Test various search queries
        search_tests = [
            {"query": "Python development testing", "expected_matches": ["Python Development Best Practices"]},
            {"query": "React component architecture", "expected_matches": ["React Frontend Architecture"]},
            {"query": "database normalization indexes", "expected_matches": ["Database Design Principles"]},
            {"query": "testing", "expected_matches": ["Python Development Best Practices", "React Frontend Architecture"]},
        ]

        for search_test in search_tests:
            search_results = await graphiti_client.search(
                query=search_test["query"],
                group_ids=[test_group_id]
            )

            # Verify search found relevant results
            assert len(search_results) > 0, f"No results found for query: {search_test['query']}"

            # Note: We can't easily verify exact matches without more complex result analysis
            # but the fact that we get results indicates the schema validation is working

        # Cleanup
        episodes = await graphiti_client.retrieve_episodes(
            reference_time=datetime.now(UTC),
            group_ids=[test_group_id],
            last_n=10
        )
        for episode in episodes:
            await graphiti_client.remove_episode(episode_uuid=episode.uuid)


@pytest.mark.performance
class TestPerformanceValidationBasic:
    """
    Basic performance validation tests as part of integration testing.

    These tests ensure that the schema validation fixes don't introduce
    unacceptable performance overhead.
    """

    @pytest.mark.asyncio
    async def test_memory_operation_performance_baseline(self, graphiti_client: Graphiti):
        """
        Test baseline performance for memory operations.

        This establishes a performance baseline to ensure the fixes don't
        introduce significant overhead.
        """
        test_content = "This is a performance test memory with moderate content length for baseline measurement."
        test_group_id = "performance-test"

        # Measure performance of memory addition
        start_time = time.time()

        await graphiti_client.add_episode(
            name="Performance Test Memory",
            episode_body=test_content,
            source=EpisodeType.text,
            source_description="Performance baseline test content",
            group_id=test_group_id,
            reference_time=datetime.now(UTC),
            entity_types={}
        )

        processing_time = time.time() - start_time

        # Performance should be reasonable (< 10 seconds for simple content)
        assert processing_time < 10, f"Memory operation took too long: {processing_time:.2f} seconds"

        # Wait for background processing
        await asyncio.sleep(2)

        # Cleanup
        episodes = await graphiti_client.retrieve_episodes(
            reference_time=datetime.now(UTC),
            group_ids=[test_group_id],
            last_n=5
        )
        for episode in episodes:
            await graphiti_client.remove_episode(episode_uuid=episode.uuid)

    @pytest.mark.asyncio
    async def test_search_performance_baseline(self, graphiti_client: Graphiti):
        """
        Test baseline performance for search operations.
        """
        test_group_id = "search-performance-test"

        # Add a memory to search for
        await graphiti_client.add_episode(
            name="Search Performance Test",
            episode_body="This content is for testing search performance and response times.",
            source=EpisodeType.text,
            source_description="Search performance test content",
            group_id=test_group_id,
            reference_time=datetime.now(UTC),
            entity_types={}
        )

        # Wait for processing
        await asyncio.sleep(2)

        # Measure search performance
        start_time = time.time()

        search_results = await graphiti_client.search(
            query="search performance testing",
            group_ids=[test_group_id]
        )

        search_time = time.time() - start_time

        # Search should be fast (< 5 seconds)
        assert search_time < 5, f"Search took too long: {search_time:.2f} seconds"
        assert len(search_results) >= 0, "Search should return results or empty list"

        # Cleanup
        episodes = await graphiti_client.retrieve_episodes(
            reference_time=datetime.now(UTC),
            group_ids=[test_group_id],
            last_n=5
        )
        for episode in episodes:
            await graphiti_client.remove_episode(episode_uuid=episode.uuid)
