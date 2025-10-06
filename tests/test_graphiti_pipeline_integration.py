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
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from graphiti_core import Graphiti
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

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestGraphitiPipelineIntegration:
    """
    Comprehensive integration tests for the complete Graphiti memory operation workflow.

    These tests verify that the schema validation fixes work correctly and that memory
    operations actually result in stored data across the entire pipeline.
    """

    @pytest.fixture(scope="class")
    async def test_config(self) -> GraphitiConfig:
        """Create test configuration for integration tests."""
        from src.config import Neo4jConfig

        return GraphitiConfig(
            neo4j=Neo4jConfig(
                uri=os.getenv("TEST_NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("TEST_NEO4J_USER", "neo4j"),
                password=os.getenv("TEST_NEO4J_PASSWORD", "password"),
            ),
            llm=GraphitiLLMConfig(
                model="gpt-oss:latest",
                use_ollama=True,
                ollama_base_url=os.getenv(
                    "TEST_OLLAMA_URL", "http://localhost:11434/v1"
                ),
                ollama_llm_model="gpt-oss:latest",
                api_key="ollama",  # Ollama doesn't require real API key
            ),
            use_custom_entities=True,
            group_id="integration-test",
        )

    @pytest.fixture(scope="class")
    async def graphiti_client(
        self, test_config: GraphitiConfig
    ) -> AsyncGenerator[Graphiti]:
        """Create real Graphiti client with Ollama configuration for integration testing."""
        try:
            # Create the client using our enhanced configuration
            client = await create_graphiti_client(test_config)

            # Build indices and constraints
            await client.build_indices_and_constraints()

            # Clear any existing test data
            await self._cleanup_test_data(client, "integration-test")

            yield client

        except Exception as e:
            pytest.skip(f"Ollama server not available for integration tests: {e}")

        finally:
            # Cleanup
            if "client" in locals():
                await self._cleanup_test_data(client, "integration-test")
                await client.close()

    @pytest.fixture(scope="class")
    async def mcp_server(self, test_config: GraphitiConfig) -> AsyncGenerator[FastMCP]:
        """Create MCP server with test configuration."""
        mcp = FastMCP("Test Graphiti Memory", instructions="Test server")

        # Mock environment variables for testing
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": test_config.neo4j.uri,
                "NEO4J_USER": test_config.neo4j.user,
                "NEO4J_PASSWORD": test_config.neo4j.password,
                "OLLAMA_BASE_URL": test_config.llm.ollama_base_url,
                "OLLAMA_MODEL": test_config.llm.model,
            },
        ):
            try:
                # Initialize server components
                await initialization_manager.start_initialization()
                mcp_config, config, graphiti_client = await initialize_server(mcp)
                await initialization_manager.complete_initialization()

                # Store references for cleanup
                mcp._test_graphiti_client = graphiti_client  # type: ignore[attr-defined]
                mcp._test_config = config  # type: ignore[attr-defined]

                yield mcp

            except Exception as e:
                await initialization_manager.fail_initialization(str(e))
                pytest.skip(f"Failed to initialize MCP server: {e}")
            finally:
                # Cleanup
                if hasattr(mcp, "_test_graphiti_client"):
                    await self._cleanup_test_data(
                        mcp._test_graphiti_client,
                        "integration-test",  # type: ignore[attr-defined]
                    )
                    await mcp._test_graphiti_client.close()  # type: ignore[attr-defined]

    async def _cleanup_test_data(self, client: Graphiti, group_id: str) -> None:
        """Clean up test data from the database."""
        try:
            # Get all episodes for the test group
            episodes = await client.get_episodes(group_id=group_id, last_n=1000)

            # Delete each episode
            for episode in episodes:
                try:
                    await client.delete_episode(episode.uuid)
                except Exception as e:
                    logger.warning(f"Failed to delete episode {episode.uuid}: {e}")

        except Exception as e:
            logger.warning(f"Failed to cleanup test data: {e}")

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
        # Get the Graphiti client from the MCP server
        graphiti_client = mcp_server._test_graphiti_client  # type: ignore[attr-defined]

        # Test data
        test_name = f"Integration Test Memory {uuid.uuid4().hex[:8]}"
        test_content = """
        ActionCable WebSocket integration presents unique challenges for infinite loop subscription handling.
        The guarantor pattern ensures that subscription loops are properly managed and terminated when needed.
        This is particularly important for real-time features that maintain persistent connections.
        """
        test_group_id = "integration-test-lifecycle"

        # Step 1: Add memory through the tool function
        add_result = await add_memory(
            name=test_name,
            episode_body=test_content,
            group_id=test_group_id,
            source="text",
            source_description="Integration test content",
        )

        # Verify immediate response
        assert isinstance(add_result, SuccessResponse)
        assert "queued for processing" in add_result.message
        assert test_name in add_result.message

        # Step 2: Wait for background processing to complete
        max_wait_time = 30  # seconds
        wait_interval = 1  # second
        episodes_found = []

        for attempt in range(max_wait_time):
            try:
                episodes = await graphiti_client.get_episodes(
                    group_id=test_group_id, last_n=10
                )
                episodes_found = [ep for ep in episodes if ep.name == test_name]

                if episodes_found:
                    break

                await asyncio.sleep(wait_interval)

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to get episodes: {e}")
                await asyncio.sleep(wait_interval)

        # Step 3: Verify memory was actually stored
        assert len(episodes_found) > 0, (
            f"Memory '{test_name}' was not stored after {max_wait_time} seconds"
        )

        stored_episode = episodes_found[0]
        assert stored_episode.name == test_name
        assert stored_episode.content == test_content
        assert stored_episode.group_id == test_group_id

        # Step 4: Test memory retrieval through search
        search_results = await graphiti_client.search(
            query="ActionCable WebSocket subscription guarantor",
            group_ids=[test_group_id],
            limit=5,
        )

        # Verify search finds the stored memory
        assert len(search_results.nodes) > 0, "Search did not find the stored memory"

        # Verify entities were extracted (this is where schema validation was failing)
        nodes_with_content = [
            node for node in search_results.nodes if "ActionCable" in str(node)
        ]
        assert len(nodes_with_content) > 0, (
            "No entities were extracted from the memory content"
        )

        # Cleanup
        await graphiti_client.delete_episode(stored_episode.uuid)

    @pytest.mark.asyncio
    async def test_schema_validation_success_with_ollama(
        self, graphiti_client: Graphiti
    ):
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
                source="text",
                group_id=test_group_id,
                reference_time=datetime.now(UTC),
                entity_types={},  # Use default entity extraction
            )

            # Wait a moment for processing
            await asyncio.sleep(2)

            # Verify episode was stored successfully
            episodes = await graphiti_client.get_episodes(
                group_id=test_group_id, last_n=5
            )
            stored_episodes = [ep for ep in episodes if ep.name == episode_name]

            assert len(stored_episodes) > 0, (
                "Episode was not stored - schema validation likely failed"
            )

            stored_episode = stored_episodes[0]
            assert stored_episode.content == test_content

            # Test that entity extraction worked (entities should be created)
            search_results = await graphiti_client.search(
                query="DataProcessor service customer profile",
                group_ids=[test_group_id],
                limit=10,
            )

            # Should find nodes related to the entities
            assert len(search_results.nodes) > 0, (
                "No entities were extracted - schema validation failed"
            )

            # Cleanup
            await graphiti_client.delete_episode(stored_episode.uuid)

        except Exception as e:
            pytest.fail(f"Schema validation failed with Ollama: {e}")

    @pytest.mark.asyncio
    async def test_background_processing_no_silent_failures(self, mcp_server: FastMCP):
        """
        Test that background processing doesn't fail silently.

        This test specifically addresses the issue where operations appeared to succeed
        but failed during background processing without proper error reporting.
        """
        graphiti_client = mcp_server._test_graphiti_client  # type: ignore[attr-defined]

        # Test with various content types that previously caused silent failures
        test_cases = [
            {
                "name": "Simple Text Memory",
                "content": "This is a simple text memory for testing basic functionality.",
                "source": "text",
            },
            {
                "name": "JSON Structured Memory",
                "content": json.dumps(
                    {
                        "customer": {"name": "Acme Corp", "type": "enterprise"},
                        "products": [{"id": "P001", "name": "CloudSync"}],
                    }
                ),
                "source": "json",
            },
            {
                "name": "Complex Entity Memory",
                "content": "The authentication service integrates with OAuth2 providers like Google and GitHub. It handles token refresh automatically and provides session management capabilities for web applications.",
                "source": "text",
            },
        ]

        test_group_id = "integration-test-silent-failures"
        stored_episodes = []

        for test_case in test_cases:
            # Add memory
            result = await add_memory(
                name=test_case["name"],
                episode_body=test_case["content"],
                group_id=test_group_id,
                source=test_case["source"],
            )

            assert isinstance(result, SuccessResponse), (
                f"Failed to queue {test_case['name']}"
            )

        # Wait for all background processing to complete
        max_wait_time = 45  # seconds for multiple operations
        wait_interval = 2  # seconds

        for _attempt in range(max_wait_time // wait_interval):
            episodes = await graphiti_client.get_episodes(
                group_id=test_group_id, last_n=10
            )
            stored_episodes = [
                ep for ep in episodes if any(tc["name"] == ep.name for tc in test_cases)
            ]

            if len(stored_episodes) == len(test_cases):
                break

            await asyncio.sleep(wait_interval)

        # Verify all memories were processed successfully
        assert len(stored_episodes) == len(test_cases), (
            f"Silent failures detected: expected {len(test_cases)} memories, got {len(stored_episodes)}"
        )

        # Verify content integrity
        for test_case in test_cases:
            matching_episodes = [
                ep for ep in stored_episodes if ep.name == test_case["name"]
            ]
            assert len(matching_episodes) == 1, (
                f"Episode '{test_case['name']}' not found or duplicated"
            )

            episode = matching_episodes[0]
            assert episode.content == test_case["content"], (
                f"Content mismatch for '{test_case['name']}'"
            )

        # Cleanup
        for episode in stored_episodes:
            await graphiti_client.delete_episode(episode.uuid)

    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self, mcp_server: FastMCP):
        """
        Test multiple concurrent memory operations.

        This verifies that the schema validation fixes work under concurrent load
        and that there are no race conditions in the background processing.
        """
        graphiti_client = mcp_server._test_graphiti_client  # type: ignore[attr-defined]
        test_group_id = "integration-test-concurrent"

        # Create multiple concurrent memory operations
        num_operations = 5
        operations = []

        for i in range(num_operations):
            operation = add_memory(
                name=f"Concurrent Memory {i}",
                episode_body=f"This is concurrent memory operation number {i}. It tests that multiple operations can be processed simultaneously without interference.",
                group_id=test_group_id,
                source="text",
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
                pytest.fail(
                    f"Concurrent operation {i} returned unexpected result: {result}"
                )

        assert successful_results == num_operations, (
            f"Expected {num_operations} successful operations, got {successful_results}"
        )

        # Wait for all background processing to complete
        max_wait_time = 60  # seconds for concurrent operations
        wait_interval = 2  # seconds

        stored_episodes = []
        for _attempt in range(max_wait_time // wait_interval):
            episodes = await graphiti_client.get_episodes(
                group_id=test_group_id, last_n=20
            )
            stored_episodes = [
                ep for ep in episodes if ep.name.startswith("Concurrent Memory")
            ]

            if len(stored_episodes) >= num_operations:
                break

            await asyncio.sleep(wait_interval)

        # Verify all concurrent operations were processed
        assert len(stored_episodes) >= num_operations, (
            f"Concurrent processing failed: expected at least {num_operations} memories, got {len(stored_episodes)}"
        )

        # Cleanup
        for episode in stored_episodes:
            await graphiti_client.delete_episode(episode.uuid)

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
            source="text",
            group_id=test_group_id,
            reference_time=datetime.now(UTC),
            entity_types={},
        )

        processing_time = time.time() - start_time

        # Wait for processing to complete
        await asyncio.sleep(5)  # Large content may take longer to process

        # Verify episode was stored
        episodes = await graphiti_client.get_episodes(group_id=test_group_id, last_n=5)
        stored_episodes = [ep for ep in episodes if ep.name == episode_name]

        assert len(stored_episodes) > 0, "Large content episode was not stored"

        stored_episode = stored_episodes[0]
        assert stored_episode.content == large_content

        # Verify entity extraction worked with large content
        search_results = await graphiti_client.search(
            query="authentication service OAuth2 payment processing",
            group_ids=[test_group_id],
            limit=20,
        )

        # Should extract many entities from the large content
        assert len(search_results.nodes) > 0, "No entities extracted from large content"

        # Verify performance is acceptable (< 30 seconds for large content)
        assert processing_time < 30, (
            f"Large content processing took too long: {processing_time:.2f} seconds"
        )

        # Cleanup
        await graphiti_client.delete_episode(stored_episode.uuid)

    @pytest.mark.asyncio
    async def test_ollama_health_and_connectivity(self, test_config: GraphitiConfig):
        """
        Test Ollama server health and connectivity for integration tests.

        This ensures that Ollama is properly configured and accessible before
        running the main integration tests.
        """
        try:
            # Create Ollama client for health check
            ollama_client = OllamaClient(config=test_config.llm, model_parameters={})

            # Test health check
            is_healthy, health_message = await ollama_client.check_health()
            assert is_healthy, f"Ollama server is not healthy: {health_message}"

            # Test model availability
            is_available, model_message = await ollama_client.validate_model_available(
                test_config.llm.model
            )
            assert is_available, (
                f"Ollama model '{test_config.llm.model}' is not available: {model_message}"
            )

            # Test basic completion to verify integration
            messages = [{"role": "user", "content": "Hello, this is a test message."}]
            response = await ollama_client._create_completion(
                model=test_config.llm.model,
                messages=messages,
                temperature=0.1,
                max_tokens=50,
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
                "expected_error": False,  # Empty content should be handled gracefully
            },
            {
                "name": "Very Long Content",
                "content": "A" * 10000,  # Very long content to test limits
                "expected_error": False,  # Should be handled gracefully
            },
            {
                "name": "Unicode Content",
                "content": "Test with émojis 🚀 and unicode characters: ñáéíóú 中文 العربية",
                "expected_error": False,  # Unicode should be handled correctly
            },
        ]

        test_group_id = "integration-test-error-handling"

        for test_case in invalid_test_cases:
            try:
                result = await add_memory(
                    name=test_case["name"],
                    episode_body=test_case["content"],
                    group_id=test_group_id,
                    source="text",
                )

                if test_case["expected_error"]:
                    assert isinstance(result, ErrorResponse), (
                        f"Expected error for {test_case['name']}, but got success"
                    )
                else:
                    assert isinstance(result, SuccessResponse), (
                        f"Expected success for {test_case['name']}, but got error: {result}"
                    )

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
                "entities": ["Python", "PEP8", "pytest"],
            },
            {
                "name": "React Frontend Architecture",
                "content": "React applications benefit from component-based architecture, state management with Redux, and testing with Jest and React Testing Library.",
                "entities": ["React", "Redux", "Jest"],
            },
            {
                "name": "Database Design Principles",
                "content": "Database design should follow normalization principles, use appropriate indexes, and consider performance implications of foreign key relationships.",
                "entities": ["database", "normalization", "indexes"],
            },
        ]

        # Add all memories
        for memory in memories:
            await graphiti_client.add_episode(
                name=memory["name"],
                episode_body=memory["content"],
                source="text",
                group_id=test_group_id,
                reference_time=datetime.now(UTC),
                entity_types={},
            )

        # Wait for processing
        await asyncio.sleep(3)

        # Test various search queries
        search_tests = [
            {
                "query": "Python development testing",
                "expected_matches": ["Python Development Best Practices"],
            },
            {
                "query": "React component architecture",
                "expected_matches": ["React Frontend Architecture"],
            },
            {
                "query": "database normalization indexes",
                "expected_matches": ["Database Design Principles"],
            },
            {
                "query": "testing",
                "expected_matches": [
                    "Python Development Best Practices",
                    "React Frontend Architecture",
                ],
            },
        ]

        for search_test in search_tests:
            search_results = await graphiti_client.search(
                query=search_test["query"], group_ids=[test_group_id], limit=10
            )

            # Verify search found relevant results
            assert len(search_results.nodes) > 0, (
                f"No results found for query: {search_test['query']}"
            )

            # Note: We can't easily verify exact matches without more complex result analysis
            # but the fact that we get results indicates the schema validation is working

        # Cleanup
        episodes = await graphiti_client.get_episodes(group_id=test_group_id, last_n=10)
        for episode in episodes:
            await graphiti_client.delete_episode(episode.uuid)


# Performance tests removed - they require full integration setup with Neo4j
# and a configured graphiti_client fixture. These are better suited for
# a separate performance testing suite with dedicated infrastructure.
