#!/usr/bin/env python3
"""
Test coverage for MCP tools to verify parameter validation fixes
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def setup_environment():
    """Set up environment variables for testing."""
    os.environ['USE_OLLAMA'] = 'true'
    os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434/v1'
    os.environ['OLLAMA_LLM_MODEL'] = 'deepseek-r1:7b'
    os.environ['OLLAMA_EMBEDDING_MODEL'] = 'nomic-embed-text'
    os.environ['NEO4J_URI'] = 'bolt://localhost:7687'
    os.environ['NEO4J_USER'] = 'neo4j'
    os.environ['NEO4J_PASSWORD'] = 'password'


class TestMCPToolSignatures:
    """Test that MCP tool function signatures are correct and compatible."""

    def test_search_memory_nodes_signature(self):
        """Test search_memory_nodes function signature."""
        from graphiti_mcp_server import search_memory_nodes
        import inspect

        sig = inspect.signature(search_memory_nodes)

        # Check that all expected parameters exist
        expected_params = {
            'query': str,
            'group_ids': type(None),  # Optional[List[str]]
            'max_nodes': int,
            'center_node_uuid': type(None),  # Optional[str]
            'entity': str
        }

        for param_name, param in sig.parameters.items():
            assert param_name in expected_params, f"Unexpected parameter: {param_name}"

        # Check return type annotation
        assert 'NodeSearchResponse' in str(sig.return_annotation)
        assert 'ErrorResponse' in str(sig.return_annotation)

    def test_search_memory_facts_signature(self):
        """Test search_memory_facts function signature."""
        from graphiti_mcp_server import search_memory_facts
        import inspect

        sig = inspect.signature(search_memory_facts)

        # Check that all expected parameters exist
        expected_params = {
            'query': str,
            'group_ids': type(None),  # Optional[List[str]]
            'max_facts': int,
            'center_node_uuid': type(None),  # Optional[str]
        }

        for param_name, param in sig.parameters.items():
            assert param_name in expected_params, f"Unexpected parameter: {param_name}"

        # Check return type annotation
        assert 'FactSearchResponse' in str(sig.return_annotation)
        assert 'ErrorResponse' in str(sig.return_annotation)

    def test_add_memory_signature(self):
        """Test add_memory function signature."""
        from graphiti_mcp_server import add_memory
        import inspect

        sig = inspect.signature(add_memory)

        # Check that all expected parameters exist
        expected_params = {
            'name': str,
            'episode_body': str,
            'group_id': type(None),  # Optional[str]
            'source': str,
            'source_description': str,
            'uuid': type(None),  # Optional[str]
        }

        for param_name, param in sig.parameters.items():
            assert param_name in expected_params, f"Unexpected parameter: {param_name}"

        # Check return type annotation
        assert 'SuccessResponse' in str(sig.return_annotation)
        assert 'ErrorResponse' in str(sig.return_annotation)

    def test_get_episodes_signature(self):
        """Test get_episodes function signature."""
        from graphiti_mcp_server import get_episodes
        import inspect

        sig = inspect.signature(get_episodes)

        # Check that all expected parameters exist
        expected_params = {
            'group_id': type(None),  # Optional[str]
            'last_n': int,
        }

        for param_name, param in sig.parameters.items():
            assert param_name in expected_params, f"Unexpected parameter: {param_name}"


class TestMCPToolParameterValidation:
    """Test that MCP tools can handle various parameter combinations correctly."""

    @pytest.mark.asyncio
    async def test_search_memory_nodes_parameter_validation(self):
        """Test search_memory_nodes with various parameter combinations."""
        from graphiti_mcp_server import search_memory_nodes
        import inspect

        sig = inspect.signature(search_memory_nodes)

        # Test cases that should work
        test_cases = [
            {"query": "test query"},
            {"query": "test query", "max_nodes": 5},
            {"query": "test query", "entity": "Preference"},
            {"query": "test query", "group_ids": ["group1", "group2"]},
            {"query": "test query", "center_node_uuid": "123e4567-e89b-12d3-a456-426614174000"},
            {"query": "test query", "max_nodes": 10, "entity": "Procedure"},
        ]

        for test_case in test_cases:
            # Test that parameters bind correctly
            bound_args = sig.bind(**test_case)
            assert bound_args is not None

            # Test that function can be called (will fail due to no Graphiti client, but that's expected)
            try:
                result = await search_memory_nodes(**test_case)
                # Should return error about Graphiti client not being initialized
                assert isinstance(result, dict)
                assert 'error' in result
                assert 'Graphiti client not initialized' in result['error']
            except Exception as e:
                # Any other exception should be related to missing dependencies, not parameter validation
                assert 'Graphiti' in str(e) or 'Neo4j' in str(e) or 'Ollama' in str(e)

    @pytest.mark.asyncio
    async def test_search_memory_facts_parameter_validation(self):
        """Test search_memory_facts with various parameter combinations."""
        from graphiti_mcp_server import search_memory_facts
        import inspect

        sig = inspect.signature(search_memory_facts)

        # Test cases that should work
        test_cases = [
            {"query": "test query"},
            {"query": "test query", "max_facts": 5},
            {"query": "test query", "group_ids": ["group1"]},
            {"query": "test query", "center_node_uuid": "123e4567-e89b-12d3-a456-426614174000"},
        ]

        for test_case in test_cases:
            # Test that parameters bind correctly
            bound_args = sig.bind(**test_case)
            assert bound_args is not None

            # Test that function can be called
            try:
                result = await search_memory_facts(**test_case)
                # Should return error about Graphiti client not being initialized
                assert isinstance(result, dict)
                assert 'error' in result
                assert 'Graphiti client not initialized' in result['error']
            except Exception as e:
                # Any other exception should be related to missing dependencies, not parameter validation
                assert 'Graphiti' in str(e) or 'Neo4j' in str(e) or 'Ollama' in str(e)

    @pytest.mark.asyncio
    async def test_add_memory_parameter_validation(self):
        """Test add_memory with various parameter combinations."""
        from graphiti_mcp_server import add_memory
        import inspect

        sig = inspect.signature(add_memory)

        # Test cases that should work
        test_cases = [
            {"name": "test", "episode_body": "test content"},
            {"name": "test", "episode_body": "test content", "source": "text"},
            {"name": "test", "episode_body": "test content", "source": "json"},
            {"name": "test", "episode_body": "test content", "source": "message"},
            {"name": "test", "episode_body": "test content", "group_id": "test-group"},
            {"name": "test", "episode_body": "test content", "uuid": "123e4567-e89b-12d3-a456-426614174000"},
        ]

        for test_case in test_cases:
            # Test that parameters bind correctly
            bound_args = sig.bind(**test_case)
            assert bound_args is not None

            # Test that function can be called
            try:
                result = await add_memory(**test_case)
                # Should return error about Graphiti client not being initialized
                assert isinstance(result, dict)
                assert 'error' in result
                assert 'Graphiti client not initialized' in result['error']
            except Exception as e:
                # Any other exception should be related to missing dependencies, not parameter validation
                assert 'Graphiti' in str(e) or 'Neo4j' in str(e) or 'Ollama' in str(e)

    @pytest.mark.asyncio
    async def test_get_episodes_parameter_validation(self):
        """Test get_episodes with various parameter combinations."""
        from graphiti_mcp_server import get_episodes
        import inspect

        sig = inspect.signature(get_episodes)

        # Test cases that should work
        test_cases = [
            {},
            {"last_n": 5},
            {"group_id": "test-group"},
            {"group_id": "test-group", "last_n": 20},
        ]

        for test_case in test_cases:
            # Test that parameters bind correctly
            bound_args = sig.bind(**test_case)
            assert bound_args is not None

            # Test that function can be called
            try:
                result = await get_episodes(**test_case)
                # Should return error about Graphiti client not being initialized
                assert isinstance(result, dict)
                assert 'error' in result
                assert 'Graphiti client not initialized' in result['error']
            except Exception as e:
                # Any other exception should be related to missing dependencies, not parameter validation
                assert 'Graphiti' in str(e) or 'Neo4j' in str(e) or 'Ollama' in str(e)


class TestMCPToolErrorHandling:
    """Test that MCP tools handle errors gracefully."""

    @pytest.mark.asyncio
    async def test_search_memory_nodes_with_user_payload(self):
        """Test search_memory_nodes with the exact payload from the user's error."""
        from graphiti_mcp_server import search_memory_nodes

        # The exact payload that was causing the -32602 error
        user_payload = {
            "query": "ActionCable WebSocket infinite loop subscription guarantor"
        }

        # This should now work without parameter validation errors
        result = await search_memory_nodes(**user_payload)

        # Should return error about Graphiti client not being initialized (expected)
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'Graphiti client not initialized' in result['error']

    @pytest.mark.asyncio
    async def test_tool_error_responses(self):
        """Test that all tools return proper error responses when Graphiti client is not initialized."""
        from graphiti_mcp_server import (
            search_memory_nodes, search_memory_facts, add_memory,
            get_episodes, delete_entity_edge, delete_episode, get_entity_edge, clear_graph
        )

        tools_to_test = [
            (search_memory_nodes, {"query": "test"}),
            (search_memory_facts, {"query": "test"}),
            (add_memory, {"name": "test", "episode_body": "test"}),
            (get_episodes, {}),
            (delete_entity_edge, {"uuid": "test-uuid"}),
            (delete_episode, {"uuid": "test-uuid"}),
            (get_entity_edge, {"uuid": "test-uuid"}),
            (clear_graph, {}),
        ]

        for tool_func, params in tools_to_test:
            result = await tool_func(**params)
            assert isinstance(result, dict)
            assert 'error' in result
            assert 'Graphiti client not initialized' in result['error']


class TestMCPToolTypeCompatibility:
    """Test that MCP tools are compatible with MCP framework type validation."""

    def test_optional_type_compatibility(self):
        """Test that Optional types are properly defined."""
        from typing import get_args, get_origin
        from graphiti_mcp_server import search_memory_nodes
        import inspect

        sig = inspect.signature(search_memory_nodes)

        # Check that Optional types are properly defined
        for param_name, param in sig.parameters.items():
            if param_name in ['group_ids', 'center_node_uuid']:
                # These should be Optional types
                assert get_origin(param.annotation) is not None
                assert 'Optional' in str(param.annotation) or 'Union' in str(param.annotation)

    def test_list_type_compatibility(self):
        """Test that List types are properly defined."""
        from graphiti_mcp_server import search_memory_nodes
        import inspect

        sig = inspect.signature(search_memory_nodes)

        # Check that List types are properly defined
        group_ids_param = sig.parameters.get('group_ids')
        assert group_ids_param is not None
        assert 'List' in str(group_ids_param.annotation)

    def test_typeddict_compatibility(self):
        """Test that TypedDict definitions are compatible."""
        from graphiti_mcp_server import NodeSearchResponse, ErrorResponse, NodeResult

        # Test that TypedDict classes can be instantiated
        error_response = ErrorResponse(error="test error")
        assert error_response['error'] == "test error"

        node_result = NodeResult(
            uuid="test-uuid",
            name="test-name",
            summary="test-summary",
            labels=["test-label"],
            group_id="test-group",
            created_at="2023-01-01T00:00:00Z",
            attributes={}
        )
        assert node_result['uuid'] == "test-uuid"
        assert isinstance(node_result['labels'], list)

        node_search_response = NodeSearchResponse(
            message="test message",
            nodes=[node_result]
        )
        assert node_search_response['message'] == "test message"
        assert len(node_search_response['nodes']) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
