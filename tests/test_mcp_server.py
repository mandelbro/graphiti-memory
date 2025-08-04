#!/usr/bin/env python3
"""
Test MCP server initialization and tool registration
"""

import os
import sys
from pathlib import Path

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


class TestMCPServerInitialization:
    """Test MCP server initialization and configuration."""

    def test_mcp_server_import(self):
        """Test that the MCP server module can be imported."""
        from graphiti_mcp_server import mcp

        assert mcp is not None
        assert hasattr(mcp, 'tool')
        assert hasattr(mcp, 'resource')

    def test_mcp_server_configuration(self):
        """Test MCP server configuration."""
        from graphiti_mcp_server import mcp

        # Check that the server has the expected configuration
        assert hasattr(mcp, 'settings')
        assert hasattr(mcp.settings, 'port')
        assert mcp.settings.port == 8020  # Default port

    def test_mcp_server_instructions(self):
        """Test that MCP server has proper instructions."""
        from graphiti_mcp_server import mcp

        # Check that instructions are set
        assert hasattr(mcp, 'instructions')
        assert mcp.instructions is not None
        assert len(mcp.instructions) > 0
        assert 'Graphiti' in mcp.instructions


class TestMCPServerTools:
    """Test that MCP tools are properly registered."""

    def test_tool_registration(self):
        """Test that all expected tools are registered."""
        from graphiti_mcp_server import mcp

        # Check that the server has tools registered
        # Note: We can't directly access the tools due to FastMCP implementation,
        # but we can verify the tool decorators are working by checking if functions exist

        # Import all tool functions to verify they exist
        from graphiti_mcp_server import (
            search_memory_nodes,
            search_memory_facts,
            add_memory,
            get_episodes,
            delete_entity_edge,
            delete_episode,
            get_entity_edge,
            clear_graph
        )

        # Verify all functions exist and are callable
        assert callable(search_memory_nodes)
        assert callable(search_memory_facts)
        assert callable(add_memory)
        assert callable(get_episodes)
        assert callable(delete_entity_edge)
        assert callable(delete_episode)
        assert callable(get_entity_edge)
        assert callable(clear_graph)

    def test_tool_decorators(self):
        """Test that tool decorators are properly applied."""
        from graphiti_mcp_server import (
            search_memory_nodes,
            search_memory_facts,
            add_memory,
            get_episodes,
            delete_entity_edge,
            delete_episode,
            get_entity_edge,
            clear_graph
        )

        # Check that functions have the expected attributes from the @mcp.tool() decorator
        # This is a basic check that the decorator was applied
        tools = [
            search_memory_nodes,
            search_memory_facts,
            add_memory,
            get_episodes,
            delete_entity_edge,
            delete_episode,
            get_entity_edge,
            clear_graph
        ]

        for tool in tools:
            assert hasattr(tool, '__name__')
            assert hasattr(tool, '__annotations__')
            assert tool.__name__ in [
                'search_memory_nodes',
                'search_memory_facts',
                'add_memory',
                'get_episodes',
                'delete_entity_edge',
                'delete_episode',
                'get_entity_edge',
                'clear_graph'
            ]


class TestMCPServerConfiguration:
    """Test MCP server configuration classes."""

    def test_graphiti_config_import(self):
        """Test that GraphitiConfig can be imported and instantiated."""
        from graphiti_mcp_server import GraphitiConfig

        # Test that the class can be instantiated
        config = GraphitiConfig()
        assert config is not None
        assert hasattr(config, 'llm')
        assert hasattr(config, 'embedder')
        assert hasattr(config, 'neo4j')

    def test_llm_config_import(self):
        """Test that GraphitiLLMConfig can be imported and instantiated."""
        from graphiti_mcp_server import GraphitiLLMConfig

        # Test that the class can be instantiated
        config = GraphitiLLMConfig()
        assert config is not None
        assert hasattr(config, 'model')
        assert hasattr(config, 'use_ollama')
        assert hasattr(config, 'ollama_base_url')

    def test_embedder_config_import(self):
        """Test that GraphitiEmbedderConfig can be imported and instantiated."""
        from graphiti_mcp_server import GraphitiEmbedderConfig

        # Test that the class can be instantiated
        config = GraphitiEmbedderConfig()
        assert config is not None
        assert hasattr(config, 'model')
        assert hasattr(config, 'use_ollama')
        assert hasattr(config, 'ollama_base_url')

    def test_neo4j_config_import(self):
        """Test that Neo4jConfig can be imported and instantiated."""
        from graphiti_mcp_server import Neo4jConfig

        # Test that the class can be instantiated
        config = Neo4jConfig()
        assert config is not None
        assert hasattr(config, 'uri')
        assert hasattr(config, 'user')
        assert hasattr(config, 'password')


class TestMCPServerTypes:
    """Test MCP server type definitions."""

    def test_response_types(self):
        """Test that response types are properly defined."""
        from graphiti_mcp_server import (
            ErrorResponse, SuccessResponse, NodeSearchResponse,
            FactSearchResponse, EpisodeSearchResponse, StatusResponse
        )

        # Test that all response types can be instantiated
        error_response = ErrorResponse(error="test error")
        assert error_response['error'] == "test error"

        success_response = SuccessResponse(message="test message")
        assert success_response['message'] == "test message"

        node_search_response = NodeSearchResponse(message="test", nodes=[])
        assert node_search_response['message'] == "test"
        assert node_search_response['nodes'] == []

        fact_search_response = FactSearchResponse(message="test", facts=[])
        assert fact_search_response['message'] == "test"
        assert fact_search_response['facts'] == []

        episode_search_response = EpisodeSearchResponse(message="test", episodes=[])
        assert episode_search_response['message'] == "test"
        assert episode_search_response['episodes'] == []

        status_response = StatusResponse(status="ok", message="test")
        assert status_response['status'] == "ok"
        assert status_response['message'] == "test"

    def test_entity_types(self):
        """Test that entity types are properly defined."""
        from graphiti_mcp_server import Requirement, Preference, Procedure

        # Test that entity types can be instantiated
        requirement = Requirement(
            project_name="test project",
            description="test requirement"
        )
        assert requirement.project_name == "test project"
        assert requirement.description == "test requirement"

        preference = Preference(
            category="test category",
            description="test preference"
        )
        assert preference.category == "test category"
        assert preference.description == "test preference"

        procedure = Procedure(
            description="test procedure"
        )
        assert procedure.description == "test procedure"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
