"""Tests for schema generation in FastMCP and Pydantic models."""

import json
import asyncio
from typing import List, Union

import pytest
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel


# Shared model classes for testing
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str


class NodeResult(BaseModel):
    """Node result model."""
    uuid: str
    name: str
    labels: List[str]


class NodeSearchResponse(BaseModel):
    """Node search response model."""
    message: str
    nodes: List[NodeResult]


class UnionTestModel(BaseModel):
    """Test model for Union type schema generation."""
    result: Union[NodeSearchResponse, ErrorResponse]


def test_pydantic_schema_generation():
    """Test direct Pydantic schema generation for NodeSearchResponse."""
    schema = NodeSearchResponse.model_json_schema()

    # Verify basic schema structure
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema

    # Verify required properties
    properties = schema["properties"]
    assert "message" in properties
    assert "nodes" in properties

    # Verify message property
    assert properties["message"]["type"] == "string"

    # Verify nodes property is an array
    assert properties["nodes"]["type"] == "array"
    assert "items" in properties["nodes"]


def test_pydantic_union_schema_generation():
    """Test Pydantic schema generation for Union types."""
    schema = UnionTestModel.model_json_schema()

    # Verify basic schema structure
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema

    # Verify result property exists
    properties = schema["properties"]
    assert "result" in properties

    # Check if $defs are available (they should be for Union types)
    has_defs = "$defs" in schema
    assert has_defs, "$defs should be available for Union types"

    # Verify the Union is properly represented
    result_property = properties["result"]
    assert "anyOf" in result_property or "oneOf" in result_property or "$ref" in result_property


def test_fastmcp_tool_registration():
    """Test that FastMCP tool registration works with Union return types."""
    # Create a test MCP server
    test_mcp = FastMCP("test")

    @test_mcp.tool()
    async def test_tool(query: str) -> Union[NodeSearchResponse, ErrorResponse]:
        """Test tool with Union return type."""
        return NodeSearchResponse(message="test", nodes=[])

    # Verify the tool function was created and decorated
    assert callable(test_tool)
    assert hasattr(test_tool, "__name__")
    assert test_tool.__name__ == "test_tool"


@pytest.mark.asyncio
async def test_tool_function_execution():
    """Test that a tool function with Union return type executes correctly."""
    async def test_tool(query: str) -> Union[NodeSearchResponse, ErrorResponse]:
        """Test tool that returns a valid response."""
        return NodeSearchResponse(
            message=f"Search completed for: {query}",
            nodes=[
                NodeResult(uuid="test-uuid", name="test-node", labels=["TestLabel"])
            ]
        )

    # Execute the tool function directly
    result = await test_tool("test query")

    # Verify the result structure
    assert isinstance(result, NodeSearchResponse)
    assert result.message == "Search completed for: test query"
    assert len(result.nodes) == 1
    assert result.nodes[0].uuid == "test-uuid"
    assert result.nodes[0].name == "test-node"
    assert result.nodes[0].labels == ["TestLabel"]


def test_error_response_schema():
    """Test ErrorResponse model schema generation."""
    schema = ErrorResponse.model_json_schema()

    # Verify basic schema structure
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema

    # Verify error property
    properties = schema["properties"]
    assert "error" in properties
    assert properties["error"]["type"] == "string"

    # Verify required fields
    assert "required" in schema
    assert "error" in schema["required"]


def test_node_result_schema():
    """Test NodeResult model schema generation."""
    schema = NodeResult.model_json_schema()

    # Verify basic schema structure
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema

    # Verify all required properties
    properties = schema["properties"]
    required_fields = ["uuid", "name", "labels"]

    for field in required_fields:
        assert field in properties

    # Verify specific field types
    assert properties["uuid"]["type"] == "string"
    assert properties["name"]["type"] == "string"
    assert properties["labels"]["type"] == "array"
    assert properties["labels"]["items"]["type"] == "string"
