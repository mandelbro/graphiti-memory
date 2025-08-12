"""Main server configuration classes for the Graphiti MCP server."""

import argparse

from pydantic import BaseModel, Field

from .database_config import Neo4jConfig
from .embedder_config import GraphitiEmbedderConfig
from .llm_config import GraphitiLLMConfig


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti client.

    Centralizes all configuration parameters for the Graphiti client.
    """

    llm: GraphitiLLMConfig = Field(default_factory=GraphitiLLMConfig)
    embedder: GraphitiEmbedderConfig = Field(default_factory=GraphitiEmbedderConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    group_id: str | None = None
    use_custom_entities: bool = False
    destroy_graph: bool = False

    @classmethod
    def from_env(cls) -> "GraphitiConfig":
        """Create a configuration instance from environment variables."""
        return cls(
            llm=GraphitiLLMConfig.from_env(),
            embedder=GraphitiEmbedderConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
        )

    @classmethod
    def from_yaml_and_env(cls) -> "GraphitiConfig":
        """Create a configuration instance from YAML files and environment variables."""
        return cls(
            llm=GraphitiLLMConfig.from_yaml_and_env(),
            embedder=GraphitiEmbedderConfig.from_env(),  # TODO: Add YAML support for embedder
            neo4j=Neo4jConfig.from_env(),  # TODO: Add YAML support for Neo4j
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> "GraphitiConfig":
        """Create configuration from CLI arguments, falling back to YAML and environment variables."""
        # Start with YAML and environment configuration
        config = cls.from_yaml_and_env()

        # Apply CLI overrides
        if args.group_id:
            config.group_id = args.group_id
        else:
            config.group_id = "default"

        config.use_custom_entities = args.use_custom_entities
        config.destroy_graph = args.destroy_graph

        # Update LLM config using CLI args
        config.llm = GraphitiLLMConfig.from_cli_and_env(args)

        # Update embedder config using CLI args
        config.embedder = GraphitiEmbedderConfig.from_cli_and_env(args)

        return config


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str = "sse"  # Default to SSE transport
    port: int = 8020  # Default port for SSE transport

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> "MCPConfig":
        """Create MCP configuration from CLI arguments."""
        return cls(transport=args.transport, port=args.port)
