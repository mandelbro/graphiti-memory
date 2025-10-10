"""
Database configuration for Graphiti MCP Server.

This module contains the Neo4jConfig class that handles all
Neo4j database connection configuration parameters.
"""

import os

from pydantic import BaseModel


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j database connection."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "demodemo"  # Must match docker-compose.yml default

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Create Neo4j configuration from environment variables."""
        return cls(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "demodemo"),  # Must match docker-compose.yml default
        )
