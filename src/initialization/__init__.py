"""
Initialization module for Graphiti MCP Server.

This module contains functions for initializing the Graphiti client
and setting up the MCP server configuration.
"""

from .graphiti_client import initialize_graphiti
from .server_setup import initialize_server, run_mcp_server

__all__ = [
    "initialize_graphiti",
    "initialize_server",
    "run_mcp_server",
]
