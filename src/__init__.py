"""
Graphiti MCP Server - Memory service for AI agents built on temporal knowledge graphs

This package provides a Model Context Protocol (MCP) server that exposes
Graphiti's temporal knowledge graph capabilities to AI agents and assistants.
"""

__version__ = "0.4.0"
__author__ = "Made by Montes"
__email__ = "chris@montesmakes.co"

from .graphiti_mcp_server import main

__all__ = ["main"]
