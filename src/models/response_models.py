"""
Response Models for Graphiti MCP Server

This module contains response model classes used for API responses
and data transfer between components.
"""

from typing import Any

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    details: dict[str, Any] | None = None


class SuccessResponse(BaseModel):
    """Standard success response model."""

    message: str


class NodeResult(BaseModel):
    """Model representing a node search result."""

    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(BaseModel):
    """Response model for node search operations."""

    message: str
    nodes: list[NodeResult]


class FactSearchResponse(BaseModel):
    """Response model for fact search operations."""

    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(BaseModel):
    """Response model for episode search operations."""

    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(BaseModel):
    """Response model for status operations."""

    status: str
    message: str
    details: dict[str, Any] | None = None
