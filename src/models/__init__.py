"""
Models package for Graphiti MCP Server

This package contains all model definitions used throughout the application,
organized into entity types and response models.
"""

# Import all entity type models
from .entity_types import (
    ENTITY_TYPES,
    Preference,
    Procedure,
    Requirement,
)

# Import all response models
from .response_models import (
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeResult,
    NodeSearchResponse,
    StatusResponse,
    SuccessResponse,
)

# Define what gets exported when using "from models import *"
__all__ = [
    # Entity types
    "Requirement",
    "Preference",
    "Procedure",
    "ENTITY_TYPES",
    # Response models
    "ErrorResponse",
    "SuccessResponse",
    "NodeResult",
    "NodeSearchResponse",
    "FactSearchResponse",
    "EpisodeSearchResponse",
    "StatusResponse",
]
