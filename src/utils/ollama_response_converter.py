"""
Ollama Response Conversion Utilities.

This module provides utilities for converting between Ollama native responses
and OpenAI-compatible formats, including message formatting, mock response classes,
and comprehensive schema mapping for Graphiti Core compatibility.
"""

import logging
import time
from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel


class MockMessage:
    """Mock OpenAI message class for Ollama compatibility."""

    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"
        self.parsed: BaseModel | None = None  # Can hold parsed structured output
        self.refusal = None  # Ollama doesn't have refusal mechanism

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation compatible with Pydantic model_dump()."""
        return {
            "content": self.content,
            "role": self.role,
            "parsed": self.parsed,
            "refusal": self.refusal,
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": None,
        }


class MockChoice:
    """Mock OpenAI choice class for Ollama compatibility."""

    def __init__(self, message_content: str):
        self.message = MockMessage(message_content)
        self.index = 0
        self.finish_reason = "stop"


class MockResponse:
    """Mock OpenAI response class for Ollama compatibility."""

    def __init__(self, content: str, model: str):
        self.choices = [MockChoice(content)]
        self.model = model
        self.id = f"chatcmpl-{int(time.time())}"
        self.created = int(time.time())
        self.object = "chat.completion"


class OllamaResponseConverter:
    """
    Comprehensive converter for Ollama responses to Graphiti-compatible schemas.

    Provides methods to convert messages, responses, and handle schema mapping
    between Ollama response formats and expected Graphiti Core schemas.
    This addresses schema validation failures in background processing.
    """

    def __init__(self):
        """Initialize the response converter with logging."""
        self.logger = logging.getLogger(__name__)

    def convert_structured_response(self, response_data: dict | list, target_schema: type) -> dict:
        """
        Convert Ollama response format to target schema format.
        Handles both direct responses and nested validation scenarios.

        Args:
            response_data: Raw response data from Ollama (dict or list)
            target_schema: Target Pydantic model schema class

        Returns:
            dict: Converted response data compatible with target schema

        Raises:
            ValueError: If conversion fails and no fallback is possible
        """
        # Check for invalid data types upfront
        if response_data is None:
            raise ValueError(f"Cannot convert None response data to {target_schema.__name__}")

        try:
            self.logger.debug(f"Converting response for schema: {target_schema.__name__}")

            # Detect ExtractedEntities schema requirement
            if self._is_extracted_entities_schema(target_schema):
                return self._convert_to_extracted_entities(response_data)

            # Detect ExtractedEdges schema requirement
            if self._is_extracted_edges_schema(target_schema):
                return self._convert_to_extracted_edges(response_data)

            # Detect NodeResolutions schema requirement
            if self._is_node_resolutions_schema(target_schema):
                return self._convert_to_node_resolutions(response_data)

            # Handle other schema patterns
            return self._convert_generic_schema(response_data, target_schema)

        except Exception as e:
            self.logger.warning(f"Schema conversion failed for {target_schema.__name__}: {e}")
            # Graceful fallback - return original data for valid types
            if isinstance(response_data, dict):
                return response_data
            elif isinstance(response_data, list) and response_data:
                return response_data[0] if len(response_data) == 1 else {"items": response_data}
            else:
                raise ValueError(f"Cannot convert response data to {target_schema.__name__}: {e}") from e

    def _convert_to_extracted_entities(self, data: dict | list) -> dict:
        """
        Convert Ollama entity array to ExtractedEntities format.

        Maps "entity" → "name" fields and wraps in expected structure.

        Args:
            data: Ollama response data (list of entities or dict)

        Returns:
            dict: Data in ExtractedEntities format
        """
        self.logger.debug("Converting to ExtractedEntities format")

        # Handle list format (typical Ollama entity response)
        if isinstance(data, list):
            mapped_entities = []
            for item in data:
                if isinstance(item, dict):
                    # Handle field name mapping: "entity" or "entity_name" -> "name"
                    mapped_item = item.copy()
                    if "entity" in mapped_item and "name" not in mapped_item:
                        mapped_item["name"] = mapped_item.pop("entity")
                        self.logger.debug(f"Mapped 'entity' field to 'name': {mapped_item.get('name')}")
                    elif "entity_name" in mapped_item and "name" not in mapped_item:
                        mapped_item["name"] = mapped_item.pop("entity_name")
                        self.logger.debug(f"Mapped 'entity_name' field to 'name': {mapped_item.get('name')}")
                    mapped_entities.append(mapped_item)
                else:
                    self.logger.warning(f"Unexpected entity item type: {type(item)}")

            return {"extracted_entities": mapped_entities}

        # Handle dict format (already wrapped or single entity)
        elif isinstance(data, dict):
            if "extracted_entities" in data:
                # Already in correct format, but check for field mapping
                entities = data["extracted_entities"]
                if isinstance(entities, list):
                    mapped_entities = []
                    for item in entities:
                        if isinstance(item, dict):
                            mapped_item = item.copy()
                            if "entity" in mapped_item and "name" not in mapped_item:
                                mapped_item["name"] = mapped_item.pop("entity")
                            mapped_entities.append(mapped_item)
                        else:
                            mapped_entities.append(item)
                    return {"extracted_entities": mapped_entities}
                return data
            else:
                # Single entity dict, wrap it
                mapped_item = data.copy()
                if "entity" in mapped_item and "name" not in mapped_item:
                    mapped_item["name"] = mapped_item.pop("entity")
                return {"extracted_entities": [mapped_item]}

        # Fallback for unexpected data types
        self.logger.warning(f"Unexpected data type for ExtractedEntities: {type(data)}")
        return {"extracted_entities": []}

    def _is_extracted_entities_schema(self, schema: type) -> bool:
        """
        Detect if target schema is ExtractedEntities type.

        Args:
            schema: Pydantic model class to check

        Returns:
            bool: True if schema expects ExtractedEntities format
        """
        # Check for ExtractedEntities pattern by looking for extracted_entities field
        if hasattr(schema, 'model_fields'):
            fields = schema.model_fields
            if 'extracted_entities' in fields:
                self.logger.debug(f"Detected ExtractedEntities schema: {schema.__name__}")
                return True

        # Check class name as backup
        if 'ExtractedEntities' in schema.__name__:
            self.logger.debug(f"Detected ExtractedEntities by name: {schema.__name__}")
            return True

        return False

    def _is_extracted_edges_schema(self, schema: type) -> bool:
        """
        Detect if target schema is ExtractedEdges type.

        Args:
            schema: Pydantic model class to check

        Returns:
            bool: True if schema expects ExtractedEdges format
        """
        # Check for ExtractedEdges pattern by looking for edges field
        if hasattr(schema, 'model_fields'):
            fields = schema.model_fields
            if 'edges' in fields:
                self.logger.debug(f"Detected ExtractedEdges schema: {schema.__name__}")
                return True

        # Check class name as backup
        if 'ExtractedEdges' in schema.__name__:
            self.logger.debug(f"Detected ExtractedEdges by name: {schema.__name__}")
            return True

        return False

    def _is_node_resolutions_schema(self, schema: type) -> bool:
        """
        Detect if target schema is NodeResolutions type.

        Args:
            schema: Pydantic model class to check

        Returns:
            bool: True if schema expects NodeResolutions format
        """
        # Check for NodeResolutions pattern by looking for entity_resolutions field
        if hasattr(schema, 'model_fields'):
            fields = schema.model_fields
            if 'entity_resolutions' in fields:
                self.logger.debug(f"Detected NodeResolutions schema: {schema.__name__}")
                return True

        # Check class name as backup
        if 'NodeResolutions' in schema.__name__:
            self.logger.debug(f"Detected NodeResolutions by name: {schema.__name__}")
            return True

        return False

    def _convert_to_extracted_edges(self, data: dict | list) -> dict:
        """
        Convert Ollama edge array to ExtractedEdges format.

        Maps field names:
        - "subject_id" → "source_entity_id"
        - "object_id" → "target_entity_id"
        - "fact_text" → "fact"

        Args:
            data: Ollama response data (list of edges or dict)

        Returns:
            dict: Data in ExtractedEdges format
        """
        self.logger.debug("Converting to ExtractedEdges format")

        # Handle list format (typical Ollama edge response)
        if isinstance(data, list):
            mapped_edges = []
            for item in data:
                if isinstance(item, dict):
                    # Handle field name mapping
                    mapped_item = item.copy()
                    if "subject_id" in mapped_item and "source_entity_id" not in mapped_item:
                        mapped_item["source_entity_id"] = mapped_item.pop("subject_id")
                        self.logger.debug(f"Mapped 'subject_id' to 'source_entity_id'")
                    if "object_id" in mapped_item and "target_entity_id" not in mapped_item:
                        mapped_item["target_entity_id"] = mapped_item.pop("object_id")
                        self.logger.debug(f"Mapped 'object_id' to 'target_entity_id'")
                    if "fact_text" in mapped_item and "fact" not in mapped_item:
                        mapped_item["fact"] = mapped_item.pop("fact_text")
                        self.logger.debug(f"Mapped 'fact_text' to 'fact'")
                    mapped_edges.append(mapped_item)
                else:
                    self.logger.warning(f"Unexpected edge item type: {type(item)}")

            return {"edges": mapped_edges}

        # Handle dict format (already wrapped)
        if isinstance(data, dict):
            if "edges" in data:
                # Already wrapped, but may need field mapping
                mapped_data = data.copy()
                if "edges" in mapped_data:
                    mapped_edges = []
                    for edge in mapped_data["edges"]:
                        if isinstance(edge, dict):
                            mapped_edge = edge.copy()
                            if "subject_id" in mapped_edge and "source_entity_id" not in mapped_edge:
                                mapped_edge["source_entity_id"] = mapped_edge.pop("subject_id")
                            if "object_id" in mapped_edge and "target_entity_id" not in mapped_edge:
                                mapped_edge["target_entity_id"] = mapped_edge.pop("object_id")
                            if "fact_text" in mapped_edge and "fact" not in mapped_edge:
                                mapped_edge["fact"] = mapped_edge.pop("fact_text")
                            mapped_edges.append(mapped_edge)
                    mapped_data["edges"] = mapped_edges
                return mapped_data
            else:
                # Single edge dict, wrap it
                mapped_item = data.copy()
                if "subject_id" in mapped_item and "source_entity_id" not in mapped_item:
                    mapped_item["source_entity_id"] = mapped_item.pop("subject_id")
                if "object_id" in mapped_item and "target_entity_id" not in mapped_item:
                    mapped_item["target_entity_id"] = mapped_item.pop("object_id")
                if "fact_text" in mapped_item and "fact" not in mapped_item:
                    mapped_item["fact"] = mapped_item.pop("fact_text")
                return {"edges": [mapped_item]}

        # Fallback for unexpected data types
        self.logger.warning(f"Unexpected data type for ExtractedEdges: {type(data)}")
        return {"edges": []}

    def _convert_to_node_resolutions(self, data: dict | list) -> dict:
        """
        Convert Ollama node resolution array to NodeResolutions format.

        Maps field names:
        - "duplicate_idx" → "duplicates" (converts to list format)

        Args:
            data: Ollama response data (list of resolutions or dict)

        Returns:
            dict: Data in NodeResolutions format
        """
        self.logger.debug("Converting to NodeResolutions format")

        # Handle list format (typical Ollama resolution response)
        if isinstance(data, list):
            mapped_resolutions = []
            for item in data:
                if isinstance(item, dict):
                    # Handle field name mapping
                    mapped_item = item.copy()
                    if "duplicate_idx" in mapped_item and "duplicates" not in mapped_item:
                        # Convert duplicate_idx to duplicates list format (keep both fields)
                        duplicate_idx = mapped_item["duplicate_idx"]  # Keep original field
                        if duplicate_idx >= 0:
                            mapped_item["duplicates"] = [duplicate_idx]
                        else:
                            mapped_item["duplicates"] = []
                        self.logger.debug(f"Added 'duplicates' field based on 'duplicate_idx'")
                    mapped_resolutions.append(mapped_item)
                else:
                    self.logger.warning(f"Unexpected resolution item type: {type(item)}")

            return {"entity_resolutions": mapped_resolutions}

        # Handle dict format (already wrapped)
        if isinstance(data, dict):
            if "entity_resolutions" in data:
                # Already wrapped, but may need field mapping
                mapped_data = data.copy()
                if "entity_resolutions" in mapped_data:
                    mapped_resolutions = []
                    for resolution in mapped_data["entity_resolutions"]:
                        if isinstance(resolution, dict):
                            mapped_resolution = resolution.copy()
                            if "duplicate_idx" in mapped_resolution and "duplicates" not in mapped_resolution:
                                duplicate_idx = mapped_resolution["duplicate_idx"]  # Keep original field
                                if duplicate_idx >= 0:
                                    mapped_resolution["duplicates"] = [duplicate_idx]
                                else:
                                    mapped_resolution["duplicates"] = []
                            mapped_resolutions.append(mapped_resolution)
                    mapped_data["entity_resolutions"] = mapped_resolutions
                return mapped_data
            else:
                # Single resolution dict, wrap it
                mapped_item = data.copy()
                if "duplicate_idx" in mapped_item and "duplicates" not in mapped_item:
                    duplicate_idx = mapped_item["duplicate_idx"]  # Keep original field
                    if duplicate_idx >= 0:
                        mapped_item["duplicates"] = [duplicate_idx]
                    else:
                        mapped_item["duplicates"] = []
                return {"entity_resolutions": [mapped_item]}

        # Fallback for unexpected data types
        self.logger.warning(f"Unexpected data type for NodeResolutions: {type(data)}")
        return {"entity_resolutions": []}

    def _convert_generic_schema(self, data: dict | list, target_schema: type) -> dict:
        """
        Handle other schema conversion patterns.

        Args:
            data: Response data to convert
            target_schema: Target schema class

        Returns:
            dict: Converted data or original data if no conversion needed
        """
        self.logger.debug(f"Generic schema conversion for: {target_schema.__name__}")

        # For list responses, try to use first item if target expects dict
        if isinstance(data, list) and data:
            if len(data) == 1:
                return data[0] if isinstance(data[0], dict) else {"value": data[0]}
            else:
                # Multiple items, check if schema can handle arrays
                if hasattr(target_schema, 'model_fields'):
                    fields = target_schema.model_fields
                    # Look for array-like fields
                    for field_name, field_info in fields.items():
                        if hasattr(field_info, 'annotation'):
                            annotation = str(field_info.annotation)
                            if 'list' in annotation.lower() or 'List' in annotation:
                                return {field_name: data}

                # Default: return first item
                return data[0] if isinstance(data[0], dict) else {"items": data}

        # For dict responses, return as-is
        if isinstance(data, dict):
            return data

        # For other types, wrap in generic structure
        return {"value": data}

    @staticmethod
    def messages_to_prompt(messages: list[ChatCompletionMessageParam]) -> str:
        """
        Convert OpenAI messages format to a simple prompt for native Ollama API.

        Args:
            messages: List of chat completion messages in OpenAI format

        Returns:
            str: Formatted prompt string for Ollama native API
        """
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n".join(prompt_parts)

    @staticmethod
    def convert_native_response_to_openai(
        native_response: dict, model: str
    ) -> MockResponse:
        """
        Convert native Ollama response to OpenAI-compatible format.

        Args:
            native_response: Response from Ollama native API
            model: Model name used for the request

        Returns:
            MockResponse: OpenAI-compatible response object
        """
        return MockResponse(native_response.get("response", ""), model)

    @staticmethod
    def set_parsed_response(response: MockResponse, parsed_model: BaseModel) -> None:
        """
        Set the parsed field on a mock response for structured output.

        Args:
            response: The mock response to update
            parsed_model: The parsed Pydantic model instance
        """
        if response.choices and response.choices[0].message:
            response.choices[0].message.parsed = parsed_model
