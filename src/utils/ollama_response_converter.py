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
                    # Handle field name mapping: "entity" -> "name"
                    mapped_item = item.copy()
                    if "entity" in mapped_item and "name" not in mapped_item:
                        mapped_item["name"] = mapped_item.pop("entity")
                        self.logger.debug(f"Mapped entity field: {mapped_item.get('name')}")
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
