"""
Comprehensive test suite for the enhanced OllamaResponseConverter.

This test suite verifies all conversion scenarios work correctly and edge cases
are handled for the schema mapping functionality that fixes Graphiti pipeline
processing issues.
"""

from typing import cast
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from src.utils.ollama_response_converter import (
    MockChoice,
    MockMessage,
    MockResponse,
    OllamaResponseConverter,
)

# Test Models - Matching Graphiti Core Schemas

class ExtractedEntity(BaseModel):
    """Test model matching graphiti_core ExtractedEntity schema."""
    name: str = Field(..., description="Name of the entity")
    entity_type_id: int = Field(..., description="ID of the entity type")
    description: str = Field(default="", description="Entity description")


class ExtractedEntities(BaseModel):
    """Test model matching graphiti_core ExtractedEntities schema."""
    extracted_entities: list[ExtractedEntity] = Field(..., description="List of extracted entities")


class SimpleModel(BaseModel):
    """Simple test model for generic schema testing."""
    value: str = Field(..., description="A simple value")
    count: int = Field(default=0, description="A count value")


class ListModel(BaseModel):
    """Test model with list field for array handling."""
    items: list[str] = Field(..., description="List of items")


class ComplexModel(BaseModel):
    """Complex test model with multiple fields."""
    title: str = Field(..., description="Title")
    data: list[dict] = Field(default=[], description="Data items")
    metadata: dict = Field(default={}, description="Metadata")


class TestOllamaResponseConverter:
    """Comprehensive test suite for OllamaResponseConverter."""

    @pytest.fixture
    def converter(self):
        """Create a fresh converter instance for each test."""
        return OllamaResponseConverter()

    @pytest.fixture
    def mock_logger(self, converter):
        """Mock logger for testing logging output."""
        with patch.object(converter, 'logger') as mock_log:
            yield mock_log

    # Schema Detection Tests

    def test_extracted_entities_schema_detection_by_field(self, converter):
        """Test detection of ExtractedEntities schema by field presence."""
        assert converter._is_extracted_entities_schema(ExtractedEntities) is True

    def test_extracted_entities_schema_detection_by_name(self, converter):
        """Test detection of ExtractedEntities schema by class name."""
        class CustomExtractedEntities(BaseModel):
            other_field: str

        assert converter._is_extracted_entities_schema(CustomExtractedEntities) is True

    def test_non_extracted_entities_schema_detection(self, converter):
        """Test that other schemas are not detected as ExtractedEntities."""
        assert converter._is_extracted_entities_schema(SimpleModel) is False
        assert converter._is_extracted_entities_schema(ListModel) is False

    # Entity Conversion Tests - Core Functionality

    def test_entity_field_mapping_basic(self, converter):
        """Test basic entity → name field mapping."""
        ollama_response = [
            {"entity_type_id": 0, "entity": "HTTP Client Error Handling"},
            {"entity_type_id": 1, "entity": "Testing Pattern"}
        ]

        result = converter._convert_to_extracted_entities(ollama_response)

        assert "extracted_entities" in result
        assert len(result["extracted_entities"]) == 2

        # Check first entity
        first_entity = result["extracted_entities"][0]
        assert first_entity["name"] == "HTTP Client Error Handling"
        assert first_entity["entity_type_id"] == 0
        assert "entity" not in first_entity  # Original field should be removed

        # Check second entity
        second_entity = result["extracted_entities"][1]
        assert second_entity["name"] == "Testing Pattern"
        assert second_entity["entity_type_id"] == 1

    def test_entity_field_mapping_already_has_name(self, converter):
        """Test that existing 'name' fields are preserved."""
        ollama_response = [
            {"entity_type_id": 0, "name": "Existing Name", "entity": "Should Be Ignored"},
            {"entity_type_id": 1, "entity": "New Entity"}
        ]

        result = converter._convert_to_extracted_entities(ollama_response)

        assert len(result["extracted_entities"]) == 2

        # First entity should keep existing name
        first_entity = result["extracted_entities"][0]
        assert first_entity["name"] == "Existing Name"
        assert "entity" in first_entity  # Should preserve both when name exists

        # Second entity should map entity → name
        second_entity = result["extracted_entities"][1]
        assert second_entity["name"] == "New Entity"
        assert "entity" not in second_entity

    def test_entity_field_mapping_empty_array(self, converter):
        """Test handling of empty entity arrays."""
        result = converter._convert_to_extracted_entities([])

        assert result == {"extracted_entities": []}

    def test_entity_field_mapping_malformed_data(self, converter, mock_logger):
        """Test graceful handling of malformed entity data."""
        ollama_response = [
            {"entity_type_id": 0, "entity": "Valid Entity"},
            "invalid_item",  # This should be handled gracefully
            42,  # This should also be handled gracefully
            {"entity_type_id": 1, "entity": "Another Valid Entity"}
        ]

        result = converter._convert_to_extracted_entities(ollama_response)

        # Should only process valid dict entities, filtering out invalid items
        assert len(result["extracted_entities"]) == 2
        assert result["extracted_entities"][0]["name"] == "Valid Entity"
        assert result["extracted_entities"][1]["name"] == "Another Valid Entity"

        # Should log warnings for invalid items
        mock_logger.warning.assert_called()

    def test_entity_field_mapping_dict_already_wrapped(self, converter):
        """Test handling of already wrapped ExtractedEntities format."""
        ollama_response = {
            "extracted_entities": [
                {"entity_type_id": 0, "entity": "Should Be Mapped"},
                {"entity_type_id": 1, "name": "Already Correct"}
            ]
        }

        result = converter._convert_to_extracted_entities(ollama_response)

        assert len(result["extracted_entities"]) == 2
        assert result["extracted_entities"][0]["name"] == "Should Be Mapped"
        assert result["extracted_entities"][1]["name"] == "Already Correct"

    def test_entity_field_mapping_single_entity_dict(self, converter):
        """Test wrapping single entity dict into array format."""
        ollama_response = {"entity_type_id": 0, "entity": "Single Entity"}

        result = converter._convert_to_extracted_entities(ollama_response)

        assert "extracted_entities" in result
        assert len(result["extracted_entities"]) == 1
        assert result["extracted_entities"][0]["name"] == "Single Entity"

    def test_entity_field_mapping_unexpected_type(self, converter, mock_logger):
        """Test handling of unexpected data types."""
        result = converter._convert_to_extracted_entities("unexpected_string")

        assert result == {"extracted_entities": []}
        mock_logger.warning.assert_called()

    # Format Conversion Tests

    def test_full_conversion_extracted_entities(self, converter):
        """Test complete conversion flow for ExtractedEntities."""
        ollama_response = [
            {"entity_type_id": 0, "entity": "Memory System"},
            {"entity_type_id": 1, "entity": "Schema Validation"}
        ]

        result = converter.convert_structured_response(ollama_response, ExtractedEntities)

        # Should be properly formatted for ExtractedEntities model
        model_instance = ExtractedEntities(**result)
        assert len(model_instance.extracted_entities) == 2
        assert model_instance.extracted_entities[0].name == "Memory System"
        assert model_instance.extracted_entities[1].name == "Schema Validation"

    def test_full_conversion_simple_model_list(self, converter):
        """Test conversion of list response for simple model."""
        ollama_response = [{"value": "test_value", "count": 5}]

        result = converter.convert_structured_response(ollama_response, SimpleModel)

        # Should return first item for single-item list
        assert result == {"value": "test_value", "count": 5}

    def test_full_conversion_simple_model_dict(self, converter):
        """Test conversion of dict response for simple model."""
        ollama_response = {"value": "test_value", "count": 10}

        result = converter.convert_structured_response(ollama_response, SimpleModel)

        # Should return dict as-is
        assert result == {"value": "test_value", "count": 10}

    def test_full_conversion_list_model_array(self, converter):
        """Test conversion for model expecting list field."""
        ollama_response = ["item1", "item2", "item3"]

        result = converter.convert_structured_response(ollama_response, ListModel)

        # Should detect list field and wrap appropriately
        assert "items" in result
        assert result["items"] == ["item1", "item2", "item3"]

    # Error Handling Tests

    def test_conversion_fallback_on_exception(self, converter, mock_logger):
        """Test graceful fallback when conversion fails."""
        # Mock the conversion method to raise an exception
        with patch.object(converter, '_is_extracted_entities_schema', side_effect=Exception("Test error")):
            ollama_response = {"test": "data"}

            result = converter.convert_structured_response(ollama_response, SimpleModel)

            # Should fall back to original data
            assert result == {"test": "data"}
            mock_logger.warning.assert_called()

    def test_conversion_fallback_list_single_item(self, converter):
        """Test fallback behavior for single-item list."""
        with patch.object(converter, '_convert_generic_schema', side_effect=Exception("Test error")):
            ollama_response = [{"test": "data"}]

            result = converter.convert_structured_response(ollama_response, SimpleModel)

            # Should return first item
            assert result == {"test": "data"}

    def test_conversion_fallback_list_multiple_items(self, converter):
        """Test fallback behavior for multi-item list."""
        with patch.object(converter, '_convert_generic_schema', side_effect=Exception("Test error")):
            ollama_response = [{"test": "data1"}, {"test": "data2"}]

            result = converter.convert_structured_response(ollama_response, SimpleModel)

            # Should wrap in items structure
            assert result == {"items": [{"test": "data1"}, {"test": "data2"}]}

    def test_conversion_failure_with_invalid_data(self, converter):
        """Test exception handling for truly invalid data."""
        with pytest.raises(ValueError, match="Cannot convert None response data"):
            converter.convert_structured_response(None, SimpleModel)

    def test_logging_output_debug_messages(self, converter, mock_logger):
        """Test that appropriate debug logging occurs."""
        ollama_response = [{"entity_type_id": 0, "entity": "Test Entity"}]

        converter.convert_structured_response(ollama_response, ExtractedEntities)

        # Should log debug messages
        mock_logger.debug.assert_called()

    # Generic Schema Conversion Tests

    def test_generic_conversion_single_list_item(self, converter):
        """Test generic conversion with single list item."""
        ollama_response = [{"value": "single_item"}]

        result = converter._convert_generic_schema(ollama_response, SimpleModel)

        assert result == {"value": "single_item"}

    def test_generic_conversion_multiple_list_items_with_list_field(self, converter):
        """Test generic conversion detecting list field in target schema."""
        ollama_response = ["item1", "item2", "item3"]

        result = converter._convert_generic_schema(ollama_response, ListModel)

        assert result == {"items": ["item1", "item2", "item3"]}

    def test_generic_conversion_dict_passthrough(self, converter):
        """Test generic conversion with dict data."""
        ollama_response = {"key": "value", "number": 42}

        result = converter._convert_generic_schema(ollama_response, SimpleModel)

        assert result == {"key": "value", "number": 42}

    def test_generic_conversion_primitive_value(self, converter):
        """Test generic conversion with primitive value."""
        ollama_response = "simple_string"

        result = converter._convert_generic_schema(ollama_response, SimpleModel)

        assert result == {"value": "simple_string"}

    # Integration Tests with Real Schemas

    def test_integration_with_complex_ollama_response(self, converter):
        """Test with complex, realistic Ollama response."""
        ollama_response = [
            {
                "entity_type_id": 0,
                "entity": "Graphiti Memory System",
                "description": "AI memory management system"
            },
            {
                "entity_type_id": 1,
                "entity": "Schema Validation Pipeline",
                "description": "Background processing component"
            },
            {
                "entity_type_id": 2,
                "entity": "Ollama Integration",
                "description": "LLM client integration"
            }
        ]

        result = converter.convert_structured_response(ollama_response, ExtractedEntities)

        # Verify complete conversion
        model_instance = ExtractedEntities(**result)
        assert len(model_instance.extracted_entities) == 3

        entities = model_instance.extracted_entities
        assert entities[0].name == "Graphiti Memory System"
        assert entities[0].description == "AI memory management system"
        assert entities[1].name == "Schema Validation Pipeline"
        assert entities[2].name == "Ollama Integration"

    def test_performance_with_large_response(self, converter):
        """Test performance with large response data."""
        # Create large response with 1000 entities
        large_response = [
            {"entity_type_id": i, "entity": f"Entity {i}"}
            for i in range(1000)
        ]

        # This should complete without timeout (pytest default is 60s)
        result = converter.convert_structured_response(large_response, ExtractedEntities)

        assert len(result["extracted_entities"]) == 1000
        assert result["extracted_entities"][0]["name"] == "Entity 0"
        assert result["extracted_entities"][999]["name"] == "Entity 999"

    def test_memory_usage_and_cleanup(self, converter):
        """Test that converter doesn't leak memory during operations."""
        import gc

        # Get initial object count
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform multiple conversions
        for i in range(100):
            ollama_response = [{"entity_type_id": i, "entity": f"Entity {i}"}]
            converter.convert_structured_response(ollama_response, ExtractedEntities)

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count shouldn't grow significantly (allowing for some test overhead)
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Memory leak detected: {object_growth} new objects"

    def test_thread_safety_basic(self, converter):
        """Basic test for thread safety of converter operations."""
        import threading

        results = []
        errors = []

        def worker():
            try:
                ollama_response = [{"entity_type_id": 1, "entity": "Concurrent Entity"}]
                result = converter.convert_structured_response(ollama_response, ExtractedEntities)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=worker) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All threads should complete successfully
        assert len(errors) == 0
        assert len(results) == 10

        # All results should be identical
        for result in results:
            assert len(result["extracted_entities"]) == 1
            assert result["extracted_entities"][0]["name"] == "Concurrent Entity"


class TestMockMessage:
    """Test suite for MockMessage utility class."""

    def test_mock_message_initialization(self):
        """Test MockMessage initialization and basic properties."""
        content = "Test message content"
        message = MockMessage(content)

        assert message.content == content
        assert message.role == "assistant"
        assert message.parsed is None
        assert message.refusal is None

    def test_mock_message_model_dump(self):
        """Test MockMessage model_dump method."""
        content = "Test content"
        message = MockMessage(content)

        result = message.model_dump()

        expected = {
            "content": content,
            "role": "assistant",
            "parsed": None,
            "refusal": None,
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": None,
        }

        assert result == expected

    def test_mock_message_with_parsed_content(self):
        """Test MockMessage with parsed content."""
        message = MockMessage("test")
        test_model = SimpleModel(value="test", count=42)
        message.parsed = test_model

        dump = message.model_dump()
        assert dump["parsed"] == test_model


class TestMockChoice:
    """Test suite for MockChoice utility class."""

    def test_mock_choice_initialization(self):
        """Test MockChoice initialization."""
        content = "Choice content"
        choice = MockChoice(content)

        assert isinstance(choice.message, MockMessage)
        assert choice.message.content == content
        assert choice.index == 0
        assert choice.finish_reason == "stop"


class TestMockResponse:
    """Test suite for MockResponse utility class."""

    def test_mock_response_initialization(self):
        """Test MockResponse initialization."""
        content = "Response content"
        model = "llama3.2"
        response = MockResponse(content, model)

        assert len(response.choices) == 1
        assert isinstance(response.choices[0], MockChoice)
        assert response.choices[0].message.content == content  # type: ignore
        assert response.model == model
        assert response.object == "chat.completion"
        assert isinstance(response.id, str)
        assert response.id.startswith("chatcmpl-")
        assert isinstance(response.created, int)


class TestStaticUtilityMethods:
    """Test suite for static utility methods."""

    def test_messages_to_prompt_basic(self):
        """Test basic message to prompt conversion."""
        messages = cast(list[ChatCompletionMessageParam], [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ])

        result = OllamaResponseConverter.messages_to_prompt(messages)

        expected = "System: You are a helpful assistant\nUser: Hello\nAssistant: Hi there!"
        assert result == expected

    def test_messages_to_prompt_empty_content(self):
        """Test message to prompt conversion with empty content."""
        messages = cast(list[ChatCompletionMessageParam], [
            {"role": "user"},  # Missing content
            {"role": "assistant", "content": "Response"}
        ])

        result = OllamaResponseConverter.messages_to_prompt(messages)

        expected = "User: \nAssistant: Response"
        assert result == expected

    def test_messages_to_prompt_unknown_role(self):
        """Test message to prompt conversion with unknown role."""
        messages = cast(list[ChatCompletionMessageParam], [
            {"role": "unknown", "content": "Some content"},
            {"role": "user", "content": "User message"}
        ])

        result = OllamaResponseConverter.messages_to_prompt(messages)

        # Unknown roles are ignored
        expected = "User: User message"
        assert result == expected

    def test_convert_native_response_to_openai(self):
        """Test conversion of native Ollama response to OpenAI format."""
        native_response = {"response": "This is the response"}
        model = "llama3.2"

        result = OllamaResponseConverter.convert_native_response_to_openai(
            native_response, model
        )

        assert isinstance(result, MockResponse)
        assert result.model == model
        assert result.choices[0].message.content == "This is the response"  # type: ignore

    def test_convert_native_response_empty(self):
        """Test conversion with empty native response."""
        native_response = {}
        model = "llama3.2"

        result = OllamaResponseConverter.convert_native_response_to_openai(
            native_response, model
        )

        assert isinstance(result, MockResponse)
        assert result.model == model
        assert result.choices[0].message is not None
        assert result.choices[0].message.content == ""

    def test_set_parsed_response(self):
        """Test setting parsed response on mock response."""
        response = MockResponse("content", "model")
        parsed_model = SimpleModel(value="test", count=5)

        OllamaResponseConverter.set_parsed_response(response, parsed_model)

        assert response.choices[0].message is not None
        assert response.choices[0].message.parsed == parsed_model

    def test_set_parsed_response_no_choices(self):
        """Test setting parsed response on response with no choices."""
        response = MockResponse("content", "model")
        response.choices = []  # Clear choices
        parsed_model = SimpleModel(value="test", count=5)

        # Should not raise an exception
        OllamaResponseConverter.set_parsed_response(response, parsed_model)

    def test_set_parsed_response_no_message(self):
        """Test setting parsed response on choice with no message."""
        # Create a MockChoice that simulates no message condition
        class MockChoiceNoMessage:
            def __init__(self):
                self.message = None
                self.index = 0
                self.finish_reason = "stop"

        response = MockResponse("content", "model")
        response.choices[0] = MockChoiceNoMessage()  # type: ignore
        parsed_model = SimpleModel(value="test", count=5)

        # Should not raise an exception
        OllamaResponseConverter.set_parsed_response(response, parsed_model)


class TestAdditionalEdgeCases:
    """Additional edge case tests to achieve 100% coverage."""

    @pytest.fixture
    def converter(self):
        """Create a fresh converter instance for each test."""
        return OllamaResponseConverter()

    @pytest.fixture
    def mock_logger(self, converter):
        """Mock logger for testing logging output."""
        with patch.object(converter, 'logger') as mock_log:
            yield mock_log

    def test_conversion_with_actual_exception_in_generic_schema(self, converter, mock_logger):
        """Test exception handling in generic schema conversion."""
        with patch.object(converter, '_convert_generic_schema', side_effect=Exception("Test error")):
            ollama_response = [{"test": "data1"}, {"test": "data2"}]

            result = converter.convert_structured_response(ollama_response, SimpleModel)

            # Should fall back to items structure for multi-item list
            assert result == {"items": [{"test": "data1"}, {"test": "data2"}]}

    def test_conversion_with_exception_and_invalid_fallback(self, converter):
        """Test exception handling when fallback also fails."""
        with patch.object(converter, '_is_extracted_entities_schema', side_effect=Exception("Test error")):
            # Empty list should cause fallback to fail
            ollama_response = []

            with pytest.raises(ValueError, match="Cannot convert response data"):
                converter.convert_structured_response(ollama_response, SimpleModel)

    def test_generic_conversion_list_non_dict_single_item(self, converter):
        """Test generic conversion with single non-dict list item."""
        ollama_response = ["simple_string"]

        result = converter._convert_generic_schema(ollama_response, SimpleModel)

        assert result == {"value": "simple_string"}

    def test_generic_conversion_primitive_non_string(self, converter):
        """Test generic conversion with primitive non-string value."""
        ollama_response = 42

        result = converter._convert_generic_schema(ollama_response, SimpleModel)

        assert result == {"value": 42}

    def test_extracted_entities_already_wrapped_non_list(self, converter):
        """Test extracted entities conversion with non-list wrapped format."""
        ollama_response = {
            "extracted_entities": "not_a_list"
        }

        result = converter._convert_to_extracted_entities(ollama_response)

        # Should return as-is since it's not a list
        assert result == {"extracted_entities": "not_a_list"}

    def test_extracted_entities_with_non_dict_item_in_wrapped_list(self, converter):
        """Test extracted entities with non-dict items in wrapped list."""
        ollama_response = {
            "extracted_entities": [
                {"entity_type_id": 0, "entity": "Valid Entity"},
                "invalid_item",
                {"entity_type_id": 1, "entity": "Another Valid Entity"}
            ]
        }

        result = converter._convert_to_extracted_entities(ollama_response)

        # Should preserve non-dict items as-is
        assert len(result["extracted_entities"]) == 3
        assert result["extracted_entities"][0]["name"] == "Valid Entity"
        assert result["extracted_entities"][1] == "invalid_item"
        assert result["extracted_entities"][2]["name"] == "Another Valid Entity"

    def test_generic_conversion_multiple_non_dict_items_no_list_fields(self, converter):
        """Test generic conversion with multiple non-dict items and no list fields in target schema."""
        ollama_response = ["item1", "item2", "item3"]

        # SimpleModel has no list fields, so should wrap in items structure
        result = converter._convert_generic_schema(ollama_response, SimpleModel)

        assert result == {"items": ["item1", "item2", "item3"]}
