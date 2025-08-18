"""
Tests for Phase 2: Enhanced Structured Response Handling in OllamaClient.

This module tests the enhanced structured completion functionality that was
implemented in Phase 2 of the Ollama OpenAI compatibility fix.
"""

from unittest.mock import MagicMock, patch

import pytest
from graphiti_core.llm_client.config import LLMConfig
from pydantic import BaseModel, Field

from src.ollama_client import OllamaClient


class EntityTestModel(BaseModel):
    """Test entity for structured response validation."""

    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    confidence: float = Field(..., description="Confidence score")


class EntityExtractionModel(BaseModel):
    """Test model for entity extraction responses (matches Graphiti expected format)."""

    entity_name: str = Field(..., description="Name of the extracted entity")
    entity_type_id: int = Field(..., description="ID of the entity type")
    description: str = Field(default="", description="Entity description")


class ExtractedEntity(BaseModel):
    """Test model matching graphiti_core ExtractedEntity schema."""

    name: str = Field(..., description="Name of the extracted entity")
    entity_type_id: int = Field(..., description="ID of the entity type")


class ExtractedEntities(BaseModel):
    """Test model matching graphiti_core ExtractedEntities schema."""

    extracted_entities: list[ExtractedEntity] = Field(..., description="List of extracted entities")


class TestOllamaPhase2StructuredResponses:
    """Test suite for Phase 2 enhanced structured response handling."""

    @pytest.fixture
    def ollama_config(self):
        """Create test configuration for Ollama."""
        return LLMConfig(
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="test-key",
            temperature=0.1,
        )

    @pytest.fixture
    def ollama_client(self, ollama_config):
        """Create OllamaClient instance for testing."""
        return OllamaClient(
            config=ollama_config, model_parameters={"num_ctx": 4096, "top_p": 0.9}
        )

    @pytest.mark.asyncio
    async def test_structured_completion_with_valid_json(self, ollama_client):
        """Test that valid JSON responses are properly parsed and structured."""
        # Mock the native API call to return valid JSON
        json_response = '{"name": "John Doe", "type": "person", "confidence": 0.95}'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": json_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify the response structure
            assert result is not None
            assert result.choices[0].message.content == json_response

            # Verify that parsed field is populated with the correct model
            parsed = result.choices[0].message.parsed
            assert parsed is not None
            assert isinstance(parsed, EntityTestModel)
            assert parsed.name == "John Doe"
            assert parsed.type == "person"
            assert parsed.confidence == 0.95

    @pytest.mark.asyncio
    async def test_structured_completion_with_invalid_json(self, ollama_client):
        """Test graceful handling of invalid JSON responses."""
        # Mock the native API call to return invalid JSON
        invalid_json = 'This is not valid JSON: {"name": "incomplete'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": invalid_json}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify that parsing failed gracefully
            assert result is not None
            assert result.choices[0].message.content == invalid_json
            assert result.choices[0].message.parsed is None

    @pytest.mark.asyncio
    async def test_structured_completion_with_validation_error(self, ollama_client):
        """Test handling of JSON that doesn't match the response model."""
        # Mock the native API call to return JSON with wrong structure
        invalid_structure = '{"wrong_field": "value", "another_field": 123}'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": invalid_structure}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify that validation failed gracefully
            assert result is not None
            assert result.choices[0].message.content == invalid_structure
            assert result.choices[0].message.parsed is None

    @pytest.mark.asyncio
    async def test_structured_completion_with_non_json_content(self, ollama_client):
        """Test handling of non-JSON responses."""
        # Mock the native API call to return regular text
        text_response = "This is a regular text response without JSON structure."

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": text_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify that non-JSON content is handled gracefully
            assert result is not None
            assert result.choices[0].message.content == text_response
            assert result.choices[0].message.parsed is None

    @pytest.mark.asyncio
    async def test_mock_message_model_dump_with_parsed_data(self, ollama_client):
        """Test that MockMessage properly implements model_dump() with parsed data."""
        # Mock the native API call to return structured JSON data
        json_response = '{"name": "Test Entity", "type": "test", "confidence": 0.8}'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": json_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Test model_dump method with parsed data
            dump = result.choices[0].message.model_dump()

            assert isinstance(dump, dict)
            assert dump["content"] == json_response
            assert dump["role"] == "assistant"
            assert dump["parsed"] is not None
            assert isinstance(dump["parsed"], EntityTestModel)
            assert dump["refusal"] is None

    @pytest.mark.asyncio
    async def test_logging_during_structured_parsing(self, ollama_client, caplog):
        """Test that appropriate log messages are generated during parsing."""
        import logging

        caplog.set_level(logging.WARNING)

        # Mock the native API call to return invalid JSON
        invalid_json = '{"invalid": json'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": invalid_json}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify warning was logged
            assert (
                "Failed to parse JSON response from Ollama model llama3.1:8b"
                in caplog.text
            )

    @pytest.mark.asyncio
    async def test_structured_completion_preserves_original_functionality(
        self, ollama_client
    ):
        """Test that the enhanced method still works when JSON parsing is not applicable."""
        # Mock the native API call to return a standard text response
        text_response = "Standard text response without JSON"

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": text_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Generate text"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityTestModel,
            )

            # Verify original functionality is preserved
            assert result is not None
            assert result.choices[0].message.content == text_response
            assert result.choices[0].message.parsed is None

    @pytest.mark.asyncio
    async def test_structured_completion_with_json_array(self, ollama_client):
        """CRITICAL TEST: Test that JSON array responses are properly parsed (Task ID: 004)."""
        # Mock the native API call to return JSON array (typical Ollama entity extraction response)
        json_array_response = '[{"entity_name": "PR #47", "entity_type_id": 0, "description": "GitHub pull request"}]'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": json_array_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entities"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityExtractionModel,
            )

            # Verify the response structure
            assert result is not None
            assert result.choices[0].message.content == json_array_response

            # CRITICAL: Verify that parsed field is populated from the JSON array
            parsed = result.choices[0].message.parsed
            assert parsed is not None
            assert isinstance(parsed, EntityExtractionModel)
            assert parsed.entity_name == "PR #47"
            assert parsed.entity_type_id == 0
            assert parsed.description == "GitHub pull request"

    @pytest.mark.asyncio
    async def test_structured_completion_with_empty_json_array(self, ollama_client):
        """Test handling of empty JSON array responses."""
        # Mock the native API call to return empty JSON array
        empty_array_response = "[]"

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": empty_array_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entities"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityExtractionModel,
            )

            # Verify the response handles empty array gracefully
            assert result is not None
            assert result.choices[0].message.content == empty_array_response
            # For empty arrays, we expect parsed to be None since we can't create
            # a valid model instance from an empty array
            parsed = result.choices[0].message.parsed
            assert parsed is None, "Empty JSON arrays should result in parsed=None"

    @pytest.mark.asyncio
    async def test_structured_completion_with_multiple_item_json_array(
        self, ollama_client
    ):
        """Test handling of JSON arrays with multiple items (uses first item)."""
        # Mock the native API call to return JSON array with multiple entities
        multi_item_array = '[{"entity_name": "First Entity", "entity_type_id": 1, "description": "First"}, {"entity_name": "Second Entity", "entity_type_id": 2, "description": "Second"}]'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": multi_item_array}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entities"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityExtractionModel,
            )

            # Verify the response uses the first item from the array
            assert result is not None
            assert result.choices[0].message.content == multi_item_array

            # Should use the first item from the array
            parsed = result.choices[0].message.parsed
            assert parsed is not None
            assert isinstance(parsed, EntityExtractionModel)
            assert parsed.entity_name == "First Entity"
            assert parsed.entity_type_id == 1
            assert parsed.description == "First"

    @pytest.mark.asyncio
    async def test_json_array_vs_object_regression(self, ollama_client):
        """Regression test: Ensure both JSON objects and arrays work correctly."""
        test_cases = [
            # JSON object (original functionality)
            (
                '{"name": "Test Entity", "type": "test_type", "confidence": 0.9}',
                EntityTestModel,
                "object",
            ),
            # JSON array (new functionality - CRITICAL FIX)
            (
                '[{"entity_name": "Array Entity", "entity_type_id": 5, "description": "From array"}]',
                EntityExtractionModel,
                "array",
            ),
        ]

        for json_response, model_class, test_type in test_cases:
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"response": json_response}
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response

                result = await ollama_client._create_structured_completion(
                    model="llama3.1:8b",
                    messages=[{"role": "user", "content": f"Test {test_type}"}],
                    temperature=0.1,
                    max_tokens=100,
                    response_model=model_class,
                )

                # Both should work without errors
                assert result is not None, f"Failed for {test_type}: {json_response}"
                assert result.choices[0].message.content == json_response
                assert result.choices[0].message.parsed is not None, (
                    f"Parsing failed for {test_type}: {json_response}"
                )
                assert isinstance(result.choices[0].message.parsed, model_class)

    @pytest.mark.asyncio
    async def test_invalid_json_formats_fallback(self, ollama_client):
        """Test graceful fallback for various invalid JSON formats."""
        invalid_cases = [
            '"not json"',  # String (not object or array)
            "not json at all",  # Plain text
            '{"incomplete": json',  # Malformed JSON
            "",  # Empty response
            "   ",  # Whitespace only
        ]

        for invalid_json in invalid_cases:
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"response": invalid_json}
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response

                result = await ollama_client._create_structured_completion(
                    model="llama3.1:8b",
                    messages=[{"role": "user", "content": "Test invalid"}],
                    temperature=0.1,
                    max_tokens=100,
                    response_model=EntityTestModel,
                )

                # Should handle gracefully with parsed=None
                assert result is not None, f"Failed to handle: {invalid_json!r}"
                assert result.choices[0].message.content == invalid_json
                assert result.choices[0].message.parsed is None, (
                    f"Should have parsed=None for: {invalid_json!r}"
                )

    def test_mock_message_type_annotations(self):
        """Test that MockMessage has proper type annotations for parsed field."""
        from src.ollama_client import OllamaClient

        # This test verifies that the MockMessage class properly accepts BaseModel instances
        # in the parsed field, which was the main issue fixed in the type annotations
        client = OllamaClient()

        # Create a mock response to access MockMessage class
        mock_response = client._response_converter.convert_native_response_to_openai(
            {"response": "test"}, "test-model"
        )
        message = mock_response.choices[0].message

        # Test that we can assign a BaseModel instance to parsed
        test_entity = EntityTestModel(name="test", type="test", confidence=0.5)
        message.parsed = test_entity

        # Verify the assignment worked
        assert message.parsed == test_entity
        assert isinstance(message.parsed, EntityTestModel)

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, ollama_client):
        """Test that timeout errors are properly handled with helpful messages."""
        import httpx

        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock a timeout exception
            mock_post.side_effect = httpx.TimeoutException("Request timed out")

            with pytest.raises(TimeoutError) as exc_info:
                await ollama_client._create_completion(
                    model="llama3.1:8b",
                    messages=[{"role": "user", "content": "Test timeout"}],
                    temperature=0.1,
                    max_tokens=100,
                )

            # Verify the error message is helpful
            assert "timed out for model 'llama3.1:8b'" in str(exc_info.value)
            assert "reducing context size" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_http_status_error_handling(self, ollama_client):
        """Test that HTTP status errors are properly handled with specific messages."""
        import httpx

        # Test 404 error
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.text = "Model not found"

            http_error = httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(),
                response=mock_response
            )
            mock_post.side_effect = http_error

            with pytest.raises(ValueError) as exc_info:
                await ollama_client._create_completion(
                    model="nonexistent-model",
                    messages=[{"role": "user", "content": "Test 404"}],
                    temperature=0.1,
                    max_tokens=100,
                )

            # Verify the error message is specific
            assert "Model 'nonexistent-model' not found on Ollama server" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, ollama_client):
        """Test that connection errors are properly handled with helpful messages."""
        import httpx

        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock a connection error
            mock_post.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(ConnectionError) as exc_info:
                await ollama_client._create_completion(
                    model="llama3.1:8b",
                    messages=[{"role": "user", "content": "Test connection"}],
                    temperature=0.1,
                    max_tokens=100,
                )

            # Verify the error message is helpful
            assert "Cannot connect to Ollama server" in str(exc_info.value)
            assert "Is Ollama running?" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extracted_entities_schema_mapping(self, ollama_client):
        """CRITICAL TEST: Test schema mapping for ExtractedEntities with field name conversion."""
        # Mock Ollama response with "entity" field that needs mapping to "name"
        ollama_response = '[{"entity_type_id": 0, "entity": "HTTP Client Error Handling"}, {"entity_type_id": 1, "entity": "Testing Pattern"}]'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": ollama_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="gpt-oss:latest",
                messages=[{"role": "user", "content": "Extract entities"}],
                temperature=0.1,
                max_tokens=100,
                response_model=ExtractedEntities,
            )

            # Verify the response structure
            assert result is not None
            assert result.choices[0].message.content == ollama_response

            # CRITICAL: Verify that parsed field contains properly mapped ExtractedEntities
            parsed = result.choices[0].message.parsed
            assert parsed is not None
            assert isinstance(parsed, ExtractedEntities)

            # Verify the structure and field mapping
            assert hasattr(parsed, 'extracted_entities')
            assert len(parsed.extracted_entities) == 2

            # Check first entity
            first_entity = parsed.extracted_entities[0]
            assert isinstance(first_entity, ExtractedEntity)
            assert first_entity.name == "HTTP Client Error Handling"  # "entity" -> "name" mapping
            assert first_entity.entity_type_id == 0

            # Check second entity
            second_entity = parsed.extracted_entities[1]
            assert isinstance(second_entity, ExtractedEntity)
            assert second_entity.name == "Testing Pattern"  # "entity" -> "name" mapping
            assert second_entity.entity_type_id == 1

    @pytest.mark.asyncio
    async def test_extracted_entities_empty_array_handling(self, ollama_client):
        """Test handling of empty arrays for ExtractedEntities model."""
        # Mock Ollama response with empty array
        ollama_response = '[]'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": ollama_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="gpt-oss:latest",
                messages=[{"role": "user", "content": "Extract entities"}],
                temperature=0.1,
                max_tokens=100,
                response_model=ExtractedEntities,
            )

            # Verify the response structure
            assert result is not None
            assert result.choices[0].message.content == ollama_response

            # Empty arrays should now result in valid model with empty list (enhanced behavior)
            parsed = result.choices[0].message.parsed
            assert parsed is not None
            assert isinstance(parsed, ExtractedEntities)
            assert parsed.extracted_entities == []

    @pytest.mark.asyncio
    async def test_non_extracted_entities_array_unchanged(self, ollama_client):
        """Test that non-ExtractedEntities models still work with original array handling."""
        # Mock Ollama response for non-ExtractedEntities model (should use first item)
        ollama_response = '[{"entity_name": "First Item", "entity_type_id": 5, "description": "First"}]'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": ollama_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "Extract entity"}],
                temperature=0.1,
                max_tokens=100,
                response_model=EntityExtractionModel,  # Not ExtractedEntities
            )

            # Verify the response structure
            assert result is not None
            assert result.choices[0].message.content == ollama_response

            # Should use first item directly (original behavior)
            parsed = result.choices[0].message.parsed
            assert parsed is not None
            assert isinstance(parsed, EntityExtractionModel)
            assert parsed.entity_name == "First Item"
            assert parsed.entity_type_id == 5
            assert parsed.description == "First"

    @pytest.mark.asyncio
    async def test_schema_mapping_robustness(self, ollama_client):
        """Test schema mapping handles various field combinations correctly."""
        # Mock response with mixed field presence
        ollama_response = '[{"entity_type_id": 0, "entity": "Has Entity Field"}, {"entity_type_id": 1, "name": "Already Has Name", "entity": "Should Prefer Name"}]'

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": ollama_response}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await ollama_client._create_structured_completion(
                model="gpt-oss:latest",
                messages=[{"role": "user", "content": "Extract entities"}],
                temperature=0.1,
                max_tokens=100,
                response_model=ExtractedEntities,
            )

            # Verify parsing succeeded
            parsed = result.choices[0].message.parsed
            assert parsed is not None
            assert len(parsed.extracted_entities) == 2

            # First entity: should map "entity" to "name"
            first_entity = parsed.extracted_entities[0]
            assert first_entity.name == "Has Entity Field"
            assert first_entity.entity_type_id == 0

            # Second entity: should preserve existing "name" field (not override with "entity")
            second_entity = parsed.extracted_entities[1]
            assert second_entity.name == "Already Has Name"  # Should keep original "name"
            assert second_entity.entity_type_id == 1
