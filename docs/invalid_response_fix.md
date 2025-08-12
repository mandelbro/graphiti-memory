# Invalid Response from LLM Error - Research and Fix Plan

## Executive Summary

The error "Invalid response from LLM" occurs when the Graphiti MCP server attempts to use structured completions with Ollama, but Ollama doesn't support OpenAI's structured output format. While the custom `OllamaClient` was designed to handle this, there's a gap in the structured response handling that causes the validation to fail.

## Root Cause Analysis

### Problem Description
```
ERROR - Error in generating LLM response: Invalid response from LLM: {'content': '', 'refusal': None, 'role': 'assistant', 'annotations': None, 'audio': None, 'function_call': None, 'tool_calls': None, 'parsed': None}
```

### Technical Root Cause

1. **Structured Completion Request**: The Graphiti core library requests structured completions for entity extraction by calling `_create_structured_completion()`.

2. **Ollama Compatibility Gap**: Ollama's OpenAI-compatible API (`/v1/chat/completions`) doesn't properly support structured output features, particularly the `parsed` field that OpenAI uses for structured responses.

3. **Validation Failure**: The base `openai_base_client.py` validates structured responses by checking for either `response_object.parsed` or `response_object.refusal` to be set. When both are `None`, it raises the "Invalid response from LLM" error.

4. **Mock Response Issue**: The `OllamaClient`'s `MockMessage` class correctly sets `parsed = None` and `refusal = None`, but this causes the validation in `_handle_structured_response()` to fail.

### Code Flow Analysis

```
graphiti_core calls -> _generate_response()
-> _create_structured_completion() (OllamaClient)
-> _create_completion() (fallback)
-> MockResponse with parsed=None
-> _handle_structured_response()
-> Exception: "Invalid response from LLM"
```

## Investigation Details

### Files Examined
- `/Users/cmontes/AI/graphiti/mcp_server/src/ollama_client.py` - Custom Ollama wrapper
- `/Users/cmontes/AI/graphiti/mcp_server/src/graphiti_mcp_server.py` - MCP server configuration
- `/Users/cmontes/AI/graphiti/graphiti_core/llm_client/openai_base_client.py` - Base LLM client
- `/Users/cmontes/AI/graphiti/mcp_server/config/providers/ollama.yml` - Ollama configuration

### Key Findings

1. **OllamaClient Design**: The `OllamaClient` is correctly designed to use Ollama's native `/api/generate` endpoint instead of the OpenAI-compatible `/v1/chat/completions`.

2. **Structured Output Gap**: The `_create_structured_completion()` method falls back to regular completion but doesn't handle the structured output parsing correctly.

3. **Response Validation**: The base client expects structured responses to have either `parsed` or `refusal` populated, but Ollama responses have neither.

4. **Configuration Conflict**: The base URL in configuration includes `/v1` but the OllamaClient strips it to use the native API.

## Proposed Solution

### Fix 1: Enhanced OllamaClient Structured Response Handling

Modify the `OllamaClient` to properly handle structured completions by parsing the text response and populating the `parsed` field:

```python
async def _create_structured_completion(
    self,
    model: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float | None,
    max_tokens: int,
    response_model: type[BaseModel],
):
    """Create a structured completion with proper parsing for Ollama."""
    # Get regular completion from Ollama
    response = await self._create_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Parse the text response as JSON and create structured response
    try:
        content = response.choices[0].message.content
        if content:
            parsed_data = json.loads(content)
            parsed_model = response_model(**parsed_data)

            # Update the mock message with parsed data
            response.choices[0].message.parsed = parsed_model
            response.choices[0].message.content = content

        return response
    except (json.JSONDecodeError, ValidationError) as e:
        # If parsing fails, fall back to regular completion
        logger.warning(f"Failed to parse structured response from Ollama: {e}")
        return response
```

### Fix 2: Alternative Bypass Strategy

If structured parsing proves problematic, modify the `_handle_structured_response` logic to handle Ollama responses:

```python
def _handle_structured_response(self, response: Any) -> dict[str, Any]:
    """Handle structured response parsing and validation."""
    response_object = response.choices[0].message

    if response_object.parsed:
        return response_object.parsed.model_dump()
    elif response_object.refusal:
        raise RefusalError(response_object.refusal)
    elif hasattr(self, '_is_ollama_client') and self._is_ollama_client:
        # For Ollama, attempt to parse content as JSON
        content = response_object.content or '{}'
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"content": content}
    else:
        raise Exception(f'Invalid response from LLM: {response_object.model_dump()}')
```

### Fix 3: Configuration Validation

Add validation to ensure Ollama is properly configured and models are available:

```python
async def validate_ollama_setup(self):
    """Validate that Ollama is running and models are available."""
    try:
        # Check if Ollama is responding
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.ollama_base_url.replace('/v1', '')}/api/tags")
            response.raise_for_status()

            # Check if required models are available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]

            if self.model not in model_names:
                raise ValueError(f"Model {self.model} not found in Ollama. Available: {model_names}")

    except Exception as e:
        logger.error(f"Ollama validation failed: {e}")
        raise
```

## Implementation Plan

### Phase 1: Immediate Fix (High Priority)
1. **Implement Fix 1**: Update `OllamaClient._create_structured_completion()` to properly handle JSON parsing
2. **Add Error Handling**: Improve error messages to distinguish between Ollama and OpenAI client issues
3. **Test with Memory Operations**: Verify the fix works with `add_memory` tool calls

### Phase 2: Robust Solution (Medium Priority)
1. **Implement Configuration Validation**: Add startup checks for Ollama availability and model presence
2. **Add Logging**: Enhanced logging to track which client and endpoint is being used
3. **Fallback Strategy**: Implement graceful degradation when structured output fails

### Phase 3: Long-term Improvements (Low Priority)
1. **Comprehensive Testing**: Add unit tests for Ollama structured response handling
2. **Documentation**: Update configuration documentation with Ollama-specific guidance
3. **Performance Optimization**: Consider caching model availability checks

## Testing Strategy

### Unit Tests
```python
async def test_ollama_structured_completion():
    """Test that OllamaClient properly handles structured completions."""
    client = OllamaClient(config=test_config)

    # Mock Ollama response
    mock_response = {"response": '{"name": "test", "type": "entity"}'}

    # Test structured completion
    response = await client._create_structured_completion(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": "Extract entity"}],
        temperature=0.1,
        max_tokens=100,
        response_model=TestEntity
    )

    assert response.choices[0].message.parsed is not None
    assert response.choices[0].message.content is not None
```

### Integration Tests
1. **End-to-End Memory Addition**: Test complete flow from `add_memory` to successful entity extraction
2. **Error Scenarios**: Test behavior when Ollama is down or models are unavailable
3. **Configuration Variants**: Test different Ollama model configurations

## Expected Outcomes

### Success Metrics
1. **Zero "Invalid response from LLM" errors** when using Ollama with proper models
2. **Successful entity extraction** from memory operations using structured completions
3. **Clear error messages** when Ollama is misconfigured or unavailable

### Rollback Plan
If fixes cause issues:
1. Revert to standard OpenAI client temporarily
2. Set `USE_OLLAMA=false` in environment variables
3. Use OpenAI API key for immediate functionality

## Risk Assessment

### Low Risk
- Changes are isolated to `OllamaClient` class
- Fallback behavior maintains existing functionality
- No changes to database or core Graphiti logic

### Potential Issues
- JSON parsing may fail for non-JSON Ollama responses
- Performance impact from additional parsing steps
- Model compatibility varies across different Ollama models

## References

- [Ollama OpenAI Compatibility Documentation](https://github.com/ollama/ollama/blob/main/docs/openai.md)
- [GraphitiCore LLM Client Architecture](../../../graphiti_core/llm_client/)
- [MCP Proxy Configuration](../../../mcpo/mcp-proxy-config.json)

---

**Document Version**: 1.0
**Created**: 2025-08-09
**Last Updated**: 2025-08-09
**Status**: Draft - Pending Implementation
