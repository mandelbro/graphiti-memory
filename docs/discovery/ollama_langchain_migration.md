# DISCOVERY: Ollama Client Library Migration

**Date**: 2025-08-22T15:12:16+00:00
**Project**: Graphiti Memory / MCP Server
**Type**: Technical Debt Reduction & Dependency Migration
**Author**: Engineering Discovery (Norwood)

## The One-Liner
Replace our 314-line home-rolled Ollama client with battle-tested libraries (LangChain or official SDK) to reduce maintenance burden and unlock advanced features like native structured outputs and tool-calling.

## The Problem
Our `src/ollama_client.py` has grown into a mini HTTP framework with its own response conversion, connection pooling, health checking, and schema mapping. Every time Ollama updates their API or we need new features, we're on the hook for implementation and testing.

**Current Pain Points:**
- **Maintenance Overhead**: 314 lines of transport code + 10 test files = ongoing maintenance debt
- **Feature Lag**: Missing native structured outputs, tool-calling, reasoning mode
- **Conversion Complexity**: Custom response converter with edge case handling for JSON/XML wrapping
- **Parameter Coverage**: Playing catch-up with new Ollama parameters and options

## The Solution
Implement a dual-path migration strategy:

1. **Primary Path**: LangChain `langchain-ollama` for rich feature support
2. **Secondary Path**: Official `ollama` SDK for minimal dependencies
3. **Feature Flag**: Environment-controlled switching between implementations
4. **Backward Compatibility**: Keep existing interfaces intact during migration

## The Work

### Effort Estimate
- **Phase 0 (Prep)**: 4-8 hours - Dependencies, feature flag setup
- **Phase 1 (Adapters)**: 16-24 hours - LangChain + Official SDK adapters
- **Phase 2 (Integration)**: 8-12 hours - Wire up feature flag, config updates
- **Phase 3 (Test Refactoring)**: 12-16 hours - Clean up obsolete tests, validate parity
- **Phase 4 (Rollout)**: 4-8 hours - Default switching, monitoring

**Total**: 44-68 hours (roughly 1.5-2 sprint capacity for one engineer)

### Timeline
- **Week 1**: Phases 0-1 (Foundation + adapters)
- **Week 2**: Phase 2-3 (Integration + test cleanup)
- **Week 3**: Phase 4 + monitoring (Rollout)

### Team Needs
- **Primary**: 1 senior engineer familiar with async Python, Pydantic validation
- **Supporting**: 1 engineer for test validation and edge case coverage

## The Risks

### 🔴 High Risk: Parameter Parity Gaps
**Probability**: Likely - LangChain might not expose every obscure Ollama parameter
**Impact**: Feature regression for power users with custom configurations
**Mitigation**: Map all current `model_parameters` usage, validate coverage during Phase 1
**Detection**: Comprehensive parameter validation tests

### 🟡 Medium Risk: Response Format Differences
**Probability**: Possible - Different libraries handle edge cases differently
**Impact**: Structured parsing failures in production
**Mitigation**: Extensive test coverage for all response formats (JSON, XML-wrapped, arrays)
**Detection**: Side-by-side validation with current implementation

### 🟡 Medium Risk: Performance Regression
**Probability**: Possible - Additional abstraction layers
**Impact**: Increased latency for LLM operations
**Mitigation**: Benchmark against current implementation, keep official SDK as fallback
**Detection**: Performance monitoring during Phase 4

### 🟢 Low Risk: Dependency Drift
**Probability**: Low - Both libraries are actively maintained
**Impact**: Breaking changes in future updates
**Mitigation**: Pin versions, maintain feature flag for rollback
**Detection**: Automated dependency scanning

---

## Current State Analysis

### The Code We're Replacing
- **Location**: `src/ollama_client.py` (314 lines)
- **What It Does**: HTTP transport + response conversion + health checking + structured parsing
- **What It Should Do**: Just handle Ollama-specific parameter passing
- **Why The Gap Exists**: Started simple, grew organically as we needed more features

### Dependencies & Coupling
- **Hard Dependencies**: `httpx`, `graphiti_core.llm_client.openai_base_client`, `pydantic`
- **Soft Dependencies**: `src.utils.ollama_health_validator`, `src.utils.ollama_response_converter`
- **Hidden Dependencies**: Assumes OpenAI message format compatibility throughout Graphiti

### Technical Debt Inventory

**Critical (The Ticking Time Bombs)**
- Custom HTTP client management: Connection pooling logic that duplicates httpx functionality | Fix effort: Replace with library-managed clients
- Response format conversion: Complex JSON/XML parsing with edge case handling | Fix effort: Use library native structured outputs

**Annoying (Death by a Thousand Cuts)**
- Parameter mapping: Manual translation between OpenAI params and Ollama options | Fix effort: Let libraries handle this
- Health checking: Custom validation logic that could be simpler | Fix effort: Use library error handling

**Philosophical (The "Why Did They Do It This Way?" Category)**
- Extending BaseOpenAI client: Forces Ollama into OpenAI compatibility box | Migration effort: Clean interface separation

---

## Solution Architecture

### Architecture Decision: Dual-Path Migration with Feature Flags

#### The Situation
We need to migrate from home-rolled client without breaking existing functionality, while evaluating which library best fits our needs long-term.

#### The Options We Considered

1. **The Conservative Choice**: Gradual refactoring of existing client
   - Why it might work: Minimal disruption, incremental improvement
   - Why it might suck: Still maintaining custom HTTP transport, no access to advanced features
   - Effort: 20-30 hours of continued maintenance

2. **The Modern Approach**: Full migration to LangChain `langchain-ollama`
   - Why it might work: Rich feature set, structured outputs, tool-calling, active maintenance
   - Why it might suck: Heavier dependency, potential over-engineering for simple use cases
   - Effort: 30-40 hours

3. **The Pragmatic Middle**: Dual-path with feature flag
   - Why it might work: Best of both worlds, validation path, rollback capability
   - Why it might suck: More initial complexity, two codepaths to maintain temporarily
   - Effort: 44-68 hours upfront, reduced long-term maintenance

#### What We're Going With
**Dual-path with feature flag** because it gives us validation, rollback capability, and lets us determine the best long-term solution through real-world usage.

#### Trade-offs We're Accepting
- **Temporary complexity** (dual implementations) in exchange for **migration safety** and **validation capability**
- **Higher upfront effort** in exchange for **reduced long-term maintenance** and **access to advanced features**

### Implementation Strategy

#### Phase 0: Foundation (The "Don't Break Anything" Phase)
**Goal**: Set up infrastructure without touching production behavior

```python
# Add to pyproject.toml dependencies
langchain-ollama = "^0.1.0"  # For rich features
ollama = "^0.3.0"           # For minimal dependencies

# Environment configuration
GRAPHITI_OLLAMA_BACKEND = "homegrown"  # default: current behavior
# Options: "homegrown" | "langchain" | "official"
```

**Success Criteria**: Dependencies installed, no behavior changes

#### Phase 1: New Adapters (The "Build the Future" Phase)
**Goal**: Implement new adapter classes behind feature flag

```python
# src/clients/langchain_ollama_adapter.py
class LangChainOllamaAdapter(BaseOpenAIClient):
    async def _create_structured_completion(self, ...):
        # Use ChatOllama.with_structured_output()
        # No manual JSON parsing required
        pass

    async def _create_completion(self, ...):
        # Use ChatOllama.invoke() or .stream()
        # Pass keep_alive, num_ctx directly
        pass

# src/clients/official_ollama_adapter.py
class OfficialOllamaAdapter(BaseOpenAIClient):
    async def _create_completion(self, ...):
        # Use ollama.AsyncClient.chat()
        # Minimal response conversion
        pass
```

**Success Criteria**: Both adapters pass basic functionality tests

#### Phase 2: Integration (The "Wire It Up" Phase)
**Goal**: Connect feature flag to adapter selection

```python
# src/initialization/graphiti_client.py
def create_ollama_client(config: LLMConfig) -> BaseOpenAIClient:
    backend = os.getenv("GRAPHITI_OLLAMA_BACKEND", "homegrown")

    if backend == "langchain":
        return LangChainOllamaAdapter(config)
    elif backend == "official":
        return OfficialOllamaAdapter(config)
    else:  # "homegrown"
        return OllamaClient(config)
```

**Rollback Plan**: Set `GRAPHITI_OLLAMA_BACKEND=homegrown` to return to current behavior

#### Phase 3: Test Migration (The "Clean House" Phase)
**Goal**: Update test suite for new reality
*See detailed breakdown in Test Refactoring Strategy section below*

---

## Test Refactoring Strategy

### Current Test Inventory Analysis

#### Tests That Become **OBSOLETE** (Delete Entirely)
```
tests/test_ollama_response_converter.py (687 lines)
```
**Why obsolete**: Testing custom response conversion logic that libraries handle natively
**Library equivalent**: LangChain's `with_structured_output()` handles this automatically
**Action**: DELETE - No migration needed

```
tests/test_ollama_connection_pooling.py
```
**Why obsolete**: Testing custom httpx client management
**Library equivalent**: Libraries manage their own HTTP clients
**Action**: DELETE - No migration needed

```
tests/test_ollama_mock_message_model_dump.py
```
**Why obsolete**: Testing specific OpenAI message format conversion quirks
**Library equivalent**: Libraries handle message format internally
**Action**: DELETE - No migration needed

#### Tests That Need **HEAVY REFACTORING** (Rewrite for Interface)
```
tests/test_ollama_client_comprehensive.py (865+ lines)
```
**Current focus**: Testing internal HTTP mechanics, response parsing
**New focus**: Test adapter interface compliance, parameter passing
**Refactor approach**: Keep ~30% of tests (interface validation), rewrite HTTP/parsing tests as integration tests

```
tests/test_ollama_structured_responses.py
```
**Current focus**: Testing custom JSON/XML parsing edge cases
**New focus**: Test that structured outputs work end-to-end with library native methods
**Refactor approach**: Replace parsing tests with validation that `with_structured_output()` works

#### Tests That **TRANSLATE DIRECTLY** (Minor Updates)
```
tests/test_keep_alive_parameter.py
tests/test_ollama_context_parameters.py
tests/test_ollama_config_integration.py
```
**Why they translate**: Testing parameter passing and configuration, which adapters must preserve
**Update needed**: Change from testing HTTP payload inspection to testing parameter reach library
**Effort**: ~20% rewrite per file

#### Tests That **STAY MOSTLY INTACT** (Compatibility Layer)
```
tests/test_ollama_health_check.py
tests/test_ollama_model_validation.py
```
**Why they stay**: Testing health/validation interface that adapters must implement
**Update needed**: Verify adapters expose same health check interface
**Effort**: ~10% updates per file

### Refactoring Implementation Plan

#### Step 1: Delete Obsolete Tests (Immediate)
```bash
# Remove tests that test implementation details we're abandoning
rm tests/test_ollama_response_converter.py
rm tests/test_ollama_connection_pooling.py
rm tests/test_ollama_mock_message_model_dump.py
```
**Time savings**: ~40% reduction in Ollama test maintenance

#### Step 2: Create New Integration Test Suite
```python
# tests/test_ollama_adapter_parity.py
class TestOllamaAdapterParity:
    """Ensure all adapters behave identically for core functionality."""

    @pytest.mark.parametrize("backend", ["langchain", "official", "homegrown"])
    async def test_parameter_passing_parity(self, backend):
        # Test keep_alive, num_ctx, temperature work the same
        pass

    @pytest.mark.parametrize("backend", ["langchain", "official"])
    async def test_structured_output_parity(self, backend):
        # Test structured outputs work identically
        pass
```

#### Step 3: Refactor Heavy Tests
```python
# tests/test_ollama_client_comprehensive.py (Before: 865 lines)
# After refactoring, target ~200-300 lines focused on:
class TestOllamaClientInterface:
    """Test adapter interface compliance, not HTTP implementation."""

    async def test_completion_interface_compliance(self):
        # Test that adapters satisfy BaseOpenAIClient contract
        pass

    async def test_error_handling_consistency(self):
        # Test that all adapters handle errors consistently
        pass
```

#### Step 4: Update Parameter Tests
```python
# tests/test_keep_alive_parameter.py - Update approach:
# OLD: Test HTTP payload contains keep_alive
# NEW: Test that keep_alive affects model behavior (integration test)

async def test_keep_alive_affects_model_persistence(self):
    """Test keep_alive parameter controls model loading behavior."""
    # Call model with keep_alive=30
    # Verify subsequent calls are faster (model stayed loaded)
```

### Test Refactoring Effort Breakdown
- **Deleted tests**: -40% maintenance burden
- **Heavy refactoring**: 3 files × 8 hours = 24 hours
- **Translation updates**: 3 files × 2 hours = 6 hours
- **New integration suite**: 8 hours
- **Validation and debugging**: 4 hours

**Total refactoring effort**: 42 hours (included in overall 44-68 hour estimate)

---

## Technical Deep Dives

### Library Evaluation: LangChain vs Official SDK

#### LangChain `langchain-ollama`
**The Pitch**: "Rich integration with structured outputs, tool-calling, and reasoning"

**The Reality** (Based on API docs and GitHub activity):
- **Actual Usage**: 50M+ downloads for LangChain ecosystem, active development
- **Pain Points**: Heavier dependency tree, potential over-engineering for simple use cases
- **Hidden Costs**: LangChain ecosystem lock-in, learning curve for new patterns

**Code Sample**:
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3",
    keep_alive="5m",
    num_ctx=4096,
    temperature=0.7,
    base_url="http://localhost:11434",
)

# Native structured outputs - no manual parsing
structured_llm = llm.with_structured_output(YourPydanticModel)
result = await structured_llm.ainvoke(messages)
```

**Verdict**: Use for primary path - rich features justify complexity

#### Official `ollama` SDK
**The Pitch**: "First-party support, minimal dependencies, direct API access"

**The Reality**:
- **Actual Usage**: 1M+ PyPI downloads, maintained by Ollama team
- **Pain Points**: More manual work for structured outputs, fewer abstractions
- **Hidden Costs**: We implement more scaffolding ourselves

**Code Sample**:
```python
import ollama

client = ollama.AsyncClient(host='http://localhost:11434')
response = await client.chat(
    model='llama3',
    messages=messages,
    options={
        'temperature': 0.7,
        'num_ctx': 4096,
    },
    keep_alive='5m'
)
# Manual structured parsing still needed
```

**Verdict**: Use for secondary path - minimal surface area, full control

### Performance Analysis

#### Current Baseline
- **Latency**: ~200-500ms per completion (includes custom HTTP client overhead)
- **Memory**: ~50MB for client + connection pools
- **Bottleneck**: JSON parsing and validation in our custom converter

#### Expected Improvements
- **LangChain Path**: -10-20ms (library-optimized HTTP handling), +structured output efficiency
- **Official SDK Path**: -30-40ms (minimal abstraction overhead)
- **Memory**: -20MB (eliminate custom HTTP client and conversion logic)

**How We Know**: Both libraries use optimized HTTP clients and eliminate our conversion overhead

---

## Risk Deep Dive

### Dependency Analysis
```
Current dependencies (Ollama-related):
├── httpx (custom HTTP client management)
├── pydantic (response validation)
└── Custom utilities (3 files, ~500 lines)

Proposed dependencies:
├── langchain-ollama (~15 direct dependencies)
├── ollama (~3 direct dependencies)
└── Eliminated custom utilities
```

**Bus Factor Analysis**:
- **LangChain**: Large team, enterprise backing (reasonable)
- **Official SDK**: Ollama team (single company, but first-party)
- **Maintenance Window**: Both actively maintained, regular releases

---

## Success Metrics

### Technical Metrics
- **Performance**: No more than 10% latency increase (target: 5-10% improvement)
- **Memory**: Reduce memory footprint by 15-25%
- **Test Coverage**: Maintain >90% coverage with 40% fewer test files
- **Feature Parity**: 100% parameter compatibility with current implementation

### Delivery Metrics
- **Phase Completion**: Each phase completed within estimated time windows
- **Rollback Capability**: Feature flag allows instant rollback with <5 minute deployment
- **Integration Tests**: All existing Graphiti integration tests pass with new adapters

### Quality Metrics
- **Bug Rate**: No increase in Ollama-related bug reports post-migration
- **Developer Experience**: Reduced complexity for adding new Ollama features
- **Maintenance Overhead**: 40% reduction in Ollama-specific test maintenance

---

## Deliverables Checklist

### Technical Implementation
- [ ] LangChain adapter with structured output support
- [ ] Official SDK adapter with minimal dependencies
- [ ] Feature flag system with environment controls
- [ ] Parameter parity validation across all adapters
- [ ] Connection management handled by libraries
- [ ] Error handling consistency across adapters

### Test Suite Refactoring
- [ ] Remove 3 obsolete test files (~40% reduction in maintenance)
- [ ] Create adapter parity test suite
- [ ] Refactor comprehensive test to focus on interface compliance
- [ ] Update parameter tests for integration validation
- [ ] Validate structured output behavior across adapters

### Quality Assurance
- [ ] Performance benchmarks vs current implementation
- [ ] Memory usage validation
- [ ] All existing integration tests pass with new adapters
- [ ] Feature flag rollback capability verified
- [ ] Production deployment plan with monitoring

### Documentation
- [ ] Migration guide for teams using custom model_parameters
- [ ] Troubleshooting guide for adapter selection
- [ ] Performance comparison documentation
- [ ] Rollback procedures for emergency situations

---

## References

### Research Documentation
- [Ollama Client Library Alternatives Research](../research/ollama_client_library_alternatives.md)

### External Dependencies
- [LangChain ChatOllama API](https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html)
- [Official Ollama Python SDK](https://github.com/ollama/ollama-python)
- [Ollama OpenAI Compatibility](https://ollama.com/blog/openai-compatibility)
- [Graphiti Quick Start](https://github.com/getzep/graphiti?tab=readme-ov-file#quick-start)

### Internal Context
- Current Implementation: `src/ollama_client.py`
- Test Suite: `tests/test_ollama_*.py` (10 files)
- Configuration: `src/config/llm_config.py`
- Integration Points: `src/initialization/graphiti_client.py`

