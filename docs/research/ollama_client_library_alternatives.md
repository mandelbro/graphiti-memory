### Evaluating Alternatives to the Home‑Rolled Ollama Client and Migration Plan

#### Executive Summary

- We can simplify maintenance and gain feature depth by replacing our custom `src/ollama_client.py` with either:
  - Official Ollama Python SDK (`ollama`), or
  - LangChain’s `langchain-ollama` integration.

- Recommendation: Prefer `langchain-ollama` as the primary path for rich features (native structured outputs via JSON schema, tool-calling, reasoning, consistent async/streaming), while keeping a secondary option using the official `ollama` `AsyncClient` for minimal dependencies and pure native control.

- We will introduce a feature flag to switch between implementations and validate parity with our existing test suite before fully switching over.

References:
- LangChain ChatOllama API (supports `keep_alive`, `num_ctx`, `num_predict`, JSON schema outputs, tool-calling): [ChatOllama API](https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html)
- Official Ollama Python SDK (sync/async, native options, streaming): [ollama/ollama-python](https://github.com/ollama/ollama-python)
- Ollama OpenAI‑compatibility context: [Ollama OpenAI compatibility](https://ollama.com/blog/openai-compatibility)
- Graphiti OpenAI‑compatible guidance: [Graphiti Quick Start](https://github.com/getzep/graphiti?tab=readme-ov-file#quick-start)

---

#### Current State (Home‑Rolled Client)

Our `src/ollama_client.py` provides:
- Transport via `httpx` with connection pooling and timeouts
- Native `/api/generate` calls to preserve Ollama options (temperature, `num_predict`, and arbitrary `options`), plus `keep_alive`
- Health/model checks and error handling
- Response conversion to OpenAI format, and structured parsing using a Pydantic model + converter

This works but adds ongoing maintenance for: API drift, parameter coverage, response conversion, and structured output resilience.

---

#### Alternatives Assessed

1) Official Ollama Python SDK (`ollama`)
- Capabilities
  - Native Chat/Generate/Embed APIs; sync/async clients; streaming; httpx customization
  - Pass native options and `keep_alive` directly in requests
  - Simple, first‑party maintenance reduces risk of API drift
- Fit for our needs
  - Excellent for direct control of Ollama behavior while reducing our transport/HTTP code
  - We would keep a thin adapter to remain compatible with our internal interfaces and OpenAI‑style expectations where needed
- Reference: [ollama/ollama-python](https://github.com/ollama/ollama-python)

2) LangChain `langchain-ollama`
- Capabilities
  - Exposes Ollama parameters including `keep_alive`, `num_ctx`, `num_predict`, `top_p`, etc.
  - Async/streaming, tool-calling, and native structured outputs via JSON schema (`with_structured_output`)
  - Reasoning mode for models that support it; httpx client kwargs for pooling/timeouts; `validate_model_on_init`
- Fit for our needs
  - Provides a robust, high‑level feature set that aligns with Graphiti’s structured memory operations
  - Minimizes our custom response conversion and validation logic
- Reference: [ChatOllama API](https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html)

Note on OpenAI‑compat approaches: While Graphiti supports “OpenAI‑compatible APIs” for other providers, Ollama’s native features (e.g., `keep_alive`, granular options, JSON schema structured outputs) are more directly and consistently exposed via the two options above than via a generic OpenAI shim.
- Context: [Ollama OpenAI compatibility](https://ollama.com/blog/openai-compatibility), [Graphiti Quick Start](https://github.com/getzep/graphiti?tab=readme-ov-file#quick-start)

---

#### Capability Mapping (High‑Level)

- Transport & Async/Streaming
  - Official `ollama`: Yes (sync/async, streaming)
  - LangChain `ChatOllama`: Yes (invoke/stream/async variants)

- Model Options & `keep_alive`
  - Official `ollama`: Full native options and `keep_alive`
  - LangChain `ChatOllama`: Exposes `keep_alive`, `num_ctx`, `num_predict`, temperature, top_k/top_p, etc.

- Structured Outputs
  - Official `ollama`: We implement our own parsing/validation on top
  - LangChain `ChatOllama`: Built‑in JSON schema outputs via `with_structured_output`

- Tool‑Calling
  - Official `ollama`: Possible, but we’d implement scaffolding
  - LangChain `ChatOllama`: Supported; integrates with LangChain tool schemas

- Health/Model Validation
  - Official `ollama`: We keep a short utility or use SDK errors
  - LangChain `ChatOllama`: `validate_model_on_init` and surface errors cleanly

---

#### Recommendation

- Primary: Integrate `langchain-ollama` as the default Ollama path for Graphiti, leveraging:
  - Native JSON‑schema structured outputs for Pydantic validation
  - Tool‑calling and reasoning support where applicable
  - Direct exposure of `keep_alive` and other advanced parameters

- Secondary: Provide an alternative adapter backed by the official `ollama` `AsyncClient` for teams that want the thinnest dependency surface and full native control.

Both implementations sit behind a feature flag so we can A/B validate and roll out incrementally.

---

#### Migration Plan (Phased)

Phase 0 – Preparation
- Add dependencies (exact pins to be finalized during implementation):
  - `langchain-ollama` (primary)
  - `ollama` (secondary)
- Introduce `GRAPHITI_OLLAMA_BACKEND` (env or config): `homegrown | langchain | official`
  - Default to `langchain` for development branches once tests pass; keep `homegrown` as fallback initially.

Phase 1 – New Adapters (No Behavior Change by Default)
- Implement `LangchainOllamaAdapter` that conforms to our internal client interface (currently satisfied by `BaseOpenAIClient` expectations in Graphiti). Map:
  - Messages → `ChatOllama.invoke/stream`
  - Parameters: temperature, `max_tokens` → `num_predict`, `keep_alive`, `num_ctx`, `top_p`, etc.
  - Structured responses → use `with_structured_output` for Pydantic models
  - Connection settings → pass `async_client_kwargs`/`client_kwargs` for pooling/timeouts
- Implement `OfficialOllamaAdapter` backed by `ollama.AsyncClient`:
  - Map our message format to SDK’s chat/generate
  - Pass native `options` and `keep_alive`
  - Keep minimal response conversion to our internal format

Phase 2 – Feature Flag Integration
- Wire `GRAPHITI_OLLAMA_BACKEND` to choose among `homegrown | langchain | official`
- Keep our existing health/model checks where helpful; otherwise rely on SDK/adapter errors

Phase 3 – Validation and Parity Tests
- Run and fix tests as needed, ensuring no regressions:
  - `tests/test_keep_alive_parameter.py`
  - `tests/test_ollama_connection_pooling.py`
  - `tests/test_ollama_structured_responses.py`
  - Full suite for MCP/server flows that touch LLM behavior

Phase 4 – Rollout
- Switch default to `langchain` in development
- Monitor performance/latency and memory usage
- After soak, set default in mainline; retain env flag for rollback

---

#### Risks & Mitigations

- Parameter parity (e.g., advanced sampling options)
  - Validate coverage; for gaps, prefer the official SDK path for that use case

- Structured parsing differences
  - Prefer `with_structured_output` for Pydantic validation; ensure our schemas align

- Streaming behavior differences
  - Exercise streaming tests; adjust adapters’ stream handling accordingly

- Dependency surface and version drift
  - Pin versions and track changelogs; keep `official` adapter as a minimal fallback

---

#### Acceptance Criteria

- All existing Ollama tests pass with `GRAPHITI_OLLAMA_BACKEND=langchain` and `official`
- No performance regression beyond agreed thresholds (latency and memory)
- `keep_alive` behavior validated across both adapters
- Structured outputs validated end‑to‑end with our Pydantic models

---

#### References

- LangChain ChatOllama API: [ChatOllama API](https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html)
- Official Ollama Python SDK: [ollama/ollama-python](https://github.com/ollama/ollama-python)
- Ollama OpenAI‑compat blog: [Ollama OpenAI compatibility](https://ollama.com/blog/openai-compatibility)
- Graphiti Quick Start (OpenAI compatible note): [Graphiti Quick Start](https://github.com/getzep/graphiti?tab=readme-ov-file#quick-start)
