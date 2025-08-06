# Configuration Directory

This directory contains YAML-based configuration files for the Graphiti MCP Server. The configuration system supports a clear hierarchy of precedence:

1. **Default values** (lowest priority) - defined in code
2. **YAML configuration files** - defined in this directory
3. **Environment variables** - set in your environment or .env file
4. **CLI arguments** (highest priority) - passed when starting the server

## Directory Structure

```
config/
├── providers/           # Provider-specific configurations
│   ├── ollama.yml      # Ollama configuration and model parameters
│   ├── openai.yml      # OpenAI configuration
│   └── azure_openai.yml # Azure OpenAI configuration
├── database/
│   └── neo4j.yml       # Neo4j database configuration
├── server.yml          # General server configuration
└── README.md           # This file
```

## Provider Configuration

### Ollama (providers/ollama.yml)

The Ollama configuration supports model-specific parameters that are passed directly to the Ollama API:

```yaml
llm:
  model: "deepseek-r1:7b"
  base_url: "http://localhost:11434/v1"
  temperature: 0.1
  max_tokens: 8192
  model_parameters:
    num_ctx: 4096          # Context window size
    num_predict: -1        # Number of tokens to predict
    repeat_penalty: 1.1    # Penalty for repeating tokens
    top_k: 40             # Limit token selection to top K
    top_p: 0.9            # Cumulative probability cutoff
```

**Supported Ollama Model Parameters:**
- `num_ctx`: Context window size (number of tokens to consider)
- `num_predict`: Number of tokens to predict (-1 for unlimited)
- `repeat_penalty`: Penalty for repeating tokens (1.0 = no penalty)
- `top_k`: Limit next token selection to K most probable tokens
- `top_p`: Cumulative probability cutoff for token selection
- `temperature`: Model-level temperature (can override general temperature)
- `seed`: Random seed for reproducible outputs
- `stop`: Array of stop sequences

### OpenAI (providers/openai.yml)

Standard OpenAI configuration with model parameters:

```yaml
llm:
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 8192
  model_parameters:
    presence_penalty: 0.0
    frequency_penalty: 0.0
    top_p: 1.0
```

### Azure OpenAI (providers/azure_openai.yml)

Azure OpenAI configuration (endpoints and keys still use environment variables):

```yaml
llm:
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 8192
```

## Environment Variable Override

You can still use environment variables to override any YAML configuration:

```bash
# Override Ollama model
export OLLAMA_LLM_MODEL="llama2:13b"

# Override temperature
export LLM_TEMPERATURE="0.7"

# Override max tokens
export LLM_MAX_TOKENS="16384"
```

## Security Note

**API keys and sensitive credentials should still be set via environment variables for security:**

- `OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `NEO4J_PASSWORD`
- etc.

## Configuration Precedence Examples

If you have:
1. YAML config: `temperature: 0.1`
2. Environment variable: `LLM_TEMPERATURE=0.5`
3. CLI argument: `--temperature 0.7`

The final temperature will be `0.7` (CLI argument wins).

## Adding New Providers

To add a new provider:

1. Create `providers/new_provider.yml`
2. Add provider-specific configuration structure
3. Update the ConfigLoader to support the new provider
4. Extend GraphitiLLMConfig to handle the new provider

## Testing Configuration

You can test your configuration by running:

```bash
# Test with current configuration
uv run src/graphiti_mcp_server.py --help

# Test with specific provider
USE_OLLAMA=true uv run src/graphiti_mcp_server.py --transport sse
```

Check the logs to see which configuration values are being used.
