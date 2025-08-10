# Configuration Directory

This directory contains YAML-based configuration files for the Graphiti MCP Server. The configuration system supports different hierarchies for different types of configuration:

## Provider Configuration Hierarchy
For provider-specific settings (models, URLs, parameters):
1. **Default values** (lowest priority) - defined in code
2. **Base YAML files** - e.g., `providers/ollama.yml`
3. **Local override files** - e.g., `providers/ollama.local.yml`
4. **CLI arguments** (highest priority) - passed when starting the server

## Other Configuration Hierarchy
For general settings and sensitive data:
1. **Default values** (lowest priority) - defined in code
2. **YAML configuration files** - defined in this directory
3. **Environment variables** - for sensitive data like API keys
4. **CLI arguments** (highest priority) - passed when starting the server

## Directory Structure

```
config/
├── providers/              # Provider-specific configurations
│   ├── ollama.yml         # Base Ollama configuration
│   ├── ollama.local.yml   # Local Ollama overrides (optional)
│   ├── openai.yml         # Base OpenAI configuration
│   ├── openai.local.yml   # Local OpenAI overrides (optional)
│   ├── azure_openai.yml   # Base Azure OpenAI configuration
│   ├── azure_openai.local.yml # Local Azure OpenAI overrides (optional)
│   └── ollama.local.yml.example # Example local override file
├── database/
│   └── neo4j.yml          # Neo4j database configuration
├── server.yml             # General server configuration
└── README.md              # This file
```

## Local Override Files

Local override files (`.local.yml`) allow you to customize configuration without modifying the base files. This is perfect for:
- Personal development settings
- Environment-specific configurations
- Experimenting with different models or parameters

**Example**: Copy `providers/ollama.local.yml.example` to `providers/ollama.local.yml` and modify as needed.

**Git**: Local override files should be added to `.gitignore` to prevent committing personal settings.

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

## Environment Variables

Environment variables are now primarily used for sensitive credentials and system-level configuration:

```bash
# Required for OpenAI/Azure providers
export OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# System configuration
export USE_OLLAMA="true"
export NEO4J_PASSWORD="your-password"
```

**Note**: Provider-specific settings (models, URLs, parameters) are no longer configurable via environment variables. Use local override files instead.

## Configuration Examples

### Using Local Overrides

1. **Base configuration** (`providers/ollama.yml`):
```yaml
llm:
  model: "gpt-oss:latest"
  temperature: 0.1
  max_tokens: 50000
```

2. **Local override** (`providers/ollama.local.yml`):
```yaml
llm:
  model: "llama3.1:8b"
  temperature: 0.3
```

3. **Result**: Model and temperature from local file, max_tokens from base file.

### CLI Override Example

```bash
# Override any setting via CLI
uv run src/graphiti_mcp_server.py --temperature 0.7
```

CLI arguments always take highest precedence.

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
