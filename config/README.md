# Configuration Directory

This directory contains YAML-based configuration files for the Graphiti MCP Server. The configuration system supports a modern unified approach alongside backward-compatible provider-specific configurations.

## Quick Start (Recommended)

### 1. Create Your Unified Config

The **unified configuration** is the recommended approach - it allows you to configure both LLM and embedder in a single file with support for mixed providers.

```bash
# Copy the template
cp config/config.local.yml.example config/config.local.yml

# Edit with your settings
vim config/config.local.yml
```

### 2. Set Required Environment Variables

```bash
# For OpenAI/Bedrock LLM (skip if using Ollama)
export OPENAI_API_KEY="your-api-key-here"

# For Neo4j (always required)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
```

### 3. Start the Server

```bash
uv run src/graphiti_mcp_server.py --transport stdio
```

See the [Unified Configuration Guide](../docs/UNIFIED_CONFIG_GUIDE.md) for detailed setup instructions and examples.

## Configuration Hierarchy

Settings are loaded with this precedence (highest to lowest):

### For All Settings
1. **CLI arguments** (highest priority) - e.g., `--model`, `--temperature`
2. **Environment variables** - for sensitive data like `OPENAI_API_KEY`, `NEO4J_PASSWORD`
3. **Unified config** - `config/config.local.yml` (recommended)
4. **Provider-specific configs** - `providers/*.local.yml` (backward compatibility only)
5. **Default values** (lowest priority) - hardcoded fallbacks

## Directory Structure

```
config/
├── config.local.yml.example   # Template for unified configuration (recommended)
├── config.local.yml           # Your local unified config (git-ignored)
├── config.yml                 # Example unified configuration
├── providers/                 # Legacy provider-specific configs (backward compatibility)
│   ├── ollama.yml            # Base Ollama configuration
│   ├── ollama.local.yml.example # Example local override
│   ├── openai.yml            # Base OpenAI configuration
│   └── azure_openai.yml      # Base Azure OpenAI configuration
├── database/
│   └── neo4j.yml             # Neo4j database configuration
├── server.yml                # General server configuration
└── README.md                 # This file
```

## Unified Configuration (Recommended)

The unified configuration allows you to configure both LLM and embedder in a single file, with support for mixing providers.

**Example: Enterprise Gateway LLM + Local Ollama Embeddings**

```yaml
llm:
  model: "gpt-4o"
  base_url: "https://your-enterprise-gateway.com"
  temperature: 0.1
  max_tokens: 8192

embedder:
  model: "nomic-embed-text"
  base_url: "http://localhost:11434/v1"
  dimension: 768
```

**Benefits:**
- ✅ Single file configuration
- ✅ Mix and match providers (e.g., enterprise LLM with local embeddings)
- ✅ Automatic provider detection from `base_url`
- ✅ Clear, maintainable configuration
- ✅ Easy to version control (as example files)

### Provider Detection

The system automatically detects which provider to use based on `base_url`:

| URL Pattern | Detected Provider |
|-------------|-------------------|
| `localhost:11434` or `127.0.0.1:11434` | Ollama |
| `azure.com` in hostname | Azure OpenAI |
| Everything else | OpenAI-compatible |

Each component (LLM and embedder) is detected independently, allowing mixed configurations.

## Legacy Provider-Specific Configuration

For backward compatibility, provider-specific configurations are still supported. However, **unified config is preferred** for new setups.

### Provider Configuration Files

- `providers/ollama.yml` - Base Ollama configuration
- `providers/openai.yml` - Base OpenAI configuration
- `providers/azure_openai.yml` - Base Azure OpenAI configuration

### Local Override Files

Local override files (`.local.yml`) allow you to customize configuration without modifying base files:

```bash
# Example: Create Ollama local override
cp providers/ollama.local.yml.example providers/ollama.local.yml
vim providers/ollama.local.yml
```

**Note:** If `config/config.local.yml` exists, it takes precedence over provider-specific configs.

## Environment Variables

Environment variables are primarily used for sensitive credentials and system-level configuration:

### Required Variables

```bash
# For OpenAI/Azure/Bedrock providers
export OPENAI_API_KEY="your-api-key"

# For Azure OpenAI (if using Azure)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-02-01"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment"

# For Neo4j (always required)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"

# Optional: SSL certificates
export SSL_CERT_FILE="/path/to/cert.pem"
export SSL_CA_BUNDLE="/path/to/ca-bundle.crt"
```

**Important:** Provider-specific settings (models, URLs, parameters) should be configured via YAML files, not environment variables. This provides cleaner configuration management and better version control.

## Model-Specific Parameters

Each provider supports specific parameters in the `model_parameters` section:

### OpenAI/Bedrock Parameters

```yaml
model_parameters:
  presence_penalty: 0.0
  frequency_penalty: 0.0
  top_p: 1.0
  n: 1
  stream: false
```

### Ollama Parameters

```yaml
model_parameters:
  num_ctx: 4096          # Context window size
  num_predict: -1        # Number of tokens to predict
  repeat_penalty: 1.1    # Penalty for repeating tokens
  top_k: 40             # Limit token selection to top K
  top_p: 0.9            # Cumulative probability cutoff
  keep_alive: "5m"      # How long to keep model in memory
  seed: 42              # Random seed for reproducibility
```

## Common Configuration Scenarios

### Development: All Ollama (Free & Offline)

```yaml
llm:
  model: "llama3.1:8b"
  base_url: "http://localhost:11434/v1"
  temperature: 0.1
  max_tokens: 10000

embedder:
  model: "nomic-embed-text"
  base_url: "http://localhost:11434/v1"
  dimension: 768
```

**Requirements:** Ollama running locally with models pulled

### Production: Enterprise Gateway

```yaml
llm:
  model: "gpt-4o"
  base_url: "https://your-enterprise-gateway.com"
  temperature: 0.1
  max_tokens: 8192

embedder:
  model: "text-embedding-3-small"
  base_url: "https://your-enterprise-gateway.com/embeddings"
  dimension: 1536
```

**Requirements:** `OPENAI_API_KEY` environment variable

### Hybrid: Enterprise LLM + Local Embeddings (Recommended)

```yaml
llm:
  model: "gpt-4o"
  base_url: "https://your-enterprise-gateway.com"
  temperature: 0.1
  max_tokens: 8192

embedder:
  model: "nomic-embed-text"
  base_url: "http://localhost:11434/v1"
  dimension: 768
```

**Requirements:** `OPENAI_API_KEY` + Ollama running locally

**Benefits:** Enterprise-grade LLM reasoning with fast, free local embeddings

## Testing Configuration

Test your configuration changes:

```bash
# Start the server
uv run src/graphiti_mcp_server.py --transport stdio

# Check startup logs for:
# "Using [Provider] LLM: [model] at [url]"
# "Using [Provider] embedder: [model] at [url]"

# Test with CLI overrides
uv run src/graphiti_mcp_server.py --temperature 0.7 --transport stdio
```

## Git Ignore

Local configuration files are automatically ignored by git:

```gitignore
# .gitignore includes:
*.local.yml            # All local override files
!*.local.yml.example   # Except example files
.env                   # Environment variable files
```

This prevents committing personal settings and API keys.

## Migrating from Provider-Specific to Unified Config

If you have existing provider-specific configs:

1. Create `config/config.local.yml` from the example
2. Copy your LLM settings from your old provider file
3. Copy your embedder settings (can be from a different provider!)
4. Test: `uv run src/graphiti_mcp_server.py --transport stdio`
5. Optional: Remove old `*.local.yml` files from `providers/`

The system will automatically use unified config if it exists.

## See Also

- [Unified Configuration Guide](../docs/UNIFIED_CONFIG_GUIDE.md) - Comprehensive setup guide
- [Environment Setup Guide](../ENVIRONMENT_SETUP.md) - Environment variable configuration
- [Main README](../README.md) - Getting started guide

## Support

For issues or questions:
- Check the [Unified Configuration Guide](../docs/UNIFIED_CONFIG_GUIDE.md) troubleshooting section
- Review startup logs for configuration warnings
- Ensure YAML syntax is valid: `python3 -c "import yaml; print(yaml.safe_load(open('config/config.local.yml')))"`
