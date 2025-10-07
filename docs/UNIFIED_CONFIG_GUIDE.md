# Unified Configuration Guide

This guide explains the unified configuration system for the Graphiti MCP Server. The unified config simplifies setup by allowing you to configure both LLM and embeddings in a single file, with support for mixed providers.

## Overview

The unified configuration system provides a simpler, more flexible way to configure your Graphiti MCP server:

- **Single config file**: `config/config.local.yml` for all settings
- **Mixed providers**: Use different providers for LLM and embeddings
- **Auto-detection**: Provider type detected from `base_url`
- **Backward compatible**: Old provider-specific configs still work

## Quick Start

### 1. Create Your Config File

Copy the template:

```bash
cp config/config.local.yml.example config/config.local.yml
```

Or create your own `config/config.local.yml`:

```yaml
llm:
  model: "claude-sonnet-4-20250514"
  base_url: "https://your-gateway.com"
  temperature: 0.1
  max_tokens: 25000

embedder:
  model: "nomic-embed-text"
  base_url: "http://localhost:11434/v1"
  dimension: 768
```

### 2. Set Environment Variables

```bash
# For Bedrock/OpenAI LLM (required if not using Ollama)
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

## Configuration Examples

### Example 1: Bedrock LLM + Local Ollama Embeddings (Recommended)

This is the optimal configuration for most use cases - enterprise LLM with fast local embeddings.

```yaml
llm:
  model: "claude-sonnet-4-20250514"
  base_url: "https://eng-ai-model-gateway.sfproxy.devx-preprod.aws-esvc1-useast2.aws.sfdc.cl"
  temperature: 0.1
  max_tokens: 25000

  model_parameters:
    presence_penalty: 0.0
    frequency_penalty: 0.0
    top_p: 1.0

embedder:
  model: "nomic-embed-text"
  base_url: "http://localhost:11434/v1"
  dimension: 768

  model_parameters:
    num_ctx: 4096
```

**Required:**
- `OPENAI_API_KEY` for Bedrock gateway
- Ollama running locally with `nomic-embed-text` model

**Benefits:**
- Enterprise-grade LLM for complex reasoning
- Fast local embeddings (no API latency)
- Lower cost (embeddings are free)
- Works offline for embedding generation

### Example 2: All Ollama (Local Development)

Perfect for development, testing, or offline work.

```yaml
llm:
  model: "llama3.1:8b"
  base_url: "http://localhost:11434/v1"
  temperature: 0.1
  max_tokens: 10000

  model_parameters:
    num_ctx: 4096
    keep_alive: "5m"

embedder:
  model: "nomic-embed-text"
  base_url: "http://localhost:11434/v1"
  dimension: 768
```

**Required:**
- Ollama running locally
- Models pulled: `ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`

**Benefits:**
- Completely free
- Works offline
- Fast iteration
- No API keys needed

### Example 3: All OpenAI/Bedrock (Enterprise)

Full enterprise gateway setup.

```yaml
llm:
  model: "gpt-4o"
  base_url: "https://your-enterprise-gateway.com"
  temperature: 0.1
  max_tokens: 8192

embedder:
  model: "text-embedding-3-small"
  base_url: "https://your-enterprise-gateway.com/bedrock/embeddings"
  dimension: 1536
```

**Required:**
- `OPENAI_API_KEY` for gateway access

**Benefits:**
- Centralized billing/monitoring
- Enterprise security compliance
- Consistent provider

### Example 4: Azure OpenAI

For Azure-hosted OpenAI services.

```yaml
llm:
  model: "gpt-4"  # Your deployment model
  temperature: 0.1
  max_tokens: 8192

embedder:
  model: "text-embedding-3-small"
  dimension: 1536
```

**Required Environment Variables:**
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-02-01"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment"
export OPENAI_API_KEY="your-azure-key"

# For embeddings (if separate deployment)
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="your-embedding-deployment"
```

## How Provider Detection Works

The system automatically detects which provider to use based on the `base_url`:

| URL Pattern | Detected Provider |
|-------------|-------------------|
| `localhost:11434` or `127.0.0.1:11434` | Ollama |
| `azure.com` in hostname | Azure OpenAI |
| Everything else | OpenAI-compatible |

**Note:** Detection is fully automatic based on the `base_url` in your configuration. No environment variables needed for detection. Each component (LLM and embedder) is detected independently, allowing mixed provider configurations.

## Configuration Hierarchy

Settings are loaded with this precedence (highest to lowest):

1. **CLI arguments** (e.g., `--model`, `--temperature`)
2. **Environment variables** (e.g., `OPENAI_API_KEY`, `OLLAMA_BASE_URL`)
3. **Unified config** (`config/config.local.yml`)
4. **Provider-specific configs** (`config/providers/*.local.yml`) - deprecated
5. **Default values** (hardcoded fallbacks)

## Migrating from Provider-Specific Configs

If you have existing provider-specific configs:

### Old Setup (Multiple Files)
```
config/providers/openai.local.yml
config/providers/ollama.local.yml
```

### New Setup (Single File)
```
config/config.local.yml
```

**Migration Steps:**

1. Create `config/config.local.yml`
2. Copy your LLM settings from the old file
3. Copy your embedder settings (can be from a different provider file!)
4. Test with `uv run src/graphiti_mcp_server.py --transport stdio`
5. Remove old provider-specific files (optional - they're still supported)

## Advanced Configuration

### Model-Specific Parameters

Each provider supports specific parameters in `model_parameters`:

**OpenAI/Bedrock:**
```yaml
model_parameters:
  presence_penalty: 0.0
  frequency_penalty: 0.0
  top_p: 1.0
  n: 1
  stream: false
```

**Ollama:**
```yaml
model_parameters:
  num_ctx: 4096
  num_predict: -1
  repeat_penalty: 1.1
  top_k: 40
  top_p: 0.9
  keep_alive: "5m"
```

### Multiple Environments

Use different config files for different environments:

```bash
# Development
cp config/config.dev.yml config/config.local.yml

# Production
cp config/config.prod.yml config/config.local.yml
```

Or use environment-specific variable prefixes:

```bash
# Development
export DEV_OPENAI_API_KEY="dev-key"

# Production
export PROD_OPENAI_API_KEY="prod-key"
```

## Troubleshooting

### Config Not Loading

**Symptom:** Server uses default values instead of your config

**Solution:**
```bash
# Check if config.local.yml exists
ls -la config/config.local.yml

# Check YAML syntax
python3 -c "import yaml; print(yaml.safe_load(open('config/config.local.yml')))"

# Check logs for warnings
uv run src/graphiti_mcp_server.py --transport stdio 2>&1 | grep -i "config"
```

### Wrong Provider Detected

**Symptom:** Server uses Ollama when you want OpenAI, or vice versa

**Solution:**
```bash
# Check your base_url in config.local.yml
# Provider is auto-detected from the URL:
#   localhost:11434 -> Ollama
#   azure.com -> Azure OpenAI
#   everything else -> OpenAI-compatible

# If you want to use Ollama, set base_url to:
#   base_url: "http://localhost:11434/v1"

# If you want to use OpenAI/Bedrock, set base_url to your gateway:
#   base_url: "https://your-gateway.com"
```

### Missing API Key

**Symptom:** `OPENAI_API_KEY must be set` error

**Solution:**
```bash
# Check if env var is set
echo $OPENAI_API_KEY

# Set it properly
export OPENAI_API_KEY="your-key-here"

# Or add to .env file
echo 'OPENAI_API_KEY="your-key-here"' >> .env
```

### Ollama Connection Failed

**Symptom:** `Connection refused` to `localhost:11434`

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check if model is available
ollama list | grep nomic-embed-text

# Pull model if missing
ollama pull nomic-embed-text
```

## Best Practices

### 1. Keep Secrets in Environment Variables

✅ **DO:**
```yaml
# config.local.yml
llm:
  model: "gpt-4"
  base_url: "https://api.openai.com/v1"
```

```bash
# .env
OPENAI_API_KEY="sk-..."
```

❌ **DON'T:**
```yaml
# config.local.yml
llm:
  api_key: "sk-..."  # Never commit API keys!
```

### 2. Use .gitignore

Add to `.gitignore`:
```
config/config.local.yml
config/**/*.local.yml
.env
```

### 3. Document Your Setup

Create a `config/README.md` or `config/config.local.yml.example` with:
- Which providers you're using
- Required environment variables
- Setup instructions for new team members

### 4. Test Configuration Changes

```bash
# Always test after config changes
uv run src/graphiti_mcp_server.py --transport stdio

# Check the startup logs for:
# "Using [Provider] LLM: [model] at [url]"
# "Using [Provider] embedder: [model] at [url]"
```

## See Also

- [Mixed Provider Setup Guide](MIXED_PROVIDER_SETUP.md) - Detailed guide for mixing providers
- [Ollama Setup Guide](../README.md#ollama-setup) - Installing and configuring Ollama
- [Configuration Reference](../config/README.md) - All configuration options
