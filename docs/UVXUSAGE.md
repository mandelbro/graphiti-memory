# Using Graphiti MCP Server with uvx

This guide shows how to use the Graphiti MCP Server as a Python package that can be installed and run with `uvx` (uv's tool runner).

## What is uvx?

[uvx](https://docs.astral.sh/uv/guides/tools/) is uv's equivalent to `pipx` - it allows you to install and run Python CLI tools in isolated environments without affecting your system Python.

## Quick Start with uvx

### 1. Install and Run Directly (Recommended)

You can run the Graphiti MCP Server directly without installation:

```bash
# Run with default settings
uvx montesmakes.graphiti-memory

# Run with custom configuration
uvx montesmakes.graphiti-memory --transport stdio --port 8021 --group-id my-project

# Run with Ollama configuration
uvx montesmakes.graphiti-memory --ollama-llm-model llama3.2:3b --ollama-embedding-model all-minilm-l6-v2

# Run with OpenAI instead of Ollama
uvx --with openai montesmakes.graphiti-memory --use-ollama false --model gpt-4o-mini
```

### 2. Install as a Tool (For Persistent Use)

If you plan to use it frequently, install it as a tool:

```bash
# Install the tool
uv tool install montesmakes.graphiti-memory

# Run the installed tool
montesmakes.graphiti-memory --help

# Update to the latest version
uv tool upgrade montesmakes.graphiti-memory

# Uninstall when no longer needed
uv tool uninstall montesmakes.graphiti-memory
```

## Environment Setup

Before running, ensure you have the required services:

### Option A: Using Ollama (Default)

1. **Install and start Ollama:**
   ```bash
   # Install Ollama (visit https://ollama.ai for instructions)
   ollama serve
   ```

2. **Pull required models:**
   ```bash
   ollama pull deepseek-r1:7b      # LLM model
   ollama pull nomic-embed-text    # Embedding model
   ```

3. **Start Neo4j:**
   ```bash
   docker run -p 7474:7474 -p 7687:7687 neo4j:5.26.2
   ```

4. **Run the server:**
   ```bash
   uvx montesmakes.graphiti-memory
   ```

### Option B: Using OpenAI

1. **Set environment variables:**
   ```bash
   export USE_OLLAMA=false
   export OPENAI_API_KEY=your_api_key_here
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=password
   ```

2. **Run with OpenAI:**
   ```bash
   uvx montesmakes.graphiti-memory --use-ollama false --model gpt-4o-mini
   ```

## Configuration Examples

### Basic Usage

```bash
# Default configuration (Ollama + SSE transport)
uvx montesmakes.graphiti-memory

# Use stdio transport (for Claude Desktop)
uvx montesmakes.graphiti-memory --transport stdio

# Specify a custom group ID
uvx montesmakes.graphiti-memory --group-id my-knowledge-graph
```

### Ollama Customization

```bash
# Use different models
uvx montesmakes.graphiti-memory \
  --ollama-llm-model llama3.2:8b \
  --ollama-embedding-model all-minilm-l6-v2 \
  --ollama-embedding-dim 384

# Connect to remote Ollama
uvx montesmakes.graphiti-memory \
  --ollama-base-url http://remote-server:11434/v1

# Adjust LLM parameters
uvx montesmakes.graphiti-memory \
  --temperature 0.2 \
  --max-tokens 16384
```

### Advanced Configuration

```bash
# Development mode with custom entities
uvx montesmakes.graphiti-memory \
  --transport sse \
  --port 8021 \
  --use-custom-entities \
  --group-id development

# Production setup with specific models
uvx montesmakes.graphiti-memory \
  --host 0.0.0.0 \
  --port 8020 \
  --ollama-llm-model mistral:7b \
  --temperature 0.1 \
  --group-id production
```

## Integration with MCP Clients

### Claude Desktop (stdio transport)

Configure Claude Desktop to use the uvx-installed package:

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "command": "uvx",
      "args": [
        "montesmakes.graphiti-memory",
        "--transport", "stdio",
        "--group-id", "claude-memory"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password"
      }
    }
  }
}
```

### Cursor IDE (SSE transport)

1. Start the server:
   ```bash
   uvx montesmakes.graphiti-memory --transport sse --port 8020
   ```

2. Configure Cursor:
   ```json
   {
     "mcpServers": {
       "graphiti-memory": {
         "url": "http://localhost:8020/sse"
       }
     }
   }
   ```

## Troubleshooting

### Package Not Found

If `uvx montesmakes.graphiti-memory` fails:

```bash
# Install from local build
uvx --from ./dist/graphiti_mcp_server-0.4.0-py3-none-any.whl montesmakes.graphiti-memory

# Install from git repository
uvx --from git+https://github.com/mandelbro/graphiti-memory.git montesmakes.graphiti-memory
```

### Dependency Issues

```bash
# Install with specific dependencies
uvx --with graphiti-core>=0.18.5 --with openai>=1.99.9 montesmakes.graphiti-memory

# Force reinstall
uvx --isolated montesmakes.graphiti-memory
```

### Service Connection Issues

```bash
# Check Neo4j connection
docker ps | grep neo4j

# Check Ollama status
curl http://localhost:11434/v1/models

# Verify environment variables
uvx montesmakes.graphiti-memory --help
```

## Benefits of uvx

1. **Isolation**: Runs in its own environment without conflicting dependencies
2. **Convenience**: No need to manage virtual environments manually
3. **Clean**: Automatically handles dependency resolution and cleanup
4. **Fast**: Caches environments for quick startup
5. **Portable**: Works consistently across different systems

## Publishing to PyPI

Once published to PyPI, users can install directly:

```bash
# Install from PyPI (when published)
uvx montesmakes.graphiti-memory

# Or install specific version
uvx montesmakes.graphiti-memory==0.4.0
```

## Development Workflow

For developers working on the package:

```bash
# Test local build
uv build
uvx --from ./dist/graphiti_mcp_server-0.4.0-py3-none-any.whl montesmakes.graphiti-memory

# Test editable installation
uvx --with-editable . montesmakes.graphiti-memory

# Publish to PyPI
uv publish
```
