#!/bin/bash
# Environment Setup Script
# Source this file to set up required environment variables for testing and running the MCP server
#
# Usage:
#   source scripts/setup_env.sh
#   OR
#   . scripts/setup_env.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up Graphiti MCP Server environment...${NC}"
echo ""

# Set OPENAI_API_KEY if not already exported
if [ -z "$OPENAI_API_KEY" ]; then
    # Try to get from .env file first
    if [ -f .env ] && grep -q "^OPENAI_API_KEY=" .env; then
        export OPENAI_API_KEY=$(grep "^OPENAI_API_KEY=" .env | cut -d= -f2- | tr -d '"' | tr -d "'")
        if [ ! -z "$OPENAI_API_KEY" ]; then
            masked_key=$(echo $OPENAI_API_KEY | sed 's/\(.\{5\}\).*\(.\{4\}\)/\1***\2/')
            echo -e "${GREEN}✓${NC} OPENAI_API_KEY loaded from .env: $masked_key"
        fi
    fi

    # If still not set, prompt user
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${YELLOW}⚠${NC} OPENAI_API_KEY not set"
        echo "  Please set it manually:"
        echo "    export OPENAI_API_KEY='your-key-here'"
        echo ""
    fi
else
    masked_key=$(echo $OPENAI_API_KEY | sed 's/\(.\{5\}\).*\(.\{4\}\)/\1***\2/')
    echo -e "${GREEN}✓${NC} OPENAI_API_KEY already set: $masked_key"
    # Make sure it's exported
    export OPENAI_API_KEY
fi

# Set SSL certificate paths for enterprise endpoints
if [ -f "$HOME/.certs/ca-bundle.pem" ]; then
    export SSL_CERT_FILE="$HOME/.certs/ca-bundle.pem"
    export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
    export CURL_CA_BUNDLE="$SSL_CERT_FILE"
    echo -e "${GREEN}✓${NC} SSL certificates configured: $SSL_CERT_FILE"
elif [ ! -z "$SSL_CERT_FILE" ]; then
    export SSL_CERT_FILE
    export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
    export CURL_CA_BUNDLE="$SSL_CERT_FILE"
    echo -e "${GREEN}✓${NC} SSL certificates configured: $SSL_CERT_FILE"
else
    echo -e "${YELLOW}⚠${NC} SSL certificates not found (OK if not using enterprise gateway endpoints)"
fi

# Set Neo4j connection if not already set
if [ -z "${NEO4J_URI:-}" ]; then
    export NEO4J_URI="bolt://localhost:7687"
    echo -e "${BLUE}ℹ${NC} NEO4J_URI set to default: $NEO4J_URI"
else
    export NEO4J_URI
    echo -e "${GREEN}✓${NC} NEO4J_URI: $NEO4J_URI"
fi

if [ -z "${NEO4J_USER:-}" ]; then
    export NEO4J_USER="neo4j"
    echo -e "${BLUE}ℹ${NC} NEO4J_USER set to default: $NEO4J_USER"
else
    export NEO4J_USER
    echo -e "${GREEN}✓${NC} NEO4J_USER: $NEO4J_USER"
fi

if [ -z "${NEO4J_PASSWORD:-}" ]; then
    export NEO4J_PASSWORD="demodemo"
    echo -e "${BLUE}ℹ${NC} NEO4J_PASSWORD set to default: ********"
else
    export NEO4J_PASSWORD
    echo -e "${GREEN}✓${NC} NEO4J_PASSWORD: ********"
fi

echo ""
echo -e "${GREEN}Environment setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Test configuration: scripts/test_model_compatibility.sh"
echo "  2. Start MCP server: uv run src/graphiti_mcp_server.py"
echo ""
