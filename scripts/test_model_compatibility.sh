#!/bin/bash
# Test Model Compatibility Script
# Validates that configured models work with their specified parameters
# Tests actual API calls to catch issues like temperature constraints

set -e

# Colors for output (define early so we can use them)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables from .env file if it exists
ENV_FILE_LOADED=false
if [ -f .env ]; then
    set -a  # Automatically export all variables
    source .env
    set +a
    ENV_FILE_LOADED=true
    echo -e "${GREEN}✓${NC} Loaded environment variables from .env file"
fi

print_header() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }

# Parse YAML function (simple grep-based parser)
get_yaml_value() {
    local file=$1
    local section=$2
    local key=$3

    # Extract value from YAML using awk
    awk -v section="$section" -v key="$key" '
        BEGIN { in_section=0 }
        /^[a-z_]+:/ {
            in_section=0
            if ($0 ~ "^" section ":") in_section=1
        }
        in_section && $0 ~ "^  " key ":" {
            sub(/^  [^:]+: */, "")
            gsub(/"/, "")
            print
            exit
        }
    ' "$file"
}

# Test 1: Load Configuration
print_header "Test 1: Configuration Loading"

CONFIG_FILE="${1:-config/config.local.yml}"

if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    print_info "Usage: $0 [config_file]"
    print_info "Default: config/config.local.yml"
    exit 1
fi

print_success "Found config file: $CONFIG_FILE"

# Extract LLM configuration
llm_model=$(get_yaml_value "$CONFIG_FILE" "llm" "model")
llm_base_url=$(get_yaml_value "$CONFIG_FILE" "llm" "base_url")
llm_temperature=$(get_yaml_value "$CONFIG_FILE" "llm" "temperature")
llm_max_tokens=$(get_yaml_value "$CONFIG_FILE" "llm" "max_tokens")
llm_use_ollama=$(get_yaml_value "$CONFIG_FILE" "llm" "use_ollama")

# Extract embedder configuration
embedder_model=$(get_yaml_value "$CONFIG_FILE" "embedder" "model")
embedder_base_url=$(get_yaml_value "$CONFIG_FILE" "embedder" "base_url")
embedder_dimension=$(get_yaml_value "$CONFIG_FILE" "embedder" "dimension")

echo ""
print_info "LLM Configuration:"
echo "  Model: $llm_model"
echo "  Base URL: $llm_base_url"
echo "  Temperature: $llm_temperature"
echo "  Max Tokens: $llm_max_tokens"
echo "  Use Ollama: $llm_use_ollama"

echo ""
print_info "Embedder Configuration:"
echo "  Model: $embedder_model"
echo "  Base URL: $embedder_base_url"
echo "  Dimension: $embedder_dimension"

# Test 2: Check Prerequisites
print_header "Test 2: Prerequisites Check"

# Check API Key
if [ -z "$OPENAI_API_KEY" ]; then
    print_error "OPENAI_API_KEY not found"
    echo ""
    if [ -f .env ]; then
        print_info "Checked .env file but OPENAI_API_KEY is not set there."
        print_info "Add it to .env file:"
        echo ""
        echo "  echo 'OPENAI_API_KEY=your-key-here' >> .env"
        echo ""
    else
        print_info "No .env file found. Create one with:"
        echo ""
        echo "  echo 'OPENAI_API_KEY=your-key-here' > .env"
        echo ""
    fi
    print_info "OR export it directly:"
    echo ""
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
    exit 1
fi
masked_key=$(echo $OPENAI_API_KEY | sed 's/\(.\{5\}\).*\(.\{4\}\)/\1*************\2/')
if [ "$ENV_FILE_LOADED" = true ]; then
    print_success "API Key loaded from .env: $masked_key"
else
    print_success "API Key set: $masked_key"
fi

# Check SSL certificate for remote endpoints
needs_ssl=false
if [[ "$llm_base_url" == https://* ]] || [[ "$embedder_base_url" == https://* ]]; then
    needs_ssl=true
fi

if [ "$needs_ssl" = true ]; then
    if [ -z "$SSL_CERT_FILE" ] || [ ! -f "$SSL_CERT_FILE" ]; then
        print_warning "SSL certificate not configured for enterprise endpoint"
        print_info "Set: export SSL_CERT_FILE=/path/to/cert.pem"
        print_info "Attempting to continue without SSL verification..."
        ssl_opt="-k"
    else
        print_success "SSL certificate configured: $SSL_CERT_FILE"
        ssl_opt="--cacert $SSL_CERT_FILE"
    fi
else
    ssl_opt=""
fi

# Check if jq is available for better JSON parsing
if command -v jq &> /dev/null; then
    print_success "jq available for JSON parsing"
    has_jq=true
else
    print_warning "jq not found, using python for JSON parsing"
    has_jq=false
fi

# Test 3: LLM Model Compatibility Test
print_header "Test 3: LLM Model Compatibility"

print_info "Testing model: $llm_model"
print_info "Testing temperature: $llm_temperature"
echo ""

# Determine endpoint based on base_url format
if [[ "$llm_base_url" == */v1 ]]; then
    llm_endpoint="$llm_base_url/chat/completions"
else
    llm_endpoint="$llm_base_url/v1/chat/completions"
fi

print_info "Testing endpoint: $llm_endpoint"
echo ""

# Test with configured parameters
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    $ssl_opt \
    -X POST "$llm_endpoint" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    --max-time 30 \
    -d "{
        \"model\": \"$llm_model\",
        \"messages\": [
            {\"role\": \"user\", \"content\": \"Say 'OK' if you receive this.\"}
        ],
        \"temperature\": $llm_temperature,
        \"max_tokens\": 10
    }" 2>&1)

http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

if [ "$http_status" = "200" ]; then
    print_success "LLM model works with configured parameters! (HTTP $http_status)"

    # Extract and display response
    if [ "$has_jq" = true ]; then
        content=$(echo "$body" | jq -r '.choices[0].message.content // "N/A"' 2>/dev/null)
    else
        content=$(echo "$body" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('choices', [{}])[0].get('message', {}).get('content', 'N/A'))" 2>/dev/null || echo "N/A")
    fi

    if [ "$content" != "N/A" ]; then
        print_info "Model response: $content"
    fi

    llm_compatible=true
else
    print_error "LLM model FAILED with configured parameters (HTTP $http_status)"
    llm_compatible=false

    # Try to extract error message
    if [ "$has_jq" = true ]; then
        error_msg=$(echo "$body" | jq -r '.error.message // .message // "Unknown error"' 2>/dev/null)
        error_type=$(echo "$body" | jq -r '.error.type // "unknown"' 2>/dev/null)
    else
        error_msg=$(echo "$body" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('error', {}).get('message', data.get('message', 'Unknown error')))" 2>/dev/null || echo "Unknown error")
        error_type=$(echo "$body" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('error', {}).get('type', 'unknown'))" 2>/dev/null || echo "unknown")
    fi

    echo ""
    print_error "Error Type: $error_type"
    print_error "Error Message: $error_msg"
    echo ""

    # Check for common parameter constraint issues
    if echo "$error_msg" | grep -qi "temperature"; then
        print_warning "Temperature constraint detected!"
        print_info "Trying with different temperature values..."
        echo ""

        # Test with temperature = 1.0
        print_info "Testing with temperature=1.0..."
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
            $ssl_opt \
            -X POST "$llm_endpoint" \
            -H "Authorization: Bearer $OPENAI_API_KEY" \
            -H "Content-Type: application/json" \
            --max-time 30 \
            -d "{
                \"model\": \"$llm_model\",
                \"messages\": [{\"role\": \"user\", \"content\": \"OK\"}],
                \"temperature\": 1.0,
                \"max_tokens\": 10
            }" 2>&1)

        alt_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)

        if [ "$alt_status" = "200" ]; then
            print_success "Model works with temperature=1.0"
            print_warning "Your config has temperature=$llm_temperature which is NOT compatible"
            print_info "Recommendation: Update config.local.yml with temperature >= 1.0"
        else
            print_error "Model still fails with temperature=1.0"
        fi
        echo ""
    fi

    if echo "$error_msg" | grep -qi "model"; then
        print_warning "Model availability issue detected!"
        print_info "The model '$llm_model' may not be available on this endpoint"
        print_info "Checking available models..."
        echo ""

        # Try to list available models
        models_response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
            $ssl_opt \
            -X GET "${llm_base_url}/v1/models" \
            -H "Authorization: Bearer $OPENAI_API_KEY" \
            --max-time 30 2>&1)

        models_status=$(echo "$models_response" | grep "HTTP_STATUS" | cut -d: -f2)
        models_body=$(echo "$models_response" | sed '/HTTP_STATUS/d')

        if [ "$models_status" = "200" ]; then
            print_info "Available models:"
            if [ "$has_jq" = true ]; then
                echo "$models_body" | jq -r '.data[]? | "  • \(.id)"' 2>/dev/null | head -20
            else
                echo "$models_body" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for model in data.get('data', [])[:20]:
        print(f\"  • {model.get('id', 'N/A')}\")
except: pass
" 2>/dev/null
            fi
        fi
    fi
fi

# Test 4: Embedder Model Compatibility Test
print_header "Test 4: Embedder Model Compatibility"

print_info "Testing model: $embedder_model"
echo ""

# Determine endpoint based on base_url format
if [[ "$embedder_base_url" == */v1 ]]; then
    embedder_endpoint="$embedder_base_url/embeddings"
else
    embedder_endpoint="$embedder_base_url/v1/embeddings"
fi

print_info "Testing endpoint: $embedder_endpoint"
echo ""

# Test embedder
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    $ssl_opt \
    -X POST "$embedder_endpoint" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    --max-time 30 \
    -d "{
        \"model\": \"$embedder_model\",
        \"input\": \"test embedding\"
    }" 2>&1)

http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

if [ "$http_status" = "200" ]; then
    print_success "Embedder model works! (HTTP $http_status)"

    # Check embedding dimension
    if [ "$has_jq" = true ]; then
        actual_dim=$(echo "$body" | jq -r '.data[0].embedding | length' 2>/dev/null)
    else
        actual_dim=$(echo "$body" | python3 -c "import json,sys; data=json.load(sys.stdin); print(len(data.get('data', [{}])[0].get('embedding', [])))" 2>/dev/null || echo "N/A")
    fi

    if [ "$actual_dim" != "N/A" ]; then
        print_info "Embedding dimension: $actual_dim"

        if [ "$actual_dim" = "$embedder_dimension" ]; then
            print_success "Dimension matches config ($embedder_dimension)"
        else
            print_warning "Dimension mismatch!"
            print_warning "Config specifies: $embedder_dimension"
            print_warning "Model returns: $actual_dim"
            print_info "Recommendation: Update config.local.yml with dimension: $actual_dim"
        fi
    fi

    embedder_compatible=true
else
    print_error "Embedder model FAILED (HTTP $http_status)"
    embedder_compatible=false

    # Extract error message
    if [ "$has_jq" = true ]; then
        error_msg=$(echo "$body" | jq -r '.error.message // .message // "Unknown error"' 2>/dev/null)
    else
        error_msg=$(echo "$body" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('error', {}).get('message', data.get('message', 'Unknown error')))" 2>/dev/null || echo "Unknown error")
    fi

    echo ""
    print_error "Error Message: $error_msg"
    echo ""

    if echo "$error_msg" | grep -qi "model"; then
        print_warning "Model availability issue detected!"
        print_info "The model '$embedder_model' may not be available on this endpoint"

        # For Ollama, suggest pulling the model
        if [[ "$embedder_base_url" == *"localhost"* ]] || [[ "$embedder_base_url" == *"11434"* ]]; then
            print_info "This appears to be a local Ollama instance"
            print_info "Try: ollama pull $embedder_model"
        fi
    fi
fi

# Test 5: Integration Test (if both pass)
if [ "$llm_compatible" = true ] && [ "$embedder_compatible" = true ]; then
    print_header "Test 5: Integration Test"

    print_info "Testing LLM and Embedder together..."
    echo ""

    # Create a test that uses both
    print_info "1. Generating text with LLM..."
    llm_response=$(curl -s $ssl_opt \
        -X POST "$llm_endpoint" \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        -H "Content-Type: application/json" \
        --max-time 30 \
        -d "{
            \"model\": \"$llm_model\",
            \"messages\": [{\"role\": \"user\", \"content\": \"In one sentence, what is knowledge graph?\"}],
            \"temperature\": $llm_temperature,
            \"max_tokens\": 50
        }")

    if [ "$has_jq" = true ]; then
        llm_text=$(echo "$llm_response" | jq -r '.choices[0].message.content // "N/A"')
    else
        llm_text=$(echo "$llm_response" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('choices', [{}])[0].get('message', {}).get('content', 'N/A'))")
    fi

    if [ "$llm_text" != "N/A" ] && [ ! -z "$llm_text" ]; then
        print_success "LLM generated text: ${llm_text:0:100}..."

        print_info "2. Creating embedding from LLM output..."
        embed_response=$(curl -s $ssl_opt \
            -X POST "$embedder_endpoint" \
            -H "Authorization: Bearer $OPENAI_API_KEY" \
            -H "Content-Type: application/json" \
            --max-time 30 \
            -d "{
                \"model\": \"$embedder_model\",
                \"input\": \"$llm_text\"
            }")

        if [ "$has_jq" = true ]; then
            embed_dim=$(echo "$embed_response" | jq -r '.data[0].embedding | length' 2>/dev/null)
        else
            embed_dim=$(echo "$embed_response" | python3 -c "import json,sys; data=json.load(sys.stdin); print(len(data.get('data', [{}])[0].get('embedding', [])))" 2>/dev/null || echo "0")
        fi

        if [ "$embed_dim" -gt 0 ]; then
            print_success "Successfully created embedding (dimension: $embed_dim)"
            print_success "Integration test PASSED!"
        else
            print_error "Failed to create embedding from LLM output"
        fi
    else
        print_warning "Could not get LLM response for integration test"
    fi
fi

# Summary
print_header "Compatibility Summary"

echo ""
if [ "$llm_compatible" = true ]; then
    print_success "LLM Configuration: COMPATIBLE"
    echo "  Model: $llm_model"
    echo "  Temperature: $llm_temperature"
    echo "  Endpoint: $llm_endpoint"
else
    print_error "LLM Configuration: INCOMPATIBLE"
    echo "  Model: $llm_model"
    echo "  Temperature: $llm_temperature"
    echo "  Endpoint: $llm_endpoint"
    echo ""
    print_info "Action Required: Fix LLM configuration before using MCP server"
fi

echo ""
if [ "$embedder_compatible" = true ]; then
    print_success "Embedder Configuration: COMPATIBLE"
    echo "  Model: $embedder_model"
    echo "  Endpoint: $embedder_endpoint"
else
    print_error "Embedder Configuration: INCOMPATIBLE"
    echo "  Model: $embedder_model"
    echo "  Endpoint: $embedder_endpoint"
    echo ""
    print_info "Action Required: Fix embedder configuration before using MCP server"
fi

echo ""
if [ "$llm_compatible" = true ] && [ "$embedder_compatible" = true ]; then
    print_success "✓ All models are compatible with the MCP server!"
    echo ""
    print_info "You can safely start the MCP server with these settings."
    exit 0
else
    print_error "✗ Configuration issues detected"
    echo ""
    print_info "Please fix the issues above before starting the MCP server."
    exit 1
fi
