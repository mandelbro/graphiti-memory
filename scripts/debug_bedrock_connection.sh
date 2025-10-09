#!/bin/bash
# Debug script for testing Bedrock gateway connectivity
# Tests SSL certificates, API key, and endpoint accessibility

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    set -a  # Automatically export all variables
    source .env
    set +a
    echo -e "${GREEN}✓${NC} Loaded environment variables from .env file"
    echo ""
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

# Helper function to list available models
list_available_models() {
    local base_url=$1

    echo ""
    print_info "Checking available models at: $base_url/v1/models"
    echo ""

    response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
        --cacert "$SSL_CERT_FILE" \
        -X GET "$base_url/v1/models" \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        -H "Content-Type: application/json" \
        --max-time 30 2>&1)

    http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)
    body=$(echo "$response" | sed '/HTTP_STATUS/d')

    if [ "$http_status" = "200" ]; then
        print_success "Successfully retrieved model list (HTTP $http_status)"
        echo ""
        print_info "Available models:"
        echo "$body" | python3 -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    models = data.get('data', [])
    if models:
        for model in models:
            model_id = model.get('id', 'N/A')
            owned_by = model.get('owned_by', 'N/A')
            print(f'  • {model_id} (owned by: {owned_by})')
    else:
        print('  No models found in response')
except Exception as e:
    print(f'  Error parsing model list: {e}')
" 2>/dev/null || echo "$body"
    else
        print_error "Failed to retrieve model list (HTTP $http_status)"
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
    fi
    echo ""
}

# Test 1: Check SSL Certificate
print_header "Test 1: SSL Certificate Check"

if [ -f "$SSL_CERT_FILE" ]; then
    print_success "Certificate file exists: $SSL_CERT_FILE"

    # Check if readable
    if [ -r "$SSL_CERT_FILE" ]; then
        print_success "Certificate file is readable"

        # Count certificates in bundle
        cert_count=$(grep -c "BEGIN CERTIFICATE" "$SSL_CERT_FILE" || echo "0")
        print_info "Certificate bundle contains $cert_count certificate(s)"
    else
        print_error "Certificate file is not readable"
        exit 1
    fi
else
    print_error "Certificate file not found: $SSL_CERT_FILE"
    exit 1
fi

# Test 2: Check Environment Variables
print_header "Test 2: Environment Variables"

echo "Checking SSL certificate variables:"
if [ ! -z "$SSL_CERT_FILE" ]; then
    print_success "SSL_CERT_FILE: $SSL_CERT_FILE"
else
    print_warning "SSL_CERT_FILE not set"
    export SSL_CERT_FILE="$SSL_CERT_FILE"
    print_info "Set SSL_CERT_FILE=$SSL_CERT_FILE"
fi

if [ ! -z "$REQUESTS_CA_BUNDLE" ]; then
    print_success "REQUESTS_CA_BUNDLE: $REQUESTS_CA_BUNDLE"
else
    print_warning "REQUESTS_CA_BUNDLE not set"
    export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
    print_info "Set REQUESTS_CA_BUNDLE=$SSL_CERT_FILE"
fi

if [ ! -z "$CURL_CA_BUNDLE" ]; then
    print_success "CURL_CA_BUNDLE: $CURL_CA_BUNDLE"
else
    print_warning "CURL_CA_BUNDLE not set"
    export CURL_CA_BUNDLE="$SSL_CERT_FILE"
    print_info "Set CURL_CA_BUNDLE=$SSL_CERT_FILE"
fi

echo ""
echo "Checking API key:"
if [ ! -z "$OPENAI_API_KEY" ]; then
    masked_key=$(echo $OPENAI_API_KEY | sed 's/\(.\{5\}\).*\(.\{4\}\)/\1*************\2/')
    print_success "OPENAI_API_KEY: $masked_key"
else
    print_error "OPENAI_API_KEY not set"
    print_info "Please set: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Test 3: Load Configuration
print_header "Test 3: Configuration Check"

CONFIG_FILE="config/providers/openai.local.yml"
if [ -f "$CONFIG_FILE" ]; then
    print_success "Found config file: $CONFIG_FILE"

    # Extract base URLs
    llm_base_url=$(grep -A 5 "^llm:" "$CONFIG_FILE" | grep "base_url:" | head -1 | awk '{print $2}' | tr -d '"')
    embedder_base_url=$(grep -A 5 "^embedder:" "$CONFIG_FILE" | grep "base_url:" | head -1 | awk '{print $2}' | tr -d '"')
    llm_model=$(grep -A 5 "^llm:" "$CONFIG_FILE" | grep "model:" | head -1 | awk '{print $2}' | tr -d '"')
    embedder_model=$(grep -A 5 "^embedder:" "$CONFIG_FILE" | grep "model:" | head -1 | awk '{print $2}' | tr -d '"')

    print_info "LLM Base URL: $llm_base_url"
    print_info "LLM Model: $llm_model"
    print_info "Embedder Base URL: $embedder_base_url"
    print_info "Embedder Model: $embedder_model"
else
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Test 4: Test SSL Connection with curl
print_header "Test 4: SSL Connection Test (curl)"

# Extract gateway hostname from the LLM base URL
GATEWAY_HOST="${llm_base_url#https://}"
GATEWAY_HOST="${GATEWAY_HOST#http://}"
GATEWAY_HOST="${GATEWAY_HOST%%/*}"

print_info "Testing SSL connection to: $GATEWAY_HOST"
echo ""

if curl --cacert "$SSL_CERT_FILE" -s -o /dev/null -w "%{http_code}" "https://$GATEWAY_HOST" > /tmp/curl_test.txt 2>&1; then
    status_code=$(cat /tmp/curl_test.txt)
    if [ "$status_code" != "000" ]; then
        print_success "SSL connection successful (HTTP $status_code)"
    else
        print_error "SSL connection failed"
        curl --cacert "$SSL_CERT_FILE" -v "https://$GATEWAY_HOST" 2>&1 | head -20
        exit 1
    fi
else
    print_error "curl command failed"
    exit 1
fi

# Test 5: Test LLM Endpoint with curl
print_header "Test 5: LLM Endpoint Test"

print_info "Testing: $llm_base_url/v1/chat/completions"
echo ""

response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    --cacert "$SSL_CERT_FILE" \
    -X POST "$llm_base_url/v1/chat/completions" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    --max-time 30 \
    -d '{
        "model": "'"$llm_model"'",
        "messages": [
            {"role": "user", "content": "Say: Connection test successful"}
        ],
        "max_tokens": 50
    }' 2>&1)

http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

if [ "$http_status" = "200" ]; then
    print_success "LLM endpoint works! (HTTP $http_status)"
    # Try to extract response content
    content=$(echo "$body" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('choices', [{}])[0].get('message', {}).get('content', 'N/A'))" 2>/dev/null || echo "")
    if [ ! -z "$content" ] && [ "$content" != "N/A" ]; then
        print_info "Response: $content"
    fi
elif [ "$http_status" = "401" ]; then
    print_error "Authentication failed (HTTP 401)"
    print_info "Your API key may not be valid for this endpoint"
    error_msg=$(echo "$body" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('error', {}).get('message', 'Unknown error'))" 2>/dev/null || echo "")
    if [ ! -z "$error_msg" ]; then
        print_info "Error: $error_msg"
    fi
elif [ "$http_status" = "404" ]; then
    print_error "Endpoint not found (HTTP 404)"
    print_warning "The /v1/chat/completions path may not exist"
    print_info "Try updating base_url to include the correct path"
    list_available_models "$llm_base_url"
else
    print_error "Request failed (HTTP $http_status)"
    echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"

    # Check if error message suggests checking available models
    if echo "$body" | grep -q "model"; then
        print_warning "Error mentions model - checking available models..."
        list_available_models "$llm_base_url"
    fi
fi

# Test 6: Test Embedder Endpoint
print_header "Test 6: Embedder Endpoint Test"

print_info "Testing: $embedder_base_url/embeddings"
echo ""

response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    --cacert "$SSL_CERT_FILE" \
    -X POST "$embedder_base_url/embeddings" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    --max-time 30 \
    -d '{
        "model": "'"$embedder_model"'",
        "input": "Connection test"
    }' 2>&1)

http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

if [ "$http_status" = "200" ]; then
    print_success "Embedder endpoint works! (HTTP $http_status)"
    dimension=$(echo "$body" | python3 -c "import json,sys; data=json.load(sys.stdin); print(len(data.get('data', [{}])[0].get('embedding', [])))" 2>/dev/null || echo "N/A")
    if [ "$dimension" != "N/A" ]; then
        print_info "Embedding dimension: $dimension"
    fi
elif [ "$http_status" = "401" ]; then
    print_error "Authentication failed (HTTP 401)"
    print_info "Your API key may not be valid for this endpoint"
    list_available_models "$embedder_base_url"
elif [ "$http_status" = "404" ]; then
    print_error "Endpoint not found (HTTP 404)"
    print_warning "The embeddings path may not exist at this base_url"
    print_info "Try: $llm_base_url/v1/embeddings"
    list_available_models "$embedder_base_url"
else
    print_error "Request failed (HTTP $http_status)"
    echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"

    # Check if error message suggests checking available models
    if echo "$body" | grep -q "model\|/v1/models"; then
        print_warning "Error mentions models - checking available models..."
        list_available_models "$embedder_base_url"
    fi
fi

# Test 7: Python SSL Test
print_header "Test 7: Python SSL Verification"

print_info "Testing Python SSL certificate handling..."
echo ""

# Export variables for Python script
export LLM_BASE_URL="$llm_base_url"

python3 << 'EOF'
import os
import sys
import ssl
import httpx

cert_path = os.path.expanduser(os.getenv("SSL_CERT_FILE", ""))
# Extract gateway host from LLM_BASE_URL environment variable
base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com")
gateway_host = base_url.replace("https://", "").replace("http://", "").split("/")[0]

print(f"Certificate path: {cert_path}")
print(f"File exists: {os.path.exists(cert_path)}")
print("")

# Test 1: SSL context
try:
    ssl_context = ssl.create_default_context(cafile=cert_path)
    print("✓ SSL context created successfully")
except Exception as e:
    print(f"✗ Failed to create SSL context: {e}")
    sys.exit(1)

# Test 2: httpx with certificate
print("\nTesting httpx with certificate...")
try:
    with httpx.Client(verify=cert_path, timeout=10.0) as client:
        response = client.get(f"https://{gateway_host}")
        print(f"✓ Connection successful (HTTP {response.status_code})")
except httpx.ConnectError as e:
    print(f"✗ Connection failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"⚠ Request completed but with error: {e}")

print("\n✓ Python SSL verification works!")
EOF

if [ $? -eq 0 ]; then
    print_success "Python can use SSL certificates correctly"
else
    print_error "Python SSL verification failed"
    exit 1
fi

# Summary
print_header "Debug Summary"

print_success "All connection tests completed!"
echo ""
print_info "Next steps:"
echo "  1. If LLM endpoint failed: Check the base_url path in config"
echo "  2. If embedder endpoint failed: Check the embedder base_url path"
echo "  3. If authentication failed: Verify your OPENAI_API_KEY is correct"
echo "  4. If model errors occurred: Check the available models list shown above"
echo "  5. Run the actual tests: scripts/test_bedrock_endpoint.sh --connection-only"
echo ""
