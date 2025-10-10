#!/usr/bin/env zsh
# Configuration Test Script for Graphiti MCP Server
#
# This script validates your entire configuration setup including:
# - Config file validity
# - LLM connectivity
# - Embedder connectivity
# - Neo4j connectivity
# - Environment variables

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print section header
print_header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# Print success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Print error message
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Print info message
print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNINGS=0

# Test result tracking
test_pass() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    PASSED_TESTS=$((PASSED_TESTS + 1))
    print_success "$1"
}

test_fail() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    FAILED_TESTS=$((FAILED_TESTS + 1))
    print_error "$1"
}

test_warn() {
    WARNINGS=$((WARNINGS + 1))
    print_warning "$1"
}

# Banner
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Graphiti Configuration Test Suite       ║${NC}"
echo -e "${CYAN}╔════════════════════════════════════════════╗${NC}"
echo ""

# Check if .env file exists and source it
if [ -f .env ]; then
    print_success "Found .env file, loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
else
    test_warn "No .env file found, using system environment variables"
fi

# Check and setup SSL certificates for internal endpoints
if [ -f "$SSL_CERT_FILE" ]; then
    print_success "Found SSL certificate bundle: $SSL_CERT_FILE"
    export SSL_CERT_FILE="$SSL_CERT_FILE"
    export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
    export CURL_CA_BUNDLE="$SSL_CERT_FILE"
    print_info "SSL environment variables configured for Python and curl"
else
    print_info "No custom SSL certificate found (using system defaults)"
fi

# ============================================================================
# Test 1: Configuration File Validation
# ============================================================================
print_header "Test 1: Configuration File Validation"

# Check for config.local.yml
if [ -f "config/config.local.yml" ]; then
    test_pass "config/config.local.yml exists"

    # Validate YAML syntax
    if python3 -c "import yaml; yaml.safe_load(open('config/config.local.yml'))" 2>/dev/null; then
        test_pass "config/config.local.yml has valid YAML syntax"

        # Extract and display configuration
        config_summary=$(python3 << 'EOF'
import yaml
with open('config/config.local.yml') as f:
    config = yaml.safe_load(f)

llm = config.get('llm', {})
embedder = config.get('embedder', {})

print(f"LLM Model: {llm.get('model', 'N/A')}")
print(f"LLM Base URL: {llm.get('base_url', 'N/A')}")
print(f"Embedder Model: {embedder.get('model', 'N/A')}")
print(f"Embedder Base URL: {embedder.get('base_url', 'N/A')}")
EOF
)
        echo "$config_summary" | while IFS= read -r line; do
            print_info "$line"
        done
    else
        test_fail "config/config.local.yml has invalid YAML syntax"
    fi
else
    test_fail "config/config.local.yml not found"
    print_info "Create it with: cp config/config.local.yml.example config/config.local.yml"
fi

# ============================================================================
# Test 2: Environment Variables
# ============================================================================
print_header "Test 2: Environment Variables"

# Check OPENAI_API_KEY
if [ ! -z "$OPENAI_API_KEY" ]; then
    test_pass "OPENAI_API_KEY is set"
    # Mask the key for security
    masked_key="${OPENAI_API_KEY:0:7}...${OPENAI_API_KEY: -4}"
    print_info "Value: $masked_key"
else
    test_warn "OPENAI_API_KEY not set (required for OpenAI/Bedrock LLM)"
fi

# Check Neo4j variables
if [ ! -z "$NEO4J_URI" ]; then
    test_pass "NEO4J_URI is set: $NEO4J_URI"
else
    test_fail "NEO4J_URI not set"
fi

if [ ! -z "$NEO4J_USER" ]; then
    test_pass "NEO4J_USER is set: $NEO4J_USER"
else
    test_fail "NEO4J_USER not set"
fi

if [ ! -z "$NEO4J_PASSWORD" ]; then
    test_pass "NEO4J_PASSWORD is set"
else
    test_fail "NEO4J_PASSWORD not set"
fi

# ============================================================================
# Test 3: Python Dependencies
# ============================================================================
print_header "Test 3: Python Dependencies"

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    test_pass "Python 3 is installed: $python_version"
else
    test_fail "Python 3 not found"
    exit 1
fi

# Check if uv is available
if command -v uv &> /dev/null; then
    uv_version=$(uv --version)
    test_pass "uv is installed: $uv_version"
else
    test_warn "uv not installed (optional, but recommended)"
fi

# Test Python imports
python3 << 'EOF'
import sys
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

try:
    import yaml
    print("✓ yaml")
except ImportError:
    print("✗ yaml (required)")
    sys.exit(1)

try:
    import neo4j
    print("✓ neo4j")
except ImportError:
    print("✗ neo4j (required)")
    sys.exit(1)

try:
    from graphiti_core import Graphiti
    print("✓ graphiti_core")
except ImportError:
    print("✗ graphiti_core (required)")
    sys.exit(1)

try:
    from openai import OpenAI
    print("✓ openai")
except ImportError:
    print("✗ openai (required)")
    sys.exit(1)

EOF

if [ $? -eq 0 ]; then
    test_pass "All required Python packages are installed"
else
    test_fail "Some required Python packages are missing"
    print_info "Run: uv sync --extra dev"
fi

# ============================================================================
# Test 4: Neo4j Connectivity
# ============================================================================
print_header "Test 4: Neo4j Connectivity"

if [ ! -z "$NEO4J_URI" ] && [ ! -z "$NEO4J_USER" ] && [ ! -z "$NEO4J_PASSWORD" ]; then
    print_info "NEO4J_URI: $NEO4J_URI"

    # Check if using Docker hostname (will hang if trying to connect)
    if [[ "$NEO4J_URI" =~ "://neo4j:" ]] || [[ "$NEO4J_URI" =~ "://db:" ]]; then
        test_warn "Docker hostname detected in NEO4J_URI"
        print_info "Cannot test connection with Docker hostname outside container"
        print_info "Running inside Docker? Make sure: docker compose up"
        print_info "Running locally? Change NEO4J_URI to: bolt://localhost:7687"
    else
        # Safe to test connection
        print_info "Testing connection..."

        neo4j_result=$(python3 -c "
import sys
from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USER', '$NEO4J_PASSWORD'), connection_timeout=5)
    driver.verify_connectivity()
    driver.close()
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>&1)

        if echo "$neo4j_result" | grep -q "SUCCESS"; then
            test_pass "Neo4j connection successful"
        else
            test_fail "Neo4j connection failed"
            error_msg=$(echo "$neo4j_result" | grep "ERROR:" | sed 's/ERROR: //')
            if [ ! -z "$error_msg" ]; then
                print_info "$error_msg"
            fi
            print_info "Make sure Neo4j is running:"
            print_info "  docker run -p 7474:7474 -p 7687:7687 neo4j:5.26.2"
        fi
    fi
else
    test_fail "Cannot test Neo4j - missing environment variables"
fi

# ============================================================================
# Test 5: LLM Configuration
# ============================================================================
print_header "Test 5: LLM Configuration"

llm_test=$(python3 << 'EOFPYTHON'
import sys
import os

sys.path.insert(0, 'src')

try:
    from config import GraphitiLLMConfig

    config = GraphitiLLMConfig.from_yaml_and_env()

    print(f"Model: {config.model}")
    print(f"Use Ollama: {config.use_ollama}")

    if config.use_ollama:
        print(f"Ollama Base URL: {config.ollama_base_url}")
        print(f"Ollama Model: {config.ollama_llm_model}")

    print("SUCCESS")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOFPYTHON
)

if [ $? -eq 0 ]; then
    test_pass "LLM configuration loaded successfully"
    echo "$llm_test" | grep -v "SUCCESS" | while IFS= read -r line; do
        print_info "$line"
    done
else
    test_fail "LLM configuration failed to load"
    echo "$llm_test"
fi

# ============================================================================
# Test 6: Embedder Configuration
# ============================================================================
print_header "Test 6: Embedder Configuration"

embedder_test=$(python3 << 'EOFPYTHON'
import sys
import os

sys.path.insert(0, 'src')

try:
    from config import GraphitiEmbedderConfig

    config = GraphitiEmbedderConfig.from_yaml_and_env()

    print(f"Model: {config.model}")
    print(f"Use Ollama: {config.use_ollama}")

    if config.use_ollama:
        print(f"Ollama Base URL: {config.ollama_base_url}")
        print(f"Ollama Model: {config.ollama_embedding_model}")
        print(f"Dimension: {config.ollama_embedding_dim}")
    else:
        if hasattr(config, 'base_url') and config.base_url:
            print(f"Base URL: {config.base_url}")
        if hasattr(config, 'dimension') and config.dimension:
            print(f"Dimension: {config.dimension}")

    print("SUCCESS")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOFPYTHON
)

if [ $? -eq 0 ]; then
    test_pass "Embedder configuration loaded successfully"
    echo "$embedder_test" | grep -v "SUCCESS" | while IFS= read -r line; do
        print_info "$line"
    done
else
    test_fail "Embedder configuration failed to load"
    echo "$embedder_test"
fi

# ============================================================================
# Test 7: Ollama Connectivity (if applicable)
# ============================================================================
print_header "Test 7: Ollama Connectivity"

# Check if config uses Ollama
uses_ollama=$(python3 << 'EOF'
import yaml
try:
    with open('config/config.local.yml') as f:
        config = yaml.safe_load(f)

    llm_url = config.get('llm', {}).get('base_url', '')
    embedder_url = config.get('embedder', {}).get('base_url', '')

    if 'localhost:11434' in llm_url or 'localhost:11434' in embedder_url:
        print("YES")
    else:
        print("NO")
except:
    print("NO")
EOF
)

if [ "$uses_ollama" = "YES" ]; then
    print_info "Configuration uses Ollama, testing connectivity..."

    # Check if Ollama is installed
    if command -v ollama &> /dev/null; then
        test_pass "Ollama is installed"

        # Check if Ollama is running
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            test_pass "Ollama is running"

            # Check for required models
            models=$(curl -s http://localhost:11434/api/tags | python3 -c "import json,sys; data=json.load(sys.stdin); print(' '.join([m['name'] for m in data.get('models', [])]))" 2>/dev/null)

            # Check for embedding model
            if echo "$models" | grep -q "nomic-embed-text"; then
                test_pass "nomic-embed-text model is available"
            else
                test_warn "nomic-embed-text model not found"
                print_info "Run: ollama pull nomic-embed-text"
            fi

            # Check for LLM model if using Ollama
            llm_model=$(python3 -c "import yaml; config=yaml.safe_load(open('config/config.local.yml')); print(config.get('llm',{}).get('model',''))" 2>/dev/null)
            if [ ! -z "$llm_model" ] && [ "$llm_model" != "None" ]; then
                if echo "$models" | grep -q "$llm_model"; then
                    test_pass "LLM model '$llm_model' is available"
                else
                    test_warn "LLM model '$llm_model' not found in Ollama"
                    print_info "Run: ollama pull $llm_model"
                fi
            fi
        else
            test_fail "Ollama is not running"
            print_info "Start Ollama with: ollama serve"
        fi
    else
        test_fail "Ollama is not installed"
        print_info "Install: curl -fsSL https://ollama.com/install.sh | sh"
    fi
else
    print_info "Configuration does not use Ollama (skipping Ollama tests)"
fi

# ============================================================================
# Test 8: LLM API Connectivity
# ============================================================================
print_header "Test 8: LLM API Connectivity"

# Get LLM base URL
llm_base_url=$(python3 -c "import yaml; config=yaml.safe_load(open('config/config.local.yml')); print(config.get('llm',{}).get('base_url',''))" 2>/dev/null)

if [ ! -z "$llm_base_url" ] && [ "$llm_base_url" != "None" ]; then
    # Skip if it's localhost (Ollama tested separately)
    if [[ ! "$llm_base_url" =~ "localhost" ]]; then
        print_info "Testing connectivity to: $llm_base_url"

        # Build curl command with SSL certificate if available
        curl_cmd="curl -s -o /dev/null -w %{http_code} --max-time 5"
        if [ -f "$SSL_CERT_FILE" ]; then
            curl_cmd="$curl_cmd --cacert $SSL_CERT_FILE"
            print_info "Using SSL certificate for connection test"
        fi

        # Try to reach the endpoint
        http_code=$(eval "$curl_cmd $llm_base_url" 2>&1)

        # Accept any valid HTTP response code (2xx, 3xx redirects, 4xx client errors from server)
        if echo "$http_code" | grep -q "200\|201\|204\|301\|302\|303\|307\|308\|400\|401\|403\|404\|405"; then
            test_pass "LLM endpoint is reachable (HTTP $http_code)"
            if [ "$http_code" = "404" ]; then
                print_info "Got 404 - base_url may need /v1 path appended"
            elif echo "$http_code" | grep -q "301\|302\|303\|307\|308"; then
                print_info "Got redirect - endpoint is responding correctly"
            fi
        else
            test_warn "LLM endpoint may not be reachable (HTTP $http_code - check VPN/network)"
        fi
    fi
fi

# ============================================================================
# Test 9: Embedder API Connectivity
# ============================================================================
print_header "Test 9: Embedder API Connectivity"

# Get embedder base URL
embedder_base_url=$(python3 -c "import yaml; config=yaml.safe_load(open('config/config.local.yml')); print(config.get('embedder',{}).get('base_url',''))" 2>/dev/null)

if [ ! -z "$embedder_base_url" ] && [ "$embedder_base_url" != "None" ]; then
    # Skip if it's localhost (Ollama tested separately)
    if [[ ! "$embedder_base_url" =~ "localhost" ]]; then
        print_info "Testing connectivity to: $embedder_base_url"

        # Build curl command with SSL certificate if available
        curl_cmd="curl -s -o /dev/null -w %{http_code} --max-time 5"
        if [ -f "$SSL_CERT_FILE" ]; then
            curl_cmd="$curl_cmd --cacert $SSL_CERT_FILE"
            print_info "Using SSL certificate for connection test"
        fi

        # Try to reach the endpoint
        http_code=$(eval "$curl_cmd $embedder_base_url" 2>&1)

        # Accept any valid HTTP response code (2xx, 3xx redirects, 4xx client errors from server)
        if echo "$http_code" | grep -q "200\|201\|204\|301\|302\|303\|307\|308\|400\|401\|403\|404\|405"; then
            test_pass "Embedder endpoint is reachable (HTTP $http_code)"
            if [ "$http_code" = "404" ]; then
                print_info "Got 404 - base_url may need correct path appended"
            elif echo "$http_code" | grep -q "301\|302\|303\|307\|308"; then
                print_info "Got redirect - endpoint is responding correctly"
            fi
        else
            test_warn "Embedder endpoint may not be reachable (HTTP $http_code - check VPN/network)"
        fi
    else
        print_info "Using local Ollama for embeddings (tested in Test 7)"
    fi
fi

# ============================================================================
# Test 10: Python SSL Certificate Handling
# ============================================================================
print_header "Test 10: Python SSL Certificate Handling"

if [ -f "$SSL_CERT_FILE" ]; then
    print_info "Testing Python SSL certificate configuration..."

    python3 << 'EOF'
import os
import sys

# Check environment variables
ssl_cert_file = os.getenv("SSL_CERT_FILE")
requests_ca_bundle = os.getenv("REQUESTS_CA_BUNDLE")
curl_ca_bundle = os.getenv("CURL_CA_BUNDLE")

if ssl_cert_file:
    print(f"✓ SSL_CERT_FILE: {ssl_cert_file}")
else:
    print("⚠ SSL_CERT_FILE not set")

if requests_ca_bundle:
    print(f"✓ REQUESTS_CA_BUNDLE: {requests_ca_bundle}")
else:
    print("⚠ REQUESTS_CA_BUNDLE not set")

if curl_ca_bundle:
    print(f"✓ CURL_CA_BUNDLE: {curl_ca_bundle}")
else:
    print("⚠ CURL_CA_BUNDLE not set")

# Test if the certificate file is readable
if ssl_cert_file and os.path.exists(ssl_cert_file):
    print(f"✓ Certificate file exists and is readable")
else:
    print("✗ Certificate file not found or not readable")
    sys.exit(1)

print("\n✓ Python SSL environment is properly configured")
EOF

    if [ $? -eq 0 ]; then
        test_pass "Python SSL certificate handling is configured correctly"
    else
        test_fail "Python SSL certificate configuration has issues"
    fi
else
    print_info "No custom SSL certificate configured (using system defaults)"
fi

# ============================================================================
# Summary
# ============================================================================
print_header "Test Summary"

echo -e "${CYAN}Total Tests:${NC} $TOTAL_TESTS"
echo -e "${GREEN}Passed:${NC} $PASSED_TESTS"
echo -e "${RED}Failed:${NC} $FAILED_TESTS"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   ✓ All critical tests passed!            ║${NC}"
    echo -e "${GREEN}║   Your configuration is ready to use.     ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
    echo ""

    if [ $WARNINGS -gt 0 ]; then
        print_warning "There are $WARNINGS warning(s) - review them above"
    fi

    print_info "Start the server with:"
    echo "  uv run src/graphiti_mcp_server.py --transport stdio"
    echo ""
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║   ✗ Configuration has issues               ║${NC}"
    echo -e "${RED}║   Fix the failed tests before continuing.  ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════╝${NC}"
    echo ""
    print_info "Review the failed tests above and fix the issues"
    echo ""
    exit 1
fi
