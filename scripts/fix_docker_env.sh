#!/bin/bash
# Script to configure Docker-specific SSL environment variables in .env file
# This is useful when your host machine uses custom CA certificates that need
# to be available inside Docker containers.

set -e

ENV_FILE="${1:-.env}"

echo "Configuring Docker SSL environment variables..."
echo ""

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found"
    echo "Please create a .env file first or copy from .env.example"
    exit 1
fi

# Check if Docker SSL variables already exist
if grep -q "SSL_CERT_FILE_DOCKER" "$ENV_FILE"; then
    echo "✓ Docker SSL variables already configured in $ENV_FILE"
    exit 0
fi

# Prompt for custom certificate setup
echo "This script will add Docker SSL certificate configuration to your .env file."
echo ""
echo "Do you use custom CA certificates that need to be mounted in Docker? (y/n)"
read -r USE_CUSTOM_CERTS

if [[ "$USE_CUSTOM_CERTS" != "y" && "$USE_CUSTOM_CERTS" != "Y" ]]; then
    echo "Skipping SSL certificate configuration."
    exit 0
fi

echo ""
echo "Enter the path to your CA certificate directory on the host:"
echo "Example: \${HOME}/.certs or /etc/ssl/certs"
read -r CERT_DIR

echo ""
echo "Enter the filename of your CA bundle inside that directory:"
echo "Example: ca-bundle.pem or ca-certificates.crt"
read -r CERT_FILENAME

# Add Docker SSL variables
echo "" >> "$ENV_FILE"
echo "# SSL Certificate Configuration for Docker Container" >> "$ENV_FILE"
echo "# Directory containing custom CA certificates (host path)" >> "$ENV_FILE"
echo "SSL_CERT_DIR=$CERT_DIR" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"
echo "# CA certificate filename" >> "$ENV_FILE"
echo "SSL_CERT_FILENAME=$CERT_FILENAME" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"
echo "# Docker container paths (automatically derived from above)" >> "$ENV_FILE"
echo "SSL_CERT_FILE_DOCKER=/app/.certs/$CERT_FILENAME" >> "$ENV_FILE"
echo "REQUESTS_CA_BUNDLE_DOCKER=/app/.certs/$CERT_FILENAME" >> "$ENV_FILE"

echo ""
echo "✓ Successfully configured Docker SSL variables in $ENV_FILE"
echo ""

# Uncomment volumes section in docker-compose.yml
if grep -q "^    # volumes:" docker-compose.yml; then
    echo "Enabling SSL certificate volume mount in docker-compose.yml..."
    # Create backup
    cp docker-compose.yml docker-compose.yml.bak
    # Uncomment the volumes section for graphiti-mcp service
    sed -i.tmp '/graphiti-mcp:/,/^  [a-z]/ {
        s/^    # volumes:/    volumes:/
        s/^    #   - \${SSL_CERT_DIR}:/      - ${SSL_CERT_DIR}:/
    }' docker-compose.yml && rm docker-compose.yml.tmp
    echo "✓ Enabled volume mount in docker-compose.yml"
    echo "  (Backup saved as docker-compose.yml.bak)"
else
    echo "⚠ Volume mount may already be enabled in docker-compose.yml"
fi

echo ""
echo "Next steps:"
echo "1. Verify the configuration in $ENV_FILE"
echo "2. Verify docker-compose.yml has volumes section uncommented"
echo "3. Run: just rebuild"
