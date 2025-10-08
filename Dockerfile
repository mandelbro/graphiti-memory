# syntax=docker/dockerfile:1.9
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the installer script to a system location
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    chmod +x /usr/local/bin/uv

# Add to PATH (though /usr/local/bin should already be in PATH)
ENV PATH="/usr/local/bin:${PATH}"

# Configure uv for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MCP_SERVER_HOST="0.0.0.0" \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN groupadd -r app && useradd -r -d /app -g app app

# Copy project files for dependency installation (better caching)
COPY pyproject.toml uv.lock README.md ./

# Install custom CA certificate into system trust store if provided
# This persists the certificate in the image for both build and runtime
RUN --mount=type=secret,id=ssl_cert,target=/tmp/ca-cert.pem,required=false \
    sh -c ' \
        if [ -f /tmp/ca-cert.pem ] && [ -s /tmp/ca-cert.pem ]; then \
            echo "Installing custom CA certificate into system trust store"; \
            cp /tmp/ca-cert.pem /usr/local/share/ca-certificates/custom-ca.crt; \
            update-ca-certificates; \
            echo "Certificate installed and will persist in container"; \
        else \
            echo "No custom CA certificate provided - using system defaults"; \
        fi \
    '

# Install dependencies including dev dependencies (after certificate is installed, if provided)
# We install dev dependencies here so they're available at runtime without re-downloading
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Change ownership to app user
RUN chown -Rv app:app /app

# Switch to non-root user
USER app

# Expose ports
# Note: These are the default ports. Actual ports can be configured via:
# - MCP_SERVER_PORT environment variable (default: 8020)
# - MCP_INTERNAL_PORT environment variable (default: 8021)
# The actual port mapping is handled by docker-compose.yml
EXPOSE 8020 8021

# Command to run the application
CMD ["uv", "run", "python", "src/graphiti_mcp_server.py"]
