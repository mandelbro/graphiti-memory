#!/usr/bin/env python3
"""
Test script to verify if LLM Gateway supports OpenAI-compatible API.

This script tests both:
1. The documented native LLM Gateway API format
2. The suspected OpenAI-compatible proxy endpoint
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import httpx
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_ssl_verify_setting() -> bool | str:
    """
    Get the appropriate SSL verification setting.

    Returns:
        - Path to certificate bundle if SSL_CERT_FILE is set
        - False if no certificate is configured (development/testing only)
    """
    ssl_cert_file = os.getenv("SSL_CERT_FILE")
    if ssl_cert_file:
        cert_path = os.path.expanduser(ssl_cert_file)
        if os.path.exists(cert_path):
            logger.info(f"Using SSL certificate bundle: {cert_path}")
            return cert_path
        else:
            logger.warning(f"SSL_CERT_FILE set but file not found: {cert_path}")

    # Fall back to disabling verification (development only)
    logger.warning("SSL verification disabled - set SSL_CERT_FILE environment variable")
    return False


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.END}\n")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def load_config() -> dict:
    """Load configuration from openai.local.yml."""
    config_path = Path("config/providers/openai.local.yml")

    if not config_path.exists():
        print_error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print_success(f"Loaded configuration from {config_path}")
    return config


async def test_openai_compatible_endpoint(config: dict, api_key: str):
    """Test the OpenAI-compatible endpoint format."""
    print_section("Testing OpenAI-Compatible Endpoint")

    # Extract base URL from config
    base_url = config.get("llm", {}).get("base_url")
    if not base_url:
        print_error("No base_url found in configuration")
        return False

    # Construct OpenAI-compatible endpoint
    # The OpenAI SDK typically uses /v1/chat/completions
    if base_url.endswith("/v1"):
        chat_url = f"{base_url}/chat/completions"
    elif base_url.endswith("/"):
        chat_url = f"{base_url}v1/chat/completions"
    else:
        chat_url = f"{base_url}/v1/chat/completions"

    print(f"Testing endpoint: {chat_url}")

    # OpenAI-compatible request format
    request_data = {
        "model": config.get("llm", {}).get("model", "gpt-3.5-turbo"),
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Say 'OpenAI compatibility confirmed' and nothing else.",
            },
        ],
        "max_tokens": 50,
        "temperature": 0.0,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        ssl_verify = get_ssl_verify_setting()
        async with httpx.AsyncClient(verify=ssl_verify, timeout=30.0) as client:
            response = await client.post(chat_url, json=request_data, headers=headers)

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print_success("OpenAI-compatible endpoint working!")

                # Check response structure
                if "choices" in result:
                    print_success("Response has OpenAI 'choices' structure")
                    content = result["choices"][0]["message"]["content"]
                    print(f"Response: {content}")
                    return True
                else:
                    print_warning("Response doesn't match OpenAI structure")
                    print(f"Response: {result}")
                    return False
            else:
                print_error(f"Request failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False

    except Exception as e:
        print_error(f"Error testing OpenAI-compatible endpoint: {e}")
        return False


async def test_native_gateway_endpoint(config: dict, api_key: str):
    """Test the native LLM Gateway endpoint format."""
    print_section("Testing Native LLM Gateway Endpoint")

    base_url = config.get("llm", {}).get("base_url")
    if not base_url:
        print_error("No base_url found in configuration")
        return False

    # Try to construct native gateway URL
    # Based on OpenAPI spec, it should be /v1.0/chat/generations
    native_url = base_url.replace("/api/v1", "/v1.0/chat/generations")

    print(f"Testing endpoint: {native_url}")

    # Native LLM Gateway request format (from OpenAPI spec)
    request_data = {
        "model": "llmgateway__OpenAIGPT35Turbo",
        "messages": [
            {
                "role": "user",
                "content": "Say 'Native gateway format confirmed' and nothing else.",
            }
        ],
        "generation_settings": {"max_tokens": 50, "temperature": 0.0},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "x-client-feature-id": "GraphitiTest",
        "x-sfdc-app-context": "GraphitiMemory",
        "x-sfdc-core-tenant-id": "test-tenant",
    }

    try:
        ssl_verify = get_ssl_verify_setting()
        async with httpx.AsyncClient(verify=ssl_verify, timeout=30.0) as client:
            response = await client.post(native_url, json=request_data, headers=headers)

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print_success("Native LLM Gateway endpoint working!")

                # Check response structure
                if "generations" in result:
                    print_success("Response has native 'generations' structure")
                    content = result["generations"][0]["text"]
                    print(f"Response: {content}")
                    return True
                else:
                    print_warning("Response doesn't match native structure")
                    print(f"Response: {result}")
                    return False
            else:
                print_error(f"Request failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False

    except Exception as e:
        print_error(f"Error testing native gateway endpoint: {e}")
        return False


async def main():
    """Main test function."""
    print_section("LLM Gateway OpenAI Compatibility Test")

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    print_success("API key found in environment")

    # Load configuration
    config = load_config()
    print(f"Base URL: {config.get('llm', {}).get('base_url')}")
    print(f"Model: {config.get('llm', {}).get('model')}")

    # Test both endpoint formats
    openai_compatible = await test_openai_compatible_endpoint(config, api_key)
    native_gateway = await test_native_gateway_endpoint(config, api_key)

    # Summary
    print_section("Test Results Summary")

    if openai_compatible:
        print_success("✓ OpenAI-compatible endpoint: WORKING")
        print("  → You can use this gateway as a drop-in OpenAI replacement")
    else:
        print_error("✗ OpenAI-compatible endpoint: NOT WORKING")

    if native_gateway:
        print_success("✓ Native LLM Gateway endpoint: WORKING")
        print("  → You need to use the native format with special headers")
    else:
        print_error("✗ Native LLM Gateway endpoint: NOT WORKING")

    # Recommendations
    print_section("Recommendations")

    if openai_compatible:
        print("✓ Your configuration is correct for OpenAI compatibility")
        print("✓ Continue using the current setup with Graphiti")
        print(f"✓ Use base_url: {config.get('llm', {}).get('base_url')}")
    elif native_gateway:
        print("⚠ Only native format works - OpenAI SDK may not be compatible")
        print("⚠ You may need a translation layer or custom client")
    else:
        print("✗ Neither endpoint format worked")
        print("✗ Check authentication, headers, or network connectivity")


if __name__ == "__main__":
    asyncio.run(main())
