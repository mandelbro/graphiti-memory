"""SSL/TLS certificate utilities for secure connections to internal endpoints."""

import logging
import os
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


def get_ssl_verify_setting() -> bool | str:
    """
    Get the appropriate SSL verification setting for httpx clients.

    Checks multiple environment variables in order of precedence:
    1. SSL_CERT_FILE - Standard Python SSL certificate file
    2. REQUESTS_CA_BUNDLE - Requests library certificate bundle
    3. CURL_CA_BUNDLE - cURL certificate bundle

    Returns:
        - Path to certificate bundle (str) if found
        - True to use default system certificates
        - False if no certificate is configured (development/testing only)

    Environment Variables:
        SSL_CERT_FILE: Path to PEM certificate bundle
        REQUESTS_CA_BUNDLE: Path to certificate bundle (requests library)
        CURL_CA_BUNDLE: Path to certificate bundle (curl)
    """
    # Check SSL_CERT_FILE first (standard Python SSL)
    ssl_cert_file = os.getenv("SSL_CERT_FILE")
    if ssl_cert_file:
        cert_path = Path(ssl_cert_file).expanduser()
        if cert_path.exists():
            logger.info(f"Using SSL certificate bundle from SSL_CERT_FILE: {cert_path}")
            return str(cert_path)
        else:
            logger.warning(f"SSL_CERT_FILE set but file not found: {cert_path}")

    # Check REQUESTS_CA_BUNDLE (requests library)
    requests_ca_bundle = os.getenv("REQUESTS_CA_BUNDLE")
    if requests_ca_bundle:
        cert_path = Path(requests_ca_bundle).expanduser()
        if cert_path.exists():
            logger.info(
                f"Using SSL certificate bundle from REQUESTS_CA_BUNDLE: {cert_path}"
            )
            return str(cert_path)
        else:
            logger.warning(f"REQUESTS_CA_BUNDLE set but file not found: {cert_path}")

    # Check CURL_CA_BUNDLE (curl)
    curl_ca_bundle = os.getenv("CURL_CA_BUNDLE")
    if curl_ca_bundle:
        cert_path = Path(curl_ca_bundle).expanduser()
        if cert_path.exists():
            logger.info(
                f"Using SSL certificate bundle from CURL_CA_BUNDLE: {cert_path}"
            )
            return str(cert_path)
        else:
            logger.warning(f"CURL_CA_BUNDLE set but file not found: {cert_path}")

    # No certificate configured - use default system certificates
    logger.debug(
        "No custom SSL certificate configured, using system default certificates"
    )
    return True


def create_ssl_context_httpx_client(
    timeout: float = 30.0,
    max_connections: int = 100,
    max_keepalive_connections: int = 20,
) -> httpx.AsyncClient:
    """
    Create an httpx AsyncClient with proper SSL certificate configuration.

    This function creates a client that respects SSL_CERT_FILE and related
    environment variables for connecting to internal endpoints with custom CAs.

    Args:
        timeout: Request timeout in seconds (default: 30.0)
        max_connections: Maximum number of connections (default: 100)
        max_keepalive_connections: Maximum keepalive connections (default: 20)

    Returns:
        httpx.AsyncClient: Configured async HTTP client with SSL support

    Example:
        >>> client = create_ssl_context_httpx_client()
        >>> # Use with OpenAI SDK
        >>> from openai import AsyncOpenAI
        >>> openai_client = AsyncOpenAI(http_client=client)
    """
    ssl_verify = get_ssl_verify_setting()

    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
    )

    client = httpx.AsyncClient(
        verify=ssl_verify,
        timeout=timeout,
        limits=limits,
    )

    if isinstance(ssl_verify, str):
        logger.info(f"Created httpx client with custom SSL certificate: {ssl_verify}")
    elif ssl_verify is True:
        logger.debug("Created httpx client with default system SSL certificates")
    else:
        logger.warning(
            "Created httpx client with SSL verification DISABLED (development only)"
        )

    return client


def get_ssl_verify_for_openai() -> bool | str:
    """
    Get SSL verification setting specifically for OpenAI SDK clients.

    The OpenAI SDK accepts either:
    - str: path to certificate bundle
    - bool: True for default certs, False to disable verification

    Returns:
        SSL verification setting compatible with OpenAI SDK
    """
    ssl_verify = get_ssl_verify_setting()

    # OpenAI SDK only accepts str or bool, not SSLContext
    if isinstance(ssl_verify, (bool, str)):
        return ssl_verify

    # Default to True (use system certificates)
    return True
