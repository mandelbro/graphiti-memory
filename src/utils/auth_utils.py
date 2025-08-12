"""Authentication utilities for Graphiti MCP Server.

This module contains authentication-related helper functions,
particularly for Azure credential management.
"""

from collections.abc import Callable

from azure.identity import DefaultAzureCredential, get_bearer_token_provider


def create_azure_credential_token_provider() -> Callable[[], str]:
    """Create an Azure credential token provider for managed identity authentication.

    Returns:
        A callable that returns authentication tokens for Azure Cognitive Services.
    """
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return token_provider
