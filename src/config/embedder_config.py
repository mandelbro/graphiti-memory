"""
Embedder configuration for Graphiti MCP Server.

This module contains the GraphitiEmbedderConfig class that handles all
embedding-related configuration parameters for OpenAI, Azure OpenAI, and Ollama.
"""

import argparse
import logging
import os

from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from src.config_loader import config_loader
from src.utils import create_azure_credential_token_provider
from src.utils.ssl_utils import create_ssl_context_httpx_client, get_ssl_verify_setting

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDER_MODEL = "nomic-embed-text"


class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the embedder client.

    Centralizes all embedding-related configuration parameters.
    """

    model: str = DEFAULT_EMBEDDER_MODEL
    api_key: str | None = None
    dimension: int | None = None  # Embedding dimension for OpenAI/Azure
    base_url: str | None = None  # Base URL for OpenAI-compatible endpoints
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False
    # Ollama configuration
    use_ollama: bool = True  # Default to Ollama
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_embedding_model: str = DEFAULT_EMBEDDER_MODEL
    ollama_embedding_dim: int = 768

    @classmethod
    def from_yaml_and_env(cls) -> "GraphitiEmbedderConfig":
        """Create embedder configuration from provider YAML and environment variables."""
        # Check USE_OLLAMA environment variable first for provider detection
        use_ollama_env = os.environ.get("USE_OLLAMA", "").lower() == "true"

        # Try to load unified config first
        try:
            yaml_config = config_loader.load_unified_config()
            if not yaml_config:
                # Fall back to provider-specific config
                # Check for USE_OLLAMA first
                if use_ollama_env:
                    yaml_config = config_loader.load_provider_config("ollama")
                else:
                    # Check for Azure OpenAI (needs explicit env var)
                    azure_openai_endpoint = os.environ.get(
                        "AZURE_OPENAI_EMBEDDING_ENDPOINT", None
                    ) or os.environ.get("AZURE_OPENAI_ENDPOINT", None)
                    if azure_openai_endpoint is not None:
                        yaml_config = config_loader.load_provider_config("azure_openai")
                    else:
                        # Default to openai config (works for OpenAI-compatible APIs)
                        yaml_config = config_loader.load_provider_config("openai")

            embed_config = yaml_config.get("embedder", {})
        except Exception as e:
            logger.warning(f"Failed to load embedder YAML configuration: {e}")
            embed_config = {}

        # Get base_url to detect provider type
        base_url = embed_config.get("base_url", "")

        # Detect if this is Ollama based on USE_OLLAMA env var or base_url pattern
        # Ollama URLs contain localhost:11434 or have 'ollama' in the hostname
        is_ollama = (
            use_ollama_env
            or "localhost:11434" in base_url
            or "127.0.0.1:11434" in base_url
            or ("ollama" in base_url.lower() and "localhost" in base_url.lower())
        )

        if is_ollama:
            # Ollama configuration
            ollama_base_url = config_loader.get_env_value(
                "OLLAMA_BASE_URL",
                embed_config.get("base_url", "http://localhost:11434/v1"),
            )
            ollama_embedding_model = config_loader.get_env_value(
                "OLLAMA_EMBEDDING_MODEL",
                embed_config.get("model", DEFAULT_EMBEDDER_MODEL),
            )
            ollama_embedding_dim = config_loader.get_env_value(
                "OLLAMA_EMBEDDING_DIM", embed_config.get("dimension", 768), int
            )

            logger.info(
                f"Using Ollama embedder: {ollama_embedding_model} at {ollama_base_url}"
            )

            return cls(
                model=ollama_embedding_model,
                api_key="abc",  # Ollama doesn't require a real API key
                use_ollama=True,
                ollama_base_url=ollama_base_url,
                ollama_embedding_model=ollama_embedding_model,
                ollama_embedding_dim=ollama_embedding_dim,
            )

        # OpenAI or Azure OpenAI
        model = embed_config.get("model", "text-embedding-3-small")
        dimension = embed_config.get("dimension")

        azure_openai_endpoint = os.environ.get(
            "AZURE_OPENAI_EMBEDDING_ENDPOINT", None
        ) or os.environ.get("AZURE_OPENAI_ENDPOINT", None)

        # Backward compatibility: allow EMBEDDER_MODEL_NAME to override model for
        # OpenAI and Azure OpenAI providers (does not affect Ollama)
        env_model_override = os.environ.get("EMBEDDER_MODEL_NAME")
        if env_model_override:
            model = env_model_override

        if azure_openai_endpoint is not None:
            # Azure OpenAI setup
            azure_openai_api_version = os.environ.get(
                "AZURE_OPENAI_EMBEDDING_API_VERSION", None
            ) or os.environ.get("AZURE_OPENAI_API_VERSION", None)
            azure_openai_deployment_name = os.environ.get(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", None
            ) or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", None)
            azure_openai_use_managed_identity = (
                os.environ.get("AZURE_OPENAI_USE_MANAGED_IDENTITY", "false").lower()
                == "true"
            )

            if azure_openai_deployment_name is None:
                logger.error(
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set"
                )
                raise ValueError(
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set"
                )

            api_key = (
                None
                if azure_openai_use_managed_identity
                else os.environ.get("AZURE_OPENAI_EMBEDDING_API_KEY", None)
                or os.environ.get("OPENAI_API_KEY", None)
            )

            logger.info(
                f"Using Azure OpenAI embedder: {model} at {azure_openai_endpoint}"
            )

            return cls(
                model=model,
                api_key=api_key,
                dimension=dimension,
                base_url=base_url,
                azure_openai_endpoint=azure_openai_endpoint,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                use_ollama=False,
            )

        # OpenAI setup
        if base_url:
            logger.info(f"Using OpenAI-compatible embedder: {model} at {base_url}")
        else:
            logger.info(f"Using OpenAI embedder: {model}")

        return cls(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            dimension=dimension,
            base_url=base_url,
            use_ollama=False,
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> "GraphitiEmbedderConfig":
        """Create embedder configuration from CLI arguments, falling back to YAML and environment variables."""
        # Start with YAML+env based config
        config = cls.from_yaml_and_env()

        # CLI arguments override environment variables when provided
        if hasattr(args, "use_ollama") and args.use_ollama is not None:
            config.use_ollama = args.use_ollama

        if hasattr(args, "ollama_base_url") and args.ollama_base_url:
            config.ollama_base_url = args.ollama_base_url

        if hasattr(args, "ollama_embedding_model") and args.ollama_embedding_model:
            config.ollama_embedding_model = args.ollama_embedding_model
            if config.use_ollama:
                config.model = args.ollama_embedding_model

        if hasattr(args, "ollama_embedding_dim") and args.ollama_embedding_dim:
            config.ollama_embedding_dim = args.ollama_embedding_dim

        return config

    def create_client(self) -> EmbedderClient | None:
        if self.use_ollama:
            # Ollama setup
            embedder_config = OpenAIEmbedderConfig(
                api_key="abc",  # Ollama doesn't require a real API key
                embedding_model=self.ollama_embedding_model,
                embedding_dim=self.ollama_embedding_dim,
                base_url=self.ollama_base_url,
            )
            return OpenAIEmbedder(config=embedder_config)

        if self.azure_openai_endpoint is not None:
            # Azure OpenAI API setup
            if self.azure_openai_use_managed_identity:
                # Use managed identity for authentication
                token_provider = create_azure_credential_token_provider()

                # Create httpx client with SSL support
                http_client = create_ssl_context_httpx_client()

                return AzureOpenAIEmbedderClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        azure_ad_token_provider=token_provider,
                        http_client=http_client,
                    ),
                    model=self.model,
                )
            elif self.api_key:
                # Use API key for authentication

                # Create httpx client with SSL support
                http_client = create_ssl_context_httpx_client()

                return AzureOpenAIEmbedderClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        api_key=self.api_key,
                        http_client=http_client,
                    ),
                    model=self.model,
                )
            else:
                logger.error("OPENAI_API_KEY must be set when using Azure OpenAI API")
                return None
        else:
            # OpenAI API setup
            if not self.api_key:
                return None

            # Use base_url and dimension from config (already loaded from YAML)
            config_params = {
                "api_key": self.api_key,
                "embedding_model": self.model,
                "base_url": self.base_url,
            }
            # Only include embedding_dim if it's specified
            if self.dimension is not None:
                config_params["embedding_dim"] = self.dimension

            embedder_config = OpenAIEmbedderConfig(**config_params)

            embedder = OpenAIEmbedder(config=embedder_config)

            # For OpenAI-compatible endpoints with custom SSL requirements,
            # patch the underlying httpx client to use SSL certificates
            ssl_verify = get_ssl_verify_setting()
            if isinstance(ssl_verify, str) or ssl_verify is False:
                # Custom certificate or SSL disabled - need to configure httpx client
                import httpx

                if hasattr(embedder, "client") and hasattr(embedder.client, "_client"):
                    # Patch the internal httpx client with SSL configuration
                    embedder.client._client = httpx.AsyncClient(
                        verify=ssl_verify, timeout=30.0
                    )
                    logger.info(
                        "Configured OpenAI embedder with custom SSL certificate"
                    )

            return embedder
