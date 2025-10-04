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

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDER_MODEL = "nomic-embed-text"


class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the embedder client.

    Centralizes all embedding-related configuration parameters.
    """

    model: str = DEFAULT_EMBEDDER_MODEL
    api_key: str | None = None
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
        # Decide provider based on USE_OLLAMA
        use_ollama = (
            config_loader.get_env_value("USE_OLLAMA", "true", str).lower() == "true"
        )

        if use_ollama:
            # Load Ollama YAML (with local overrides) for embedder
            try:
                yaml_config = config_loader.load_provider_config("ollama")
                embed_config = yaml_config.get("embedder", {})
            except Exception as e:
                logger.warning(f"Failed to load Ollama embedder YAML: {e}")
                embed_config = {}

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

            return cls(
                model=ollama_embedding_model,
                api_key="abc",  # Ollama doesn't require a real API key
                use_ollama=True,
                ollama_base_url=ollama_base_url,
                ollama_embedding_model=ollama_embedding_model,
                ollama_embedding_dim=ollama_embedding_dim,
            )

        # OpenAI or Azure OpenAI
        azure_openai_endpoint = os.environ.get(
            "AZURE_OPENAI_EMBEDDING_ENDPOINT", None
        ) or os.environ.get("AZURE_OPENAI_ENDPOINT", None)

        try:
            if azure_openai_endpoint is not None:
                yaml_config = config_loader.load_provider_config("azure_openai")
            else:
                yaml_config = config_loader.load_provider_config("openai")
            embed_config = yaml_config.get("embedder", {})
        except Exception as e:
            logger.warning(f"Failed to load OpenAI/Azure embedder YAML: {e}")
            embed_config = {}

        model = embed_config.get("model", "text-embedding-3-small")

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

            return cls(
                model=model,
                api_key=api_key,
                azure_openai_endpoint=azure_openai_endpoint,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                use_ollama=False,
            )

        # OpenAI setup
        return cls(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY"),
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
                return AzureOpenAIEmbedderClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        azure_ad_token_provider=token_provider,
                    ),
                    model=self.model,
                )
            elif self.api_key:
                # Use API key for authentication
                return AzureOpenAIEmbedderClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        api_key=self.api_key,
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

            embedder_config = OpenAIEmbedderConfig(
                api_key=self.api_key, embedding_model=self.model
            )

            return OpenAIEmbedder(config=embedder_config)
