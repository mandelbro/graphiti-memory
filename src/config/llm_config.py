"""LLM configuration for the Graphiti MCP server."""

import argparse
import logging
import os
from typing import Any

from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field

from src.config_loader import config_loader
from src.ollama_client import OllamaClient
from src.utils.auth_utils import create_azure_credential_token_provider

# Get logger for this module
logger = logging.getLogger(__name__)

# Constants for default models
DEFAULT_LLM_MODEL = "deepseek-r1:7b"
SMALL_LLM_MODEL = "deepseek-r1:7b"


class GraphitiLLMConfig(BaseModel):
    """Configuration for the LLM client.

    Centralizes all LLM-specific configuration parameters including API keys and model selection.
    """

    api_key: str | None = None
    model: str = DEFAULT_LLM_MODEL
    small_model: str = SMALL_LLM_MODEL
    temperature: float = 0.0
    max_tokens: int = int(os.environ.get("LLM_MAX_TOKENS", "8192"))
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False
    # Ollama configuration
    use_ollama: bool = True  # Default to Ollama
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_llm_model: str = DEFAULT_LLM_MODEL
    ollama_model_parameters: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml_and_env(cls) -> "GraphitiLLMConfig":
        """Create LLM configuration from YAML files and environment variables."""
        # Check if Ollama should be used (default to True)
        use_ollama = (
            config_loader.get_env_value("USE_OLLAMA", "true", str).lower() == "true"
        )

        if use_ollama:
            # Load Ollama YAML configuration (with local overrides)
            try:
                yaml_config = config_loader.load_provider_config("ollama")
                llm_config = yaml_config.get("llm", {})
            except Exception as e:
                logger.warning(f"Failed to load Ollama YAML configuration: {e}")
                llm_config = {}

            # Use YAML config values with fallback to defaults, then override with environment variables
            ollama_base_url = config_loader.get_env_value(
                "OLLAMA_BASE_URL",
                llm_config.get("base_url", "http://localhost:11434/v1"),
            )
            ollama_llm_model = config_loader.get_env_value(
                "OLLAMA_LLM_MODEL", llm_config.get("model", DEFAULT_LLM_MODEL)
            )
            temperature = config_loader.get_env_value(
                "LLM_TEMPERATURE", llm_config.get("temperature", 0.0), float
            )
            max_tokens = config_loader.get_env_value(
                "LLM_MAX_TOKENS", llm_config.get("max_tokens", 8192), int
            )

            # Get Ollama model parameters from YAML
            ollama_model_parameters = llm_config.get("model_parameters", {})

            return cls(
                api_key="abc",  # Ollama doesn't require a real API key
                model=ollama_llm_model,
                small_model=ollama_llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                use_ollama=True,
                ollama_base_url=ollama_base_url,
                ollama_llm_model=ollama_llm_model,
                ollama_model_parameters=ollama_model_parameters,
            )
        else:
            # Load OpenAI or Azure OpenAI configuration
            # Try Azure OpenAI first, then OpenAI
            azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)

            try:
                if azure_openai_endpoint is not None:
                    # Azure OpenAI configuration (with local overrides)
                    yaml_config = config_loader.load_provider_config("azure_openai")
                else:
                    # OpenAI configuration (with local overrides)
                    yaml_config = config_loader.load_provider_config("openai")

                llm_config = yaml_config.get("llm", {})
            except Exception as e:
                logger.warning(
                    f"Failed to load OpenAI/Azure OpenAI YAML configuration: {e}"
                )
                llm_config = {}

            # Use YAML config values with fallback to defaults
            model = llm_config.get("model", DEFAULT_LLM_MODEL)
            small_model = llm_config.get("small_model", SMALL_LLM_MODEL)
            temperature = llm_config.get("temperature", 0.0)
            max_tokens = llm_config.get("max_tokens", 8192)

            if azure_openai_endpoint is not None:
                # Azure OpenAI setup - still use environment variables for sensitive config
                azure_openai_api_version = os.environ.get(
                    "AZURE_OPENAI_API_VERSION", None
                )
                azure_openai_deployment_name = os.environ.get(
                    "AZURE_OPENAI_DEPLOYMENT_NAME", None
                )
                azure_openai_use_managed_identity = (
                    os.environ.get("AZURE_OPENAI_USE_MANAGED_IDENTITY", "false").lower()
                    == "true"
                )

                if azure_openai_deployment_name is None:
                    logger.error(
                        "AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set"
                    )
                    raise ValueError(
                        "AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set"
                    )

                api_key = (
                    None
                    if azure_openai_use_managed_identity
                    else os.environ.get("OPENAI_API_KEY", None)
                )

                return cls(
                    azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                    azure_openai_endpoint=azure_openai_endpoint,
                    api_key=api_key,
                    azure_openai_api_version=azure_openai_api_version,
                    azure_openai_deployment_name=azure_openai_deployment_name,
                    model=model,
                    small_model=small_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_ollama=False,
                )
            else:
                # OpenAI setup - still use environment variables for API key
                return cls(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    model=model,
                    small_model=small_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_ollama=False,
                )

    @classmethod
    def from_env(cls) -> "GraphitiLLMConfig":
        """Create LLM configuration from environment variables."""
        # Check if Ollama should be used (default to True)
        use_ollama = os.environ.get("USE_OLLAMA", "true").lower() == "true"

        if use_ollama:
            # Ollama configuration
            ollama_base_url = os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434/v1"
            )
            ollama_llm_model = os.environ.get("OLLAMA_LLM_MODEL", DEFAULT_LLM_MODEL)

            return cls(
                api_key="abc",  # Ollama doesn't require a real API key
                model=ollama_llm_model,
                small_model=ollama_llm_model,
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "8192")),
                use_ollama=True,
                ollama_base_url=ollama_base_url,
                ollama_llm_model=ollama_llm_model,
            )

        # OpenAI/Azure OpenAI configuration (existing logic)
        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get("MODEL_NAME", "")
        model = model_env if model_env.strip() else DEFAULT_LLM_MODEL

        # Get small_model from environment, or use default if not set or empty
        small_model_env = os.environ.get("SMALL_MODEL_NAME", "")
        small_model = small_model_env if small_model_env.strip() else SMALL_LLM_MODEL

        azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)
        azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", None)
        azure_openai_deployment_name = os.environ.get(
            "AZURE_OPENAI_DEPLOYMENT_NAME", None
        )
        azure_openai_use_managed_identity = (
            os.environ.get("AZURE_OPENAI_USE_MANAGED_IDENTITY", "false").lower()
            == "true"
        )

        if azure_openai_endpoint is None:
            # Setup for OpenAI API
            # Log if empty model was provided
            if model_env == "":
                logger.debug(
                    f"MODEL_NAME environment variable not set, using default: {DEFAULT_LLM_MODEL}"
                )
            elif not model_env.strip():
                logger.warning(
                    f"Empty MODEL_NAME environment variable, using default: {DEFAULT_LLM_MODEL}"
                )

            return cls(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=model,
                small_model=small_model,
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "8192")),
                use_ollama=False,
            )
        else:
            # Setup for Azure OpenAI API
            # Log if empty deployment name was provided
            if azure_openai_deployment_name is None:
                logger.error(
                    "AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set"
                )

                raise ValueError(
                    "AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set"
                )
            if not azure_openai_use_managed_identity:
                # api key
                api_key = os.environ.get("OPENAI_API_KEY", None)
            else:
                # Managed identity
                api_key = None

            return cls(
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                azure_openai_endpoint=azure_openai_endpoint,
                api_key=api_key,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
                model=model,
                small_model=small_model,
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "8192")),
                use_ollama=False,
            )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> "GraphitiLLMConfig":
        """Create LLM configuration from CLI arguments, falling back to YAML and environment variables."""
        # Start with YAML and environment-based config
        config = cls.from_yaml_and_env()

        # CLI arguments override environment variables when provided
        if hasattr(args, "use_ollama") and args.use_ollama is not None:
            config.use_ollama = args.use_ollama

        if hasattr(args, "ollama_base_url") and args.ollama_base_url:
            config.ollama_base_url = args.ollama_base_url

        if hasattr(args, "ollama_llm_model") and args.ollama_llm_model:
            config.ollama_llm_model = args.ollama_llm_model
            if config.use_ollama:
                config.model = args.ollama_llm_model
                config.small_model = args.ollama_llm_model

        if hasattr(args, "model") and args.model:
            # Only use CLI model if it's not empty and not using Ollama
            if args.model.strip() and not config.use_ollama:
                config.model = args.model
            elif args.model.strip() == "":
                # Log that empty model was provided and default is used
                logger.warning(
                    f"Empty model name provided, using default: {DEFAULT_LLM_MODEL}"
                )

        if hasattr(args, "small_model") and args.small_model:
            if args.small_model.strip() and not config.use_ollama:
                config.small_model = args.small_model
            elif args.small_model.strip() == "":
                logger.warning(
                    f"Empty small_model name provided, using default: {SMALL_LLM_MODEL}"
                )

        if hasattr(args, "temperature") and args.temperature is not None:
            config.temperature = args.temperature

        if hasattr(args, "max_tokens") and args.max_tokens is not None:
            config.max_tokens = args.max_tokens

        return config

    def create_client(self) -> LLMClient:
        """Create an LLM client based on this configuration.

        Returns:
            LLMClient instance
        """

        if self.use_ollama:
            # Ollama setup
            llm_client_config = LLMConfig(
                api_key="abc",  # Ollama doesn't require a real API key
                model=self.ollama_llm_model,
                small_model=self.ollama_llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                base_url=self.ollama_base_url,
            )
            return OllamaClient(
                config=llm_client_config, model_parameters=self.ollama_model_parameters
            )

        if self.azure_openai_endpoint is not None:
            # Azure OpenAI API setup
            if self.azure_openai_use_managed_identity:
                # Use managed identity for authentication
                token_provider = create_azure_credential_token_provider()
                return AzureOpenAILLMClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        azure_ad_token_provider=token_provider,
                    ),
                    config=LLMConfig(
                        api_key=self.api_key,
                        model=self.model,
                        small_model=self.small_model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    ),
                )
            elif self.api_key:
                # Use API key for authentication
                return AzureOpenAILLMClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        api_key=self.api_key,
                    ),
                    config=LLMConfig(
                        api_key=self.api_key,
                        model=self.model,
                        small_model=self.small_model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    ),
                )
            else:
                raise ValueError(
                    "OPENAI_API_KEY must be set when using Azure OpenAI API"
                )

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set when using OpenAI API")

        llm_client_config = LLMConfig(
            api_key=self.api_key, model=self.model, small_model=self.small_model
        )

        # Set temperature and max_tokens
        llm_client_config.temperature = self.temperature
        llm_client_config.max_tokens = self.max_tokens

        return OpenAIClient(config=llm_client_config)
