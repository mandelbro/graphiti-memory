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
from src.utils.ssl_utils import create_ssl_context_httpx_client, get_ssl_verify_setting

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
        # Try to load unified config first
        # Check USE_OLLAMA environment variable first for provider detection
        use_ollama_env = os.environ.get("USE_OLLAMA", "").lower() == "true"

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
                        "AZURE_OPENAI_ENDPOINT", None
                    )
                    if azure_openai_endpoint is not None:
                        yaml_config = config_loader.load_provider_config("azure_openai")
                    else:
                        # Default to openai config (works for OpenAI-compatible APIs)
                        yaml_config = config_loader.load_provider_config("openai")

            llm_config = yaml_config.get("llm", {})
        except Exception as e:
            logger.warning(f"Failed to load LLM YAML configuration: {e}")
            llm_config = {}

        # Get base_url to detect provider type
        base_url = llm_config.get("base_url", "")

        # Check explicit use_ollama setting in YAML (has highest priority)
        yaml_use_ollama = llm_config.get("use_ollama")

        # Detect if this is Ollama based on explicit setting, env var, or base_url pattern
        # Priority: 1) Explicit YAML use_ollama, 2) USE_OLLAMA env var, 3) base_url pattern
        if yaml_use_ollama is not None:
            # Explicit YAML setting takes precedence
            is_ollama = yaml_use_ollama
        else:
            # Fall back to env var and URL pattern detection
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
            ollama_model_parameters = llm_config.get("model_parameters", {})

            logger.info(f"Using Ollama LLM: {ollama_llm_model} at {ollama_base_url}")

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
            # OpenAI or Azure OpenAI configuration
            model = llm_config.get("model", DEFAULT_LLM_MODEL)
            small_model = llm_config.get("small_model", SMALL_LLM_MODEL)
            temperature = llm_config.get("temperature", 0.0)
            max_tokens = llm_config.get("max_tokens", 8192)

            azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)

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

                logger.info(
                    f"Using Azure OpenAI LLM: {model} at {azure_openai_endpoint}"
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
                openai_base_url = llm_config.get("base_url")
                if openai_base_url:
                    logger.info(
                        f"Using OpenAI-compatible LLM: {model} at {openai_base_url}"
                    )
                else:
                    logger.info(f"Using OpenAI LLM: {model}")

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
        """Create LLM configuration from environment variables only.

        Detection logic (in order):
        1. If OLLAMA_BASE_URL or OLLAMA_LLM_MODEL is set -> Use Ollama
        2. If AZURE_OPENAI_ENDPOINT is set -> Use Azure OpenAI
        3. Otherwise -> Use OpenAI (or OpenAI-compatible API)
        """
        # Check if Ollama-specific env vars are set
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
        ollama_llm_model = os.environ.get("OLLAMA_LLM_MODEL")

        if ollama_base_url or ollama_llm_model:
            # Ollama configuration
            ollama_base_url = ollama_base_url or "http://localhost:11434/v1"
            ollama_llm_model = ollama_llm_model or DEFAULT_LLM_MODEL

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
                openai_enable_temperature = (
                    os.environ.get("OPENAI_ENABLE_TEMPERATURE", "false").lower()
                    == "true"
                )

                config = LLMConfig(
                    api_key=self.api_key,
                    model=self.model,
                    small_model=self.small_model,
                )
                # Always set max_tokens
                config.max_tokens = self.max_tokens
                # Only include temperature when explicitly enabled
                if openai_enable_temperature:
                    config.temperature = self.temperature

                # Create httpx client with SSL support
                http_client = create_ssl_context_httpx_client()

                return AzureOpenAILLMClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        azure_ad_token_provider=token_provider,
                        http_client=http_client,
                    ),
                    config=config,
                )
            elif self.api_key:
                # Use API key for authentication
                openai_enable_temperature = (
                    os.environ.get("OPENAI_ENABLE_TEMPERATURE", "false").lower()
                    == "true"
                )

                config = LLMConfig(
                    api_key=self.api_key,
                    model=self.model,
                    small_model=self.small_model,
                )
                # Always set max_tokens
                config.max_tokens = self.max_tokens
                # Only include temperature when explicitly enabled
                if openai_enable_temperature:
                    config.temperature = self.temperature

                # Create httpx client with SSL support
                http_client = create_ssl_context_httpx_client()

                return AzureOpenAILLMClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        api_key=self.api_key,
                        http_client=http_client,
                    ),
                    config=config,
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

        # Always set max_tokens
        llm_client_config.max_tokens = self.max_tokens

        # Only include temperature for OpenAI when explicitly enabled
        openai_enable_temperature = (
            os.environ.get("OPENAI_ENABLE_TEMPERATURE", "false").lower() == "true"
        )
        if openai_enable_temperature:
            llm_client_config.temperature = self.temperature

        # Note: For OpenAI-compatible endpoints requiring custom SSL certificates
        # (e.g., internal corporate gateways), ensure SSL_CERT_FILE environment
        # variable is set. The OpenAI SDK will respect this for HTTPS connections.
        # For Azure OpenAI, SSL support is configured via http_client parameter above.

        client = OpenAIClient(config=llm_client_config)

        # For OpenAI-compatible endpoints with custom SSL requirements,
        # patch the underlying httpx client to use SSL certificates
        ssl_verify = get_ssl_verify_setting()
        if isinstance(ssl_verify, str) or ssl_verify is False:
            # Custom certificate or SSL disabled - need to configure httpx client
            import httpx

            if hasattr(client, "client") and hasattr(client.client, "_client"):
                # Patch the internal httpx client with SSL configuration
                client.client._client = httpx.AsyncClient(
                    verify=ssl_verify, timeout=30.0
                )
                logger.info("Configured OpenAI client with custom SSL certificate")

        return client
