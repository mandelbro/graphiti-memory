"""
Ollama Health Check and Model Validation Utilities.

This module provides health checking and model validation functionality
for Ollama servers, including intelligent caching to minimize API calls.
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)


class OllamaHealthValidator:
    """
    Utility class for Ollama server health checks and model validation.

    Provides cached health checking and model availability validation
    to ensure reliable communication with Ollama servers.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the health validator.

        Args:
            base_url: Base URL for the Ollama server
        """
        self.base_url = base_url

        # Connection pooling infrastructure
        self._http_client: httpx.AsyncClient | None = None

        # Health check caching infrastructure (5-minute TTL)
        self._health_check_cache: dict[str, tuple[tuple[bool, str], float]] = {}

        # Model validation caching infrastructure
        self._model_cache: dict[str, bool | str] = {}

    async def __aenter__(self) -> "OllamaHealthValidator":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - close HTTP client if it exists and is not closed."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def _get_http_client(self) -> httpx.AsyncClient:
        """
        Get or create a shared HTTP client with connection pooling.

        Creates a new httpx.AsyncClient if none exists or current one is closed.
        Configures connection limits and timeouts for optimal performance.

        Returns:
            httpx.AsyncClient: The shared HTTP client instance
        """
        if self._http_client is None or self._http_client.is_closed:
            # Create new HTTP client with connection pooling configuration
            limits = httpx.Limits(
                max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0
            )

            timeout = httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=2.0)

            self._http_client = httpx.AsyncClient(limits=limits, timeout=timeout)

        return self._http_client

    async def check_health(self) -> tuple[bool, str]:
        """
        Check Ollama server health with intelligent caching.

        Validates Ollama server availability by making a GET request to the /api/tags
        endpoint. Results are cached for 5 minutes to avoid health check spam.

        Returns:
            tuple[bool, str]: (is_healthy, message) where is_healthy indicates if the
                              server is accessible and message contains details
        """
        cache_key = f"health_{self.base_url}"
        current_time = time.time()

        # Check cache first (5-minute TTL)
        if cache_key in self._health_check_cache:
            cached_result, cached_time = self._health_check_cache[cache_key]
            if current_time - cached_time < 300:  # 5 minutes = 300 seconds
                return cached_result

        # Perform health check
        try:
            # Construct the health check URL
            base_url = self.base_url or "http://localhost:11434"
            native_url = base_url.replace("/v1", "").rstrip("/")
            health_url = f"{native_url}/api/tags"

            # Make the health check request
            client = await self._get_http_client()
            response = await client.get(health_url, timeout=1.0)
            response.raise_for_status()

            # Server is healthy
            result = (True, f"Ollama server at {base_url} is healthy and accessible")

        except httpx.ConnectError:
            # Connection failed - server not running or unreachable
            result = (
                False,
                f"Cannot connect to Ollama server at {self.base_url}. Is Ollama running?",
            )

        except httpx.TimeoutException:
            # Server not responding in time
            result = (
                False,
                f"Ollama server at {self.base_url} is not responding. Server may be overloaded.",
            )

        except Exception as e:
            # Generic error
            result = (False, f"Ollama health check failed: {e}")

        # Cache the result (both positive and negative)
        self._health_check_cache[cache_key] = (result, current_time)

        return result

    async def validate_model_available(self, model: str) -> tuple[bool, str]:
        """
        Validate if the requested model is available on the Ollama server.

        Checks the model cache first to avoid repeated API calls, then queries
        the /api/tags endpoint to fetch available models. Results are cached
        for future requests.

        Args:
            model: The model name to validate

        Returns:
            tuple[bool, str]: (is_available, message) where is_available indicates
                             if the model is available and message contains details
                             or error information
        """
        # Check cache first
        cache_key = f"{model}_available"
        error_cache_key = f"{model}_error_msg"

        if cache_key in self._model_cache:
            is_cached_available = self._model_cache[cache_key]
            if is_cached_available:
                return (True, "Model available")
            else:
                # Return cached error message if available
                cached_error_msg = str(
                    self._model_cache.get(error_cache_key, f"Model '{model}' not found")
                )
                return (False, cached_error_msg)

        try:
            # Construct the tags URL
            base_url = self.base_url or "http://localhost:11434"
            native_url = base_url.replace("/v1", "").rstrip("/")
            tags_url = f"{native_url}/api/tags"

            # Make the API request
            client = await self._get_http_client()
            response = await client.get(tags_url, timeout=5.0)
            response.raise_for_status()
            tags_data = response.json()

            # Extract available models
            models_data = tags_data.get("models")
            if models_data is None:
                # Missing 'models' field - graceful fallback
                logger.warning(
                    f"No 'models' field in Ollama tags response, allowing request for '{model}' to proceed"
                )
                return (
                    True,
                    "Model validation skipped due to error: Missing 'models' field in response",
                )

            available_models = [
                model_info.get("name", "") for model_info in models_data
            ]

            # Check if the requested model is available
            model_available = model in available_models

            # Cache the result (both positive and negative)
            self._model_cache[cache_key] = model_available

            if model_available:
                return (True, "Model available")
            else:
                # Prepare helpful error message with first 5 available models
                if available_models:
                    first_models = available_models[:5]
                    models_display = ", ".join(first_models)
                else:
                    models_display = "(none)"

                error_msg = (
                    f"Model '{model}' not found. Available models: {models_display}"
                )
                # Cache the error message too
                self._model_cache[error_cache_key] = error_msg
                return (False, error_msg)

        except Exception as e:
            # Log warning but don't cache failures (might be temporary)
            logger.warning(f"Model validation failed for '{model}': {e}")

            # Graceful fallback - allow the request to proceed
            # Don't block on validation failures
            return (True, f"Model validation skipped due to error: {e}")
