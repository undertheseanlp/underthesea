import os

DEFAULT_MODEL = "gpt-4o-mini"


class LLM:
    """
    LLM client supporting OpenAI and Azure OpenAI.

    Environment variables:
    - OpenAI: OPENAI_API_KEY, OPENAI_MODEL (optional)
    - Azure OpenAI: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
                    AZURE_OPENAI_DEPLOYMENT (optional), AZURE_OPENAI_API_VERSION (optional)
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
    ):
        """
        Initialize LLM client.

        Parameters
        ----------
        provider : str, optional
            Provider name: "openai" or "azure". Auto-detected from env vars if not specified.
        model : str, optional
            Model/deployment name. Falls back to OPENAI_MODEL, AZURE_OPENAI_DEPLOYMENT,
            or "gpt-4o-mini".
        api_key : str, optional
            API key. Falls back to OPENAI_API_KEY or AZURE_OPENAI_API_KEY.
        azure_endpoint : str, optional
            Azure OpenAI endpoint. Falls back to AZURE_OPENAI_ENDPOINT.
        azure_api_version : str, optional
            Azure API version. Falls back to AZURE_OPENAI_API_VERSION or "2024-02-01".
        """
        self._provider = provider or self._detect_provider()
        self._model = model or self._detect_model()
        self._client = self._create_client(api_key, azure_endpoint, azure_api_version)

    def _detect_provider(self) -> str:
        """Auto-detect provider from environment variables."""
        if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
            return "azure"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY for OpenAI or "
            "AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT for Azure OpenAI."
        )

    def _detect_model(self) -> str:
        """Auto-detect model from environment variables."""
        if self._provider == "azure":
            return os.environ.get("AZURE_OPENAI_DEPLOYMENT", DEFAULT_MODEL)
        return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    def _create_client(
        self,
        api_key: str | None,
        azure_endpoint: str | None,
        azure_api_version: str | None,
    ):
        """Create the appropriate client based on provider."""
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as err:
            raise ImportError(
                "openai package required. Install with: pip install underthesea[agent]"
            ) from err

        if self._provider == "azure":
            key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
            version = (
                azure_api_version
                or os.environ.get("AZURE_OPENAI_API_VERSION")
                or "2024-02-01"
            )

            if not key:
                raise ValueError("Azure OpenAI API key required. Set AZURE_OPENAI_API_KEY.")
            if not endpoint:
                raise ValueError("Azure OpenAI endpoint required. Set AZURE_OPENAI_ENDPOINT.")

            return AzureOpenAI(
                api_key=key,
                azure_endpoint=endpoint,
                api_version=version,
            )
        else:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY.")

            return OpenAI(api_key=key)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Send chat messages and get a response.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of messages with "role" and "content" keys.
        model : str, optional
            Model name (OpenAI) or deployment name (Azure). Uses default if not specified.
        temperature : float
            Sampling temperature.

        Returns
        -------
        str
            Assistant response content.
        """
        response = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    @property
    def provider(self) -> str:
        """Get the current provider name."""
        return self._provider

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model
