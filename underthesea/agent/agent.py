from underthesea.agent.llm import LLM


class _AgentInstance:
    """Vietnamese-focused conversational AI agent using OpenAI or Azure OpenAI."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant specialized in Vietnamese language and NLP tasks."

    def __init__(self):
        self._llm: LLM | None = None
        self._system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self._history: list[dict[str, str]] = []

    def _ensure_llm(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
    ):
        """Initialize LLM client if not already done."""
        if self._llm is not None:
            return

        self._llm = LLM(
            provider=provider,
            model=model,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        )

    def __call__(
        self,
        message: str,
        model: str | None = None,
        system_prompt: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
    ) -> str:
        """
        Send a message and get a response.

        Parameters
        ----------
        message : str
            User message
        model : str, optional
            Model name (OpenAI) or deployment name (Azure).
            Falls back to OPENAI_MODEL, AZURE_OPENAI_DEPLOYMENT, or "gpt-4o-mini".
        system_prompt : str, optional
            Custom system prompt
        provider : str, optional
            Provider: "openai" or "azure". Auto-detected if not specified.
        api_key : str, optional
            API key. Falls back to OPENAI_API_KEY or AZURE_OPENAI_API_KEY.
        azure_endpoint : str, optional
            Azure OpenAI endpoint. Falls back to AZURE_OPENAI_ENDPOINT.
        azure_api_version : str, optional
            Azure API version. Falls back to AZURE_OPENAI_API_VERSION.

        Returns
        -------
        str
            Assistant response
        """
        self._ensure_llm(provider, model, api_key, azure_endpoint, azure_api_version)

        if system_prompt:
            self._system_prompt = system_prompt

        self._history.append({"role": "user", "content": message})

        messages = [{"role": "system", "content": self._system_prompt}] + self._history

        assistant_message = self._llm.chat(messages, model=model)
        self._history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def reset(self):
        """Clear conversation history."""
        self._history = []

    @property
    def history(self) -> list[dict[str, str]]:
        """Get conversation history."""
        return self._history.copy()


agent = _AgentInstance()
