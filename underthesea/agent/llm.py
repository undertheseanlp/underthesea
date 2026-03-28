"""LLM - auto-detect provider from environment variables."""

from __future__ import annotations

import os

from underthesea.agent.providers.base import BaseProvider


class LLM(BaseProvider):
    """Auto-detecting LLM client. Picks the right provider from environment variables.

    Detection order:
    1. AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT → AzureOpenAI
    2. OPENAI_API_KEY → OpenAI
    3. ANTHROPIC_API_KEY → Anthropic
    4. GOOGLE_API_KEY → Gemini

    Examples
    --------
    >>> from underthesea.agent import LLM
    >>> llm = LLM()  # auto-detect from env

    For explicit provider selection, use the provider classes directly:
    >>> from underthesea.agent import OpenAI, AzureOpenAI, Anthropic, Gemini
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        # Azure-specific
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
        # Anthropic-specific
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        resolved = provider or self._detect_provider_name()

        if resolved == "anthropic":
            from underthesea.agent.providers.anthropic_provider import Anthropic
            self._backend = Anthropic(
                api_key=api_key, model=model, base_url=base_url, max_tokens=max_tokens
            )
        elif resolved == "azure":
            from underthesea.agent.providers.openai_provider import AzureOpenAI
            self._backend = AzureOpenAI(
                api_key=api_key, endpoint=azure_endpoint, deployment=model,
                api_version=azure_api_version,
            )
        elif resolved == "gemini":
            from underthesea.agent.providers.gemini_provider import Gemini
            self._backend = Gemini(api_key=api_key, model=model)
        else:
            from underthesea.agent.providers.openai_provider import OpenAI
            self._backend = OpenAI(api_key=api_key, model=model, base_url=base_url)

        self._provider = self._backend.name
        self._model = self._backend.default_model

    @staticmethod
    def _detect_provider_name() -> str:
        if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
            return "azure"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("GOOGLE_API_KEY"):
            return "gemini"
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY for OpenAI or "
            "AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT for Azure OpenAI."
        )

    # --- chat() returns str for backward compat ---

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> str:
        """Send chat messages and get a response string."""
        result = self._backend.chat(
            messages, model=model, temperature=temperature, tools=tools, tool_choice=tool_choice
        )
        return result.content

    def chat_raw(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ):
        """Send chat messages and get a ProviderMessage (with tool_calls etc.)."""
        return self._backend.chat(
            messages, model=model, temperature=temperature, tools=tools, tool_choice=tool_choice
        )

    def supports_tool_calling(self):
        return self._backend.supports_tool_calling()

    @property
    def name(self):
        return self._backend.name

    @property
    def default_model(self):
        return self._backend.default_model

    @property
    def backend(self) -> BaseProvider:
        return self._backend

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model
