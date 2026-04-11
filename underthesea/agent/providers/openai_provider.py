"""OpenAI and Azure OpenAI providers - zero external dependencies."""

import os

from underthesea.agent.providers import _http
from underthesea.agent.providers.base import BaseProvider, ProviderMessage, StreamDelta, ToolCall


def _parse_openai_response(data: dict) -> ProviderMessage:
    """Parse OpenAI JSON response into ProviderMessage."""
    msg = data["choices"][0]["message"]
    tool_calls = []
    for tc in msg.get("tool_calls") or []:
        tool_calls.append(
            ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=tc["function"]["arguments"])
        )
    return ProviderMessage(content=msg.get("content"), tool_calls=tool_calls, raw=data)


def _stream_openai(url, headers, body):
    """Shared streaming logic for OpenAI/Azure. Yields StreamDelta."""
    body["stream"] = True
    for chunk in _http.stream_sse(url, headers, body):
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        finish = choices[0].get("finish_reason")

        # Text content
        text = delta.get("content")

        # Tool calls
        tool_name = tool_id = tool_args = None
        for tc in delta.get("tool_calls") or []:
            func = tc.get("function", {})
            tool_id = tc.get("id") or tool_id
            tool_name = func.get("name") or tool_name
            tool_args = func.get("arguments")

        if text or tool_name or tool_id or tool_args or finish:
            yield StreamDelta(
                content=text,
                tool_name=tool_name,
                tool_call_id=tool_id,
                tool_arguments_delta=tool_args,
                finish_reason=finish,
            )


class OpenAI(BaseProvider):
    """OpenAI API provider (zero dependencies).

    >>> from underthesea.agent import OpenAI
    >>> llm = OpenAI(api_key="sk-...")
    >>> llm = OpenAI()  # uses OPENAI_API_KEY env var
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY.")
        self._model = model or os.environ.get("OPENAI_MODEL", self.DEFAULT_MODEL)
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL", self.BASE_URL)).rstrip("/")

    def _url(self):
        return f"{self._base_url}/chat/completions"

    def _headers(self):
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}

    def _body(self, messages, model, temperature, tools, tool_choice):
        body = {"model": model or self._model, "messages": messages, "temperature": temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        return body

    def chat(self, messages, model=None, temperature=0.7, tools=None, tool_choice=None):
        data = _http.post_json(
            self._url(), self._headers(),
            self._body(messages, model, temperature, tools, tool_choice),
        )
        return _parse_openai_response(data)

    def chat_stream(self, messages, model=None, temperature=0.7, tools=None, tool_choice=None):
        yield from _stream_openai(
            self._url(), self._headers(),
            self._body(messages, model, temperature, tools, tool_choice),
        )

    def supports_tool_calling(self):
        return True

    @property
    def name(self):
        return "openai"

    @property
    def default_model(self):
        return self._model

    @property
    def model(self):
        return self._model


class AzureOpenAI(BaseProvider):
    """Azure OpenAI provider (zero dependencies).

    >>> from underthesea.agent import AzureOpenAI
    >>> llm = AzureOpenAI(api_key="...", endpoint="https://my.openai.azure.com", deployment="gpt-4")
    >>> llm = AzureOpenAI()  # uses AZURE_OPENAI_* env vars
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
    ):
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._endpoint = (endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
        self._model = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", self.DEFAULT_MODEL)
        self._api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-02-01"

        if not self._api_key:
            raise ValueError("Azure OpenAI API key required. Set AZURE_OPENAI_API_KEY.")
        if not self._endpoint:
            raise ValueError("Azure OpenAI endpoint required. Set AZURE_OPENAI_ENDPOINT.")

    def _url(self, model=None):
        deploy = model or self._model
        return f"{self._endpoint}/openai/deployments/{deploy}/chat/completions?api-version={self._api_version}"

    def _headers(self):
        return {"Content-Type": "application/json", "api-key": self._api_key}

    def _body(self, messages, temperature, tools, tool_choice):
        body = {"messages": messages, "temperature": temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        return body

    def chat(self, messages, model=None, temperature=0.7, tools=None, tool_choice=None):
        data = _http.post_json(
            self._url(model), self._headers(),
            self._body(messages, temperature, tools, tool_choice),
        )
        return _parse_openai_response(data)

    def chat_stream(self, messages, model=None, temperature=0.7, tools=None, tool_choice=None):
        yield from _stream_openai(
            self._url(model), self._headers(),
            self._body(messages, temperature, tools, tool_choice),
        )

    def supports_tool_calling(self):
        return True

    @property
    def name(self):
        return "azure"

    @property
    def default_model(self):
        return self._model

    @property
    def model(self):
        return self._model
