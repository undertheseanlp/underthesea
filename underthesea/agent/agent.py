import json

from underthesea.agent.llm import LLM
from underthesea.agent.tools import Tool


class Agent:
    """Agent with custom tools support using OpenAI function calling."""

    DEFAULT_INSTRUCTION = "You are a helpful assistant."

    def __init__(
        self,
        name: str,
        tools: list[Tool] | None = None,
        instruction: str | None = None,
        max_iterations: int = 10,
    ):
        """
        Initialize an Agent.

        Parameters
        ----------
        name : str
            Agent name.
        tools : list[Tool], optional
            List of tools available to the agent.
        instruction : str, optional
            System instruction for the agent.
        max_iterations : int
            Maximum number of tool calling iterations.
        """
        self.name = name
        self.tools = tools or []
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self.max_iterations = max_iterations
        self._llm: LLM | None = None
        self._history: list[dict] = []

    def _ensure_llm(self, **kwargs):
        """Initialize LLM client if not already done."""
        if self._llm is None:
            self._llm = LLM(**kwargs)

    def __call__(
        self,
        message: str,
        model: str | None = None,
        **llm_kwargs,
    ) -> str:
        """
        Send message and get response, using tools if available.

        Parameters
        ----------
        message : str
            User message.
        model : str, optional
            Model name to use.
        **llm_kwargs
            Additional arguments passed to LLM initialization.

        Returns
        -------
        str
            Assistant response.
        """
        self._ensure_llm(**llm_kwargs)
        self._history.append({"role": "user", "content": message})

        if self.tools:
            return self._call_with_tools(model)

        messages = [{"role": "system", "content": self.instruction}] + self._history
        response = self._llm.chat(messages, model=model)
        self._history.append({"role": "assistant", "content": response})
        return response

    def _call_with_tools(self, model: str | None) -> str:
        """Handle message with tool calling loop."""
        messages = [{"role": "system", "content": self.instruction}] + self._history
        openai_tools = [t.to_openai_tool() for t in self.tools]
        tool_map = {t.name: t for t in self.tools}

        for _ in range(self.max_iterations):
            response = self._llm._client.chat.completions.create(
                model=model or self._llm._model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    tool = tool_map[tc.function.name]
                    result = tool.execute(json.loads(tc.function.arguments))
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )
            else:
                content = msg.content
                self._history.append({"role": "assistant", "content": content})
                return content

        raise RuntimeError("Max tool iterations reached")

    def reset(self):
        """Clear conversation history."""
        self._history = []

    @property
    def history(self) -> list[dict]:
        """Get conversation history."""
        return self._history.copy()


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
