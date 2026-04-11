import json
import os

from underthesea.agent.llm import LLM
from underthesea.agent.providers.base import BaseProvider
from underthesea.agent.tools import Tool
from underthesea.agent.trace.base import BaseTracer, extract_usage
from underthesea.agent.trace.decorator import _current_trace_id, _current_tracer

_auto_tracer = None


def _tracing_disabled() -> bool:
    """Check if tracing is disabled via environment variable."""
    return os.environ.get("UNDERTHESEA_TRACE_DISABLED", "").lower() in ("1", "true", "yes")


def _default_trace_dir() -> str:
    """``~/.underthesea/traces`` — the default directory for auto-traces."""
    return os.path.join(os.path.expanduser("~"), ".underthesea", "traces")


def _get_auto_tracer() -> BaseTracer:
    """Return a shared LocalTracer that writes to ``~/.underthesea/traces``.

    Traces are saved automatically — like Claude Code — so every agent call
    is recorded for later inspection.  Disable with
    ``UNDERTHESEA_TRACE_DISABLED=1``, change the directory with
    ``UNDERTHESEA_TRACE_DIR``.
    """
    global _auto_tracer
    if _auto_tracer is None:
        from underthesea.agent.trace.local import LocalTracer
        trace_dir = os.environ.get("UNDERTHESEA_TRACE_DIR", _default_trace_dir())
        _auto_tracer = LocalTracer(trace_dir=trace_dir, console=True)
    return _auto_tracer


class Agent:
    """Agent with custom tools support and multi-provider LLM backends."""

    DEFAULT_INSTRUCTION = "You are a helpful assistant."

    def __init__(
        self,
        name: str,
        tools: list[Tool] | None = None,
        instruction: str | None = None,
        max_iterations: int = 10,
        provider: BaseProvider | LLM | None = None,
        tracer: BaseTracer | None = None,
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
        provider : BaseProvider or LLM, optional
            LLM provider or LLM instance. If LLM is passed, its backend provider is used.
            If not specified, auto-detects from environment variables.
        tracer : BaseTracer, optional
            Tracer for observability (e.g. LocalTracer, LangfuseTracer).
            If omitted, the agent inherits the active ``@trace()`` context
            automatically.
        """
        self.name = name
        self.tools = tools or []
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self.max_iterations = max_iterations
        # Accept LLM (auto-detect) or any BaseProvider
        if isinstance(provider, LLM):
            self._provider = provider.backend
        else:
            self._provider = provider
        self._tracer = tracer
        self._history: list[dict] = []

    def _ensure_provider(self, **kwargs):
        """Initialize provider if not already set."""
        if self._provider is None:
            self._provider = LLM(**kwargs).backend

    def _resolve_tracing(self) -> tuple[BaseTracer | None, str | None, bool]:
        """Resolve tracer and trace_id from explicit config, @trace context, or auto.

        Resolution order:
        1. Explicit ``tracer=`` on the Agent → creates its own trace
        2. Active ``@trace()`` context      → attaches spans to existing trace
        3. Auto local tracer                → creates its own trace (like Claude Code)

        Disable everything with ``UNDERTHESEA_TRACE_DISABLED=1``.

        Returns (tracer, trace_id, owns_trace).
        - owns_trace=True  → caller must call end_trace()
        - owns_trace=False → spans attach to an existing trace
        """
        if _tracing_disabled():
            return None, None, False

        # 1. Explicit tracer on the Agent → always creates its own trace
        if self._tracer:
            return self._tracer, None, True

        # 2. Inherit from active @trace() context
        ctx_tracer = _current_tracer.get()
        ctx_trace_id = _current_trace_id.get()
        if ctx_tracer and ctx_trace_id:
            return ctx_tracer, ctx_trace_id, False

        # 3. Auto local trace — every call is recorded by default
        return _get_auto_tracer(), None, True

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
        self._ensure_provider(**llm_kwargs)
        self._history.append({"role": "user", "content": message})

        tracer, trace_id, owns_trace = self._resolve_tracing()

        if tracer and owns_trace:
            trace_id = tracer.start_trace(
                name=self.name, input=message,
                metadata={"model": model or getattr(self._provider, "default_model", None)},
            )

        try:
            if self.tools:
                response = self._call_with_tools(model, tracer=tracer, trace_id=trace_id)
            else:
                response = self._call_simple(model, tracer=tracer, trace_id=trace_id)

            if tracer and owns_trace and trace_id:
                tracer.end_trace(trace_id, output=response)
            return response
        except Exception as e:
            if tracer and owns_trace and trace_id:
                tracer.end_trace(trace_id, status="error", error=str(e))
            raise

    def _call_simple(
        self, model: str | None,
        tracer: BaseTracer | None = None, trace_id: str | None = None,
    ) -> str:
        """Handle message without tools."""
        messages = [{"role": "system", "content": self.instruction}] + self._history

        span_id = None
        if tracer and trace_id:
            span_id = tracer.start_generation(
                trace_id, name="llm.chat",
                model=model or getattr(self._provider, "default_model", None),
                input=messages,
            )

        result = self._provider.chat(messages, model=model)

        if tracer and span_id:
            usage = extract_usage(result.raw) if hasattr(result, "raw") else None
            tracer.end_generation(span_id, output=result.content, usage=usage)

        response = result.content
        self._history.append({"role": "assistant", "content": response})
        return response

    def _call_with_tools(
        self, model: str | None,
        tracer: BaseTracer | None = None, trace_id: str | None = None,
    ) -> str:
        """Handle message with tool calling loop."""
        messages = [{"role": "system", "content": self.instruction}] + self._history
        openai_tools = [t.to_openai_tool() for t in self.tools]
        tool_map = {t.name: t for t in self.tools}
        gen_count = 0

        for _ in range(self.max_iterations):
            gen_count += 1
            resolved_model = model or getattr(self._provider, "default_model", None)

            # -- trace: LLM generation --
            span_id = None
            if tracer and trace_id:
                span_id = tracer.start_generation(
                    trace_id, name=f"llm.chat #{gen_count}",
                    model=resolved_model, input=messages,
                )

            result = self._provider.chat(
                messages, model=model, tools=openai_tools, tool_choice="auto"
            )

            if tracer and span_id:
                usage = extract_usage(result.raw) if hasattr(result, "raw") else None
                gen_output = {
                    "content": result.content,
                    "tool_calls": (
                        [{"name": tc.name, "arguments": tc.arguments}
                         for tc in result.tool_calls]
                        if result.tool_calls else None
                    ),
                }
                tracer.end_generation(span_id, output=gen_output, usage=usage)

            if result.tool_calls:
                assistant_msg = {"role": "assistant", "content": result.content}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in result.tool_calls
                ]
                messages.append(assistant_msg)

                for tc in result.tool_calls:
                    tool = tool_map[tc.name]
                    args = json.loads(tc.arguments)

                    # -- trace: tool execution --
                    tool_span_id = None
                    if tracer and trace_id:
                        tool_span_id = tracer.start_span(
                            trace_id, name=f"tool.{tc.name}", input=args,
                        )

                    try:
                        tool_result = tool.execute(args)
                    except Exception as e:
                        if tracer and tool_span_id:
                            tracer.end_span(
                                tool_span_id, status="error", error=str(e),
                            )
                        raise

                    if tracer and tool_span_id:
                        tracer.end_span(tool_span_id, output=tool_result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })
            else:
                content = result.content
                self._history.append({"role": "assistant", "content": content})
                return content

        raise RuntimeError("Max tool iterations reached")

    def stream(self, message: str, model: str | None = None, **llm_kwargs):
        """Stream a response, yielding text chunks.

        Parameters
        ----------
        message : str
            User message.
        model : str, optional
            Model override.

        Yields
        ------
        str
            Text chunks as they arrive.
        """
        self._ensure_provider(**llm_kwargs)
        self._history.append({"role": "user", "content": message})
        messages = [{"role": "system", "content": self.instruction}] + self._history

        tracer, trace_id, owns_trace = self._resolve_tracing()

        if tracer and owns_trace:
            trace_id = tracer.start_trace(name=self.name, input=message)

        span_id = None
        if tracer and trace_id:
            span_id = tracer.start_generation(
                trace_id, name="llm.chat_stream",
                model=model or getattr(self._provider, "default_model", None),
                input=messages,
            )

        full_content = []
        for delta in self._provider.chat_stream(messages, model=model):
            if delta.content:
                full_content.append(delta.content)
                yield delta.content

        response = "".join(full_content)
        self._history.append({"role": "assistant", "content": response})

        if tracer and span_id:
            tracer.end_generation(span_id, output=response)
        if tracer and owns_trace and trace_id:
            tracer.end_trace(trace_id, output=response)

    def reset(self):
        """Clear conversation history."""
        self._history = []

    @property
    def history(self) -> list[dict]:
        """Get conversation history."""
        return self._history.copy()

    @property
    def tracer(self) -> BaseTracer | None:
        return self._tracer

    @tracer.setter
    def tracer(self, value: BaseTracer | None):
        self._tracer = value


class _AgentInstance:
    """Vietnamese-focused conversational AI agent."""

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
        if self._llm is not None:
            return
        self._llm = LLM(
            provider=provider, model=model, api_key=api_key,
            azure_endpoint=azure_endpoint, azure_api_version=azure_api_version,
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
        tracer: BaseTracer | None = None,
    ) -> str:
        self._ensure_llm(provider, model, api_key, azure_endpoint, azure_api_version)

        if system_prompt:
            self._system_prompt = system_prompt

        # Resolve tracer: explicit > context > auto
        if _tracing_disabled():
            tracer = None
        elif tracer is None:
            ctx_tracer = _current_tracer.get()
            tracer = ctx_tracer if ctx_tracer else _get_auto_tracer()

        self._history.append({"role": "user", "content": message})
        messages = [{"role": "system", "content": self._system_prompt}] + self._history

        trace_id = None
        span_id = None
        owns_trace = False
        ctx_trace_id = _current_trace_id.get()

        if tracer:
            if ctx_trace_id:
                # Inside @trace context → add spans to existing trace
                trace_id = ctx_trace_id
            else:
                # Standalone / auto → create new trace
                trace_id = tracer.start_trace(
                    name="agent", input=message,
                    metadata={
                        "model": model or self._llm.model,
                        "provider": self._llm.provider,
                    },
                )
                owns_trace = True
            span_id = tracer.start_generation(
                trace_id, name="llm.chat",
                model=model or self._llm.model, input=messages,
            )

        try:
            assistant_message = self._llm.chat(messages, model=model)
        except Exception as e:
            if tracer and span_id:
                tracer.end_generation(span_id, status="error", error=str(e))
            if tracer and owns_trace and trace_id:
                tracer.end_trace(trace_id, status="error", error=str(e))
            raise

        if tracer and span_id:
            tracer.end_generation(span_id, output=assistant_message)
        if tracer and owns_trace and trace_id:
            tracer.end_trace(trace_id, output=assistant_message)

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
