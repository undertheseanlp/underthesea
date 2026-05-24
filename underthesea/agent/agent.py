import json
import os
from concurrent.futures import ThreadPoolExecutor

from underthesea.agent.llm import LLM
from underthesea.agent.providers.base import BaseProvider
from underthesea.agent.tools import Tool
from underthesea.agent.trace.base import BaseTracer, extract_usage
from underthesea.agent.trace.decorator import _current_trace_id, _current_tracer

_auto_tracer = None


def _empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _safe_load_args(raw: str) -> tuple[dict, str | None]:
    """Parse a JSON-encoded tool-arguments blob, tolerating empty strings."""
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        return {}, str(e)
    if not isinstance(parsed, dict):
        return {}, f"Tool arguments must decode to an object, got {type(parsed).__name__}"
    return parsed, None


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
        parallel_tools: bool = True,
        max_tool_workers: int = 8,
        tool_error_handling: str = "recover",
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
        parallel_tools : bool
            When the model returns multiple tool calls in one response, execute
            them concurrently in a thread pool.  Set to ``False`` to keep
            sequential execution (useful when tools share mutable state).
        max_tool_workers : int
            Upper bound on threads used for parallel tool execution.
        tool_error_handling : {"recover", "raise"}
            ``"recover"`` (default) catches tool exceptions and feeds the error
            back to the model as the tool result, so the agent can self-correct.
            ``"raise"`` propagates the original behaviour of aborting the run.
        """
        if tool_error_handling not in ("recover", "raise"):
            raise ValueError(
                f"tool_error_handling must be 'recover' or 'raise', got {tool_error_handling!r}",
            )
        self.name = name
        self.tools = tools or []
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self.max_iterations = max_iterations
        self.parallel_tools = parallel_tools
        self.max_tool_workers = max_tool_workers
        self.tool_error_handling = tool_error_handling
        # Accept LLM (auto-detect) or any BaseProvider
        if isinstance(provider, LLM):
            self._provider = provider.backend
        else:
            self._provider = provider
        self._tracer = tracer
        self._history: list[dict] = []
        self._last_usage: dict[str, int] = _empty_usage()
        self._total_usage: dict[str, int] = _empty_usage()

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
        self._last_usage = _empty_usage()

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

        usage = extract_usage(result.raw) if hasattr(result, "raw") else None
        self._accumulate_usage(usage)
        if tracer and span_id:
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

            usage = extract_usage(result.raw) if hasattr(result, "raw") else None
            self._accumulate_usage(usage)
            if tracer and span_id:
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

                tool_messages = self._run_tool_calls(
                    result.tool_calls, tool_map, tracer, trace_id,
                )
                messages.extend(tool_messages)
            else:
                content = result.content
                self._history.append({"role": "assistant", "content": content})
                return content

        raise RuntimeError("Max tool iterations reached")

    def _run_tool_calls(
        self,
        tool_calls,
        tool_map: dict[str, Tool],
        tracer: BaseTracer | None,
        trace_id: str | None,
    ) -> list[dict]:
        """Execute a batch of tool calls and return the resulting tool messages.

        Calls run concurrently in a thread pool when ``parallel_tools`` is set
        and there is more than one call.  Order is preserved so the model sees
        results in the same order it requested them.
        """
        if not tool_calls:
            return []

        run_one = self._run_single_tool

        if self.parallel_tools and len(tool_calls) > 1:
            workers = min(self.max_tool_workers, len(tool_calls))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(
                    lambda tc: run_one(tc, tool_map, tracer, trace_id),
                    tool_calls,
                ))
        else:
            results = [run_one(tc, tool_map, tracer, trace_id) for tc in tool_calls]
        return results

    def _run_single_tool(
        self,
        tc,
        tool_map: dict[str, Tool],
        tracer: BaseTracer | None,
        trace_id: str | None,
    ) -> dict:
        """Run one tool call and return the matching ``role=tool`` message.

        Errors are JSON-encoded and surfaced to the model unless the agent is
        configured with ``tool_error_handling="raise"``.
        """
        args, parse_error = _safe_load_args(tc.arguments)

        tool_span_id = None
        if tracer and trace_id:
            tool_span_id = tracer.start_span(
                trace_id, name=f"tool.{tc.name}", input=args,
            )

        try:
            if parse_error is not None:
                raise ValueError(f"Invalid tool arguments JSON: {parse_error}")
            tool = tool_map.get(tc.name)
            if tool is None:
                raise KeyError(f"Unknown tool: {tc.name!r}")
            tool_result = tool.execute(args)
        except Exception as e:
            if tracer and tool_span_id:
                tracer.end_span(tool_span_id, status="error", error=str(e))
            if self.tool_error_handling == "raise":
                raise
            tool_result = json.dumps(
                {"error": str(e), "tool": tc.name},
                ensure_ascii=False,
            )
        else:
            if tracer and tool_span_id:
                tracer.end_span(tool_span_id, output=tool_result)

        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": tool_result,
        }

    def _accumulate_usage(self, usage: dict | None) -> None:
        """Track per-call and lifetime token usage on the agent."""
        if not usage:
            return
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(key, 0) or 0
            self._last_usage[key] = self._last_usage.get(key, 0) + value
            self._total_usage[key] = self._total_usage.get(key, 0) + value

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
        """Clear conversation history and per-call usage (lifetime totals kept)."""
        self._history = []
        self._last_usage = _empty_usage()

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

    @property
    def last_usage(self) -> dict[str, int]:
        """Token usage from the most recent ``agent(...)`` call."""
        return dict(self._last_usage)

    @property
    def total_usage(self) -> dict[str, int]:
        """Cumulative token usage across the agent's lifetime."""
        return dict(self._total_usage)


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
