"""Langfuse tracer — sends traces to Langfuse for remote observability.

Compatible with Langfuse v4+ which uses the ``start_observation`` API.
Observations are nested under a root agent span for proper hierarchy.
"""

from __future__ import annotations

import os

from underthesea.agent.trace.base import BaseTracer


class LangfuseTracer(BaseTracer):
    """Send agent traces to a Langfuse instance.

    Requires the ``langfuse`` package (``pip install langfuse``).

    Parameters
    ----------
    public_key : str, optional
        Langfuse public key. Falls back to ``LANGFUSE_PUBLIC_KEY`` env var.
    secret_key : str, optional
        Langfuse secret key. Falls back to ``LANGFUSE_SECRET_KEY`` env var.
    host : str, optional
        Langfuse host URL. Falls back to ``LANGFUSE_HOST`` env var
        (default ``https://cloud.langfuse.com``).

    Examples
    --------
    >>> from underthesea.agent import Agent, LangfuseTracer
    >>> tracer = LangfuseTracer()
    >>> bot = Agent("demo", tracer=tracer)
    >>> bot("Xin chao!")
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ):
        try:
            from langfuse import Langfuse  # noqa: F401
        except ImportError:
            raise ImportError(
                "langfuse package is required for LangfuseTracer. "
                "Install it with: pip install langfuse"
            ) from None

        self._langfuse = Langfuse(
            public_key=public_key or os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=secret_key or os.environ.get("LANGFUSE_SECRET_KEY"),
            host=host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        # trace_id → root observation (as_type="agent")
        self._traces: dict[str, object] = {}
        # span_id → child observation
        self._spans: dict[str, object] = {}

    # ------------------------------------------------------------------
    # BaseTracer implementation
    # ------------------------------------------------------------------

    def start_trace(self, *, name: str, input: str, metadata: dict | None = None) -> str:
        trace_id = self._langfuse.create_trace_id()
        obs = self._langfuse.start_observation(
            trace_context={"trace_id": trace_id},
            name=name,
            as_type="agent",
            input=input,
            metadata=metadata,
        )
        self._traces[trace_id] = obs
        return trace_id

    def end_trace(
        self, trace_id: str, *, output: str | None = None,
        status: str = "ok", error: str | None = None,
    ):
        obs = self._traces.pop(trace_id, None)
        if not obs:
            return
        update_kwargs: dict = {}
        if output is not None:
            update_kwargs["output"] = output
        if error:
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = f"{status}: {error}"
        else:
            update_kwargs["status_message"] = status
        if update_kwargs:
            obs.update(**update_kwargs)
        obs.end()
        self._langfuse.flush()

    def start_generation(
        self, trace_id: str, *, name: str, model: str | None = None,
        input: object = None, metadata: dict | None = None,
    ) -> str:
        parent = self._traces.get(trace_id)
        if not parent:
            raise ValueError(f"Unknown trace_id: {trace_id}")
        gen = parent.start_observation(
            name=name,
            as_type="generation",
            input=input,
            model=model,
            metadata=metadata,
        )
        self._spans[gen.id] = gen
        return gen.id

    def end_generation(
        self, span_id: str, *, output: object = None,
        usage: dict | None = None, status: str = "ok", error: str | None = None,
    ):
        gen = self._spans.pop(span_id, None)
        if not gen:
            return
        update_kwargs: dict = {}
        if output is not None:
            update_kwargs["output"] = output
        if usage:
            update_kwargs["usage_details"] = {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
                "total": usage.get("total_tokens", 0),
            }
        if error:
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = f"error: {error}"
        else:
            update_kwargs["status_message"] = status
        if update_kwargs:
            gen.update(**update_kwargs)
        gen.end()

    def start_span(
        self, trace_id: str, *, name: str,
        input: object = None, metadata: dict | None = None,
    ) -> str:
        parent = self._traces.get(trace_id)
        if not parent:
            raise ValueError(f"Unknown trace_id: {trace_id}")
        span = parent.start_observation(
            name=name,
            as_type="tool",
            input=input,
            metadata=metadata,
        )
        self._spans[span.id] = span
        return span.id

    def end_span(
        self, span_id: str, *, output: object = None,
        status: str = "ok", error: str | None = None,
    ):
        span = self._spans.pop(span_id, None)
        if not span:
            return
        update_kwargs: dict = {}
        if output is not None:
            update_kwargs["output"] = output
        if error:
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = f"error: {error}"
        else:
            update_kwargs["status_message"] = status
        if update_kwargs:
            span.update(**update_kwargs)
        span.end()

    def flush(self):
        """Flush pending events to Langfuse."""
        self._langfuse.flush()

    def shutdown(self):
        """Shutdown the Langfuse client cleanly."""
        self._langfuse.shutdown()
