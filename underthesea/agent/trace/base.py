"""Base tracer interface for agent observability."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTracer(ABC):
    """Abstract base class for agent tracers.

    Tracers capture the lifecycle of agent interactions:
    trace (top-level) -> generation (LLM call) / span (tool call).
    """

    @abstractmethod
    def start_trace(self, *, name: str, input: str, metadata: dict | None = None) -> str:
        """Start a new trace. Returns trace_id."""

    @abstractmethod
    def end_trace(
        self, trace_id: str, *, output: str | None = None,
        status: str = "ok", error: str | None = None,
    ):
        """End a trace."""

    @abstractmethod
    def start_generation(
        self, trace_id: str, *, name: str, model: str | None = None,
        input: object = None, metadata: dict | None = None,
    ) -> str:
        """Start an LLM generation span. Returns span_id."""

    @abstractmethod
    def end_generation(
        self, span_id: str, *, output: object = None,
        usage: dict | None = None, status: str = "ok", error: str | None = None,
    ):
        """End an LLM generation span."""

    @abstractmethod
    def start_span(
        self, trace_id: str, *, name: str,
        input: object = None, metadata: dict | None = None,
    ) -> str:
        """Start a generic span (e.g. tool call). Returns span_id."""

    @abstractmethod
    def end_span(
        self, span_id: str, *, output: object = None,
        status: str = "ok", error: str | None = None,
    ):
        """End a generic span."""


def extract_usage(raw: object) -> dict | None:
    """Extract normalized token usage from a raw provider response.

    Returns dict with keys: input_tokens, output_tokens, total_tokens.
    """
    if not raw or not isinstance(raw, dict):
        return None
    # OpenAI / Azure OpenAI
    if "usage" in raw:
        u = raw["usage"]
        input_t = u.get("prompt_tokens") or u.get("input_tokens", 0)
        output_t = u.get("completion_tokens") or u.get("output_tokens", 0)
        return {
            "input_tokens": input_t,
            "output_tokens": output_t,
            "total_tokens": u.get("total_tokens", input_t + output_t),
        }
    # Gemini
    if "usageMetadata" in raw:
        u = raw["usageMetadata"]
        input_t = u.get("promptTokenCount", 0)
        output_t = u.get("candidatesTokenCount", 0)
        return {
            "input_tokens": input_t,
            "output_tokens": output_t,
            "total_tokens": u.get("totalTokenCount", input_t + output_t),
        }
    return None
