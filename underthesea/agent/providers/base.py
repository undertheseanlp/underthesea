"""Base provider interface for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """Provider-agnostic tool call representation."""

    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class ProviderMessage:
    """Provider-agnostic message returned from chat."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: object = None  # Original provider-specific response


@dataclass
class StreamDelta:
    """A single chunk from a streaming response."""

    content: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_arguments_delta: str | None = None
    finish_reason: str | None = None


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> ProviderMessage:
        """Send messages and get a response."""
        ...

    def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> Generator[StreamDelta]:
        """Stream a response as deltas. Override in subclasses."""
        # Default: fall back to non-streaming and yield a single delta
        result = self.chat(messages, model=model, temperature=temperature,
                           tools=tools, tool_choice=tool_choice)
        if result.content:
            yield StreamDelta(content=result.content)
        for tc in result.tool_calls:
            yield StreamDelta(tool_call_id=tc.id, tool_name=tc.name,
                              tool_arguments_delta=tc.arguments)
        yield StreamDelta(finish_reason="stop")

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        ...
