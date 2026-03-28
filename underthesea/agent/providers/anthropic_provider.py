"""Anthropic Claude provider - zero external dependencies."""

import json
import os

from underthesea.agent.providers import _http
from underthesea.agent.providers.base import BaseProvider, ProviderMessage, StreamDelta, ToolCall


class Anthropic(BaseProvider):
    """Anthropic Claude API provider (zero dependencies).

    >>> from underthesea.agent import Anthropic
    >>> llm = Anthropic(api_key="sk-ant-...")
    >>> llm = Anthropic()  # uses ANTHROPIC_API_KEY env var
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    BASE_URL = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY.")
        self._model = model or os.environ.get("ANTHROPIC_MODEL", self.DEFAULT_MODEL)
        self._base_url = (base_url or self.BASE_URL).rstrip("/")
        self._max_tokens = max_tokens

    def chat(self, messages, model=None, temperature=0.7, tools=None, tool_choice=None):
        system_prompt = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered.append(msg)

        body = {
            "model": model or self._model,
            "messages": self._convert_messages(filtered),
            "max_tokens": self._max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            body["system"] = system_prompt
        if tools:
            body["tools"] = self._convert_tools(tools)
            if tool_choice:
                body["tool_choice"] = self._convert_tool_choice(tool_choice)

        data = _http.post_json(
            f"{self._base_url}/v1/messages",
            {
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": self.API_VERSION,
            },
            body,
        )
        return self._parse_response(data)

    def _convert_messages(self, messages):
        converted = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "tool":
                tool_results = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": messages[i]["tool_call_id"],
                        "content": messages[i]["content"],
                    })
                    i += 1
                converted.append({"role": "user", "content": tool_results})

            elif msg["role"] == "assistant" and "tool_calls" in msg:
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    tc_func = tc.get("function", tc)
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", tc_func.get("id")),
                        "name": tc_func.get("name", tc.get("name")),
                        "input": json.loads(tc_func.get("arguments", tc.get("arguments", "{}"))),
                    })
                converted.append({"role": "assistant", "content": content_blocks})
                i += 1

            else:
                converted.append(msg)
                i += 1

        return converted

    def _convert_tools(self, openai_tools):
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in openai_tools
        ]

    def _convert_tool_choice(self, choice):
        return {"type": {"auto": "auto", "none": "none", "required": "any"}.get(choice, "auto")}

    def _parse_response(self, data):
        content = None
        tool_calls = []
        for block in data["content"]:
            if block["type"] == "text":
                content = block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(
                    ToolCall(id=block["id"], name=block["name"], arguments=json.dumps(block["input"]))
                )
        return ProviderMessage(content=content, tool_calls=tool_calls, raw=data)

    def chat_stream(self, messages, model=None, temperature=0.7, tools=None, tool_choice=None):
        system_prompt = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered.append(msg)

        body = {
            "model": model or self._model,
            "messages": self._convert_messages(filtered),
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system_prompt:
            body["system"] = system_prompt
        if tools:
            body["tools"] = self._convert_tools(tools)
            if tool_choice:
                body["tool_choice"] = self._convert_tool_choice(tool_choice)

        current_tool_id = None
        current_tool_name = None

        for chunk in _http.stream_sse(
            f"{self._base_url}/v1/messages",
            {
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": self.API_VERSION,
            },
            body,
        ):
            event_type = chunk.get("type", "")

            if event_type == "content_block_start":
                block = chunk.get("content_block", {})
                if block.get("type") == "tool_use":
                    current_tool_id = block["id"]
                    current_tool_name = block["name"]
                    yield StreamDelta(tool_call_id=current_tool_id, tool_name=current_tool_name)

            elif event_type == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield StreamDelta(content=delta["text"])
                elif delta.get("type") == "input_json_delta":
                    yield StreamDelta(
                        tool_call_id=current_tool_id,
                        tool_arguments_delta=delta.get("partial_json", ""),
                    )

            elif event_type == "content_block_stop":
                current_tool_id = None
                current_tool_name = None

            elif event_type == "message_delta":
                stop = chunk.get("delta", {}).get("stop_reason")
                if stop:
                    yield StreamDelta(finish_reason=stop)

    def supports_tool_calling(self):
        return True

    @property
    def name(self):
        return "anthropic"

    @property
    def default_model(self):
        return self._model

    @property
    def model(self):
        return self._model
