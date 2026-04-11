"""Google Gemini provider - zero external dependencies."""

import json
import os
import urllib.parse

from underthesea.agent.providers import _http
from underthesea.agent.providers.base import BaseProvider, ProviderMessage, StreamDelta, ToolCall


class Gemini(BaseProvider):
    """Google Gemini API provider (zero dependencies).

    >>> from underthesea.agent import Gemini
    >>> llm = Gemini(api_key="...")
    >>> llm = Gemini()  # uses GOOGLE_API_KEY env var
    """

    DEFAULT_MODEL = "gemini-2.0-flash"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY.")
        self._model = model or os.environ.get("GEMINI_MODEL", self.DEFAULT_MODEL)

    def chat(self, messages, model=None, temperature=0.7, tools=None, tool_choice=None):
        mdl = model or self._model
        url = (
            f"{self.BASE_URL}/models/{mdl}:generateContent"
            f"?key={urllib.parse.quote(self._api_key)}"
        )

        system_text, contents = self._convert_messages(messages)
        body = {"contents": contents, "generationConfig": {"temperature": temperature}}

        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if tools:
            body["tools"] = [{"functionDeclarations": self._convert_tools(tools)}]
            if tool_choice:
                mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(tool_choice, "AUTO")
                body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}

        data = _http.post_json(url, {"Content-Type": "application/json"}, body)
        return self._parse_response(data)

    def _convert_messages(self, messages):
        system_text = None
        contents = []

        for msg in messages:
            role = msg["role"]

            if role == "system":
                system_text = msg["content"]

            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})

            elif role == "assistant":
                parts = []
                if msg.get("content"):
                    parts.append({"text": msg["content"]})
                for tc in msg.get("tool_calls") or []:
                    tc_func = tc.get("function", tc)
                    parts.append({
                        "functionCall": {
                            "name": tc_func.get("name", tc.get("name")),
                            "args": json.loads(tc_func.get("arguments", tc.get("arguments", "{}"))),
                        }
                    })
                contents.append({"role": "model", "parts": parts})

            elif role == "tool":
                raw_content = msg["content"]
                response = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                contents.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": "function", "response": response}}],
                })

        return system_text, contents

    def _convert_tools(self, openai_tools):
        declarations = []
        for tool in openai_tools:
            func = tool["function"]
            params = func.get("parameters", {})
            declarations.append({
                "name": func["name"],
                "description": func["description"],
                "parameters": self._upcase_types(params),
            })
        return declarations

    def _upcase_types(self, schema):
        """Gemini requires UPPERCASE type strings."""
        if not isinstance(schema, dict):
            return schema
        result = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                result[k] = v.upper()
            elif isinstance(v, dict):
                result[k] = self._upcase_types(v)
            else:
                result[k] = v
        return result

    def _parse_response(self, data):
        content = None
        tool_calls = []
        parts = data["candidates"][0]["content"]["parts"]

        for part in parts:
            if "text" in part:
                content = part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=fc.get("id", f"call_{fc['name']}"),
                        name=fc["name"],
                        arguments=json.dumps(fc.get("args", {})),
                    )
                )

        return ProviderMessage(content=content, tool_calls=tool_calls, raw=data)

    def chat_stream(self, messages, model=None, temperature=0.7, tools=None, tool_choice=None):
        mdl = model or self._model
        url = (
            f"{self.BASE_URL}/models/{mdl}:streamGenerateContent"
            f"?alt=sse&key={urllib.parse.quote(self._api_key)}"
        )

        system_text, contents = self._convert_messages(messages)
        body = {"contents": contents, "generationConfig": {"temperature": temperature}}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if tools:
            body["tools"] = [{"functionDeclarations": self._convert_tools(tools)}]
            if tool_choice:
                mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(tool_choice, "AUTO")
                body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}

        for chunk in _http.stream_sse(url, {"Content-Type": "application/json"}, body):
            for candidate in chunk.get("candidates", []):
                finish = candidate.get("finishReason")
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        yield StreamDelta(content=part["text"])
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        yield StreamDelta(
                            tool_call_id=fc.get("id", f"call_{fc['name']}"),
                            tool_name=fc["name"],
                            tool_arguments_delta=json.dumps(fc.get("args", {})),
                        )
                if finish:
                    yield StreamDelta(finish_reason=finish)

    def supports_tool_calling(self):
        return True

    @property
    def name(self):
        return "gemini"

    @property
    def default_model(self):
        return self._model

    @property
    def model(self):
        return self._model
