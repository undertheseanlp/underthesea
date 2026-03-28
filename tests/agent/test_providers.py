"""Tests for provider abstraction layer."""

import json
import os
from unittest import TestCase
from unittest.mock import patch

from underthesea.agent.providers.base import BaseProvider, ProviderMessage, ToolCall

# Helper to mock post_json responses
MOCK_POST = "underthesea.agent.providers._http.post_json"


def _openai_response(content="Hello!", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
        msg["content"] = None
    return {"choices": [{"message": msg, "finish_reason": "stop"}]}


def _anthropic_response(content="Hello!", tool_calls=None):
    blocks = []
    if content:
        blocks.append({"type": "text", "text": content})
    for tc in tool_calls or []:
        blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]})
    return {"content": blocks, "stop_reason": "end_turn"}


class TestProviderMessage(TestCase):
    def test_message_without_tool_calls(self):
        msg = ProviderMessage(content="Hello!")
        self.assertEqual(msg.content, "Hello!")
        self.assertEqual(msg.tool_calls, [])

    def test_message_with_tool_calls(self):
        tc = ToolCall(id="call_1", name="search", arguments='{"q": "test"}')
        msg = ProviderMessage(content=None, tool_calls=[tc])
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].name, "search")


class TestBaseProvider(TestCase):
    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            BaseProvider()


class TestOpenAI(TestCase):
    def test_missing_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            from underthesea.agent.providers.openai_provider import OpenAI
            with self.assertRaises(ValueError):
                OpenAI()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    def test_basic_init(self):
        from underthesea.agent.providers.openai_provider import OpenAI
        p = OpenAI()
        self.assertEqual(p.name, "openai")
        self.assertEqual(p.model, "gpt-4o-mini")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    @patch(MOCK_POST, return_value=_openai_response("Hi!"))
    def test_chat(self, mock_post):
        from underthesea.agent.providers.openai_provider import OpenAI
        result = OpenAI().chat([{"role": "user", "content": "Hello"}])
        self.assertIsInstance(result, ProviderMessage)
        self.assertEqual(result.content, "Hi!")
        self.assertEqual(result.tool_calls, [])

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    @patch(MOCK_POST, return_value=_openai_response(tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": '{"x": 1}'}}
    ]))
    def test_chat_with_tool_calls(self, mock_post):
        from underthesea.agent.providers.openai_provider import OpenAI
        result = OpenAI().chat([{"role": "user", "content": "calc"}], tools=[{}])
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "calc")


class TestAzureOpenAI(TestCase):
    def test_missing_endpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            from underthesea.agent.providers.openai_provider import AzureOpenAI
            with self.assertRaises(ValueError):
                AzureOpenAI(api_key="key")

    def test_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from underthesea.agent.providers.openai_provider import AzureOpenAI
            with self.assertRaises(ValueError):
                AzureOpenAI(endpoint="https://test.azure.com")

    @patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "https://e"}, clear=True)
    def test_basic_init(self):
        from underthesea.agent.providers.openai_provider import AzureOpenAI
        p = AzureOpenAI()
        self.assertEqual(p.name, "azure")

    @patch.dict(os.environ, {}, clear=True)
    @patch(MOCK_POST, return_value=_openai_response("Azure!"))
    def test_explicit_params(self, mock_post):
        from underthesea.agent.providers.openai_provider import AzureOpenAI
        p = AzureOpenAI(api_key="k", endpoint="https://e", deployment="gpt-4", api_version="2024-06-01")
        result = p.chat([{"role": "user", "content": "Hi"}])
        self.assertEqual(result.content, "Azure!")
        self.assertEqual(p.model, "gpt-4")
        # Verify URL contains deployment and api-version
        call_url = mock_post.call_args[0][0]
        self.assertIn("gpt-4", call_url)
        self.assertIn("2024-06-01", call_url)


class TestAnthropic(TestCase):
    def test_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from underthesea.agent.providers.anthropic_provider import Anthropic
            with self.assertRaises(ValueError):
                Anthropic()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True)
    def test_basic_init(self):
        from underthesea.agent.providers.anthropic_provider import Anthropic
        p = Anthropic()
        self.assertEqual(p.name, "anthropic")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True)
    @patch(MOCK_POST, return_value=_anthropic_response("Xin chao!"))
    def test_chat(self, mock_post):
        from underthesea.agent.providers.anthropic_provider import Anthropic
        result = Anthropic().chat([
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ])
        self.assertEqual(result.content, "Xin chao!")
        # Verify system extracted
        call_body = mock_post.call_args[0][2]
        self.assertEqual(call_body["system"], "Be helpful.")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True)
    @patch(MOCK_POST, return_value=_anthropic_response(tool_calls=[
        {"id": "toolu_1", "name": "calc", "input": {"x": 1}}
    ]))
    def test_chat_with_tool_calls(self, mock_post):
        from underthesea.agent.providers.anthropic_provider import Anthropic
        result = Anthropic().chat(
            [{"role": "user", "content": "calc"}],
            tools=[{"type": "function", "function": {"name": "calc", "description": "Calc", "parameters": {}}}],
        )
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "calc")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}, clear=True)
    def test_tool_format_conversion(self):
        from underthesea.agent.providers.anthropic_provider import Anthropic
        p = Anthropic()
        converted = p._convert_tools([{
            "type": "function",
            "function": {"name": "s", "description": "d", "parameters": {"type": "object"}},
        }])
        self.assertEqual(converted[0]["name"], "s")
        self.assertIn("input_schema", converted[0])

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}, clear=True)
    def test_message_conversion(self):
        from underthesea.agent.providers.anthropic_provider import Anthropic
        p = Anthropic()
        converted = p._convert_messages([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": '{"r": 1}'},
        ])
        self.assertEqual(converted[1]["content"][0]["type"], "tool_use")
        self.assertEqual(converted[2]["content"][0]["type"], "tool_result")


class TestGemini(TestCase):
    def test_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from underthesea.agent.providers.gemini_provider import Gemini
            with self.assertRaises(ValueError):
                Gemini()

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_basic_init(self):
        from underthesea.agent.providers.gemini_provider import Gemini
        p = Gemini()
        self.assertEqual(p.name, "gemini")
        self.assertEqual(p.model, "gemini-2.0-flash")

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    @patch(MOCK_POST, return_value={
        "candidates": [{"content": {"parts": [{"text": "Hi!"}], "role": "model"}}]
    })
    def test_chat(self, mock_post):
        from underthesea.agent.providers.gemini_provider import Gemini
        result = Gemini().chat([{"role": "user", "content": "Hello"}])
        self.assertEqual(result.content, "Hi!")

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    @patch(MOCK_POST, return_value={
        "candidates": [{"content": {"parts": [
            {"functionCall": {"name": "calc", "args": {"x": 1}}}
        ], "role": "model"}}]
    })
    def test_chat_with_tool_calls(self, mock_post):
        from underthesea.agent.providers.gemini_provider import Gemini
        result = Gemini().chat(
            [{"role": "user", "content": "calc"}],
            tools=[{"type": "function", "function": {"name": "calc", "description": "Calc", "parameters": {}}}],
        )
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "calc")


class TestLLMAutoDetect(TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_detects_openai(self):
        from underthesea.agent import LLM
        llm = LLM()
        self.assertEqual(llm.provider, "openai")

    @patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "https://e"}, clear=True)
    def test_detects_azure(self):
        from underthesea.agent import LLM
        llm = LLM()
        self.assertEqual(llm.provider, "azure")

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "k", "AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "https://e"
    }, clear=True)
    def test_azure_preferred(self):
        from underthesea.agent import LLM
        self.assertEqual(LLM().provider, "azure")

    @patch.dict(os.environ, {}, clear=True)
    def test_no_credentials_raises(self):
        from underthesea.agent import LLM
        with self.assertRaises(ValueError):
            LLM()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True)
    @patch(MOCK_POST, return_value=_openai_response("Hi"))
    def test_chat_returns_str(self, mock_post):
        from underthesea.agent import LLM
        result = LLM().chat([{"role": "user", "content": "Hi"}])
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Hi")


class TestAgentWithProviders(TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True)
    @patch(MOCK_POST, return_value=_openai_response("Hello!"))
    def test_agent_with_openai(self, mock_post):
        from underthesea.agent import Agent
        from underthesea.agent.providers.openai_provider import OpenAI
        agent = Agent(name="t", provider=OpenAI())
        self.assertEqual(agent("Hi"), "Hello!")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True)
    @patch(MOCK_POST, return_value=_openai_response("Hello!"))
    def test_agent_with_llm(self, mock_post):
        from underthesea.agent import Agent, LLM
        agent = Agent(name="t", provider=LLM())
        self.assertEqual(agent("Hi"), "Hello!")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True)
    @patch(MOCK_POST, return_value=_openai_response("Hello!"))
    def test_agent_auto_detects(self, mock_post):
        from underthesea.agent import Agent
        agent = Agent(name="t")
        self.assertEqual(agent("Hi"), "Hello!")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True)
    @patch(MOCK_POST)
    def test_agent_tool_calling(self, mock_post):
        from underthesea.agent import Agent, Tool
        from underthesea.agent.providers.openai_provider import OpenAI

        mock_post.side_effect = [
            _openai_response(tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "HN"}'}}
            ]),
            _openai_response("Weather is 25C."),
        ]

        def get_weather(location: str) -> dict:
            """Get weather."""
            return {"temp": 25}

        agent = Agent(name="w", tools=[Tool(get_weather)], provider=OpenAI())
        self.assertEqual(agent("Weather?"), "Weather is 25C.")
        self.assertEqual(mock_post.call_count, 2)


class TestImports(TestCase):
    def test_all_imports(self):
        from underthesea.agent import (
            Agent, Anthropic, AzureOpenAI, BaseProvider, Gemini, LLM, OpenAI, Session, Tool,
        )
        for cls in [Agent, Anthropic, AzureOpenAI, BaseProvider, Gemini, LLM, OpenAI, Session, Tool]:
            self.assertIsNotNone(cls)
