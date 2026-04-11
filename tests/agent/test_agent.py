"""Tests for _AgentInstance (the global `agent` singleton)."""

import os
from unittest import TestCase
from unittest.mock import patch

MOCK_POST = "underthesea.agent.providers._http.post_json"


def _resp(content="Xin chào!"):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


class TestAgent(TestCase):
    def setUp(self):
        from underthesea import agent
        agent.reset()
        agent._llm = None
        agent._system_prompt = agent.DEFAULT_SYSTEM_PROMPT

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Xin chào!"))
    def test_agent_basic_call(self, mock_post):
        from underthesea import agent
        response = agent("Hello")
        self.assertEqual(response, "Xin chào!")
        self.assertEqual(len(agent.history), 2)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Response"))
    def test_agent_custom_model(self, mock_post):
        from underthesea import agent
        agent("Hello", model="gpt-4")
        call_body = mock_post.call_args[0][2]
        self.assertEqual(call_body["model"], "gpt-4")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Response"))
    def test_agent_custom_system_prompt(self, mock_post):
        from underthesea import agent
        agent("Hello", system_prompt="Bạn là trợ lý tiếng Việt")
        self.assertEqual(agent._system_prompt, "Bạn là trợ lý tiếng Việt")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Response"))
    def test_agent_reset(self, mock_post):
        from underthesea import agent
        agent("Test")
        self.assertEqual(len(agent.history), 2)
        agent.reset()
        self.assertEqual(len(agent.history), 0)

    @patch.dict(os.environ, {}, clear=True)
    def test_agent_missing_api_key(self):
        from underthesea import agent
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("AZURE_OPENAI_API_KEY", None)
        os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
        agent._llm = None
        with self.assertRaises(ValueError):
            agent("Hello")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Response"))
    def test_agent_history_is_copy(self, mock_post):
        from underthesea import agent
        agent("Test")
        history = agent.history
        history.clear()
        self.assertEqual(len(agent.history), 2)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST)
    def test_agent_conversation_flow(self, mock_post):
        from underthesea import agent
        mock_post.side_effect = [_resp("Xin chào!"), _resp("NLP là xử lý ngôn ngữ tự nhiên.")]
        r1 = agent("Hello")
        self.assertEqual(r1, "Xin chào!")
        r2 = agent("NLP là gì?")
        self.assertEqual(r2, "NLP là xử lý ngôn ngữ tự nhiên.")
        self.assertEqual(len(agent.history), 4)


class TestAgentAzure(TestCase):
    def setUp(self):
        from underthesea import agent
        agent.reset()
        agent._llm = None
        agent._system_prompt = agent.DEFAULT_SYSTEM_PROMPT

    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "azure-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
    }, clear=True)
    @patch(MOCK_POST, return_value=_resp("Azure response"))
    def test_agent_azure_auto_detect(self, mock_post):
        from underthesea import agent
        response = agent("Hello")
        self.assertEqual(response, "Azure response")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"})
    @patch(MOCK_POST, return_value=_resp("Azure response"))
    def test_agent_azure_explicit_provider(self, mock_post):
        from underthesea import agent
        response = agent(
            "Hello", provider="azure",
            api_key="my-azure-key", azure_endpoint="https://my.openai.azure.com",
        )
        self.assertEqual(response, "Azure response")
