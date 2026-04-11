"""Tests for LLM auto-detect wrapper."""

import os
from unittest import TestCase
from unittest.mock import patch

MOCK_POST = "underthesea.agent.providers._http.post_json"


def _openai_resp(content="Hello!"):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


class TestLLMOpenAI(TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_llm_openai_auto_detect(self):
        from underthesea.agent.llm import LLM
        llm = LLM()
        self.assertEqual(llm.provider, "openai")
        self.assertEqual(llm.model, "gpt-4o-mini")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4"}, clear=True)
    def test_llm_openai_model_from_env(self):
        from underthesea.agent.llm import LLM
        llm = LLM()
        self.assertEqual(llm.model, "gpt-4")

    @patch.dict(os.environ, {}, clear=True)
    def test_llm_openai_explicit_key(self):
        from underthesea.agent.llm import LLM
        llm = LLM(provider="openai", api_key="my-key")
        self.assertEqual(llm.provider, "openai")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_llm_openai_explicit_model(self):
        from underthesea.agent.llm import LLM
        llm = LLM(model="gpt-4-turbo")
        self.assertEqual(llm.model, "gpt-4-turbo")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    @patch(MOCK_POST, return_value=_openai_resp("Hello!"))
    def test_llm_chat(self, mock_post):
        from underthesea.agent.llm import LLM
        llm = LLM()
        response = llm.chat([{"role": "user", "content": "Hi"}])
        self.assertEqual(response, "Hello!")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    @patch(MOCK_POST, return_value=_openai_resp("Hello!"))
    def test_llm_chat_uses_default_model(self, mock_post):
        from underthesea.agent.llm import LLM
        llm = LLM()
        llm.chat([{"role": "user", "content": "Hi"}])
        call_body = mock_post.call_args[0][2]
        self.assertEqual(call_body["model"], "gpt-4o-mini")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    @patch(MOCK_POST, return_value=_openai_resp("Hello!"))
    def test_llm_chat_override_model(self, mock_post):
        from underthesea.agent.llm import LLM
        llm = LLM()
        llm.chat([{"role": "user", "content": "Hi"}], model="gpt-4")
        call_body = mock_post.call_args[0][2]
        self.assertEqual(call_body["model"], "gpt-4")

    @patch.dict(os.environ, {}, clear=True)
    def test_llm_no_credentials(self):
        from underthesea.agent.llm import LLM
        with self.assertRaises(ValueError) as ctx:
            LLM()
        self.assertIn("No API key found", str(ctx.exception))


class TestLLMAzure(TestCase):
    @patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "https://e"}, clear=True)
    def test_llm_azure_auto_detect(self):
        from underthesea.agent.llm import LLM
        llm = LLM()
        self.assertEqual(llm.provider, "azure")
        self.assertEqual(llm.model, "gpt-4o-mini")

    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "https://e",
        "AZURE_OPENAI_DEPLOYMENT": "my-gpt4-deployment",
    }, clear=True)
    def test_llm_azure_deployment_from_env(self):
        from underthesea.agent.llm import LLM
        llm = LLM()
        self.assertEqual(llm.model, "my-gpt4-deployment")

    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "https://e",
        "AZURE_OPENAI_API_VERSION": "2024-05-01-preview",
    }, clear=True)
    @patch(MOCK_POST, return_value=_openai_resp())
    def test_llm_azure_custom_version(self, mock_post):
        from underthesea.agent.llm import LLM
        llm = LLM()
        llm.chat([{"role": "user", "content": "Hi"}])
        call_url = mock_post.call_args[0][0]
        self.assertIn("2024-05-01-preview", call_url)

    @patch.dict(os.environ, {}, clear=True)
    @patch(MOCK_POST, return_value=_openai_resp())
    def test_llm_azure_explicit_params(self, mock_post):
        from underthesea.agent.llm import LLM
        llm = LLM(
            provider="azure", model="my-deployment",
            api_key="my-azure-key", azure_endpoint="https://my.openai.azure.com",
            azure_api_version="2024-06-01",
        )
        self.assertEqual(llm.provider, "azure")
        self.assertEqual(llm.model, "my-deployment")

    @patch.dict(os.environ, {}, clear=True)
    def test_llm_azure_missing_endpoint(self):
        from underthesea.agent.llm import LLM
        with self.assertRaises(ValueError) as ctx:
            LLM(provider="azure", api_key="my-key")
        self.assertIn("endpoint required", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_llm_azure_missing_key(self):
        from underthesea.agent.llm import LLM
        with self.assertRaises(ValueError) as ctx:
            LLM(provider="azure", azure_endpoint="https://test.openai.azure.com")
        self.assertIn("API key required", str(ctx.exception))


class TestLLMProviderPriority(TestCase):
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "openai-key",
        "AZURE_OPENAI_API_KEY": "azure-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
    }, clear=True)
    def test_azure_preferred_when_both_available(self):
        from underthesea.agent.llm import LLM
        llm = LLM()
        self.assertEqual(llm.provider, "azure")
