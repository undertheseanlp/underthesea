import os
from unittest import TestCase
from unittest.mock import Mock, patch


class TestAgent(TestCase):
    def setUp(self):
        """Reset agent state before each test."""
        from underthesea import agent

        agent.reset()
        agent._llm = None
        agent._system_prompt = agent.DEFAULT_SYSTEM_PROMPT

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_agent_basic_call(self, mock_openai):
        from underthesea import agent

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Xin chào!"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        response = agent("Hello")

        self.assertEqual(response, "Xin chào!")
        self.assertEqual(len(agent.history), 2)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_agent_custom_model(self, mock_openai):
        from underthesea import agent

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        agent("Hello", model="gpt-4")

        # Verify model was passed to chat
        call_args = mock_openai.return_value.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs["model"], "gpt-4")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_agent_custom_system_prompt(self, mock_openai):
        from underthesea import agent

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        custom_prompt = "Bạn là trợ lý tiếng Việt"
        agent("Hello", system_prompt=custom_prompt)

        self.assertEqual(agent._system_prompt, custom_prompt)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_agent_reset(self, mock_openai):
        from underthesea import agent

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

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
    @patch("openai.OpenAI")
    def test_agent_history_is_copy(self, mock_openai):
        from underthesea import agent

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        agent("Test")

        history = agent.history
        history.clear()
        self.assertEqual(len(agent.history), 2)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_agent_conversation_flow(self, mock_openai):
        from underthesea import agent

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        mock_response.choices[0].message.content = "Xin chào!"
        response1 = agent("Hello")
        self.assertEqual(response1, "Xin chào!")

        mock_response.choices[0].message.content = "NLP là xử lý ngôn ngữ tự nhiên."
        response2 = agent("NLP là gì?")
        self.assertEqual(response2, "NLP là xử lý ngôn ngữ tự nhiên.")

        self.assertEqual(len(agent.history), 4)


class TestAgentAzure(TestCase):
    def setUp(self):
        """Reset agent state before each test."""
        from underthesea import agent

        agent.reset()
        agent._llm = None
        agent._system_prompt = agent.DEFAULT_SYSTEM_PROMPT

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "azure-test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        },
        clear=True,
    )
    @patch("openai.AzureOpenAI")
    def test_agent_azure_auto_detect(self, mock_azure):
        from underthesea import agent

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Azure response"
        mock_azure.return_value.chat.completions.create.return_value = mock_response

        response = agent("Hello")

        self.assertEqual(response, "Azure response")
        mock_azure.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"})
    @patch("openai.AzureOpenAI")
    def test_agent_azure_explicit_provider(self, mock_azure):
        from underthesea import agent

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Azure response"
        mock_azure.return_value.chat.completions.create.return_value = mock_response

        response = agent(
            "Hello",
            provider="azure",
            api_key="my-azure-key",
            azure_endpoint="https://my.openai.azure.com",
        )

        self.assertEqual(response, "Azure response")
        mock_azure.assert_called_once()
