"""Tests for Tool class, default tools, and Agent with tools."""

import os
from unittest import TestCase
from unittest.mock import patch

MOCK_POST = "underthesea.agent.providers._http.post_json"


def _resp(content="Hello!"):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


class TestDefaultTools(TestCase):
    def test_default_tools_import(self):
        from underthesea.agent import (
            calculator_tool, core_tools, current_datetime_tool, default_tools,
            fetch_url_tool, json_parse_tool, list_directory_tool, python_tool,
            read_file_tool, shell_tool, string_length_tool, system_tools,
            web_search_tool, web_tools, wikipedia_tool, write_file_tool,
        )
        self.assertEqual(len(core_tools), 4)
        self.assertEqual(len(web_tools), 3)
        self.assertEqual(len(system_tools), 5)
        self.assertEqual(len(default_tools), 12)

    def test_current_datetime_tool(self):
        from underthesea.agent import current_datetime_tool
        result = current_datetime_tool()
        self.assertIn("datetime", result)
        self.assertIn("date", result)

    def test_calculator_tool(self):
        from underthesea.agent import calculator_tool
        self.assertEqual(calculator_tool(expression="2 + 3 * 4")["result"], 14)
        self.assertAlmostEqual(calculator_tool(expression="pi * 2")["result"], 6.283, places=2)

    def test_calculator_tool_error(self):
        from underthesea.agent import calculator_tool
        self.assertIn("error", calculator_tool(expression="invalid"))

    def test_string_length_tool(self):
        from underthesea.agent import string_length_tool
        r = string_length_tool(text="Hello World")
        self.assertEqual(r["characters"], 11)
        self.assertEqual(r["words"], 2)

    def test_json_parse_tool(self):
        from underthesea.agent import json_parse_tool
        r = json_parse_tool(json_string='{"name": "test"}')
        self.assertTrue(r["success"])
        self.assertFalse(json_parse_tool(json_string="bad")["success"])

    def test_list_directory_tool(self):
        from underthesea.agent import list_directory_tool
        r = list_directory_tool(path=".")
        self.assertIn("files", r)
        self.assertIn("directories", r)

    def test_python_tool(self):
        from underthesea.agent import python_tool
        self.assertEqual(python_tool(code="print(2 + 2)")["output"], "4")

    def test_tools_to_openai_format(self):
        from underthesea.agent import default_tools
        for tool in default_tools:
            fmt = tool.to_openai_tool()
            self.assertEqual(fmt["type"], "function")
            self.assertIn("name", fmt["function"])

    def test_create_agent_with_default_tools(self):
        from underthesea.agent import Agent, default_tools
        agent = Agent(name="assistant", tools=default_tools, instruction="You are helpful.")
        self.assertEqual(len(agent.tools), 12)


class TestTool(TestCase):
    def test_tool_from_function(self):
        from underthesea.agent import Tool

        def greet(name: str) -> str:
            """Greet a person."""
            return f"Hello, {name}!"

        tool = Tool(greet)
        self.assertEqual(tool.name, "greet")
        self.assertEqual(tool.description, "Greet a person.")
        self.assertIn("name", tool.parameters["required"])

    def test_tool_custom_name_and_description(self):
        from underthesea.agent import Tool
        tool = Tool(lambda x: x * 2, name="doubler", description="Doubles")
        self.assertEqual(tool.name, "doubler")

    def test_tool_parameter_types(self):
        from underthesea.agent import Tool

        def func(s: str, i: int, f: float, b: bool, lst: list) -> dict:
            return {}

        props = Tool(func).parameters["properties"]
        self.assertEqual(props["s"]["type"], "string")
        self.assertEqual(props["i"]["type"], "integer")
        self.assertEqual(props["f"]["type"], "number")
        self.assertEqual(props["b"]["type"], "boolean")
        self.assertEqual(props["lst"]["type"], "array")

    def test_tool_optional_parameters(self):
        from underthesea.agent import Tool

        def func(required: str, optional: str = "default") -> str:
            return required + optional

        tool = Tool(func)
        self.assertIn("required", tool.parameters["required"])
        self.assertNotIn("optional", tool.parameters["required"])

    def test_tool_to_openai_format(self):
        from underthesea.agent import Tool

        def search(query: str) -> list:
            """Search for items."""
            return []

        fmt = Tool(search).to_openai_tool()
        self.assertEqual(fmt["type"], "function")
        self.assertEqual(fmt["function"]["name"], "search")

    def test_tool_to_anthropic_format(self):
        from underthesea.agent import Tool

        def search(query: str) -> list:
            """Search."""
            return []

        fmt = Tool(search).to_anthropic_tool()
        self.assertEqual(fmt["name"], "search")
        self.assertIn("input_schema", fmt)

    def test_tool_execute(self):
        from underthesea.agent import Tool
        tool = Tool(lambda a, b: a + b)
        self.assertEqual(tool.execute({"a": 2, "b": 3}), "5")

    def test_tool_callable(self):
        from underthesea.agent import Tool
        tool = Tool(lambda x, y: x * y)
        self.assertEqual(tool(x=3, y=4), 12)


class TestAgentWithTools(TestCase):
    def test_agent_creation_with_tools(self):
        from underthesea.agent import Agent, Tool
        agent = Agent(name="t", tools=[Tool(lambda x: x)], instruction="Custom")
        self.assertEqual(len(agent.tools), 1)
        self.assertEqual(agent.instruction, "Custom")

    def test_agent_default_instruction(self):
        from underthesea.agent import Agent
        self.assertEqual(Agent(name="t").instruction, Agent.DEFAULT_INSTRUCTION)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Hello!"))
    def test_agent_without_tools(self, mock_post):
        from underthesea.agent import Agent
        self.assertEqual(Agent(name="t")("Hi"), "Hello!")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value={"choices": [{"message": {
        "role": "assistant", "content": "I can help!", "tool_calls": None
    }, "finish_reason": "stop"}]})
    def test_agent_with_tools_no_tool_call(self, mock_post):
        from underthesea.agent import Agent, Tool
        agent = Agent(name="t", tools=[Tool(lambda: "12:00")])
        self.assertEqual(agent("Hello"), "I can help!")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST)
    def test_agent_with_tool_call(self, mock_post):
        from underthesea.agent import Agent, Tool

        def get_weather(location: str) -> dict:
            """Get weather."""
            return {"location": location, "temp": 25, "condition": "sunny"}

        mock_post.side_effect = [
            {"choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Hanoi"}'}}]
            }, "finish_reason": "tool_calls"}]},
            _resp("The weather in Hanoi is 25C and sunny."),
        ]

        agent = Agent(name="w", tools=[Tool(get_weather, description="Get weather")])
        self.assertEqual(agent("Weather in Hanoi?"), "The weather in Hanoi is 25C and sunny.")
        self.assertEqual(mock_post.call_count, 2)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Response"))
    def test_agent_reset(self, mock_post):
        from underthesea.agent import Agent
        agent = Agent(name="t")
        agent("Msg 1")
        self.assertEqual(len(agent.history), 2)
        agent.reset()
        self.assertEqual(len(agent.history), 0)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST)
    def test_agent_max_iterations(self, mock_post):
        from underthesea.agent import Agent, Tool

        mock_post.return_value = {"choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "inf", "arguments": "{}"}}]
        }, "finish_reason": "tool_calls"}]}

        agent = Agent(name="t", tools=[Tool(lambda: "r", name="inf")], max_iterations=3)
        with self.assertRaises(RuntimeError) as ctx:
            agent("Do something")
        self.assertIn("Max tool iterations reached", str(ctx.exception))

    def test_agent_history_is_copy(self):
        from underthesea.agent import Agent
        agent = Agent(name="t")
        agent._history = [{"role": "user", "content": "test"}]
        history = agent.history
        history.clear()
        self.assertEqual(len(agent.history), 1)
