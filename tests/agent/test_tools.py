import os
from unittest import TestCase
from unittest.mock import Mock, patch


class TestDefaultTools(TestCase):
    def test_default_tools_import(self):
        from underthesea.agent import (
            calculator_tool,
            core_tools,
            current_datetime_tool,
            default_tools,
            fetch_url_tool,
            json_parse_tool,
            list_directory_tool,
            python_tool,
            read_file_tool,
            shell_tool,
            string_length_tool,
            system_tools,
            web_search_tool,
            web_tools,
            wikipedia_tool,
            write_file_tool,
        )

        self.assertEqual(len(core_tools), 4)
        self.assertEqual(len(web_tools), 3)
        self.assertEqual(len(system_tools), 5)
        self.assertEqual(len(default_tools), 12)

        self.assertEqual(current_datetime_tool.name, "get_current_datetime")
        self.assertEqual(calculator_tool.name, "calculator")
        self.assertEqual(web_search_tool.name, "web_search")
        self.assertEqual(wikipedia_tool.name, "wikipedia")
        self.assertEqual(shell_tool.name, "shell")
        self.assertEqual(python_tool.name, "python")

    def test_current_datetime_tool(self):
        from underthesea.agent import current_datetime_tool

        result = current_datetime_tool()
        self.assertIn("datetime", result)
        self.assertIn("date", result)
        self.assertIn("time", result)
        self.assertIn("weekday", result)

    def test_calculator_tool(self):
        from underthesea.agent import calculator_tool

        result = calculator_tool(expression="2 + 3 * 4")
        self.assertEqual(result["result"], 14)

        result = calculator_tool(expression="sqrt(16)")
        self.assertEqual(result["result"], 4.0)

        result = calculator_tool(expression="pi * 2")
        self.assertAlmostEqual(result["result"], 6.283, places=2)

    def test_calculator_tool_error(self):
        from underthesea.agent import calculator_tool

        result = calculator_tool(expression="invalid")
        self.assertIn("error", result)

    def test_string_length_tool(self):
        from underthesea.agent import string_length_tool

        result = string_length_tool(text="Hello World")
        self.assertEqual(result["characters"], 11)
        self.assertEqual(result["words"], 2)

    def test_json_parse_tool(self):
        from underthesea.agent import json_parse_tool

        result = json_parse_tool(json_string='{"name": "test", "value": 123}')
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["name"], "test")

        result = json_parse_tool(json_string="invalid json")
        self.assertFalse(result["success"])

    def test_list_directory_tool(self):
        from underthesea.agent import list_directory_tool

        result = list_directory_tool(path=".")
        self.assertIn("files", result)
        self.assertIn("directories", result)

    def test_python_tool(self):
        from underthesea.agent import python_tool

        result = python_tool(code="print(2 + 2)")
        self.assertEqual(result["output"], "4")

        result = python_tool(code="for i in range(3): print(i)")
        self.assertIn("0", result["output"])
        self.assertIn("1", result["output"])
        self.assertIn("2", result["output"])

    def test_tools_to_openai_format(self):
        from underthesea.agent import default_tools

        for tool in default_tools:
            openai_format = tool.to_openai_tool()
            self.assertEqual(openai_format["type"], "function")
            self.assertIn("name", openai_format["function"])
            self.assertIn("description", openai_format["function"])
            self.assertIn("parameters", openai_format["function"])

    def test_create_agent_with_default_tools(self):
        from underthesea.agent import Agent, default_tools

        agent = Agent(
            name="assistant",
            tools=default_tools,
            instruction="You are a helpful assistant.",
        )

        self.assertEqual(agent.name, "assistant")
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
        self.assertEqual(tool.parameters["type"], "object")
        self.assertEqual(tool.parameters["properties"]["name"]["type"], "string")
        self.assertIn("name", tool.parameters["required"])

    def test_tool_custom_name_and_description(self):
        from underthesea.agent import Tool

        def my_func(x: int) -> int:
            return x * 2

        tool = Tool(my_func, name="doubler", description="Doubles a number")

        self.assertEqual(tool.name, "doubler")
        self.assertEqual(tool.description, "Doubles a number")

    def test_tool_parameter_types(self):
        from underthesea.agent import Tool

        def func_with_types(
            s: str, i: int, f: float, b: bool, lst: list
        ) -> dict:
            return {}

        tool = Tool(func_with_types)

        props = tool.parameters["properties"]
        self.assertEqual(props["s"]["type"], "string")
        self.assertEqual(props["i"]["type"], "integer")
        self.assertEqual(props["f"]["type"], "number")
        self.assertEqual(props["b"]["type"], "boolean")
        self.assertEqual(props["lst"]["type"], "array")

    def test_tool_optional_parameters(self):
        from underthesea.agent import Tool

        def func_with_default(required: str, optional: str = "default") -> str:
            return required + optional

        tool = Tool(func_with_default)

        self.assertIn("required", tool.parameters["required"])
        self.assertNotIn("optional", tool.parameters["required"])

    def test_tool_to_openai_format(self):
        from underthesea.agent import Tool

        def search(query: str) -> list:
            """Search for items."""
            return []

        tool = Tool(search)
        openai_format = tool.to_openai_tool()

        self.assertEqual(openai_format["type"], "function")
        self.assertEqual(openai_format["function"]["name"], "search")
        self.assertEqual(openai_format["function"]["description"], "Search for items.")
        self.assertIn("parameters", openai_format["function"])

    def test_tool_execute(self):
        from underthesea.agent import Tool

        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool(add)
        result = tool.execute({"a": 2, "b": 3})

        self.assertEqual(result, "5")

    def test_tool_execute_dict_result(self):
        from underthesea.agent import Tool

        def get_weather(location: str) -> dict:
            return {"location": location, "temp": 25}

        tool = Tool(get_weather)
        result = tool.execute({"location": "Hanoi"})

        self.assertIn("Hanoi", result)
        self.assertIn("25", result)

    def test_tool_callable(self):
        from underthesea.agent import Tool

        def multiply(x: int, y: int) -> int:
            return x * y

        tool = Tool(multiply)
        result = tool(x=3, y=4)

        self.assertEqual(result, 12)


class TestAgentWithTools(TestCase):
    def test_agent_creation_with_tools(self):
        from underthesea.agent import Agent, Tool

        def dummy_tool(x: str) -> str:
            return x

        agent = Agent(
            name="test_agent",
            tools=[Tool(dummy_tool)],
            instruction="Custom instruction",
        )

        self.assertEqual(agent.name, "test_agent")
        self.assertEqual(len(agent.tools), 1)
        self.assertEqual(agent.instruction, "Custom instruction")

    def test_agent_default_instruction(self):
        from underthesea.agent import Agent

        agent = Agent(name="test")

        self.assertEqual(agent.instruction, Agent.DEFAULT_INSTRUCTION)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_agent_without_tools(self, mock_openai):
        from underthesea.agent import Agent

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        agent = Agent(name="simple_agent")
        response = agent("Hi")

        self.assertEqual(response, "Hello!")
        self.assertEqual(len(agent.history), 2)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_agent_with_tools_no_tool_call(self, mock_openai):
        from underthesea.agent import Agent, Tool

        def get_time() -> str:
            return "12:00"

        mock_response = Mock()
        mock_message = Mock()
        mock_message.tool_calls = None
        mock_message.content = "I can help you with that!"
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = mock_message
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        agent = Agent(name="helper", tools=[Tool(get_time)])
        response = agent("Hello")

        self.assertEqual(response, "I can help you with that!")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_agent_with_tool_call(self, mock_openai):
        from underthesea.agent import Agent, Tool

        def get_weather(location: str) -> dict:
            """Get weather for a location."""
            return {"location": location, "temp": 25, "condition": "sunny"}

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "Hanoi"}'

        mock_message_with_tool = Mock()
        mock_message_with_tool.tool_calls = [mock_tool_call]

        mock_message_final = Mock()
        mock_message_final.tool_calls = None
        mock_message_final.content = "The weather in Hanoi is 25C and sunny."

        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message = mock_message_with_tool

        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message = mock_message_final

        mock_openai.return_value.chat.completions.create.side_effect = [
            mock_response_1,
            mock_response_2,
        ]

        agent = Agent(
            name="weather_agent",
            tools=[Tool(get_weather, description="Get weather for a city")],
        )
        response = agent("What's the weather in Hanoi?")

        self.assertEqual(response, "The weather in Hanoi is 25C and sunny.")
        self.assertEqual(mock_openai.return_value.chat.completions.create.call_count, 2)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_agent_reset(self, mock_openai):
        from underthesea.agent import Agent

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        agent = Agent(name="test")
        agent("Message 1")
        self.assertEqual(len(agent.history), 2)

        agent.reset()
        self.assertEqual(len(agent.history), 0)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_agent_max_iterations(self, mock_openai):
        from underthesea.agent import Agent, Tool

        def infinite_tool() -> str:
            return "result"

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "infinite_tool"
        mock_tool_call.function.arguments = "{}"

        mock_message = Mock()
        mock_message.tool_calls = [mock_tool_call]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = mock_message
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        agent = Agent(
            name="test",
            tools=[Tool(infinite_tool)],
            max_iterations=3,
        )

        with self.assertRaises(RuntimeError) as ctx:
            agent("Do something")

        self.assertIn("Max tool iterations reached", str(ctx.exception))

    def test_agent_history_is_copy(self):
        from underthesea.agent import Agent

        agent = Agent(name="test")
        agent._history = [{"role": "user", "content": "test"}]

        history = agent.history
        history.clear()

        self.assertEqual(len(agent.history), 1)
