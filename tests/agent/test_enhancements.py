"""Tests for the v9.4+ agent enhancements:

- Parallel tool execution
- Tool error recovery (errors fed back to the model)
- Token usage tracking on the Agent
- ``@tool`` decorator and richer Tool schema extraction
- HTTP retry with exponential backoff
"""

from __future__ import annotations

import os
import threading
import time
from typing import Literal, Optional
from unittest import TestCase
from unittest.mock import patch

MOCK_POST = "underthesea.agent.providers._http.post_json"


def _resp(content="Hello!", *, usage=None):
    data = {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}
    if usage is not None:
        data["usage"] = usage
    return data


def _tool_call_resp(calls, *, usage=None):
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in calls
        ],
    }
    data = {"choices": [{"message": msg, "finish_reason": "tool_calls"}]}
    if usage is not None:
        data["usage"] = usage
    return data


# ---------------------------------------------------------------------------
# @tool decorator + Tool schema enhancements
# ---------------------------------------------------------------------------


class TestToolDecorator(TestCase):
    def test_bare_decorator(self):
        from underthesea.agent import Tool, tool

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        self.assertIsInstance(add, Tool)
        self.assertEqual(add.name, "add")
        self.assertEqual(add.description, "Add two numbers.")
        self.assertEqual(add(a=2, b=3), 5)

    def test_decorator_with_overrides(self):
        from underthesea.agent import tool

        @tool(name="adder", description="Sum two ints")
        def _impl(a: int, b: int) -> int:
            return a + b

        self.assertEqual(_impl.name, "adder")
        self.assertEqual(_impl.description, "Sum two ints")


class TestToolSchema(TestCase):
    def test_optional_type(self):
        from underthesea.agent import Tool

        def func(name: str, age: Optional[int] = None) -> dict:
            return {"name": name, "age": age}

        schema = Tool(func).parameters
        self.assertEqual(schema["properties"]["age"]["type"], "integer")
        self.assertNotIn("age", schema["required"])
        self.assertIn("name", schema["required"])

    def test_list_with_item_type(self):
        from underthesea.agent import Tool

        def func(tags: list[str]) -> int:
            return len(tags)

        schema = Tool(func).parameters
        self.assertEqual(schema["properties"]["tags"]["type"], "array")
        self.assertEqual(schema["properties"]["tags"]["items"], {"type": "string"})

    def test_dict_with_value_type(self):
        from underthesea.agent import Tool

        def func(counts: dict[str, int]) -> int:
            return sum(counts.values())

        schema = Tool(func).parameters
        prop = schema["properties"]["counts"]
        self.assertEqual(prop["type"], "object")
        self.assertEqual(prop["additionalProperties"], {"type": "integer"})

    def test_literal_becomes_enum(self):
        from underthesea.agent import Tool

        def func(mode: Literal["fast", "slow"]) -> str:
            return mode

        schema = Tool(func).parameters
        prop = schema["properties"]["mode"]
        self.assertEqual(prop["type"], "string")
        self.assertEqual(prop["enum"], ["fast", "slow"])

    def test_param_descriptions_parsed_from_docstring(self):
        from underthesea.agent import Tool

        def func(city: str, units: str = "metric") -> dict:
            """Get the weather forecast.

            Args:
                city: City to look up.
                units: Units system (metric or imperial).
            """
            return {}

        schema = Tool(func).parameters
        self.assertEqual(schema["properties"]["city"]["description"], "City to look up.")
        self.assertEqual(
            schema["properties"]["units"]["description"],
            "Units system (metric or imperial).",
        )

    def test_default_value_recorded(self):
        from underthesea.agent import Tool

        def func(lang: str = "vi") -> str:
            return lang

        schema = Tool(func).parameters
        self.assertEqual(schema["properties"]["lang"]["default"], "vi")
        self.assertNotIn("lang", schema["required"])


# ---------------------------------------------------------------------------
# Parallel tool execution
# ---------------------------------------------------------------------------


class TestParallelToolExecution(TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_parallel_tools_run_concurrently(self):
        from underthesea.agent import Agent, Tool

        barrier = threading.Barrier(2, timeout=5)

        def slow_a() -> str:
            barrier.wait()
            return "a"

        def slow_b() -> str:
            barrier.wait()
            return "b"

        agent = Agent(
            name="t",
            tools=[Tool(slow_a, name="slow_a"), Tool(slow_b, name="slow_b")],
            parallel_tools=True,
        )

        side = [
            _tool_call_resp([
                {"id": "1", "name": "slow_a", "arguments": "{}"},
                {"id": "2", "name": "slow_b", "arguments": "{}"},
            ]),
            _resp("done"),
        ]

        with patch(MOCK_POST, side_effect=side):
            result = agent("Go")
        # If execution were sequential, the barrier would deadlock and time out.
        self.assertEqual(result, "done")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_sequential_when_disabled(self):
        from underthesea.agent import Agent, Tool

        calls: list[str] = []

        def record(name: str) -> str:
            calls.append(name)
            return name

        agent = Agent(
            name="t",
            tools=[Tool(record)],
            parallel_tools=False,
        )

        side = [
            _tool_call_resp([
                {"id": "1", "name": "record", "arguments": '{"name": "a"}'},
                {"id": "2", "name": "record", "arguments": '{"name": "b"}'},
            ]),
            _resp("done"),
        ]

        with patch(MOCK_POST, side_effect=side):
            agent("Go")
        self.assertEqual(calls, ["a", "b"])

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_tool_results_order_preserved(self):
        from underthesea.agent import Agent, Tool

        def echo(text: str) -> str:
            # Vary the sleep so a, b, c return out of order if parallel.
            delays = {"a": 0.02, "b": 0.0, "c": 0.01}
            time.sleep(delays.get(text, 0))
            return text

        agent = Agent(name="t", tools=[Tool(echo)], parallel_tools=True)
        captured: list[list[dict]] = []

        def fake_post(url, headers, body, **kw):
            captured.append(body["messages"])
            if len(captured) == 1:
                return _tool_call_resp([
                    {"id": "1", "name": "echo", "arguments": '{"text": "a"}'},
                    {"id": "2", "name": "echo", "arguments": '{"text": "b"}'},
                    {"id": "3", "name": "echo", "arguments": '{"text": "c"}'},
                ])
            return _resp("ok")

        with patch(MOCK_POST, side_effect=fake_post):
            agent("Go")
        second_call_msgs = captured[1]
        tool_msgs = [m for m in second_call_msgs if m.get("role") == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["1", "2", "3"])


# ---------------------------------------------------------------------------
# Tool error recovery
# ---------------------------------------------------------------------------


class TestToolErrorRecovery(TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_recover_default_feeds_error_back(self):
        from underthesea.agent import Agent, Tool

        def fail() -> str:
            raise RuntimeError("boom")

        agent = Agent(name="t", tools=[Tool(fail)])
        sent_messages: list[list[dict]] = []

        def fake_post(url, headers, body, **kw):
            sent_messages.append(body["messages"])
            if len(sent_messages) == 1:
                return _tool_call_resp([{"id": "1", "name": "fail", "arguments": "{}"}])
            return _resp("Sorry, the tool failed.")

        with patch(MOCK_POST, side_effect=fake_post):
            response = agent("Run it")

        self.assertEqual(response, "Sorry, the tool failed.")
        tool_msgs = [m for m in sent_messages[1] if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("boom", tool_msgs[0]["content"])
        self.assertIn("error", tool_msgs[0]["content"])

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_raise_mode_propagates(self):
        from underthesea.agent import Agent, Tool

        def fail() -> str:
            raise RuntimeError("boom")

        agent = Agent(
            name="t",
            tools=[Tool(fail)],
            tool_error_handling="raise",
        )

        with patch(MOCK_POST, return_value=_tool_call_resp(
            [{"id": "1", "name": "fail", "arguments": "{}"}]
        )):
            with self.assertRaises(RuntimeError) as ctx:
                agent("Run it")
        self.assertIn("boom", str(ctx.exception))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_invalid_arguments_recovered(self):
        from underthesea.agent import Agent, Tool

        def add(a: int, b: int) -> int:
            return a + b

        agent = Agent(name="t", tools=[Tool(add)])

        side = [
            _tool_call_resp([{"id": "1", "name": "add", "arguments": "not-json"}]),
            _resp("got it"),
        ]
        with patch(MOCK_POST, side_effect=side):
            response = agent("compute")
        self.assertEqual(response, "got it")

    def test_invalid_tool_error_handling_rejected(self):
        from underthesea.agent import Agent
        with self.assertRaises(ValueError):
            Agent(name="t", tool_error_handling="ignore")


# ---------------------------------------------------------------------------
# Token usage tracking
# ---------------------------------------------------------------------------


class TestUsageTracking(TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_usage_tracked_for_simple_call(self):
        from underthesea.agent import Agent

        agent = Agent(name="t")
        usage = {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}

        with patch(MOCK_POST, return_value=_resp("hi", usage=usage)):
            agent("Hello")

        self.assertEqual(agent.last_usage["input_tokens"], 10)
        self.assertEqual(agent.last_usage["output_tokens"], 7)
        self.assertEqual(agent.last_usage["total_tokens"], 17)
        self.assertEqual(agent.total_usage["total_tokens"], 17)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_usage_accumulates_across_tool_iterations(self):
        from underthesea.agent import Agent, Tool

        agent = Agent(name="t", tools=[Tool(lambda: "x", name="x")])
        usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

        side = [
            _tool_call_resp([{"id": "1", "name": "x", "arguments": "{}"}], usage=usage),
            _resp("done", usage=usage),
        ]
        with patch(MOCK_POST, side_effect=side):
            agent("Go")

        self.assertEqual(agent.last_usage["total_tokens"], 16)
        self.assertEqual(agent.total_usage["total_tokens"], 16)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_last_usage_resets_per_call_but_total_keeps_growing(self):
        from underthesea.agent import Agent

        agent = Agent(name="t")
        usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}

        with patch(MOCK_POST, return_value=_resp("a", usage=usage)):
            agent("First")
            agent("Second")

        self.assertEqual(agent.last_usage["total_tokens"], 5)
        self.assertEqual(agent.total_usage["total_tokens"], 10)


# ---------------------------------------------------------------------------
# HTTP retry with exponential backoff
# ---------------------------------------------------------------------------


class TestHttpRetry(TestCase):
    def test_retries_on_5xx(self):
        import urllib.error
        from io import BytesIO

        from underthesea.agent.providers import _http

        attempts: list[int] = []

        def fake_urlopen(req, timeout=None):
            attempts.append(1)
            if len(attempts) < 3:
                err = urllib.error.HTTPError(
                    url=req.full_url, code=503, msg="busy",
                    hdrs=None, fp=BytesIO(b"upstream busy"),
                )
                raise err

            class _OK:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *a):
                    return False

                def read(self_inner):
                    return b'{"ok": true}'

            return _OK()

        with patch.object(_http.urllib.request, "urlopen", side_effect=fake_urlopen), \
                patch.object(_http.time, "sleep") as mock_sleep:
            result = _http.post_json(
                "http://x", {}, {}, max_retries=3, base_delay=0.0, max_delay=0.0,
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(attempts), 3)
        # Slept twice (between attempts 1->2 and 2->3).
        self.assertEqual(mock_sleep.call_count, 2)

    def test_does_not_retry_on_4xx(self):
        import urllib.error
        from io import BytesIO

        from underthesea.agent.providers import _http
        from underthesea.agent.providers._http import LLMError

        attempts: list[int] = []

        def fake_urlopen(req, timeout=None):
            attempts.append(1)
            raise urllib.error.HTTPError(
                url=req.full_url, code=401, msg="unauth",
                hdrs=None, fp=BytesIO(b"bad key"),
            )

        with patch.object(_http.urllib.request, "urlopen", side_effect=fake_urlopen), \
                patch.object(_http.time, "sleep"):
            with self.assertRaises(LLMError) as ctx:
                _http.post_json("http://x", {}, {}, max_retries=3, base_delay=0.0)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(len(attempts), 1)

    def test_gives_up_after_max_retries(self):
        import urllib.error
        from io import BytesIO

        from underthesea.agent.providers import _http
        from underthesea.agent.providers._http import LLMError

        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(
                url=req.full_url, code=503, msg="busy",
                hdrs=None, fp=BytesIO(b"still busy"),
            )

        with patch.object(_http.urllib.request, "urlopen", side_effect=fake_urlopen), \
                patch.object(_http.time, "sleep"):
            with self.assertRaises(LLMError):
                _http.post_json(
                    "http://x", {}, {}, max_retries=2, base_delay=0.0, max_delay=0.0,
                )
