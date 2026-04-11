"""Tests for the agent trace module (LocalTracer, LangfuseTracer, Agent integration)."""

import json
import os
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

MOCK_POST = "underthesea.agent.providers._http.post_json"


def _resp(content="Hello!", tool_calls=None):
    """Build a mock OpenAI-style response."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def _tool_call_resp(name, arguments, content=None, call_id="call_1"):
    """Build a mock response with a tool call."""
    return _resp(
        content=content,
        tool_calls=[{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    )


# ======================================================================
# BaseTracer / extract_usage
# ======================================================================


class TestExtractUsage(TestCase):
    def test_openai_usage(self):
        from underthesea.agent.trace.base import extract_usage
        raw = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        usage = extract_usage(raw)
        self.assertEqual(usage, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    def test_anthropic_usage(self):
        from underthesea.agent.trace.base import extract_usage
        raw = {"usage": {"input_tokens": 20, "output_tokens": 10}}
        usage = extract_usage(raw)
        self.assertEqual(usage["input_tokens"], 20)
        self.assertEqual(usage["output_tokens"], 10)

    def test_gemini_usage(self):
        from underthesea.agent.trace.base import extract_usage
        raw = {"usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 15, "totalTokenCount": 45}}
        usage = extract_usage(raw)
        self.assertEqual(usage, {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45})

    def test_none_input(self):
        from underthesea.agent.trace.base import extract_usage
        self.assertIsNone(extract_usage(None))
        self.assertIsNone(extract_usage({}))
        self.assertIsNone(extract_usage("not a dict"))


# ======================================================================
# LocalTracer
# ======================================================================


class TestLocalTracer(TestCase):
    def setUp(self):
        self.trace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.trace_dir, ignore_errors=True)

    def _make_tracer(self, console=False):
        from underthesea.agent.trace.local import LocalTracer
        return LocalTracer(trace_dir=self.trace_dir, console=console)

    def test_basic_trace_lifecycle(self):
        tracer = self._make_tracer()
        tid = tracer.start_trace(name="test", input="hello")
        self.assertIsInstance(tid, str)
        self.assertEqual(len(tid), 12)

        tracer.end_trace(tid, output="world")

        # Verify file was created
        trace = tracer.get_trace(tid)
        self.assertIsNotNone(trace)
        self.assertEqual(trace["name"], "test")
        self.assertEqual(trace["input"], "hello")
        self.assertEqual(trace["output"], "world")
        self.assertEqual(trace["status"], "ok")
        self.assertIn("duration_ms", trace)

    def test_generation_span(self):
        tracer = self._make_tracer()
        tid = tracer.start_trace(name="test", input="hi")

        sid = tracer.start_generation(tid, name="llm.chat", model="gpt-4")
        tracer.end_generation(
            sid, output="response",
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        tracer.end_trace(tid, output="done")

        trace = tracer.get_trace(tid)
        self.assertEqual(len(trace["spans"]), 1)
        span = trace["spans"][0]
        self.assertEqual(span["type"], "generation")
        self.assertEqual(span["model"], "gpt-4")
        self.assertEqual(span["usage"]["input_tokens"], 100)
        self.assertIn("duration_ms", span)

    def test_tool_span(self):
        tracer = self._make_tracer()
        tid = tracer.start_trace(name="test", input="calc")

        sid = tracer.start_span(tid, name="tool.calculator", input={"expr": "1+1"})
        tracer.end_span(sid, output="2")
        tracer.end_trace(tid, output="done")

        trace = tracer.get_trace(tid)
        self.assertEqual(len(trace["spans"]), 1)
        span = trace["spans"][0]
        self.assertEqual(span["type"], "span")
        self.assertEqual(span["name"], "tool.calculator")
        self.assertEqual(span["input"], {"expr": "1+1"})
        self.assertEqual(span["output"], "2")

    def test_error_trace(self):
        tracer = self._make_tracer()
        tid = tracer.start_trace(name="test", input="fail")
        tracer.end_trace(tid, status="error", error="something broke")

        trace = tracer.get_trace(tid)
        self.assertEqual(trace["status"], "error")
        self.assertEqual(trace["error"], "something broke")

    def test_list_traces(self):
        tracer = self._make_tracer()
        for i in range(3):
            tid = tracer.start_trace(name=f"test-{i}", input=f"msg-{i}")
            tracer.end_trace(tid, output=f"out-{i}")

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 3)

    def test_list_traces_limit(self):
        tracer = self._make_tracer()
        for i in range(5):
            tid = tracer.start_trace(name=f"test-{i}", input=f"msg-{i}")
            tracer.end_trace(tid, output=f"out-{i}")

        traces = tracer.list_traces(limit=2)
        self.assertEqual(len(traces), 2)

    def test_get_nonexistent_trace(self):
        tracer = self._make_tracer()
        self.assertIsNone(tracer.get_trace("nonexistent"))

    def test_print_trace(self):
        tracer = self._make_tracer()
        tid = tracer.start_trace(name="print-test", input="hello")
        sid = tracer.start_generation(tid, name="llm.chat", model="gpt-4")
        tracer.end_generation(
            sid, output="response",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        tracer.end_trace(tid, output="done")

        # Should not raise
        trace = tracer.get_trace(tid)
        tracer.print_trace(trace)
        tracer.print_trace(tid)

    def test_console_output(self):
        """Console mode should not raise."""
        tracer = self._make_tracer(console=True)
        tid = tracer.start_trace(name="console-test", input="hello")
        sid = tracer.start_generation(tid, name="llm.chat", model="gpt-4")
        tracer.end_generation(sid, output="response")
        sid2 = tracer.start_span(tid, name="tool.calc", input={"x": 1})
        tracer.end_span(sid2, output="result")
        tracer.end_trace(tid, output="done")

    def test_multiple_spans(self):
        tracer = self._make_tracer()
        tid = tracer.start_trace(name="multi", input="test")

        g1 = tracer.start_generation(tid, name="llm #1", model="gpt-4")
        tracer.end_generation(g1, output={"content": None, "tool_calls": [{"name": "calc"}]})

        t1 = tracer.start_span(tid, name="tool.calc", input={"expr": "2+2"})
        tracer.end_span(t1, output="4")

        g2 = tracer.start_generation(tid, name="llm #2", model="gpt-4")
        tracer.end_generation(g2, output="The answer is 4")

        tracer.end_trace(tid, output="The answer is 4")

        trace = tracer.get_trace(tid)
        self.assertEqual(len(trace["spans"]), 3)
        self.assertEqual(trace["spans"][0]["type"], "generation")
        self.assertEqual(trace["spans"][1]["type"], "span")
        self.assertEqual(trace["spans"][2]["type"], "generation")


# ======================================================================
# LangfuseTracer (mocked)
# ======================================================================


def _make_langfuse_tracer():
    """Create a LangfuseTracer with a mocked langfuse backend."""
    from underthesea.agent.trace.langfuse_tracer import LangfuseTracer
    mock_instance = MagicMock()
    tracer = LangfuseTracer.__new__(LangfuseTracer)
    tracer._langfuse = mock_instance
    tracer._traces = {}
    tracer._spans = {}
    return tracer, mock_instance


class TestLangfuseTracer(TestCase):
    def test_start_end_trace(self):
        tracer, mock_lf = _make_langfuse_tracer()
        mock_lf.create_trace_id.return_value = "trace-123"

        mock_obs = MagicMock()
        mock_lf.start_observation.return_value = mock_obs

        tid = tracer.start_trace(name="test", input="hello")
        self.assertEqual(tid, "trace-123")
        mock_lf.start_observation.assert_called_once()

        tracer.end_trace(tid, output="world")
        mock_obs.update.assert_called_once()
        mock_obs.end.assert_called_once()
        mock_lf.flush.assert_called_once()

    def test_generation(self):
        tracer, mock_lf = _make_langfuse_tracer()

        mock_parent = MagicMock()
        tracer._traces["trace-1"] = mock_parent

        mock_gen = MagicMock()
        mock_gen.id = "gen-1"
        mock_parent.start_observation.return_value = mock_gen

        sid = tracer.start_generation("trace-1", name="llm.chat", model="gpt-4")
        self.assertEqual(sid, "gen-1")
        mock_parent.start_observation.assert_called_once_with(
            name="llm.chat", as_type="generation", input=None, model="gpt-4", metadata=None,
        )

        tracer.end_generation(
            sid, output="response",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        mock_gen.update.assert_called_once()
        mock_gen.end.assert_called_once()
        call_kwargs = mock_gen.update.call_args[1]
        self.assertEqual(call_kwargs["usage_details"]["input"], 10)

    def test_span(self):
        tracer, mock_lf = _make_langfuse_tracer()

        mock_parent = MagicMock()
        tracer._traces["trace-1"] = mock_parent

        mock_span = MagicMock()
        mock_span.id = "span-1"
        mock_parent.start_observation.return_value = mock_span

        sid = tracer.start_span("trace-1", name="tool.calc", input={"x": 1})
        self.assertEqual(sid, "span-1")
        mock_parent.start_observation.assert_called_once_with(
            name="tool.calc", as_type="tool", input={"x": 1}, metadata=None,
        )

        tracer.end_span(sid, output="2")
        mock_span.update.assert_called_once()
        mock_span.end.assert_called_once()

    def test_unknown_trace_id_raises(self):
        tracer, _ = _make_langfuse_tracer()
        with self.assertRaises(ValueError):
            tracer.start_generation("unknown", name="test")
        with self.assertRaises(ValueError):
            tracer.start_span("unknown", name="test")

    def test_error_generation(self):
        tracer, _ = _make_langfuse_tracer()
        mock_parent = MagicMock()
        tracer._traces["trace-1"] = mock_parent

        mock_gen = MagicMock()
        mock_gen.id = "gen-err"
        mock_parent.start_observation.return_value = mock_gen

        sid = tracer.start_generation("trace-1", name="llm.chat", model="gpt-4")
        tracer.end_generation(sid, status="error", error="timeout")
        call_kwargs = mock_gen.update.call_args[1]
        self.assertEqual(call_kwargs["level"], "ERROR")

    def test_flush_and_shutdown(self):
        tracer, mock_lf = _make_langfuse_tracer()
        tracer.flush()
        mock_lf.flush.assert_called_once()
        tracer.shutdown()
        mock_lf.shutdown.assert_called_once()


# ======================================================================
# Agent integration with tracer
# ======================================================================


class TestAgentWithTracer(TestCase):
    def setUp(self):
        self.trace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.trace_dir, ignore_errors=True)

    def _make_tracer(self):
        from underthesea.agent.trace.local import LocalTracer
        return LocalTracer(trace_dir=self.trace_dir, console=False)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Hello world!"))
    def test_agent_class_with_tracer(self, mock_post):
        from underthesea.agent import Agent
        tracer = self._make_tracer()
        bot = Agent("test-bot", tracer=tracer)
        result = bot("Hi")
        self.assertEqual(result, "Hello world!")

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(trace["name"], "test-bot")
        self.assertEqual(trace["input"], "Hi")
        self.assertEqual(trace["output"], "Hello world!")
        self.assertEqual(trace["status"], "ok")
        # Should have one generation span
        self.assertEqual(len(trace["spans"]), 1)
        self.assertEqual(trace["spans"][0]["type"], "generation")
        self.assertEqual(trace["spans"][0]["usage"]["input_tokens"], 100)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, side_effect=[
        _tool_call_resp("calculator", {"expression": "2+2"}),
        _resp("The answer is 4"),
    ])
    def test_agent_with_tools_and_tracer(self, mock_post):
        from underthesea.agent import Agent, calculator_tool
        tracer = self._make_tracer()
        bot = Agent("calc-bot", tools=[calculator_tool], tracer=tracer)
        result = bot("What is 2+2?")
        self.assertEqual(result, "The answer is 4")

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        # 2 generations + 1 tool call = 3 spans
        self.assertEqual(len(trace["spans"]), 3)
        types = [s["type"] for s in trace["spans"]]
        self.assertEqual(types, ["generation", "span", "generation"])
        # Tool span
        tool_span = trace["spans"][1]
        self.assertEqual(tool_span["name"], "tool.calculator")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, side_effect=RuntimeError("API error"))
    def test_agent_error_traced(self, mock_post):
        from underthesea.agent import Agent
        tracer = self._make_tracer()
        bot = Agent("error-bot", tracer=tracer)
        with self.assertRaises(RuntimeError):
            bot("fail")

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["status"], "error")
        self.assertIn("API error", traces[0]["error"])

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("Chao ban!"))
    def test_agent_singleton_with_tracer(self, mock_post):
        from underthesea import agent
        agent.reset()
        agent._llm = None
        tracer = self._make_tracer()
        result = agent("Xin chao", tracer=tracer)
        self.assertEqual(result, "Chao ban!")

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["input"], "Xin chao")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("No trace"))
    def test_agent_without_tracer(self, mock_post):
        """Agent should work normally when no tracer is set."""
        from underthesea.agent import Agent
        bot = Agent("no-trace")
        result = bot("Hi")
        self.assertEqual(result, "No trace")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("traced"))
    def test_tracer_property(self, mock_post):
        from underthesea.agent import Agent
        tracer = self._make_tracer()
        bot = Agent("test", tracer=tracer)
        self.assertIs(bot.tracer, tracer)

        bot.tracer = None
        self.assertIsNone(bot.tracer)


# ======================================================================
# @trace decorator
# ======================================================================


class TestTraceDecorator(TestCase):
    def setUp(self):
        self.trace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.trace_dir, ignore_errors=True)

    def _make_tracer(self):
        from underthesea.agent.trace.local import LocalTracer
        return LocalTracer(trace_dir=self.trace_dir, console=False)

    def test_basic_decorator(self):
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def greet(name: str) -> str:
            return f"Hello {name}"

        result = greet("World")
        self.assertEqual(result, "Hello World")

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["name"], "greet")
        self.assertEqual(traces[0]["output"], "Hello World")
        self.assertEqual(traces[0]["status"], "ok")

    def test_custom_name(self):
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer, name="my-custom-step")
        def step():
            return 42

        step()
        traces = tracer.list_traces()
        self.assertEqual(traces[0]["name"], "my-custom-step")

    def test_input_captured(self):
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def add(a: int, b: int) -> int:
            return a + b

        add(3, 7)
        traces = tracer.list_traces()
        input_data = json.loads(traces[0]["input"])
        self.assertEqual(input_data["a"], 3)
        self.assertEqual(input_data["b"], 7)

    def test_error_traced(self):
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def fail():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            fail()

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["status"], "error")
        self.assertIn("boom", traces[0]["error"])

    def test_nested_creates_spans(self):
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def inner_a(x):
            return x * 2

        @trace(tracer)
        def inner_b(x):
            return x + 1

        @trace(tracer)
        def outer(x):
            a = inner_a(x)
            b = inner_b(a)
            return b

        result = outer(5)
        self.assertEqual(result, 11)  # (5*2)+1

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 1)  # Only one trace, not three
        t = traces[0]
        self.assertEqual(t["name"], "outer")
        self.assertEqual(len(t["spans"]), 2)
        self.assertEqual(t["spans"][0]["name"], "inner_a")
        self.assertEqual(t["spans"][1]["name"], "inner_b")

    def test_nested_error_in_span(self):
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def bad_step():
            raise RuntimeError("inner fail")

        @trace(tracer)
        def pipeline():
            return bad_step()

        with self.assertRaises(RuntimeError):
            pipeline()

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 1)
        t = traces[0]
        self.assertEqual(t["status"], "error")
        self.assertEqual(len(t["spans"]), 1)
        self.assertEqual(t["spans"][0]["status"], "error")
        self.assertIn("inner fail", t["spans"][0]["error"])

    def test_separate_calls_create_separate_traces(self):
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def work(x):
            return x

        work(1)
        work(2)
        traces = tracer.list_traces()
        self.assertEqual(len(traces), 2)

    def test_preserves_function_metadata(self):
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def documented_func():
            """This is my docstring."""
            return True

        self.assertEqual(documented_func.__name__, "documented_func")
        self.assertEqual(documented_func.__doc__, "This is my docstring.")

    def test_kwargs_captured(self):
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def configure(host: str = "localhost", port: int = 8080):
            return f"{host}:{port}"

        configure(port=3000)
        traces = tracer.list_traces()
        input_data = json.loads(traces[0]["input"])
        self.assertEqual(input_data["host"], "localhost")
        self.assertEqual(input_data["port"], 3000)

    def test_return_none(self):
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def noop():
            pass

        result = noop()
        self.assertIsNone(result)
        traces = tracer.list_traces()
        self.assertEqual(traces[0]["output"], "None")


# ======================================================================
# Agent inherits @trace context (no explicit tracer)
# ======================================================================


class TestAgentInheritsTraceContext(TestCase):
    def setUp(self):
        self.trace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.trace_dir, ignore_errors=True)

    def _make_tracer(self):
        from underthesea.agent.trace.local import LocalTracer
        return LocalTracer(trace_dir=self.trace_dir, console=False)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("inherited!"))
    def test_agent_inherits_trace_context(self, mock_post):
        """Agent without explicit tracer should add spans to the @trace context."""
        from underthesea.agent import Agent
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def pipeline(msg):
            bot = Agent("inner-bot")  # no tracer set
            return bot(msg)

        result = pipeline("hello")
        self.assertEqual(result, "inherited!")

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 1)
        t = traces[0]
        self.assertEqual(t["name"], "pipeline")
        # Agent's LLM call should appear as a span
        self.assertTrue(len(t["spans"]) >= 1)
        gen_spans = [s for s in t["spans"] if s["type"] == "generation"]
        self.assertTrue(len(gen_spans) >= 1)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, side_effect=[
        _tool_call_resp("calculator", {"expression": "3+3"}),
        _resp("The answer is 6"),
    ])
    def test_agent_with_tools_inherits_context(self, mock_post):
        """Agent with tools should create tool + generation spans in @trace."""
        from underthesea.agent import Agent, calculator_tool
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def pipeline(msg):
            bot = Agent("calc-bot", tools=[calculator_tool])
            return bot(msg)

        result = pipeline("3+3?")
        self.assertEqual(result, "The answer is 6")

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 1)
        t = traces[0]
        self.assertEqual(t["name"], "pipeline")
        # 2 generations + 1 tool = 3 spans
        self.assertEqual(len(t["spans"]), 3)
        types = [s["type"] for s in t["spans"]]
        self.assertEqual(types, ["generation", "span", "generation"])

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch(MOCK_POST, return_value=_resp("response"))
    def test_agent_singleton_inherits_context(self, mock_post):
        """The global `agent()` singleton also inherits @trace context."""
        from underthesea import agent
        from underthesea.agent.trace import trace
        agent.reset()
        agent._llm = None
        tracer = self._make_tracer()

        @trace(tracer)
        def pipeline(msg):
            return agent(msg)

        result = pipeline("hi")
        self.assertEqual(result, "response")

        traces = tracer.list_traces()
        self.assertEqual(len(traces), 1)
        t = traces[0]
        self.assertEqual(t["name"], "pipeline")
        gen_spans = [s for s in t["spans"] if s["type"] == "generation"]
        self.assertTrue(len(gen_spans) >= 1)

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "UNDERTHESEA_TRACE_DISABLED": "1",
    })
    @patch(MOCK_POST, return_value=_resp("no context"))
    def test_agent_no_trace_when_disabled(self, mock_post):
        """Agent with tracing disabled should not trace."""
        from underthesea.agent import Agent
        tracer = self._make_tracer()
        bot = Agent("solo")
        result = bot("hi")
        self.assertEqual(result, "no context")
        self.assertEqual(len(tracer.list_traces()), 0)


# ======================================================================
# Env var toggle: UNDERTHESEA_TRACE_DISABLED
# ======================================================================


class TestTracingDisabledEnvVar(TestCase):
    def setUp(self):
        self.trace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.trace_dir, ignore_errors=True)

    def _make_tracer(self):
        from underthesea.agent.trace.local import LocalTracer
        return LocalTracer(trace_dir=self.trace_dir, console=False)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "UNDERTHESEA_TRACE_DISABLED": "1"})
    @patch(MOCK_POST, return_value=_resp("no trace"))
    def test_explicit_tracer_disabled_by_env(self, mock_post):
        from underthesea.agent import Agent
        tracer = self._make_tracer()
        bot = Agent("bot", tracer=tracer)
        result = bot("hi")
        self.assertEqual(result, "no trace")
        self.assertEqual(len(tracer.list_traces()), 0)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "UNDERTHESEA_TRACE_DISABLED": "true"})
    @patch(MOCK_POST, return_value=_resp("disabled"))
    def test_env_var_true_string(self, mock_post):
        from underthesea.agent import Agent
        tracer = self._make_tracer()
        bot = Agent("bot", tracer=tracer)
        bot("hi")
        self.assertEqual(len(tracer.list_traces()), 0)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "UNDERTHESEA_TRACE_DISABLED": "0"})
    @patch(MOCK_POST, return_value=_resp("enabled"))
    def test_env_var_zero_means_enabled(self, mock_post):
        from underthesea.agent import Agent
        tracer = self._make_tracer()
        bot = Agent("bot", tracer=tracer)
        bot("hi")
        self.assertEqual(len(tracer.list_traces()), 1)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "UNDERTHESEA_TRACE_DISABLED": "1"})
    @patch(MOCK_POST, return_value=_resp("disabled"))
    def test_decorator_disabled_by_env(self, mock_post):
        """@trace decorator context should not propagate when disabled."""
        from underthesea.agent import Agent
        from underthesea.agent.trace import trace
        tracer = self._make_tracer()

        @trace(tracer)
        def pipeline(msg):
            bot = Agent("bot")
            return bot(msg)

        # The decorator itself still runs (it doesn't check env),
        # but the Agent skips tracing
        pipeline("hi")
        traces = tracer.list_traces()
        # Decorator creates a trace but Agent adds no spans
        if traces:
            agent_gens = [s for s in traces[0].get("spans", []) if s["type"] == "generation"]
            self.assertEqual(len(agent_gens), 0)


# ======================================================================
# Auto local trace (like Claude Code)
# ======================================================================


class TestAutoTrace(TestCase):
    def setUp(self):
        self.trace_dir = tempfile.mkdtemp()
        # Reset global auto tracer so each test gets a fresh one
        import sys  # noqa: I001

        import underthesea.agent  # noqa: F401 — ensure module is loaded
        self._agent_mod = sys.modules["underthesea.agent.agent"]
        self._orig_auto_tracer = self._agent_mod._auto_tracer
        self._agent_mod._auto_tracer = None

    def tearDown(self):
        self._agent_mod._auto_tracer = self._orig_auto_tracer
        shutil.rmtree(self.trace_dir, ignore_errors=True)

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "UNDERTHESEA_TRACE_DIR": "",  # will be overridden below
    })
    @patch(MOCK_POST, return_value=_resp("auto traced!"))
    def test_auto_trace_agent_class(self, mock_post):
        """Agent without explicit tracer should auto-trace to local files."""
        os.environ["UNDERTHESEA_TRACE_DIR"] = self.trace_dir
        self._agent_mod._auto_tracer = None  # force re-creation with new dir

        from underthesea.agent import Agent
        bot = Agent("auto-bot")
        result = bot("hello")
        self.assertEqual(result, "auto traced!")

        # Verify trace file was created in the auto directory
        from underthesea.agent.trace.local import LocalTracer
        reader = LocalTracer(trace_dir=self.trace_dir, console=False)
        traces = reader.list_traces()
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["name"], "auto-bot")
        self.assertEqual(traces[0]["input"], "hello")
        self.assertEqual(traces[0]["status"], "ok")
        # Should have generation span
        gen_spans = [s for s in traces[0]["spans"] if s["type"] == "generation"]
        self.assertEqual(len(gen_spans), 1)

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "UNDERTHESEA_TRACE_DIR": "",
    })
    @patch(MOCK_POST, return_value=_resp("auto singleton"))
    def test_auto_trace_agent_singleton(self, mock_post):
        """Global agent() singleton should auto-trace."""
        os.environ["UNDERTHESEA_TRACE_DIR"] = self.trace_dir
        self._agent_mod._auto_tracer = None

        from underthesea import agent
        agent.reset()
        agent._llm = None
        result = agent("hi")
        self.assertEqual(result, "auto singleton")

        from underthesea.agent.trace.local import LocalTracer
        reader = LocalTracer(trace_dir=self.trace_dir, console=False)
        traces = reader.list_traces()
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["input"], "hi")

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "UNDERTHESEA_TRACE_DIR": "",
        "UNDERTHESEA_TRACE_DISABLED": "1",
    })
    @patch(MOCK_POST, return_value=_resp("no auto"))
    def test_auto_trace_disabled_by_env(self, mock_post):
        """Auto-trace should not run when UNDERTHESEA_TRACE_DISABLED=1."""
        os.environ["UNDERTHESEA_TRACE_DIR"] = self.trace_dir
        self._agent_mod._auto_tracer = None

        from underthesea.agent import Agent
        bot = Agent("bot")
        result = bot("hi")
        self.assertEqual(result, "no auto")

        from underthesea.agent.trace.local import LocalTracer
        reader = LocalTracer(trace_dir=self.trace_dir, console=False)
        self.assertEqual(len(reader.list_traces()), 0)

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "UNDERTHESEA_TRACE_DIR": "",
    })
    @patch(MOCK_POST, return_value=_resp("explicit wins"))
    def test_explicit_tracer_overrides_auto(self, mock_post):
        """Explicit tracer should be used instead of auto-tracer."""
        os.environ["UNDERTHESEA_TRACE_DIR"] = self.trace_dir
        self._agent_mod._auto_tracer = None

        explicit_dir = tempfile.mkdtemp()
        try:
            from underthesea.agent import Agent
            from underthesea.agent.trace.local import LocalTracer
            explicit_tracer = LocalTracer(trace_dir=explicit_dir, console=False)
            bot = Agent("bot", tracer=explicit_tracer)
            bot("hi")

            # Trace should be in explicit dir, not auto dir
            self.assertEqual(len(explicit_tracer.list_traces()), 1)
            auto_reader = LocalTracer(trace_dir=self.trace_dir, console=False)
            self.assertEqual(len(auto_reader.list_traces()), 0)
        finally:
            shutil.rmtree(explicit_dir, ignore_errors=True)
