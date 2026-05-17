"""Tests for underthesea.agent.server."""
import json
import unittest
from unittest import TestCase

try:
    from starlette.testclient import TestClient

    from underthesea.agent import Agent, Tool
    from underthesea.agent.server import (
        _auto_agent_card,
        _record_tool,
        _spawn_session_agent,
        _trace_var,
        _wrap_tools_inplace,
        make_app,
    )
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False


def _double(x: int) -> int:
    """Doubles input."""
    return x * 2


@unittest.skipUnless(SERVER_AVAILABLE, "a2a-sdk[http-server] not installed")
class TestRecordTool(TestCase):
    def test_records_call_when_trace_active(self):
        wrapped = _record_tool(_double)
        trace: list = []
        token = _trace_var.set(trace)
        try:
            result = wrapped(x=5)
        finally:
            _trace_var.reset(token)
        self.assertEqual(result, 10)
        self.assertEqual(
            trace, [{"tool": "_double", "args": {"x": 5}, "result": 10}]
        )

    def test_silent_when_no_trace(self):
        wrapped = _record_tool(_double)
        self.assertEqual(wrapped(x=3), 6)

    def test_preserves_function_metadata(self):
        wrapped = _record_tool(_double)
        self.assertEqual(wrapped.__name__, "_double")
        self.assertEqual(wrapped.__doc__, "Doubles input.")


@unittest.skipUnless(SERVER_AVAILABLE, "a2a-sdk[http-server] not installed")
class TestWrapToolsInplace(TestCase):
    def test_wraps_func_and_sets_marker(self):
        tool = Tool(_double)
        original = tool.func
        _wrap_tools_inplace([tool])
        self.assertIsNot(tool.func, original)
        self.assertTrue(getattr(tool.func, "__uts_server_wrapped__", False))

    def test_idempotent(self):
        tool = Tool(_double)
        _wrap_tools_inplace([tool])
        first = tool.func
        _wrap_tools_inplace([tool])
        self.assertIs(tool.func, first)

    def test_wrapped_func_records_via_contextvar(self):
        tool = Tool(_double)
        _wrap_tools_inplace([tool])
        trace: list = []
        token = _trace_var.set(trace)
        try:
            result = tool.execute({"x": 7})
        finally:
            _trace_var.reset(token)
        self.assertEqual(json.loads(result), 14)
        self.assertEqual(trace[0]["args"], {"x": 7})


@unittest.skipUnless(SERVER_AVAILABLE, "a2a-sdk[http-server] not installed")
class TestSpawnSessionAgent(TestCase):
    def test_clones_template_with_isolated_history(self):
        template = Agent(
            name="T", instruction="instr",
            tools=[Tool(_double)], max_iterations=7,
        )
        template._history.append({"role": "user", "content": "prior"})

        session = _spawn_session_agent(template)

        self.assertEqual(session.name, "T")
        self.assertEqual(session.instruction, "instr")
        self.assertEqual(session.max_iterations, 7)
        self.assertIs(session.tools[0], template.tools[0])
        self.assertEqual(len(session.history), 0)

    def test_shares_provider_and_tracer(self):
        template = Agent(name="T", instruction="i", tools=[Tool(_double)])
        sentinel_provider = object()
        sentinel_tracer = object()
        template._provider = sentinel_provider
        template._tracer = sentinel_tracer

        session = _spawn_session_agent(template)

        self.assertIs(session._provider, sentinel_provider)
        self.assertIs(session._tracer, sentinel_tracer)


@unittest.skipUnless(SERVER_AVAILABLE, "a2a-sdk[http-server] not installed")
class TestAutoAgentCard(TestCase):
    def test_required_fields_present(self):
        agent = Agent(
            name="MathAgent", instruction="Be helpful",
            tools=[Tool(_double)],
        )
        card = _auto_agent_card(agent, "http://localhost:8000/a2a/math")

        self.assertEqual(card["name"], "MathAgent")
        self.assertEqual(card["url"], "http://localhost:8000/a2a/math")
        self.assertEqual(card["preferredTransport"], "JSONRPC")
        self.assertTrue(card["capabilities"]["streaming"])
        self.assertEqual(len(card["skills"]), 1)
        self.assertEqual(card["skills"][0]["id"], "_double")
        self.assertEqual(card["skills"][0]["description"], "Doubles input.")

    def test_truncates_long_instruction(self):
        agent = Agent(name="X", instruction="a" * 500, tools=[])
        card = _auto_agent_card(agent, "http://x/")
        self.assertLessEqual(len(card["description"]), 200)


@unittest.skipUnless(SERVER_AVAILABLE, "a2a-sdk[http-server] not installed")
class TestMakeApp(TestCase):
    def _agent(self):
        return Agent(
            name="TestAgent", instruction="i",
            tools=[Tool(_double)],
        )

    def test_card_route_serves_auto_generated_card(self):
        app = make_app(self._agent(), path="/a2a/x", host="127.0.0.1", port=8765)
        client = TestClient(app)
        r = client.get("/a2a/x/.well-known/agent-card.json")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["name"], "TestAgent")
        self.assertEqual(body["url"], "http://127.0.0.1:8765/a2a/x")

    def test_card_route_serves_override(self):
        custom = _auto_agent_card(self._agent(), "http://x:1/y")
        custom["name"] = "Overridden"
        app = make_app(
            self._agent(), path="/a2a/x", agent_card=custom,
            host="127.0.0.1", port=8765,
        )
        client = TestClient(app)
        body = client.get("/a2a/x/.well-known/agent-card.json").json()
        self.assertEqual(body["name"], "Overridden")

    def test_rpc_endpoint_mounted(self):
        app = make_app(self._agent(), path="/a2a/x", host="127.0.0.1", port=8765)
        paths = {getattr(r, "path", None) for r in app.routes}
        self.assertIn("/a2a/x", paths)
        self.assertIn("/a2a/x/.well-known/agent-card.json", paths)

    def test_tools_wrapped_after_make_app(self):
        agent = self._agent()
        make_app(agent, path="/a2a/x", host="127.0.0.1", port=8765)
        self.assertTrue(
            getattr(agent.tools[0].func, "__uts_server_wrapped__", False)
        )


if __name__ == "__main__":
    unittest.main()
