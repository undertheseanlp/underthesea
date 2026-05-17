"""Tests for underthesea.agent.server."""
import json
import unittest
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

try:
    import httpx

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


def _make_agent() -> "Agent":
    return Agent(
        name="TestAgent", instruction="i",
        tools=[Tool(_double)],
    )


def _client(app):
    """httpx AsyncClient that drives the raw ASGI app in-process."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://t",
    )


def _parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in text.split("\n")
        if line.startswith("data: ")
    ]


@unittest.skipUnless(SERVER_AVAILABLE, "httpx not installed")
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
            trace, [{"name": "_double", "args": {"x": 5}, "result": 10}]
        )

    def test_silent_when_no_trace(self):
        wrapped = _record_tool(_double)
        self.assertEqual(wrapped(x=3), 6)

    def test_preserves_function_metadata(self):
        wrapped = _record_tool(_double)
        self.assertEqual(wrapped.__name__, "_double")
        self.assertEqual(wrapped.__doc__, "Doubles input.")


@unittest.skipUnless(SERVER_AVAILABLE, "httpx not installed")
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


@unittest.skipUnless(SERVER_AVAILABLE, "httpx not installed")
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


@unittest.skipUnless(SERVER_AVAILABLE, "httpx not installed")
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


@unittest.skipUnless(SERVER_AVAILABLE, "httpx not installed")
class TestMakeApp(IsolatedAsyncioTestCase):
    async def test_card_route_serves_auto_generated_card(self):
        app = make_app(_make_agent(), path="/a2a/x", host="127.0.0.1", port=8765)
        async with _client(app) as c:
            r = await c.get("/a2a/x/.well-known/agent-card.json")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["name"], "TestAgent")
        self.assertEqual(body["url"], "http://127.0.0.1:8765/a2a/x")

    async def test_card_route_serves_override(self):
        custom = _auto_agent_card(_make_agent(), "http://x:1/y")
        custom["name"] = "Overridden"
        app = make_app(
            _make_agent(), path="/a2a/x", agent_card=custom,
            host="127.0.0.1", port=8765,
        )
        async with _client(app) as c:
            r = await c.get("/a2a/x/.well-known/agent-card.json")
        self.assertEqual(r.json()["name"], "Overridden")

    async def test_unknown_route_returns_404(self):
        app = make_app(_make_agent(), path="/a2a/x", host="127.0.0.1", port=8765)
        async with _client(app) as c:
            r = await c.get("/no-such-route")
        self.assertEqual(r.status_code, 404)

    async def test_options_returns_cors_preflight(self):
        app = make_app(_make_agent(), path="/a2a/x", host="127.0.0.1", port=8765)
        async with _client(app) as c:
            r = await c.request(
                "OPTIONS", "/a2a/x", headers={"origin": "http://app.local"},
            )
        self.assertEqual(r.status_code, 204)
        self.assertIn("access-control-allow-origin", r.headers)
        self.assertEqual(r.headers["access-control-allow-origin"], "http://app.local")

    def test_tools_wrapped_after_make_app(self):
        agent = _make_agent()
        make_app(agent, path="/a2a/x", host="127.0.0.1", port=8765)
        self.assertTrue(
            getattr(agent.tools[0].func, "__uts_server_wrapped__", False)
        )


@unittest.skipUnless(SERVER_AVAILABLE, "httpx not installed")
class TestRpcStream(IsolatedAsyncioTestCase):
    async def test_stream_emits_full_a2a_event_sequence(self):
        """End-to-end: POST message/stream → SSE with all expected events."""
        agent = Agent(name="A", instruction="i", tools=[Tool(_double)])

        def fake_call(self_agent, message, **kwargs):
            trace = _trace_var.get()
            if trace is not None:
                trace.append({"name": "_double", "args": {"x": 1}, "result": 2})
            return "fake reply"

        with patch.object(Agent, "__call__", fake_call):
            app = make_app(agent, path="/a2a/x", host="127.0.0.1", port=8765)
            async with _client(app) as c:
                r = await c.post("/a2a/x", json={
                    "jsonrpc": "2.0", "id": "rpc-1",
                    "method": "message/stream",
                    "params": {"message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hi"}],
                        "messageId": "m1",
                        "contextId": "ctx-stream-test",
                    }},
                })

        self.assertEqual(r.status_code, 200)
        events = _parse_sse(r.text)

        for ev in events:
            self.assertEqual(ev["jsonrpc"], "2.0")
            self.assertEqual(ev["id"], "rpc-1")
            self.assertEqual(ev["result"]["contextId"], "ctx-stream-test")

        kinds = [ev["result"]["kind"] for ev in events]
        self.assertEqual(
            kinds,
            ["status-update", "status-update",
             "artifact-update", "artifact-update", "status-update"],
        )

        statuses = [
            ev["result"]["status"]["state"]
            for ev in events if ev["result"]["kind"] == "status-update"
        ]
        self.assertEqual(statuses, ["submitted", "working", "completed"])
        self.assertTrue(events[-1]["result"]["final"])

        artifacts = [
            ev["result"]["artifact"]
            for ev in events if ev["result"]["kind"] == "artifact-update"
        ]
        self.assertEqual([a["name"] for a in artifacts], ["tool_call", "reply"])
        self.assertEqual(
            artifacts[0]["parts"][0]["data"]["tool_call"],
            {"name": "_double", "args": {"x": 1}, "result": 2},
        )
        self.assertEqual(artifacts[1]["parts"][0]["text"], "fake reply")

    async def test_stream_generates_ids_when_missing(self):
        agent = Agent(name="A", instruction="i", tools=[Tool(_double)])

        def fake_call(_self, _message, **_kwargs):
            return "ok"

        with patch.object(Agent, "__call__", fake_call):
            app = make_app(agent, path="/a2a/x", host="127.0.0.1", port=8765)
            async with _client(app) as c:
                r = await c.post("/a2a/x", json={
                    "jsonrpc": "2.0", "id": "rpc-2",
                    "method": "message/stream",
                    "params": {"message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hi"}],
                        "messageId": "m1",
                    }},
                })

        events = _parse_sse(r.text)
        ctx_ids = {ev["result"]["contextId"] for ev in events}
        task_ids = {ev["result"]["taskId"] for ev in events}
        self.assertEqual(len(ctx_ids), 1)
        self.assertEqual(len(task_ids), 1)
        self.assertTrue(next(iter(ctx_ids)))
        self.assertTrue(next(iter(task_ids)))


if __name__ == "__main__":
    unittest.main()
