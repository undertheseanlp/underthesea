"""Tests for ClaudeBridge — subprocess wrapper around `claude` CLI."""

import asyncio
from unittest import TestCase
from unittest.mock import patch

from underthesea.agent.assistant.bridge import (
    BridgeError,
    ClaudeBridge,
    ClaudeNotFoundError,
    StreamEvent,
)


def _async(coro):
    return asyncio.run(coro)


class FakeStream:
    """Async iterator over predetermined byte lines."""

    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)

    async def read(self) -> bytes:
        return b""


class TestStreamEvent(TestCase):
    def test_text_chunk_extracts_text(self):
        ev = StreamEvent(
            type="assistant",
            raw={"message": {"content": [{"type": "text", "text": "Hello"}]}},
        )
        self.assertTrue(ev.is_text_chunk)
        self.assertEqual(ev.text, "Hello")

    def test_concatenates_multiple_text_blocks(self):
        ev = StreamEvent(
            type="assistant",
            raw={
                "message": {
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world"},
                    ]
                }
            },
        )
        self.assertEqual(ev.text, "Hello world")

    def test_thinking_block_is_not_text_chunk(self):
        ev = StreamEvent(
            type="assistant",
            raw={"message": {"content": [{"type": "thinking", "thinking": "..."}]}},
        )
        self.assertFalse(ev.is_text_chunk)
        self.assertEqual(ev.text, "")

    def test_tool_use_extracted(self):
        ev = StreamEvent(
            type="assistant",
            raw={
                "message": {
                    "content": [{"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}}]
                }
            },
        )
        self.assertEqual(ev.tool_use["name"], "Bash")

    def test_result_event_has_cost(self):
        ev = StreamEvent(type="result", raw={"total_cost_usd": 0.0123})
        self.assertTrue(ev.is_done)
        self.assertAlmostEqual(ev.cost_usd, 0.0123)


class TestClaudeBridgeCheck(TestCase):
    def test_missing_binary_raises(self):
        bridge = ClaudeBridge(binary="definitely-not-claude-xyz-1234")
        with self.assertRaises(ClaudeNotFoundError) as ctx:
            bridge.check()
        self.assertIn("not found", str(ctx.exception))
        self.assertIn("claude login", str(ctx.exception))

    @patch("underthesea.agent.assistant.bridge.shutil.which", return_value="/usr/local/bin/claude")
    def test_present_binary_returns_path(self, _):
        bridge = ClaudeBridge()
        self.assertEqual(bridge.check(), "/usr/local/bin/claude")


class TestBuildArgs(TestCase):
    @patch("underthesea.agent.assistant.bridge.shutil.which", return_value="/x/claude")
    def test_default_args(self, _):
        bridge = ClaudeBridge()
        args = bridge._build_args("hi")
        self.assertEqual(args[0], "claude")
        self.assertIn("--print", args)
        self.assertIn("--output-format", args)
        self.assertIn("stream-json", args)
        self.assertEqual(args[-1], "hi")
        self.assertNotIn("--model", args)
        self.assertNotIn("--resume", args)

    def test_model_added_when_set(self):
        bridge = ClaudeBridge(model="haiku")
        args = bridge._build_args("hi")
        i = args.index("--model")
        self.assertEqual(args[i + 1], "haiku")

    def test_resume_added_when_session_id(self):
        bridge = ClaudeBridge()
        bridge.session_id = "abc-123"
        args = bridge._build_args("hi")
        i = args.index("--resume")
        self.assertEqual(args[i + 1], "abc-123")


class TestParseStream(TestCase):
    def test_parses_jsonl_and_skips_empty(self):
        lines = [
            b'{"type":"system","session_id":"abc"}\n',
            b"\n",
            b'{"type":"assistant","message":{"content":[]},"session_id":"abc"}\n',
            b'not-json\n',
            b'{"type":"result","total_cost_usd":0.01,"session_id":"abc"}\n',
        ]

        async def collect():
            events = []
            async for ev in ClaudeBridge._parse_stream(FakeStream(lines)):
                events.append(ev)
            return events

        events = _async(collect())
        types = [e.type for e in events]
        self.assertEqual(types, ["system", "assistant", "result"])
        self.assertEqual(events[0].session_id, "abc")
        self.assertTrue(events[-1].is_done)


class TestSendSessionTracking(TestCase):
    def test_send_updates_session_id_from_first_event(self):
        """Mock the subprocess so we can verify session_id pickup without a real claude binary."""

        class FakeProc:
            def __init__(self):
                self.stdout = FakeStream(
                    [
                        b'{"type":"system","session_id":"xyz-456","subtype":"init"}\n',
                        b'{"type":"result","session_id":"xyz-456","total_cost_usd":0.0}\n',
                    ]
                )
                self.stderr = FakeStream([])
                self.returncode = 0

            async def wait(self):
                return 0

        async def fake_exec(*args, **kwargs):
            return FakeProc()

        bridge = ClaudeBridge()

        async def go():
            collected = []
            with patch("underthesea.agent.assistant.bridge.shutil.which", return_value="/x/claude"):
                with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
                    async for ev in bridge.send("hi"):
                        collected.append(ev)
            return collected

        events = _async(go())
        self.assertEqual(bridge.session_id, "xyz-456")
        self.assertEqual(len(events), 2)

    def test_nonzero_exit_raises(self):
        class FakeProcFail:
            def __init__(self):
                self.stdout = FakeStream([])
                self.stderr = FakeStream([b"boom\n"])

                async def _wait():
                    return 2

                self.wait = _wait
                self.returncode = 2

            async def stderr_read(self):
                return b"boom"

        class StderrFail:
            async def read(self):
                return b"boom"

        class FakeProc2:
            def __init__(self):
                self.stdout = FakeStream([])
                self.stderr = StderrFail()
                self.returncode = 2

            async def wait(self):
                return 2

        async def fake_exec(*args, **kwargs):
            return FakeProc2()

        bridge = ClaudeBridge()

        async def go():
            with patch("underthesea.agent.assistant.bridge.shutil.which", return_value="/x/claude"):
                with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
                    async for _ in bridge.send("hi"):
                        pass

        with self.assertRaises(BridgeError) as ctx:
            _async(go())
        self.assertIn("boom", str(ctx.exception))
