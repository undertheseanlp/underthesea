"""Snapshot tests for the assistant TUI via pytest-textual-snapshot.

Run:
    pip install -e '.[assistant]' '.[test-tui]'
    pytest tests/agent/assistant/test_tui_snapshot.py

First run generates baseline SVGs under
``tests/agent/assistant/__snapshots__/``. Subsequent runs diff against
the baseline. Update intentional UI changes with ``pytest --snapshot-update``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from underthesea.agent.assistant.bridge import StreamEvent
from underthesea.agent.assistant.session import Session
from underthesea.agent.assistant.tui import build_app


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeBridge:
    """ClaudeBridge stand-in that yields canned StreamEvents.

    Mirrors the public attributes that the TUI reads (cwd, model,
    session_id) and exposes an ``async def send(prompt)`` async generator.
    """

    def __init__(
        self,
        *,
        events: list[StreamEvent] | None = None,
        cwd: str = "/workspace/underthesea-agent",
        model: str = "haiku",
    ) -> None:
        self.cwd = cwd
        self.model = model
        self.session_id: str | None = None
        self._events = list(events or [])

    def check(self) -> str:
        return "/fake/claude"

    async def send(self, prompt: str):
        for ev in self._events:
            if ev.session_id and not self.session_id:
                self.session_id = ev.session_id
            yield ev


def _session(tmp_path: Path, name: str = "snapshot") -> Session:
    return Session(name=name, path=tmp_path / f"{name}.md")


# Canonical "one good turn" event sequence for the streaming snapshot.
_GOOD_TURN = [
    StreamEvent(
        type="system",
        raw={"subtype": "init", "session_id": "fake1234-5678-9abc-def0-000000000000"},
        session_id="fake1234-5678-9abc-def0-000000000000",
    ),
    StreamEvent(
        type="assistant",
        raw={
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": "Chào bạn! Mình là Uts 🪸 — đang chạy trên Claude qua Underthesea Assistant.",
                    }
                ]
            }
        },
        session_id="fake1234-5678-9abc-def0-000000000000",
    ),
    StreamEvent(
        type="result",
        raw={
            "total_cost_usd": 0.0123,
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 25,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
        session_id="fake1234-5678-9abc-def0-000000000000",
    ),
]


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


def test_empty_app(snap_compare, tmp_path):
    """Baseline: fresh session, no turns yet."""
    session = _session(tmp_path)
    bridge = FakeBridge()
    app = build_app(session, bridge)
    assert snap_compare(app, terminal_size=(80, 20))


def test_resumed_session_renders_history(snap_compare, tmp_path):
    """A session loaded from disk should re-render its prior turns."""
    session = _session(tmp_path, name="resumed")
    session.append("user", "Trước đó bạn đã hỏi gì?")
    session.append(
        "assistant",
        "Đây là câu trả lời cũ — render lại khi resume session.",
    )
    session.claude_session_id = "abc12345-6789-0000-1111-222222222222"

    bridge = FakeBridge()
    bridge.session_id = session.claude_session_id

    app = build_app(session, bridge)
    assert snap_compare(app, terminal_size=(80, 20))


def test_after_user_turn(snap_compare, tmp_path):
    """Drive a full turn: type, enter, stream done. Captures user bar +
    assistant reply + status bar with token count + cost cleared."""
    session = _session(tmp_path, name="after-turn")
    bridge = FakeBridge(events=_GOOD_TURN)
    app = build_app(session, bridge)

    async def drive(pilot):
        await pilot.press(*"xin chào")
        await pilot.press("enter")
        # Pump the event loop until the bridge.send generator is exhausted.
        await pilot.pause()
        await pilot.pause()

    assert snap_compare(app, run_before=drive, terminal_size=(80, 20))
