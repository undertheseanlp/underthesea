"""Tests for Session — markdown-backed conversation log."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest import TestCase

from underthesea.agent.assistant.session import Session, Turn


class TestSession(TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_open_creates_empty_session_when_missing(self):
        s = Session.open("new-chat", self.tmpdir)
        self.assertEqual(s.name, "new-chat")
        self.assertEqual(s.turns, [])
        self.assertIsNone(s.claude_session_id)

    def test_append_and_save_persists_turns(self):
        s = Session.open("chat", self.tmpdir)
        s.append("user", "Hello")
        s.append("assistant", "Hi there")
        s.claude_session_id = "abc-123"
        s.model = "sonnet"
        s.save()

        content = (self.tmpdir / "chat.md").read_text()
        self.assertIn("# session: chat", content)
        self.assertIn("> claude_session_id: abc-123", content)
        self.assertIn("> model: sonnet", content)
        self.assertIn("## user", content)
        self.assertIn("Hello", content)
        self.assertIn("## assistant", content)
        self.assertIn("Hi there", content)

    def test_save_then_reopen_roundtrips(self):
        s1 = Session.open("trip", self.tmpdir)
        s1.append("user", "first")
        s1.append("assistant", "second")
        s1.claude_session_id = "uuid-xyz"
        s1.model = "haiku"
        s1.save()

        s2 = Session.open("trip", self.tmpdir)
        self.assertEqual(s2.claude_session_id, "uuid-xyz")
        self.assertEqual(s2.model, "haiku")
        self.assertEqual(len(s2.turns), 2)
        self.assertEqual(s2.turns[0].role, "user")
        self.assertEqual(s2.turns[0].content, "first")
        self.assertEqual(s2.turns[1].role, "assistant")
        self.assertEqual(s2.turns[1].content, "second")

    def test_multiline_content_preserved(self):
        s1 = Session.open("ml", self.tmpdir)
        body = "line one\nline two\nline three"
        s1.append("assistant", body)
        s1.save()

        s2 = Session.open("ml", self.tmpdir)
        self.assertEqual(s2.turns[0].content, body)

    def test_atomic_save_does_not_leave_tmp(self):
        s = Session.open("atomic", self.tmpdir)
        s.append("user", "x")
        s.save()
        files = sorted(p.name for p in self.tmpdir.iterdir())
        self.assertEqual(files, ["atomic.md"])

    def test_turn_default_ts_is_set(self):
        t = Turn(role="user", content="hi")
        self.assertIsInstance(t.ts, datetime)
