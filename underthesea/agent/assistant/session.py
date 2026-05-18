"""Session — markdown-backed conversation log for the assistant.

Each session is one file under ~/.underthesea/assistant/<name>.md (or a custom
directory). The format is human-readable and intentionally compatible with the
future MarkdownMemory abstraction (M2 of the roadmap).

File layout:

    # session: <name>
    > created: 2026-05-17T18:00:00
    > claude_session_id: <uuid>
    > model: claude-sonnet-4-6

    ## user — 2026-05-17T18:00:01
    Hello

    ## assistant — 2026-05-17T18:00:02
    Hi! How can I help?
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_DIR = Path.home() / ".underthesea" / "assistant"


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    ts: datetime = field(default_factory=datetime.now)


class Session:
    """Markdown-backed chat log.

    Usage:
        s = Session.open("my-chat")
        s.append("user", "Hello")
        s.append("assistant", "Hi!")
        s.save()
    """

    HEADER_RE = re.compile(r"^# session: (.+)$", re.MULTILINE)
    TURN_RE = re.compile(r"^## (user|assistant) — (.+)$", re.MULTILINE)

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.turns: list[Turn] = []
        self.claude_session_id: str | None = None
        self.model: str | None = None

    @classmethod
    def open(cls, name: str, directory: Path | None = None) -> Session:
        """Open or create a session by name. Loads existing turns if present."""
        directory = directory or DEFAULT_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.md"
        s = cls(name=name, path=path)
        if path.exists():
            s._load()
        return s

    def append(self, role: str, content: str) -> Turn:
        turn = Turn(role=role, content=content)
        self.turns.append(turn)
        return turn

    def save(self) -> None:
        """Persist the session to disk. Overwrites the file atomically."""
        text = self._serialize()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.path)

    def _serialize(self) -> str:
        lines = [f"# session: {self.name}"]
        created = self.turns[0].ts if self.turns else datetime.now()
        lines.append(f"> created: {created.isoformat(timespec='seconds')}")
        if self.claude_session_id:
            lines.append(f"> claude_session_id: {self.claude_session_id}")
        if self.model:
            lines.append(f"> model: {self.model}")
        lines.append("")
        for t in self.turns:
            lines.append(f"## {t.role} — {t.ts.isoformat(timespec='seconds')}")
            lines.append(t.content.rstrip())
            lines.append("")
        return "\n".join(lines) + "\n"

    def _load(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("> claude_session_id: "):
                self.claude_session_id = line.split(": ", 1)[1].strip()
            elif line.startswith("> model: "):
                self.model = line.split(": ", 1)[1].strip()

        # Split on turn headers and reconstruct.
        chunks = re.split(r"^## (user|assistant) — (\S+)\s*$", text, flags=re.MULTILINE)
        # chunks = [preamble, role1, ts1, body1, role2, ts2, body2, ...]
        for i in range(1, len(chunks) - 2, 3):
            role = chunks[i]
            ts_str = chunks[i + 1]
            body = chunks[i + 2].strip()
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                ts = datetime.now()
            self.turns.append(Turn(role=role, content=body, ts=ts))
