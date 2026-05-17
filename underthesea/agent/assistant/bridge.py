"""ClaudeBridge — async subprocess wrapper around the `claude` CLI.

Spawns `claude --print --output-format=stream-json --verbose` for each user
turn and yields parsed stream events. Auth comes from the user's existing
`claude login`, so no API key is required.

Schema of events (a few examples observed in the wild):
    {"type": "system", "subtype": "init", "session_id": "...", "model": "..."}
    {"type": "assistant", "message": {"content": [{"type": "text", "text": "..."}]}, "session_id": "..."}
    {"type": "result", "subtype": "success", "result": "Hello", "total_cost_usd": 0.04, ...}
"""
from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass


class BridgeError(RuntimeError):
    """Raised when the claude subprocess fails or emits an unrecoverable event."""


class ClaudeNotFoundError(BridgeError):
    """Raised when the `claude` binary is missing from PATH."""

    INSTALL_URL = "https://docs.claude.com/en/docs/claude-code/quickstart"

    def __init__(self) -> None:
        super().__init__(
            "`claude` CLI not found in PATH.\n"
            f"Install Claude Code: {self.INSTALL_URL}\n"
            "Then run `claude login` to authenticate."
        )


@dataclass
class StreamEvent:
    """One parsed JSONL line from `claude --output-format=stream-json`."""

    type: str
    raw: dict
    session_id: str | None = None

    @property
    def is_text_chunk(self) -> bool:
        if self.type != "assistant":
            return False
        for block in self.raw.get("message", {}).get("content", []):
            if block.get("type") == "text":
                return True
        return False

    @property
    def text(self) -> str:
        """Concatenated text from all text blocks in an assistant event."""
        if self.type != "assistant":
            return ""
        parts = []
        for block in self.raw.get("message", {}).get("content", []):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    @property
    def tool_use(self) -> dict | None:
        """First tool_use block in this assistant event, if any."""
        if self.type != "assistant":
            return None
        for block in self.raw.get("message", {}).get("content", []):
            if block.get("type") == "tool_use":
                return block
        return None

    @property
    def is_done(self) -> bool:
        return self.type == "result"

    @property
    def cost_usd(self) -> float | None:
        if self.type == "result":
            return self.raw.get("total_cost_usd")
        return None


class ClaudeBridge:
    """Subprocess bridge to the local `claude` CLI.

    Usage:
        bridge = ClaudeBridge()
        async for event in bridge.send("Hello"):
            if event.is_text_chunk:
                print(event.text, end="")
            elif event.is_done:
                print(f"\\n(cost: ${event.cost_usd:.4f})")
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        binary: str = "claude",
        cwd: str | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.binary = binary
        self.model = model
        self.cwd = cwd
        self.extra_args = extra_args or []
        self.session_id: str | None = None

    def check(self) -> str:
        """Resolve the `claude` binary path. Raises ClaudeNotFoundError if missing."""
        path = shutil.which(self.binary)
        if not path:
            raise ClaudeNotFoundError()
        return path

    def _build_args(self, prompt: str) -> list[str]:
        args = [
            self.binary,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if self.model:
            args += ["--model", self.model]
        if self.session_id:
            args += ["--resume", self.session_id]
        args += self.extra_args
        args.append(prompt)
        return args

    async def send(self, prompt: str) -> AsyncIterator[StreamEvent]:
        """Send a user prompt and yield StreamEvents as they arrive.

        Updates self.session_id from the first system event so the next call
        continues the conversation via --resume.
        """
        self.check()
        args = self._build_args(prompt)
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None

        try:
            async for event in self._parse_stream(proc.stdout):
                if event.session_id and not self.session_id:
                    self.session_id = event.session_id
                yield event
        finally:
            await proc.wait()
            if proc.returncode and proc.returncode != 0:
                stderr_bytes = await proc.stderr.read()
                raise BridgeError(
                    f"claude exited with code {proc.returncode}: "
                    f"{stderr_bytes.decode(errors='replace').strip()}"
                )

    @staticmethod
    async def _parse_stream(
        stdout: asyncio.StreamReader,
    ) -> AsyncIterator[StreamEvent]:
        async for line in stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield StreamEvent(
                type=obj.get("type", "unknown"),
                raw=obj,
                session_id=obj.get("session_id"),
            )

    def reset(self) -> None:
        """Forget the current session so the next send starts fresh."""
        self.session_id = None
