"""Context management for multi-session agents.

Implements context reset with structured handoff, following the Anthropic
recommendation that context resets outperform compaction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class HandoffData:
    """Structured data for context handoff between sessions."""

    session_id: int
    previous_session_summary: str
    current_subtask_id: int | None = None
    context_for_next_session: str = ""
    artifacts: list[str] = field(default_factory=list)
    warnings: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ContextManager:
    """Manages context reset and structured handoff between agent sessions."""

    def __init__(self, handoff_dir: str | Path):
        self._dir = Path(handoff_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save_handoff(self, handoff: HandoffData) -> Path:
        """Save handoff data for the next session."""
        path = self._dir / f"handoff_session_{handoff.session_id}.json"
        data = {
            "session_id": handoff.session_id,
            "previous_session_summary": handoff.previous_session_summary,
            "current_subtask_id": handoff.current_subtask_id,
            "context_for_next_session": handoff.context_for_next_session,
            "artifacts": handoff.artifacts,
            "warnings": handoff.warnings,
            "created_at": handoff.created_at,
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def load_latest_handoff(self) -> HandoffData | None:
        """Load the most recent handoff file."""
        handoff_files = sorted(self._dir.glob("handoff_session_*.json"))
        if not handoff_files:
            return None
        with open(handoff_files[-1]) as f:
            data = json.load(f)
        return HandoffData(**data)

    def load_handoff(self, session_id: int) -> HandoffData | None:
        """Load a specific session's handoff."""
        path = self._dir / f"handoff_session_{session_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return HandoffData(**data)

    def build_context_prompt(self, handoff: HandoffData) -> str:
        """Build a context prompt from handoff data for the new session."""
        parts = [
            f"## Context from previous session (Session {handoff.session_id})",
            f"\n### Summary\n{handoff.previous_session_summary}",
        ]
        if handoff.context_for_next_session:
            parts.append(f"\n### Instructions\n{handoff.context_for_next_session}")
        if handoff.artifacts:
            parts.append("\n### Artifacts\n" + "\n".join(f"- {a}" for a in handoff.artifacts))
        if handoff.warnings:
            parts.append(f"\n### Warnings\n{handoff.warnings}")
        return "\n".join(parts)
