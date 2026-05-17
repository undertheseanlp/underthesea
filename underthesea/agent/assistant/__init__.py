"""Underthesea Assistant — TUI agent bridging to the local `claude` CLI.

This module spawns Claude Code as a subprocess and uses the user's existing
`claude login` subscription, so no API key is required and no token cost is
incurred beyond what Claude Code itself uses.

Public API:
    ClaudeBridge: subprocess wrapper that streams events from `claude --print`.
    Session: markdown-backed conversation log persisted under
        ~/.underthesea/assistant/<name>.md.
    run_tui: launch the Textual chat UI (requires the `[assistant]` extra).
"""
from underthesea.agent.assistant.bridge import (
    BridgeError,
    ClaudeBridge,
    ClaudeNotFoundError,
    StreamEvent,
)
from underthesea.agent.assistant.session import Session

__all__ = [
    "BridgeError",
    "ClaudeBridge",
    "ClaudeNotFoundError",
    "Session",
    "StreamEvent",
    "build_app",
    "run_tui",
]


def __getattr__(name):
    """Lazy re-export of TUI helpers so importing this module does not
    require the optional ``textual`` dependency until those names are used.
    """
    if name in {"build_app", "run_tui"}:
        from underthesea.agent.assistant import tui as _tui

        return getattr(_tui, name)
    raise AttributeError(f"module 'underthesea.agent.assistant' has no attribute {name!r}")
