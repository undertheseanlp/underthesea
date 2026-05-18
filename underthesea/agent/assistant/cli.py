"""Click command implementation for `underthesea chat`.

Imported lazily from underthesea.cli to keep startup time small for users who
do not use the chat TUI.
"""
from __future__ import annotations

import sys
from datetime import datetime

import click

from underthesea.agent.assistant.bridge import ClaudeBridge, ClaudeNotFoundError
from underthesea.agent.assistant.session import DEFAULT_DIR, Session


def _new_session_name() -> str:
    """Timestamp-based session name, e.g. ``2026-05-17_19-30-15``."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@click.command(name="chat")
@click.option(
    "--session",
    "session_name",
    default=None,
    help=(
        "Session name. Each invocation starts a fresh timestamped session "
        "by default; pass an existing name to resume a saved chat."
    ),
)
@click.option(
    "--memory-dir",
    type=click.Path(file_okay=False),
    default=None,
    help=f"Directory for session files. Defaults to {DEFAULT_DIR}.",
)
@click.option(
    "--model",
    default=None,
    help="Override the Claude model (e.g. 'sonnet', 'haiku', 'opus').",
)
@click.option(
    "--check",
    is_flag=True,
    help="Verify the `claude` CLI is installed and exit.",
)
def chat_command(
    session_name: str | None,
    memory_dir: str | None,
    model: str | None,
    check: bool,
) -> None:
    """Launch the Underthesea chat TUI.

    Uses your local `claude` CLI subscription as the LLM backend — no API key
    or token cost. Make sure you have run `claude login` once.

    Each invocation starts a fresh timestamped session unless ``--session``
    names a saved one to resume.
    """
    bridge = ClaudeBridge(model=model)

    try:
        binary_path = bridge.check()
    except ClaudeNotFoundError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    if check:
        click.echo(f"claude: {binary_path}")
        if model:
            click.echo(f"model: {model}")
        click.echo("OK")
        return

    from pathlib import Path

    directory = Path(memory_dir) if memory_dir else None
    if session_name is None:
        session_name = _new_session_name()
        click.echo(f"new session: {session_name}", err=True)
    session = Session.open(session_name, directory)
    if session.claude_session_id:
        bridge.session_id = session.claude_session_id
    if session.model:
        bridge.model = bridge.model or session.model
    elif model:
        session.model = model

    from underthesea.agent.assistant.tui import run_tui

    try:
        run_tui(session, bridge)
    except ImportError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
