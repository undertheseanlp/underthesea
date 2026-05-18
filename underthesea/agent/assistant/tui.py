"""Textual TUI app for the assistant — OpenClaw-inspired minimal layout.

No header bar, no widget borders, no per-message role labels. User input
shows in a full-width slate bar; assistant replies render as plain text.
A two-line status bar at the bottom mirrors OpenClaw's footer (state +
agent / session / model / tokens).

Imports of `textual` are deferred so the rest of the assistant module
can be imported without the `[assistant]` extra installed.
"""
from __future__ import annotations

from pathlib import Path

from underthesea.agent.assistant.bridge import ClaudeBridge
from underthesea.agent.assistant.session import Session

# Claude default context window. Used for the "tokens X/Y (Z%)" indicator.
CONTEXT_WINDOW = 200_000


def run_tui(session: Session, bridge: ClaudeBridge) -> None:
    """Launch the chat TUI. Blocks until the user quits.

    Raises ImportError with a helpful message if the `[assistant]` extra
    is not installed.
    """
    build_app(session, bridge).run()


def build_app(session: Session, bridge: ClaudeBridge):
    """Construct the Textual app without running it.

    Useful for snapshot tests (pytest-textual-snapshot) and other
    inspection. Returns a non-running ``App`` instance.
    """
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import VerticalScroll
        from textual.widgets import Input, Static
    except ImportError as e:
        raise ImportError(
            "The assistant TUI requires the `[assistant]` extra:\n"
            "    pip install 'underthesea[assistant]'"
        ) from e

    class UserMessage(Static):
        DEFAULT_CSS = """
        UserMessage {
            background: #2a3447;
            color: #e6e6e6;
            padding: 1 2;
            margin: 0 0 1 0;
            width: 100%;
        }
        """

    class AssistantMessage(Static):
        DEFAULT_CSS = """
        AssistantMessage {
            color: #e6e6e6;
            padding: 0 2;
            margin: 0 0 1 0;
        }
        """

    class StatusBar(Static):
        DEFAULT_CSS = """
        StatusBar {
            height: 1;
            background: #0d1117;
            color: #8b949e;
            padding: 0 1;
        }
        """

    class AssistantApp(App):
        CSS = """
        Screen {
            background: #0d1117;
        }
        VerticalScroll#chat {
            background: #0d1117;
            scrollbar-size: 0 0;
            padding: 1 0 0 0;
        }
        Input {
            dock: bottom;
            border: none;
            background: #161b22;
            color: #e6e6e6;
            padding: 0 2;
            height: 1;
        }
        Input:focus {
            border: none;
        }
        """
        BINDINGS = [
            Binding("ctrl+c", "quit", "Quit", show=False),
            Binding("ctrl+l", "clear", "Clear", show=False),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.session = session
            self.bridge = bridge
            self.tokens_used = 0
            self.status_state = "idle"

        def compose(self) -> ComposeResult:
            yield VerticalScroll(id="chat")
            yield StatusBar(id="status")
            yield Input(placeholder="", id="input")

        def on_mount(self) -> None:
            chat = self.query_one("#chat", VerticalScroll)
            for turn in self.session.turns:
                if turn.role == "user":
                    chat.mount(UserMessage(turn.content))
                else:
                    chat.mount(AssistantMessage(turn.content))
            self._refresh_status()
            self.query_one("#input", Input).focus()

        def _refresh_status(self) -> None:
            workspace = Path(self.bridge.cwd or ".").resolve()
            workspace_name = workspace.name or "."
            sid = self.bridge.session_id
            sid_short = f"{sid[:8]}…" if sid else "—"
            model = self.bridge.model or "default"
            tokens_k = self.tokens_used / 1000
            window_k = CONTEXT_WINDOW // 1000

            status = (
                f"{self.status_state} | "
                f"{workspace_name}/{self.session.name} | "
                f"{model} | {sid_short} | "
                f"{tokens_k:.1f}k/{window_k}k"
            )
            self.query_one("#status", StatusBar).update(status)

        async def on_input_submitted(self, message: Input.Submitted) -> None:
            text = message.value.strip()
            if not text:
                return
            input_widget = self.query_one("#input", Input)
            input_widget.value = ""
            input_widget.disabled = True

            chat = self.query_one("#chat", VerticalScroll)
            chat.mount(UserMessage(text))
            self.session.append("user", text)

            reply_widget = AssistantMessage("")
            chat.mount(reply_widget)
            chat.scroll_end(animate=False)

            self.status_state = "thinking…"
            self._refresh_status()

            parts: list[str] = []
            try:
                async for event in self.bridge.send(text):
                    if event.type == "system" and event.session_id:
                        self.session.claude_session_id = event.session_id
                        self._refresh_status()
                    if event.is_text_chunk:
                        chunk = event.text
                        if chunk:
                            parts.append(chunk)
                            reply_widget.update("".join(parts))
                            chat.scroll_end(animate=False)
                    elif event.is_done:
                        usage = event.raw.get("usage", {}) or {}
                        self.tokens_used += (
                            (usage.get("input_tokens") or 0)
                            + (usage.get("output_tokens") or 0)
                            + (usage.get("cache_read_input_tokens") or 0)
                            + (usage.get("cache_creation_input_tokens") or 0)
                        )
            except Exception as e:  # noqa: BLE001
                reply_widget.update(f"[error] {e}")
            else:
                reply = "".join(parts).strip()
                if reply:
                    self.session.append("assistant", reply)
                    self.session.save()
            finally:
                self.status_state = "idle"
                self._refresh_status()
                input_widget.disabled = False
                input_widget.focus()

        def action_clear(self) -> None:
            chat = self.query_one("#chat", VerticalScroll)
            for child in list(chat.children):
                child.remove()

    return AssistantApp()
