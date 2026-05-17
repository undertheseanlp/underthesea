"""Textual TUI app for the assistant.

Imports of `textual` are deferred so the rest of the assistant module can be
imported without the `[assistant]` extra installed.
"""
from __future__ import annotations

from underthesea.agent.assistant.bridge import ClaudeBridge
from underthesea.agent.assistant.session import Session


def run_tui(session: Session, bridge: ClaudeBridge) -> None:
    """Launch the chat TUI. Blocks until the user quits.

    Raises ImportError with a helpful message if the `[assistant]` extra is
    not installed.
    """
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Vertical
        from textual.widgets import Footer, Header, Input, RichLog
    except ImportError as e:
        raise ImportError(
            "The assistant TUI requires the `[assistant]` extra:\n"
            "    pip install 'underthesea[assistant]'"
        ) from e

    class AssistantApp(App):
        CSS = """
        Screen { layout: vertical; }
        RichLog { border: solid $accent; padding: 1; }
        Input { dock: bottom; }
        """
        BINDINGS = [
            Binding("ctrl+c", "quit", "Quit"),
            Binding("ctrl+l", "clear", "Clear"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.session = session
            self.bridge = bridge

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Vertical(RichLog(id="log", wrap=True, markup=True))
            yield Input(placeholder="Type a message and press Enter...", id="input")
            yield Footer()

        def on_mount(self) -> None:
            log = self.query_one("#log", RichLog)
            log.write(f"[dim]session: {self.session.name}[/dim]")
            if self.bridge.model:
                log.write(f"[dim]model: {self.bridge.model}[/dim]")
            if self.bridge.session_id:
                log.write(f"[dim]resuming claude session: {self.bridge.session_id}[/dim]")
            log.write("")
            for turn in self.session.turns:
                self._render_turn(turn.role, turn.content)
            self.query_one("#input", Input).focus()

        def _render_turn(self, role: str, content: str) -> None:
            log = self.query_one("#log", RichLog)
            colour = "cyan" if role == "user" else "green"
            log.write(f"[bold {colour}]{role}[/bold {colour}]: {content}")
            log.write("")

        async def on_input_submitted(self, message: Input.Submitted) -> None:
            text = message.value.strip()
            if not text:
                return
            input_widget = self.query_one("#input", Input)
            input_widget.value = ""
            input_widget.disabled = True

            self.session.append("user", text)
            self._render_turn("user", text)

            log = self.query_one("#log", RichLog)
            log.write("[bold green]assistant[/bold green]: ", expand=True)
            collected: list[str] = []
            try:
                async for event in self.bridge.send(text):
                    if event.type == "system" and event.session_id:
                        self.session.claude_session_id = event.session_id
                    if event.is_text_chunk:
                        chunk = event.text
                        if chunk:
                            collected.append(chunk)
                            log.write(chunk)
                    elif (tool := event.tool_use) is not None:
                        log.write(f"[dim]→ tool: {tool.get('name')}[/dim]")
                    elif event.is_done:
                        cost = event.cost_usd
                        if cost is not None:
                            log.write(f"[dim](cost: ${cost:.4f})[/dim]")
            except Exception as e:  # noqa: BLE001
                log.write(f"[bold red]error:[/bold red] {e}")
            else:
                reply = "".join(collected)
                if reply:
                    self.session.append("assistant", reply)
                    self.session.save()
            finally:
                log.write("")
                input_widget.disabled = False
                input_widget.focus()

        def action_clear(self) -> None:
            self.query_one("#log", RichLog).clear()

    AssistantApp().run()
