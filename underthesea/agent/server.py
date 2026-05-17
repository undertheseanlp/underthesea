"""A2A-compatible HTTP server for Agent instances.

Wraps an :class:`Agent` so it speaks the A2A protocol — JSON-RPC over HTTP
with SSE streaming, plus a discoverable AgentCard at
``{path}/.well-known/agent-card.json``. One :class:`Agent` is spawned per
A2A ``context_id`` so each conversation keeps its own history.

Requires ``a2a-sdk[http-server]`` (Starlette + sse-starlette).

Example
-------
::

    from underthesea.agent import Agent, Tool
    from underthesea.agent.server import serve

    def add(a: int, b: int) -> int:
        '''Add two integers.'''
        return a + b

    agent = Agent(name="MathAgent", instruction="...", tools=[Tool(add)])
    serve(agent, port=8000, path="/a2a/math")

Use :func:`make_app` instead of :func:`serve` if you need to mount extra
routes (e.g. a static UI) on the returned Starlette app.
"""
import asyncio
import contextvars
import functools
import json
import uuid
from collections.abc import Callable

import uvicorn
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.simple_request_context_builder import (
    SimpleRequestContextBuilder,
)
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import AgentCard, Part
from a2a.utils.parts import get_text_parts
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.routing import Route

from underthesea.agent.agent import Agent
from underthesea.agent.tools import Tool

# Tool calls happen deep inside Agent._call_with_tools; a ContextVar lets us
# observe each call without changing the Agent API.
_trace_var: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "underthesea_agent_server_trace", default=None
)


def _record_tool(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapped(**kwargs):
        result = func(**kwargs)
        trace = _trace_var.get()
        if trace is not None:
            trace.append({"tool": func.__name__, "args": kwargs, "result": result})
        return result
    wrapped.__uts_server_wrapped__ = True
    return wrapped


def _wrap_tools_inplace(tools: list[Tool]) -> None:
    for tool in tools:
        if not getattr(tool.func, "__uts_server_wrapped__", False):
            tool.func = _record_tool(tool.func)


def _spawn_session_agent(template: Agent) -> Agent:
    """Fresh Agent sharing the template's tools/instruction/provider, isolated history."""
    return Agent(
        name=template.name,
        tools=template.tools,
        instruction=template.instruction,
        max_iterations=template.max_iterations,
        provider=template._provider,
        tracer=template._tracer,
    )


def _auto_agent_card(agent: Agent, url: str) -> dict:
    return {
        "name": agent.name,
        "description": (agent.instruction or "")[:200],
        "url": url,
        "preferredTransport": "JSONRPC",
        "protocolVersion": "0.3.0",
        "version": "1.0.0",
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "skills": [
            {"id": t.name, "name": t.name, "description": t.description}
            for t in agent.tools
        ],
    }


class _SessionAgentExecutor(AgentExecutor):
    """A2A executor that spawns one Agent per A2A ``context_id``."""

    def __init__(self, template: Agent):
        self._template = template
        self._sessions: dict[str, Agent] = {}

    async def execute(self, context, event_queue) -> None:
        updater = TaskUpdater(
            event_queue,
            task_id=context.task_id or str(uuid.uuid4()),
            context_id=context.context_id or str(uuid.uuid4()),
        )
        await updater.submit()
        await updater.start_work()

        text_parts = get_text_parts(context.message.parts if context.message else [])
        user_message = (text_parts[0] if text_parts else "").strip()

        agent = self._sessions.setdefault(
            updater.context_id, _spawn_session_agent(self._template)
        )
        trace: list[dict] = []
        token = _trace_var.set(trace)
        try:
            reply = await asyncio.to_thread(agent, user_message)
        finally:
            _trace_var.reset(token)

        for step in trace:
            await updater.add_artifact(
                [Part.model_validate(
                    {"kind": "data", "data": {"tool_call": step}}
                )],
                name="tool_call",
            )
        await updater.add_artifact(
            [Part.model_validate({"kind": "text", "text": reply})],
            name="reply",
        )
        await updater.complete()

    async def cancel(self, context, event_queue) -> None:
        pass


def make_app(
    agent: Agent,
    *,
    path: str,
    agent_card: dict | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    allow_origins: list[str] | None = None,
):
    """Build a Starlette ASGI app serving ``agent`` over A2A.

    Parameters
    ----------
    agent : Agent
        Template agent; per-session copies are spawned to isolate history.
    path : str
        JSON-RPC endpoint, e.g. ``"/a2a/my_agent"``. The agent card is mounted
        at ``f"{path}/.well-known/agent-card.json"``.
    agent_card : dict, optional
        Override the auto-generated AgentCard.
    host, port : str, int
        Used only to fill the ``url`` field of the auto-generated card.
    allow_origins : list[str], optional
        CORS origins. Defaults to ``["*"]``.
    """
    _wrap_tools_inplace(agent.tools)

    card_dict = agent_card or _auto_agent_card(agent, f"http://{host}:{port}{path}")
    card_bytes = json.dumps(card_dict).encode("utf-8")
    card_path = f"{path}/.well-known/agent-card.json"

    async def serve_card(_request):
        return Response(content=card_bytes, media_type="application/json")

    handler = DefaultRequestHandler(
        agent_executor=_SessionAgentExecutor(agent),
        task_store=InMemoryTaskStore(),
        request_context_builder=SimpleRequestContextBuilder(),
    )
    app = A2AStarletteApplication(
        agent_card=AgentCard.model_validate(card_dict),
        http_handler=handler,
    ).build(rpc_url=path, agent_card_url=card_path)
    # Ours wins because Starlette uses first-match routing.
    app.routes.insert(0, Route(card_path, serve_card, methods=["GET"]))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def serve(
    agent: Agent,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/a2a",
    agent_card: dict | None = None,
    log_level: str = "info",
):
    """Run a blocking uvicorn server hosting ``agent`` over A2A.

    Convenience entrypoint — use :func:`make_app` if you need to attach extra
    routes (static UI, custom endpoints) before starting the server.
    """
    app = make_app(agent, path=path, agent_card=agent_card, host=host, port=port)
    print(f"{agent.name} listening on http://{host}:{port}")
    print(f"  RPC:  POST  {path}")
    print(f"  Card: GET   {path}/.well-known/agent-card.json")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
