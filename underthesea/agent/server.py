"""A2A-compatible HTTP server for Agent instances — raw ASGI, no web framework.

Implements the A2A wire protocol — JSON-RPC 2.0 ``message/stream`` over HTTP
with SSE streaming — as a plain ASGI callable. No starlette / fastapi
dependency at the library level; users plug the returned app into any ASGI
server they prefer (uvicorn, hypercorn, daphne, ...).

Endpoints
---------
- ``POST {path}``                              — JSON-RPC ``message/stream``
- ``GET  {path}/.well-known/agent-card.json``  — discoverable AgentCard
- ``OPTIONS *``                                — CORS preflight

One :class:`Agent` is spawned per A2A ``contextId`` so each conversation
keeps its own history. Tool calls are streamed live as ``tool_call``
artifacts via a ContextVar hook around each ``Tool.func``.

Example
-------
::

    from underthesea.agent import Agent, Tool
    from underthesea.agent.server import serve  # needs `pip install uvicorn`

    def add(a: int, b: int) -> int:
        '''Add two integers.'''
        return a + b

    agent = Agent(name="MathAgent", instruction="...", tools=[Tool(add)])
    serve(agent, port=8000, path="/a2a/math")

To use a different ASGI server::

    from underthesea.agent.server import make_app
    app = make_app(agent, path="/a2a/math")
    # then: hypercorn module:app, daphne module:app, etc.
"""
import asyncio
import contextvars
import functools
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime

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
            trace.append({"name": func.__name__, "args": kwargs, "result": result})
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
            {"id": t.name, "name": t.name, "description": t.description, "tags": []}
            for t in agent.tools
        ],
    }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _extract_user_text(parts: list[dict]) -> str:
    for p in parts or []:
        if p.get("kind") == "text":
            return (p.get("text") or "").strip()
    return ""


def _sse_event(rpc_id, result: dict) -> bytes:
    payload = {"id": rpc_id, "jsonrpc": "2.0", "result": result}
    return f"data: {json.dumps(payload)}\n\n".encode()


# ---------- raw ASGI helpers ----------


async def _read_body(receive) -> bytes:
    body, more = b"", True
    while more:
        msg = await receive()
        body += msg.get("body", b"")
        more = msg.get("more_body", False)
    return body


async def _send_bytes(
    send, status: int, body: bytes, content_type: str, extra_headers=()
) -> None:
    headers = [
        (b"content-type", content_type.encode()),
        (b"content-length", str(len(body)).encode()),
        *extra_headers,
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body, "more_body": False})


async def _send_sse(
    send, generator: AsyncIterator[bytes], extra_headers=()
) -> None:
    headers = [
        (b"content-type", b"text/event-stream"),
        (b"cache-control", b"no-cache"),
        (b"x-accel-buffering", b"no"),
        *extra_headers,
    ]
    await send({"type": "http.response.start", "status": 200, "headers": headers})
    async for chunk in generator:
        await send({"type": "http.response.body", "body": chunk, "more_body": True})
    await send({"type": "http.response.body", "body": b"", "more_body": False})


def _cors_headers(request_origin: str | None, allow_origins: list[str]) -> list:
    if not allow_origins:
        return []
    if "*" in allow_origins:
        allowed = request_origin or "*"
    elif request_origin in allow_origins:
        allowed = request_origin
    else:
        return []
    return [
        (b"access-control-allow-origin", allowed.encode()),
        (b"access-control-allow-credentials", b"true"),
        (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
        (b"access-control-allow-headers", b"*"),
    ]


def _origin_of(scope) -> str | None:
    for k, v in scope.get("headers", []):
        if k == b"origin":
            return v.decode()
    return None


async def _handle_lifespan(receive, send) -> None:
    while True:
        msg = await receive()
        if msg["type"] == "lifespan.startup":
            await send({"type": "lifespan.startup.complete"})
        elif msg["type"] == "lifespan.shutdown":
            await send({"type": "lifespan.shutdown.complete"})
            return


class _SessionRouter:
    """Maps A2A contextId → spawned Agent; owns the streaming RPC handler."""

    def __init__(self, template: Agent):
        self._template = template
        self._sessions: dict[str, Agent] = {}

    def _session(self, context_id: str) -> Agent:
        if context_id not in self._sessions:
            self._sessions[context_id] = _spawn_session_agent(self._template)
        return self._sessions[context_id]

    async def handle_rpc(self, receive, send, cors_headers) -> None:
        body = json.loads(await _read_body(receive))
        rpc_id = body.get("id")
        params = body.get("params", {})
        message = params.get("message", {})

        user_text = _extract_user_text(message.get("parts", []))
        context_id = message.get("contextId") or str(uuid.uuid4())
        task_id = message.get("taskId") or str(uuid.uuid4())
        agent = self._session(context_id)

        async def stream():
            for state in ("submitted", "working"):
                yield _sse_event(rpc_id, {
                    "contextId": context_id, "taskId": task_id,
                    "kind": "status-update", "final": False,
                    "status": {"state": state, "timestamp": _now_iso()},
                })

            trace: list[dict] = []
            token = _trace_var.set(trace)
            try:
                reply = await asyncio.to_thread(agent, user_text)
            finally:
                _trace_var.reset(token)

            for step in trace:
                yield _sse_event(rpc_id, {
                    "contextId": context_id, "taskId": task_id,
                    "kind": "artifact-update",
                    "artifact": {
                        "artifactId": str(uuid.uuid4()),
                        "name": "tool_call",
                        "parts": [{"kind": "data", "data": {"tool_call": step}}],
                    },
                })
            yield _sse_event(rpc_id, {
                "contextId": context_id, "taskId": task_id,
                "kind": "artifact-update",
                "artifact": {
                    "artifactId": str(uuid.uuid4()),
                    "name": "reply",
                    "parts": [{"kind": "text", "text": reply}],
                },
            })
            yield _sse_event(rpc_id, {
                "contextId": context_id, "taskId": task_id,
                "kind": "status-update", "final": True,
                "status": {"state": "completed", "timestamp": _now_iso()},
            })

        await _send_sse(send, stream(), extra_headers=cors_headers)


# ---------- public API ----------


ASGIApp = Callable[[dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]], Awaitable[None]]


def make_app(
    agent: Agent,
    *,
    path: str,
    agent_card: dict | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    allow_origins: list[str] | None = None,
) -> ASGIApp:
    """Build a raw ASGI app serving ``agent`` over A2A.

    Returns an async callable ``(scope, receive, send)`` suitable for any
    ASGI server (uvicorn, hypercorn, daphne) and composable with any ASGI
    router (Starlette, FastAPI) via Mount.

    Parameters
    ----------
    agent : Agent
        Template agent; per-session copies share its tools/instruction/provider.
    path : str
        JSON-RPC endpoint, e.g. ``"/a2a/my_agent"``. The agent card is mounted
        at ``f"{path}/.well-known/agent-card.json"``.
    agent_card : dict, optional
        Override the auto-generated AgentCard (served verbatim).
    host, port : str, int
        Used only to fill the ``url`` field of the auto-generated card.
    allow_origins : list[str], optional
        CORS origins. Defaults to ``["*"]``.
    """
    _wrap_tools_inplace(agent.tools)

    card_dict = agent_card or _auto_agent_card(agent, f"http://{host}:{port}{path}")
    card_bytes = json.dumps(card_dict).encode("utf-8")
    card_path = f"{path}/.well-known/agent-card.json"

    router = _SessionRouter(agent)
    allow = allow_origins if allow_origins is not None else ["*"]

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            await _handle_lifespan(receive, send)
            return
        if scope["type"] != "http":
            return

        cors = _cors_headers(_origin_of(scope), allow)
        method, p = scope["method"], scope["path"]

        if method == "OPTIONS":
            await _send_bytes(send, 204, b"", "text/plain", cors)
            return
        if method == "GET" and p == card_path:
            await _send_bytes(send, 200, card_bytes, "application/json", cors)
            return
        if method == "POST" and p == path:
            await router.handle_rpc(receive, send, cors)
            return
        await _send_bytes(send, 404, b"Not Found", "text/plain", cors)

    return app


def serve(
    agent: Agent,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/a2a",
    agent_card: dict | None = None,
    log_level: str = "info",
) -> None:
    """Run a blocking uvicorn server hosting ``agent`` over A2A.

    Convenience entrypoint — lazy-imports uvicorn. Use :func:`make_app` if you
    need to mount extra routes (static UI, custom endpoints) or pick a
    different ASGI server.
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "uvicorn is required for serve(). "
            "Install with: pip install 'underthesea[agent-server]'"
        ) from e

    app = make_app(agent, path=path, agent_card=agent_card, host=host, port=port)
    print(f"{agent.name} listening on http://{host}:{port}")
    print(f"  RPC:  POST  {path}")
    print(f"  Card: GET   {path}/.well-known/agent-card.json")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
