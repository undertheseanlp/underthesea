"""Decorator-based tracing for functions and methods."""

from __future__ import annotations

import contextvars
import functools
import inspect
import json

from underthesea.agent.trace.base import BaseTracer

# Context variable to track the active trace — enables automatic nesting.
_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)
_current_tracer: contextvars.ContextVar[BaseTracer | None] = contextvars.ContextVar(
    "tracer", default=None
)


def trace(tracer: BaseTracer, name: str | None = None):
    """Decorator that traces a function call.

    At the top level the decorator creates a **trace**.  When called inside
    an already-traced function it creates a **span** (child) instead, so
    nesting works automatically.

    Parameters
    ----------
    tracer : BaseTracer
        Tracer instance (``LocalTracer``, ``LangfuseTracer``, …).
    name : str, optional
        Override the span/trace name.  Defaults to ``func.__name__``.

    Examples
    --------
    Simple — every call creates its own trace::

        from underthesea.agent import LocalTracer
        from underthesea.agent.trace import trace

        tracer = LocalTracer()

        @trace(tracer)
        def greet(name: str) -> str:
            return f"Hello {name}"

        greet("World")

    Nested — inner calls become spans of the outer trace::

        @trace(tracer)
        def pipeline(text):
            cleaned = preprocess(text)
            return classify(cleaned)

        @trace(tracer)
        def preprocess(text):
            return text.strip().lower()

        @trace(tracer)
        def classify(text):
            return "positive"

        pipeline("  Hello!  ")
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trace_name = name or func.__name__
            input_data = _format_input(func, args, kwargs)
            parent_trace_id = _current_trace_id.get()

            if parent_trace_id is None:
                # Top-level → new trace
                trace_id = tracer.start_trace(name=trace_name, input=input_data)
                token_tid = _current_trace_id.set(trace_id)
                token_tr = _current_tracer.set(tracer)
                try:
                    result = func(*args, **kwargs)
                    tracer.end_trace(trace_id, output=_safe_repr(result))
                    return result
                except Exception as e:
                    tracer.end_trace(trace_id, status="error", error=str(e))
                    raise
                finally:
                    _current_trace_id.reset(token_tid)
                    _current_tracer.reset(token_tr)
            else:
                # Nested → span under the active trace
                active_tracer = _current_tracer.get() or tracer
                span_id = active_tracer.start_span(
                    parent_trace_id, name=trace_name, input=input_data,
                )
                try:
                    result = func(*args, **kwargs)
                    active_tracer.end_span(span_id, output=_safe_repr(result))
                    return result
                except Exception as e:
                    active_tracer.end_span(span_id, status="error", error=str(e))
                    raise

        return wrapper

    return decorator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _format_input(func, args, kwargs) -> str:
    """Build a readable string from function arguments."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
        # Remove 'self' / 'cls' for methods
        params.pop("self", None)
        params.pop("cls", None)
        return json.dumps(params, ensure_ascii=False, default=str)
    except Exception:
        # Fallback: positional + keyword representation
        parts = [repr(a) for a in args]
        parts += [f"{k}={v!r}" for k, v in kwargs.items()]
        return ", ".join(parts)


def _safe_repr(obj) -> str:
    """Convert a return value to a string, safely."""
    if obj is None:
        return "None"
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(obj)
