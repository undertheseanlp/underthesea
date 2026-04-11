"""Local file-based tracer — saves traces as JSON for offline inspection."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from underthesea.agent.trace.base import BaseTracer


class LocalTracer(BaseTracer):
    """Saves each agent call as a JSON trace file.

    Parameters
    ----------
    trace_dir : str
        Directory for trace files (created automatically). Default ``".traces"``.
    console : bool
        Print trace events to stderr/stdout while running. Default ``True``.

    Examples
    --------
    >>> from underthesea.agent import Agent, LocalTracer, calculator_tool
    >>> tracer = LocalTracer()
    >>> bot = Agent("demo", tools=[calculator_tool], tracer=tracer)
    >>> bot("What is 2+2?")
    """

    def __init__(self, trace_dir: str = ".traces", console: bool = True):
        self._trace_dir = Path(trace_dir)
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._console = console
        # In-flight state
        self._traces: dict[str, dict] = {}
        self._spans: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # BaseTracer implementation
    # ------------------------------------------------------------------

    def start_trace(self, *, name: str, input: str, metadata: dict | None = None) -> str:
        trace_id = uuid4().hex[:12]
        self._traces[trace_id] = {
            "id": trace_id,
            "name": name,
            "input": input,
            "start_time": _now(),
            "spans": [],
            "metadata": metadata or {},
        }
        if self._console:
            _print(f">> Trace [{trace_id}] {name}")
            _print(f"   Input: {_trunc(str(input))}")
        return trace_id

    def end_trace(
        self, trace_id: str, *, output: str | None = None,
        status: str = "ok", error: str | None = None,
    ):
        trace = self._traces.pop(trace_id, None)
        if not trace:
            return
        trace["output"] = output
        trace["end_time"] = _now()
        trace["status"] = status
        if error:
            trace["error"] = error
        trace["duration_ms"] = _duration_ms(trace["start_time"], trace["end_time"])

        ts = datetime.fromisoformat(trace["start_time"]).strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_trace_{trace_id}.json"
        path = self._trace_dir / filename
        with open(path, "w") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False, default=str)

        if self._console:
            icon = "[ok]" if status == "ok" else "[error]"
            _print(f"<< Trace [{trace_id}] {icon} {trace['duration_ms']}ms -> {path}")

    def start_generation(
        self, trace_id: str, *, name: str, model: str | None = None,
        input: object = None, metadata: dict | None = None,
    ) -> str:
        span_id = uuid4().hex[:12]
        self._spans[span_id] = {
            "id": span_id,
            "trace_id": trace_id,
            "name": name,
            "type": "generation",
            "model": model,
            "input": input,
            "start_time": _now(),
            "metadata": metadata or {},
        }
        if self._console:
            _print(f"   |-- Generation: {name} ({model or '?'})")
        return span_id

    def end_generation(
        self, span_id: str, *, output: object = None,
        usage: dict | None = None, status: str = "ok", error: str | None = None,
    ):
        span = self._spans.pop(span_id, None)
        if not span:
            return
        span["output"] = output
        span["end_time"] = _now()
        span["status"] = status
        span["duration_ms"] = _duration_ms(span["start_time"], span["end_time"])
        if usage:
            span["usage"] = usage
        if error:
            span["error"] = error

        trace = self._traces.get(span["trace_id"])
        if trace:
            trace["spans"].append(span)

        if self._console:
            parts = [f"   |     {span['duration_ms']}ms"]
            if usage:
                parts.append(
                    f" | {usage.get('input_tokens', '?')}->{usage.get('output_tokens', '?')} tokens"
                )
            if status != "ok":
                parts.append(f" | {status}: {error or ''}")
            _print("".join(parts))

    def start_span(
        self, trace_id: str, *, name: str,
        input: object = None, metadata: dict | None = None,
    ) -> str:
        span_id = uuid4().hex[:12]
        self._spans[span_id] = {
            "id": span_id,
            "trace_id": trace_id,
            "name": name,
            "type": "span",
            "input": input,
            "start_time": _now(),
            "metadata": metadata or {},
        }
        if self._console:
            _print(f"   |-- Tool: {name}")
        return span_id

    def end_span(
        self, span_id: str, *, output: object = None,
        status: str = "ok", error: str | None = None,
    ):
        span = self._spans.pop(span_id, None)
        if not span:
            return
        span["output"] = output
        span["end_time"] = _now()
        span["status"] = status
        span["duration_ms"] = _duration_ms(span["start_time"], span["end_time"])
        if error:
            span["error"] = error

        trace = self._traces.get(span["trace_id"])
        if trace:
            trace["spans"].append(span)

        if self._console:
            result_str = _trunc(str(output), 80) if output else ""
            _print(f"   |     {span['duration_ms']}ms | {result_str}")

    # ------------------------------------------------------------------
    # Utility: browse saved traces
    # ------------------------------------------------------------------

    def list_traces(self, limit: int = 20) -> list[dict]:
        """List recent traces (newest first)."""
        files = sorted(
            self._trace_dir.glob("*_trace_*.json"),
            key=os.path.getmtime,
            reverse=True,
        )
        traces = []
        for f in files[:limit]:
            with open(f) as fh:
                traces.append(json.load(fh))
        return traces

    def get_trace(self, trace_id: str) -> dict | None:
        """Load a single trace by ID."""
        matches = list(self._trace_dir.glob(f"*_trace_{trace_id}.json"))
        if matches:
            with open(matches[0]) as f:
                return json.load(f)
        return None

    def print_trace(self, trace: dict | str):
        """Pretty-print a trace to the console."""
        if isinstance(trace, str):
            trace = self.get_trace(trace)
        if not trace:
            print("Trace not found.")
            return

        w = 60
        print(f"\n{'=' * w}")
        print(f"Trace: {trace['name']} [{trace['id']}]")
        print(f"Status: {trace['status']} | Duration: {trace.get('duration_ms', '?')}ms")
        print(f"Time: {trace['start_time']}")
        print(f"{'-' * w}")
        print(f"Input: {trace['input']}")
        print(f"{'-' * w}")

        spans = trace.get("spans", [])
        for i, span in enumerate(spans):
            is_last = i == len(spans) - 1
            prefix = "`--" if is_last else "|--"
            tag = "[LLM]" if span["type"] == "generation" else "[Tool]"

            print(f"{prefix} {tag} {span['name']} ({span.get('duration_ms', '?')}ms)")

            indent = "    " if is_last else "|   "
            if span["type"] == "generation":
                if span.get("model"):
                    print(f"{indent}Model: {span['model']}")
                usage = span.get("usage")
                if usage:
                    print(
                        f"{indent}Tokens: {usage.get('input_tokens', '?')} in"
                        f" -> {usage.get('output_tokens', '?')} out"
                    )
            else:
                if span.get("input"):
                    print(f"{indent}Input: {_trunc(str(span['input']), 100)}")
                if span.get("output"):
                    print(f"{indent}Output: {_trunc(str(span['output']), 100)}")

        print(f"{'-' * w}")
        print(f"Output: {_trunc(str(trace.get('output', '')), 200)}")
        print(f"{'=' * w}\n")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat()


def _duration_ms(start: str, end: str) -> int:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    return int((e - s).total_seconds() * 1000)


def _trunc(s: str, max_len: int = 120) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s


def _print(msg: str):
    print(msg)
