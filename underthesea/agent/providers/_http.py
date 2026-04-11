"""Shared HTTP helper using only Python stdlib."""

import json
import urllib.error
import urllib.request
from collections.abc import Generator


class LLMError(Exception):
    """Raised when an LLM API returns an error."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


def post_json(url: str, headers: dict, body: dict, timeout: int = 120) -> dict:
    """POST JSON and return parsed response."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise LLMError(e.code, error_body) from e


def stream_sse(url: str, headers: dict, body: dict, timeout: int = 300) -> Generator[dict]:
    """POST JSON and yield parsed SSE data events.

    Yields dicts from each `data: {...}` line. Stops on `data: [DONE]` or stream end.
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise LLMError(e.code, error_body) from e

    try:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
    finally:
        resp.close()
