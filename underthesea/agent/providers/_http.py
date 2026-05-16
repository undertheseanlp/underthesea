"""Shared HTTP helper using only Python stdlib."""

import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Generator


class LLMError(Exception):
    """Raised when an LLM API returns an error."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


# Status codes worth retrying. 408/425 are transient client-side hints; 429 is
# rate limiting; 5xx are upstream failures.
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 529}


def _default_max_retries() -> int:
    """Override default with ``UNDERTHESEA_LLM_MAX_RETRIES`` env var."""
    try:
        return max(0, int(os.environ.get("UNDERTHESEA_LLM_MAX_RETRIES", "3")))
    except ValueError:
        return 3


def _retry_sleep(attempt: int, base_delay: float, max_delay: float) -> float:
    """Exponential backoff with full jitter."""
    delay = min(max_delay, base_delay * (2 ** attempt))
    return random.uniform(0, delay)


def post_json(
    url: str,
    headers: dict,
    body: dict,
    timeout: int = 120,
    max_retries: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> dict:
    """POST JSON and return parsed response.

    Retries transient failures (429, 5xx, connection / timeout errors) with
    exponential backoff plus jitter.  Set ``max_retries=0`` to disable, or use
    the ``UNDERTHESEA_LLM_MAX_RETRIES`` env var to change the default.
    """
    if max_retries is None:
        max_retries = _default_max_retries()

    data = json.dumps(body).encode("utf-8")
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            err = LLMError(e.code, error_body)
            if attempt < max_retries and e.code in _RETRYABLE_STATUS:
                time.sleep(_retry_sleep(attempt, base_delay, max_delay))
                last_exc = err
                continue
            raise err from e
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            if attempt < max_retries:
                time.sleep(_retry_sleep(attempt, base_delay, max_delay))
                last_exc = e
                continue
            raise

    # Defensive — loop above always returns or raises, but keep mypy happy.
    if last_exc:
        raise last_exc
    raise RuntimeError("post_json exhausted retries without raising")  # pragma: no cover


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
