"""Tool class for agent function calling."""
import inspect
import json
import re
import typing
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin


class Tool:
    """Wraps a function as an agent tool for OpenAI function calling."""

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ):
        """
        Initialize a Tool.

        Parameters
        ----------
        func : Callable
            The function to wrap as a tool.
        name : str, optional
            Tool name. Defaults to function name.
        description : str, optional
            Tool description. Defaults to function docstring (summary line).
        """
        self.func = func
        self.name = name or func.__name__
        docstring = func.__doc__ or ""
        self._param_docs = _parse_param_docs(docstring)
        self.description = description or _docstring_summary(docstring) or f"Tool: {self.name}"
        self.parameters = self._extract_parameters(func)

    def _extract_parameters(self, func: Callable) -> dict:
        """Extract JSON schema from function signature."""
        sig = inspect.signature(func)
        try:
            hints = typing.get_type_hints(func)
        except Exception:
            hints = getattr(func, "__annotations__", {}) or {}

        properties: dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = hints.get(param_name, str)
            schema = _python_type_to_schema(param_type)
            doc = self._param_docs.get(param_name)
            schema["description"] = doc or f"Parameter {param_name}"
            if param.default is not inspect.Parameter.empty:
                # JSON schema "default" is informational; many providers honour it.
                try:
                    json.dumps(param.default)
                    schema["default"] = param.default
                except TypeError:
                    pass
            else:
                required.append(param_name)
            properties[param_name] = schema

        return {"type": "object", "properties": properties, "required": required}

    def to_openai_tool(self) -> dict:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_tool(self) -> dict:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def execute(self, arguments: dict) -> str:
        """Execute tool and return JSON-serialized result."""
        result = self.func(**arguments)
        return json.dumps(result, ensure_ascii=False, default=str)

    def __call__(self, **kwargs) -> Any:
        """Call the underlying function directly."""
        return self.func(**kwargs)


def tool(
    _func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable], Tool]:
    """Decorator that turns a function into a :class:`Tool`.

    Examples
    --------
    >>> @tool
    ... def add(a: int, b: int) -> int:
    ...     '''Add two numbers.'''
    ...     return a + b
    >>> add.name
    'add'

    >>> @tool(name="search", description="Search the web")
    ... def _search(query: str) -> list:
    ...     ...
    """
    def wrap(fn: Callable) -> Tool:
        return Tool(fn, name=name, description=description)

    if _func is None:
        return wrap
    return wrap(_func)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_PRIMITIVE_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _python_type_to_schema(py_type: Any) -> dict:
    """Convert a Python type annotation to a JSON schema fragment.

    Handles primitives, ``list[T]``, ``dict[K, V]``, ``Optional[T]``,
    ``Union[...]`` and ``Literal[...]``.  Unknown types fall back to ``string``
    so the LLM still gets a usable schema.
    """
    if py_type is inspect.Parameter.empty or py_type is None:
        return {"type": "string"}

    if py_type in _PRIMITIVE_TYPES:
        return {"type": _PRIMITIVE_TYPES[py_type]}

    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Literal:
        return {"type": _literal_type(args), "enum": list(args)}

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[T] — schema is T, nullability handled by "required".
            return _python_type_to_schema(non_none[0])
        # Multi-type union — pick the first known primitive, fall back to string.
        for arg in non_none:
            if arg in _PRIMITIVE_TYPES:
                return {"type": _PRIMITIVE_TYPES[arg]}
        return {"type": "string"}

    if origin in (list, tuple, set, frozenset):
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _python_type_to_schema(args[0])
        return schema

    if origin is dict:
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = _python_type_to_schema(args[1])
        return schema

    return {"type": "string"}


def _literal_type(values: tuple) -> str:
    """Infer the JSON type that fits all literal values."""
    types = {type(v) for v in values}
    if types <= {str}:
        return "string"
    if types <= {int, bool}:
        return "integer" if types == {int} else "boolean"
    if types <= {int, float}:
        return "number"
    return "string"


_PARAM_HEADER_RE = re.compile(
    r"^\s*(Args|Arguments|Parameters|Params)\s*:?\s*$", re.IGNORECASE,
)
_PARAM_LINE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*[:\-]\s*(?P<desc>.+)$",
)


def _docstring_summary(docstring: str) -> str:
    """Return the first non-empty line of a docstring."""
    for line in (docstring or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _parse_param_docs(docstring: str) -> dict[str, str]:
    """Parse Google- and NumPy-style parameter descriptions from a docstring.

    Recognises sections headed by ``Args:``, ``Arguments:``, ``Parameters`` or
    ``Params:``.  Lines look like ``name: description`` or ``name (type):
    description``.  Returns ``{}`` when nothing parses — callers should fall
    back to a generic description.
    """
    if not docstring:
        return {}

    lines = inspect.cleandoc(docstring).splitlines()
    in_section = False
    params: dict[str, str] = {}
    current: str | None = None

    for line in lines:
        if _PARAM_HEADER_RE.match(line):
            in_section = True
            current = None
            continue
        if in_section and not line.strip():
            # Blank line ends Google-style block but not NumPy-style; be lenient
            # and just reset the "current" continuation tracker.
            current = None
            continue
        if in_section and line and not line.startswith((" ", "\t")) and ":" not in line:
            # New unindented non-param header — section likely ended.
            in_section = False
            current = None
            continue
        if in_section:
            match = _PARAM_LINE_RE.match(line)
            if match:
                current = match.group("name")
                params[current] = match.group("desc").strip()
            elif current and line.strip():
                params[current] = f"{params[current]} {line.strip()}"

    return params
