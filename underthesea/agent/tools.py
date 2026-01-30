"""Tool class for agent function calling."""
import inspect
import json
from collections.abc import Callable
from typing import Any


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
            Tool description. Defaults to function docstring.
        """
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or f"Tool: {self.name}"
        self.parameters = self._extract_parameters(func)

    def _extract_parameters(self, func: Callable) -> dict:
        """Extract JSON schema from function signature."""
        sig = inspect.signature(func)
        hints = getattr(func, "__annotations__", {})

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue
            param_type = hints.get(param_name, str)
            properties[param_name] = {
                "type": self._python_type_to_json(param_type),
                "description": f"Parameter {param_name}",
            }
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {"type": "object", "properties": properties, "required": required}

    def _python_type_to_json(self, py_type) -> str:
        """Convert Python type to JSON schema type."""
        mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
        }
        return mapping.get(py_type, "string")

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

    def execute(self, arguments: dict) -> str:
        """Execute tool and return JSON result."""
        result = self.func(**arguments)
        return json.dumps(result, ensure_ascii=False, default=str)

    def __call__(self, **kwargs) -> Any:
        """Call the underlying function directly."""
        return self.func(**kwargs)
