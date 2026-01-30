"""Default tools for agent - common utilities like LangChain/OpenAI tools."""
import json
import math
import subprocess
from datetime import datetime

from underthesea.agent.tools import Tool

# ============================================================================
# Date/Time Tools
# ============================================================================


def _get_current_datetime() -> dict:
    """Get the current date and time."""
    now = datetime.now()
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timestamp": int(now.timestamp()),
    }


current_datetime_tool = Tool(
    _get_current_datetime,
    name="get_current_datetime",
    description="Get the current date, time, and weekday. Use when user asks about current time or date.",
)


# ============================================================================
# Calculator / Math Tools
# ============================================================================


def _calculator(expression: str) -> dict:
    """
    Evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, abs, round, pi, e
    """
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


calculator_tool = Tool(
    _calculator,
    name="calculator",
    description="Evaluate mathematical expressions. Supports basic arithmetic (+, -, *, /, **) and functions (sqrt, sin, cos, tan, log, abs, round, pi, e). Example: 'sqrt(16) + 2 * 3'",
)


# ============================================================================
# Web/URL Tools
# ============================================================================


def _fetch_url(url: str) -> dict:
    """Fetch content from a URL."""
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read().decode("utf-8", errors="ignore")
            # Limit content size
            if len(content) > 10000:
                content = content[:10000] + "... [truncated]"
            return {
                "url": url,
                "status": response.status,
                "content": content,
            }
    except Exception as e:
        return {"url": url, "error": str(e)}


fetch_url_tool = Tool(
    _fetch_url,
    name="fetch_url",
    description="Fetch and read content from a URL. Returns the text content of a webpage.",
)


def _web_search(query: str) -> dict:
    """
    Search the web using DuckDuckGo (no API key required).
    Returns top search results.
    """
    try:
        import re
        import urllib.parse
        import urllib.request

        # Use DuckDuckGo HTML search
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible; underthesea-agent)"}
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")

        # Extract results (simplified parsing)
        results = []
        # Find result blocks
        pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, html)

        for href, title in matches[:5]:
            results.append({"title": title.strip(), "url": href})

        return {"query": query, "results": results}
    except Exception as e:
        return {"query": query, "error": str(e), "results": []}


web_search_tool = Tool(
    _web_search,
    name="web_search",
    description="Search the web for information. Returns top search results with titles and URLs. Use for finding current information, facts, or news.",
)


# ============================================================================
# Wikipedia Tool
# ============================================================================


def _wikipedia_search(query: str, lang: str = "vi") -> dict:
    """
    Search and get summary from Wikipedia.
    Supports Vietnamese (vi) and English (en).
    """
    try:
        import urllib.parse
        import urllib.request

        encoded_query = urllib.parse.quote(query)
        api_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"

        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "underthesea-agent/1.0"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        return {
            "title": data.get("title", ""),
            "summary": data.get("extract", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except Exception as e:
        return {"query": query, "error": str(e)}


wikipedia_tool = Tool(
    _wikipedia_search,
    name="wikipedia",
    description="Search Wikipedia and get article summary. Use 'lang' parameter: 'vi' for Vietnamese (default), 'en' for English.",
)


# ============================================================================
# Shell/Command Tool
# ============================================================================


def _run_shell_command(command: str) -> dict:
    """
    Run a shell command and return the output.
    Use with caution - only for safe commands.
    """
    # Block dangerous commands
    dangerous = ["rm -rf", "mkfs", "dd if=", ":(){", "fork bomb", "> /dev/"]
    for d in dangerous:
        if d in command.lower():
            return {"command": command, "error": "Command blocked for safety"}

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout or result.stderr
        if len(output) > 5000:
            output = output[:5000] + "... [truncated]"
        return {
            "command": command,
            "output": output,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"command": command, "error": "Command timed out after 30 seconds"}
    except Exception as e:
        return {"command": command, "error": str(e)}


shell_tool = Tool(
    _run_shell_command,
    name="shell",
    description="Run a shell command and get the output. Use for system operations, file listing, etc. Example: 'ls -la', 'pwd', 'cat file.txt'",
)


# ============================================================================
# Python Code Execution Tool
# ============================================================================


def _run_python(code: str) -> dict:
    """
    Execute Python code and return the result.
    The code should use print() to output results.
    """
    try:
        # Create a restricted globals dict
        allowed_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "bool": bool,
                "abs": abs,
                "sum": sum,
                "min": min,
                "max": max,
                "round": round,
                "sorted": sorted,
                "reversed": reversed,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "any": any,
                "all": all,
            }
        }

        # Capture stdout
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            exec(code, allowed_globals)
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        return {"code": code, "output": output.strip()}
    except Exception as e:
        return {"code": code, "error": str(e)}


python_tool = Tool(
    _run_python,
    name="python",
    description="Execute Python code. Use print() to output results. Supports basic Python operations and data structures.",
)


# ============================================================================
# JSON Tool
# ============================================================================


def _parse_json(json_string: str) -> dict:
    """Parse a JSON string and return the parsed object."""
    try:
        parsed = json.loads(json_string)
        return {"success": True, "data": parsed}
    except json.JSONDecodeError as e:
        return {"success": False, "error": str(e)}


json_parse_tool = Tool(
    _parse_json,
    name="parse_json",
    description="Parse a JSON string into a Python object. Useful for processing API responses or structured data.",
)


# ============================================================================
# String Tools
# ============================================================================


def _string_length(text: str) -> dict:
    """Count characters and words in text."""
    words = text.split()
    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "characters": len(text),
        "words": len(words),
        "lines": text.count("\n") + 1,
    }


string_length_tool = Tool(
    _string_length,
    name="string_length",
    description="Count the number of characters, words, and lines in a text.",
)


# ============================================================================
# File Tools
# ============================================================================


def _read_file(file_path: str) -> dict:
    """Read content from a file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        if len(content) > 10000:
            content = content[:10000] + "... [truncated]"
        return {"file_path": file_path, "content": content}
    except Exception as e:
        return {"file_path": file_path, "error": str(e)}


def _write_file(file_path: str, content: str) -> dict:
    """Write content to a file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"file_path": file_path, "success": True, "bytes_written": len(content)}
    except Exception as e:
        return {"file_path": file_path, "error": str(e)}


def _list_directory(path: str = ".") -> dict:
    """List files and directories in a path."""
    import os

    try:
        items = os.listdir(path)
        files = []
        dirs = []
        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                dirs.append(item)
            else:
                files.append(item)
        return {"path": path, "directories": sorted(dirs), "files": sorted(files)}
    except Exception as e:
        return {"path": path, "error": str(e)}


read_file_tool = Tool(
    _read_file,
    name="read_file",
    description="Read and return the content of a file.",
)

write_file_tool = Tool(
    _write_file,
    name="write_file",
    description="Write content to a file. Creates the file if it doesn't exist.",
)

list_directory_tool = Tool(
    _list_directory,
    name="list_directory",
    description="List all files and directories in a given path. Defaults to current directory.",
)


# ============================================================================
# Tool Collections
# ============================================================================

# Core utility tools (safe, no external dependencies)
core_tools = [
    current_datetime_tool,
    calculator_tool,
    string_length_tool,
    json_parse_tool,
]

# Web tools (requires network access)
web_tools = [
    web_search_tool,
    fetch_url_tool,
    wikipedia_tool,
]

# System tools (file and shell operations)
system_tools = [
    read_file_tool,
    write_file_tool,
    list_directory_tool,
    shell_tool,
    python_tool,
]

# All default tools
default_tools = core_tools + web_tools + system_tools
