# Default Tools Reference

This document provides detailed documentation for all built-in tools in the underthesea agent module.

## Tool Collections

```python
from underthesea.agent import (
    default_tools,   # All 12 tools
    core_tools,      # 4 safe tools
    web_tools,       # 3 web tools
    system_tools,    # 5 system tools
)
```

| Collection | Tools | Safety Level |
|------------|-------|--------------|
| `core_tools` | 4 | Safe - No external access |
| `web_tools` | 3 | Network - HTTP requests |
| `system_tools` | 5 | System - File/shell access |
| `default_tools` | 12 | All of the above |

---

## Core Tools

### current_datetime_tool

Get the current date, time, and weekday.

**Function:** `get_current_datetime() -> dict`

**Parameters:** None

**Returns:**
```python
{
    "datetime": "2025-01-30T10:30:00.123456",  # ISO format
    "date": "2025-01-30",                       # YYYY-MM-DD
    "time": "10:30:00",                         # HH:MM:SS
    "weekday": "Thursday",                      # Day name
    "timestamp": 1738236600                     # Unix timestamp
}
```

**Example:**
```python
from underthesea.agent import current_datetime_tool

result = current_datetime_tool()
print(f"Today is {result['weekday']}, {result['date']}")
```

---

### calculator_tool

Evaluate mathematical expressions safely.

**Function:** `calculator(expression: str) -> dict`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `expression` | `str` | Yes | Math expression to evaluate |

**Supported Operations:**
| Category | Operations |
|----------|------------|
| Arithmetic | `+`, `-`, `*`, `/`, `**` (power), `%` (modulo) |
| Functions | `sqrt`, `sin`, `cos`, `tan`, `log`, `log10`, `exp` |
| Rounding | `abs`, `round`, `floor`, `ceil` |
| Aggregation | `min`, `max`, `sum`, `pow` |
| Constants | `pi` (3.14159...), `e` (2.71828...) |

**Returns:**
```python
# Success
{"expression": "sqrt(16) + 2", "result": 6.0}

# Error
{"expression": "invalid", "error": "name 'invalid' is not defined"}
```

**Examples:**
```python
from underthesea.agent import calculator_tool

# Basic arithmetic
calculator_tool(expression="2 + 3 * 4")  # result: 14

# Functions
calculator_tool(expression="sqrt(144)")  # result: 12.0
calculator_tool(expression="sin(pi/2)")  # result: 1.0

# Complex expressions
calculator_tool(expression="log(e**2)")  # result: 2.0
calculator_tool(expression="round(pi, 2)")  # result: 3.14
```

**Security:** Uses restricted `eval` with only safe math functions allowed.

---

### string_length_tool

Count characters, words, and lines in text.

**Function:** `string_length(text: str) -> dict`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | `str` | Yes | Text to analyze |

**Returns:**
```python
{
    "text": "Hello World...",  # First 100 chars (truncated if longer)
    "characters": 11,           # Total character count
    "words": 2,                 # Word count (split by whitespace)
    "lines": 1                  # Line count (newlines + 1)
}
```

**Example:**
```python
from underthesea.agent import string_length_tool

result = string_length_tool(text="Hello World\nThis is a test")
# {'text': 'Hello World\nThis is a test', 'characters': 27, 'words': 5, 'lines': 2}
```

---

### json_parse_tool

Parse a JSON string into a Python object.

**Function:** `parse_json(json_string: str) -> dict`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `json_string` | `str` | Yes | JSON string to parse |

**Returns:**
```python
# Success
{"success": True, "data": <parsed object>}

# Error
{"success": False, "error": "Expecting property name: line 1 column 2"}
```

**Example:**
```python
from underthesea.agent import json_parse_tool

# Valid JSON
result = json_parse_tool(json_string='{"name": "test", "value": 123}')
# {'success': True, 'data': {'name': 'test', 'value': 123}}

# Invalid JSON
result = json_parse_tool(json_string='not json')
# {'success': False, 'error': '...'}
```

---

## Web Tools

### web_search_tool

Search the web using DuckDuckGo (no API key required).

**Function:** `web_search(query: str) -> dict`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | `str` | Yes | Search query |

**Returns:**
```python
{
    "query": "python tutorial",
    "results": [
        {"title": "Python Tutorial", "url": "https://..."},
        {"title": "Learn Python", "url": "https://..."},
        # Up to 5 results
    ]
}
```

**Example:**
```python
from underthesea.agent import web_search_tool

result = web_search_tool(query="Vietnamese NLP")
for r in result["results"]:
    print(f"{r['title']}: {r['url']}")
```

**Notes:**
- Uses DuckDuckGo HTML search
- Returns up to 5 results
- 10 second timeout
- No API key required

---

### fetch_url_tool

Fetch and read content from a URL.

**Function:** `fetch_url(url: str) -> dict`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `url` | `str` | Yes | URL to fetch |

**Returns:**
```python
# Success
{
    "url": "https://example.com",
    "status": 200,
    "content": "<html>..."  # Up to 10,000 chars
}

# Error
{"url": "https://...", "error": "HTTP Error 404: Not Found"}
```

**Example:**
```python
from underthesea.agent import fetch_url_tool

result = fetch_url_tool(url="https://example.com")
if "error" not in result:
    print(f"Status: {result['status']}")
    print(f"Content length: {len(result['content'])}")
```

**Notes:**
- 10 second timeout
- Content truncated to 10,000 characters
- UTF-8 decoding with error handling

---

### wikipedia_tool

Search Wikipedia and get article summary.

**Function:** `wikipedia_search(query: str, lang: str = "vi") -> dict`

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | `str` | Yes | - | Search query / article title |
| `lang` | `str` | No | `"vi"` | Language code (`vi`, `en`, etc.) |

**Returns:**
```python
# Success
{
    "title": "Hà Nội",
    "summary": "Hà Nội là thủ đô của Việt Nam...",
    "url": "https://vi.wikipedia.org/wiki/Hà_Nội"
}

# Not found
{"query": "xyz123", "error": "HTTP Error 404: Not Found"}
```

**Example:**
```python
from underthesea.agent import wikipedia_tool

# Vietnamese Wikipedia
result = wikipedia_tool(query="Hà Nội", lang="vi")
print(result["summary"])

# English Wikipedia
result = wikipedia_tool(query="Vietnam", lang="en")
print(result["summary"])
```

**Notes:**
- Uses Wikipedia REST API
- Returns article summary (not full text)
- 10 second timeout

---

## System Tools

### read_file_tool

Read content from a file.

**Function:** `read_file(file_path: str) -> dict`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file_path` | `str` | Yes | Path to file |

**Returns:**
```python
# Success
{"file_path": "/path/to/file.txt", "content": "file contents..."}

# Error
{"file_path": "/path/to/file.txt", "error": "No such file or directory"}
```

**Example:**
```python
from underthesea.agent import read_file_tool

result = read_file_tool(file_path="README.md")
if "error" not in result:
    print(result["content"])
```

**Notes:**
- UTF-8 encoding
- Content truncated to 10,000 characters

---

### write_file_tool

Write content to a file.

**Function:** `write_file(file_path: str, content: str) -> dict`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file_path` | `str` | Yes | Path to file |
| `content` | `str` | Yes | Content to write |

**Returns:**
```python
# Success
{"file_path": "/path/to/file.txt", "success": True, "bytes_written": 123}

# Error
{"file_path": "/path/to/file.txt", "error": "Permission denied"}
```

**Example:**
```python
from underthesea.agent import write_file_tool

result = write_file_tool(file_path="output.txt", content="Hello World")
print(f"Wrote {result['bytes_written']} bytes")
```

**Notes:**
- Creates file if not exists
- Overwrites existing content
- UTF-8 encoding

---

### list_directory_tool

List files and directories in a path.

**Function:** `list_directory(path: str = ".") -> dict`

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `path` | `str` | No | `"."` | Directory path |

**Returns:**
```python
{
    "path": "/home/user",
    "directories": ["Documents", "Downloads"],  # Sorted
    "files": ["file.txt", "notes.md"]           # Sorted
}
```

**Example:**
```python
from underthesea.agent import list_directory_tool

result = list_directory_tool(path=".")
print(f"Directories: {result['directories']}")
print(f"Files: {result['files']}")
```

---

### shell_tool

Run a shell command and return the output.

**Function:** `run_shell_command(command: str) -> dict`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `command` | `str` | Yes | Shell command to execute |

**Returns:**
```python
# Success
{
    "command": "ls -la",
    "output": "total 16\ndrwxr-xr-x...",
    "return_code": 0
}

# Error
{"command": "invalid_cmd", "output": "command not found", "return_code": 127}

# Blocked
{"command": "rm -rf /", "error": "Command blocked for safety"}
```

**Blocked Commands:**
```python
["rm -rf", "mkfs", "dd if=", ":(){", "fork bomb", "> /dev/"]
```

**Example:**
```python
from underthesea.agent import shell_tool

result = shell_tool(command="pwd")
print(result["output"])

result = shell_tool(command="git status")
print(result["output"])
```

**Notes:**
- 30 second timeout
- Output truncated to 5,000 characters
- Dangerous commands blocked

---

### python_tool

Execute Python code in a restricted environment.

**Function:** `run_python(code: str) -> dict`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `code` | `str` | Yes | Python code to execute |

**Returns:**
```python
# Success
{"code": "print(2 + 2)", "output": "4"}

# Error
{"code": "import os", "error": "name 'os' is not defined"}
```

**Allowed Built-ins:**
```python
print, len, range, str, int, float, list, dict, set, tuple, bool,
abs, sum, min, max, round, sorted, reversed, enumerate, zip, map, filter, any, all
```

**Example:**
```python
from underthesea.agent import python_tool

# Simple calculation
result = python_tool(code="print(sum(range(10)))")
# {'code': '...', 'output': '45'}

# Multiple operations
result = python_tool(code="""
for i in range(3):
    print(f"Number: {i}")
""")
# {'code': '...', 'output': 'Number: 0\nNumber: 1\nNumber: 2'}
```

**Notes:**
- Captures stdout via `print()`
- Restricted globals (no imports, no file access)
- Use for safe calculations only

---

## Creating Custom Tools

### Basic Tool

```python
from underthesea.agent import Agent, Tool

def my_tool(param1: str, param2: int = 10) -> dict:
    """Tool description shown to the LLM."""
    return {"result": f"{param1} x {param2}"}

tool = Tool(my_tool)
agent = Agent(name="my_agent", tools=[tool])
```

### With Custom Name/Description

```python
tool = Tool(
    my_tool,
    name="custom_name",
    description="Custom description for the LLM"
)
```

### Tool Schema

```python
tool = Tool(my_tool)

# View OpenAI format
print(tool.to_openai_tool())
# {
#     "type": "function",
#     "function": {
#         "name": "my_tool",
#         "description": "Tool description shown to the LLM.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "param1": {"type": "string", "description": "Parameter param1"},
#                 "param2": {"type": "integer", "description": "Parameter param2"}
#             },
#             "required": ["param1"]
#         }
#     }
# }
```

### Direct Execution

```python
# Via Tool object
result = tool(param1="test", param2=5)

# Via execute (returns JSON string)
result_json = tool.execute({"param1": "test", "param2": 5})
```
