# Agents

This document provides a technical overview of the agent module in underthesea, including architecture, tools, and comparison with other popular agent frameworks.

## Overview

The agent module provides conversational AI capabilities with support for:

1. **Simple Agent** (`agent`): Singleton instance for quick conversational AI
2. **Custom Agent** (`Agent`): Class-based agent with custom tools support
3. **Tool System** (`Tool`): Function-to-tool wrapper with OpenAI function calling
4. **LLM Client** (`LLM`): Provider-agnostic LLM client (OpenAI, Azure OpenAI)

```
User Message → [Agent] → [LLM] → Tool Calls? → [Tool Execution] → Response
                  ↑                    ↓
                  └── History ←────────┘
```

## Installation

```bash
pip install "underthesea[agent]"

# Set API key
export OPENAI_API_KEY="sk-..."
# Or for Azure OpenAI:
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://xxx.openai.azure.com"
```

## Architecture

### Module Structure

```
underthesea/agent/
├── __init__.py          # Exports: agent, Agent, Tool, LLM, default_tools
├── agent.py             # _AgentInstance (singleton) and Agent class
├── llm.py               # LLM client for OpenAI/Azure
├── tools.py             # Tool class for function wrapping
└── default_tools.py     # Pre-built utility tools
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │   LLM    │  │  Tools   │  │ History  │  │ Instruction │ │
│  │ (client) │  │  (list)  │  │  (list)  │  │   (str)     │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────────────┘ │
│       │             │             │                         │
│       ▼             ▼             ▼                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    __call__()                         │  │
│  │  1. Add user message to history                       │  │
│  │  2. If tools: _call_with_tools()                      │  │
│  │     - Loop: LLM → Tool calls? → Execute → Repeat      │  │
│  │  3. Else: Simple chat completion                      │  │
│  │  4. Add assistant response to history                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. LLM Client

The `LLM` class provides a unified interface for OpenAI and Azure OpenAI.

**Provider Detection:**

| Priority | Condition | Provider |
|----------|-----------|----------|
| 1 | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` set | Azure |
| 2 | `OPENAI_API_KEY` set | OpenAI |
| 3 | Neither set | Error |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | Auto-detect | `"openai"` or `"azure"` |
| `model` | `str` | `gpt-4o-mini` | Model/deployment name |
| `api_key` | `str` | From env | API key |
| `azure_endpoint` | `str` | From env | Azure endpoint URL |
| `azure_api_version` | `str` | `2024-02-01` | Azure API version |

### 2. Tool Class

The `Tool` class wraps Python functions for OpenAI function calling.

**Automatic Schema Extraction:**

```python
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get weather for a location."""
    return {"temp": 25, "unit": unit}

tool = Tool(get_weather)
# Extracts:
# - name: "get_weather"
# - description: "Get weather for a location."
# - parameters: {"location": {"type": "string", "required": true},
#                "unit": {"type": "string", "required": false}}
```

**Type Mapping:**

| Python Type | JSON Schema Type |
|-------------|------------------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` | `array` |

### 3. Agent Class

The `Agent` class supports custom tools via OpenAI function calling.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Agent identifier |
| `tools` | `list[Tool]` | `[]` | Available tools |
| `instruction` | `str` | `"You are a helpful assistant."` | System prompt |
| `max_iterations` | `int` | `10` | Max tool calling loops |

**Tool Calling Flow:**

```
1. User sends message
2. Agent builds messages: [system, ...history, user]
3. Send to LLM with tools
4. If LLM returns tool_calls:
   a. Execute each tool
   b. Add tool results to messages
   c. Go to step 3 (repeat until no tool calls or max_iterations)
5. Return final assistant message
```

## Default Tools

### Tool Collections

| Collection | Count | Description |
|------------|-------|-------------|
| `default_tools` | 12 | All default tools |
| `core_tools` | 4 | Safe utilities (no external calls) |
| `web_tools` | 3 | Web/network operations |
| `system_tools` | 5 | File and system operations |

### Core Tools

| Tool | Function | Description |
|------|----------|-------------|
| `current_datetime_tool` | `get_current_datetime()` | Returns date, time, weekday, timestamp |
| `calculator_tool` | `calculator(expression)` | Evaluates math (sqrt, sin, cos, log, pi, e) |
| `string_length_tool` | `string_length(text)` | Counts characters, words, lines |
| `json_parse_tool` | `parse_json(json_string)` | Parses JSON strings |

### Web Tools

| Tool | Function | Description |
|------|----------|-------------|
| `web_search_tool` | `web_search(query)` | DuckDuckGo search (no API key) |
| `fetch_url_tool` | `fetch_url(url)` | Fetch URL content |
| `wikipedia_tool` | `wikipedia(query, lang)` | Wikipedia search (vi/en) |

### System Tools

| Tool | Function | Description |
|------|----------|-------------|
| `read_file_tool` | `read_file(path)` | Read file content |
| `write_file_tool` | `write_file(path, content)` | Write to file |
| `list_directory_tool` | `list_directory(path)` | List files/directories |
| `shell_tool` | `run_shell(command)` | Execute shell command |
| `python_tool` | `run_python(code)` | Execute Python code |

## Usage Examples

### Simple Agent (Singleton)

```python
from underthesea import agent

# Basic conversation
response = agent("Xin chào!")
print(response)

# With custom system prompt
response = agent("NLP là gì?", system_prompt="Bạn là chuyên gia NLP")

# Check history
print(agent.history)

# Reset conversation
agent.reset()
```

### Custom Agent with Tools

```python
from underthesea.agent import Agent, Tool

def search_database(query: str) -> list:
    """Search internal database."""
    return [{"id": 1, "title": f"Result for {query}"}]

def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    return {"status": "sent", "to": to}

agent = Agent(
    name="assistant",
    tools=[
        Tool(search_database),
        Tool(send_email, description="Send email to user"),
    ],
    instruction="You are a helpful office assistant."
)

response = agent("Find documents about AI and email the results to john@example.com")
```

### Using Default Tools

```python
from underthesea.agent import Agent, default_tools, core_tools, web_tools

# All tools
full_agent = Agent(name="full", tools=default_tools)

# Safe agent (no system access)
safe_agent = Agent(name="safe", tools=core_tools)

# Web-enabled agent
web_agent = Agent(name="web", tools=core_tools + web_tools)

# Use agent
response = full_agent("What time is it and what's the weather in Hanoi?")
```

### Direct Tool Usage

```python
from underthesea.agent import calculator_tool, wikipedia_tool

# Use tools without LLM
result = calculator_tool(expression="sqrt(144) + 2**10")
print(result)  # {'expression': '...', 'result': 1036.0}

wiki = wikipedia_tool(query="Hà Nội", lang="vi")
print(wiki["summary"])
```

## Comparison with Other Frameworks

### Feature Comparison

| Feature | underthesea | LangChain | OpenAI SDK | CrewAI | Phidata |
|---------|-------------|-----------|------------|--------|---------|
| **Setup Complexity** | Low | Medium | Low | Medium | Low |
| **Tool Definition** | Function + decorator | Class-based | Function | Class-based | Toolkit |
| **Multi-agent** | Manual | LangGraph | Built-in | Built-in | Built-in |
| **Memory** | In-memory | Multiple | Session | Built-in | Built-in |
| **MCP Support** | No | Yes | Yes | No | Yes |
| **Providers** | OpenAI, Azure | 100+ | OpenAI | OpenAI+ | 50+ |

### Tool Count Comparison

| Framework | Built-in Tools | Tool Categories |
|-----------|----------------|-----------------|
| **underthesea** | 12 | Core, Web, System |
| LangChain | 50+ | Search, Code, API, DB |
| OpenAI SDK | 5 hosted + local | Search, File, Code, Computer |
| CrewAI | 30+ | File, Web, Search, Document |
| Phidata/Agno | 45+ | Search, Finance, News, DB, Media |
| smolagents | 10 | Search, Web, Code, Speech |
| Pydantic AI | 7 | Search, Code, Image, Memory |

### Design Philosophy

| Framework | Philosophy |
|-----------|------------|
| **underthesea** | Simple, Vietnamese NLP focused, minimal dependencies |
| LangChain | Comprehensive, composable chains, large ecosystem |
| OpenAI SDK | Official, production-ready, OpenAI optimized |
| CrewAI | Role-based multi-agent collaboration |
| Phidata | Performance-focused, 45+ pre-built toolkits |
| smolagents | Lightweight, HuggingFace integration |

## Performance Considerations

### Latency Sources

| Source | Typical Time | Notes |
|--------|--------------|-------|
| LLM API call | 500ms - 5s | Depends on model and prompt length |
| Tool execution | Variable | Depends on tool (web search ~1s) |
| First call | +2-5s | Client initialization |

### Best Practices

1. **Reuse Agent Instances**: Avoid creating new agents per request
2. **Limit Tools**: Only include necessary tools (reduces prompt size)
3. **Set max_iterations**: Prevent infinite tool loops
4. **Use core_tools for Safety**: Avoid system tools in production

### Memory Usage

| Component | Memory |
|-----------|--------|
| Agent instance | ~1KB |
| LLM client | ~5KB |
| Per tool | ~500B |
| History (per message) | ~1KB |

## Security Considerations

### Tool Safety Levels

| Level | Tools | Risk |
|-------|-------|------|
| **Safe** | `core_tools` | No external access |
| **Network** | `web_tools` | HTTP requests only |
| **System** | `system_tools` | File/shell access |

### Recommendations

1. **Production**: Use only `core_tools` unless necessary
2. **Shell Tool**: Blocks dangerous commands (rm -rf, mkfs, etc.)
3. **Python Tool**: Runs in restricted globals
4. **File Tools**: Validate paths before use

### Blocked Shell Commands

```python
dangerous = ["rm -rf", "mkfs", "dd if=", ":(){", "fork bomb", "> /dev/"]
```

## API Reference

### Module Exports

```python
from underthesea.agent import (
    # Core
    agent,              # Singleton agent instance
    Agent,              # Agent class with tools
    LLM,                # LLM client
    Tool,               # Function-to-tool wrapper

    # Tool collections
    default_tools,      # All 12 tools
    core_tools,         # 4 safe tools
    web_tools,          # 3 web tools
    system_tools,       # 5 system tools

    # Individual tools
    current_datetime_tool,
    calculator_tool,
    string_length_tool,
    json_parse_tool,
    web_search_tool,
    fetch_url_tool,
    wikipedia_tool,
    read_file_tool,
    write_file_tool,
    list_directory_tool,
    shell_tool,
    python_tool,
)
```

## Testing

```bash
# Run all agent tests
uv run python -m unittest discover tests.agent

# Run specific test modules
uv run python -m unittest tests.agent.test_agent
uv run python -m unittest tests.agent.test_tools
uv run python -m unittest tests.agent.test_llm

# Lint
ruff check underthesea/agent/
```

## Changelog

### Unreleased

- Add `Agent` class with custom tools support (GH-712)
- Add `Tool` class for function wrapping
- Add 12 default tools: calculator, datetime, web_search, wikipedia, shell, python, file operations

### v9.1.5 (2026-01-29)

- Add `agent` singleton and `LLM` client (GH-745)
- Support OpenAI and Azure OpenAI providers

## References

### OpenAI Function Calling

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI Tools API Reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools)

### Popular Agent Frameworks

- [LangChain](https://python.langchain.com/) - Comprehensive AI application framework
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) - Official OpenAI agent framework
- [CrewAI](https://docs.crewai.com/) - Role-based multi-agent framework
- [Phidata/Agno](https://docs.phidata.com/) - High-performance agent framework
- [smolagents](https://huggingface.co/docs/smolagents/) - Lightweight HuggingFace agents
- [Pydantic AI](https://ai.pydantic.dev/) - Type-safe agent framework
- [Google ADK](https://google.github.io/adk-docs/) - Google's Agent Development Kit
- [AWS Strands](https://strandsagents.com/) - AWS agent SDK

### Related Documentation

- [underthesea Agent API](../../api/agent.md)
- [Issue GH-712: Agents with LLMs](https://github.com/undertheseanlp/underthesea/issues/712)
