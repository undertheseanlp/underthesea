# agent

Conversational AI agent for Vietnamese language tasks using OpenAI or Azure OpenAI.

## Usage

```python
from underthesea import agent

response = agent("Xin chào, NLP là gì?")
print(response)
# NLP (Natural Language Processing) là xử lý ngôn ngữ tự nhiên...
```

## Installation

```bash
pip install "underthesea[agent]"
```

## Configuration

### Environment Variables

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY="sk-..."
    export OPENAI_MODEL="gpt-4o-mini"  # optional
    ```

=== "Azure OpenAI"

    ```bash
    export AZURE_OPENAI_API_KEY="..."
    export AZURE_OPENAI_ENDPOINT="https://xxx.openai.azure.com"
    export AZURE_OPENAI_DEPLOYMENT="my-gpt4-deployment"  # optional
    export AZURE_OPENAI_API_VERSION="2024-02-01"  # optional
    ```

### Environment Variables Reference

| Provider | Variable | Required | Description |
|----------|----------|----------|-------------|
| OpenAI | `OPENAI_API_KEY` | Yes | OpenAI API key |
| OpenAI | `OPENAI_MODEL` | No | Model name (default: `gpt-4o-mini`) |
| Azure | `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| Azure | `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint URL |
| Azure | `AZURE_OPENAI_DEPLOYMENT` | No | Deployment name (default: `gpt-4o-mini`) |
| Azure | `AZURE_OPENAI_API_VERSION` | No | API version (default: `2024-02-01`) |

## Function Signature

```python
def agent(
    message: str,
    model: str | None = None,
    system_prompt: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
) -> str
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | | User message to send |
| `model` | `str` | `None` | Model/deployment name. Falls back to env var or `gpt-4o-mini` |
| `system_prompt` | `str` | `None` | Custom system prompt |
| `provider` | `str` | `None` | Provider: `"openai"` or `"azure"`. Auto-detected if not specified |
| `api_key` | `str` | `None` | API key. Falls back to environment variable |
| `azure_endpoint` | `str` | `None` | Azure endpoint URL. Falls back to `AZURE_OPENAI_ENDPOINT` |
| `azure_api_version` | `str` | `None` | Azure API version. Falls back to env var or `2024-02-01` |

## Returns

| Type | Description |
|------|-------------|
| `str` | Assistant response |

## Methods

### `agent.reset()`

Clear conversation history.

```python
agent.reset()
```

### `agent.history`

Get conversation history (read-only copy).

```python
history = agent.history
# [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
```

## Examples

### Basic Conversation

```python
from underthesea import agent

# First message
response = agent("Xin chào!")
print(response)
# Xin chào! Tôi có thể giúp gì cho bạn?

# Follow-up (history is maintained)
response = agent("NLP là gì?")
print(response)
# NLP (Natural Language Processing) là...

# Check history
print(len(agent.history))  # 4 (2 user + 2 assistant messages)

# Reset conversation
agent.reset()
print(len(agent.history))  # 0
```

### Custom System Prompt

```python
from underthesea import agent

response = agent(
    "Chào bạn",
    system_prompt="Bạn là trợ lý chuyên về ẩm thực Việt Nam."
)
```

### Custom Model

```python
from underthesea import agent

# Use GPT-4
response = agent("Giải thích machine learning", model="gpt-4")

# Use GPT-4 Turbo
response = agent("Giải thích deep learning", model="gpt-4-turbo")
```

### Azure OpenAI

```python
from underthesea import agent

# Using environment variables (recommended)
# export AZURE_OPENAI_API_KEY="..."
# export AZURE_OPENAI_ENDPOINT="https://xxx.openai.azure.com"
# export AZURE_OPENAI_DEPLOYMENT="my-gpt4"

response = agent("Xin chào")

# Or explicit configuration
response = agent(
    "Xin chào",
    provider="azure",
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    model="your-deployment-name"
)
```

### Vietnamese NLP Assistant

```python
from underthesea import agent, word_tokenize, pos_tag

# Use agent to explain NLP concepts
text = "Học máy là gì?"
response = agent(text)
print(response)

# Combine with other underthesea functions
text = "Việt Nam là quốc gia xinh đẹp"
tokens = word_tokenize(text)
tags = pos_tag(text)

# Ask agent to explain the results
response = agent(f"Giải thích kết quả POS tagging: {tags}")
print(response)
```

## LLM Class

For more control, use the `LLM` class directly:

```python
from underthesea.agent import LLM

# Initialize
llm = LLM(model="gpt-4")

# Chat with custom messages
messages = [
    {"role": "system", "content": "You are a Vietnamese language expert."},
    {"role": "user", "content": "Explain word segmentation in Vietnamese."}
]
response = llm.chat(messages)
print(response)

# Check provider
print(llm.provider)  # 'openai' or 'azure'
print(llm.model)     # 'gpt-4'
```

## Agent Class with Tools

Create custom agents with function calling support using OpenAI's tools API.

### Basic Usage

```python
from underthesea.agent import Agent, Tool

# Define a tool as a Python function
def get_weather(location: str) -> dict:
    """Get current weather for a location."""
    return {"location": location, "temp": 25, "condition": "sunny"}

# Create agent with tools
my_agent = Agent(
    name="weather_assistant",
    tools=[Tool(get_weather, description="Get weather for a city")],
    instruction="You are a helpful weather assistant."
)

# Use the agent - it will automatically call tools when needed
response = my_agent("What's the weather in Hanoi?")
print(response)
# The weather in Hanoi is 25°C and sunny.

# Reset conversation
my_agent.reset()
```

### Agent Constructor

```python
Agent(
    name: str,
    tools: list[Tool] | None = None,
    instruction: str | None = None,
    max_iterations: int = 10,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | | Agent name |
| `tools` | `list[Tool]` | `None` | List of tools available to the agent |
| `instruction` | `str` | `"You are a helpful assistant."` | System instruction |
| `max_iterations` | `int` | `10` | Maximum tool calling iterations |

### Tool Class

Wrap Python functions as agent tools:

```python
from underthesea.agent import Tool

def search(query: str, limit: int = 10) -> list:
    """Search for items matching the query."""
    return [{"title": f"Result for {query}"}]

tool = Tool(
    func=search,
    name="web_search",           # Optional, defaults to function name
    description="Search the web" # Optional, defaults to docstring
)

# Convert to OpenAI format
openai_format = tool.to_openai_tool()

# Execute directly
result = tool(query="python", limit=5)
```

### Tool Constructor

```python
Tool(
    func: Callable,
    name: str | None = None,
    description: str | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | | The function to wrap |
| `name` | `str` | `None` | Tool name (defaults to function name) |
| `description` | `str` | `None` | Tool description (defaults to docstring) |

### Supported Parameter Types

The Tool class automatically extracts JSON schema from function signatures:

| Python Type | JSON Schema Type |
|-------------|------------------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` | `array` |

### Multiple Tools Example

```python
from underthesea.agent import Agent, Tool

def get_weather(location: str) -> dict:
    """Get current weather for a location."""
    return {"location": location, "temp": 25}

def search_news(query: str) -> str:
    """Search Vietnamese news articles."""
    return f"Found articles about: {query}"

def translate_text(text: str, target_lang: str = "en") -> str:
    """Translate Vietnamese text."""
    return f"Translated: {text}"

agent = Agent(
    name="multi_tool_agent",
    tools=[
        Tool(get_weather),
        Tool(search_news),
        Tool(translate_text, description="Translate Vietnamese to other languages"),
    ],
    instruction="You are a helpful Vietnamese assistant with access to weather, news, and translation tools."
)

# The agent decides which tool to use based on the query
response = agent("Thời tiết ở Đà Nẵng thế nào?")  # Uses get_weather
response = agent("Tin tức về AI hôm nay")          # Uses search_news
response = agent("Dịch 'Xin chào' sang tiếng Anh") # Uses translate_text
```

### Agent without Tools

Agent also works without tools as a simple conversational agent:

```python
from underthesea.agent import Agent

simple_agent = Agent(
    name="chatbot",
    instruction="You are a friendly Vietnamese chatbot."
)

response = simple_agent("Xin chào!")
print(response)
```

## Default Tools

Pre-built tools similar to LangChain/OpenAI tools for common tasks.

### Tool Collections

| Collection | Tools | Description |
|------------|-------|-------------|
| `default_tools` | 12 tools | All default tools |
| `core_tools` | 4 tools | Safe utilities (datetime, calculator, string, json) |
| `web_tools` | 3 tools | Web operations (search, fetch, wikipedia) |
| `system_tools` | 5 tools | System operations (file, shell, python) |

### Core Tools

| Tool | Description |
|------|-------------|
| `current_datetime_tool` | Get current date, time, weekday |
| `calculator_tool` | Evaluate math expressions (supports sqrt, sin, cos, log, pi, e) |
| `string_length_tool` | Count characters, words, lines in text |
| `json_parse_tool` | Parse JSON strings |

### Web Tools

| Tool | Description |
|------|-------------|
| `web_search_tool` | Search the web using DuckDuckGo (no API key) |
| `fetch_url_tool` | Fetch content from a URL |
| `wikipedia_tool` | Search Wikipedia (supports Vietnamese and English) |

### System Tools

| Tool | Description |
|------|-------------|
| `read_file_tool` | Read content from a file |
| `write_file_tool` | Write content to a file |
| `list_directory_tool` | List files and directories |
| `shell_tool` | Run shell commands |
| `python_tool` | Execute Python code |

### Using Default Tools

```python
from underthesea.agent import Agent, default_tools

# Create agent with all default tools
my_agent = Agent(
    name="assistant",
    tools=default_tools,
    instruction="You are a helpful assistant with access to various tools."
)

# Agent can now use any tool automatically
my_agent("What time is it?")           # Uses current_datetime_tool
my_agent("Calculate sqrt(144) + 10")   # Uses calculator_tool
my_agent("Search for Python tutorials") # Uses web_search_tool
my_agent("List files in current dir")  # Uses list_directory_tool
```

### Using Specific Tool Collections

```python
from underthesea.agent import Agent, core_tools, web_tools

# Safe agent with only core tools (no system access)
safe_agent = Agent(
    name="safe_assistant",
    tools=core_tools,
)

# Web-enabled agent
web_agent = Agent(
    name="web_assistant",
    tools=core_tools + web_tools,
)
```

### Using Tools Directly

```python
from underthesea.agent import calculator_tool, current_datetime_tool, wikipedia_tool

# Call tools directly (without LLM)
result = calculator_tool(expression="2 ** 10")
print(result)  # {'expression': '2 ** 10', 'result': 1024}

now = current_datetime_tool()
print(now)  # {'datetime': '...', 'date': '...', 'weekday': 'Friday', ...}

wiki = wikipedia_tool(query="Hà Nội", lang="vi")
print(wiki["summary"])  # Wikipedia summary about Hanoi
```

## Notes

- The agent maintains conversation history across calls
- Use `agent.reset()` to start a new conversation
- Azure OpenAI is preferred when both credentials are available
- Default system prompt focuses on Vietnamese language and NLP tasks
- First call may be slower due to client initialization
- Agent with tools uses OpenAI's function calling API
- Tools are automatically called when the model decides they are needed
- `max_iterations` prevents infinite tool calling loops
