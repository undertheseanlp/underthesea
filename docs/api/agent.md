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

## Notes

- The agent maintains conversation history across calls
- Use `agent.reset()` to start a new conversation
- Azure OpenAI is preferred when both credentials are available
- Default system prompt focuses on Vietnamese language and NLP tasks
- First call may be slower due to client initialization
