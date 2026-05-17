<p align="center">
  <br>
  <img src="https://raw.githubusercontent.com/undertheseanlp/underthesea/main/docs/static/img/logo.png"/>
  <br/>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/pypi/v/underthesea.svg">
  </a>
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue">
  </a>
  <a href="http://undertheseanlp.com/">
    <img src="https://img.shields.io/badge/demo-live-brightgreen">
  </a>
  <a href="https://undertheseanlp.github.io/underthesea/">
    <img src="https://img.shields.io/badge/docs-live-brightgreen">
  </a>
  <a href="https://colab.research.google.com/drive/1gD8dSMSE_uNacW4qJ-NSnvRT85xo9ZY2">
    <img src="https://img.shields.io/badge/colab-ff9f01?logo=google-colab&logoColor=white">
  </a>
  <a href="https://www.facebook.com/undertheseanlp/">
    <img src="https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white">
  </a>
  <a href="https://www.youtube.com/channel/UC9Jv1Qg49uprg6SjkyAqs9A">
    <img src="https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white">
  </a>
</p>

<br/>

<p align="center">
  <a href="https://github.com/undertheseanlp/underthesea/blob/main/docs/contribute/SPONSORS.md">
    <img src="https://img.shields.io/badge/sponsors-30-red?style=social&logo=GithubSponsors">
  </a>
</p>

<h3 align="center">
Open-source Agentic AI Toolkit
</h3>

`Underthesea` is:

🌊 **An Agentic AI Toolkit.** Since v9.3.0, Underthesea is an open-source Agentic AI Toolkit with built-in Vietnamese NLP capabilities. It provides multi-provider AI Agent support and a suite of Python modules for [Vietnamese Natural Language Processing](https://github.com/undertheseanlp/underthesea).

🎁 [**Support Us!**](#-support-us) Every bit of support helps us achieve our goals. Thank you so much. 💝💝💝

## Installation

```bash
$ pip install underthesea
```

## Agent

Multi-provider AI Agent with **zero external dependencies**. Communicates with LLM APIs using only Python stdlib (`urllib` + `json`) — no `openai`, `anthropic`, or `google-genai` packages required.

**Providers:** OpenAI | Azure OpenAI | Anthropic Claude | Google Gemini

### Quick Start

```bash
# Pick one provider:
$ export OPENAI_API_KEY=sk-...
# or Azure:
$ export AZURE_OPENAI_API_KEY=... && export AZURE_OPENAI_ENDPOINT=https://...
# or Anthropic:
$ export ANTHROPIC_API_KEY=sk-ant-...
# or Gemini:
$ export GOOGLE_API_KEY=...
```

```python
from underthesea.agent import Agent, LLM

agent = Agent(name="assistant", provider=LLM())
agent("Hello!")
```

### Providers

Each provider is its own class, following the [Anthropic SDK](https://docs.anthropic.com/en/api/client-sdks) pattern.

```python
from underthesea.agent import Agent, OpenAI, AzureOpenAI, Anthropic, Gemini, LLM

# OpenAI
agent = Agent(name="bot", provider=OpenAI(api_key="sk-..."))

# Azure OpenAI
agent = Agent(name="bot", provider=AzureOpenAI(
    api_key="...",
    endpoint="https://my.openai.azure.com",
    deployment="gpt-4",
))

# Anthropic Claude
agent = Agent(name="bot", provider=Anthropic(api_key="sk-ant-..."))

# Google Gemini
agent = Agent(name="bot", provider=Gemini(api_key="..."))

# Auto-detect from environment variables
agent = Agent(name="bot", provider=LLM())
```

### Streaming

```python
for chunk in agent.stream("Explain what an AI agent is"):
    print(chunk, end="", flush=True)
```

### Tool Calling

```python
from underthesea.agent import Agent, Tool, OpenAI

def get_weather(location: str) -> dict:
    """Get current weather for a location."""
    return {"location": location, "temp": 25, "condition": "sunny"}

agent = Agent(
    name="assistant",
    provider=OpenAI(),
    tools=[Tool(get_weather)],
    instruction="You are a helpful assistant.",
)

agent("What's the weather in Hanoi?")
# 'The weather in Hanoi is 25°C and sunny.'
```

### Default Tools

12 built-in tools: calculator, datetime, web search, wikipedia, file I/O, shell, python exec.

```python
from underthesea.agent import Agent, default_tools, LLM

agent = Agent(name="assistant", provider=LLM(), tools=default_tools)
agent("Calculate sqrt(144) + 10")
```

### Multi-Session

Long-running agents with context reset and structured handoff between sessions, following [Anthropic harness patterns](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents).

```python
from underthesea.agent import Agent, Session, AzureOpenAI

agent = Agent(name="researcher", provider=AzureOpenAI(...))
session = Session(agent, progress_file="progress.json")
session.create_task("Analyze documents", [
    "Read and classify documents",
    "Summarize each group",
    "Write final report",
])
session.run_until_complete(max_sessions=5)
```

### Tracing

Every agent call is automatically traced to `~/.underthesea/traces/`. Disable with `UNDERTHESEA_TRACE_DISABLED=1`.

```python
from underthesea.agent import Agent, LangfuseTracer, calculator_tool

# Auto local trace (default) — zero config
agent = Agent(name="bot", tools=[calculator_tool])
agent("What is 2+2?")
# >> Trace [a1b2c3] bot
#    |-- Generation: llm.chat #1 (gpt-4.1-mini) ... 1200ms | 100->18 tokens
#    |-- Tool: tool.calculator ... 0ms
#    |-- Generation: llm.chat #2 (gpt-4.1-mini) ... 800ms | 150->12 tokens
# << Trace [a1b2c3] [ok] 2000ms -> ~/.underthesea/traces/20260411_trace_a1b2c3.json

# Langfuse (pip install langfuse)
agent = Agent(name="bot", tools=[calculator_tool], tracer=LangfuseTracer())

# @trace decorator — nested functions become child spans
from underthesea.agent.trace import trace, LocalTracer

@trace(LocalTracer())
def pipeline(text):
    return Agent(name="bot")(text)  # auto-inherits trace context
```

### Serving (A2A)

Expose any agent over the [A2A protocol](https://github.com/google-agentic-commerce/AP2) — JSON-RPC `message/stream` over HTTP+SSE with a discoverable `AgentCard` and an optional bundled chat UI. The server is a raw ASGI app with **no web-framework dep** in the base install — plug it into any ASGI server you like.

```bash
# Optional convenience extra: uvicorn + starlette + httpx
$ pip install 'underthesea[agent-server]'
```

```python
from underthesea.agent import Agent, Tool
from underthesea.agent.server import serve

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

agent = Agent(name="MathAgent", instruction="...", tools=[Tool(add)])
serve(agent, port=8000, path="/a2a/math", ui=True)
# → GET  http://127.0.0.1:8000/a2a/math/ui            (bundled chat UI)
# → GET  http://127.0.0.1:8000/a2a/math/.well-known/agent-card.json
# → POST http://127.0.0.1:8000/a2a/math               (JSON-RPC message/stream, SSE)
```

For custom routing or your own ASGI server, use `make_app()`:

```python
from underthesea.agent.server import make_app

app = make_app(agent, path="/a2a/math")  # → raw ASGI callable
# uvicorn module:app
# hypercorn module:app
# daphne module:app
```

One `Agent` is spawned per A2A `contextId` so each conversation keeps its own history. Tool calls are streamed live as `tool_call` artifacts; the agent's reply arrives as a `text` artifact when the loop finishes.

### Architecture

```
underthesea.agent
├── providers/
│   ├── OpenAI          # api.openai.com
│   ├── AzureOpenAI     # *.openai.azure.com
│   ├── Anthropic       # api.anthropic.com
│   └── Gemini          # generativelanguage.googleapis.com
├── trace/
│   ├── LocalTracer     # JSON files to ~/.underthesea/traces/
│   ├── LangfuseTracer  # Langfuse v4 observability
│   └── @trace          # Decorator with auto-nesting
├── server/
│   ├── make_app        # Raw ASGI callable (JSON-RPC + SSE + AgentCard)
│   ├── serve           # uvicorn convenience entrypoint
│   └── static/         # Bundled chat UI (ui=True)
├── Agent               # Tool calling loop + streaming
├── LLM                 # Auto-detect provider from env vars
├── Session             # Multi-session orchestration
├── Tool                # Function → tool wrapper
└── default_tools       # 12 built-in tools
```

## Vietnamese NLP

See full documentation at [NLP.md](NLP.md).

| Pipeline | Usage |
|----------|-------|
| Sentence Segmentation | `sent_tokenize(text)` |
| Text Normalization | `text_normalize(text)` |
| Word Segmentation | `word_tokenize(text)` |
| POS Tagging | `pos_tag(text)` |
| Chunking | `chunk(text)` |
| Named Entity Recognition | `ner(text)` |
| Text Classification | `classify(text)` |
| Sentiment Analysis | `sentiment(text)` |
| Language Detection | `lang_detect(text)` |
| Dependency Parsing | `dependency_parse(text)` |
| Translation | `translate(text)` |
| Text-to-Speech | `tts(text)` |

```python
from underthesea import word_tokenize, ner, sentiment

word_tokenize("Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò")
# ["Chàng trai", "9X", "Quảng Trị", "khởi nghiệp", "từ", "nấm", "sò"]

ner("Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump")
# [... ('Việt Nam', 'Np', 'B-NP', 'B-LOC'), ... ('Donald', 'Np', 'B-NP', 'B-PER'), ('Trump', 'Np', 'B-NP', 'I-PER')]

sentiment("Sản phẩm hơi nhỏ nhưng chất lượng tốt, đóng gói cẩn thận.")
# 'positive'
```

## Contributing

Do you want to contribute with underthesea development? Great! Please read more details at [Contributing Guide](https://undertheseanlp.github.io/underthesea/docs/developer/contributing)

## 💝 Support Us

If you found this project helpful and would like to support our work, you can just buy us a coffee ☕.

Your support is our biggest encouragement 🎁!

<img src="https://raw.githubusercontent.com/undertheseanlp/underthesea/main/docs/static/img/support.png"/>
