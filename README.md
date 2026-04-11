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

### Architecture

```
underthesea.agent
├── providers/
│   ├── OpenAI          # api.openai.com
│   ├── AzureOpenAI     # *.openai.azure.com
│   ├── Anthropic       # api.anthropic.com
│   └── Gemini          # generativelanguage.googleapis.com
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
