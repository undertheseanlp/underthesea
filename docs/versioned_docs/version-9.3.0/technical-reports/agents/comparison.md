# Agent Frameworks Comparison (2025)

This document provides a comprehensive comparison of popular AI agent frameworks and SDKs as of 2025.

## Framework Overview

| Framework | Organization | GitHub Stars | Language | License |
|-----------|--------------|--------------|----------|---------|
| [LangChain](https://python.langchain.com/) | LangChain | 100k+ | Python, JS | MIT |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | LangChain | 14k+ | Python, JS | MIT |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | OpenAI | 11k+ | Python, TS | MIT |
| [AutoGen](https://github.com/microsoft/autogen) | Microsoft | 45k+ | Python, .NET | MIT |
| [Semantic Kernel](https://github.com/microsoft/semantic-kernel) | Microsoft | 25k+ | Python, C#, Java | MIT |
| [CrewAI](https://docs.crewai.com/) | CrewAI | 25k+ | Python | MIT |
| [Phidata/Agno](https://docs.phidata.com/) | Phidata | 18k+ | Python | MIT |
| [smolagents](https://huggingface.co/docs/smolagents/) | HuggingFace | 15k+ | Python | Apache 2.0 |
| [Pydantic AI](https://ai.pydantic.dev/) | Pydantic | 10k+ | Python | MIT |
| [Google ADK](https://google.github.io/adk-docs/) | Google | 8k+ | Python, TS, Java | Apache 2.0 |
| [LlamaIndex](https://docs.llamaindex.ai/) | LlamaIndex | 40k+ | Python, TS | MIT |
| [Haystack](https://docs.haystack.deepset.ai/) | deepset | 18k+ | Python | Apache 2.0 |
| [AWS Strands](https://strandsagents.com/) | AWS | 3k+ | Python, TS | Apache 2.0 |
| [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) | Anthropic | New | Python, TS | MIT |
| [IBM BeeAI](https://github.com/i-am-bee/beeai-framework) | IBM/Linux Foundation | 3k+ | Python, TS | Apache 2.0 |
| [Letta/MemGPT](https://github.com/letta-ai/letta) | Letta | 12k+ | Python | Apache 2.0 |
| [CAMEL-AI](https://github.com/camel-ai/camel) | CAMEL-AI | 6k+ | Python | Apache 2.0 |
| [Dify](https://dify.ai/) | Dify | 60k+ | Python, TS | Apache 2.0 |
| [Flowise](https://flowiseai.com/) | Flowise (Workday) | 35k+ | JS/TS | Apache 2.0 |
| [Langflow](https://www.langflow.org/) | Langflow | 40k+ | Python | MIT |

## Built-in Tools Comparison

### OpenAI Agents SDK

| Category | Tools |
|----------|-------|
| **Hosted** | `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool`, `HostedMCPTool` |
| **Local** | `ComputerTool`, `ShellTool`, `ApplyPatchTool`, `FunctionTool` |

### HuggingFace smolagents

| Category | Tools |
|----------|-------|
| **Search** | `DuckDuckGoSearchTool`, `GoogleSearchTool`, `ApiWebSearchTool`, `WebSearchTool`, `WikipediaSearchTool` |
| **Web** | `VisitWebpageTool` |
| **Code** | `PythonInterpreterTool` |
| **User** | `UserInputTool`, `FinalAnswerTool` |
| **Audio** | `SpeechToTextTool` |

### Pydantic AI

| Tool | Description |
|------|-------------|
| `WebSearchTool` | Search the web |
| `WebFetchTool` | Fetch web pages |
| `CodeExecutionTool` | Execute code in sandbox |
| `ImageGenerationTool` | Generate images |
| `FileSearchTool` | RAG/vector search |
| `MemoryTool` | Persistent memory |
| `MCPServerTool` | MCP integration |

### Phidata/Agno (45+ Toolkits)

| Category | Tools |
|----------|-------|
| **Search** | DuckDuckGo, GoogleSearch, Exa, SearxNG, Serpapi, Tavily |
| **Finance** | YFinanceTools, OpenBB |
| **News** | Newspaper4k, HackerNews, Arxiv, Pubmed |
| **Database** | DuckDb, Postgres, SQL |
| **Code** | Python, Shell, Calculator |
| **Files** | File, CSV, Pandas |
| **Web** | Apify, Crawl4AI, Firecrawl, Spider, Website, JinaReader |
| **Media** | Dalle, YouTube, MLXTranscribe, ModelsLabs |
| **Communication** | Email, Slack, Resend |
| **Services** | GitHub, Jira, Twitter, Zendesk, CalCom, Wikipedia |

### CrewAI (30+ Tools)

| Category | Tools |
|----------|-------|
| **File** | FileReadTool, DirectoryReadTool, DirectorySearchTool |
| **Web** | ScrapeWebsiteTool, WebsiteSearchTool, FirecrawlCrawlWebsiteTool |
| **Search** | SerperDevTool, EXASearchTool, BraveSearchTool |
| **Documents** | PDFSearchTool, DOCXSearchTool, TXTSearchTool, MDXSearchTool |
| **Data** | CSVSearchTool, JSONSearchTool, XMLSearchTool |
| **Code** | CodeInterpreterTool, CodeDocsSearchTool, GithubSearchTool |
| **Database** | PGSearchTool |
| **Media** | YoutubeVideoSearchTool, YoutubeChannelSearchTool, DALL-E Tool |

### LangChain

| Category | Tools |
|----------|-------|
| **Search** | DuckDuckGoSearchRun, GoogleSearchRun, BingSearchRun, WikipediaQueryRun, ArxivQueryRun |
| **Code** | PythonREPL, ShellTool |
| **Math** | LLMMathChain |
| **HTTP** | RequestsGetTool, RequestsPostTool |
| **APIs** | OpenWeatherMapQueryRun, NewsAPITool |

### underthesea (12 Tools)

| Category | Tools |
|----------|-------|
| **Core** | current_datetime_tool, calculator_tool, string_length_tool, json_parse_tool |
| **Web** | web_search_tool, fetch_url_tool, wikipedia_tool |
| **System** | read_file_tool, write_file_tool, list_directory_tool, shell_tool, python_tool |

## Feature Matrix

| Feature | underthesea | LangChain | OpenAI SDK | CrewAI | Phidata | smolagents |
|---------|:-----------:|:---------:|:----------:|:------:|:-------:|:----------:|
| Simple API | ✅ | ⚠️ | ✅ | ⚠️ | ✅ | ✅ |
| Multi-agent | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MCP Support | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Memory | In-memory | Multiple | Session | Built-in | Built-in | ❌ |
| Streaming | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Async | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Type Safety | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| Visual Builder | ❌ | Flowise | ❌ | ❌ | ❌ | ❌ |
| Tracing | ❌ | LangSmith | Built-in | ❌ | Built-in | ❌ |
| Human-in-loop | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |

Legend: ✅ Full support | ⚠️ Partial | ❌ Not supported

## Provider Support

| Framework | OpenAI | Azure | Anthropic | Google | AWS Bedrock | Local/Ollama |
|-----------|:------:|:-----:|:---------:|:------:|:-----------:|:------------:|
| underthesea | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| LangChain | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| OpenAI SDK | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CrewAI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Phidata | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| smolagents | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| Pydantic AI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Google ADK | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |

## Use Case Recommendations

### Simple Chatbot / Q&A

| Recommendation | Frameworks |
|----------------|------------|
| **Best** | underthesea, Pydantic AI, smolagents |
| **Good** | OpenAI SDK, Phidata |
| **Overkill** | LangChain, CrewAI, AutoGen |

### Multi-Agent Collaboration

| Recommendation | Frameworks |
|----------------|------------|
| **Best** | CrewAI, AutoGen, CAMEL-AI |
| **Good** | LangGraph, OpenAI SDK, Phidata |
| **Limited** | underthesea, smolagents |

### RAG / Document Q&A

| Recommendation | Frameworks |
|----------------|------------|
| **Best** | LlamaIndex, LangChain, Haystack |
| **Good** | CrewAI (with tools), Dify |
| **Limited** | underthesea, smolagents |

### Code Generation / Automation

| Recommendation | Frameworks |
|----------------|------------|
| **Best** | Claude Agent SDK, OpenAI SDK |
| **Good** | LangChain, AutoGen |
| **Limited** | underthesea |

### Enterprise / Production

| Recommendation | Frameworks |
|----------------|------------|
| **Best** | Semantic Kernel, AWS Strands, IBM BeeAI |
| **Good** | LangChain + LangSmith, OpenAI SDK |
| **Prototype** | underthesea, Phidata, smolagents |

### Visual / No-Code

| Recommendation | Frameworks |
|----------------|------------|
| **Best** | Flowise, Dify, Langflow, n8n |
| **Code-first** | All others |

## Performance Benchmarks

### GAIA Benchmark (General AI Assistants)

| Framework | Score | Notes |
|-----------|-------|-------|
| CAMEL-AI OWL | 58.18% | #1 on leaderboard |
| AutoGen | ~55% | Multi-agent |
| OpenAI SDK | ~50% | With tools |

### Agent Instantiation Speed

| Framework | Time | Notes |
|-----------|------|-------|
| Phidata/Agno | 2μs | Claims 5000x faster than LangGraph |
| smolagents | ~10μs | Lightweight |
| LangGraph | ~10ms | Full features |

### Memory Efficiency

| Framework | Memory | Notes |
|-----------|--------|-------|
| Phidata/Agno | Low | Claims 50x more efficient |
| smolagents | Low | Minimal dependencies |
| LangChain | High | Large ecosystem |

## Protocol Support

### MCP (Model Context Protocol)

| Framework | MCP Client | MCP Server | Notes |
|-----------|:----------:|:----------:|-------|
| OpenAI SDK | ✅ | ❌ | HostedMCPTool |
| LangChain | ✅ | ✅ | Full support |
| Pydantic AI | ✅ | ❌ | MCPServerTool |
| Google ADK | ✅ | ✅ | Native support |
| Dify | ✅ | ✅ | MCP Apps |
| Langflow | ✅ | ✅ | v1.7+ |
| smolagents | ✅ | ❌ | Latest specs |
| underthesea | ❌ | ❌ | Not supported |

### A2A (Agent-to-Agent Protocol)

| Framework | Support | Notes |
|-----------|:-------:|-------|
| IBM BeeAI | ✅ | ACP merged with A2A |
| AWS Strands | ✅ | Built-in |
| Google ADK | ✅ | Native |
| Others | ❌ | Not yet |

## Conclusion

### When to Use underthesea Agent

**Good for:**
- Simple Vietnamese NLP chatbots
- Quick prototyping
- Minimal dependencies
- OpenAI/Azure OpenAI only

**Not ideal for:**
- Multi-agent systems
- Production at scale
- Complex workflows
- Multiple LLM providers

### Migration Path

If you outgrow underthesea agents:

1. **More providers** → Pydantic AI, Phidata
2. **Multi-agent** → CrewAI, AutoGen
3. **Production** → LangChain + LangSmith, AWS Strands
4. **Visual** → Flowise, Dify, Langflow
