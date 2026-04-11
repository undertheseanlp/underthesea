# Agent Harness - Current State Assessment

> Last updated: 2026-04-11

## 1. Architecture Overview

Agent harness (`underthesea/agent/`) gồm 15 source files, 2046 LOC, tổ chức thành 4 layers:

```
underthesea/agent/
├── __init__.py              # Public API exports
├── agent.py                 # Agent class + _AgentInstance singleton
├── llm.py                   # LLM auto-detect wrapper
├── tools.py                 # Tool abstraction (function → JSON schema)
├── default_tools.py         # 12 built-in tools
├── providers/
│   ├── base.py              # BaseProvider abstract class + data models
│   ├── _http.py             # Raw HTTP via urllib (zero deps)
│   ├── openai_provider.py   # OpenAI + AzureOpenAI
│   ├── anthropic_provider.py # Anthropic Claude
│   └── gemini_provider.py   # Google Gemini
└── harness/
    ├── session.py           # SessionManager (multi-session orchestration)
    ├── progress.py          # ProgressTracker (JSON-based subtask tracking)
    └── context.py           # ContextManager (structured handoff)
```

## 2. Providers

| Provider | Chat | Streaming | Tool Calling | Default Model |
|----------|------|-----------|--------------|---------------|
| OpenAI | ✅ | ✅ SSE | ✅ | gpt-4o-mini |
| AzureOpenAI | ✅ | ✅ SSE | ✅ | gpt-4o-mini |
| Anthropic | ✅ | ✅ Event-based | ✅ | claude-sonnet-4-20250514 |
| Gemini | ✅ | ✅ Part-based | ✅ | gemini-2.0-flash |

**LLM auto-detect** từ env vars theo thứ tự: Azure > OpenAI > Anthropic > Gemini.

**Zero external dependencies**: tất cả HTTP calls qua `urllib.request` + `json` (stdlib). Không import `openai`, `anthropic`, hay `google-genai`.

## 3. Features Implemented

### Core LLM
- Multi-provider abstraction (single interface)
- Auto-detect provider từ environment variables
- Chat (streaming + non-streaming)
- Tool calling loop với iteration limit
- Conversation history management
- System prompt customization
- Temperature control, model override per call

### Tool Ecosystem
- Function → Tool with auto JSON schema extraction via `inspect`
- 12 built-in tools: date/time, calculator, web search, URL fetch, Wikipedia, file I/O, shell, Python
- Sandboxing cho shell/Python execution
- Dual format: OpenAI + Anthropic compatible

### Multi-session Harness
- Progress tracking (JSON persistence)
- Subtask status management (PENDING → IN_PROGRESS → COMPLETED/FAILED/SKIPPED)
- Context reset between sessions (theo Anthropic recommendation)
- Structured handoff (summary, instructions, artifacts, warnings)

## 4. Test Coverage

6 test files, 1078 LOC, 101 tests — tất cả mock HTTP, không gọi real API.

| File | Lines | Coverage |
|------|-------|----------|
| test_agent.py | 110 | Agent singleton, history, reset |
| test_llm.py | 140 | Auto-detection, provider priority |
| test_providers.py | 308 | All 4 providers: chat, tools, format |
| test_tools.py | 230 | Tool creation, execution, schemas |
| test_harness.py | 295 | Progress, context, session manager |

## 5. Features Missing

### P0 — Critical gaps
- **Error handling & retry**: no exponential backoff, no rate limit handling, no network error recovery
- **Logging/observability**: không có logger, không trace, không metrics
- **Token counting**: không enforce context window limits

### P1 — Important
- **Async support**: hoàn toàn synchronous, không async/await
- **Structured output**: không JSON schema validation cho response
- **Context window management**: không summarization, không compression
- **Cost tracking**: không đếm tokens, không tính cost

### P2 — Nice to have
- **Memory/RAG**: no long-term memory, no vector search
- **Multi-agent orchestration**: no agent-to-agent communication
- **Guardrails**: no input/output filtering, no prompt injection protection
- **Vision/image support**: text-only
- **Plugin system**: no extension mechanism
- **Config file support**: chỉ env vars

## 6. Code Quality Notes

### Hardcoded values
- Default models hardcoded trong mỗi provider
- Anthropic max_tokens: 4096
- Azure API version: "2024-02-01"
- Web fetch truncate: 10000 chars
- Shell output truncate: 5000 chars

### Inconsistencies
- Agent history lưu raw OpenAI format, chưa normalized
- Tool format khác nhau giữa OpenAI/Anthropic nhưng không documented
- Streaming fallback trong BaseProvider chưa handle tool_calls đúng

### Potential issues
- `LLMError` raised nhưng không được catch/handle upstream
- Shell tool blocking list chưa comprehensive
- Gemini type uppercasing có thể fail với nested structures
