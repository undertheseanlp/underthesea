# Agent Harness - Roadmap

> Last updated: 2026-04-11

## Phase 1: Production Readiness (v9.3.x)

Mục tiêu: agent harness có thể chạy trong production với độ tin cậy cao.

### 1.1 Error Handling & Retry
- [ ] Exponential backoff với jitter cho API calls
- [ ] Retry trên rate limit (429), server error (5xx), network timeout
- [ ] Configurable retry policy (max_retries, base_delay, max_delay)
- [ ] Provider fallback: tự động chuyển sang provider khác khi primary fail

### 1.2 Logging & Observability
- [ ] Structured logging qua Python `logging` module
- [ ] Log levels: DEBUG (raw request/response), INFO (call summary), WARNING (retry), ERROR
- [ ] Token usage tracking per call (prompt_tokens, completion_tokens)
- [ ] Cost estimation per provider
- [ ] Callback hooks cho monitoring integration

### 1.3 Context Window Management
- [ ] Token counting per provider (tiktoken cho OpenAI, character-based estimate cho others)
- [ ] Auto-truncation khi vượt context limit
- [ ] Conversation summarization strategy

### 1.4 Configuration
- [ ] Config dataclass thay vì hardcoded values (timeouts, truncation limits, default models)
- [ ] Config from file (TOML/YAML) ngoài env vars
- [ ] Per-provider config overrides

---

## Phase 2: Advanced Capabilities (v9.4.x)

Mục tiêu: mở rộng capabilities để hỗ trợ các use case phức tạp hơn.

### 2.1 Async Support
- [ ] `AsyncBaseProvider` with `async chat()` / `async chat_stream()`
- [ ] `Agent.acall()` / `Agent.astream()` async variants
- [ ] Async tool execution
- [ ] Connection pooling cho async HTTP

### 2.2 Structured Output
- [ ] JSON mode / response_format cho OpenAI & Gemini
- [ ] Pydantic model validation cho tool responses
- [ ] Schema-constrained generation

### 2.3 Multi-modal
- [ ] Image input support (vision) cho OpenAI, Anthropic, Gemini
- [ ] File/document upload
- [ ] Audio input (Gemini)

### 2.4 Memory & RAG
- [ ] Short-term memory: conversation buffer with sliding window
- [ ] Long-term memory: persistent key-value store
- [ ] Vector search integration interface (pluggable backend)
- [ ] Retrieval-augmented generation pipeline

---

## Phase 3: Multi-Agent & Orchestration (v9.5.x)

Mục tiêu: hỗ trợ multi-agent workflows và complex orchestration patterns.

### 3.1 Multi-Agent
- [ ] Agent-to-agent message passing
- [ ] Supervisor agent pattern (orchestrator + workers)
- [ ] Agent registry và discovery
- [ ] Shared context / blackboard pattern

### 3.2 Workflow Engine
- [ ] DAG-based task execution
- [ ] Parallel tool execution
- [ ] Conditional branching
- [ ] Human-in-the-loop checkpoints

### 3.3 Safety & Guardrails
- [ ] Input validation / content filtering
- [ ] Output guardrails (PII detection, toxicity)
- [ ] Prompt injection detection
- [ ] Rate limiting per user/session
- [ ] Audit logging

---

## Phase 4: Ecosystem & Developer Experience (v9.6.x)

Mục tiêu: developer-friendly ecosystem.

### 4.1 Plugin System
- [ ] Provider plugin interface (third-party providers)
- [ ] Tool plugin registry
- [ ] Memory backend plugins
- [ ] Middleware/hooks system

### 4.2 CLI & DevTools
- [ ] `underthesea agent chat` — interactive CLI chat
- [ ] `underthesea agent benchmark` — provider comparison tool
- [ ] `underthesea agent inspect` — debug/trace viewer
- [ ] Playground web UI (extensions/apps/chat evolution)

### 4.3 Documentation & Examples
- [ ] API reference docs (Docusaurus)
- [ ] Cookbook: common patterns (RAG, multi-agent, tool chains)
- [ ] Provider migration guide
- [ ] Performance benchmarks

---

## Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| Error handling & retry | High | Low | **P0** |
| Logging | High | Low | **P0** |
| Token counting | High | Medium | **P0** |
| Config dataclass | Medium | Low | **P1** |
| Async support | High | High | **P1** |
| Structured output | Medium | Medium | **P1** |
| Multi-modal (vision) | Medium | Medium | **P2** |
| Memory/RAG | High | High | **P2** |
| Multi-agent | Medium | High | **P3** |
| Plugin system | Medium | High | **P3** |
| Guardrails | Medium | Medium | **P2** |
