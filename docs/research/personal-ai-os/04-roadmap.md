# Personal AI OS — Roadmap (for Underthesea)

> Last updated: 2026-05-17

Underthesea **không** đặt mục tiêu trở thành Personal AI OS đầy đủ (đó là phạm vi của OpenClaw). Roadmap này chọn lọc **pattern phù hợp với Vietnamese NLP context** và lộ trình thử nghiệm.

## Phase 0 — Status hiện tại (đã có)

- ✅ `Agent` class với tool loop (OpenAI function-calling)
- ✅ Multi-provider LLM (`OpenAI`, `Azure`, `Anthropic`, `Gemini`)
- ✅ `SessionManager` + `ContextManager` — context reset + handoff (Anthropic pattern)
- ✅ `WikiAgent` — markdown-as-memory cho personal KB (giống Karpathy thesis)
- ✅ Tracing (`LocalTracer`, `LangfuseTracer`) + auto-trace

## Phase 1 — Reliability patterns (1-2 tháng)

Mượn từ OpenClaw mà KHÔNG cần daemon:

- [ ] **Lane Queue** cho `default_tools` có side-effect (`write_file`, `shell`, `python`)
  - Implement: simple `asyncio.Queue` per lane name
  - Test: 2 tool call `write_file` đồng thời → serialize
- [ ] **Semantic Snapshot** cho `fetch_url_tool`
  - Implement: dùng `playwright` accessibility tree thay raw HTML
  - Fallback: HTML nếu không có browser
  - Test: token count giảm ≥ 5x trên top 10 site Việt Nam
- [ ] **Tool dedup** trong cùng iteration
  - Hash tool name + args, skip duplicate trong 1 turn

**Deliverable**: `underthesea.agent.runtime` module với `LaneQueue` + `SnapshotTool`.

## Phase 2 — Memory-first abstraction (2-3 tháng)

- [ ] Tách `WikiAgent` thành `Memory` interface chung
  - `MarkdownMemory(dir)` — current `WikiAgent` style
  - `VectorMemory(dir)` — embed + retrieve qua FAISS/Chroma
  - `HybridMemory` — markdown làm source-of-truth, vector làm index
- [ ] Agent có thể attach memory: `Agent(name, memory=MarkdownMemory("./mem"))`
- [ ] Auto-commit memory changes (git-backed, optional)

**Deliverable**: `underthesea.agent.memory` module.

## Phase 3 — Ambient triggers (3-4 tháng)

- [ ] `Heartbeat` scheduler — cron-style trong Python process
  - `Heartbeat.add("0 7 * * *", agent_callable)`
  - Có thể standalone hoặc embed vào caller process
- [ ] `Watcher` abstractions:
  - `FileWatcher` (qua `watchdog`)
  - `EmailWatcher` (IMAP/Gmail API)
  - `WebhookReceiver` (FastAPI subapp)
- [ ] Vietnamese-specific: thử nghiệm `ZaloWatcher` (Zalo Official Account API)

**Deliverable**: `underthesea.agent.ambient` module.

## Phase 4 — Multi-agent orchestrator (4-6 tháng)

- [ ] `Orchestrator` route message tới agent phù hợp dựa trên intent
- [ ] Built-in specialized agents cho NLP Việt Nam:
  - `TokenizerAgent`, `NERAgent`, `SentimentAgent`, `SummaryAgent`
  - `WikiAgent` (đã có)
- [ ] Handoff giữa agent qua `Memory` chung

**Deliverable**: `underthesea.agent.orchestrator` module + reference agents.

## Phase 5 — Optional: Gateway daemon (nếu cộng đồng yêu cầu)

Đây là phạm vi **lớn**, có thể spin-off thành package riêng (`underthesea-os`):

- [ ] WebSocket gateway daemon
- [ ] Device pairing + token
- [ ] Node protocol (laptop ↔ phone capability advertise)
- [ ] Channel integration: Zalo, Telegram, iMessage, Slack
- [ ] CLI launcher: `underthesea-os start`

**Quyết định Go/No-go**: cuối Phase 4, đánh giá nhu cầu thực tế.

## Quyết định cần làm sớm

1. **Scope ownership**: `underthesea.agent.*` có nên tách thành package riêng (`uts-agent`)?
2. **Cloud vs local**: target user dev (local) hay end-user (cloud-hosted)?
3. **Memory persistence**: file-based default? SQLite optional?
4. **Vietnamese channel priority**: Zalo > Messenger > email — đúng không?

## Non-goals

- ❌ Không clone OpenClaw 1:1 — quá rộng so với core NLP mission
- ❌ Không xây UI app (macOS/iOS/Android) — để cộng đồng làm
- ❌ Không cạnh tranh Siri / Apple Intelligence ở consumer space
- ❌ Không lock vào 1 LLM provider — giữ multi-provider

## Success metrics (cuối 2026)

- Phase 1-2 ship → ≥ 3 user thực tế dùng `WikiAgent` cho personal KB tiếng Việt
- Phase 3 ship → ≥ 1 case study: ambient agent cho email Vietnamese triage
- Repo `underthesea` thêm ≥ 500 ⭐ nhờ agent capability
- Có ≥ 1 PR cộng đồng cho Zalo/Messenger watcher
