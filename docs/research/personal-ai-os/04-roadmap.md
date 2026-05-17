# Personal AI OS — Roadmap (for Underthesea)

> Last updated: 2026-05-17
> Aligned với plan `~/.claude/plans/xu-t-roadmap-cho-mighty-candy.md` đã duyệt 2026-05-17.

Underthesea **không** đặt mục tiêu trở thành Personal AI OS đầy đủ (đó là phạm vi của OpenClaw). Roadmap này chọn lọc **pattern phù hợp** và lộ trình ship + hype theo 3 milestone trong 6 tháng. Positioning thuần **global** — không dùng Vietnamese identity làm marketing lever.

## Phase 0 — Status hiện tại (đã có ở v9.5.0)

- ✅ `Agent` class với tool loop (OpenAI function-calling)
- ✅ Multi-provider LLM (`OpenAI`, `Azure`, `Anthropic`, `Gemini`)
- ✅ `SessionManager` + `ContextManager` — context reset + handoff (Anthropic pattern)
- ✅ `WikiAgent` — markdown-as-memory cho personal KB (giống Karpathy thesis)
- ✅ Tracing (`LocalTracer`, `LangfuseTracer`) + auto-trace
- ✅ A2A-compatible agent server (`underthesea.agent.server`) — bundled chat UI, per-session isolation, tool streaming
- ✅ Zero external deps cho core (`urllib` + `json` only)

## Milestone 1 — "Smallest Agent Framework + Free Assistant TUI" (T1-T2, ~2.5 tháng)

Mượn pattern từ OpenClaw mà KHÔNG cần daemon. Mục tiêu: positioning + first hype moment (Show HN).

### Runtime patterns
- [ ] **Lane Queue** cho `default_tools` có side-effect (`write_file`, `shell`, `python`)
  - Implement: simple `asyncio.Queue` per lane name
  - Decorator: `@lane("messaging")`
  - Test: 2 tool call `write_file` đồng thời → serialize
- [ ] **Semantic Snapshot** cho `fetch_url_tool`
  - Dùng `playwright` accessibility tree thay raw HTML
  - Fallback: HTML nếu không có browser
  - Test: token count giảm ≥ 5x trên top 10 site phổ biến (HN, GitHub, Wikipedia, Reddit, SO…)
- [ ] **Tool dedup** trong cùng iteration
  - Hash tool name + args, skip duplicate trong 1 turn

### Underthesea Assistant TUI (NEW — OpenClaw-style claude-cli bridge)
- [ ] **CLI subcommand**: `underthesea assistant` — launch TUI app
- [ ] **TUI** với `textual` — chat panel + input box + status bar (streaming token-by-token)
- [ ] **ClaudeBridge**: spawn `claude --print --output-format=stream-json` subprocess, parse stream
  - Dùng `claude login` subscription của user → **free** (không tốn API token)
  - Đây là USP cực mạnh: "Free Personal AI Assistant on your Claude Pro/Max subscription"
- [ ] **Session/history**: persist conversation vào markdown file (precursor của MarkdownMemory M2)
- [ ] Optional extra `[assistant]`: `textual`, `rich` (không cần `claude-agent-sdk` vì dùng subprocess)

### Marketing infra
- [ ] **README rewrite** + CONTRIBUTING.md + issue/PR templates + Discord setup

**Deliverable**: `underthesea.agent.runtime` (LaneQueue + SnapshotTool) + `underthesea.agent.assistant` (TUI + ClaudeBridge).

**Hype**: Show HN dual angle: *"Smallest agent framework + free TUI assistant using your Claude subscription"* — combo này độc nhất, đủ wow cho HN front page. dev.to blog, Twitter thread, 5 AI newsletter outreach. Demo GIF 15s của TUI là asset chính.

## Milestone 2 — "Personal AI OS in Pure Python" (T3-T4, ~2 tháng)

Memory-first + ambient triggers + killer demo app.

### Memory layer
- [ ] Tách `WikiAgent` thành `Memory` interface chung
  - `MarkdownMemory(dir)` — current `WikiAgent` style (default)
  - `VectorMemory(dir)` — embed + retrieve qua FAISS/Chroma (optional)
  - `HybridMemory` — markdown làm source-of-truth, vector làm index
- [ ] Agent có thể attach memory: `Agent(name, memory=MarkdownMemory("./mem"))`
- [ ] Auto-commit memory changes (git-backed, optional)

### Ambient triggers
- [ ] `Heartbeat` scheduler — cron-style trong Python process
  - `Heartbeat.add("0 7 * * *", agent_callable)`
- [ ] `Watcher` abstractions:
  - `TelegramWatcher` — Bot API (default, dễ onboard nhất)
  - `SlackWatcher` — Socket Mode / Events API
  - `EmailWatcher` — IMAP polling
  - `FileWatcher` (qua `watchdog`) — optional
  - `WebhookReceiver` (FastAPI subapp) — optional
- [ ] ~~`ZaloWatcher`~~ — Deprioritized 2026-05-17. Channel global hơn (Telegram/Slack) phủ rộng audience hơn nhiều lần.

### Killer demo
- [ ] `examples/personal-ai-os/` — Telegram bot tự triage tin nhắn + reply + nhớ context qua MarkdownMemory + morning brief 7AM
- [ ] One-command setup: `python -m examples.personal-ai-os start`
- [ ] GIF demo + README riêng

**Deliverable**: `underthesea.agent.memory` + `underthesea.agent.ambient` modules + working demo app.

**Hype**: dev.to/Substack blog "Self-hosted Jarvis in 50 LOC", YouTube live coding, PyCon US/EuroPython CFP, podcast pitch (Latent Space, Practical AI), Reddit cross-post (r/Python, r/selfhosted, r/LocalLLaMA).

## Milestone 3 — "Growth Amplification" (T5-T6, ~2 tháng)

Không build feature mới. Polish + deploy + launch + retrospective.

- [ ] **HuggingFace Space** deploy demo M2 — public link, anybody can try
- [ ] **Benchmark suite** `benchmarks/agent_comparison/` — Underthesea vs LangGraph/CrewAI/AutoGen: latency, token cost, cold-start, LOC, deps count. Reproducible Colab notebook.
- [ ] **Bug bash** — fix mọi issue user đã tạo từ M1+M2, target 0 critical bug trước Product Hunt
- [ ] **Telegram bot template** (cookiecutter) — `pip install underthesea-telegram-bot`
- [ ] **Product Hunt launch** — chuẩn bị visual + 5-day pre-launch warmup
- [ ] **Benchmark blog** đăng HN + r/MachineLearning + Lobsters
- [ ] **Online hackathon** "Build with Underthesea" — Devpost / MLH cross-promote
- [ ] **Retrospective blog** — số liệu thật sau 6 tháng (stars, downloads, sponsors)
- [ ] **Discord events** — weekly office hour, AMA, showcase

**Deliverable**: HF Space public, benchmark numbers công khai, ≥ 3 viral content moment.

## Phase 5 — Optional: Gateway daemon (chỉ nếu cộng đồng yêu cầu)

Đây là phạm vi **lớn**, có thể spin-off thành package riêng (`underthesea-os`):

- [ ] WebSocket gateway daemon
- [ ] Device pairing + token
- [ ] Node protocol (laptop ↔ phone capability advertise)
- [ ] Channel integration mở rộng: WhatsApp, iMessage, Discord webhook
- [ ] CLI launcher: `underthesea-os start`

**Quyết định Go/No-go**: cuối M3, đánh giá nhu cầu thực tế từ Discord/issue tracker.

## Scope cut (so với roadmap cũ)

- ❌ **Multi-agent orchestrator + specialized NLP agents** — cut khỏi 6-month plan. Single Agent + Lane Queue đã đủ cho "smallest agent framework" narrative. Multi-agent là enterprise pattern, không phải hype-friendly.
- ❌ **Vietnamese-specific channels (Zalo)** — cut. Telegram/Slack/Email phủ audience global rộng hơn.
- ❌ **Vietnamese-specific case studies** — cut khỏi success metrics. User VN cũ không phải audience growth target.

## Quyết định cần làm sớm

1. **Scope ownership**: tách `underthesea.agent` thành package riêng (`uts-agent`)? **Recommend tách**, vì brand "Underthesea" gắn liền NLP Việt Nam có thể conflict với global positioning mới.
2. **Cloud vs local**: target dev (local) hay end-user (managed)? **Recommend local-first**, managed = HF Space demo cho try-before-install.
3. **Memory persistence**: file-based default? SQLite optional? **Recommend MarkdownMemory default**, git-backed auto-commit là optional.
4. **README rebrand**: đổi tagline + visual identity, hay giữ name `underthesea`? **Recommend giữ name, đổi tagline** (đổi tên = mất 9 năm SEO/credibility).

## Non-goals

- ❌ Không clone OpenClaw 1:1 — quá rộng so với core mission
- ❌ Không xây UI app (macOS/iOS/Android) — để cộng đồng làm
- ❌ Không cạnh tranh Siri / Apple Intelligence ở consumer space
- ❌ Không lock vào 1 LLM provider — giữ multi-provider làm USP
- ❌ Không dùng Vietnamese identity làm marketing lever
- ❌ Không train model mới — reuse pipeline đã có

## Success metrics (cuối tháng 11/2026)

- ≥ 5,000 GitHub ⭐ growth (baseline tháng 5/2026)
- ≥ 100 GitHub sponsors (3.3x từ 30 hiện tại)
- ≥ 3 viral content moment (Show HN front page, Twitter ≥ 1k like, Reddit top-10)
- ≥ 1 killer demo app deployed với active user thật
- Discord ≥ 500 thành viên
- ≥ 1 case study người dùng thật post lên Twitter/blog của họ
