# Personal AI OS — Research

> Last updated: 2026-05-17

## 1. Trend overview

Personal AI OS là sự hội tụ của 4 trend agentic AI nổi bật trong 2024-2026:

### 1.1 Always-on / Ambient agents
- Khái niệm "ambient agents" được Harrison Chase (LangChain CEO) phổ biến 2025
- Agent **chủ động** monitor inbox/event, không chờ user prompt
- Đối lập với "reactive chatbot"

### 1.2 Personal AI OS (self-hosted)
- Phong trào "own your AI" — không lệ thuộc cloud của OpenAI/Anthropic
- Driven by privacy concerns + cost của subscription
- Mở đường cho Ollama, LocalAI, OpenClaw

### 1.3 Memory-first architecture
- Mem0, Letta (MemGPT), Cognee
- Karpathy 2025: "wiki as personal memory" — markdown human-readable
- Khác hướng vector DB-only

### 1.4 Multi-channel / Multi-device
- Apple Intelligence (cross-device, on-device)
- Humane AI Pin, Rabbit R1 (dedicated device, không cần phone)
- OpenClaw: open-source version của ý tưởng này

## 2. Landscape (2026)

### 2.1 Personal AI OS / Always-on
| Project | License | Notes |
|---|---|---|
| **OpenClaw** | MIT | Viral 2026, 347k ⭐, multi-channel daemon |
| **Open Interpreter** | AGPL | Earlier, code execution focus |
| **Letta (MemGPT)** | Apache | Memory-first, less channel coverage |
| **Pieces OS** | Closed | Code-snippet focused, on-device |
| **Cluely** | Closed | Screen + audio ambient |
| **Rewind** | Closed | macOS recall, on-device |

### 2.2 Memory layer (có thể plug vào Personal AI OS)
| Project | Approach |
|---|---|
| **Mem0** | Long-term memory với vector + graph |
| **Letta** | Tiered memory (in-context + recall) |
| **Cognee** | Knowledge graph memory |
| **Karpathy wiki** | Plain markdown, LLM-maintained |

### 2.3 Multi-agent frameworks (developer-facing — KHÁC layer)
- LangGraph, CrewAI, AutoGen: enterprise pipeline
- KHÔNG phải Personal AI OS — chúng là building block để xây Personal AI OS

## 3. Key technical patterns

### 3.1 Gateway pattern (OpenClaw)
- 1 daemon owns tất cả messaging surface
- WebSocket + JSON Schema validation
- Device pairing trước khi cấp token
- Phù hợp với mental model "OS service"

### 3.2 Lane Queue
- Serial-by-default cho action có side-effect
- Tránh race: 2 tin nhắn cùng lúc, 2 file write cùng lúc
- Có "express lane" cho read-only

### 3.3 Semantic Snapshot (web browsing)
- Parse accessibility tree thay vì HTML/screenshot
- Giảm 10-20x token vs raw HTML
- Tăng accuracy với screen reader semantics
- Tham khảo: Browser Use, Playwright accessibility snapshot

### 3.4 Heartbeat / proactive triggers
- Cron scheduler nội bộ
- Inbox watcher (IMAP/Gmail API/Slack RTM)
- Filesystem watcher
- Webhook receiver

### 3.5 Memory as Markdown
- Karpathy thesis: vector DB là premature optimization cho personal scale
- Markdown: human-readable, git-friendly, grep-able, LLM-native
- Trade-off: kém scale hơn vector cho > 10k docs

### 3.6 Node mesh
- gRPC / WebSocket node-to-gateway
- Capability advertisement (camera, screen, location...)
- Trust model: pairing + device token

## 4. Caveats & open problems

### 4.1 Privacy vs convenience
- Self-hosted = không có cloud nhưng yêu cầu maintenance
- Multi-channel listener = cần credentials cho mọi service → attack surface lớn

### 4.2 Cost
- LLM call cho ambient agent có thể nổ chi phí
- Heartbeat mỗi 5 phút × 24h × 30 ngày = 8640 call/tháng
- Cần model local (Ollama) hoặc small model (Haiku) cho background task

### 4.3 Reliability
- Long-running daemon dễ bị memory leak, hang
- Cần process supervisor (systemd, launchd)
- State recovery sau crash

### 4.4 Trigger explosion
- Quá nhiều inbox watcher → noise
- Cần rate limit + dedup pattern

### 4.5 Vietnamese context
- Channel chính: Zalo (API hạn chế), Facebook Messenger
- Email tiếng Việt cần segmentation/normalization (đây là điểm Underthesea có thể mạnh)
- Voice (TTS/STT) tiếng Việt vẫn yếu so với English

## 5. References

- [OpenClaw architecture docs](https://docs.openclaw.ai/concepts/architecture)
- [OpenClaw April 2026 update](https://www.clawbot.blog/blog/openclaw-the-rise-of-an-open-source-ai-agent-framework-april-2026-update/)
- [Reference Architecture: OpenClaw (Feb 2026)](https://robotpaper.ai/reference-architecture-openclaw-early-feb-2026-edition-opus-4-6/)
- [Lessons from OpenClaw's Architecture - Agentailor](https://blog.agentailor.com/posts/openclaw-architecture-lessons-for-agent-builders)
- [Harrison Chase on Ambient Agents (LangChain blog)](https://blog.langchain.dev) — concept origin
- [Karpathy: building a personal wiki with LLM](https://karpathy.ai) — markdown memory thesis
- [Mem0 GitHub](https://github.com/mem0ai/mem0)
- [Letta (MemGPT) GitHub](https://github.com/letta-ai/letta)
