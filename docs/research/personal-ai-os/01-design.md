# Personal AI OS — Design Document

> Last updated: 2026-05-17

## 1. Problem

Hầu hết "agent framework" hiện nay (LangGraph, CrewAI, AutoGen) thiết kế cho **dev xây enterprise pipeline** — mỗi lần dùng phải:

1. Viết Python code, deploy lên server
2. Trigger qua HTTP request hoặc CLI
3. Agent chạy xong → tắt, mất state

Đây không phải cách end-user dùng AI hàng ngày. Một **personal assistant thực sự** cần:

- **Always-on** — nghe inbox, message, calendar 24/7
- **Multi-channel** — WhatsApp, iMessage, Slack, Telegram cùng lúc
- **Multi-device** — laptop, phone, tablet đều là endpoint
- **Persistent memory** — nhớ context xuyên qua phiên/ngày/tuần
- **Proactive** — tự action khi có trigger, không chờ user prompt

Trend này có tên: **Personal AI OS** (hoặc Ambient Agent, Always-on Agent). OpenClaw là reference implementation viral nhất 2026.

## 2. Đặc trưng của Personal AI OS

### 2.1 Daemon-first, không request-response

- Long-lived process, mở socket nghe events
- Khác với CLI agent (Claude Code, Gemini CLI): chạy → trả → tắt
- Khác với chat UI (ChatGPT): user phải mở app, gõ prompt

### 2.2 Gateway pattern

Một **Gateway** duy nhất làm:
- Message broker giữa channel ↔ agent
- Session/device pairing + auth
- Event bus (chat, presence, health, schedule)
- Wire protocol (WebSocket + JSON Schema)

### 2.3 Memory-first, không stateless

- Persistent context (Markdown files / structured store)
- Cross-session handoff
- Shared memory giữa các sub-agent

### 2.4 Ambient triggers (Heartbeat)

- Cron-like scheduler nội bộ
- Mailbox/inbox watcher
- File system / app event listener
- "Khi X xảy ra → agent action Y" thay vì "user hỏi → agent trả"

### 2.5 Multi-device mesh

- Mỗi device (laptop/phone/headless) là **Node**
- Nodes expose capabilities: camera, screen, mic, location, canvas
- Gateway route action tới Node phù hợp

### 2.6 Reliability patterns

- **Lane Queue** — serial-by-default cho side-effect (gửi tin nhắn, ghi file)
- **Semantic Snapshot** — accessibility tree thay vì HTML thô
- Idempotent action retry

## 3. Architecture (reference: OpenClaw)

```
┌─────────────────────────────────────────────────────┐
│  Channels: WhatsApp / iMessage / Slack / Telegram   │
└───────────────────────┬─────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │   Gateway daemon   │ ← WS + JSON Schema, auth, pairing
              │   (lane queue)     │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  Orchestrator      │ ← routes to specialized agent
              │  + Agent Runtime   │
              └─────────┬──────────┘
                        │
        ┌───────────────┼────────────────┐
        │               │                │
  ┌─────▼────┐   ┌──────▼─────┐   ┌──────▼─────┐
  │ Skills   │   │ Memory     │   │ Heartbeat  │
  │ (tools)  │   │ (Markdown) │   │ (schedule) │
  └─────┬────┘   └────────────┘   └────────────┘
        │
  ┌─────▼──────────────────────────────────────┐
  │  Nodes (macOS / iOS / Android / headless)  │
  │  → camera, screen, canvas, location, mic   │
  └────────────────────────────────────────────┘
```

## 4. So với Underthesea Agent hiện tại

| Layer | Underthesea (current) | Personal AI OS target |
|---|---|---|
| Entry | Sync Python call | WebSocket daemon |
| Channels | Stdin/CLI | WhatsApp, Slack, ... |
| Memory | History list + Handoff JSON | Markdown shared store |
| Trigger | User prompt | Inbox watcher + cron |
| Devices | Single process | Node mesh |
| Reliability | None | Lane queue, semantic snapshot |
| Multi-agent | Single agent | Orchestrator + specialized agents |

Underthesea hiện ở **Layer 1 (SDK)** — nếu muốn lên Personal AI OS phải thêm Gateway + Heartbeat + Node protocol.

## 5. Non-goals (cho Underthesea)

- **Không** clone toàn bộ OpenClaw — quá nặng cho mục tiêu NLP library
- **Không** target end-user; vẫn target developer
- **Có thể** mượn patterns chọn lọc: lane queue, ambient trigger, markdown memory (đã có trong `WikiAgent`)

## 6. Open questions

1. Có nên tách `underthesea.agent` thành package độc lập để target Personal AI OS không?
2. Vietnamese-specific channels nào quan trọng (Zalo, Lotus)?
3. Self-hosted vs cloud trade-off cho user Việt Nam?
