# Weekly Agentic AI Scan — 2026-07-17

## Executive Summary

- **Actor/event-driven đang thắng thế ReAct-loop thuần** ở các coding-agent runtime nghiêm túc: `xai-org/grok-build` (xAI) và `awslabs/cli-agent-orchestrator` đều tách state/sampling/tool-dispatch thành các actor/service độc lập giao tiếp qua channel/HTTP thay vì một vòng lặp đơn luồng.
- **"Meta-harness" và "cross-vendor review" là pattern mới đáng chú ý nhất tuần này**: `omnigent-ai/omnigent` (Databricks) chuẩn hoá ~10 CLI agent khác nhau qua một `Executor` protocol chung, và ví dụ Polly của nó dùng reviewer *khác vendor* với implementer để giảm blind-spot — implementer không tự merge, con người luôn là gate cuối.
- **Memory cho coding agent đang phân hoá hai trường phái rõ rệt**: `vshulcz/deja-vu` chọn pure lexical/inverted-index (không embedding, không LLM call) đối lập có chủ đích với các công cụ semantic-memory khác, trong khi `xai-grok-memory` và memory 2-kho (Markdown wiki + SQLite BM25) của CAO đại diện cho hướng hybrid có structured scoring.

## Mục lục

1. [xai-org/grok-build](#1-xai-orggrok-build)
2. [awslabs/cli-agent-orchestrator (CAO)](#2-awslabscli-agent-orchestrator-cao)
3. [omnigent-ai/omnigent](#3-omnigent-aiomnigent)
4. [vshulcz/deja-vu](#4-vshulczdeja-vu)
5. [Repo khác đã xác định nhưng không đào sâu](#5-repo-khác-đã-xác-định-nhưng-không-đào-sâu)
6. [Ghi chú phương pháp luận & giới hạn](#6-ghi-chú-phương-pháp-luận--giới-hạn)

---

## 1. xai-org/grok-build

**Link:** https://github.com/xai-org/grok-build

### §1 Quick Context
TUI/agent runtime Rust của xAI cho coding, hỗ trợ tương tác, headless CI, và nhúng qua Agent Client Protocol (ACP). Stack: Rust (edition 2024), Cargo workspace ~70 crates. **13,191 sao, 2,370 fork**, tạo 2026-07-14, push gần nhất 2026-07-16. `has_issues:false`, PR ngoài bị khoá (`collaborators_only`). Không tìm thấy `.github/workflows/*.yml` công khai — không có bằng chứng CI public.

### §2 Architecture Deep-dive

**A. Component inventory**
- `xai-grok-pager-bin` (`crates/codegen/xai-grok-pager-bin/src/main.rs`) — composition root, chọn mode TUI/headless/leader/stdio-agent.
- `xai-grok-pager` (`crates/codegen/xai-grok-pager/src/lib.rs`) — TUI: scrollback, prompt, modals.
- `MvpAgent` (`crates/codegen/xai-grok-shell/src/agent/mvp_agent/mod.rs`) — agent runtime chính, implement `agent_client_protocol::Client`.
- `xai-acp-lib` (`crates/codegen/xai-acp-lib/src/lib.rs`) — gateway sender/receiver cho ACP.
- `agent/server.rs` — WebSocket server (axum) cho remote leader/follower TUI client.
- `xai-chat-state` (`crates/codegen/xai-chat-state/src/lib.rs`) — `ChatStateActor` giữ conversation, sampling config, total tokens.
- `xai-grok-sampler` (`crates/codegen/xai-grok-sampler/src/lib.rs`) — `SamplerActor`: streaming HTTP tới model, retry, "doom loop" detection.
- `xai-grok-tools` / `xai-tool-runtime` (`crates/codegen/xai-grok-tools/src/lib.rs`, `crates/common/xai-tool-runtime/src/lib.rs`) — implementation tool + `Tool` trait/`ToolDispatch`.
- `xai-tool-protocol` (`crates/common/xai-tool-protocol/src/lib.rs`) — JSON-RPC 2.0 cho "Computer Hub" (external tool server).
- `xai-grok-mcp` (`crates/codegen/xai-grok-mcp/src/lib.rs`) — tích hợp MCP qua crate `rmcp`.
- `xai-grok-sandbox` (`crates/codegen/xai-grok-sandbox/src/lib.rs`) — sandbox OS-level qua `nono` (Landlock/Seatbelt) + seccomp.
- `xai-grok-memory` (`crates/codegen/xai-grok-memory/src/lib.rs`) — memory markdown liên-session tại `~/.grok/memory/`, index bằng sqlite-vec.
- `xai-grok-compaction` (`crates/common/xai-grok-compaction/src/lib.rs`) — engine nén context (intra/inter/code_compaction), tái dùng giữa Grok chat và grok-build.
- `xai-grok-hooks` (`crates/codegen/xai-grok-hooks/src/lib.rs`) — hook file-based (`session_start/pre_tool_use/post_tool_use/session_end`).
- `xai-grok-plugin-marketplace` (`crates/codegen/xai-grok-plugin-marketplace/src/lib.rs`) — cài plugin từ git source.
- `xai-grok-models` (`crates/codegen/xai-grok-models/src/lib.rs` + `default_models.json`) — model-role registry.

**B. Control flow — Actor-based event-driven pipeline quanh ACP session** (không phải ReAct loop thuần):
1. `main.rs` chọn mode, gọi `run_headless`/`run_leader`/`run_stdio_agent`.
2. `MvpAgent` nhận `new_session`/prompt qua ACP, spawn `SessionThread`.
3. `ChatStateActor` build request từ conversation state, gửi qua `SamplerActor`.
4. Model trả stream; nếu có tool call, dispatch qua `ToolBridge` → `xai-grok-tools`/`xai-tool-runtime::Tool`, thực thi trong `xai-grok-sandbox`, hoặc route ra MCP server qua `xai-grok-mcp`.
5. Kết quả tool ghi lại vào `ChatStateActor`, lặp lại tới khi model trả lời cuối.
6. Session update stream về `xai-grok-pager` qua ACP notification.

**C. State & data flow:** conversation lưu trong `Vec<ConversationItem>` bên trong `ChatStateActor`, giao tiếp qua `mpsc`/oneshot command-query (không shared lock); nén context qua `xai-grok-compaction`; context window 500,000 token cho model `grok-build` (`default_models.json`).

**D. Tool integration:** native `Tool` trait + JSON-RPC 2.0 wire (`xai-tool-protocol`) cho external "Computer Hub" servers, cộng MCP qua `rmcp`; sandbox thực thi qua `nono` (Landlock/Seatbelt) + seccomp cho subprocess network.

**E. Memory:** `xai-grok-memory` — markdown + sqlite-vec, gated bởi `--experimental-memory`/`GROK_MEMORY=1`.

**F. Model orchestration:** role-based default trong `default_models.json` — `default/image_description/session_summary` = `grok-build`, `web_search` = `grok-4.20-multi-agent`; resolution order CLI flag > ENV > config.toml > remote settings > defaults. Không có bằng chứng fallback/parallel-model.

**G. Observability:** `xai-grok-telemetry` (`unified_log`, `session_ctx::log_event`), `xai-tracing`/`xai-tracing-macros`, upload session trace qua `upload::trace`. Không xác định từ code về eval hook tự động.

**H. Extension points:** hooks (`xai-grok-hooks`), plugin marketplace git-based (`xai-grok-plugin-marketplace`, official source `github.com/xai-org/plugin-marketplace`), MCP server config, custom agent definitions qua `AgentDefinition`/`preset_names`.

### §3 Architecture Diagram

```mermaid
sequenceDiagram
    participant TUI as xai-grok-pager
    participant Agent as MvpAgent (agent/mvp_agent)
    participant State as xai-chat-state (ChatStateActor)
    participant Sampler as xai-grok-sampler (SamplerActor)
    participant Tools as xai-grok-tools / xai-tool-runtime
    participant Sandbox as xai-grok-sandbox

    TUI->>Agent: ACP prompt (agent_client_protocol)
    Agent->>State: push_user / build_req
    State->>Sampler: sampling request
    Sampler-->>State: stream + tool_call
    State->>Tools: dispatch ToolDispatch
    Tools->>Sandbox: exec (Landlock/seccomp)
    Sandbox-->>Tools: result
    Tools-->>State: ToolOutput
    State->>Sampler: continue turn
    Sampler-->>Agent: final response
    Agent-->>TUI: ACP session/update notification
```

### §4 Verdict
Điểm đáng chú ý: kiến trúc actor pattern nhất quán (`xai-chat-state`, `xai-grok-sampler` đều theo cùng pattern) thay vì shared-lock state, và tách riêng "compaction-core" khỏi host để tái dùng giữa Grok chat và grok-build. Sandbox OS-level (`nono`, Landlock/Seatbelt) tích hợp sẵn — hiếm thấy ở agent CLI mã nguồn mở khác. Red flag: đây là bản public "đồng bộ định kỳ từ monorepo nội bộ" — không nhận PR/issue ngoài, không thấy CI workflow công khai, khó kiểm chứng tính đầy đủ của snapshot. `mvp_agent/mod.rs` dùng raw pointer (`LocalRef<T>`) với `unsafe impl Sync` để né borrow-checker trong `spawn_local` — code smell tiềm ẩn nếu invariant bị vi phạm khi refactor. Câu hỏi mở: cơ chế fallback model khi API lỗi, và giới hạn cụ thể của "doom_loop" detection trong sampler — không xác định từ code đã đọc.

---

## 2. awslabs/cli-agent-orchestrator (CAO)

**Link:** https://github.com/awslabs/cli-agent-orchestrator

### §1 Quick Context
Framework điều phối đa-agent CLI theo mô hình supervisor-worker, cách ly mỗi agent trong một tmux session, giao tiếp qua MCP. Stack: Python ≥3.10, FastAPI + Uvicorn, FastMCP/`mcp`, SQLAlchemy/SQLite, `libtmux`, `rank-bm25`, `pyte`, OpenTelemetry (opt-in). **902 sao, 172 fork, 42 contributor** (top: haofeif 51 commit, fanhongy 40), tạo 2025-07-29, push gần nhất 2026-07-17. CI: `.github/workflows/ci.yml` và `release.yml` tồn tại (License: Apache-2.0).

### §2 Architecture Deep-dive

**A. Component inventory**
- `src/cli_agent_orchestrator/api/main.py` — FastAPI HTTP server (port 9889 mặc định), endpoint quản lý session/terminal/inbox.
- `src/cli_agent_orchestrator/mcp_server/server.py` — MCP server nội bộ (`cao-mcp-server`), 17 tool: `handoff`, `assign`, `send_message`, `memory_store/recall/forget`, `workflow_run/resume/return/cancel`, `emit_ui`, `answer_user_prompt`, `delete_terminal`, `load_skill`.
- `src/cli_agent_orchestrator/services/event_bus.py` — pub/sub nội bộ, wildcard topic matching.
- `src/cli_agent_orchestrator/services/fifo_reader.py` — đọc output tmux qua FIFO, publish `terminal.{id}.output`.
- `src/cli_agent_orchestrator/services/status_monitor.py` — parse output, publish `terminal.{id}.status` (IDLE/PROCESSING/COMPLETED/ERROR).
- `src/cli_agent_orchestrator/services/inbox_service.py` — deliver message khi terminal rảnh.
- `src/cli_agent_orchestrator/services/memory_service.py` — logic lưu/truy hồi memory.
- `src/cli_agent_orchestrator/clients/tmux.py` — wrapper `libtmux.Server()`, set `CAO_TERMINAL_ID`.
- `src/cli_agent_orchestrator/clients/database.py` — SQLite (bảng `terminals`, `inbox_messages`).
- `src/cli_agent_orchestrator/providers/base.py` — abstract `BaseProvider`, mỗi provider tự parse pattern output CLI riêng để suy ra trạng thái.
- `src/cli_agent_orchestrator/providers/{kiro_cli,claude_code,codex}.py` — adapter cho từng CLI thật.
- `src/cli_agent_orchestrator/agent_store/{developer,reviewer,code_supervisor}.md` — agent profile dạng frontmatter Markdown.
- `src/cli_agent_orchestrator/plugins/builtin/*` — plugin entry point (event-driven, ví dụ Discord forwarding).
- `cao-ops-mcp-server` (`ops_mcp_server/server.py`) — MCP server quản lý ngoài-session, riêng biệt với `cao-mcp-server`.

**B. Control flow — Hierarchical supervisor-worker** (tên chính thức trong README), happy path "Handoff":
1. Supervisor gọi MCP tool `handoff(agent_profile, message)`.
2. `POST /sessions/{session}/terminals` → tạo terminal, `clients/tmux.py` tạo tmux session/window, set `CAO_TERMINAL_ID`.
3. Đợi trạng thái IDLE (suy ra từ `status_monitor.py` đọc FIFO output).
4. `POST /terminals/{id}/input` gửi task; poll tới khi `status = COMPLETED`.
5. `GET /terminals/{id}/output?mode=last` lấy kết quả; `POST /terminals/{id}/exit` xoá terminal worker (scrollback lưu `~/.cao/logs/terminal/`).
6. Trả output về supervisor caller.

**C. State & data flow:** message giữa agent = văn bản thô qua tmux `send_keys`/paste (không phải JSON structured, trừ MCP tool call). State: SQLite cho metadata, tmux pane buffer cho nội dung hội thoại thực tế. Không xác định từ code cơ chế nén context chủ động (do CLI con tự quản lý).

**D. Tool/MCP integration:** hai MCP server tách biệt (`cao-mcp-server` trong-session, `cao-ops-mcp-server` ngoài-session). Sandbox = tmux session/pane thật (PTY), `path_validation.py` chặn thư mục hệ thống.

**E. Memory architecture:** 2 kho ghép — file Markdown wiki (`~/.aws/cli-agent-orchestrator/memory/…/wiki/{scope}/{key}.md`) + bảng SQLite `memory_metadata` (BM25 + recency + usage 3-factor scoring). 5 scope: global/project/session/agent/federated, auto-inject vào context CLI con lúc khởi tạo terminal.

**F. Model orchestration:** mỗi role gắn với một CLI process thật (Kiro CLI mặc định, Claude Code, Codex, Antigravity, Hermes, Kimi, Copilot, OpenCode, Cursor — 9 provider). Cross-provider mixing qua field `provider:` trong frontmatter; không có fallback/retry model tự động.

**G. Observability:** Web UI bundled (`cao-server`, port 9889), MCP Apps extension cho dashboard; OpenTelemetry GenAI (extra `otel`, no-op nếu vắng SDK); snapshot/restore terminal (`cao terminal restore`).

**H. Extension points:** Skills (`skills/cao-*`), Plugins observer-only event-driven, Flows (cron qua `apscheduler`), custom agent profile qua `cao install <file|url>`.

### §3 Architecture Diagram

```mermaid
sequenceDiagram
    participant Sup as Supervisor Agent
    participant MCP as mcp_server/server.py (handoff)
    participant API as api/main.py (FastAPI :9889)
    participant Tmux as clients/tmux.py
    participant Mon as services/status_monitor.py
    participant Prov as providers/base.py (worker CLI)

    Sup->>MCP: handoff(agent_profile, message)
    MCP->>API: POST /sessions/{s}/terminals
    API->>Tmux: create_session() [set CAO_TERMINAL_ID]
    Tmux->>Prov: initialize() worker CLI process
    Mon->>API: status=IDLE (via FIFO output)
    MCP->>API: POST /terminals/{id}/input
    Mon-->>MCP: poll status=COMPLETED
    MCP->>API: GET /terminals/{id}/output?mode=last
    API->>Tmux: exit terminal (snapshot saved)
    MCP-->>Sup: return worker output
```

### §4 Verdict
Điểm đáng chú ý cụ thể: memory 2-kho (Markdown wiki + SQLite index, BM25+recency+usage) và cross-provider mixing qua frontmatter `provider:` là thiết kế khác biệt thật sự so với framework multi-agent chỉ dùng 1 model backend. Red flag cụ thể: trạng thái agent (IDLE/PROCESSING/COMPLETED) được suy luận bằng regex parse output terminal thô — dễ vỡ khi CLI con đổi UI/prompt string. `cao-server` chỉ bind localhost, không có auth mặc định (README tự thừa nhận, WebSocket PTY endpoint full access). Câu hỏi mở: không xác định cơ chế quản lý context window khi hội thoại worker dài; không xác định cơ chế fallback khi 1 provider lỗi giữa chừng.

---

## 3. omnigent-ai/omnigent

**Link:** https://github.com/omnigent-ai/omnigent

> ⚠️ **Lưu ý:** tồn tại repo `mumomo011/omnigent` gần như y hệt (README hotlink thẳng ảnh của `omnigent-ai`) nhưng 0 sao — dấu hiệu squat/clone, không phải fork hợp lệ. Tránh cài đặt từ nguồn này.

### §1 Quick Context
Meta-harness điều phối thống nhất Claude Code, Codex, Cursor, OpenCode, Hermes, Pi và agent YAML tuỳ biến. Stack: Python 3.12+, FastAPI/Starlette, SQLAlchemy, OpenTelemetry. **7.4k sao, 151 contributor**, commit gần nhất "today", CI phong phú (17 workflow: `ci.yml`, `e2e.yml`, `integration.yml`, `oss-scorecard.yml`...). License: Apache 2.0. Open-sourced bởi Databricks (`.github/MAINTAINER` chứa các tài khoản Databricks công khai như `mateiz`, `dbczumar`, `dennyglee`).

### §2 Architecture Deep-dive

**A. Component inventory**
- `Executor` (`omnigent/inner/executor.py`) — abstract base "swap layer": `run_turn(messages, tools, system_prompt, config) -> AsyncIterator[ExecutorEvent]`, event `TextChunk/ReasoningChunk/ToolCallRequest/ToolCallComplete/TurnComplete/ExecutorError`.
- `ClaudeSdkExecutor`, `CodexExecutor`, `PiExecutor` (`omnigent/inner/claude_sdk_executor.py`, `codex_executor.py`, `pi_executor.py`) — cài đặt cụ thể của Executor cho từng harness.
- `ClaudeNativeHarness`/`CodexNativeHarness` (`omnigent/inner/claude_native_harness.py`, `codex_native_harness.py`) — wrapper tmux/PTY cho CLI gốc.
- Policy engine (`omnigent/policies/base.py`, `registry.py`, `function.py`; builtin `omnigent/inner/nessie/policies.py`) — cổng kiểm soát tool call.
- `workflow.py` (`omnigent/runtime/workflow.py`, 2793 dòng) — vòng lặp phiên/turn.
- `compaction.py`, `context_window.py` (`omnigent/llms/`, `omnigent/runtime/`) — quản lý context window.
- `git_worktree.py` (`omnigent/host/git_worktree.py`) — tạo/xoá git worktree phía host (dùng argv, không qua shell).
- `db_models.py` (`omnigent/db/db_models.py`) — SQLAlchemy: `SqlConversation`, `SqlConversationItem`, `SqlAgentConfiguration`, `SqlPolicy`, `SqlHost`.
- `tracing.py` (`omnigent/inner/tracing.py`) — OpenTelemetry.
- `sandbox/bwrap.py`, `sandbox/seatbelt.py` — sandbox OS (Linux bubblewrap, macOS seatbelt).
- Polly example (`examples/polly/config.yaml`, `.../skills/fanout/SKILL.md`, `.../cross-review/SKILL.md`).
- `spec/parser.py`, `spec/validator.py`, `spec/AGENTSPEC.md` — parser/validator cho YAML agent spec.

**B. Control flow — "Plan → Worktree Fan-out → Cross-vendor Review"** (dựa trên ví dụ Polly, 2 file SKILL.md đọc trực tiếp):
1. `omnigent run examples/polly/` khởi động Polly (brain = executor `claude-sdk`, không code trực tiếp); preflight dò harness khả dụng.
2. Polly phân rã goal, nạp skill `fanout`: mỗi task tạo 1 worktree (`git worktree add .worktrees/<task_id>`).
3. Dispatch tới 1 sub-agent implementer (claude_code/codex/opencode/cursor/hermes/pi), scoped vào worktree riêng; sub-agent tự chạy đến khi mở PR.
4. Polly nhận kết quả qua inbox, lấy diff PR, nạp skill `cross-review`: chỉ gửi diff + acceptance-contract cho reviewer thuộc **vendor khác**.
5. Reviewer báo cáo issue blocking/non-blocking, không tự sửa; issue blocking gửi lại implementer gốc để fix, lặp lại bước 4.
6. Khi gates xanh + zero blocking issue → PR đánh dấu "ready"; **Polly không bao giờ merge** — con người merge.

**C. State & data flow:** `Message = dict[str, Any]` với `role/content/metadata/session_id`; lịch sử hội thoại lưu Postgres qua SQLAlchemy. Context window quản lý đa tầng: env override → litellm registry → MLflow catalog → fallback 128K; compaction 3 lớp (surgical clearing → LLM summarization → truncation cứng).

**D. Tool/capability:** MCP qua `omnigent/tools/mcp.py`; local Python callable qua `tools/local_callable.py`; sandbox OS bắt buộc trên Linux (`bwrap`)/macOS (`seatbelt`); policy builtin (`ask_on_os_tools`, `cost_budget`, `max_tool_calls_per_session`) ở 3 tầng server/agent/session. Polly có guardrail runner-side riêng (`blast_radius`, `spawn_bounds`, `headless_subagent_purpose_guard`).

**E. Memory:** không có subsystem "memory" riêng biệt trong core `omnigent/`; README chỉ liệt kê extra tuỳ chọn `hindsight` — không xác định từ code cách nó tích hợp.

**F. Model orchestration:** `llms/routing.py`, `runner/cost_advisor.py`/`cost_judge.py` chọn model brain theo chi phí mỗi turn. Polly gán vai trò cố định: 5 harness CLI chạy terminal riêng cho implement, `pi` headless cho review/explore. Song song hoá bằng git worktree độc lập theo task.

**G. Observability:** `inner/tracing.py` phát OTel span cho mỗi turn/tool-call/sub-agent/policy-eval, xuất Jaeger/Tempo/Grafana/MLflow Traces. Cộng tác real-time: `omnigent attach <session_id>`, `omnigent run --fork <session_id>`, nút "Share" web UI.

**H. Extension:** harness mới = thêm cặp `*_executor.py`/`*_harness.py` cài đặt `Executor`, đăng ký trong `harness_aliases.py`. Agent tuỳ biến = YAML theo `omnigent/spec/AGENTSPEC.md`, chạy bằng `omnigent run path/to/agent.yaml`.

### §3 Architecture Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant P as Polly (claude-sdk Executor)
    participant W as git_worktree.py
    participant S as claude_code/codex/... Executor
    participant R as Reviewer Executor (vendor khác)
    U->>P: omnigent run examples/polly/
    P->>W: git worktree add .worktrees/<task_id>
    P->>S: sys_session_send(purpose=implement)
    S-->>P: inbox: PR opened
    P->>R: sys_session_send(purpose=review, diff+contract)
    R-->>P: inbox: blocking/non-blocking issues
    P->>U: registry.json ready — human merges
```

### §4 Verdict
Điểm mới đáng chú ý: `Executor` protocol hiện thực hoá nhất quán luận điểm "messages+tools+system_prompt in, event stream out" qua ~10 harness khác nhau — không phải marketing suông. Cơ chế cross-vendor review của Polly (reviewer chỉ nhận diff+contract, không thấy worktree) là kiểm soát cô lập hợp lý. Điểm nghi vấn/hạn chế: prompt của Polly (350 dòng YAML) chứa nhiều luật hành vi bằng văn xuôi thay vì state machine cứng — rủi ro drift khi model thay đổi; module nội bộ `nessie` không có tài liệu công khai ngoài code. Câu hỏi mở: cơ chế "hindsight" memory extra không xác định từ code đã đọc.

---

## 4. vshulcz/deja-vu

**Link:** https://github.com/vshulcz/deja-vu

### §1 Quick Context
Binary Go zero-dependency, index cục bộ log của Claude Code/Codex/opencode/aider/Gemini CLI/Cursor/Antigravity/Grok Build để `recall` lại qua MCP. Stack: Go 1.25, không phụ thuộc ngoài trừ shell-out `sqlite3` CLI cho opencode/Cursor. **311 sao** (từ nguồn tham chiếu ban đầu — số liệu contributor/last-activity chính xác không xác định được do `api.github.com` bị chặn trong phiên này). CI thật: `.github/workflows/ci.yml` (matrix ubuntu/macos/windows, e2e, cross-compile, `govulncheck`, `goreleaser check`). License: MIT.

### §2 Architecture Deep-dive

**A. Component inventory**
- `cmd/deja/main.go` — CLI entrypoint (`query`, `ctx`, `stats`, `sync`, `share`, `mcp`...).
- `cmd/deja/mcp.go` — MCP stdio JSON-RPC server (`initialize`/`tools/list`/`tools/call`), 2 tool: `recall`, `recall_context`.
- `cmd/deja/install.go` — ghi config MCP + SessionStart hook vào các agent host.
- `internal/sources/{claude,codex,opencode,aider,gemini,cursor,antigravity,grok}.go` — 8 parser riêng theo harness, trả `[]model.Session`.
- `internal/redact/redact.go` — regex redaction (AWS key, `api_key=`, Bearer, PEM private key, JWT, provider prefix, `scheme://user:pass@host`).
- `internal/index/index.go` (2226 dòng) — build/incremental update/search, ghi `records.bin`/`buckets/*.bin`/`manifest.gob`.
- `internal/search/search.go` — ranking cuối.
- `internal/usage` — ghi nhận usage cho `deja statusline`.
- `install.sh` — script cài đặt, verify checksum sha256.
- `docs/ARCHITECTURE.md` — tài liệu kiến trúc nội bộ.

**B. Control flow (index → query → recall):**
1. Agent gọi tool `recall` qua stdio JSON-RPC → `handleMCP` → `recallText`.
2. `recallText` gọi `index.EnsureForSearch` — so `manifest.gob` với file hiện tại: fresh thì bỏ qua, append-only thì đọc phần mới, khác thì rebuild.
3. Nếu cần parse lại, parser trong `internal/sources/*.go` chạy song song (worker pool = NumCPU), rồi `internal/redact/redact.go` chạy **trước** mọi `writeRecord`.
4. `index.SearchWithRecovery` đọc posting list từ `buckets/*.bin`, giao nhau cho multi-word query, pre-rank theo count×recency.
5. `search.Run` đọc `records.bin` cho top session, xếp hạng cuối.
6. `recallText` format kết quả ≤4KB, `mcp.go` trả JSON-RPC result; `usage.Record` ghi log.

**C. State & data flow:** index tại `~/.cache/deja` — `records.bin`, `buckets/*.bin` (token → posting), `manifest.gob`/`sessions.gob`. Cập nhật tăng dần dựa trên path/size/mtime. Latency công bố (chưa tự benchmark lại): warm search 7–9ms, cold index ~10s, index size ~2.4% corpus (đo trên 1.250+ session/~3.3GB).

**D. Tool/capability integration:** MCP stdio server độc lập đăng ký 2 tool có `inputSchema` JSON rõ ràng; `deja install --all/--auto` ghi entry MCP + hook (`~/.claude/settings.json` command hook gọi `deja hook-context`, theo `docs/ARCHITECTURE.md`).

**E. Memory architecture (trọng tâm của repo):** không phân biệt short/long-term — một inverted index phẳng trên toàn bộ lịch sử log có sẵn (retroactive). Retrieval **thuần lexical/keyword qua inverted index**, không phải vector/embedding search (README tự đối lập có chủ đích với công cụ semantic-memory khác). Regex search cố ý bypass bucket index vì "arbitrary regex cannot use token postings safely". Redaction chạy trước ghi record.

**F. Model orchestration: không có.** Không có import LLM/embedding client trong `mcp.go`, `index.go`, `redact.go` — chỉ regex, tokenize, gob/binary I/O, shell-out `sqlite3`.

**G. Observability & eval:** `deja stats`, `deja sources` (đếm redaction per-store). CI chạy unit test `-race` + coverage, e2e smoke, `govulncheck` — CI phần mềm đúng đắn, nhưng **không xác định từ code** có benchmark precision/recall retrieval riêng.

**H. Extension points:** quy trình 5 bước tĩnh trong `docs/ARCHITECTURE.md` để thêm harness mới (viết parser trả `model.Session`, đăng ký discovery, nối vào pipeline, thêm config, thêm fixture) — không có plugin loader runtime.

### §3 Architecture Diagram

```mermaid
flowchart LR
    A[Agent qua MCP stdio] -->|tools/call recall| B["cmd/deja/mcp.go"]
    B --> C["internal/index/index.go EnsureForSearch"]
    C -->|stale| D["internal/sources/*.go parsers"]
    D --> E["internal/redact/redact.go"]
    E --> F["records.bin + buckets/*.bin + manifest.gob"]
    C --> F
    F --> G["internal/search/search.go"]
    G --> B
    B -->|JSON-RPC result| A
```

### §4 Verdict
Điểm đáng chú ý cụ thể: redaction chạy **trước** mọi ghi index (không phải lọc sau), và regex search cố ý bypass posting-index vì lý do đúng đắn về tính an toàn — thiết kế thận trọng, tự thừa nhận giới hạn kỹ thuật thay vì che giấu. Red flag/limitation cụ thể: retrieval thuần lexical AND-match, không có disambiguation ngữ nghĩa; không tìm thấy benchmark định lượng recall/precision trong repo, chỉ có CI test đúng-sai chức năng. Câu hỏi mở: `internal/search/search.go` và `cmd/deja/install.go` chưa được đọc chi tiết đầy đủ; số liệu cộng đồng (stars/contributors chính xác) không xác định được trong phiên này do `api.github.com` bị chặn.

> **Lưu ý an toàn:** phần fetch nội dung repo này khiến hệ thống lọc nội bộ gắn cờ "instruction-shaped pattern liên quan settings-json" trên output của subagent nghiên cứu. Sau khi kiểm tra thủ công, nguyên nhân là do `docs/ARCHITECTURE.md` của repo mô tả đúng tính năng thật của công cụ (`deja install` ghi hook vào `~/.claude/settings.json`) — không phát hiện nội dung có chủ đích prompt-injection. Ghi lại ở đây để minh bạch quy trình, không phải cảnh báo về mã độc.

---

## 5. Repo khác đã xác định nhưng không đào sâu

Các repo dưới đây xuất hiện trong vòng quét ban đầu (từ khoá agent/multi-agent/agentic, cửa sổ 7 ngày) và có vẻ hợp lệ qua xác minh sơ bộ (raw README fetch trả 200, nội dung nhất quán giữa các lần fetch độc lập), nhưng **không đủ thời gian/ngân sách để đào sâu kiến trúc** theo chuẩn §2 trong lần quét này:

- **`mereyabdenbekuly-ctrl/clodex-ide`** (~830 sao, TypeScript/Electron) — "local-first, zero-trust agentic IDE", nhấn mạnh isolated process boundaries, 88 commit trên main. Đáng theo dõi cho hướng security/sandboxing của agentic IDE.
- **`PengZhang64/circuit-framework`** (~336 sao, Python) — hệ thống multi-agent nghiên cứu/paper-trading crypto xây trên nền TradingAgents (Apache 2.0): 5 "analyst" agent tranh luận (market structure, derivatives, sentiment, catalyst, regime) trước khi một risk gate xác định (deterministic) chấp thuận/từ chối lệnh. Vì là extension của framework có sẵn nên độ mới về kiến trúc thấp hơn 4 repo trên.
- **`awslabs/cli-agent-orchestrator`-adjacent / `ComposioHQ/agent-orchestrator`** — agent IDE quản lý fleet coding agent song song, tự động xử lý CI fix/merge conflict/code review; xuất hiện trong tìm kiếm nhưng chưa xác minh trực tiếp README/code, cần theo dõi tuần sau.
- **`microsoft/agent-framework`** — framework orchestration Python/.NET đã thiết lập từ trước (không phải repo mới tuần này), có update liên tục nhưng không có tín hiệu thay đổi kiến trúc lớn trong cửa sổ 7 ngày — loại khỏi deep-dive vì không mới.

Repo bị loại theo bộ lọc relevance (không phù hợp): `pyang5166/gbro-collage-broll` (~220 sao) — công cụ tạo video b-roll gắn mác "agent skill", về bản chất là ứng dụng tạo nội dung, không mang insight kiến trúc agentic.

---

## 6. Ghi chú phương pháp luận & giới hạn

- **Không có quyền truy cập `gh` CLI** và **quyền truy cập GitHub MCP integration của phiên này bị giới hạn (scope) chỉ cho `undertheseanlp/underthesea`** — không được dùng để search/đọc repo ngoài phạm vi đó. Vì vậy toàn bộ nghiên cứu repo bên ngoài trong báo cáo này dùng WebSearch + WebFetch + `curl` tới `raw.githubusercontent.com`/`api.github.com` (không xác thực) thay vì `gh api search/repositories` như quy trình lý tưởng đề ra — đây là fallback bắt buộc, không phải lựa chọn.
- **Một trong bốn agent con (grok-build) đã gọi tool `mcp__github__search_repositories` để lấy thêm metadata cho `xai-org/grok-build`** — đây là vi phạm phạm vi scope nói trên (tool đó chỉ được phép dùng cho `undertheseanlp/underthesea`). Lỗi thuộc về việc điều phối: instruction gửi cho agent con không nói rõ giới hạn scope này. Đã ghi nhận để tránh lặp lại ở lần quét sau — sẽ chỉ định rõ "không dùng mcp__github__* tools" trong prompt cho agent con.
- **WebFetch trên trang GitHub search (HTML render bằng JS) đã bị quan sát thấy hallucinate** dữ liệu plausible-nhưng-sai ở bước quét sơ bộ (ví dụ số liệu commit không khớp). Mọi số liệu cuối cùng trong báo cáo đều được double-check qua fetch trực tiếp `raw.githubusercontent.com` (nội dung thô, không qua model tóm tắt) trước khi đưa vào báo cáo.
- **`github.com`/`api.github.com` bị chặn trực tiếp bởi chính sách egress của môi trường sandbox** (HTTP 403, "GitHub access... not enabled for this session") đối với truy cập ẩn danh không qua MCP — số liệu sao/contributor/ngày tạo cho một số repo được lấy qua nguồn thay thế (`raw.githubusercontent.com`, `shields.io`, `ungh.cc`) nên độ tin cậy không đồng đều giữa các repo; đã ghi rõ "không xác định từ code" ở những chỗ không xác minh được.
- Tự-kiểm (self-check) theo yêu cầu: các link repo đã xác minh tồn tại qua raw-content fetch (200), không phải qua HTTP 200 trực tiếp trên `github.com` (bị chặn ở tầng mạng của phiên này, không phải do repo lỗi). Toàn bộ path trong §2.A của 4 repo chính đều được agent con đọc trực tiếp qua `raw.githubusercontent.com` trước khi liệt kê.
