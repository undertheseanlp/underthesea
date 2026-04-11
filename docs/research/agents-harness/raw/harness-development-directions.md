# Các hướng phát triển Agent Harness cho Underthesea

## Phân tích hiện trạng Agentic AI & Đề xuất theo chuẩn Industry Standard

---

## 1. Hiện trạng Agent Module

### 1.1 Kiến trúc hiện tại

Agent module (`underthesea/agent/`) hiện tại là một **OpenAI function calling wrapper đơn giản** gồm 4 file:

```
underthesea/agent/
├── agent.py          # Agent class + _AgentInstance singleton
├── llm.py            # LLM wrapper (OpenAI / Azure OpenAI)
├── tools.py          # Tool wrapper cho function calling
└── default_tools.py  # 12 tools mặc định (calculator, shell, web_search...)
```

**Luồng hoạt động hiện tại:**

```
User message
    ↓
Thêm vào history
    ↓
Gửi tới OpenAI API (kèm tool definitions)
    ↓
┌── Vòng lặp (tối đa max_iterations=10 lần) ──┐
│  LLM trả về tool_calls?                       │
│  → Có: thực thi tool, thêm kết quả, lặp lại  │
│  → Không: trả response cho user               │
└────────────────────────────────────────────────┘
```

### 1.2 Khoảng cách so với Industry Standard

So sánh với các chuẩn từ Anthropic Engineering và LangChain Deep Agents SDK:

| Khả năng | Industry Standard | Underthesea hiện tại |
|----------|-------------------|---------------------|
| **Multi-session continuity** | Context reset + structured handoff giữa các session | Không có. History mất khi process kết thúc |
| **Theo dõi tiến độ** | JSON-based feature list với trạng thái pass/fail | Không có |
| **Quản lý context** | Context reset > compaction khi context đầy | Không có. History tích lũy vô hạn đến khi tràn |
| **Tách biệt Generator/Evaluator** | Agent tạo output và agent đánh giá là 2 thực thể riêng | Không có. Một agent làm tất cả |
| **Lập kế hoạch (Planning)** | Planner agent phân rã task phức tạp | Không có. LLM tự quyết định bước tiếp theo |
| **Ủy quyền subagent** | Spawn subagents cho subtasks, bảo toàn context | Không có |
| **Tối ưu token** | Tóm tắt hội thoại, loại bỏ tool results lớn | Không có. Giữ toàn bộ history |
| **Đánh giá agent** | Eval harness với task/trial/grader/transcript | Không có |
| **Multi-provider** | Hỗ trợ nhiều LLM backends | Chỉ OpenAI + Azure OpenAI |
| **Giao tiếp giữa agents** | Qua structured files hoặc message protocols | Không có |
| **Sprint contracts** | Định nghĩa "hoàn thành" trước khi bắt đầu | Không có |
| **Xử lý lỗi nâng cao** | Retry logic, fallback strategies, graceful degradation | Cơ bản: tool lỗi trả string error, vượt iteration thì RuntimeError |

### 1.3 Điểm mạnh hiện có (để xây dựng tiếp)

- **Tool system linh hoạt**: `Tool` class tự động trích xuất JSON schema từ Python function signature
- **12 default tools** đã sẵn sàng (web search, file I/O, shell, python exec...)
- **Lazy initialization**: LLM chỉ khởi tạo khi cần
- **An toàn cơ bản**: Shell tool chặn lệnh nguy hiểm, Python tool dùng restricted globals
- **Test coverage tốt**: 3 test files bao phủ agent, llm, tools

---

## 2. Các hướng phát triển Agent Harness

### Hướng 1: Multi-Session Agent với Context Management

**Vấn đề cốt lõi**: Agents hiện tại không thể làm việc xuyên session. Khi context đầy hoặc process kết thúc, toàn bộ tiến độ mất.

**Bài học từ Anthropic**: Context reset (xóa sạch hội thoại, tạo agent mới với structured handoff) hiệu quả hơn compaction (tóm tắt hội thoại cũ). Đây là phát hiện phản trực giác nhưng được kiểm chứng nhất quán.

**Kiến trúc đề xuất**:

```
underthesea/agent/
├── harness/
│   ├── __init__.py
│   ├── session.py             # Quản lý session lifecycle
│   ├── context.py             # Context reset & handoff protocols
│   ├── progress.py            # Theo dõi tiến độ (JSON-based)
│   └── persistence.py         # Lưu trữ state giữa các session
```

**Session lifecycle**:

```
Session 1 (Khởi tạo):
  1. Nhận task từ user
  2. Tạo progress.json với danh sách subtasks
  3. Thực thi subtask đầu tiên
  4. Lưu handoff file: trạng thái + kết quả + subtask tiếp theo
  5. Kết thúc session

Session N (Tiếp tục):
  1. Đọc handoff file → hiểu context trước đó
  2. Đọc progress.json → biết subtask nào đã xong
  3. Chọn subtask tiếp theo chưa hoàn thành
  4. Thực thi
  5. Cập nhật progress + handoff file
  6. Kết thúc session
```

**Cấu trúc progress file**:

```json
{
  "task": "Phân tích và tóm tắt 100 tài liệu pháp luật",
  "created_at": "2026-03-28T10:00:00",
  "subtasks": [
    {
      "id": 1,
      "description": "Đọc và phân loại 100 tài liệu theo lĩnh vực",
      "status": "completed",
      "result_summary": "Phân loại xong: 40 dân sự, 30 hình sự, 30 hành chính"
    },
    {
      "id": 2,
      "description": "Tóm tắt nhóm tài liệu dân sự",
      "status": "in_progress",
      "result_summary": null
    }
  ]
}
```

**Cấu trúc handoff file**:

```json
{
  "session_id": 3,
  "previous_session_summary": "Hoàn thành phân loại. Bắt đầu tóm tắt nhóm dân sự, đã xong 20/40.",
  "current_subtask_id": 2,
  "context_for_next_session": "File output đang ở /tmp/summaries/. Dùng template tóm tắt ở /tmp/template.md",
  "artifacts": ["/tmp/summaries/", "/tmp/classification.json"],
  "warnings": "Tài liệu #15 bị lỗi encoding, cần xử lý riêng"
}
```

**Sử dụng**:

```python
from underthesea.agent import Agent
from underthesea.agent.harness import SessionManager

agent = Agent(
    name="legal_analyst",
    tools=[read_file_tool, write_file_tool, web_search_tool],
    instruction="Bạn là chuyên gia phân tích pháp luật Việt Nam."
)

session = SessionManager(
    agent=agent,
    progress_file="progress.json",
    handoff_dir=".agent_state/"
)

# Session tự động: đọc state cũ → thực thi → lưu state mới
session.run()

# Hoặc chạy liên tục nhiều session
session.run_until_complete(max_sessions=10)
```

**Ưu tiên**: **CAO** - Đây là nền tảng cho mọi hướng khác. Không có multi-session thì agent không thể xử lý task phức tạp.

---

### Hướng 2: Kiến trúc đa Agent (Planner / Executor / Evaluator)

**Vấn đề cốt lõi**: Một agent đơn lẻ vừa lập kế hoạch, vừa thực thi, vừa tự đánh giá → dẫn đến self-evaluation bias (luôn tự khen kết quả của mình) và over-ambition (cố làm quá nhiều cùng lúc).

**Bài học từ Anthropic**: Tách biệt tạo sinh và đánh giá "dễ giải quyết hơn nhiều so với việc buộc generator tự phê bình". Chi phí tăng 20x nhưng chất lượng output tăng đáng kể (solo $9/20 phút vs harness $200/6 giờ).

**Kiến trúc đề xuất**:

```
underthesea/agent/
├── harness/
│   ├── planner.py             # Agent lập kế hoạch
│   ├── executor.py            # Agent thực thi
│   ├── evaluator.py           # Agent đánh giá
│   ├── orchestrator.py        # Điều phối 3 agents
│   └── contracts.py           # Sprint contracts
```

**Vai trò từng agent**:

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                          │
│  Điều phối luồng: Planner → Executor → Evaluator        │
│  Quản lý vòng lặp cải thiện (5-15 iterations)           │
└────┬──────────────────┬──────────────────┬──────────────┘
     ↓                  ↓                  ↓
┌─────────┐      ┌───────────┐      ┌───────────┐
│ Planner │      │ Executor  │      │ Evaluator │
│         │      │           │      │           │
│ Nhận mô │ ──→  │ Thực thi  │ ──→  │ Kiểm tra  │
│ tả task │      │ từng bước │      │ kết quả   │
│         │      │ theo kế   │      │ theo tiêu │
│ Tạo kế  │      │ hoạch     │      │ chí đã    │
│ hoạch   │      │           │      │ thỏa thuận│
│ chi tiết│      │ Tạo       │      │           │
│ + tiêu  │      │ artifacts │      │ Phản hồi  │
│ chí đánh│      │           │      │ chi tiết  │
│ giá     │      │           │      │ cho lần   │
│         │      │           │      │ lặp tiếp  │
└─────────┘      └───────────┘      └───────────┘
```

**Sprint Contract** (thỏa thuận giữa Executor và Evaluator trước khi bắt đầu):

```json
{
  "sprint_goal": "Tạo báo cáo phân tích thị trường bất động sản Q1/2026",
  "acceptance_criteria": [
    {
      "criterion": "Bao phủ ít nhất 3 phân khúc: căn hộ, nhà phố, đất nền",
      "weight": 0.3,
      "threshold": "pass/fail"
    },
    {
      "criterion": "Mỗi phân khúc có dữ liệu giá từ ít nhất 2 nguồn",
      "weight": 0.3,
      "threshold": "pass/fail"
    },
    {
      "criterion": "Có phần dự báo xu hướng với lập luận rõ ràng",
      "weight": 0.2,
      "threshold": "LLM judge score >= 7/10"
    },
    {
      "criterion": "Báo cáo dưới 3000 từ, có biểu đồ minh họa",
      "weight": 0.2,
      "threshold": "pass/fail"
    }
  ],
  "max_iterations": 5
}
```

**Giao tiếp giữa agents qua structured files** (không qua hội thoại trực tiếp):

```python
# Planner ghi ra file
plan = planner.create_plan(task_description)
plan.save("artifacts/plan.json")

# Executor đọc plan, thực thi, ghi kết quả
result = executor.execute(plan_file="artifacts/plan.json")
result.save("artifacts/output/")

# Evaluator đọc contract + output, đánh giá
evaluation = evaluator.evaluate(
    contract_file="artifacts/contract.json",
    output_dir="artifacts/output/"
)

# Nếu chưa đạt → Executor nhận feedback, lặp lại
if not evaluation.passed:
    result = executor.revise(
        feedback_file="artifacts/feedback.json",
        previous_output="artifacts/output/"
    )
```

**Sử dụng**:

```python
from underthesea.agent.harness import Orchestrator

orchestrator = Orchestrator(
    planner_model="claude-sonnet-4-6",
    executor_model="claude-sonnet-4-6",
    evaluator_model="claude-sonnet-4-6",
    tools=[web_search_tool, read_file_tool, write_file_tool],
    max_iterations=10,
    artifacts_dir=".agent_artifacts/"
)

result = orchestrator.run(
    task="Nghiên cứu và viết báo cáo về ứng dụng AI trong y tế Việt Nam 2025-2026"
)

print(result.status)        # "completed" | "max_iterations_reached"
print(result.iterations)    # Số vòng lặp đã thực hiện
print(result.artifacts)     # Đường dẫn các file output
print(result.eval_history)  # Lịch sử đánh giá qua các vòng
```

**Ưu tiên**: **CAO** - Đây là pattern đã được Anthropic kiểm chứng tạo ra sự khác biệt lớn nhất về chất lượng output.

---

### Hướng 3: Multi-Provider & Model Routing

**Vấn đề cốt lõi**: Agent hiện tại chỉ hỗ trợ OpenAI và Azure OpenAI. Không thể dùng Claude, Gemini, local models, hay kết hợp nhiều model cho các vai trò khác nhau.

**Bài học từ industry**: Kiến trúc đa agent cần multi-provider vì:
- Planner có thể dùng model mạnh hơn (Opus), Executor dùng model nhanh hơn (Sonnet/Haiku)
- Evaluator nên dùng model khác Generator để tránh bias cùng họ model
- Local models cho xử lý dữ liệu nhạy cảm

**Kiến trúc đề xuất**:

```
underthesea/agent/
├── providers/
│   ├── __init__.py
│   ├── base.py              # Abstract base class cho providers
│   ├── openai_provider.py   # OpenAI + Azure (refactor từ llm.py hiện tại)
│   ├── anthropic_provider.py # Claude API
│   ├── google_provider.py   # Gemini API
│   ├── local_provider.py    # Ollama / vLLM / llama.cpp
│   └── router.py            # Model routing logic
```

**Provider interface chuẩn**:

```python
class BaseProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Message:
        """Gửi messages, nhận response. Hỗ trợ tool calling nếu provider cho phép."""

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Provider có hỗ trợ function/tool calling không."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên provider (openai, anthropic, google, local)."""
```

**Model router**:

```python
from underthesea.agent.providers import ModelRouter

router = ModelRouter({
    "planner": {"provider": "anthropic", "model": "claude-opus-4-6"},
    "executor": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "evaluator": {"provider": "openai", "model": "gpt-4o"},
    "fast_tasks": {"provider": "local", "model": "llama3.2"},
})

# Lấy provider theo vai trò
planner_llm = router.get("planner")
executor_llm = router.get("executor")
```

**Sử dụng trong kiến trúc đa agent**:

```python
orchestrator = Orchestrator(
    router=ModelRouter({
        "planner": {"provider": "anthropic", "model": "claude-opus-4-6"},
        "executor": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "evaluator": {"provider": "openai", "model": "gpt-4o"},
    }),
    tools=[...],
)
```

**Ưu tiên**: **CAO** - Điều kiện tiên quyết để triển khai kiến trúc đa agent hiệu quả. Hiện tại chỉ hỗ trợ OpenAI là hạn chế lớn.

---

### Hướng 4: Hệ thống Tool nâng cao

**Vấn đề cốt lõi**: Tool system hiện tại đơn giản - chỉ wrap Python function thành OpenAI format. Thiếu:
- Tool composition (kết hợp tools thành workflow)
- Tool middleware (logging, rate limiting, retry)
- Dynamic tool registration (thêm/bớt tools runtime)
- Tool sandboxing (cô lập thực thi cho an toàn)

**Kiến trúc đề xuất**:

```
underthesea/agent/
├── tools/
│   ├── __init__.py
│   ├── base.py                # Tool base class (refactor từ tools.py)
│   ├── registry.py            # Dynamic tool registry
│   ├── middleware.py           # Logging, retry, rate limit, timeout
│   ├── sandbox.py             # Tool execution sandboxing
│   ├── composite.py           # Tool composition (pipeline tools)
│   ├── builtin/
│   │   ├── core.py            # calculator, datetime, string, json
│   │   ├── web.py             # search, fetch, wikipedia
│   │   ├── system.py          # file I/O, shell, python
│   │   ├── nlp.py             # Underthesea NLP dưới dạng tools
│   │   └── data.py            # Dataset management tools
│   └── mcp/
│       └── adapter.py         # MCP (Model Context Protocol) adapter
```

**Tool middleware**:

```python
from underthesea.agent.tools import Tool, with_retry, with_logging, with_timeout

@with_logging
@with_retry(max_attempts=3, backoff=2.0)
@with_timeout(seconds=30)
def fetch_and_parse(url: str) -> dict:
    """Tải và phân tích nội dung trang web."""
    ...

tool = Tool(fetch_and_parse)
```

**NLP tools** (biến underthesea pipelines thành agent tools):

```python
from underthesea.agent.tools.builtin.nlp import (
    word_tokenize_tool,    # Tách từ tiếng Việt
    pos_tag_tool,          # Gán nhãn từ loại
    ner_tool,              # Nhận dạng thực thể
    sentiment_tool,        # Phân tích cảm xúc
    classify_tool,         # Phân loại văn bản
    translate_tool,        # Dịch Việt-Anh
)

# Agent có thể dùng NLP tools để phân tích dữ liệu
analyst = Agent(
    name="text_analyst",
    tools=[ner_tool, sentiment_tool, classify_tool, read_file_tool],
    instruction="Phân tích văn bản tiếng Việt bằng các công cụ NLP."
)
```

**MCP adapter** (hỗ trợ Model Context Protocol):

```python
from underthesea.agent.tools.mcp import MCPAdapter

# Kết nối tới MCP server bên ngoài
mcp_tools = MCPAdapter.from_server("http://localhost:8080/mcp")
agent = Agent(name="agent", tools=default_tools + mcp_tools)
```

**Ưu tiên**: **TRUNG BÌNH-CAO** - Nâng cấp đáng kể khả năng agent, đặc biệt NLP tools tạo điểm khác biệt cho underthesea.

---

### Hướng 5: Agent Evaluation Harness

**Vấn đề cốt lõi**: Không có cách đo lường chất lượng agent. Không biết agent version A có tốt hơn version B không. Không phát hiện được regression khi thay đổi prompts, tools, hoặc model.

**Bài học từ Anthropic**: "Evals tốt giúp team ship AI agents tự tin hơn." Bắt đầu với 20-50 tasks từ lỗi thực tế. Đánh giá kết quả cuối cùng (outcome), không đánh giá quy trình (transcript).

**Kiến trúc đề xuất**:

```
underthesea/agent/
├── eval/
│   ├── __init__.py
│   ├── harness.py             # Core evaluation runner
│   ├── task.py                # Định nghĩa eval task
│   ├── trial.py               # Thực thi trial cô lập
│   ├── graders/
│   │   ├── __init__.py
│   │   ├── code_grader.py     # Kiểm tra deterministic (file tồn tại, JSON đúng schema...)
│   │   ├── model_grader.py    # LLM judge đánh giá chất lượng
│   │   ├── human_grader.py    # Interface cho human review
│   │   └── composite.py       # Kết hợp nhiều grader
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── task_completion.py # Tỷ lệ hoàn thành task
│   │   ├── turn_efficiency.py # Số lượt tool call / task
│   │   ├── cost.py            # Chi phí token / task
│   │   └── pass_at_k.py       # pass@k và pass^k
│   ├── suites/
│   │   ├── __init__.py
│   │   ├── tool_use.py        # Đánh giá khả năng sử dụng tools
│   │   ├── planning.py        # Đánh giá khả năng lập kế hoạch
│   │   ├── research.py        # Đánh giá khả năng nghiên cứu
│   │   └── conversation.py    # Đánh giá chất lượng hội thoại
│   └── reporters/
│       ├── console.py
│       └── json_reporter.py
```

**Định nghĩa eval task**:

```python
from underthesea.agent.eval import EvalTask, CodeGrader, ModelGrader

task = EvalTask(
    name="research_vietnam_gdp",
    description="Tìm GDP Việt Nam 2025 và so sánh với 2024",
    agent_config={
        "tools": [web_search_tool, calculator_tool],
        "instruction": "Bạn là trợ lý nghiên cứu kinh tế.",
    },
    user_message="GDP Việt Nam năm 2025 là bao nhiêu? Tăng bao nhiêu % so với 2024?",
    graders=[
        CodeGrader(
            name="contains_numbers",
            check=lambda outcome: any(c.isdigit() for c in outcome.response)
        ),
        CodeGrader(
            name="mentions_comparison",
            check=lambda outcome: "%" in outcome.response or "tăng" in outcome.response
        ),
        ModelGrader(
            name="factual_accuracy",
            rubric="Đánh giá tính chính xác của số liệu GDP. Cho điểm 1-10.",
            threshold=7
        ),
        CodeGrader(
            name="used_search_tool",
            check=lambda outcome: any(
                tc.tool_name == "web_search" for tc in outcome.transcript.tool_calls
            )
        ),
    ],
    max_turns=15,
    timeout=120,
)
```

**Các loại đánh giá theo kiểu agent** (theo chuẩn Anthropic):

| Loại agent | Grader chính | Ví dụ |
|------------|-------------|-------|
| **Conversational** | Task completion + turn efficiency + LLM rubric | Chatbot tư vấn pháp luật: trả lời đúng + dưới 5 lượt |
| **Research** | Groundedness + coverage + source quality | Agent nghiên cứu: thông tin có nguồn + bao phủ đủ chủ đề |
| **Tool-use** | Tool selection accuracy + output correctness | Agent phân tích: chọn đúng tool + kết quả đúng |
| **Long-running** | Subtask completion rate + final outcome | Agent viết báo cáo: hoàn thành đủ phần + chất lượng tổng |

**Non-determinism metrics**:

```python
from underthesea.agent.eval import EvalHarness

harness = EvalHarness(
    suite="research",
    trials_per_task=5,   # Chạy mỗi task 5 lần
)
results = harness.run(agent)

print(results.pass_at_k(k=1))   # Xác suất đúng ít nhất 1/1 lần
print(results.pass_at_k(k=5))   # Xác suất đúng ít nhất 1/5 lần → gần 100%
print(results.pass_pow_k(k=5))  # Xác suất đúng cả 5/5 lần → yêu cầu ổn định
```

**Ưu tiên**: **CAO** - Không có eval thì không biết agent có đang tốt lên hay xấu đi. Đây là "Swiss Cheese Model" - nhiều lớp đánh giá chồng lên nhau để bắt lỗi.

---

### Hướng 6: Subagent Delegation & Task Decomposition

**Vấn đề cốt lõi**: Một agent đơn lẻ xử lý task phức tạp sẽ nhanh chóng tràn context. Cần cơ chế spawn subagents cho subtasks, mỗi subagent có context riêng.

**Bài học từ LangChain Deep Agents SDK**: Agent harness cần khả năng "spawn subagents để phân chia công việc và bảo toàn context".

**Kiến trúc đề xuất**:

```
underthesea/agent/
├── harness/
│   ├── delegation.py          # Subagent spawning & management
│   ├── task_graph.py          # DAG (Directed Acyclic Graph) của subtasks
│   └── aggregator.py          # Tổng hợp kết quả từ subagents
```

**Delegation pattern**:

```python
from underthesea.agent.harness import DelegatingAgent

agent = DelegatingAgent(
    name="research_lead",
    instruction="Bạn là trưởng nhóm nghiên cứu. Phân chia công việc cho subagents.",
    tools=[web_search_tool, read_file_tool, write_file_tool],
    subagent_config={
        "model": "claude-sonnet-4-6",
        "max_iterations": 10,
    }
)

# Agent tự động spawn subagents khi cần
result = agent("Nghiên cứu 5 công ty fintech lớn nhất Việt Nam. "
               "Mỗi công ty cần: lịch sử, sản phẩm, doanh thu, đối thủ.")

# Bên trong, agent sẽ:
# 1. Tạo 5 subagents, mỗi agent nghiên cứu 1 công ty
# 2. Mỗi subagent có context riêng, không bị tràn
# 3. Tổng hợp kết quả từ 5 subagents thành báo cáo cuối
```

**Task graph** (cho xử lý song song):

```
            ┌→ Subagent A (Công ty 1) ─┐
            ├→ Subagent B (Công ty 2) ─┤
Lead Agent ─┼→ Subagent C (Công ty 3) ─┼→ Aggregator → Báo cáo cuối
            ├→ Subagent D (Công ty 4) ─┤
            └→ Subagent E (Công ty 5) ─┘
```

**Ưu tiên**: **TRUNG BÌNH** - Cần thiết cho tasks phức tạp, nhưng nên xây trên nền Hướng 1 + 2 trước.

---

## 3. Lộ trình phát triển

### Phase 1: Nền tảng (3-4 tuần)

**Mục tiêu**: Multi-provider + Multi-session cơ bản

```
underthesea/agent/
├── providers/
│   ├── base.py                # Provider interface chuẩn
│   ├── openai_provider.py     # Refactor từ llm.py
│   └── anthropic_provider.py  # Claude API support
├── harness/
│   ├── session.py             # Session lifecycle
│   ├── context.py             # Context reset + handoff
│   └── progress.py            # Progress tracking (JSON)
```

**Sản phẩm bàn giao**:
- Agent có thể dùng Claude hoặc OpenAI
- Agent có thể lưu/khôi phục state giữa các session
- Progress tracking qua JSON file
- Structured handoff khi context cần reset

### Phase 2: Kiến trúc đa Agent (3-4 tuần)

**Mục tiêu**: Planner/Executor/Evaluator pattern hoạt động

```
underthesea/agent/
├── harness/
│   ├── planner.py
│   ├── executor.py
│   ├── evaluator.py
│   ├── orchestrator.py
│   └── contracts.py
```

**Sản phẩm bàn giao**:
- Orchestrator điều phối 3 agents
- Sprint contracts với tiêu chí đánh giá cụ thể
- Vòng lặp cải thiện (Executor → Evaluator → feedback → Executor)
- Giao tiếp giữa agents qua structured files

### Phase 3: Tool System + NLP Tools (2-3 tuần)

**Mục tiêu**: Nâng cấp tool system, tạo NLP tools

```
underthesea/agent/
├── tools/
│   ├── registry.py
│   ├── middleware.py
│   └── builtin/
│       └── nlp.py             # word_tokenize, ner, sentiment... dưới dạng tools
```

**Sản phẩm bàn giao**:
- Tool middleware (logging, retry, timeout)
- Dynamic tool registry
- NLP pipelines exposed dưới dạng agent tools
- Điểm khác biệt: agent có thể phân tích tiếng Việt natively

### Phase 4: Agent Eval + Subagent Delegation (4-6 tuần)

**Mục tiêu**: Đo lường chất lượng agent + xử lý task phức tạp

```
underthesea/agent/
├── eval/
│   ├── harness.py
│   ├── task.py, trial.py
│   ├── graders/
│   └── suites/
├── harness/
│   ├── delegation.py
│   └── task_graph.py
```

**Sản phẩm bàn giao**:
- Eval harness chạy được eval suite cho agents
- 20-50 eval tasks cho conversational + research + tool-use agents
- Subagent spawning cho task decomposition
- pass@k / pass^k metrics

---

## 4. Ánh xạ với Industry Standards

### Anthropic "Effective Harnesses for Long-Running Agents"

| Nguyên tắc | Ánh xạ sang Underthesea |
|------------|------------------------|
| Initializer Agent + Coding Agent | → Session manager với init session + continue sessions (Hướng 1) |
| Feature list dạng JSON với pass/fail | → progress.json theo dõi subtasks (Hướng 1) |
| Một feature mỗi session | → Mỗi session chỉ xử lý subtask tiếp theo chưa hoàn thành |
| Git commits + progress file | → Handoff file + progress file + artifacts directory |
| Baseline test trước khi bắt đầu | → Evaluator kiểm tra state trước khi Executor bắt đầu (Hướng 2) |

### Anthropic "Harness Design for Long-Running Apps"

| Nguyên tắc | Ánh xạ sang Underthesea |
|------------|------------------------|
| Context reset > compaction | → `context.py` implement context reset với structured handoff (Hướng 1) |
| Tách biệt Generator/Evaluator | → Executor agent + Evaluator agent riêng biệt (Hướng 2) |
| Sprint contracts | → `contracts.py` định nghĩa tiêu chí "hoàn thành" (Hướng 2) |
| 5-15 vòng lặp cải thiện | → Orchestrator vòng lặp Executor → Evaluator (Hướng 2) |
| Tìm giải pháp đơn giản nhất | → Bắt đầu Phase 1 trước, tăng complexity dần |
| Kiểm tra giả định liên tục | → Agent eval harness đo regression (Hướng 5) |

### Anthropic "Demystifying Evals for AI Agents"

| Nguyên tắc | Ánh xạ sang Underthesea |
|------------|------------------------|
| Task / Trial / Grader / Transcript | → `eval/task.py`, `eval/trial.py`, `eval/graders/` (Hướng 5) |
| Code-based + Model-based + Human graders | → 3 loại grader trong `eval/graders/` |
| pass@k / pass^k | → `eval/metrics/pass_at_k.py` |
| Bắt đầu 20-50 tasks từ lỗi thực tế | → Tạo eval tasks từ các trường hợp agent thất bại |
| Đánh giá outcome, không đánh giá process | → Graders kiểm tra kết quả cuối, không kiểm tra tool call sequence |
| Swiss Cheese Model | → Eval harness + production monitoring + transcript review |

### LangChain Deep Agents SDK

| Khái niệm | Ánh xạ sang Underthesea |
|-----------|------------------------|
| Lập kế hoạch & quản lý task | → Planner agent + progress tracking (Hướng 1 + 2) |
| Ủy quyền task (subagents) | → DelegatingAgent + task graph (Hướng 6) |
| Tích hợp hệ thống file | → Artifacts directory + structured files (Hướng 1 + 2) |
| Tối ưu token | → Context reset + handoff (Hướng 1) |

---

## 5. Kết luận

### Thứ tự ưu tiên:

1. **Hướng 1 (Multi-Session) + Hướng 3 (Multi-Provider)** - Nền tảng bắt buộc. Không có multi-session thì agent không làm được task phức tạp. Không có multi-provider thì bị khóa vào OpenAI.

2. **Hướng 2 (Planner/Executor/Evaluator)** - Đây là nơi tạo ra sự khác biệt chất lượng lớn nhất (20x theo Anthropic). Xây trên nền Phase 1.

3. **Hướng 5 (Agent Eval)** - Cần có để đo lường và cải thiện. Không có eval thì "bay mù" - không biết thay đổi nào giúp ích, thay đổi nào gây hại.

4. **Hướng 4 (Tool System) + Hướng 6 (Subagent)** - Nâng cấp khả năng agent. NLP tools tạo điểm khác biệt cho underthesea so với các agent framework khác.

### Nguyên tắc xuyên suốt:

- **Tìm giải pháp đơn giản nhất** (Anthropic): Mỗi component harness mã hóa giả định về giới hạn của model. Khi model cải thiện, bỏ bớt scaffolding không còn cần.
- **Tách biệt tạo sinh và đánh giá**: Không bao giờ để agent tự đánh giá output của mình.
- **Context reset thay vì compaction**: Luôn ưu tiên tạo agent mới với handoff rõ ràng.
- **Đánh giá kết quả, không đánh giá quy trình**: Cho phép agent sáng tạo trong cách giải quyết, chỉ cần kết quả đúng.
- **Chi phí tỷ lệ với chất lượng**: Sẵn sàng đầu tư compute cost cho kiến trúc đa agent khi cần output chất lượng cao.
