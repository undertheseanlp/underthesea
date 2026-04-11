# Agent Simulation — Design Document

> Last updated: 2026-04-11

## 1. Problem

Khi xây dựng agent (ví dụ CS Chatbot hỗ trợ cài đặt máy tính), cần cách test tự động:

- Agent có trả lời đúng không?
- Agent có gọi đúng tool không?
- Multi-turn conversation có dẫn đến kết quả mong muốn không?

Test thủ công không scale. Cần **simulated users** — LLM đóng vai user với kịch bản cụ thể.

## 2. Research Summary

### Các framework hiện có

| Framework | Approach | Key Feature |
|-----------|----------|-------------|
| DeepEval | `ConversationSimulator` | Scenario + expected_outcome + persona |
| AWS Strands | `ActorSimulator` | Auto-generate actor profile, track goal per turn |
| AutoGen | `UserProxyAgent` | Configurable human_input_mode |
| LangSmith | Thread-level eval | LLM-as-judge + human correction loop |
| Ragas | Agent metrics | ToolCallAccuracy, AgentGoalAccuracy |
| Braintrust | AutoEvals | Deterministic + LLM-as-judge combo |

### Consensus patterns

1. **Scenario object**: persona + task + expected_outcome + max_turns
2. **User simulator**: LLM đóng vai user, maintain character, track goal
3. **Conversation runner**: orchestrate user ↔ agent turns
4. **Evaluator**: deterministic checks + LLM-as-judge

### Caveat

LLM-simulated users tend to be more polite and ask more questions than real humans (arxiv:2601.17087). Use for regression testing, not as replacement for human evaluation.

## 3. Architecture

```
underthesea.agent.simulation
├── scenario.py         # Scenario dataclass + YAML loader
├── user_simulator.py   # LLM-powered simulated user
├── runner.py           # Conversation orchestrator
├── evaluator.py        # LLM-as-judge + deterministic checks
└── __init__.py
```

### Flow

```
                    ┌──────────────┐
                    │   Scenario   │
                    │  (YAML/code) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    Runner    │
                    └──┬───────┬───┘
                       │       │
              ┌────────▼──┐ ┌──▼────────┐
              │   User    │ │   Agent   │
              │ Simulator │ │ (under    │
              │  (LLM)    │ │  test)    │
              └────────┬──┘ └──┬────────┘
                       │       │
                    ┌──▼───────▼───┐
                    │  Transcript  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Evaluator   │
                    │ (LLM-judge)  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Report     │
                    └──────────────┘
```

## 4. API Design

### 4.1 Scenario

```python
from underthesea.agent.simulation import Scenario

# Define inline
scenario = Scenario(
    name="install-error-python-version",
    persona="Người dùng Windows, không rành terminal, lần đầu dùng Python",
    task="Cài underthesea nhưng gặp lỗi: 'Python 3.8 is not supported, requires >= 3.10'",
    expected_outcome="Agent hướng dẫn cài Python >= 3.10 và cài lại underthesea thành công",
    max_turns=10,
)

# Or load from YAML
scenarios = Scenario.load("scenarios/cs_chatbot.yaml")
```

**YAML format:**

```yaml
scenarios:
  - name: install-error-python-version
    persona: Người dùng Windows, không rành terminal, lần đầu dùng Python
    task: |
      Cài underthesea nhưng gặp lỗi:
      'ERROR: Package underthesea requires Python >=3.10, but you have Python 3.8.10'
    expected_outcome: Agent hướng dẫn cài Python >= 3.10 và cài lại underthesea thành công
    max_turns: 10

  - name: install-error-rust-compiler
    persona: Data scientist dùng macOS, quen pip nhưng chưa cài Rust
    task: |
      pip install underthesea bị lỗi:
      'error: can't find Rust compiler. Install Rust from https://rustup.rs'
    expected_outcome: Agent hướng dẫn cài Rust toolchain và cài lại underthesea
    max_turns: 8

  - name: word-tokenize-wrong-output
    persona: Developer Python có kinh nghiệm, đang integrate underthesea vào project
    task: |
      word_tokenize("Hà Nội") trả về ['Hà', 'Nội'] thay vì ['Hà Nội'].
      User muốn biết tại sao và cách fix.
    expected_outcome: Agent giải thích cần load model và hướng dẫn cách fix
    max_turns: 6
```

### 4.2 UserSimulator

```python
from underthesea.agent.simulation import UserSimulator

simulator = UserSimulator(
    provider=LLM(),  # or any BaseProvider
    model="gpt-4.1-mini",
)

# Generate next user message given scenario + conversation history
message = simulator.next_message(
    scenario=scenario,
    history=[
        {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì?"},
    ],
)
# Returns: UserMessage(content="Tôi cài underthesea bị lỗi...", goal_achieved=False)
```

**System prompt cho UserSimulator:**

```
You are simulating a user with the following profile:
- Persona: {persona}
- Problem: {task}
- Goal: {expected_outcome}

Rules:
1. Stay in character. Match the persona's expertise level and communication style.
2. Start by describing your problem naturally (don't dump the full error at once).
3. Follow the agent's instructions step by step. Report what happens.
4. If the agent's suggestion doesn't work, say so and describe the new error.
5. If your problem is resolved, say thank you and confirm it works.
6. Respond in Vietnamese.

Output JSON:
{"message": "your message", "goal_achieved": true/false}
```

### 4.3 ConversationRunner

```python
from underthesea.agent.simulation import run_simulation, Scenario

# Single scenario
result = run_simulation(
    agent=my_agent,
    scenario=scenario,
    provider=LLM(),  # for user simulator
)

print(result.transcript)     # List of messages
print(result.turns)          # Number of turns
print(result.goal_achieved)  # True/False (from user simulator)
print(result.duration_ms)    # Total time

# Batch run
results = run_simulation(
    agent=my_agent,
    scenarios=Scenario.load("scenarios/cs_chatbot.yaml"),
    provider=LLM(),
)

for r in results:
    print(f"{r.scenario.name}: {'PASS' if r.goal_achieved else 'FAIL'} ({r.turns} turns)")
```

**Runner logic (pseudocode):**

```python
def run_simulation(agent, scenario, provider):
    simulator = UserSimulator(provider)
    transcript = []

    # User starts the conversation
    user_msg = simulator.first_message(scenario)
    transcript.append({"role": "user", "content": user_msg.content})

    for turn in range(scenario.max_turns):
        # Agent responds
        agent_response = agent(user_msg.content)
        transcript.append({"role": "assistant", "content": agent_response})

        # User simulator responds
        user_msg = simulator.next_message(scenario, transcript)
        if user_msg.goal_achieved:
            transcript.append({"role": "user", "content": user_msg.content})
            break
        transcript.append({"role": "user", "content": user_msg.content})

    return SimulationResult(
        scenario=scenario,
        transcript=transcript,
        turns=len(transcript) // 2,
        goal_achieved=user_msg.goal_achieved,
    )
```

### 4.4 Evaluator

```python
from underthesea.agent.simulation import evaluate

# Evaluate a single result
eval_result = evaluate(
    result=result,
    provider=LLM(),  # LLM-as-judge
    criteria=[
        "goal_achieved",      # Did the agent solve the user's problem?
        "response_quality",   # Were responses clear and helpful?
        "turn_efficiency",    # Was the problem solved in few turns?
        "topic_adherence",    # Did the agent stay on topic?
    ],
)

print(eval_result.scores)
# {
#   "goal_achieved": 1.0,
#   "response_quality": 0.9,
#   "turn_efficiency": 0.8,
#   "topic_adherence": 1.0,
#   "overall": 0.925,
# }
print(eval_result.reasoning)  # LLM judge's explanation
```

**LLM-as-judge prompt:**

```
You are evaluating an AI agent's performance in a customer support conversation.

## Scenario
- User persona: {persona}
- User's problem: {task}
- Expected outcome: {expected_outcome}

## Transcript
{transcript}

## Evaluation Criteria
Score each criterion from 0.0 to 1.0:

1. goal_achieved: Was the user's problem fully resolved?
2. response_quality: Were the agent's responses clear, accurate, and helpful?
3. turn_efficiency: Was the problem solved efficiently (fewer turns = better)?
4. topic_adherence: Did the agent stay focused on the user's problem?

Output JSON:
{
  "scores": {"goal_achieved": 0.0-1.0, ...},
  "overall": 0.0-1.0,
  "reasoning": "explanation"
}
```

## 5. Example: CS Chatbot Testing

```python
from underthesea.agent import Agent, Tool, LLM
from underthesea.agent.simulation import Scenario, run_simulation, evaluate

# 1. Define the agent
def search_docs(query: str) -> str:
    """Search documentation for troubleshooting."""
    # ... search logic
    return "Found: Install Python >= 3.10 from python.org"

cs_agent = Agent(
    name="CS Chatbot",
    tools=[Tool(search_docs)],
    instruction="""Bạn là nhân viên hỗ trợ kỹ thuật cho thư viện underthesea.
    Hướng dẫn người dùng cài đặt và sử dụng underthesea.
    Trả lời bằng tiếng Việt, rõ ràng, từng bước.""",
)

# 2. Define scenarios
scenarios = [
    Scenario(
        name="python-version-error",
        persona="Sinh viên IT năm 2, dùng Windows, biết cơ bản về Python",
        task="pip install underthesea bị lỗi 'requires Python >= 3.10' nhưng đang dùng Python 3.8",
        expected_outcome="Cài được Python 3.10+ và cài underthesea thành công",
        max_turns=8,
    ),
    Scenario(
        name="rust-compiler-missing",
        persona="Data scientist, dùng macOS, chưa cài Rust",
        task="pip install underthesea lỗi 'can't find Rust compiler'",
        expected_outcome="Cài Rust toolchain và cài underthesea thành công",
        max_turns=8,
    ),
    Scenario(
        name="import-error-torch",
        persona="Researcher NLP, dùng Ubuntu, cần dùng deep learning features",
        task="from underthesea import dependency_parse → ImportError: torch not found",
        expected_outcome="Agent hướng dẫn pip install underthesea[deep]",
        max_turns=6,
    ),
]

# 3. Run simulations
provider = LLM()
results = run_simulation(agent=cs_agent, scenarios=scenarios, provider=provider)

# 4. Evaluate
for result in results:
    eval_result = evaluate(result=result, provider=provider)
    status = "PASS" if eval_result.scores["goal_achieved"] >= 0.8 else "FAIL"
    print(f"[{status}] {result.scenario.name}")
    print(f"  Turns: {result.turns}")
    print(f"  Scores: {eval_result.scores}")
    print(f"  Reasoning: {eval_result.reasoning}")
    print()
```

**Expected output:**

```
[PASS] python-version-error
  Turns: 4
  Scores: {'goal_achieved': 1.0, 'response_quality': 0.9, 'turn_efficiency': 0.9, 'topic_adherence': 1.0, 'overall': 0.95}
  Reasoning: Agent correctly identified the Python version issue and provided step-by-step instructions...

[PASS] rust-compiler-missing
  Turns: 3
  Scores: {'goal_achieved': 1.0, 'response_quality': 0.85, 'turn_efficiency': 1.0, 'topic_adherence': 1.0, 'overall': 0.96}
  Reasoning: Agent quickly identified the missing Rust compiler and provided the correct rustup command...

[FAIL] import-error-torch
  Turns: 6
  Scores: {'goal_achieved': 0.5, 'response_quality': 0.6, 'turn_efficiency': 0.4, 'topic_adherence': 0.8, 'overall': 0.575}
  Reasoning: Agent suggested pip install torch separately instead of the simpler underthesea[deep]...
```

## 6. Integration with Trace

Simulation results integrate with the existing trace module:

```python
from underthesea.agent.simulation import run_simulation
from underthesea.agent.trace import trace, LocalTracer

tracer = LocalTracer()

@trace(tracer)
def test_scenario(agent, scenario, provider):
    result = run_simulation(agent=agent, scenario=scenario, provider=provider)
    eval_result = evaluate(result=result, provider=provider)
    return eval_result

# Each simulation run is fully traced:
# trace "test_scenario"
#   └─ span "run_simulation"
#       ├─ generation "user_simulator.first_message"
#       ├─ generation "agent.llm.chat #1"
#       ├─ span "agent.tool.search_docs"
#       ├─ generation "agent.llm.chat #2"
#       ├─ generation "user_simulator.next_message"
#       └─ ...
#   └─ span "evaluate"
#       └─ generation "llm_judge"
```

## 7. Module Dependencies

```
underthesea.agent.simulation
├── depends on: underthesea.agent.Agent (agent under test)
├── depends on: underthesea.agent.LLM / BaseProvider (for user simulator + judge)
├── depends on: underthesea.agent.trace (optional, for observability)
├── optional: PyYAML (for YAML scenario loading — already a core dep)
└── no new external dependencies
```

## 8. Files to Create

```
underthesea/agent/simulation/
├── __init__.py          # Public API: Scenario, UserSimulator, run_simulation, evaluate
├── scenario.py          # Scenario dataclass + YAML loader
├── user_simulator.py    # LLM-powered user simulator
├── runner.py            # ConversationRunner + SimulationResult
└── evaluator.py         # LLM-as-judge evaluator + EvalResult

tests/agent/
└── test_simulation.py   # Tests with mocked LLM responses
```

## 9. References

- [DeepEval ConversationSimulator](https://deepeval.com/docs/conversation-simulator)
- [AWS Strands ActorSimulator](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/simulators/user_simulation/)
- [LangSmith Multi-turn Evals](https://blog.langchain.com/insights-agent-multiturn-evals-langsmith/)
- [Ragas Agent Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/)
- [OpenAI Agent Evals](https://developers.openai.com/api/docs/guides/agent-evals)
- [Lost in Simulation (arxiv:2601.17087)](https://arxiv.org/abs/2601.17087)
