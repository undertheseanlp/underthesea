# Agent Simulation — Roadmap

> Last updated: 2026-04-11

## Phase 1: Core Module (MVP)

**Goal:** Run a single scenario against an agent, get pass/fail result.

### Tasks

- [ ] `Scenario` dataclass with YAML loader
- [ ] `UserSimulator` with configurable LLM provider
- [ ] `ConversationRunner` — orchestrate user ↔ agent turns
- [ ] `SimulationResult` — transcript, turns, goal_achieved, duration
- [ ] Basic `evaluate()` — LLM-as-judge with goal_achieved score
- [ ] Unit tests with mocked LLM responses
- [ ] E2E test with live Azure OpenAI

### API surface

```python
from underthesea.agent.simulation import Scenario, run_simulation, evaluate
```

### Success criteria

- Run 3 CS chatbot scenarios
- Each produces a transcript and pass/fail evaluation
- Results integrate with existing trace module

---

## Phase 2: Batch & Reporting

**Goal:** Run multiple scenarios, generate an evaluation report.

### Tasks

- [ ] Batch `run_simulation(scenarios=[...])` — run multiple scenarios
- [ ] `SimulationReport` — aggregate results, pass/fail counts, average scores
- [ ] Console report output (table format)
- [ ] JSON report export
- [ ] Parallel scenario execution (optional)

### API surface

```python
results = run_simulation(agent=agent, scenarios=scenarios, provider=llm)
report = SimulationReport(results)
report.print()
report.save("report.json")
```

---

## Phase 3: Advanced Evaluation

**Goal:** Richer evaluation metrics beyond pass/fail.

### Tasks

- [ ] Multi-criteria evaluation: goal_achieved, response_quality, turn_efficiency, topic_adherence
- [ ] Deterministic checks: tool call accuracy (right tool, right args)
- [ ] Per-turn scoring (not just overall)
- [ ] Custom evaluation criteria (user-defined rubrics)
- [ ] Evaluation history tracking (regression detection)

### Metrics (inspired by Ragas)

| Metric | Type | Description |
|--------|------|-------------|
| `goal_achieved` | LLM-judge | Binary: was the user's problem resolved? |
| `response_quality` | LLM-judge | 0-1: clarity, accuracy, helpfulness |
| `turn_efficiency` | Deterministic | Fewer turns = better score |
| `topic_adherence` | LLM-judge | Did the agent stay on topic? |
| `tool_accuracy` | Deterministic | Correct tool + correct args? |

---

## Phase 4: CI Integration

**Goal:** Run simulation tests in CI as regression checks.

### Tasks

- [ ] CLI command: `underthesea simulate scenarios.yaml --agent cs_chatbot`
- [ ] GitHub Actions workflow for simulation tests
- [ ] Threshold-based pass/fail (e.g., overall >= 0.8)
- [ ] Diff report: compare current vs previous run
- [ ] Langfuse integration for simulation traces

---

## Phase 5: Advanced Scenarios

**Goal:** Support more complex testing patterns.

### Tasks

- [ ] Multi-agent scenarios (agent hands off to another agent)
- [ ] Adversarial user simulator (tries to break the agent)
- [ ] Long conversation scenarios (20+ turns)
- [ ] Scenario templates with variable substitution
- [ ] Scenario generation from production logs

---

## Non-Goals (for now)

- Real-time user simulation (interactive mode)
- Visual UI for scenario management
- A/B testing between agent versions (use Langfuse for this)
- Production traffic replay
