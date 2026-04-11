# Agent Simulation — Research Notes

> Last updated: 2026-04-11

## 1. Frameworks Surveyed

### DeepEval ConversationSimulator

**Approach:** Define `ConversationalGolden` with scenario, expected_outcome, user_description. Simulator drives multi-turn exchanges, stopping when expected outcome is reached.

**Key insight:** Richer persona descriptions produce more realistic and challenging conversations.

**API:**
```python
golden = ConversationalGolden(
    scenario="User wants to cancel subscription",
    expected_outcome="Agent processes cancellation and confirms",
    user_description="Frustrated customer who has been waiting 30 minutes",
)
simulator = ConversationSimulator()
simulator.simulate(agent=my_agent, goldens=[golden])
```

**Metrics:** KnowledgeRetention, ConversationCompleteness, TurnRelevancy.

**Source:** https://deepeval.com/docs/conversation-simulator

---

### AWS Strands Evals ActorSimulator

**Approach:** Auto-generates a full actor profile (experience level, communication style, budget constraints) from a test case. Tracks goal completion turn-by-turn.

**Key insight:** Actor follows up naturally on partial answers and responds to clarifying questions in character. Transcripts feed into the same eval pipeline used for single-turn tests.

**API:**
```python
test_case = TestCase(
    input_text="I need help setting up my printer",
    expected_output="Printer is set up and working",
    actor_description="Non-technical user, elderly, patient",
)
simulator = ActorSimulator(model="us.anthropic.claude-sonnet-4-20250514")
transcript = simulator.run(agent=my_agent, test_case=test_case)
```

**Source:** https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/simulators/user_simulation/

---

### AutoGen UserProxyAgent

**Approach:** Configurable `human_input_mode` ("ALWAYS"/"TERMINATE"/"NEVER") lets you script test behavior or plug in another LLM as the user.

**Key insight:** Enables automated multi-agent testing loops. Can be used for both human-in-the-loop and fully automated testing.

**Source:** https://microsoft.github.io/autogen/

---

### LangSmith Multi-turn Evals

**Approach:** Thread-level evaluation. Define scoring criteria (rubric), collect human corrections on judge scores, which are auto-inserted as few-shot examples.

**Key insight:** Human corrections on judge scores create a feedback loop that aligns the judge with human preferences over time.

**API:**
```python
evaluator = LLMAsJudge(
    criteria="helpfulness",
    rubric="Score 1-5 based on...",
    few_shot_examples=human_corrections,
)
```

**Source:** https://blog.langchain.com/insights-agent-multiturn-evals-langsmith/

---

### Ragas Agent Metrics

**Approach:** Structured metric taxonomy specifically for agent evaluation.

**Metrics:**
| Metric | Description |
|--------|-------------|
| `ToolCallAccuracy` | Right tools, right args, right order |
| `ToolCallF1` | Set-based tool comparison (precision/recall) |
| `AgentGoalAccuracy` | Did the agent achieve the user's objective (binary) |
| `TopicAdherence` | Did the agent stay on topic |

**Source:** https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/

---

### Braintrust AutoEvals

**Approach:** Combines deterministic checks (tool selection, argument format) with LLM-as-judge for open-ended quality.

**Key insight:** Recommends four test case categories: happy-path, edge cases, adversarial, and off-topic.

**Source:** https://www.braintrust.dev/articles/ai-agent-evaluation-framework

---

### OpenAI Agent Evals

**Approach:** Trace-based grading. Capture the full trace (model calls, tool calls, guardrails, handoffs), then score with structured criteria.

**Key insight:** Evaluate at the trace level, not just the final response. A correct final answer with wrong intermediate steps should score lower.

**Source:** https://developers.openai.com/api/docs/guides/agent-evals

## 2. Common Patterns

### Pattern 1: Scenario Object

Every framework uses some form of scenario definition:

```
Scenario = {
    persona: who the user is
    task: what problem they have
    expected_outcome: what success looks like
    max_turns: conversation length limit
}
```

### Pattern 2: User Simulator as LLM

A second LLM plays the user role with a system prompt that:
1. Defines the persona and problem
2. Sets rules for behavior (stay in character, follow instructions, report results)
3. Tracks goal completion (output a signal when goal is achieved)

### Pattern 3: Conversation Loop

```
user_simulator.first_message()
loop:
    agent.respond(user_message)
    user_simulator.next_message(transcript)
    if goal_achieved or max_turns: break
```

### Pattern 4: Dual Evaluation

1. **Deterministic:** Did the right tool get called? Were args correct?
2. **LLM-as-judge:** Was the response helpful? Was the goal achieved?

### Pattern 5: Batch Execution

Run multiple scenarios in sequence or parallel, aggregate results into a report.

## 3. Key Research Paper

**"Lost in Simulation: LLM-Simulated Users are Unreliable Proxies"** (arxiv:2601.17087)

Findings:
- LLM-simulated users are more polite than real users
- They ask more clarifying questions
- Success rates vary up to 9 percentage points across different simulator LLMs
- Simulated users tend to accept agent responses more readily

**Implication:** Use simulated users for regression testing and coverage, not as a replacement for human evaluation. Be aware of the "politeness bias."

## 4. Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scenario format | Dataclass + YAML | YAML for non-technical scenario authors, dataclass for programmatic use |
| User simulator output | JSON with `message` + `goal_achieved` | Clean separation of content and control signal |
| Evaluation approach | Deterministic + LLM-as-judge | Catches both hard failures (wrong tool) and soft failures (unhelpful response) |
| Integration | Works with existing Agent + Trace | No new dependencies, reuses LLM providers |
| Language | Vietnamese by default | Target use case is Vietnamese NLP toolkit |
