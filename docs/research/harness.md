# Agent Harness & Evaluation Harness Research

## Sources

1. [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) - Anthropic Engineering
2. [Harness Design for Long-Running Apps](https://www.anthropic.com/engineering/harness-design-long-running-apps) - Anthropic Engineering (Prithvi Rajasekaran)
3. [Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) - Anthropic Engineering
4. [Agent Harnesses (LangChain Deep Agents SDK)](https://docs.langchain.com/oss/python/concepts/products#agent-harnesses-like-the-deep-agents-sdk) - LangChain Docs

---

## 1. What Is an Agent Harness?

An **agent harness** (also called an agent scaffold) is the infrastructure layer that enables a model to act as an agent. It handles input processing, tool orchestration, memory management, and session continuity. The harness is distinct from the model itself -- it is the system around the model that turns it into a functioning agent.

An **evaluation harness** is the infrastructure that runs evals end-to-end: executing tasks concurrently, managing isolated environments, and aggregating results.

### Key Distinction

| Concept | Purpose |
|---------|---------|
| **Agent Harness / Scaffold** | Enables a model to act as an agent (tools, memory, orchestration) |
| **Evaluation Harness** | Runs evals end-to-end (task execution, isolation, result aggregation) |

---

## 2. The Core Problem: Long-Running Agents

Complex projects cannot complete within a single context window. This creates two primary failure modes:

1. **Over-ambition**: The agent attempts to implement everything at once, exhausts its context mid-task, and leaves incomplete, undocumented work for the next session.
2. **Premature completion**: Later agent instances see existing progress and incorrectly declare the project finished.

Additional failure modes emerge over extended runs:

- **Context degradation**: As context windows fill, models lose coherence. Some exhibit "context anxiety," prematurely concluding work as they approach perceived limits.
- **Self-evaluation bias**: When evaluating their own work, models tend to offer praise regardless of actual quality. This intensifies for subjective domains like design.

---

## 3. Harness Architecture Patterns

### 3.1 Two-Agent Pattern (Initializer + Coding Agent)

From Source 1 -- the simplest effective pattern for multi-session continuity.

**Initializer Agent** (first session only):
- Creates `init.sh` script for dev server startup
- Creates `claude-progress.txt` for activity logging
- Makes initial git commit documenting baseline files

**Coding Agent** (every subsequent session):
1. Read progress documentation and git history
2. Run baseline end-to-end tests
3. Select one incomplete feature to implement
4. Commit changes with descriptive messages
5. Update progress tracking

**Key design decisions:**
- Feature specs stored as JSON (less prone to model corruption than Markdown)
- 200+ feature items with step-by-step descriptions and `"passes": false` status markers
- Agents prohibited from removing/editing feature definitions
- Single-feature-per-session constraint prevents context exhaustion

```json
{
  "category": "functional",
  "description": "[Feature name]",
  "steps": ["[action 1]", "[action 2]"],
  "passes": false
}
```

### 3.2 Three-Agent Pattern (Planner + Generator + Evaluator)

From Source 2 -- a more sophisticated architecture inspired by GANs.

**Planner Agent**: Transforms brief prompts (1-4 sentences) into comprehensive product specifications. Emphasizes scope ambition and high-level technical direction while avoiding premature implementation details.

**Generator Agent**: Works through features systematically. Implements with chosen stack (e.g., React, Vite, FastAPI, SQLite/PostgreSQL). Performs self-evaluation before handing to QA. Maintains version control.

**Evaluator Agent**: Conducts thorough testing via Playwright. Exercises UI features, API endpoints, and database state. Negotiates "sprint contracts" upfront -- agreements defining what "done" means before code writing begins. Each criterion has hard thresholds; failures trigger detailed feedback for iteration.

**Communication protocol:** Agents exchange information through structured files, not direct conversation. This ensures faithful adherence to specifications without over-constraining implementation.

### 3.3 Generator-Evaluator Loop

From Source 2 -- specifically for quality-sensitive domains (e.g., frontend design).

- Generator creates HTML/CSS/JS from prompts
- Evaluator navigates the live page via Playwright MCP (not static screenshots)
- 5-15 iteration cycles per generation
- Strategic decision points: refine current direction or pivot aesthetically
- 4-hour runs produced noticeably better outputs

**Critical insight:** Separating generation from evaluation is "far more tractable than making a generator critical of its own work."

### 3.4 Batteries-Included Harness (LangChain Deep Agents SDK)

From Source 4 -- the most opinionated tier in the stack.

Integrated capabilities:
- **Planning and task management**: Track multiple objectives with built-in structures
- **Task delegation**: Spawn subagents to compartmentalize work and preserve context
- **File system integration**: Pluggable storage backends
- **Token optimization**: Conversation summarization and automatic eviction of large tool results

The Deep Agents SDK builds on LangGraph, adding planning, filesystem for context management, subagent spawning, and more.

---

## 4. Critical Design Principles

### 4.1 Context Resets Over Compaction

Context resets (completely clearing conversation and starting fresh agents with structured handoffs) proved more effective than compaction (summarizing earlier exchanges in place). This is counterintuitive but consistently observed.

### 4.2 Separate Generation from Evaluation

Self-evaluation bias is a persistent failure mode. Isolating generation from evaluation enables:
- Independent tuning of each component
- More effective feedback loops
- More honest quality assessment

### 4.3 Criteria-Driven Prompting

Translate subjective requirements into specific, weightable criteria. For design quality:
- **Design Quality**: Coherence across colors, typography, layout, imagery
- **Originality**: Custom decisions vs. template defaults and AI patterns
- **Craft**: Technical execution (typography hierarchy, spacing, color harmony)
- **Functionality**: User comprehension and task completion

### 4.4 Find the Simplest Solution Possible

Only increase harness complexity when necessary. Capabilities improve faster than architectural needs. Every harness component encodes assumptions about model limitations -- these deserve regular examination and potential removal.

### 4.5 Continuous Assumption Testing

As models improve, harness components may become unnecessary overhead. The Opus 4.6 example: removing sprint-level decomposition while retaining planner + evaluator still produced strong results.

### 4.6 Incremental Progress Over Monolithic Execution

Single-feature-per-session prevents context exhaustion. Git commits with clear messages enable rollback. Progress file summaries enable handoff.

---

## 5. Cost/Quality Tradeoffs

From Source 2, testing a retro video game maker:

| Approach | Time | Cost | Quality |
|----------|------|------|---------|
| Solo agent | 20 min | $9 | Functional but broken core features |
| Full harness (3-agent) | 6 hours | $200 | Comprehensive, polished, working |

The 20x cost increase yielded dramatically superior output. This frames the harness decision as a quality investment, not just engineering overhead.

---

## 6. Evaluation Harness Design

### 6.1 Core Terminology

| Term | Definition |
|------|-----------|
| **Task** | Single test with defined inputs and success criteria |
| **Trial** | One attempt at a task; multiple trials address model variability |
| **Grader** | Logic scoring agent performance; tasks contain multiple graders |
| **Transcript** | Complete trial record (outputs, tool calls, reasoning, intermediate results) |
| **Outcome** | Final environmental state (distinct from transcript claims) |
| **Evaluation Suite** | Cohesive task collection measuring specific capabilities |

### 6.2 Three Grader Types

**Code-Based Graders**: Fast, cheap, objective, reproducible, easy to debug. But brittle to valid variations that don't match expected patterns exactly.

**Model-Based Graders**: Flexible, scalable, handle open-ended tasks. But non-deterministic, higher cost, require human calibration.

**Human Graders**: Gold-standard quality. But expensive, slow, difficult to scale.

### 6.3 Non-Determinism Metrics

Two complementary metrics address model variability:

- **pass@k**: Probability of at least one correct solution in k attempts. Rises with more attempts.
- **pass^k**: Probability that all k trials succeed. Falls as consistency demands increase.

At k=1, both metrics equal per-trial success rate. At k=10, they diverge sharply -- pass@k approaches 100% while pass^k may approach 0%. These reflect fundamentally different product requirements (exploratory vs. production).

### 6.4 Evaluation by Agent Type

**Coding Agents**: Deterministic pass-fail test suites. SWE-Bench style: fix failing tests without breaking existing ones. Supplement with transcript-level grading for code quality.

**Conversational Agents**: Multidimensional success -- task completion (state checks), turn efficiency (transcript constraints), interaction quality (LLM rubrics). Adversarial user personas.

**Research Agents**: Combined graders -- groundedness checks (source support), coverage checks (required facts), source quality verification.

**Computer Use Agents**: Sandboxed environments. URL/page state verification plus backend verification that actions genuinely occurred.

### 6.5 Eval Development Roadmap

**Initial Setup (Steps 0-3):**
- Start with 20-50 tasks drawn from real failures
- Begin with manual checks already performed pre-release and bug-tracker failures
- Ensure specs are unambiguous (domain experts reach identical verdicts independently)
- Create reference solutions proving solvability
- Build balanced problem sets (test where behaviors should AND shouldn't occur)

**Infrastructure Design (Steps 4-5):**
- Maintain isolated trial environments with clean state
- Avoid shared infrastructure artifacts (cached data, resource exhaustion create correlated failures)
- Grade outputs, not specific tool call sequences (reward creative problem-solving)
- Implement partial credit for multi-component tasks
- Calibrate LLM judges against human expertise; provide "Unknown" escape valves

**Long-Term Maintenance (Steps 6-8):**
- Regularly examine transcripts to understand failures
- Monitor for eval saturation (100% pass rate = graduate to regression suite)
- Establish dedicated ownership where domain experts contribute tasks like PRs

### 6.6 Complementary Evaluation Methods (Swiss Cheese Model)

| Method | Strength | Limitation |
|--------|----------|-----------|
| Automated Evals | Reproducible, fast iteration | Requires ongoing maintenance |
| Production Monitoring | Ground truth at scale | Reactive, impacts real users |
| A/B Testing | Measures user outcomes | Slow, requires traffic |
| User Feedback | Surfaces unanticipated issues | Sparse, self-selected |
| Transcript Review | Builds failure-mode intuition | Time-intensive |
| Human Studies | Expert quality judgments | Expensive, slow |

Multiple overlapping methods catch failures each layer misses.

---

## 7. Common Failure Modes & Mitigations

| Failure Mode | Mitigation |
|-------------|-----------|
| Premature completion | Comprehensive feature JSON with explicit pass/fail status |
| Undocumented progress | Git repo + progress file; review history and baseline test first |
| Context exhaustion | Single-feature-per-session; context resets over compaction |
| Self-evaluation bias | Separate generator and evaluator agents |
| Rigid grading | Grade outcomes not sequences; partial credit; calibrate judges |
| Eval saturation | Graduate to regression suite; add harder tasks |
| Ambiguous task specs | Ensure domain experts independently reach identical verdicts |
| Configuration drift | init.sh automation; read documentation at session start |

---

## 8. Key Takeaways

1. **Harnesses are the differentiator.** The model is necessary but not sufficient. The harness architecture determines whether agents can sustain quality over extended runs.

2. **Separate generation from evaluation.** Self-critique doesn't work reliably. Use distinct agents for creating and judging work.

3. **Context resets beat compaction.** Fresh starts with structured handoffs outperform summarized continuations.

4. **Start evals early with real failures.** 20-50 tasks from actual bugs are more valuable than hundreds of synthetic tests.

5. **Harnesses must evolve with models.** Every component encodes assumptions about limitations. As models improve, strip away unnecessary scaffolding.

6. **The pattern mirrors human teams.** Shift-based handoffs, clean code states, comprehensive task decomposition, and mandatory quality verification before session end.

7. **Cost scales with quality.** 20x cost can yield dramatically better results. Frame harness decisions as quality investments.

8. **Grade outcomes, not processes.** Reward creative problem-solving rather than enforcing specific tool call sequences.
