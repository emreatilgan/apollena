# Apollena — Cross-Domain AI Research Agent System

## Project Overview

This system autonomously discovers, implements, and benchmarks novel CTR prediction algorithms by borrowing mechanisms from non-ML scientific domains. It targets the BARS / Criteo_x1 AUC leaderboard. No human intervention is required at any pipeline stage.

Reference: `architecture.docx` (v0.1, March 2026).

---

## Project Structure

```
apollena/
├── CLAUDE.md
├── orchestrator.py              # Entry point — runs the full pipeline
├── agents/
│   ├── scout.py
│   ├── theorist.py
│   ├── engineer.py
│   └── benchmarker.py
├── prompts/                     # System prompts as markdown files
│   ├── orchestrator_prompt.md
│   ├── scout_prompt.md
│   ├── theorist_prompt.md
│   ├── engineer_prompt.md
│   └── benchmarker_prompt.md
├── schemas/                     # Input/output JSON schemas per agent
├── configs/                     # FuxiCTR experiment YAML configs
├── results/                     # Benchmark output JSONs (gitignored if large)
├── sessions.json                # Session resume tracking (written by Orchestrator)
├── scout_output.json            # Scout stage output
├── hypothesis.json              # Theorist stage output
└── requirements.txt
```

FuxiCTR workspace lives at `~/Projects/FuxiCTR/` (outside this repo).

---

## Agent Architecture

Five agents. The Orchestrator coordinates; the four subagents do the work. Each subagent runs in an isolated context window — only its final JSON result is passed back.

| Agent | File | Role |
|---|---|---|
| Orchestrator | `orchestrator.py` | Pipeline coordinator, spawns subagents via `Task` tool |
| Scout | `agents/scout.py` | Literature search across target domain |
| Theorist | `agents/theorist.py` | Maps domain mechanism → architectural hypothesis |
| Engineer | `agents/engineer.py` | Implements FuxiCTR model class, runs tests |
| Benchmarker | `agents/benchmarker.py` | Runs Criteo_x1 training, captures AUC/logloss |

---

## Claude Agent SDK Configuration

Use the Claude Agent SDK (Python). Agent config per agent:

| Agent | allowed_tools | permission_mode | effort | max_turns | max_budget_usd |
|---|---|---|---|---|---|
| Orchestrator | Task, Read, Write, TodoWrite | acceptEdits | high | 50 | 5.00 |
| Scout | WebSearch, WebFetch, Read, Write | acceptEdits | high | 20 | 1.50 |
| Theorist | Read, Write | acceptEdits | max | 20 | 2.00 |
| Engineer | Read, Edit, Write, Bash, Glob, Grep | bypassPermissions | high | 30 | 2.00 |
| Benchmarker | Read, Write, Edit, Bash | bypassPermissions | medium | 15 | 1.00 |

Total max budget per pipeline run: ~$11.50. Expected cost: $3–6.

The Theorist is the only agent with `effort: max` and extended thinking enabled. It is permitted to take longer than other agents.

---

## Pipeline Stages & Validation Gates

Agents run sequentially. Each stage must pass a validation gate before the next is spawned.

| Stage | Agent | Output File | Validation Gate |
|---|---|---|---|
| 1 | Scout | `scout_output.json` | ≥3 papers returned, each with `novelty_score ≥ 6` |
| 2 | Theorist | `hypothesis.json` | Hypothesis not in BARS model list; must be differentiable |
| 3 | Engineer | model file + test file | `pytest` must pass |
| 4 | Benchmarker | `results/benchmark_result.json` | AUC is finite; training run completes without error |

If a validation gate fails, the Orchestrator re-spawns the same agent with corrected instructions. It does not restart the full pipeline.

---

## File-Based Handoff Protocol

Each agent writes its complete output to disk as a JSON file before returning. The Orchestrator reads these files and passes only the relevant subset as context to the next agent. Never pass full conversation history between agents.

---

## Session Resume Protocol

The Orchestrator writes `session_id` for each subagent run to `sessions.json`. On interruption (error_max_turns, error_max_budget_usd, API failure), resume from the last successful stage checkpoint — do not restart the full pipeline.

```json
// sessions.json structure
{
  "run_id": "...",
  "domain": "statistical_physics",
  "stages": {
    "scout": { "session_id": "...", "status": "completed", "output": "scout_output.json" },
    "theorist": { "session_id": "...", "status": "in_progress" },
    "engineer": { "status": "pending" },
    "benchmarker": { "status": "pending" }
  }
}
```

---

## FuxiCTR Model Implementation Rules

When the Engineer implements a new model:

1. Read an existing BARS model as a structural template (e.g. `fuxictr/models/DeepFM.py`)
2. Subclass `BaseModel`; implement `__init__()` and `forward()`
3. Use standard FuxiCTR constructor parameters — do not invent new ones
4. Register the new class in `fuxictr/models/__init__.py`
5. New layer goes in `fuxictr/layers/` if it's reusable; inline otherwise
6. Test file: `tests/test_{ModelName}.py` — must test forward pass shape, finite loss, no NaN gradients
7. Test command: `cd ~/Projects/FuxiCTR && python -m pytest tests/test_{ModelName}.py`
8. On test failure: read error, patch, re-run — loop up to 10 times before escalating to Orchestrator

---

## Hypothesis Validity Rules (Theorist)

A hypothesis is invalid if it:
- Is already present in the BARS model taxonomy (DeepFM, DCN, AutoInt, FinalMLP, etc.)
- Is a pure engineering variation (different activation, different normalisation, different optimizer)
- Cannot be expressed as a differentiable computation graph
- Does not trace to a mechanism from the target non-ML domain

---

## Target Domains (V1)

| Domain | Core Mechanism | Priority |
|---|---|---|
| Statistical Physics (Spin Glass) | Energy minimisation in frustrated interaction systems | HIGH |
| Neuroscience (Predictive Coding) | Hierarchical prediction error minimisation | HIGH |
| Ecology (Niche Theory) | Species occupying distinct resource niches | MED |
| Information Theory (Kolmogorov Complexity) | Algorithmic information content of sequences | MED |

Default domain for first run: `statistical_physics`.

---

## Success Criteria

- **Primary (required):** Produces a valid FuxiCTR model class with a finite, valid AUC on Criteo_x1 — fully autonomously
- **Secondary (publishable):** Top-10 AUC on BARS Criteo_x1 leaderboard
- **Stretch (fundable):** Beats current #1 AUC (0.8137 as of v0.1)
- **Negative result:** A documented failed attempt (mechanism found → implemented → AUC below baseline) is still publishable

---

## Coding Conventions

- Language: Python 3.10+
- Package manager: `pip` with `requirements.txt`
- Agent SDK: `claude-agent-sdk` (Python)
- All agent configs defined in the agent files themselves, not hardcoded in `orchestrator.py`
- Prompts live in `prompts/` as markdown files — load them at runtime, do not inline long strings
- All inter-agent data is JSON; use `json.dumps(indent=2)` for readability
- No external databases — file system only for state

---

## What Not To Do

- Do not ask for human confirmation mid-pipeline — agents must be fully autonomous
- Do not pass full conversation history between agents — only structured JSON outputs
- Do not restart the full pipeline on a single-stage failure — resume from checkpoint
- Do not implement a hypothesis that is already in BARS
- Do not hardcode file paths — use constants defined at the top of each agent file
- Do not skip the validation gates even if the output looks correct
