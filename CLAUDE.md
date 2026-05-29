# Apollena — Cross-Domain AI Research Agent System

## Project Overview

This system autonomously discovers, implements, and benchmarks novel CTR prediction algorithms by borrowing mechanisms from non-ML scientific domains. It benchmarks on the BARS Frappe_x1 dataset (CPU-friendly, ~288K samples). Criteo_x1 is available for full-scale GPU runs. No human intervention is required at any pipeline stage.

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
| Benchmarker | Read, Write, Edit, Bash | bypassPermissions | high | 50 | 3.00 |

Total max budget per pipeline run: ~$12.50. Expected cost: $4–8.

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

- **Primary (required):** Produces a valid FuxiCTR model class with a finite, valid AUC on Frappe_x1 — fully autonomously
- **Secondary (publishable):** AUC > 0.98 on Frappe_x1 (competitive with BARS leaderboard models)
- **Stretch (fundable):** Run on Criteo_x1 (GPU) and beat current #1 AUC (0.8137 as of v0.1)
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

## gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools directly.

Available gstack skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/connect-chrome`, `/qa`, `/qa-only`, `/design-review`, `/setup-browser-cookies`, `/setup-deploy`, `/setup-gbrain`, `/retro`, `/investigate`, `/document-release`, `/document-generate`, `/codex`, `/cso`, `/autoplan`, `/plan-devex-review`, `/devex-review`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`, `/learn`.

---

## What Not To Do

- Do not ask for human confirmation mid-pipeline — agents must be fully autonomous
- Do not pass full conversation history between agents — only structured JSON outputs
- Do not restart the full pipeline on a single-stage failure — resume from checkpoint
- Do not implement a hypothesis that is already in BARS
- Do not hardcode file paths — use constants defined at the top of each agent file
- Do not skip the validation gates even if the output looks correct

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec

## GBrain Configuration (configured by /setup-gbrain)
- Mode: local-stdio
- Engine: pglite
- Config file: ~/.gbrain/config.json (mode 0600)
- Setup date: 2026-05-28
- MCP registered: yes (user scope, /Users/emreatilgan/.bun/bin/gbrain serve)
- Artifacts sync: off
- Current repo policy: n/a (no origin remote)

## GBrain Search Guidance (configured by /sync-gbrain)
<!-- gstack-gbrain-search-guidance:start -->

GBrain is set up and synced on this machine. The agent should prefer gbrain
over Grep when the question is semantic or when you don't know the exact
identifier yet. Two indexed corpora available via the `gbrain` CLI:
- This repo's code (registered as `gstack-code-<repo>` source).
- `~/.gstack/` curated memory (registered as `gstack-brain-<user>` source via
  the existing federation pipeline).

Prefer gbrain when:
- "Where is X handled?" / semantic intent, no exact string yet:
    `gbrain search "<terms>"` or `gbrain query "<question>"`
- "Where is symbol Y defined?" / symbol-based code questions:
    `gbrain code-def <symbol>` or `gbrain code-refs <symbol>`
- "What calls Y?" / "What does Y depend on?":
    `gbrain code-callers <symbol>` / `gbrain code-callees <symbol>`
- "What did we decide last time?" / past plans, retros, learnings:
    `gbrain search "<terms>" --source gstack-brain-<user>`

Grep is still right for known exact strings, regex, multiline patterns, and
file globs. The brain auto-syncs incrementally on every gstack skill start.
Run `/sync-gbrain` to force-refresh, `/sync-gbrain --full` for full reindex.

<!-- gstack-gbrain-search-guidance:end -->
