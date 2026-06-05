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
├── infra/                       # GCP VM provisioning + tuning scripts
└── requirements.txt

# Runtime artifacts — generated per run, gitignored, NOT committed.
# A clean checkout contains NONE of these. Their absence is EXPECTED and
# does not mean the pipeline hasn't run — they are scratch output, deleted
# routinely between runs (and after GCP VM runs). Do not treat them as source.
#   scout_output.json             # Scout stage output (latest scout)
#   hypothesis.json               # Theorist stage output (current iteration)
#   sessions.json                 # Orchestrator session / resume state (outer loop)
#   results/benchmark_result.json # Benchmarker stage output (latest run)
#   runs/<run_id>/iter_<NNN>/     # Per-iteration archive (hypothesis + result snapshots)
```

FuxiCTR workspace lives at `~/Projects/FuxiCTR/` (outside this repo).

---

## Agent Architecture

Four LLM subagents do the work, coordinated by the Orchestrator — a **deterministic Python script**, not itself an LLM/SDK agent. Each subagent runs in an isolated context window (a separate SDK `query()` call) — only its final JSON result is passed back.

| Agent | File | Role |
|---|---|---|
| Orchestrator | `orchestrator.py` | Pipeline coordinator — plain Python; runs each subagent in-process via the SDK `query()` (no `Task` tool) |
| Scout | `agents/scout.py` | Literature search across target domain |
| Theorist | `agents/theorist.py` | Maps domain mechanism → architectural hypothesis |
| Engineer | `agents/engineer.py` | Implements FuxiCTR model class, runs tests |
| Benchmarker | `agents/benchmarker.py` | Runs Frappe_x1 Optuna tuning, captures AUC/logloss |

---

## Claude Agent SDK Configuration

Use the Claude Agent SDK (Python). The Orchestrator is a deterministic Python script (no SDK agent config). Config per **subagent**:

| Agent | allowed_tools | permission_mode | effort | max_turns | max_budget_usd |
|---|---|---|---|---|---|
| Scout | WebSearch, WebFetch, Read, Write | acceptEdits | default | 20 | 1.50 |
| Theorist | Read, Write | acceptEdits | max | 20 | 2.00 |
| Engineer | Read, Edit, Write, Bash, Glob, Grep | bypassPermissions | default | 30 | 2.00 |
| Benchmarker | Read, Write, Edit, Bash | bypassPermissions | default | 50 | 3.00 |

Total max budget per **single pass** (Scout once + one Theorist→Engineer→Benchmarker iteration): ~$8.50 (sum of the four subagents; the Orchestrator is plain Python and carries no agent budget). Expected single-pass cost: $4–8. The outer optimization loop repeats Theorist→Engineer→Benchmarker (~$5–7/iteration), so a default `--max-iterations 3` run costs ~$15–21 worst case; `--max-budget-usd` caps cumulative spend.

The Theorist is the only agent that sets `effort` (`max`) and enables extended thinking (`thinking={"type": "adaptive"}`); the other three use the SDK default effort. The Theorist is permitted to take longer than other agents.

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

### Outer Optimization Loop

Scout runs **once** to seed the literature. Stages 2–4 (Theorist → Engineer → Benchmarker) are then wrapped in an outer optimization loop: each iteration's benchmark result is fed back to the next Theorist call (as a compact `attempt_history` of prior `model_name`/`AUC`/`status`/`failure_reason` records plus the `best_so_far`), so hypotheses improve across iterations. A new hypothesis must be **distinct** from prior attempts — the Orchestrator rejects a repeated `model_name` on top of the BARS novelty check.

The loop stops on the first of: best AUC ≥ `--target-auc` (default 0.98); cumulative spend ≥ `--max-budget-usd` (optional); `--max-iterations` reached (default 3); or `--patience` consecutive no-improvement iterations (default 2). When the search stalls (`--rescout-after` no-improvement iterations, default 1), Scout is re-run for **fresh** papers before the next hypothesis. A stage that fails after `MAX_RETRIES_PER_STAGE` abandons only the current iteration (the failure is fed back to the Theorist), not the whole loop; only a Scout-bootstrap failure is fatal. `--max-iterations 1` reproduces the original single-pass behavior.

---

## File-Based Handoff Protocol

Each agent writes its complete output to disk as a JSON file before returning. The Orchestrator reads these files and passes only the relevant subset as context to the next agent. Never pass full conversation history between agents.

---

## Session Resume Protocol

The Orchestrator writes `session_id` for each subagent run to `sessions.json`. On interruption (error_max_turns, error_max_budget_usd, API failure), resume with `--resume` from the last checkpoint — do not restart the full pipeline. Resume keeps the stored `config`. Resume granularity is per-stage *within* the active iteration: a completed stage is skipped, an interrupted iteration continues from its first pending stage, and a finished iteration advances the loop (re-evaluating the stopping conditions).

```json
// sessions.json structure (outer loop)
{
  "run_id": "...",
  "domain": "statistical_physics",
  "config": { "max_iterations": 3, "target_auc": 0.98, "patience": 2, "rescout_after": 1, "max_budget_usd": null },
  "total_cost_usd": 12.4,
  "scout": { "session_id": "...", "status": "completed", "output": "scout_output.json", "runs": 1 },
  "iterations": [
    {
      "index": 0,
      "rescouted": false,
      "stages": {
        "theorist":    { "session_id": "...", "status": "completed", "output": "runs/<run_id>/iter_000/hypothesis.json" },
        "engineer":    { "session_id": "...", "status": "completed", "output": { "model_name": "...", "model_file": "...", "test_passed": true } },
        "benchmarker": { "session_id": "...", "status": "completed", "output": "runs/<run_id>/iter_000/benchmark_result.json" }
      },
      "result":  { "model_name": "...", "auc": 0.971, "logloss": 0.18, "status": "completed", "failure_reason": null },
      "attempt": { "iteration": 0, "model_name": "...", "source_mechanism": "...", "auc": 0.971, "status": "completed" }
    }
  ],
  "best": { "iteration": 0, "model_name": "...", "auc": 0.971, "hypothesis_file": "...", "result_file": "..." },
  "stall": 0,
  "stop_reason": null
}
```

---

## FuxiCTR Model Implementation Rules

When the Engineer implements a new model:

1. Read an existing BARS model as a structural template (e.g. `model_zoo/DeepFM/DeepFM_torch/src/DeepFM.py`)
2. Subclass `BaseModel` from `fuxictr.pytorch.models`; implement `__init__()` and `forward()`
3. Use standard FuxiCTR constructor parameters — do not invent new ones
4. Create a model-zoo entry under `model_zoo/{ModelName}/{ModelName}_torch/` (with `src/{ModelName}.py`, `src/__init__.py`, `config/`, `run_expid.py`, `fuxictr_version.py`) — the model is registered via its model-zoo `src/`, not a global `fuxictr/.../__init__.py`
5. New layer goes in `fuxictr/pytorch/layers/` if it's reusable; inline otherwise
6. Test file: `tests/unit_tests/test_{ModelName}.py` — must test forward pass shape, finite loss, no NaN gradients
7. Test command: `cd ~/Projects/FuxiCTR && python -m pytest tests/unit_tests/test_{ModelName}.py`
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
- Prompts live in `prompts/` as markdown files — load them at runtime, do not inline long strings. (The four subagent prompts are loaded; `orchestrator_prompt.md` is descriptive only — `orchestrator.py` is plain Python and does not load it.)
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
