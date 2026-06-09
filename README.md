# Apollena

**A cross-domain, fully autonomous AI research agent that discovers, implements, and benchmarks novel CTR-prediction algorithms — with no human in the loop.**

Apollena borrows mechanisms from non-ML sciences (statistical physics, neuroscience, ecology, information theory) and turns them into new click-through-rate (CTR) prediction architectures. It runs the full scientific discovery loop — *hypothesize → build → experiment → interpret → repeat* — end to end, autonomously, for a few dollars per run.

It benchmarks on the BARS **Frappe_x1** dataset (CPU-friendly, ~288K samples); **Criteo_x1** is available for full-scale GPU runs.

---

## How it works

Four LLM subagents do the work, coordinated by an **Orchestrator that is plain deterministic Python** (not an LLM). Each subagent runs in an isolated context window (a separate Claude Agent SDK `query()` call) and communicates *only* through JSON files on disk — full conversation history is never shared between agents.

```
Scout (once)  ─→  ┌─ outer optimization loop ───────────────────┐
                  │  Theorist → Engineer → Benchmarker           │
                  │      ▲  feedback: prior attempts + best AUC   │
                  └──────┴───────────────────────────────────────┘
```

| Agent | Role | Tools | Budget |
|---|---|---|---|
| **Scout** | Literature search across a target science domain | WebSearch, WebFetch, Read, Write | $1.50 |
| **Theorist** | Maps a domain mechanism → differentiable architecture hypothesis | Read, Write | $2.00 |
| **Engineer** | Implements the FuxiCTR model class + tests, iterates until `pytest` passes | Read, Edit, Write, Bash, Glob, Grep | $2.00 |
| **Benchmarker** | Runs Optuna hyperparameter tuning on Frappe_x1, captures AUC/logloss | Read, Write, Edit, Bash | $3.00 |

The **Theorist** is the only agent that uses extended thinking (`thinking={"type": "adaptive"}`, `effort="max"`) — the system invests its deepest reasoning in hypothesis generation.

### Pipeline stages & validation gates

Stages run sequentially; each must pass a gate before the next is spawned. A failed gate re-spawns the **same** agent with corrected instructions — it does **not** restart the pipeline.

| Stage | Agent | Output | Validation gate |
|---|---|---|---|
| 1 | Scout | `scout_output.json` | ≥3 papers, each with `novelty_score ≥ 6` |
| 2 | Theorist | `hypothesis.json` | Not in BARS taxonomy; differentiable; distinct from prior attempts |
| 3 | Engineer | model file + test file | `pytest` passes |
| 4 | Benchmarker | `results/benchmark_result.json` | AUC is finite; training completes |

### Outer optimization loop

Scout runs **once** to seed the literature. Stages 2–4 then repeat in a loop: each iteration's benchmark result is fed back to the next Theorist as a compact `attempt_history`, so hypotheses improve over time. A new hypothesis must be materially distinct from prior ones.

The loop stops on the first of:
- best AUC ≥ `--target-auc` (default `0.98`),
- cumulative spend ≥ `--max-budget-usd` (optional),
- `--max-iterations` reached (default `3`),
- `--patience` consecutive no-improvement iterations (default `2`).

When the search stalls (`--rescout-after` no-improvement iterations, default `1`), Scout re-runs for **fresh** papers before the next hypothesis. A stage that fails after its retries abandons only the current iteration (the failure is fed back to the Theorist); only a Scout-bootstrap failure is fatal. `--max-iterations 1` reproduces the original single-pass behavior.

---

## Quick start

### Prerequisites

- Python 3.10+
- A [FuxiCTR](https://github.com/reczoo/FuxiCTR) workspace at `~/Projects/FuxiCTR/` (outside this repo), with the Frappe_x1 dataset prepared
- Claude Agent SDK credentials configured in your environment

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
# Default: statistical_physics domain, up to 3 iterations, target AUC 0.98
python orchestrator.py

# Pick a domain and tune the loop
python orchestrator.py --domain neuroscience --max-iterations 5 --target-auc 0.985

# Single pass (original behavior)
python orchestrator.py --max-iterations 1

# Resume after an interruption (keeps the stored config)
python orchestrator.py --resume
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--domain` | `statistical_physics` | Target domain: `statistical_physics`, `neuroscience`, `ecology`, `information_theory` |
| `--resume` | off | Resume from the last checkpoint in `sessions.json` |
| `--max-iterations` | `3` | Max outer-loop iterations (each ~$5–7) |
| `--target-auc` | `0.98` | Stop early once best AUC reaches this |
| `--patience` | `2` | Stop after this many no-improvement iterations |
| `--rescout-after` | `1` | Re-scout for fresh papers after this many stalled iterations |
| `--max-budget-usd` | off | Optional hard cap on cumulative agent spend |

### Cost

A single pass (Scout once + one Theorist→Engineer→Benchmarker iteration) caps at ~$8.50 and typically costs **$4–8**. Each additional iteration adds ~$5–7, so a default 3-iteration run costs ~$15–21 worst case. `--max-budget-usd` caps cumulative spend.

---

## State & resume

The Orchestrator writes pipeline state (per-subagent `session_id`, status, costs, per-iteration results, and the best result so far) to `sessions.json` after every state change. On interruption (max-turns, budget, API failure), `--resume` continues from the last checkpoint at **per-stage granularity within the active iteration** — completed stages are skipped, the interrupted stage restarts — rather than restarting the whole pipeline.

---

## Project structure

```
apollena/
├── orchestrator.py          # Entry point — deterministic Python pipeline coordinator
├── agents/                  # The four LLM subagents
│   ├── scout.py
│   ├── theorist.py
│   ├── engineer.py
│   └── benchmarker.py
├── prompts/                 # System prompts as markdown, loaded at runtime
├── schemas/                 # Input/output JSON schemas per agent
├── configs/                 # FuxiCTR experiment YAML configs
├── infra/                   # GCP VM provisioning + GPU tuning scripts
├── tests/                   # Unit tests for the outer-loop decision logic
└── requirements.txt
```

### Runtime artifacts (gitignored — not committed)

These are generated per run and deleted routinely between runs. A clean checkout contains **none** of them, and their absence does **not** mean the pipeline hasn't run:

- `scout_output.json` — latest Scout output
- `hypothesis.json` — current Theorist output
- `sessions.json` — orchestrator session / resume state
- `results/benchmark_result.json` — latest Benchmarker output
- `runs/<run_id>/iter_<NNN>/` — per-iteration archive (hypothesis + result snapshots)

---

## Design principles

- **Deterministic orchestration, LLM cognition.** Control flow, state, and validation live in plain Python; the agents only reason. This is what lets the loop close reliably without a human.
- **File-based handoff.** Each agent writes its complete output to disk; the Orchestrator passes only the relevant JSON subset to the next agent. No conversation history is shared.
- **Validation gates everywhere.** Every stage must pass a gate before the next is spawned — even when the output looks correct.
- **Fully autonomous.** No human confirmation mid-pipeline. A documented failed attempt (mechanism → implemented → AUC below baseline) is itself a valid, publishable negative result.

---

## Success criteria

- **Primary (required):** a valid FuxiCTR model class with a finite AUC on Frappe_x1, produced fully autonomously.
- **Secondary (publishable):** AUC > 0.98 on Frappe_x1.
- **Stretch (fundable):** run on Criteo_x1 (GPU) and beat the current #1 BARS AUC.
- **Negative result:** a documented failed attempt below baseline is still publishable.

---

## Testing

```bash
python -m pytest tests/
```

The test suite (`tests/test_loop_logic.py`) exercises the pure decision helpers in the Orchestrator — stopping conditions, next-action selection, best-tracking, stall/re-scout logic, and hypothesis distinctness.

---

See [`CLAUDE.md`](CLAUDE.md) for the full architecture specification and contributor conventions.
