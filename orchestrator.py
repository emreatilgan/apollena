"""Orchestrator — coordinates the autonomous CTR research pipeline.

Usage:
    python orchestrator.py [--domain DOMAIN] [--resume]
                           [--max-iterations N] [--target-auc AUC]
                           [--patience N] [--rescout-after N] [--max-budget-usd USD]

Scout runs once to seed the literature. The four-stage pipeline is then wrapped
in an OUTER OPTIMIZATION LOOP: each iteration runs Theorist → Engineer →
Benchmarker, and every prior attempt's (hypothesis, AUC, status, failure_reason)
is fed back to the next Theorist call so benchmark results steer the next
hypothesis. The loop stops on an AUC target, on patience (no improvement for N
rounds), at a max iteration count, or at an optional cumulative budget. When the
search stalls, Scout is re-run for fresh papers before the next hypothesis.

Each stage still writes its output to disk and must pass a validation gate. A
stage that fails after MAX_RETRIES_PER_STAGE abandons only the current iteration
(the failure is fed back to the Theorist) — it does not halt the whole loop.
Only the one-time bootstrap Scout failure is fatal.
"""

import argparse
import json
import logging
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anyio

from agents.scout import run_scout, validate_scout_output
from agents.theorist import run_theorist, validate_theorist_output
from agents.engineer import run_engineer, validate_engineer_output, _extract_short_name
from agents.benchmarker import run_benchmarker, validate_benchmarker_output

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent
SESSIONS_FILE = PROJECT_ROOT / "sessions.json"
SCOUT_OUTPUT_FILE = PROJECT_ROOT / "scout_output.json"
HYPOTHESIS_FILE = PROJECT_ROOT / "hypothesis.json"
BENCHMARK_RESULT_FILE = PROJECT_ROOT / "results" / "benchmark_result.json"
RUNS_DIR = PROJECT_ROOT / "runs"

VALID_DOMAINS = ["statistical_physics", "neuroscience", "ecology", "information_theory"]
DEFAULT_DOMAIN = "statistical_physics"
MAX_RETRIES_PER_STAGE = 3

# Outer-loop defaults (overridable via CLI flags)
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_TARGET_AUC = 0.98
DEFAULT_PATIENCE = 2
DEFAULT_RESCOUT_AFTER = 1

ITER_STAGE_ORDER = ["theorist", "engineer", "benchmarker"]

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")


# --- Session Management ---

def load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        return json.loads(SESSIONS_FILE.read_text())
    return {}


def save_sessions(sessions: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2))


def init_session(domain: str, config: dict) -> dict:
    """Create a new session state dict for an outer-loop run."""
    return {
        "run_id": str(uuid.uuid4())[:8],
        "domain": domain,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": dict(config),
        "total_cost_usd": 0.0,
        "scout": {"status": "pending", "session_id": None, "output": None, "runs": 0},
        "iterations": [],
        "best": None,
        "stall": 0,
        "rescout_next": False,
        "rescouted_this_stall": False,
        "stop_reason": None,
    }


def _new_iteration_record(index: int, rescouted: bool) -> dict:
    return {
        "index": index,
        "rescouted": rescouted,
        "stages": {
            "theorist": {"status": "pending", "session_id": None, "output": None},
            "engineer": {"status": "pending", "session_id": None, "output": None},
            "benchmarker": {"status": "pending", "session_id": None, "output": None},
        },
        "result": None,
        "attempt": None,
    }


# --- Pure decision helpers (no I/O — unit-testable) ---

def should_stop(sessions: dict, config: dict) -> str | None:
    """Return a stop reason if the outer loop should terminate, else None.

    Assumes any in-progress iteration has already been handled by the caller.
    """
    best = sessions.get("best")
    if best and best.get("auc") is not None and best["auc"] >= config["target_auc"]:
        return "target_auc"

    max_budget = config.get("max_budget_usd")
    if max_budget and sessions.get("total_cost_usd", 0.0) >= max_budget:
        return "budget"

    if len(sessions["iterations"]) >= config["max_iterations"]:
        return "max_iterations"

    if sessions.get("stall", 0) >= config["patience"]:
        return "patience"

    return None


def next_action(sessions: dict, config: dict) -> tuple[str, object]:
    """Decide the next thing the loop should do.

    Returns one of:
      ("scout", None)          — run (or re-run) the one-time bootstrap Scout
      ("resume_iter", index)   — finish an iteration interrupted mid-flight
      ("start_iter", index)    — begin a fresh iteration
      ("stop", reason)         — terminate the loop
    """
    if sessions["scout"]["status"] != "completed":
        return ("scout", None)

    iterations = sessions["iterations"]
    if iterations and iterations[-1].get("result") is None:
        return ("resume_iter", iterations[-1]["index"])

    reason = should_stop(sessions, config)
    if reason:
        return ("stop", reason)

    return ("start_iter", len(iterations))


def update_best(sessions: dict, it: dict) -> bool:
    """Update sessions['best'] if this iteration improved AUC. Returns True if so."""
    result = it.get("result") or {}
    auc = result.get("auc")
    if auc is None:
        return False

    best = sessions.get("best")
    if best is None or best.get("auc") is None or auc > best["auc"]:
        sessions["best"] = {
            "iteration": it["index"],
            "model_name": result.get("model_name"),
            "auc": auc,
            "logloss": result.get("logloss"),
            "dataset": result.get("dataset"),
            "hypothesis_file": it["stages"]["theorist"].get("output"),
            "result_file": it["stages"]["benchmarker"].get("output"),
        }
        return True
    return False


def update_stall(sessions: dict, config: dict, improved: bool) -> None:
    """Advance the stall counter and arm a re-scout when stuck."""
    if improved:
        sessions["stall"] = 0
        sessions["rescouted_this_stall"] = False
        return

    sessions["stall"] = sessions.get("stall", 0) + 1
    if (sessions["stall"] >= config["rescout_after"]
            and not sessions.get("rescouted_this_stall", False)):
        sessions["rescout_next"] = True
        sessions["rescouted_this_stall"] = True


def build_attempt_history(sessions: dict) -> list[dict]:
    """Collect the compact per-iteration records fed back to the Theorist."""
    return [it["attempt"] for it in sessions["iterations"] if it.get("attempt")]


def hypothesis_is_distinct(hypothesis: dict, prior_attempts: list[dict]) -> bool:
    """True if the hypothesis model_name does not repeat a prior attempt's name."""
    new_name = _extract_short_name(hypothesis.get("model_name", "") or "").lower()
    if not new_name:
        return True  # let validate_theorist_output handle the empty-name case
    prior_names = {(a.get("model_name") or "").lower() for a in prior_attempts}
    return new_name not in prior_names


# --- Archival & lookups ---

def _archive_iteration_file(run_id: str, iter_index: int, src_path, dst_name: str) -> Path:
    """Snapshot a working file into runs/<run_id>/iter_<NNN>/ for history & repro."""
    src = Path(src_path)
    dst_dir = RUNS_DIR / run_id / f"iter_{iter_index:03d}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / dst_name
    if src.exists():
        shutil.copy2(src, dst)
    return dst


def _load_iteration_hypothesis(it: dict) -> dict | None:
    path = it["stages"]["theorist"].get("output")
    if path and Path(path).exists():
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return None
    return None


def _arch_idea(hypothesis: dict | None) -> str | None:
    if not hypothesis:
        return None
    new_layer = (hypothesis.get("architecture") or {}).get("new_layer")
    return new_layer[:400] if new_layer else None


def _seen_titles(sessions: dict) -> list[str]:
    """Titles of the papers currently in play — handed to a re-scout to avoid."""
    if not SCOUT_OUTPUT_FILE.exists():
        return []
    try:
        scout = json.loads(SCOUT_OUTPUT_FILE.read_text())
    except Exception:
        return []
    return [p.get("title") for p in scout.get("papers", []) if p.get("title")]


# --- Scout (one-time bootstrap + stall-breaking re-scout) ---

async def run_scout_with_retries(sessions: dict, config: dict, avoid_titles=None) -> bool:
    """Run Scout with validation + retries. Returns True if it passed the gate."""
    label = "RE-SCOUT" if avoid_titles else "SCOUT"
    log.info(f"=== {label} (domain={sessions['domain']}) ===")
    retries = 0
    while retries < MAX_RETRIES_PER_STAGE:
        try:
            sessions["scout"]["status"] = "in_progress"
            save_sessions(sessions)

            result = await run_scout(sessions["domain"], avoid_titles=avoid_titles)
            sessions["total_cost_usd"] = sessions.get("total_cost_usd", 0.0) + result.get("cost_usd", 0.0)
            sessions["scout"]["session_id"] = result.get("session_id")

            is_valid, error = validate_scout_output(result.get("output"))
            if is_valid:
                sessions["scout"]["status"] = "completed"
                sessions["scout"]["output"] = str(SCOUT_OUTPUT_FILE)
                sessions["scout"]["runs"] = sessions["scout"].get("runs", 0) + 1
                save_sessions(sessions)
                log.info("Scout validation PASSED")
                return True

            retries += 1
            log.warning(f"Scout failed validation, retry {retries}/{MAX_RETRIES_PER_STAGE}: {error}")
            sessions["scout"]["status"] = "pending"
            sessions["scout"]["error"] = error
            save_sessions(sessions)
        except Exception as e:
            retries += 1
            log.error(f"Scout raised exception: {e}, retry {retries}/{MAX_RETRIES_PER_STAGE}")
            sessions["scout"]["status"] = "error"
            sessions["scout"]["error"] = str(e)
            save_sessions(sessions)

    return False


async def maybe_rescout(sessions: dict, config: dict, iter_index: int) -> bool:
    """If a stall armed a re-scout, refresh the literature before the next iteration."""
    if not sessions.get("rescout_next"):
        return False

    avoid = _seen_titles(sessions)
    # Back up the current (valid) papers so a failed re-scout can't corrupt them.
    backup = SCOUT_OUTPUT_FILE.read_text() if SCOUT_OUTPUT_FILE.exists() else None

    log.info(f"Re-scout armed (stall={sessions.get('stall', 0)}); fetching fresh papers")
    ok = await run_scout_with_retries(sessions, config, avoid_titles=avoid)
    sessions["rescout_next"] = False
    if ok:
        _archive_iteration_file(sessions["run_id"], iter_index, SCOUT_OUTPUT_FILE, "scout_output.json")
        save_sessions(sessions)
        return True

    # Restore the prior papers and scout status so the loop keeps working.
    if backup is not None:
        SCOUT_OUTPUT_FILE.write_text(backup)
    sessions["scout"]["status"] = "completed"
    log.warning("Re-scout failed; continuing with existing papers")
    save_sessions(sessions)
    return False


# --- Per-stage runners (iteration-scoped) ---

async def _stage_theorist(sessions: dict, config: dict, it: dict, iter_index: int) -> tuple[bool, str]:
    scout_output = json.loads(SCOUT_OUTPUT_FILE.read_text())
    attempt_history = build_attempt_history(sessions)
    best = sessions.get("best")

    result = await run_theorist(
        scout_output,
        attempt_history=attempt_history or None,
        best_so_far=best,
        target_auc=config["target_auc"],
    )
    sessions["total_cost_usd"] += result.get("cost_usd", 0.0)
    it["stages"]["theorist"]["session_id"] = result.get("session_id")

    output = result.get("output")
    is_valid, error = validate_theorist_output(output)
    if is_valid and attempt_history and not hypothesis_is_distinct(output, attempt_history):
        is_valid = False
        error = f"model_name '{output.get('model_name')}' repeats a prior attempt; must be materially different"

    if is_valid:
        archive = _archive_iteration_file(sessions["run_id"], iter_index, HYPOTHESIS_FILE, "hypothesis.json")
        it["stages"]["theorist"]["status"] = "completed"
        it["stages"]["theorist"]["output"] = str(archive)
        return True, ""

    it["stages"]["theorist"]["status"] = "failed"
    it["stages"]["theorist"]["error"] = error
    return False, error


async def _stage_engineer(sessions: dict, config: dict, it: dict, iter_index: int) -> tuple[bool, str]:
    hypothesis = json.loads(HYPOTHESIS_FILE.read_text())

    result = await run_engineer(hypothesis)
    sessions["total_cost_usd"] += result.get("cost_usd", 0.0)
    it["stages"]["engineer"]["session_id"] = result.get("session_id")

    output = result.get("output")
    is_valid, error = validate_engineer_output(output)
    if is_valid:
        it["stages"]["engineer"]["status"] = "completed"
        it["stages"]["engineer"]["output"] = output
        return True, ""

    it["stages"]["engineer"]["status"] = "failed"
    it["stages"]["engineer"]["error"] = error
    return False, error


async def _stage_benchmarker(sessions: dict, config: dict, it: dict, iter_index: int) -> tuple[bool, str]:
    hypothesis = json.loads(HYPOTHESIS_FILE.read_text())
    model_name = _extract_short_name(hypothesis["model_name"])
    engineer_output = it["stages"]["engineer"].get("output") or {}
    model_file = engineer_output.get("model_file", "")

    result = await run_benchmarker(model_name, model_file)
    sessions["total_cost_usd"] += result.get("cost_usd", 0.0)
    it["stages"]["benchmarker"]["session_id"] = result.get("session_id")

    output = result.get("output")
    is_valid, error = validate_benchmarker_output(output)
    if is_valid:
        archive = _archive_iteration_file(sessions["run_id"], iter_index, BENCHMARK_RESULT_FILE, "benchmark_result.json")
        it["stages"]["benchmarker"]["status"] = "completed"
        it["stages"]["benchmarker"]["output"] = str(archive)
        return True, ""

    it["stages"]["benchmarker"]["status"] = "failed"
    it["stages"]["benchmarker"]["error"] = error
    return False, error


_STAGE_FUNCS = {
    "theorist": _stage_theorist,
    "engineer": _stage_engineer,
    "benchmarker": _stage_benchmarker,
}


# --- Iteration finalization (build result + feedback record) ---

def _finalize_completed_iteration(it: dict, iter_index: int) -> None:
    bench = json.loads(Path(it["stages"]["benchmarker"]["output"]).read_text())
    hypothesis = _load_iteration_hypothesis(it)
    model_name = (
        _extract_short_name(hypothesis["model_name"])
        if hypothesis and hypothesis.get("model_name")
        else bench.get("model_name")
    )
    source_mechanism = hypothesis.get("source_mechanism") if hypothesis else None

    it["result"] = {
        "model_name": model_name,
        "auc": bench.get("auc"),
        "logloss": bench.get("logloss"),
        "dataset": bench.get("dataset"),
        "status": "completed",
        "failure_reason": None,
    }
    it["attempt"] = {
        "iteration": iter_index,
        "model_name": model_name,
        "source_mechanism": source_mechanism,
        "architecture_idea": _arch_idea(hypothesis),
        "auc": bench.get("auc"),
        "logloss": bench.get("logloss"),
        "status": "completed",
        "failure_reason": None,
    }


def _finalize_failed_iteration(it: dict, iter_index: int, failed_stage: str) -> None:
    err = it["stages"][failed_stage].get("error", "unknown error")
    failure_reason = f"{failed_stage}: {err}"
    hypothesis = _load_iteration_hypothesis(it)
    model_name = (
        _extract_short_name(hypothesis["model_name"])
        if hypothesis and hypothesis.get("model_name")
        else None
    )
    source_mechanism = hypothesis.get("source_mechanism") if hypothesis else None

    it["result"] = {
        "model_name": model_name,
        "auc": None,
        "logloss": None,
        "dataset": None,
        "status": "failed",
        "failure_reason": failure_reason,
    }
    it["attempt"] = {
        "iteration": iter_index,
        "model_name": model_name,
        "source_mechanism": source_mechanism,
        "architecture_idea": _arch_idea(hypothesis),
        "auc": None,
        "logloss": None,
        "status": "failed",
        "failure_reason": failure_reason,
    }


async def run_iteration(sessions: dict, config: dict, iter_index: int, it: dict) -> bool:
    """Run Theorist → Engineer → Benchmarker for one iteration (with per-stage
    retries). Returns True if the iteration produced a valid benchmark, else False.
    A failed stage abandons only this iteration; the failure is recorded for the
    Theorist's feedback history.
    """
    # On resume, ensure the working hypothesis.json reflects THIS iteration.
    th = it["stages"]["theorist"]
    if th["status"] == "completed" and th.get("output") and Path(th["output"]).exists():
        shutil.copy2(th["output"], HYPOTHESIS_FILE)

    for stage in ITER_STAGE_ORDER:
        if it["stages"][stage]["status"] == "completed":
            continue  # resume: skip stages already done

        func = _STAGE_FUNCS[stage]
        retries = 0
        passed = False
        while retries < MAX_RETRIES_PER_STAGE:
            try:
                it["stages"][stage]["status"] = "in_progress"
                save_sessions(sessions)
                passed, error = await func(sessions, config, it, iter_index)
                save_sessions(sessions)
                if passed:
                    break
                retries += 1
                log.warning(
                    f"[iter {iter_index}] stage '{stage}' failed validation, "
                    f"retry {retries}/{MAX_RETRIES_PER_STAGE}: {error}"
                )
            except Exception as e:
                retries += 1
                log.error(
                    f"[iter {iter_index}] stage '{stage}' raised exception: {e}, "
                    f"retry {retries}/{MAX_RETRIES_PER_STAGE}"
                )
                it["stages"][stage]["status"] = "error"
                it["stages"][stage]["error"] = str(e)
                save_sessions(sessions)

        if not passed:
            log.error(f"[iter {iter_index}] stage '{stage}' failed after {MAX_RETRIES_PER_STAGE} retries; abandoning iteration")
            _finalize_failed_iteration(it, iter_index, stage)
            save_sessions(sessions)
            return False

    _finalize_completed_iteration(it, iter_index)
    save_sessions(sessions)
    return True


# --- Main Pipeline ---

async def run_pipeline(domain: str, resume: bool, config: dict) -> None:
    """Run the outer optimization loop, optionally resuming from a checkpoint."""
    if resume and SESSIONS_FILE.exists():
        sessions = load_sessions()
        config = sessions.get("config", config)  # stored config is authoritative on resume
        log.info(f"Resuming run {sessions['run_id']} from checkpoint (config={config})")
    else:
        sessions = init_session(domain, config)
        save_sessions(sessions)
        log.info(f"Starting new run {sessions['run_id']} (domain={domain}, config={config})")

    while True:
        action, arg = next_action(sessions, config)

        if action == "stop":
            sessions["stop_reason"] = arg
            save_sessions(sessions)
            log.info(f"Outer loop stopping: {arg}")
            break

        if action == "scout":
            if not await run_scout_with_retries(sessions, config):
                log.error("Scout bootstrap failed after retries. Pipeline halted.")
                sys.exit(1)
            continue

        # action is start_iter or resume_iter
        iter_index = arg
        if action == "start_iter":
            did_rescout = await maybe_rescout(sessions, config, iter_index)
            it = _new_iteration_record(iter_index, did_rescout)
            sessions["iterations"].append(it)
            save_sessions(sessions)
            log.info(f"=== ITERATION {iter_index} (domain={sessions['domain']}) ===")
        else:
            it = sessions["iterations"][iter_index]
            log.info(f"=== RESUME ITERATION {iter_index} ===")

        completed = await run_iteration(sessions, config, iter_index, it)
        improved = update_best(sessions, it) if completed else False
        update_stall(sessions, config, improved)
        save_sessions(sessions)

    _print_final_report(sessions)

    if sessions.get("best") is None:
        log.error("No iteration produced a valid model. Pipeline failed.")
        sys.exit(1)


def _print_final_report(sessions: dict) -> None:
    """Print a summary of the outer-loop run."""
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(f"Run ID: {sessions['run_id']}")
    log.info(f"Domain: {sessions['domain']}")
    log.info(f"Stop reason: {sessions.get('stop_reason')}")
    log.info(f"Iterations run: {len(sessions['iterations'])} | Scout runs: {sessions['scout'].get('runs', 0)}")
    log.info(f"Total agent cost: ${sessions.get('total_cost_usd', 0.0):.2f}")
    log.info("-" * 60)
    log.info("Per-iteration results:")
    for it in sessions["iterations"]:
        r = it.get("result") or {}
        auc = r.get("auc")
        auc_str = f"{auc:.5f}" if isinstance(auc, (int, float)) else "—"
        flag = " (re-scout)" if it.get("rescouted") else ""
        log.info(f"  iter {it['index']}: {r.get('model_name', '?')} | AUC={auc_str} | {r.get('status', '?')}{flag}")

    log.info("-" * 60)
    best = sessions.get("best")
    if not best:
        log.info("BEST: none — no iteration produced a valid model")
        return

    auc = best.get("auc")
    dataset = best.get("dataset", "unknown")
    log.info(f"BEST: {best.get('model_name')} (iter {best.get('iteration')}) | AUC={auc} | logloss={best.get('logloss')}")
    if dataset == "Frappe_x1" and auc and auc > 0.98:
        log.info("  >>> Excellent result on Frappe_x1! <<<")
    elif dataset == "Criteo_x1" and auc:
        if auc > 0.8137:
            log.info("  >>> NEW #1 ON BARS Criteo_x1 LEADERBOARD! <<<")
        elif auc > 0.8100:
            log.info("  >>> Top-10 candidate on BARS Criteo_x1 leaderboard")
    log.info(f"  Hypothesis: {best.get('hypothesis_file')}")
    log.info(f"  Result:     {best.get('result_file')}")


def main():
    parser = argparse.ArgumentParser(description="Apollena — Autonomous CTR Research Pipeline")
    parser.add_argument(
        "--domain",
        choices=VALID_DOMAINS,
        default=DEFAULT_DOMAIN,
        help=f"Target scientific domain (default: {DEFAULT_DOMAIN})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint in sessions.json (keeps the stored config)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Max outer-loop iterations, each ~$5-7 (default: {DEFAULT_MAX_ITERATIONS}; 1 == single pass)",
    )
    parser.add_argument(
        "--target-auc",
        type=float,
        default=DEFAULT_TARGET_AUC,
        help=f"Stop early once best AUC reaches this (default: {DEFAULT_TARGET_AUC})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Stop after this many iterations with no AUC improvement (default: {DEFAULT_PATIENCE})",
    )
    parser.add_argument(
        "--rescout-after",
        type=int,
        default=DEFAULT_RESCOUT_AFTER,
        help=f"Re-scout for fresh papers after this many stalled iterations (default: {DEFAULT_RESCOUT_AFTER})",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=None,
        help="Optional hard cap on cumulative agent spend across the whole run (default: off)",
    )
    args = parser.parse_args()

    config = {
        "max_iterations": args.max_iterations,
        "target_auc": args.target_auc,
        "patience": args.patience,
        "rescout_after": args.rescout_after,
        "max_budget_usd": args.max_budget_usd,
    }

    anyio.run(run_pipeline, args.domain, args.resume, config)


if __name__ == "__main__":
    main()
