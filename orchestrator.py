"""Orchestrator — coordinates the autonomous CTR research pipeline.

Usage:
    python orchestrator.py [--domain DOMAIN] [--resume]

Runs the full pipeline: Scout → Theorist → Engineer → Benchmarker.
Each stage writes its output to disk and must pass a validation gate
before the next stage is spawned.
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anyio

from agents.scout import run_scout, validate_scout_output
from agents.theorist import run_theorist, validate_theorist_output
from agents.engineer import run_engineer, validate_engineer_output
from agents.benchmarker import run_benchmarker, validate_benchmarker_output

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent
SESSIONS_FILE = PROJECT_ROOT / "sessions.json"
SCOUT_OUTPUT_FILE = PROJECT_ROOT / "scout_output.json"
HYPOTHESIS_FILE = PROJECT_ROOT / "hypothesis.json"
BENCHMARK_RESULT_FILE = PROJECT_ROOT / "results" / "benchmark_result.json"

VALID_DOMAINS = ["statistical_physics", "neuroscience", "ecology", "information_theory"]
DEFAULT_DOMAIN = "statistical_physics"
MAX_RETRIES_PER_STAGE = 3

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


def init_session(domain: str) -> dict:
    """Create a new session state dict."""
    return {
        "run_id": str(uuid.uuid4())[:8],
        "domain": domain,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {
            "scout": {"status": "pending", "session_id": None, "output": None},
            "theorist": {"status": "pending", "session_id": None, "output": None},
            "engineer": {"status": "pending", "session_id": None, "output": None},
            "benchmarker": {"status": "pending", "session_id": None, "output": None},
        },
    }


def get_next_stage(sessions: dict) -> str | None:
    """Return the name of the next stage to run, or None if all are completed."""
    stage_order = ["scout", "theorist", "engineer", "benchmarker"]
    for stage in stage_order:
        status = sessions["stages"][stage]["status"]
        if status != "completed":
            return stage
    return None


# --- Pipeline Stages ---

async def run_stage_scout(sessions: dict) -> bool:
    """Run the Scout stage. Returns True if validation passes."""
    domain = sessions["domain"]
    log.info(f"=== STAGE 1: SCOUT (domain={domain}) ===")

    sessions["stages"]["scout"]["status"] = "in_progress"
    save_sessions(sessions)

    result = await run_scout(domain)

    sessions["stages"]["scout"]["session_id"] = result.get("session_id")

    is_valid, error = validate_scout_output(result.get("output"))
    if is_valid:
        sessions["stages"]["scout"]["status"] = "completed"
        sessions["stages"]["scout"]["output"] = str(SCOUT_OUTPUT_FILE)
        save_sessions(sessions)
        log.info("Scout validation PASSED")
        return True
    else:
        sessions["stages"]["scout"]["status"] = "failed"
        sessions["stages"]["scout"]["error"] = error
        save_sessions(sessions)
        log.error(f"Scout validation FAILED: {error}")
        return False


async def run_stage_theorist(sessions: dict) -> bool:
    """Run the Theorist stage. Returns True if validation passes."""
    log.info("=== STAGE 2: THEORIST ===")

    scout_output = json.loads(SCOUT_OUTPUT_FILE.read_text())

    sessions["stages"]["theorist"]["status"] = "in_progress"
    save_sessions(sessions)

    result = await run_theorist(scout_output)

    sessions["stages"]["theorist"]["session_id"] = result.get("session_id")

    is_valid, error = validate_theorist_output(result.get("output"))
    if is_valid:
        sessions["stages"]["theorist"]["status"] = "completed"
        sessions["stages"]["theorist"]["output"] = str(HYPOTHESIS_FILE)
        save_sessions(sessions)
        log.info("Theorist validation PASSED")
        return True
    else:
        sessions["stages"]["theorist"]["status"] = "failed"
        sessions["stages"]["theorist"]["error"] = error
        save_sessions(sessions)
        log.error(f"Theorist validation FAILED: {error}")
        return False


async def run_stage_engineer(sessions: dict) -> bool:
    """Run the Engineer stage. Returns True if validation passes."""
    log.info("=== STAGE 3: ENGINEER ===")

    hypothesis = json.loads(HYPOTHESIS_FILE.read_text())

    sessions["stages"]["engineer"]["status"] = "in_progress"
    save_sessions(sessions)

    result = await run_engineer(hypothesis)

    sessions["stages"]["engineer"]["session_id"] = result.get("session_id")

    is_valid, error = validate_engineer_output(result.get("output"))
    if is_valid:
        sessions["stages"]["engineer"]["status"] = "completed"
        sessions["stages"]["engineer"]["output"] = result["output"]
        save_sessions(sessions)
        log.info("Engineer validation PASSED")
        return True
    else:
        sessions["stages"]["engineer"]["status"] = "failed"
        sessions["stages"]["engineer"]["error"] = error
        save_sessions(sessions)
        log.error(f"Engineer validation FAILED: {error}")
        return False


async def run_stage_benchmarker(sessions: dict) -> bool:
    """Run the Benchmarker stage. Returns True if validation passes."""
    log.info("=== STAGE 4: BENCHMARKER ===")

    hypothesis = json.loads(HYPOTHESIS_FILE.read_text())
    model_name = hypothesis["model_name"]

    engineer_output = sessions["stages"]["engineer"].get("output", {})
    model_file = engineer_output.get("model_file", "")

    sessions["stages"]["benchmarker"]["status"] = "in_progress"
    save_sessions(sessions)

    result = await run_benchmarker(model_name, model_file)

    sessions["stages"]["benchmarker"]["session_id"] = result.get("session_id")

    is_valid, error = validate_benchmarker_output(result.get("output"))
    if is_valid:
        sessions["stages"]["benchmarker"]["status"] = "completed"
        sessions["stages"]["benchmarker"]["output"] = str(BENCHMARK_RESULT_FILE)
        save_sessions(sessions)
        log.info("Benchmarker validation PASSED")
        return True
    else:
        sessions["stages"]["benchmarker"]["status"] = "failed"
        sessions["stages"]["benchmarker"]["error"] = error
        save_sessions(sessions)
        log.error(f"Benchmarker validation FAILED: {error}")
        return False


# --- Stage dispatch ---

STAGE_RUNNERS = {
    "scout": run_stage_scout,
    "theorist": run_stage_theorist,
    "engineer": run_stage_engineer,
    "benchmarker": run_stage_benchmarker,
}


# --- Main Pipeline ---

async def run_pipeline(domain: str, resume: bool = False) -> None:
    """Run the full pipeline, optionally resuming from checkpoint."""
    if resume and SESSIONS_FILE.exists():
        sessions = load_sessions()
        log.info(f"Resuming run {sessions['run_id']} from checkpoint")
    else:
        sessions = init_session(domain)
        save_sessions(sessions)
        log.info(f"Starting new run {sessions['run_id']} (domain={domain})")

    while True:
        stage = get_next_stage(sessions)
        if stage is None:
            log.info("All stages completed!")
            break

        runner = STAGE_RUNNERS[stage]
        retries = 0

        while retries < MAX_RETRIES_PER_STAGE:
            try:
                passed = await runner(sessions)
                if passed:
                    break
                retries += 1
                log.warning(f"Stage '{stage}' failed validation, retry {retries}/{MAX_RETRIES_PER_STAGE}")
                # Reset status for retry
                sessions["stages"][stage]["status"] = "pending"
                save_sessions(sessions)
            except Exception as e:
                retries += 1
                log.error(f"Stage '{stage}' raised exception: {e}, retry {retries}/{MAX_RETRIES_PER_STAGE}")
                sessions["stages"][stage]["status"] = "error"
                sessions["stages"][stage]["error"] = str(e)
                save_sessions(sessions)

        if sessions["stages"][stage]["status"] != "completed":
            log.error(f"Stage '{stage}' failed after {MAX_RETRIES_PER_STAGE} retries. Pipeline halted.")
            sys.exit(1)

    # --- Final Report ---
    _print_final_report(sessions)


def _print_final_report(sessions: dict) -> None:
    """Print a summary of the pipeline run."""
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(f"Run ID: {sessions['run_id']}")
    log.info(f"Domain: {sessions['domain']}")

    for stage_name, stage_data in sessions["stages"].items():
        status = stage_data["status"]
        log.info(f"  {stage_name}: {status}")

    if BENCHMARK_RESULT_FILE.exists():
        result = json.loads(BENCHMARK_RESULT_FILE.read_text())
        auc = result.get("auc")
        logloss = result.get("logloss")
        log.info(f"  AUC: {auc}")
        log.info(f"  Logloss: {logloss}")

        if auc and auc > 0.8137:
            log.info("  >>> NEW #1 ON BARS LEADERBOARD! <<<")
        elif auc and auc > 0.8100:
            log.info("  >>> Top-10 candidate on BARS leaderboard")


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
        help="Resume from the last checkpoint in sessions.json",
    )
    args = parser.parse_args()

    anyio.run(run_pipeline, args.domain, args.resume)


if __name__ == "__main__":
    main()
