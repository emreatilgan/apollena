"""Benchmarker Agent — runs Criteo_x1 benchmark on implemented model."""

import json
import math
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, SystemMessage

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = PROJECT_ROOT / "prompts" / "benchmarker_prompt.md"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "benchmark_result.json"
CONFIGS_DIR = PROJECT_ROOT / "configs"
FUXICTR_PATH = "/workspace/FuxiCTR"

ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash"]
PERMISSION_MODE = "bypassPermissions"
MAX_TURNS = 15
MAX_BUDGET_USD = 1.00


async def run_benchmarker(model_name: str, model_file: str) -> dict:
    """Run the Benchmarker agent. Returns the parsed benchmark result dict."""
    system_prompt = PROMPT_FILE.read_text()

    task_prompt = (
        f"Benchmark the following model on Criteo_x1:\n\n"
        f"**Model Name**: {model_name}\n"
        f"**Model File**: `{model_file}`\n"
        f"**FuxiCTR Path**: `{FUXICTR_PATH}`\n"
        f"**Configs Directory**: `{CONFIGS_DIR}`\n"
        f"**Results Output**: `{OUTPUT_FILE}`\n\n"
        f"Use standard Criteo_x1 hyperparameters:\n"
        f"- Embedding dim: 16\n"
        f"- Learning rate: 1e-3\n"
        f"- Batch size: 4096\n"
        f"- Epochs: 10\n"
        f"- Early stopping patience: 2\n"
        f"- Optimizer: Adam\n\n"
        f"Follow the instructions in your system prompt exactly."
    )

    session_id = None
    result_text = None

    async for message in query(
        prompt=task_prompt,
        options=ClaudeAgentOptions(
            cwd=FUXICTR_PATH,
            system_prompt=system_prompt,
            allowed_tools=ALLOWED_TOOLS,
            permission_mode=PERMISSION_MODE,
            allow_dangerously_skip_permissions=True,
            max_turns=MAX_TURNS,
            max_budget_usd=MAX_BUDGET_USD,
        ),
    ):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            session_id = message.session_id
        elif isinstance(message, ResultMessage):
            result_text = message.result

    output = json.loads(OUTPUT_FILE.read_text()) if OUTPUT_FILE.exists() else None
    return {"session_id": session_id, "result": result_text, "output": output}


def validate_benchmarker_output(output: dict) -> tuple[bool, str]:
    """Validate Benchmarker output against the gate criteria.

    Returns (is_valid, error_message).
    """
    if output is None:
        return False, "Benchmark result file not found"

    status = output.get("status")
    if status != "completed":
        error_msg = output.get("error_message", "Unknown error")
        return False, f"Benchmark status is '{status}': {error_msg}"

    auc = output.get("auc")
    if auc is None or not math.isfinite(auc):
        return False, f"AUC is not finite: {auc}"

    return True, ""
