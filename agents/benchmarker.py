"""Benchmarker Agent — runs Criteo_x1 benchmark on implemented model."""

import json
import logging
import math
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    UserMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)

log = logging.getLogger("benchmarker")

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = PROJECT_ROOT / "prompts" / "benchmarker_prompt.md"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "benchmark_result.json"
CONFIGS_DIR = PROJECT_ROOT / "configs"
FUXICTR_PATH = str(Path.home() / "Projects" / "FuxiCTR")

ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash"]
PERMISSION_MODE = "bypassPermissions"
MAX_TURNS = 30
MAX_BUDGET_USD = 2.00


async def run_benchmarker(model_name: str, model_file: str) -> dict:
    """Run the Benchmarker agent. Returns the parsed benchmark result dict."""
    system_prompt = PROMPT_FILE.read_text().replace("{{FUXICTR_PATH}}", FUXICTR_PATH)

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

    turn = 0
    async for message in query(
        prompt=task_prompt,
        options=ClaudeAgentOptions(
            cwd=FUXICTR_PATH,
            system_prompt=system_prompt,
            allowed_tools=ALLOWED_TOOLS,
            permission_mode=PERMISSION_MODE,
            max_turns=MAX_TURNS,
            max_budget_usd=MAX_BUDGET_USD,
        ),
    ):
        if isinstance(message, AssistantMessage):
            turn += 1
            for block in message.content:
                if isinstance(block, TextBlock):
                    preview = block.text[:200].replace("\n", " ")
                    log.info(f"[turn {turn}] Assistant text: {preview}")
                elif isinstance(block, ToolUseBlock):
                    input_preview = json.dumps(block.input, ensure_ascii=False)[:300]
                    log.info(f"[turn {turn}] Tool call: {block.name} | input: {input_preview}")
        elif isinstance(message, UserMessage):
            if isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        content_str = str(block.content or "")[:300].replace("\n", " ")
                        is_err = " (ERROR)" if block.is_error else ""
                        log.info(f"[turn {turn}] Tool result{is_err}: {content_str}")
            elif isinstance(message.content, str) and message.content:
                log.info(f"[turn {turn}] User message: {message.content[:200]}")
        elif isinstance(message, SystemMessage):
            log.info(f"[turn {turn}] System ({message.subtype}): {json.dumps(message.data, default=str)[:200]}")
        elif isinstance(message, ResultMessage):
            session_id = message.session_id
            result_text = message.result
            log.info(
                f"Agent finished: turns={message.num_turns}, cost=${message.total_cost_usd:.3f}, "
                f"stop_reason={message.stop_reason}, is_error={message.is_error}"
            )

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
