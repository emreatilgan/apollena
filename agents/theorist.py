"""Hypothesis Generator (Theorist) Agent — maps domain mechanisms to CTR architectures."""

import json
import logging
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    UserMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
)

log = logging.getLogger("theorist")

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = PROJECT_ROOT / "prompts" / "theorist_prompt.md"
OUTPUT_FILE = PROJECT_ROOT / "hypothesis.json"
SCOUT_OUTPUT_FILE = PROJECT_ROOT / "scout_output.json"

ALLOWED_TOOLS = ["Read", "Write"]
PERMISSION_MODE = "acceptEdits"
MAX_TURNS = 20
MAX_BUDGET_USD = 2.00

# Known BARS models — used for novelty checking
BARS_MODEL_LIST = [
    "LR", "FM", "FwFM", "FmFM", "FFM", "FFMv2", "HOFM",
    "DNN", "WideDeep", "DeepFM", "FNN", "IPNN", "OPNN", "PNN",
    "DCN", "DCNv2", "CrossNet", "xDeepFM", "CIN",
    "AutoInt", "AutoInt+", "InterHAt",
    "AFN", "AFN+", "DESTINE", "EulerNet",
    "FinalMLP", "FinalNet",
    "FiGNN", "Fi-GNN",
    "BST", "DIN", "DIEN", "DMIN", "DMR", "FINT",
    "MaskNet", "AOANet", "ONN", "ONNv2",
    "SAM", "FiBiNET", "EDCN", "GDCN",
    "DeepIM", "EPNet", "PPNet", "APG",
]


async def run_theorist(scout_output: dict) -> dict:
    """Run the Theorist agent with the Scout's findings. Returns the parsed output dict."""
    system_prompt = PROMPT_FILE.read_text()

    # Build context for the Theorist: papers + BARS model list
    domain = scout_output.get("domain") or scout_output.get("target_domain", "unknown")
    theorist_context = {
        "domain": domain,
        "papers": scout_output["papers"],
        "bars_model_list": BARS_MODEL_LIST,
    }

    task_prompt = (
        f"Here is the Scout's output with papers from the **{domain}** domain:\n\n"
        f"```json\n{json.dumps(theorist_context, indent=2)}\n```\n\n"
        f"Write your hypothesis to: `{OUTPUT_FILE}`\n\n"
        f"Follow the instructions in your system prompt exactly. Think deeply — you have extended thinking enabled."
    )

    session_id = None
    result_text = None

    turn = 0
    async for message in query(
        prompt=task_prompt,
        options=ClaudeAgentOptions(
            cwd=str(PROJECT_ROOT),
            system_prompt=system_prompt,
            allowed_tools=ALLOWED_TOOLS,
            permission_mode=PERMISSION_MODE,
            max_turns=MAX_TURNS,
            max_budget_usd=MAX_BUDGET_USD,
            thinking={"type": "adaptive"},
            effort="max",
        ),
    ):
        if isinstance(message, AssistantMessage):
            turn += 1
            for block in message.content:
                if isinstance(block, TextBlock):
                    preview = block.text[:200].replace("\n", " ")
                    log.info(f"[turn {turn}] Assistant text: {preview}")
                elif isinstance(block, ThinkingBlock):
                    preview = block.thinking[:200].replace("\n", " ")
                    log.info(f"[turn {turn}] Thinking: {preview}...")
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


def validate_theorist_output(output: dict) -> tuple[bool, str]:
    """Validate Theorist output against the gate criteria.

    Returns (is_valid, error_message).
    """
    if output is None:
        return False, "Hypothesis output file not found"

    model_name = output.get("model_name", "")
    if not model_name:
        return False, "No model_name in hypothesis"

    # Check not in BARS model list
    if model_name in BARS_MODEL_LIST:
        return False, f"Model name '{model_name}' already exists in BARS taxonomy"

    # Check required architecture fields
    arch = output.get("architecture", {})
    for field in ["new_layer", "layer_math", "forward_pass", "integration"]:
        if not arch.get(field):
            return False, f"Missing architecture.{field}"

    # Check differentiability argument exists
    if not output.get("differentiability_proof"):
        return False, "Missing differentiability_proof"

    # Check novelty argument exists
    if not output.get("novelty_argument"):
        return False, "Missing novelty_argument"

    return True, ""
