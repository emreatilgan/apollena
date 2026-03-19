"""Implementation Engineer Agent — implements FuxiCTR model from hypothesis."""

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
    ToolUseBlock,
    ToolResultBlock,
)

log = logging.getLogger("engineer")

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = PROJECT_ROOT / "prompts" / "engineer_prompt.md"
FUXICTR_PATH = str(Path.home() / "Projects" / "FuxiCTR")

ALLOWED_TOOLS = ["Read", "Edit", "Write", "Bash", "Glob", "Grep"]
PERMISSION_MODE = "bypassPermissions"
MAX_TURNS = 30
MAX_BUDGET_USD = 2.00


def _extract_short_name(raw_name: str) -> str:
    """Extract the short model identifier from a name like 'TAPNet (Thouless-Anderson-Palmer Network)'."""
    return raw_name.split("(")[0].split(" —")[0].split(" -")[0].strip()


async def run_engineer(hypothesis: dict) -> dict:
    """Run the Engineer agent with the Theorist's hypothesis. Returns a summary dict."""
    system_prompt = PROMPT_FILE.read_text().replace("{{FUXICTR_PATH}}", FUXICTR_PATH)
    model_name = _extract_short_name(hypothesis["model_name"])

    task_prompt = (
        f"Implement the following hypothesis as a FuxiCTR model.\n\n"
        f"**Model Name**: {model_name}\n\n"
        f"**Full Hypothesis**:\n"
        f"```json\n{json.dumps(hypothesis, indent=2)}\n```\n\n"
        f"**FuxiCTR Path**: `{FUXICTR_PATH}`\n"
        f"**Template Model**: Read `{FUXICTR_PATH}/model_zoo/DeepFM/DeepFM_torch/src/DeepFM.py` as a structural template.\n\n"
        f"After implementation, run:\n"
        f"```bash\n"
        f"cd {FUXICTR_PATH} && python -m pytest tests/unit_tests/test_{model_name}.py -v\n"
        f"```\n\n"
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

    # Try to parse structured output from the result
    output = _parse_engineer_result(model_name, result_text)
    return {"session_id": session_id, "result": result_text, "output": output}


def _parse_engineer_result(model_name: str, result_text: str | None) -> dict:
    """Extract structured info from Engineer's result text."""
    model_file = f"{FUXICTR_PATH}/model_zoo/{model_name}/{model_name}_torch/src/{model_name}.py"
    test_file = f"{FUXICTR_PATH}/tests/unit_tests/test_{model_name}.py"

    # Check if files actually exist
    model_exists = Path(model_file).exists()
    test_exists = Path(test_file).exists()

    return {
        "model_name": model_name,
        "model_file": model_file if model_exists else None,
        "test_file": test_file if test_exists else None,
        "test_passed": "passed" in (result_text or "").lower() if result_text else False,
        "files_modified": [f for f in [model_file, test_file] if Path(f).exists()],
    }


def validate_engineer_output(output: dict) -> tuple[bool, str]:
    """Validate Engineer output against the gate criteria.

    Returns (is_valid, error_message).
    """
    if output is None:
        return False, "Engineer output is None"

    if not output.get("model_file"):
        return False, "Model file was not created"

    if not output.get("test_file"):
        return False, "Test file was not created"

    if not output.get("test_passed"):
        return False, "Tests did not pass"

    return True, ""
