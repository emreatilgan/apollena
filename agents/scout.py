"""Literature Scout Agent — searches non-ML scientific domains for CTR-relevant mechanisms."""

import json
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = PROJECT_ROOT / "prompts" / "scout_prompt.md"
OUTPUT_FILE = PROJECT_ROOT / "scout_output.json"

ALLOWED_TOOLS = ["WebSearch", "WebFetch", "Read", "Write"]
PERMISSION_MODE = "acceptEdits"
MAX_TURNS = 20
MAX_BUDGET_USD = 1.50

# Domain-specific search keywords
DOMAIN_KEYWORDS = {
    "statistical_physics": [
        "spin glass", "Ising model", "frustrated magnets", "replica symmetry breaking",
        "energy landscape", "Boltzmann machine", "partition function",
        "mean field theory", "Edwards-Anderson model", "Sherrington-Kirkpatrick",
    ],
    "neuroscience": [
        "predictive coding", "prediction error minimisation", "hierarchical prediction",
        "free energy principle", "Bayesian brain", "cortical microcircuit",
        "top-down prediction", "bottom-up error signal", "neural message passing",
    ],
    "ecology": [
        "niche theory", "competitive exclusion", "resource partitioning",
        "species interaction network", "niche differentiation", "Lotka-Volterra",
        "functional trait", "niche overlap", "community assembly",
    ],
    "information_theory": [
        "Kolmogorov complexity", "algorithmic information", "minimum description length",
        "Lempel-Ziv complexity", "information bottleneck", "rate-distortion",
        "mutual information", "channel capacity", "source coding",
    ],
}


async def run_scout(domain: str) -> dict:
    """Run the Scout agent for the given domain. Returns the parsed output dict."""
    system_prompt = PROMPT_FILE.read_text()
    keywords = DOMAIN_KEYWORDS.get(domain, [])

    task_prompt = (
        f"Your target domain is: **{domain}**\n\n"
        f"Use these domain-specific keywords to guide your search:\n"
        f"{json.dumps(keywords, indent=2)}\n\n"
        f"Write your output to: `{OUTPUT_FILE}`\n\n"
        f"Follow the instructions in your system prompt exactly."
    )

    session_id = None
    result_text = None

    async for message in query(
        prompt=task_prompt,
        options=ClaudeAgentOptions(
            cwd=str(PROJECT_ROOT),
            system_prompt=system_prompt,
            allowed_tools=ALLOWED_TOOLS,
            permission_mode=PERMISSION_MODE,
            max_turns=MAX_TURNS,
            max_budget_usd=MAX_BUDGET_USD,
        ),
    ):
        if isinstance(message, ResultMessage):
            session_id = message.session_id
            result_text = message.result

    # Read the output file the agent should have written
    output = json.loads(OUTPUT_FILE.read_text()) if OUTPUT_FILE.exists() else None
    return {"session_id": session_id, "result": result_text, "output": output}


def validate_scout_output(output: dict) -> tuple[bool, str]:
    """Validate Scout output against the gate criteria.

    Returns (is_valid, error_message).
    """
    if output is None:
        return False, "Scout output file not found"

    papers = output.get("papers", [])
    if len(papers) < 3:
        return False, f"Only {len(papers)} papers returned, need at least 3"

    low_score_papers = [p for p in papers if p.get("novelty_score", 0) < 6]
    if low_score_papers:
        titles = [p["title"] for p in low_score_papers]
        return False, f"Papers with novelty_score < 6: {titles}"

    return True, ""
