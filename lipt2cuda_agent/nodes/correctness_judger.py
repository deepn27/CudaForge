"""Correctness judger node for LIPT2CudaAgent."""
from __future__ import annotations

from lipt2cuda_agent.state import LIPT2CudaAgentState
from prompts.judger_repair import build_correctness_prompts
from utils.kernel_io import extract_json


def _last_n_lines(text: str, n: int = 150) -> str:
    """Keep only the last n lines of text."""
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


def correctness_judger_node(state: LIPT2CudaAgentState) -> LIPT2CudaAgentState:
    """Analyze errors and generate repair strategy.

    This node:
    1. Extracts error log from current_kernel metrics
    2. Builds correctness judger prompts
    3. Calls LLM to analyze the error
    4. Extracts JSON strategy from response
    5. Updates state with strategy and error log for repair node

    Args:
        state: Current workflow state

    Returns:
        Updated state with judger_strategy and error_log
    """
    print("[Repair] Analyzing errors ...")

    current_kernel = state["current_kernel"]
    if current_kernel is None:
        raise ValueError("current_kernel is None in correctness_judger_node")

    # Extract error log
    error_log = _last_n_lines(
        current_kernel.metrics.get("message", "") if current_kernel.metrics else ""
    )

    # Build prompts
    problem_system_prompt, problem_prompt = build_correctness_prompts(
        error_log=error_log,
        arch_path=state["task_path"],
        cuda_code=current_kernel.code,
    )

    # Save prompt
    prompt_file = state["io_dir"] / f"round{state['round_idx']:03d}_problem_identify_prompt.txt"
    prompt_file.write_text(problem_prompt, encoding="utf-8")

    # Call LLM
    call_llm = state["call_llm"]
    raw = call_llm(
        problem_prompt,
        problem_system_prompt,
        log_path=state["log_path"],
        call_type="problem_identify",
        round_idx=state["round_idx"],
    )

    # Save raw reply
    reply_file = state["io_dir"] / f"{state['round_idx']}_raw_problem_identify_reply.txt"
    reply_file.write_text(raw, encoding="utf-8")

    # Extract JSON strategy
    problem_json = extract_json(raw)

    # Update state
    return {
        **state,
        "judger_strategy": problem_json,
        "error_log": error_log,
        "next_action": "repair",
    }
