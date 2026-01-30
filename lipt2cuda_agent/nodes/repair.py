"""Repair node for LIPT2CudaAgent."""
from __future__ import annotations

from lipt2cuda_agent.state import LIPT2CudaAgentState
from prompts.error import build_error_prompt
from utils.kernel_io import extract_code_block, save_kernel_code
from scripts.individual import KernelIndividual


def repair_node(state: LIPT2CudaAgentState) -> LIPT2CudaAgentState:
    """Repair broken kernels based on error analysis.

    This node:
    1. Builds repair prompt with old code, error log, and problem analysis
    2. Calls LLM to generate repaired code
    3. Extracts and saves the new kernel code
    4. Creates a new KernelIndividual
    5. Updates state for benchmarking

    Args:
        state: Current workflow state

    Returns:
        Updated state with repaired current_kernel
    """
    print("[Repair] Generating repaired kernel ...")

    current_kernel = state["current_kernel"]
    if current_kernel is None:
        raise ValueError("current_kernel is None in repair_node")

    # Build repair prompt
    repair_prompt = build_error_prompt(
        old_code=current_kernel.code,
        error_log=state["error_log"],
        problem=state["judger_strategy"],
        gpu_name=state["args"].gpu,
    )

    # Save prompt
    prompt_file = state["io_dir"] / f"round{state['round_idx']:03d}_repair_prompt.txt"
    prompt_file.write_text(repair_prompt, encoding="utf-8")

    # Call LLM
    call_llm = state["call_llm"]
    raw = call_llm(
        repair_prompt,
        sys_prompt=None,  # Uses default
        log_path=state["log_path"],
        call_type="repair",
        round_idx=state["round_idx"],
    )

    # Save raw reply
    reply_file = state["io_dir"] / f"{state['round_idx']}_raw_reply.txt"
    reply_file.write_text(raw, encoding="utf-8")

    # Extract code
    code = extract_code_block(raw) or raw

    # Save kernel code
    code_path = save_kernel_code(code, state["code_dir"])

    # Create new KernelIndividual
    ind = KernelIndividual(code)
    ind.code_path = code_path

    # Update state
    return {
        **state,
        "current_kernel": ind,
        "current_code": code,
        "phase": "repair",
        "next_action": "benchmark",
    }
