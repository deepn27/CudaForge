"""Optimization node for LIPT2CudaAgent."""
from __future__ import annotations

from lipt2cuda_agent.state import LIPT2CudaAgentState
from prompts.optimization import build_optimization_prompt
from utils.kernel_io import extract_code_block, save_kernel_code
from scripts.individual import KernelIndividual


def optimization_node(state: LIPT2CudaAgentState) -> LIPT2CudaAgentState:
    """Optimize runnable kernels based on performance analysis.

    This node:
    1. Builds optimization prompt with current code and strategy
    2. Calls LLM to generate optimized code
    3. Extracts and saves the new kernel code
    4. Creates a new KernelIndividual
    5. Updates state for benchmarking

    Args:
        state: Current workflow state

    Returns:
        Updated state with optimized current_kernel
    """
    print("[Optimization] Generating optimized kernel ...")

    current_kernel = state["current_kernel"]
    if current_kernel is None:
        raise ValueError("current_kernel is None in optimization_node")

    # Build optimization prompt
    opt_prompt = build_optimization_prompt(
        arch_path=current_kernel.code_path,
        gpu_name=state["args"].gpu,
        optimization_suggestion=state["judger_strategy"],
    )

    # Save prompt
    prompt_file = state["io_dir"] / f"round{state['round_idx']:03d}_opt_prompt.txt"
    prompt_file.write_text(opt_prompt, encoding="utf-8")

    # Call LLM
    call_llm = state["call_llm"]
    raw = call_llm(
        opt_prompt,
        sys_prompt=None,  # Uses default
        log_path=state["log_path"],
        call_type="optimization",
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
        "phase": "opt",
        "next_action": "benchmark",
    }
